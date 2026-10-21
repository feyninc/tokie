//! DeepSeek split rules.
//!
//! Three cuts run in order. Numbers come in groups of up to three. A CJK
//! ideograph or kana run is its own region (`U+4E00`–`U+9FA5`, hiragana,
//! katakana). The main pattern then cuts what remains: one ASCII punctuation
//! character may prefix ASCII letters, a non-punctuation character may
//! prefix a letter or mark, punctuation keeps a following newline run, and
//! whitespace keeps a newline run through its last newline. A whitespace
//! run that ends at a number or a CJK region stays whole.

use crate::walk::classify::{masks_for, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;
use unicode_general_category::{get_general_category, GeneralCategory};

/// DeepSeek pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeepSeek;

/// Pulls pieces in batches. Create one with [`DeepSeek::walk`].
pub struct DeepSeekWalk<'a> {
    inner: Walk<'a, DeepSeek>,
}

impl DeepSeek {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<DeepSeek>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> DeepSeekWalk<'_> {
        DeepSeekWalk {
            inner: Walk::new(text),
        }
    }
}

impl DeepSeekWalk<'_> {
    /// Write the next pieces into `out`. Returns how many were written.
    /// A short count means the text is finished.
    pub fn fill(&mut self, out: &mut [Span]) -> usize {
        self.inner.fill(out)
    }
}

impl Scheme for DeepSeek {
    #[inline]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance(bytes, pos)
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        window::<ISA>(bytes, pos)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Letter,
    Mark,
    Number,
    Whitespace,
    PunctSym,
    Other,
}

fn kind(ch: char) -> Kind {
    if ch.is_whitespace() {
        return Kind::Whitespace;
    }
    match get_general_category(ch) {
        GeneralCategory::UppercaseLetter
        | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter
        | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter => Kind::Letter,
        GeneralCategory::NonspacingMark
        | GeneralCategory::SpacingMark
        | GeneralCategory::EnclosingMark => Kind::Mark,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => Kind::Number,
        GeneralCategory::ConnectorPunctuation
        | GeneralCategory::DashPunctuation
        | GeneralCategory::OpenPunctuation
        | GeneralCategory::ClosePunctuation
        | GeneralCategory::InitialPunctuation
        | GeneralCategory::FinalPunctuation
        | GeneralCategory::OtherPunctuation
        | GeneralCategory::MathSymbol
        | GeneralCategory::CurrencySymbol
        | GeneralCategory::ModifierSymbol
        | GeneralCategory::OtherSymbol => Kind::PunctSym,
        _ => Kind::Other,
    }
}

fn is_cjk(ch: char) -> bool {
    let cp = ch as u32;
    (0x4E00..=0x9FA5).contains(&cp) || (0x3040..=0x30FF).contains(&cp)
}

fn is_letter_byte(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

fn is_digit_byte(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

fn is_newline_byte(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

fn is_ascii_punct(b: u8) -> bool {
    b.is_ascii_punctuation()
}

fn advance(bytes: &[u8], pos: usize) -> usize {
    let b0 = bytes[pos];
    if b0 < 0x80 {
        if is_digit_byte(b0) {
            return numbers(bytes, pos + 1, 1);
        }
        return advance_main(bytes, pos);
    }
    let (ch, n) = decode(bytes, pos);
    if is_cjk(ch) {
        return advance_cjk(bytes, pos);
    }
    if kind(ch) == Kind::Number {
        return numbers(bytes, pos + n, 1);
    }
    advance_main(bytes, pos)
}

fn advance_main(bytes: &[u8], pos: usize) -> usize {
    let b0 = bytes[pos];
    if is_letter_byte(b0) {
        return scan_lm(bytes, pos + 1, false);
    }
    if b0 == b' ' {
        let Some(b1) = bytes.get(pos + 1).copied() else {
            return pos + 1;
        };
        if is_letter_byte(b1) {
            return scan_lm(bytes, pos + 2, false);
        }
        if b1 < 0x80 {
            if is_digit_byte(b1) {
                return pos + 1;
            }
            if is_ascii_punct(b1) {
                return newlines(bytes, scan_ps(bytes, pos + 2, false));
            }
            if is_ascii_ws(b1) {
                return ws_end(bytes, pos);
            }
            return pos + 1;
        }
        let (ch, n) = decode(bytes, pos + 1);
        if is_cjk(ch) {
            return pos + 1;
        }
        let next = pos + 1 + n;
        return match kind(ch) {
            Kind::Letter | Kind::Mark => scan_lm(bytes, next, false),
            Kind::Number => pos + 1,
            Kind::Whitespace => ws_end(bytes, pos),
            Kind::PunctSym => newlines(bytes, scan_ps(bytes, next, false)),
            Kind::Other => pos + 1,
        };
    }
    if b0 < 0x80 {
        if is_newline_byte(b0) {
            return ws_end(bytes, pos);
        }
        if is_ascii_ws(b0) {
            if lm_end(bytes, pos + 1, false).is_some() {
                return scan_lm(bytes, pos + 1, false);
            }
            return ws_end(bytes, pos);
        }
        if is_ascii_punct(b0) {
            if bytes.get(pos + 1).is_some_and(|b| is_letter_byte(*b)) {
                return scan_ascii_letters(bytes, pos + 1);
            }
            return newlines(bytes, scan_ps(bytes, pos + 1, false));
        }
        if lm_end(bytes, pos + 1, false).is_some() {
            return scan_lm(bytes, pos + 1, false);
        }
        return scan_gap(bytes, pos, 1, false);
    }
    let (ch, n) = decode(bytes, pos);
    let next = pos + n;
    match kind(ch) {
        Kind::Letter | Kind::Mark => scan_lm(bytes, next, false),
        Kind::Whitespace => {
            if lm_end(bytes, next, false).is_some() {
                return scan_lm(bytes, next, false);
            }
            ws_end(bytes, pos)
        }
        Kind::PunctSym => newlines(bytes, scan_ps(bytes, next, false)),
        Kind::Number | Kind::Other => {
            if lm_end(bytes, next, false).is_some() {
                return scan_lm(bytes, next, false);
            }
            scan_gap(bytes, pos, n, false)
        }
    }
}

fn advance_cjk(bytes: &[u8], pos: usize) -> usize {
    let (ch, n) = decode(bytes, pos);
    let next = pos + n;
    match kind(ch) {
        Kind::Letter | Kind::Mark => scan_lm(bytes, next, true),
        Kind::PunctSym => scan_ps(bytes, next, true),
        _ => {
            if lm_end(bytes, next, true).is_some() {
                return scan_lm(bytes, next, true);
            }
            scan_gap(bytes, pos, n, true)
        }
    }
}

fn lm_end(bytes: &[u8], pos: usize, cjk: bool) -> Option<usize> {
    let b = *bytes.get(pos)?;
    if b < 0x80 {
        if !cjk && is_letter_byte(b) {
            return Some(pos + 1);
        }
        return None;
    }
    let (ch, n) = decode(bytes, pos);
    if is_cjk(ch) != cjk {
        return None;
    }
    match kind(ch) {
        Kind::Letter | Kind::Mark => Some(pos + n),
        _ => None,
    }
}

fn scan_lm(bytes: &[u8], mut pos: usize, cjk: bool) -> usize {
    let len = bytes.len();
    loop {
        if !cjk {
            pos = scan_ascii_letters(bytes, pos);
        }
        if pos >= len || bytes[pos] < 0x80 {
            return pos;
        }
        let (ch, n) = decode(bytes, pos);
        if is_cjk(ch) != cjk {
            return pos;
        }
        match kind(ch) {
            Kind::Letter | Kind::Mark => pos += n,
            _ => return pos,
        }
    }
}

fn scan_ascii_letters(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && is_letter_byte(bytes[pos]) {
        pos += 1;
    }
    pos
}

fn scan_ps(bytes: &[u8], mut pos: usize, cjk: bool) -> usize {
    let len = bytes.len();
    loop {
        if !cjk {
            while pos < len {
                let b = bytes[pos];
                if b >= 0x80 {
                    break;
                }
                if !is_ascii_punct(b) {
                    return pos;
                }
                pos += 1;
            }
        }
        if pos >= len || bytes[pos] < 0x80 {
            return pos;
        }
        let (ch, n) = decode(bytes, pos);
        if is_cjk(ch) != cjk || kind(ch) != Kind::PunctSym {
            return pos;
        }
        pos += n;
    }
}

fn newlines(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() && is_newline_byte(bytes[pos]) {
        pos += 1;
    }
    pos
}

fn numbers(bytes: &[u8], mut pos: usize, mut n: usize) -> usize {
    while n < 3 && pos < bytes.len() {
        let b = bytes[pos];
        if is_digit_byte(b) {
            pos += 1;
            n += 1;
            continue;
        }
        if b < 0x80 {
            break;
        }
        let (ch, len) = decode(bytes, pos);
        if is_cjk(ch) || kind(ch) != Kind::Number {
            break;
        }
        pos += len;
        n += 1;
    }
    pos
}

fn ws_end(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut p = start;
    let mut last_nl = 0usize;
    let mut last_char = start;
    let mut boundary = false;
    while p < len {
        let b = bytes[p];
        if is_newline_byte(b) {
            last_char = p;
            p += 1;
            last_nl = p;
        } else if b < 0x80 && is_ascii_ws(b) {
            last_char = p;
            p += 1;
        } else if b < 0x80 {
            boundary = is_digit_byte(b);
            break;
        } else {
            let (ch, n) = decode(bytes, p);
            if kind(ch) == Kind::Whitespace {
                last_char = p;
                p += n;
            } else {
                boundary = kind(ch) == Kind::Number || is_cjk(ch);
                break;
            }
        }
    }
    if last_nl != 0 {
        return last_nl;
    }
    if p >= len || boundary {
        return p;
    }
    if last_char > start {
        return last_char;
    }
    p
}

fn scan_gap(bytes: &[u8], pos: usize, first_len: usize, cjk: bool) -> usize {
    let len = bytes.len();
    let mut p = pos + first_len;
    while p < len {
        let b = bytes[p];
        let (ch, n) = if b < 0x80 {
            (char::from_u32(b as u32).unwrap(), 1)
        } else {
            decode(bytes, p)
        };
        if is_cjk(ch) != cjk || kind(ch) != Kind::Other {
            return p;
        }
        if lm_end(bytes, p + n, cjk).is_some() {
            return p;
        }
        p += n;
    }
    p
}

fn decode(bytes: &[u8], i: usize) -> (char, usize) {
    let b0 = bytes[i];
    let width = if b0 < 0xE0 {
        2
    } else if b0 < 0xF0 {
        3
    } else {
        4
    };
    let end = (i + width).min(bytes.len());
    let rest = std::str::from_utf8(&bytes[i..end]).expect("piece boundary is a char boundary");
    let ch = rest.chars().next().expect("i is inside bytes");
    (ch, ch.len_utf8())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Follow {
    Whitespace,
    Boundary,
    Letter,
    Other,
}

fn follow_at(bytes: &[u8], at: usize) -> Follow {
    if at >= bytes.len() {
        return Follow::Boundary;
    }
    let b = bytes[at];
    if b < 0x80 {
        if is_ascii_ws(b) {
            return Follow::Whitespace;
        }
        if is_digit_byte(b) {
            return Follow::Boundary;
        }
        if is_letter_byte(b) {
            return Follow::Letter;
        }
        return Follow::Other;
    }
    let (ch, _) = decode(bytes, at);
    if is_cjk(ch) || kind(ch) == Kind::Number {
        Follow::Boundary
    } else if kind(ch) == Kind::Whitespace {
        Follow::Whitespace
    } else if matches!(kind(ch), Kind::Letter | Kind::Mark) {
        Follow::Letter
    } else {
        Follow::Other
    }
}

#[inline(always)]
fn window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    let punct = punct_mask::<ISA>(bytes, pos);
    let stop = if masks.non_ascii == 0 {
        64
    } else {
        masks.non_ascii.trailing_zeros()
    };
    if stop == 0 {
        return None;
    }
    bits(bytes, pos, masks, punct, stop)
}

#[inline(always)]
fn bits(bytes: &[u8], pos: usize, masks: Masks, punct: u64, stop: u32) -> Option<Window> {
    let keep = if stop >= 64 {
        u64::MAX
    } else {
        (1u64 << stop) - 1
    };
    let letter = masks.letter & keep;
    let digit = masks.digit & keep;
    let space = masks.space & keep;
    let nl = masks.newline & keep;
    let ows = masks.other_ws & keep;
    let punct = punct & keep;
    let ws = space | nl | ows;
    let control = !(letter | digit | ws | punct) & keep;
    let eof = pos + stop as usize == bytes.len();
    let follow = follow_at(bytes, pos + stop as usize);

    let prev_punct = punct << 1;
    let prev_space = space << 1;
    let prev_letter = letter << 1;
    let prev_ows = ows << 1;
    let prev_control = control << 1;
    let punct_start = punct & !prev_punct & !prev_space;
    let mut next_letter = letter >> 1;
    if follow == Follow::Letter && stop > 0 {
        next_letter |= 1u64 << (stop - 1);
    }
    let letter_start =
        letter & !prev_letter & !(punct_start << 1) & !prev_space & !prev_ows & !prev_control;
    let control_start = control & (!prev_control | next_letter);
    let digit_start = splits3(digit);

    let abs_seed = nl & prev_punct;
    let abs = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, nl) & keep
    };
    let ws_span = ws & !abs;
    let content = (!ws & !digit) & keep;
    let mut seen = content;
    if matches!(follow, Follow::Letter | Follow::Other) && stop < 64 {
        seen |= 1u64 << stop;
    }
    let mut split_ok = ws_span & (seen >> 1);
    if matches!(follow, Follow::Letter | Follow::Other) && stop >= 64 {
        split_ok |= ws_span & (1u64 << 63);
    }
    let mut b_ws = ws_span & (!(ws_span << 1) | split_ok);
    b_ws |= (abs << 1) & !abs & ws_span;

    let nonws = !ws_span & keep;
    let mut runs_n = nl & ws_span;
    while runs_n != 0 {
        let f = runs_n.trailing_zeros();
        let below = nonws & mask_below(f);
        let a = if below == 0 {
            0
        } else {
            64 - below.leading_zeros()
        };
        let e = (nonws & mask_from(f)).trailing_zeros();
        let run = mask_from(a) & !mask_from(e);
        b_ws &= !run;
        b_ws |= 1u64 << a;
        let q = 63 - (nl & run).leading_zeros();
        if q + 1 < e {
            b_ws |= 1u64 << (q + 1);
            let tail = run & mask_from(q + 1);
            let tail_leads = tail & ws_span;
            if tail_leads != 0 {
                let last = 63 - tail_leads.leading_zeros();
                // A whitespace run that reaches a number or a CJK region stays
                // whole. Only a letter, mark, or other character pulls the
                // last whitespace character off as its prefix.
                let boundary_after = if e < stop {
                    digit & (1u64 << e) != 0
                } else {
                    matches!(follow, Follow::Boundary | Follow::Whitespace)
                };
                if last > q + 1 && !boundary_after {
                    b_ws |= 1u64 << last;
                }
            }
        }
        runs_n &= !run;
    }

    let boundary = letter_start | digit_start | punct_start | control_start | b_ws;
    let mut cut = stop;

    if follow == Follow::Whitespace && stop > 0 && ws_span & (1u64 << (stop - 1)) != 0 {
        if nonws == 0 {
            cut = 0;
        } else {
            let h = 63 - nonws.leading_zeros();
            cut = cut.min(h + 1);
        }
    }
    if follow == Follow::Whitespace && abs & (1u64 << stop.saturating_sub(1)) != 0 && stop > 0 {
        cut = cut.min(punct_owner(punct, space, stop - 1));
    }

    if cut > 0 && cut < 64 {
        let before = boundary & mask_below(cut);
        if before != 0 {
            let start_rel = 63 - before.leading_zeros();
            if advance(bytes, pos + start_rel as usize) > pos + cut as usize {
                cut = start_rel;
            }
        }
    }

    finish(boundary, cut, eof && cut >= 64)
}

fn finish(mut boundary: u64, cut: u32, finishes: bool) -> Option<Window> {
    if cut < 64 {
        boundary &= mask_below(cut);
        if cut > 0 {
            boundary |= 1u64 << cut;
        }
    }
    if boundary & 1 == 0 {
        return None;
    }
    let complete = if finishes {
        boundary.count_ones()
    } else {
        boundary.count_ones().saturating_sub(1)
    };
    if complete == 0 {
        return None;
    }
    Some(Window {
        starts: boundary,
        finishes,
    })
}

fn punct_owner(punct: u64, space: u64, bit: u32) -> u32 {
    let mut i = bit;
    while i > 0 && (1u64 << (i - 1)) & !punct != 0 && punct & (1u64 << i) == 0 {
        // Walk back across the absorbed newline run. The caller passes the
        // last absorbed newline, and every preceding absorbed byte is a newline.
        i -= 1;
        if punct & (1u64 << i) != 0 || space & (1u64 << i) != 0 {
            break;
        }
    }
    while i > 0 && punct & (1u64 << (i - 1)) != 0 {
        i -= 1;
    }
    if i > 0 && space & (1u64 << (i - 1)) != 0 {
        i -= 1;
    }
    i
}

fn mask_below(bit: u32) -> u64 {
    if bit >= 64 {
        u64::MAX
    } else if bit == 0 {
        0
    } else {
        (1u64 << bit) - 1
    }
}

fn mask_from(bit: u32) -> u64 {
    if bit >= 64 { 0 } else { u64::MAX << bit }
}

fn smear_up(seed: u64, within: u64) -> u64 {
    let mut a = seed;
    let mut m = within;
    let mut sh = 1u32;
    while sh < 64 {
        a |= (a << sh) & m;
        m &= m << sh;
        sh <<= 1;
    }
    a
}

fn splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1);
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

#[inline(always)]
fn punct_mask<const ISA: u8>(bytes: &[u8], pos: usize) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::walk::{ISA_AVX2, ISA_AVX512};
        if ISA == ISA_AVX512 {
            // SAFETY: the AVX-512 kernel is the only caller of this isa.
            return unsafe { punct_avx512(bytes, pos) };
        }
        if ISA == ISA_AVX2 {
            // SAFETY: the AVX2 kernel is the only caller of this isa.
            return unsafe { punct_avx2(bytes, pos) };
        }
    }
    let _ = ISA;
    punct_scalar(bytes, pos)
}

fn punct_scalar(bytes: &[u8], pos: usize) -> u64 {
    let mut punct = 0u64;
    for i in 0..64 {
        if is_ascii_punct(bytes[pos + i]) {
            punct |= 1u64 << i;
        }
    }
    punct
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn punct_avx512(bytes: &[u8], pos: usize) -> u64 {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(pos) as *const _);
        let range = |lo: u8, hi: u8| {
            _mm512_cmple_epu8_mask(
                _mm512_sub_epi8(v, _mm512_set1_epi8(lo as i8)),
                _mm512_set1_epi8((hi - lo) as i8),
            )
        };
        range(0x21, 0x2F) | range(0x3A, 0x40) | range(0x5B, 0x60) | range(0x7B, 0x7E)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn punct_avx2(bytes: &[u8], pos: usize) -> u64 {
    use std::arch::x86_64::*;
    unsafe {
        let pack = |m0: __m256i, m1: __m256i| -> u64 {
            (_mm256_movemask_epi8(m0) as u32 as u64)
                | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };
        let le = |v: __m256i, lim: __m256i| -> __m256i {
            _mm256_cmpeq_epi8(_mm256_min_epu8(v, lim), v)
        };
        let p = bytes.as_ptr().add(pos);
        let v0 = _mm256_loadu_si256(p as *const _);
        let v1 = _mm256_loadu_si256(p.add(32) as *const _);
        let range = |lo: u8, hi: u8| {
            let lim = _mm256_set1_epi8((hi - lo) as i8);
            let a = le(
                _mm256_sub_epi8(v0, _mm256_set1_epi8(lo as i8)),
                lim,
            );
            let b = le(
                _mm256_sub_epi8(v1, _mm256_set1_epi8(lo as i8)),
                lim,
            );
            pack(a, b)
        };
        range(0x21, 0x2F) | range(0x3A, 0x40) | range(0x5B, 0x60) | range(0x7B, 0x7E)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<String> {
        let mut walk = DeepSeek::walk(text);
        let mut buf = [Span { start: 0, end: 0 }; 32];
        let mut out = Vec::new();
        loop {
            let n = walk.fill(&mut buf);
            if n == 0 {
                break;
            }
            for span in &buf[..n] {
                out.push(text[span.start as usize..span.end as usize].to_string());
            }
        }
        out
    }

    #[test]
    fn numbers_letters_and_punctuation() {
        assert_eq!(pieces("hello"), ["hello"]);
        assert_eq!(pieces("!hello"), ["!hello"]);
        assert_eq!(pieces("!!hello"), ["!!", "hello"]);
        assert_eq!(pieces("don't"), ["don", "'t"]);
        assert_eq!(pieces("1234"), ["123", "4"]);
        assert_eq!(pieces("  1"), ["  ", "1"]);
        assert_eq!(pieces("a  1"), ["a", "  ", "1"]);
        assert_eq!(pieces(" hello"), [" hello"]);
        assert_eq!(pieces("  hello"), [" ", " hello"]);
    }

    #[test]
    fn rules_hold_inside_a_long_window() {
        let text = format!("note don't 1234  1 !!hello end {}", "word ".repeat(20));
        let got = pieces(&text);
        assert!(got.iter().any(|p| p == " don"), "{got:?}");
        assert!(got.iter().any(|p| p == "'t"), "{got:?}");
        assert!(got.iter().any(|p| p == "123"), "{got:?}");
        assert!(got.iter().any(|p| p == "4"), "{got:?}");
        assert!(got.iter().any(|p| p == "  "), "{got:?}");
        assert!(got.iter().any(|p| p == " !!"), "{got:?}");
        assert!(got.iter().any(|p| p == "hello"), "{got:?}");
    }
}
