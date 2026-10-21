//! cl100k split rules.
//!
//! The pattern is `(?i:'s|'t|'re|'ve|'m|'ll|'d)`, an optional prefix plus
//! letters, up to three numbers, an optional space plus punctuation and
//! its trailing newlines, then a whitespace run. A newline run is kept
//! whole through its last newline. A space prefixes a word or punctuation,
//! and does not prefix a number.

use crate::walk::classify::{class_of, masks_for, Class, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;

/// cl100k pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Cl100k;

/// Pulls pieces in batches. Create one with [`Cl100k::walk`].
pub struct Cl100kWalk<'a> {
    inner: Walk<'a, Cl100k>,
}

impl Cl100k {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<Cl100k>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> Cl100kWalk<'_> {
        Cl100kWalk {
            inner: Walk::new(text),
        }
    }
}

impl Cl100kWalk<'_> {
    /// Write the next pieces into `out`. Returns how many were written.
    /// A short count means the text is finished.
    pub fn fill(&mut self, out: &mut [Span]) -> usize {
        self.inner.fill(out)
    }
}

impl Scheme for Cl100k {
    #[inline]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance(bytes, pos)
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        window::<ISA>(bytes, pos)
    }
}

fn advance(bytes: &[u8], start: usize) -> usize {
    if let Some(n) = contraction_len(bytes, start) {
        return start + n;
    }
    if let Some(end) = take_letters(bytes, start) {
        return end;
    }
    if let Some(end) = take_numbers(bytes, start) {
        return end;
    }
    if let Some(end) = take_punct(bytes, start) {
        return end;
    }
    take_whitespace(bytes, start)
}

/// Case-insensitive `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, or `'d`.
fn contraction_len(bytes: &[u8], at: usize) -> Option<usize> {
    if bytes.get(at) != Some(&b'\'') {
        return None;
    }
    let b1 = bytes.get(at + 1).copied()? | 0x20;
    match b1 {
        b's' | b'd' | b'm' | b't' => Some(2),
        b'l' if bytes.get(at + 2).is_some_and(|b| (b | 0x20) == b'l') => Some(3),
        b'v' if bytes.get(at + 2).is_some_and(|b| (b | 0x20) == b'e') => Some(3),
        b'r' if bytes.get(at + 2).is_some_and(|b| (b | 0x20) == b'e') => Some(3),
        _ => None,
    }
}

struct Char {
    len: usize,
    letter: bool,
    number: bool,
    space: bool,
    crlf: bool,
    whitespace: bool,
}

fn char_at(bytes: &[u8], i: usize) -> Char {
    let b = bytes[i];
    if b < 0x80 {
        let letter = (b | 0x20).wrapping_sub(b'a') < 26;
        let number = b.wrapping_sub(b'0') < 10;
        let space = b == b' ';
        let crlf = b == b'\n' || b == b'\r';
        let whitespace = space || crlf || b.wrapping_sub(9) < 5;
        return Char {
            len: 1,
            letter,
            number,
            space,
            crlf,
            whitespace,
        };
    }
    let (ch, len) = decode_char(bytes, i);
    let class = class_of(ch);
    Char {
        len,
        letter: class == Class::Letter,
        number: class == Class::Number,
        space: false,
        crlf: false,
        whitespace: class == Class::Whitespace,
    }
}

fn decode_char(bytes: &[u8], i: usize) -> (char, usize) {
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

fn is_prefix(ch: &Char) -> bool {
    !ch.letter && !ch.number && !ch.crlf
}

fn take_letters(bytes: &[u8], start: usize) -> Option<usize> {
    let ch = char_at(bytes, start);
    if ch.letter {
        return Some(scan_letters(bytes, start + ch.len));
    }
    if is_prefix(&ch) && start + ch.len < bytes.len() {
        let next = char_at(bytes, start + ch.len);
        if next.letter {
            return Some(scan_letters(bytes, start + ch.len + next.len));
        }
    }
    None
}

fn scan_letters(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() {
        let ch = char_at(bytes, p);
        if !ch.letter {
            break;
        }
        p += ch.len;
    }
    p
}

fn take_numbers(bytes: &[u8], start: usize) -> Option<usize> {
    let ch = char_at(bytes, start);
    if !ch.number {
        return None;
    }
    let mut p = start + ch.len;
    let mut n = 1;
    while n < 3 && p < bytes.len() {
        let ch = char_at(bytes, p);
        if !ch.number {
            break;
        }
        p += ch.len;
        n += 1;
    }
    Some(p)
}

fn take_punct(bytes: &[u8], start: usize) -> Option<usize> {
    let mut p = start;
    let ch = char_at(bytes, p);
    if ch.space && p + 1 < bytes.len() {
        let next = char_at(bytes, p + 1);
        if !next.whitespace && !next.letter && !next.number {
            p += 1;
        }
    }
    let ch = char_at(bytes, p);
    if ch.whitespace || ch.letter || ch.number {
        return None;
    }
    p += ch.len;
    while p < bytes.len() {
        let ch = char_at(bytes, p);
        if ch.whitespace || ch.letter || ch.number {
            break;
        }
        p += ch.len;
    }
    while p < bytes.len() && (bytes[p] == b'\n' || bytes[p] == b'\r') {
        p += 1;
    }
    Some(p)
}

fn take_whitespace(bytes: &[u8], start: usize) -> usize {
    let mut p = start;
    let mut last_newline = None;
    while p < bytes.len() {
        let ch = char_at(bytes, p);
        if !ch.whitespace {
            break;
        }
        p += ch.len;
        if ch.crlf {
            last_newline = Some(p);
        }
    }
    if let Some(end) = last_newline {
        return end;
    }
    if p < bytes.len() && p - start > 1 {
        let mut last = p - 1;
        while last > start && bytes[last] & 0xC0 == 0x80 {
            last -= 1;
        }
        if last > start {
            return last;
        }
    }
    p.max(start + 1)
}

#[inline(always)]
fn window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    if masks.non_ascii != 0 {
        return window_unicode(bytes, pos, masks);
    }
    bits(bytes, pos, masks, Uni::EMPTY, 64, false)
}

/// Unicode bytes folded into the ASCII class masks. Empty on the ASCII path,
/// so those terms fold out of the inlined kernel.
struct Uni {
    letter: u64,
    ws: u64,
    cont: u64,
    lead2: u64,
    lead3: u64,
    lead4: u64,
}

impl Uni {
    const EMPTY: Self = Self {
        letter: 0,
        ws: 0,
        cont: 0,
        lead2: 0,
        lead3: 0,
        lead4: 0,
    };
}

/// Classify non-ASCII characters into the masks and run the same bit rules.
/// A Unicode number, or a character that does not fit in the window, ends
/// the bit run: `\p{N}{1,3}` counts characters, which the byte masks cannot.
#[inline(never)]
fn window_unicode(bytes: &[u8], pos: usize, masks: Masks) -> Option<Window> {
    let found = absorb(bytes, pos, masks.non_ascii);
    let mut stop = found.stop;
    if found.number && stop > 0 {
        let mut s = stop;
        while s > 0 && masks.digit & (1u64 << (s - 1)) != 0 {
            s -= 1;
        }
        stop = s;
    }
    if stop == 0 {
        return None;
    }
    bits(bytes, pos, masks, found.uni, stop, found.number)
}

struct Absorbed {
    uni: Uni,
    stop: u32,
    number: bool,
}

fn absorb(bytes: &[u8], pos: usize, mut bits: u64) -> Absorbed {
    let mut uni = Uni::EMPTY;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        let b = bytes[pos + i];
        if b & 0xC0 == 0x80 || pos + i + utf8_width(b) > bytes.len() {
            return Absorbed {
                uni,
                stop: i as u32,
                number: false,
            };
        }
        let (ch, n) = decode_char(bytes, pos + i);
        if i + n > 64 {
            return Absorbed {
                uni,
                stop: i as u32,
                number: false,
            };
        }
        if class_of(ch) == Class::Number {
            return Absorbed {
                uni,
                stop: i as u32,
                number: true,
            };
        }
        let char_bits = ((1u64 << n) - 1) << i;
        bits &= !char_bits;
        if n > 1 {
            uni.cont |= char_bits ^ (1u64 << i);
        }
        match n {
            2 => uni.lead2 |= 1u64 << i,
            3 => uni.lead3 |= 1u64 << i,
            4 => uni.lead4 |= 1u64 << i,
            _ => {}
        }
        match class_of(ch) {
            Class::Letter => uni.letter |= char_bits,
            Class::Whitespace => uni.ws |= char_bits,
            Class::Number | Class::Other => {}
        }
    }
    Absorbed {
        uni,
        stop: 64,
        number: false,
    }
}

fn utf8_width(b: u8) -> usize {
    if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// Piece starts for one 64-byte window. `stop` below 64 resumes the scalar
/// walker there. While more text follows, the last start is resumed rather
/// than treated as a finished piece.
#[inline(always)]
fn bits(
    bytes: &[u8],
    pos: usize,
    masks: Masks,
    uni: Uni,
    stop: u32,
    tail_nonws: bool,
) -> Option<Window> {
    let keep = if stop >= 64 {
        u64::MAX
    } else {
        (1u64 << stop) - 1
    };
    let letter = (masks.letter | uni.letter) & keep;
    let digit = masks.digit & keep;
    let space = masks.space & keep;
    let nl = masks.newline & keep;
    let ows = (masks.other_ws | uni.ws) & keep;
    let cont = uni.cont & keep;
    let lead2 = uni.lead2 & keep;
    let lead3 = uni.lead3 & keep;
    let lead4 = uni.lead4 & keep;
    let ws = space | nl | ows;
    // Continuation bytes of a punctuation character stay in this class so the
    // next character still sees "previous is punctuation".
    let punct = !(letter | digit | ws) & keep;
    let eof = pos + 64 == bytes.len();

    let len1 = !(cont | lead2 | lead3 | lead4);
    let prev_os = (punct | space) << 1;
    let two_back = ((prev_os & len1) << 1)
        | ((prev_os & lead2) << 2)
        | ((prev_os & lead3) << 3)
        | ((prev_os & lead4) << 4);
    let prev_punct = punct << 1;
    let prev_letter = letter << 1;
    let prev_space = space << 1;
    let prev_ows = ows << 1;
    let single_punct = prev_punct & !two_back;
    let letter_start = letter & !cont & !prev_letter & !prev_space & !prev_ows & !single_punct;
    let punct_start = punct & !cont & !prev_punct & !prev_space;
    let digit_start = splits3(digit);

    let abs_seed = nl & prev_punct;
    let abs_n = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, nl)
    };
    let ws_span = ws & !abs_n;
    let leads = ws_span & !cont;
    let nonws = !ws_span;
    let (next_ws, next_crlf) = byte_after(bytes, pos, eof);
    let ws_lead1 = leads & !(lead2 | lead3 | lead4);
    // Bytes past `stop` are unclassified. They are not evidence that a
    // whitespace run has ended.
    let mut seen_nonws = nonws & keep;
    if tail_nonws && stop < 64 {
        seen_nonws |= 1u64 << stop;
    }
    let mut split_ok = (ws_lead1 & (seen_nonws >> 1))
        | ((leads & lead2) & (seen_nonws >> 2))
        | ((leads & lead3) & (seen_nonws >> 3))
        | ((leads & lead4) & (seen_nonws >> 4));
    if !next_ws {
        split_ok |= (ws_lead1 & (1u64 << 63))
            | ((leads & lead2) & (1u64 << 62))
            | ((leads & lead3) & (1u64 << 61))
            | ((leads & lead4) & (1u64 << 60));
    }
    let mut b_ws = leads & (!((ws_span) << 1) | split_ok);
    // A tab or space after punctuation's trailing newlines starts the next piece.
    b_ws |= (abs_n << 1) & !abs_n & leads;

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
        let tail_leads = tail & leads;
        if tail_leads != 0 {
            b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
        }
        }
        runs_n &= !run;
    }
    b_ws &= !cont;

    let mut boundary = letter_start | digit_start | punct_start | b_ws;
    let mut cut = stop;

    if next_ws && ws_span & (1u64 << 63) != 0 {
        if nonws == 0 {
            cut = 0;
        } else {
            let h = 63 - nonws.leading_zeros();
            cut = cut.min(h + 1);
        }
    }
    if next_crlf && abs_n & (1u64 << 63) != 0 {
        cut = cut.min(punct_owner(punct, space, nl, 63));
    }

    let mut apostrophe = masks.apostrophe & boundary & keep;
    while apostrophe != 0 {
        let i = apostrophe.trailing_zeros();
        apostrophe &= apostrophe - 1;
        if i >= cut {
            break;
        }
        if i >= 61 {
            cut = cut.min(i);
            break;
        }
        let Some(k) = contraction_len(bytes, pos + i as usize) else {
            continue;
        };
        let end = i + k as u32;
        if end > 64 || end > cut {
            cut = cut.min(i);
            break;
        }
        boundary &= !(((1u64 << (k - 1)) - 1) << (i + 1));
        if end < 64 {
            boundary |= 1u64 << end;
        }
    }

    // A character that does not fit, or a Unicode number, ends the bit run.
    // If the piece already in progress continues past that point, resume at
    // the piece start so the next window sees it whole.
    if stop < 64 && cut > 0 {
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

/// Where the punctuation piece that owns the newline at `bit` begins.
/// The newlines themselves, the punctuation before them, and one ASCII
/// space prefix all belong to that piece.
fn punct_owner(punct: u64, space: u64, nl: u64, bit: u32) -> u32 {
    let mut i = bit;
    while i > 0 && nl & (1u64 << (i - 1)) != 0 {
        i -= 1;
    }
    while i > 0 && punct & (1u64 << (i - 1)) != 0 {
        i -= 1;
    }
    if i > 0 && space & (1u64 << (i - 1)) != 0 {
        i -= 1;
    }
    i
}

fn byte_after(bytes: &[u8], pos: usize, eof: bool) -> (bool, bool) {
    if eof {
        return (false, false);
    }
    let b = bytes[pos + 64];
    if b >= 0x80 {
        return (true, false);
    }
    (
        b == b' ' || b.wrapping_sub(9) < 5,
        b == b'\n' || b == b'\r',
    )
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

/// Starts of a digit run, then every third byte while the run continues.
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


#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<&str> {
        let mut buf = [Span { start: 0, end: 0 }; 16];
        let mut walk = Cl100k::walk(text);
        let mut out = Vec::new();
        loop {
            let n = walk.fill(&mut buf);
            if n == 0 {
                break;
            }
            for span in &buf[..n] {
                out.push(&text[span.start as usize..span.end as usize]);
            }
        }
        out
    }

    fn assert_pieces(text: &str, expected: &[&str]) {
        let got = pieces(text);
        assert_eq!(got, expected, "text {text:?}");
        assert_eq!(got.concat(), text, "pieces must cover {text:?}");
        assert_eq!(Cl100k::count(text), expected.len());
    }

    #[test]
    fn space_bang_newlines_inside_a_window() {
        let text = format!("{} !\n\n{}", "a".repeat(70), "b".repeat(70));
        let got = pieces(&text);
        assert!(
            got.iter().any(|p| *p == " !\n\n"),
            "pieces {got:?}"
        );
    }

    #[test]
    fn hand_cases() {
        assert_pieces("Hello world", &["Hello", " world"]);
        assert_pieces("12345", &["123", "45"]);
        assert_pieces("DON'T", &["DON", "'T"]);
        assert_pieces("$hello", &["$hello"]);
        assert_pieces("a b", &["a", " b"]);
        assert_pieces("a\nb", &["a", "\n", "b"]);
        assert_pieces("a \nb", &["a", " \n", "b"]);
        assert_pieces("a”\n\nb", &["a", "”\n\n", "b"]);
        assert_pieces("1¹23", &["1¹2", "3"]);
        assert_pieces("don'ts", &["don", "'t", "s"]);
        assert_pieces("x 42", &["x", " ", "42"]);
        assert_pieces(" 'they", &[" '", "they"]);
        assert_pieces("!'they", &["!'", "they"]);
        assert_pieces("'Stress", &["'S", "tress"]);
        assert_pieces("'reading", &["'re", "ading"]);
        assert_pieces(" !\n\nX", &[" !\n\n", "X"]);
    }

    #[test]
    fn contraction_letters_continue_inside_an_ascii_window() {
        let text = format!("note\n'Stress level {}", "word ".repeat(20));
        assert!(text.len() > 64);
        let got = pieces(&text);
        assert_eq!(&got[..5], &["note", "\n", "'S", "tress", " level"]);
    }

    #[test]
    fn space_apostrophe_inside_an_ascii_window() {
        let text = format!("the 'they said {}", "word ".repeat(20));
        assert!(text.len() > 64);
        let got = pieces(&text);
        assert_eq!(&got[..4], &["the", " '", "they", " said"]);
    }
}
