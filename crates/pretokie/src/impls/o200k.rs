//! o200k split rules, and the siblings that share them.
//!
//! A letter run splits on case. Capitals stay together until a lowercase
//! letter, and a capital after a lowercase letter starts a new piece.
//! `camelCase` is `camel` and `Case`. [`O200k`] and [`Kimi`] keep a
//! contraction on that piece, so `don't` stays whole. [`Nemotron`] leaves
//! the contraction as its own piece. [`O200k`] and [`Kimi`] take numbers in
//! groups of up to three. [`Nemotron`] takes each number character as its
//! own piece.
//! [`O200k`] and [`Nemotron`] keep a following run of newlines and slashes
//! on punctuation. [`Kimi`] keeps newlines only, and a Han run is its own
//! piece. Whitespace keeps a newline run through its last newline.

use crate::walk::classify::{masks_for, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;
use unicode_general_category::{get_general_category, GeneralCategory};

macro_rules! family {
    ($(#[$meta:meta])* $name:ident, $walk:ident, $contractions:literal, $digits3:literal, $slash:literal, $han:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        /// Pulls pieces in batches. Create one with [`$name::walk`].
        pub struct $walk<'a> {
            inner: Walk<'a, $name>,
        }

        impl $name {
            /// Number of pieces in `text`.
            pub fn count(text: &str) -> usize {
                Self::count_upto(text, usize::MAX)
            }

            /// Number of pieces, stopping once the count passes `limit`.
            ///
            /// Returns `limit + 1` when the text has more pieces than `limit`.
            pub fn count_upto(text: &str, limit: usize) -> usize {
                Walk::<$name>::new(text).count_upto(limit)
            }

            /// Write every piece of `text` that fits in `out`. Returns how many were written.
            pub fn fill(text: &str, out: &mut [Span]) -> usize {
                Self::walk(text).fill(out)
            }

            /// Cursor for pulling pieces in batches.
            pub fn walk(text: &str) -> $walk<'_> {
                $walk {
                    inner: Walk::new(text),
                }
            }
        }

        impl $walk<'_> {
            /// Write the next pieces into `out`. Returns how many were written.
            /// A short count means the text is finished.
            pub fn fill(&mut self, out: &mut [Span]) -> usize {
                self.inner.fill(out)
            }

            /// Write the next piece boundaries into `out`.
            ///
            /// `out[0]` is the absolute start. Each later entry is the next piece's
            /// exclusive end. `out` must hold `limit + 80` slots.
            pub fn bounds(&mut self, out: &mut [u32], limit: usize) -> usize {
                self.inner.bounds(out, limit)
            }
        }

        impl Scheme for $name {
            #[inline]
            fn advance(bytes: &[u8], pos: usize) -> usize {
                advance::<$contractions, $digits3, $slash, $han>(bytes, pos)
            }

            #[inline(always)]
            fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
                window::<ISA, $contractions, $digits3, $slash, $han>(bytes, pos)
            }
        }
    };
}

family!(
    /// o200k pretokenizer. Contractions stay on the word, numbers come in
    /// groups of up to three, and punctuation keeps newlines and slashes.
    O200k,
    O200kWalk,
    true,
    true,
    true,
    false
);
family!(
/// Nemotron pretokenizer. A contraction is its own piece. So is each
/// number character.
    Nemotron,
    NemotronWalk,
    false,
    false,
    true,
    false
);
family!(
    /// Kimi pretokenizer. Han is its own run, and punctuation keeps
    /// newlines only.
    Kimi,
    KimiWalk,
    true,
    true,
    false,
    true
);

#[derive(Clone, Copy)]
enum Case {
    Upper { last_soft: usize },
    Lower,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Upper,
    Lower,
    Caseless,
    Mark,
    Number,
    Whitespace,
    Other,
}

fn kind(ch: char) -> Kind {
    if ch.is_whitespace() {
        return Kind::Whitespace;
    }
    match get_general_category(ch) {
        GeneralCategory::UppercaseLetter | GeneralCategory::TitlecaseLetter => Kind::Upper,
        GeneralCategory::LowercaseLetter => Kind::Lower,
        GeneralCategory::ModifierLetter | GeneralCategory::OtherLetter => Kind::Caseless,
        GeneralCategory::NonspacingMark
        | GeneralCategory::SpacingMark
        | GeneralCategory::EnclosingMark => Kind::Mark,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => Kind::Number,
        _ => Kind::Other,
    }
}

fn is_letter_byte(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

fn is_upper_byte(b: u8) -> bool {
    b.wrapping_sub(b'A') < 26
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

fn advance<const CONTRACTIONS: bool, const DIGITS3: bool, const SLASH: bool, const HAN: bool>(
    bytes: &[u8],
    pos: usize,
) -> usize {
    let b0 = bytes[pos];
    if is_letter_byte(b0) {
        let state = if is_upper_byte(b0) {
            Case::Upper { last_soft: 0 }
        } else {
            Case::Lower
        };
        return suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, pos + 1, state));
    }
    if b0 == b' ' {
        let Some(b1) = bytes.get(pos + 1).copied() else {
            return pos + 1;
        };
        if is_letter_byte(b1) {
            let state = if is_upper_byte(b1) {
                Case::Upper { last_soft: 0 }
            } else {
                Case::Lower
            };
            return suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, pos + 2, state));
        }
        if b1 < 0x80 {
            if is_digit_byte(b1) || is_ascii_ws(b1) {
                return if is_digit_byte(b1) {
                    pos + 1
                } else {
                    ws_end(bytes, pos)
                };
            }
            return tail::<SLASH>(bytes, scan_punct(bytes, pos + 2));
        }
        return after_space::<CONTRACTIONS, SLASH, HAN>(bytes, pos);
    }
    if b0 >= 0x80 {
        let (ch, n) = decode(bytes, pos);
        let next = pos + n;
        if HAN && is_han(ch) {
            return scan_han(bytes, next);
        }
        return match kind(ch) {
            Kind::Upper => {
                suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, next, Case::Upper { last_soft: 0 }))
            }
            Kind::Lower => suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, next, Case::Lower)),
            Kind::Caseless | Kind::Mark => suffix::<CONTRACTIONS>(
                bytes,
                scan_case::<HAN>(bytes, next, Case::Upper { last_soft: next }),
            ),
            Kind::Number => numbers::<DIGITS3>(bytes, next, 1),
            Kind::Whitespace => {
                if ch == '\n' || ch == '\r' {
                    ws_end(bytes, pos)
                } else {
                    prefix_or_ws::<CONTRACTIONS, HAN>(bytes, pos, next)
                }
            }
            Kind::Other => prefix_or_punct::<CONTRACTIONS, SLASH, HAN>(bytes, next),
        };
    }
    if is_digit_byte(b0) {
        return numbers::<DIGITS3>(bytes, pos + 1, 1);
    }
    if is_newline_byte(b0) {
        return ws_end(bytes, pos);
    }
    if is_ascii_ws(b0) {
        return prefix_or_ws::<CONTRACTIONS, HAN>(bytes, pos, pos + 1);
    }
    prefix_or_punct::<CONTRACTIONS, SLASH, HAN>(bytes, pos + 1)
}

fn after_space<const CONTRACTIONS: bool, const SLASH: bool, const HAN: bool>(
    bytes: &[u8],
    pos: usize,
) -> usize {
    let (ch, n) = decode(bytes, pos + 1);
    let next = pos + 1 + n;
    match kind(ch) {
        Kind::Upper => {
            suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, next, Case::Upper { last_soft: 0 }))
        }
        Kind::Lower => suffix::<CONTRACTIONS>(bytes, scan_case::<HAN>(bytes, next, Case::Lower)),
        Kind::Caseless => {
            if HAN && is_han(ch) {
                return pos + 1;
            }
            suffix::<CONTRACTIONS>(
                bytes,
                scan_case::<HAN>(bytes, next, Case::Upper { last_soft: next }),
            )
        }
        Kind::Mark => {
            if HAN && is_han(ch) {
                return tail::<SLASH>(bytes, scan_punct(bytes, next));
            }
            suffix::<CONTRACTIONS>(
                bytes,
                scan_case::<HAN>(bytes, next, Case::Upper { last_soft: next }),
            )
        }
        Kind::Whitespace => ws_end(bytes, pos),
        Kind::Number => pos + 1,
        Kind::Other => tail::<SLASH>(bytes, scan_punct(bytes, next)),
    }
}

fn prefix_or_ws<const CONTRACTIONS: bool, const HAN: bool>(
    bytes: &[u8],
    pos: usize,
    next: usize,
) -> usize {
    if let Some(end) = prefixed_letters::<HAN>(bytes, next) {
        return suffix::<CONTRACTIONS>(bytes, end);
    }
    ws_end(bytes, pos)
}

fn prefix_or_punct<const CONTRACTIONS: bool, const SLASH: bool, const HAN: bool>(
    bytes: &[u8],
    next: usize,
) -> usize {
    if let Some(end) = prefixed_letters::<HAN>(bytes, next) {
        return suffix::<CONTRACTIONS>(bytes, end);
    }
    tail::<SLASH>(bytes, scan_punct(bytes, next))
}

fn prefixed_letters<const HAN: bool>(bytes: &[u8], at: usize) -> Option<usize> {
    if at >= bytes.len() {
        return None;
    }
    let b = bytes[at];
    if is_letter_byte(b) {
        let state = if is_upper_byte(b) {
            Case::Upper { last_soft: 0 }
        } else {
            Case::Lower
        };
        return Some(scan_case::<HAN>(bytes, at + 1, state));
    }
    if b < 0x80 {
        return None;
    }
    let (ch, n) = decode(bytes, at);
    let next = at + n;
    if HAN && is_han(ch) {
        return None;
    }
    match kind(ch) {
        Kind::Upper => Some(scan_case::<HAN>(bytes, next, Case::Upper { last_soft: 0 })),
        Kind::Lower => Some(scan_case::<HAN>(bytes, next, Case::Lower)),
        Kind::Caseless | Kind::Mark => {
            Some(scan_case::<HAN>(bytes, next, Case::Upper { last_soft: next }))
        }
        _ => None,
    }
}

fn scan_case<const HAN: bool>(bytes: &[u8], mut pos: usize, mut state: Case) -> usize {
    let len = bytes.len();
    loop {
        while pos < len {
            let b = bytes[pos];
            if is_upper_byte(b) {
                if matches!(state, Case::Lower) {
                    return pos;
                }
                pos += 1;
            } else if is_letter_byte(b) {
                state = Case::Lower;
                pos += 1;
            } else {
                break;
            }
        }
        if pos < len && bytes[pos] >= 0x80 {
            let (ch, n) = decode(bytes, pos);
            match kind(ch) {
                Kind::Upper => {
                    if matches!(state, Case::Lower) {
                        return pos;
                    }
                    pos += n;
                }
                Kind::Lower => {
                    state = Case::Lower;
                    pos += n;
                }
                Kind::Caseless | Kind::Mark => {
                    if HAN && is_han(ch) {
                        break;
                    }
                    pos += n;
                    if let Case::Upper { last_soft } = &mut state {
                        *last_soft = pos;
                    }
                }
                _ => break,
            }
            continue;
        }
        break;
    }
    match state {
        Case::Upper { last_soft } if last_soft != 0 => last_soft,
        _ => pos,
    }
}

fn suffix<const CONTRACTIONS: bool>(bytes: &[u8], end: usize) -> usize {
    if !CONTRACTIONS || bytes.get(end) != Some(&b'\'') {
        return end;
    }
    match bytes.get(end + 1).map(|b| b | 0x20) {
        Some(b's' | b'd' | b'm' | b't') => end + 2,
        Some(b'l') if bytes.get(end + 2).is_some_and(|b| b | 0x20 == b'l') => end + 3,
        Some(b'v') if bytes.get(end + 2).is_some_and(|b| b | 0x20 == b'e') => end + 3,
        Some(b'r') if bytes.get(end + 2).is_some_and(|b| b | 0x20 == b'e') => end + 3,
        Some(0xc5) if bytes.get(end + 2) == Some(&0xbf) => end + 3,
        _ => end,
    }
}

fn scan_punct(bytes: &[u8], mut pos: usize) -> usize {
    let len = bytes.len();
    while pos < len {
        let b = bytes[pos];
        if b < 0x80 {
            if is_letter_byte(b) || is_digit_byte(b) || is_ascii_ws(b) {
                return pos;
            }
            pos += 1;
            continue;
        }
        let (ch, n) = decode(bytes, pos);
        match kind(ch) {
            Kind::Other | Kind::Mark => pos += n,
            _ => return pos,
        }
    }
    pos
}

fn tail<const SLASH: bool>(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b != b'\n' && b != b'\r' && !(SLASH && b == b'/') {
            break;
        }
        pos += 1;
    }
    pos
}

fn scan_han(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 0x80 {
            return pos;
        }
        let (ch, n) = decode(bytes, pos);
        if !is_han(ch) {
            return pos;
        }
        pos += n;
    }
    pos
}

fn numbers<const DIGITS3: bool>(bytes: &[u8], mut pos: usize, mut n: usize) -> usize {
    if !DIGITS3 {
        return pos;
    }
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
        if kind(ch) != Kind::Number {
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
    while p < len {
        let b = bytes[p];
        if is_newline_byte(b) {
            last_char = p;
            p += 1;
            last_nl = p;
        } else if b < 0x80 && is_ascii_ws(b) {
            last_char = p;
            p += 1;
        } else if b >= 0x80 {
            let (ch, n) = decode(bytes, p);
            if kind(ch) == Kind::Whitespace {
                last_char = p;
                p += n;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    if last_nl != 0 {
        return last_nl;
    }
    if p >= len || last_char == start {
        return p;
    }
    last_char
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

#[inline(always)]
fn window<
    const ISA: u8,
    const CONTRACTIONS: bool,
    const DIGITS3: bool,
    const SLASH: bool,
    const HAN: bool,
>(
    bytes: &[u8],
    pos: usize,
) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    let (upper, slash) = extras::<ISA>(bytes, pos);
    if masks.non_ascii != 0 {
        return window_unicode::<CONTRACTIONS, DIGITS3, SLASH, HAN>(bytes, pos, masks, upper, slash);
    }
    bits::<CONTRACTIONS, DIGITS3, SLASH, HAN>(bytes, pos, masks, upper, slash, Uni::EMPTY, 64, false)
}

#[inline(never)]
fn window_unicode<
    const CONTRACTIONS: bool,
    const DIGITS3: bool,
    const SLASH: bool,
    const HAN: bool,
>(
    bytes: &[u8],
    pos: usize,
    masks: Masks,
    mut upper: u64,
    slash: u64,
) -> Option<Window> {
    let found = absorb::<HAN>(bytes, pos, masks.non_ascii);
    let mut stop = found.stop;
    if DIGITS3 && found.number && stop > 0 {
        let mut s = stop;
        while s > 0 && masks.digit & (1u64 << (s - 1)) != 0 {
            s -= 1;
        }
        stop = s;
    }
    if stop == 0 {
        return None;
    }
    upper |= found.uni.upper;
    bits::<CONTRACTIONS, DIGITS3, SLASH, HAN>(
        bytes,
        pos,
        masks,
        upper,
        slash,
        found.uni,
        stop,
        found.tail_nonws,
    )
}

struct Uni {
    letter: u64,
    upper: u64,
    ws: u64,
    cont: u64,
    lead2: u64,
    lead3: u64,
    lead4: u64,
}

impl Uni {
    const EMPTY: Self = Self {
        letter: 0,
        upper: 0,
        ws: 0,
        cont: 0,
        lead2: 0,
        lead3: 0,
        lead4: 0,
    };
}

struct Absorbed {
    uni: Uni,
    stop: u32,
    number: bool,
    tail_nonws: bool,
}

fn absorb<const HAN: bool>(bytes: &[u8], pos: usize, mut pending: u64) -> Absorbed {
    let mut uni = Uni::EMPTY;
    while pending != 0 {
        let i = pending.trailing_zeros() as usize;
        let b = bytes[pos + i];
        if b & 0xC0 == 0x80 || pos + i + utf8_width(b) > bytes.len() {
            return Absorbed {
                uni,
                stop: i as u32,
                number: false,
                tail_nonws: false,
            };
        }
        let (ch, n) = decode(bytes, pos + i);
        let class = kind(ch);
        if i + n > 64
            || matches!(class, Kind::Caseless | Kind::Mark)
            || (HAN && class == Kind::Other && is_han(ch))
        {
            return Absorbed {
                uni,
                stop: i as u32,
                number: class == Kind::Number,
                tail_nonws: class != Kind::Whitespace,
            };
        }
        if class == Kind::Number {
            return Absorbed {
                uni,
                stop: i as u32,
                number: true,
                tail_nonws: true,
            };
        }
        let char_bits = ((1u64 << n) - 1) << i;
        pending &= !char_bits;
        if n > 1 {
            uni.cont |= char_bits ^ (1u64 << i);
        }
        match n {
            2 => uni.lead2 |= 1u64 << i,
            3 => uni.lead3 |= 1u64 << i,
            4 => uni.lead4 |= 1u64 << i,
            _ => {}
        }
        match class {
            Kind::Upper => {
                uni.letter |= char_bits;
                uni.upper |= char_bits;
            }
            Kind::Lower => uni.letter |= char_bits,
            Kind::Whitespace => uni.ws |= char_bits,
            Kind::Caseless | Kind::Mark | Kind::Number | Kind::Other => {}
        }
    }
    Absorbed {
        uni,
        stop: 64,
        number: false,
        tail_nonws: false,
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

#[inline(always)]
fn bits<const CONTRACTIONS: bool, const DIGITS3: bool, const SLASH: bool, const HAN: bool>(
    bytes: &[u8],
    pos: usize,
    masks: Masks,
    upper_ascii: u64,
    slash: u64,
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
    let upper = (upper_ascii | uni.upper) & keep;
    let digit = masks.digit & keep;
    let space = masks.space & keep;
    let nl = masks.newline & keep;
    let ows = (masks.other_ws | uni.ws) & keep;
    let cont = uni.cont & keep;
    let slash = slash & keep;
    let ws = space | nl | ows;
    let punct = !(letter | digit | ws) & keep;
    let eof = pos + 64 == bytes.len();

    let tcls = if SLASH { nl | slash } else { nl };
    let prev_punct = punct << 1;
    let abs_seed = nl & prev_punct;
    let abs = if abs_seed == 0 {
        0
    } else {
        smear_up(abs_seed, tcls) & keep
    };
    let punct = punct & !abs;
    let ws = ws & !abs;
    let nl = nl & !abs;

    let len1 = !(cont | uni.lead2 | uni.lead3 | uni.lead4);
    let prev_os = (punct | space) << 1;
    let two_back = ((prev_os & len1) << 1)
        | ((prev_os & uni.lead2) << 2)
        | ((prev_os & uni.lead3) << 3)
        | ((prev_os & uni.lead4) << 4);
    let prev_punct = punct << 1;
    let prev_letter = letter << 1;
    let prev_space = space << 1;
    let prev_ows = ows << 1;
    let prev_lower = (letter & !upper) << 1;
    let single_punct = prev_punct & !two_back;
    let letter_start = letter & !cont & !prev_letter & !prev_space & !prev_ows & !single_punct;
    let camel = upper & !cont & prev_lower;
    let punct_start = punct & !cont & !prev_punct & !prev_space;
    let digit_start = if DIGITS3 { splits3(digit) } else { digit };

    let leads = ws & !cont;
    let nonws = !ws & keep;
    let mut seen_nonws = nonws;
    if tail_nonws && stop < 64 {
        seen_nonws |= 1u64 << stop;
    }
    let (next_ws, next_tail) = byte_after::<SLASH>(bytes, pos, eof);
    let ws_lead1 = leads & !(uni.lead2 | uni.lead3 | uni.lead4);
    let mut split_ok = (ws_lead1 & (seen_nonws >> 1))
        | ((leads & uni.lead2) & (seen_nonws >> 2))
        | ((leads & uni.lead3) & (seen_nonws >> 3))
        | ((leads & uni.lead4) & (seen_nonws >> 4));
    if !next_ws {
        split_ok |= (ws_lead1 & (1u64 << 63))
            | ((leads & uni.lead2) & (1u64 << 62))
            | ((leads & uni.lead3) & (1u64 << 61))
            | ((leads & uni.lead4) & (1u64 << 60));
    }
    let mut b_ws = leads & (!((ws) << 1) | split_ok);

    let mut runs_n = nl;
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
            let tail_bits = run & mask_from(q + 1);
            let tail_leads = tail_bits & leads;
            if tail_leads != 0 {
                b_ws |= 1u64 << (63 - tail_leads.leading_zeros());
            }
        }
        runs_n &= !run;
    }
    b_ws &= !cont;

    let mut boundary = letter_start | camel | digit_start | punct_start | b_ws | 1;
    let mut cut = stop;

    if next_ws && ws & (1u64 << 63) != 0 {
        if nonws == 0 {
            cut = 0;
        } else {
            let h = 63 - nonws.leading_zeros();
            cut = cut.min(h + 1);
        }
    }
    if next_tail && abs & (1u64 << 63) != 0 {
        cut = cut.min(tail_owner(abs, punct, space, 63));
    }

    if CONTRACTIONS {
        let mut apostrophe = masks.apostrophe & boundary & prev_letter & keep;
        while apostrophe != 0 {
            let i = apostrophe.trailing_zeros();
            apostrophe &= apostrophe - 1;
            if i >= cut {
                break;
            }
            if i <= 2 || i >= 61 || suffix_before(bytes, pos, i) {
                cut = cut.min(i);
                break;
            }
            let Some(k) = suffix_len(bytes, pos + i as usize) else {
                continue;
            };
            let end = i + k;
            if end > 64 || end > cut {
                cut = cut.min(i);
                break;
            }
            boundary &= !(((1u64 << (k - 1)) - 1) << i);
            if end < 64 {
                boundary |= 1u64 << end;
            }
        }
    }

    if cut > 0 && cut < 64 {
        let before = boundary & mask_below(cut);
        if before != 0 {
            let start_rel = 63 - before.leading_zeros();
            if advance::<CONTRACTIONS, DIGITS3, SLASH, HAN>(bytes, pos + start_rel as usize)
                > pos + cut as usize
            {
                cut = start_rel;
            }
        }
    }

    finish(boundary, cut, eof && cut >= 64)
}

fn suffix_before(bytes: &[u8], pos: usize, i: u32) -> bool {
    let p = pos + i as usize;
    if p < 2 {
        return false;
    }
    (bytes[p - 2] == b'\'' && matches!(bytes[p - 1] | 0x20, b's' | b'd' | b'm' | b't'))
        || (p >= 3
            && bytes[p - 3] == b'\''
            && (matches!(
                (bytes[p - 2] | 0x20, bytes[p - 1] | 0x20),
                (b'l', b'l') | (b'v', b'e') | (b'r', b'e')
            ) || (bytes[p - 2] == 0xC5 && bytes[p - 1] == 0xBF)))
}

fn suffix_len(bytes: &[u8], at: usize) -> Option<u32> {
    let b1 = bytes.get(at + 1).copied()? | 0x20;
    match b1 {
        b's' | b'd' | b'm' | b't' => Some(2),
        b'l' if bytes.get(at + 2).is_some_and(|b| b | 0x20 == b'l') => Some(3),
        b'v' if bytes.get(at + 2).is_some_and(|b| b | 0x20 == b'e') => Some(3),
        b'r' if bytes.get(at + 2).is_some_and(|b| b | 0x20 == b'e') => Some(3),
        _ => None,
    }
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
        stops: None,
        finishes,
    })
}

fn tail_owner(abs: u64, punct: u64, space: u64, bit: u32) -> u32 {
    let mut i = bit;
    while i > 0 && abs & (1u64 << (i - 1)) != 0 {
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

fn byte_after<const SLASH: bool>(bytes: &[u8], pos: usize, eof: bool) -> (bool, bool) {
    if eof {
        return (false, false);
    }
    let b = bytes[pos + 64];
    if b >= 0x80 {
        return (true, false);
    }
    (
        b == b' ' || b.wrapping_sub(9) < 5,
        b == b'\n' || b == b'\r' || (SLASH && b == b'/'),
    )
}

/// Unicode 16 `Script=Han`, the same ranges `\p{Han}` uses.
fn is_han(ch: char) -> bool {
    const HAN: &[(u32, u32)] = &[
        (0x2E80, 0x2E99),
        (0x2E9B, 0x2EF3),
        (0x2F00, 0x2FD5),
        (0x3005, 0x3005),
        (0x3007, 0x3007),
        (0x3021, 0x3029),
        (0x3038, 0x303B),
        (0x3400, 0x4DBF),
        (0x4E00, 0x9FFF),
        (0xF900, 0xFA6D),
        (0xFA70, 0xFAD9),
        (0x16FE2, 0x16FE3),
        (0x16FF0, 0x16FF1),
        (0x20000, 0x2A6DF),
        (0x2A700, 0x2B739),
        (0x2B740, 0x2B81D),
        (0x2B820, 0x2CEA1),
        (0x2CEB0, 0x2EBE0),
        (0x2EBF0, 0x2EE5D),
        (0x2F800, 0x2FA1D),
        (0x30000, 0x3134A),
        (0x31350, 0x323AF),
    ];
    let cp = ch as u32;
    HAN.iter().any(|&(start, end)| cp >= start && cp <= end)
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
fn extras<const ISA: u8>(bytes: &[u8], pos: usize) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        use crate::walk::{ISA_AVX2, ISA_AVX512};
        if ISA == ISA_AVX512 {
            // SAFETY: the AVX-512 kernel is the only caller of this isa.
            return unsafe { extras_avx512(bytes, pos) };
        }
        if ISA == ISA_AVX2 {
            // SAFETY: the AVX2 kernel is the only caller of this isa.
            return unsafe { extras_avx2(bytes, pos) };
        }
    }
    let _ = ISA;
    extras_scalar(bytes, pos)
}

fn extras_scalar(bytes: &[u8], pos: usize) -> (u64, u64) {
    let mut upper = 0u64;
    let mut slash = 0u64;
    for i in 0..64 {
        let b = bytes[pos + i];
        if is_upper_byte(b) {
            upper |= 1u64 << i;
        } else if b == b'/' {
            slash |= 1u64 << i;
        }
    }
    (upper, slash)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn extras_avx512(bytes: &[u8], pos: usize) -> (u64, u64) {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(bytes.as_ptr().add(pos) as *const _);
        let upper = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'A' as i8)),
            _mm512_set1_epi8(25),
        );
        let slash = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'/' as i8));
        (upper, slash)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn extras_avx2(bytes: &[u8], pos: usize) -> (u64, u64) {
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
        let upper = pack(
            le(_mm256_sub_epi8(v0, _mm256_set1_epi8(b'A' as i8)), _mm256_set1_epi8(25)),
            le(_mm256_sub_epi8(v1, _mm256_set1_epi8(b'A' as i8)), _mm256_set1_epi8(25)),
        );
        let slash = pack(
            _mm256_cmpeq_epi8(v0, _mm256_set1_epi8(b'/' as i8)),
            _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(b'/' as i8)),
        );
        (upper, slash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_match_fill() {
        let samples = [
            "",
            "Hello world",
            "it's a test\n",
            "123 456789",
            "café naïve 東京",
            "camelCase don't",
            "a/b\n\n",
        ];
        let long = "Hello world! It's a test.\n\tDon't  stop.\n".repeat(40);
        let words = "word ".repeat(300);
        for text in samples {
            crate::walk::bounds_match_fill::<O200k>(text);
            crate::walk::bounds_match_fill::<Nemotron>(text);
            crate::walk::bounds_match_fill::<Kimi>(text);
        }
        for text in [&long, &words] {
            crate::walk::bounds_match_fill::<O200k>(text);
            crate::walk::bounds_match_fill::<Nemotron>(text);
            crate::walk::bounds_match_fill::<Kimi>(text);
        }
    }


    fn pieces(text: &str, mut fill: impl FnMut(&mut [Span]) -> usize) -> Vec<String> {
        let mut buf = [Span { start: 0, end: 0 }; 32];
        let mut out = Vec::new();
        loop {
            let n = fill(&mut buf);
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
    fn camel_and_contractions() {
        let pieces = |text: &str| {
            let mut walk = O200k::walk(text);
            pieces(text, |buf| walk.fill(buf))
        };
        assert_eq!(pieces("camelCase"), ["camel", "Case"]);
        assert_eq!(pieces("HTTPResponse"), ["HTTPResponse"]);
        assert_eq!(pieces("don't"), ["don't"]);
        assert_eq!(pieces("can'ts"), ["can't", "s"]);
        assert_eq!(pieces("'sound"), ["'sound"]);
        assert_eq!(pieces("12345"), ["123", "45"]);
        assert_eq!(pieces(".\n//"), [".\n//"]);
        assert_eq!(pieces("a b"), ["a", " b"]);
        assert_eq!(pieces("x 42"), ["x", " ", "42"]);
    }

    #[test]
    fn camel_inside_a_long_window() {
        let text = format!("note camelCase end {}", "word ".repeat(20));
        let mut walk = O200k::walk(&text);
        let got = pieces(&text, |buf| walk.fill(buf));
        assert!(got.iter().any(|p| p == " camel"), "{got:?}");
        assert!(got.iter().any(|p| p == "Case"), "{got:?}");
    }

    #[test]
    fn nemotron_splits_contractions_and_digits() {
        let text = format!("note don't 12345 .\n// end {}", "word ".repeat(20));
        let mut walk = Nemotron::walk(&text);
        let got = pieces(&text, |buf| walk.fill(buf));
        assert!(got.iter().any(|p| p == " don"), "{got:?}");
        assert!(got.iter().any(|p| p == "'t"), "{got:?}");
        assert!(got.iter().any(|p| p == "1"), "{got:?}");
        assert!(got.iter().any(|p| p == " .\n//"), "{got:?}");
        assert!(!got.iter().any(|p| p == "don't"), "{got:?}");
        assert!(!got.iter().any(|p| p == "123"), "{got:?}");
    }

    #[test]
    fn kimi_splits_han_and_leaves_slashes() {
        let text = format!("note abc中文def 1〇2 .\n// end {}", "word ".repeat(20));
        let mut walk = Kimi::walk(&text);
        let got = pieces(&text, |buf| walk.fill(buf));
        assert!(got.iter().any(|p| p == " abc"), "{got:?}");
        assert!(got.iter().any(|p| p == "中文"), "{got:?}");
        assert!(got.iter().any(|p| p == "def"), "{got:?}");
        assert!(got.iter().any(|p| p == "1〇2"), "{got:?}");
        assert!(got.iter().any(|p| p == " .\n"), "{got:?}");
        assert!(got.iter().any(|p| p == "//"), "{got:?}");
    }
}
