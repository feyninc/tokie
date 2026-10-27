//! cl100k split rules, and the siblings that share them.
//!
//! The pattern is `(?i:'s|'t|'re|'ve|'m|'ll|'d)`, an optional prefix plus
//! letters, a run of numbers, an optional space plus punctuation and its
//! trailing newlines, then a whitespace run. A newline run is kept whole
//! through its last newline. A space prefixes a word or punctuation, and
//! does not prefix a number.
//!
//! [`Voyage`] (Qwen2 and Qwen3) takes each number character alone. [`Olmo`]
//! keeps cl100k's groups of three. [`Qwen`] (Qwen3.5) is Voyage with
//! combining marks counted as letters.

use crate::walk::classify::{class_of, masks_for, Class, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;
use unicode_general_category::{get_general_category, GeneralCategory};

macro_rules! family {
    ($(#[$meta:meta])* $name:ident, $walk:ident, $digits3:literal, $marks:literal) => {
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
                advance::<$digits3, $marks>(bytes, pos)
            }

            #[inline(always)]
            fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
                window::<ISA, $digits3, $marks>(bytes, pos)
            }
        }
    };
}

family!(
    /// cl100k pretokenizer. Numbers come in groups of up to three.
    Cl100k,
    Cl100kWalk,
    true,
    false
);
family!(
    /// Qwen2 and Qwen3 pretokenizer. Each number character is its own piece.
    Voyage,
    VoyageWalk,
    false,
    false
);
family!(
    /// Olmo 2 and Olmo 3 pretokenizer. Numbers come in groups of up to three.
    Olmo,
    OlmoWalk,
    true,
    false
);
family!(
    /// Qwen3.5 pretokenizer. Each number is its own piece, and a mark is a letter.
    Qwen,
    QwenWalk,
    false,
    true
);

fn advance<const DIGITS3: bool, const MARKS: bool>(bytes: &[u8], start: usize) -> usize {
    if let Some(n) = contraction_len(bytes, start) {
        return start + n;
    }
    if let Some(end) = take_letters::<MARKS>(bytes, start) {
        return end;
    }
    if let Some(end) = take_numbers::<DIGITS3, MARKS>(bytes, start) {
        return end;
    }
    if let Some(end) = take_punct::<MARKS>(bytes, start) {
        return end;
    }
    take_whitespace::<MARKS>(bytes, start)
}

fn kind<const MARKS: bool>(ch: char) -> Class {
    let class = class_of(ch);
    if MARKS && class == Class::Other && is_mark(ch) {
        Class::Letter
    } else {
        class
    }
}

fn is_mark(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::NonspacingMark
            | GeneralCategory::SpacingMark
            | GeneralCategory::EnclosingMark
    )
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

fn char_at<const MARKS: bool>(bytes: &[u8], i: usize) -> Char {
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
    let class = kind::<MARKS>(ch);
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

fn take_letters<const MARKS: bool>(bytes: &[u8], start: usize) -> Option<usize> {
    let ch = char_at::<MARKS>(bytes, start);
    if ch.letter {
        return Some(scan_letters::<MARKS>(bytes, start + ch.len));
    }
    if is_prefix(&ch) && start + ch.len < bytes.len() {
        let next = char_at::<MARKS>(bytes, start + ch.len);
        if next.letter {
            return Some(scan_letters::<MARKS>(bytes, start + ch.len + next.len));
        }
    }
    None
}

fn scan_letters<const MARKS: bool>(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() {
        let ch = char_at::<MARKS>(bytes, p);
        if !ch.letter {
            break;
        }
        p += ch.len;
    }
    p
}

fn take_numbers<const DIGITS3: bool, const MARKS: bool>(bytes: &[u8], start: usize) -> Option<usize> {
    let ch = char_at::<MARKS>(bytes, start);
    if !ch.number {
        return None;
    }
    let mut p = start + ch.len;
    let mut n = 1;
    let limit = if DIGITS3 { 3 } else { 1 };
    while n < limit && p < bytes.len() {
        let ch = char_at::<MARKS>(bytes, p);
        if !ch.number {
            break;
        }
        p += ch.len;
        n += 1;
    }
    Some(p)
}

fn take_punct<const MARKS: bool>(bytes: &[u8], start: usize) -> Option<usize> {
    let mut p = start;
    let ch = char_at::<MARKS>(bytes, p);
    if ch.space && p + 1 < bytes.len() {
        let next = char_at::<MARKS>(bytes, p + 1);
        if !next.whitespace && !next.letter && !next.number {
            p += 1;
        }
    }
    let ch = char_at::<MARKS>(bytes, p);
    if ch.whitespace || ch.letter || ch.number {
        return None;
    }
    p += ch.len;
    while p < bytes.len() {
        let ch = char_at::<MARKS>(bytes, p);
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

fn take_whitespace<const MARKS: bool>(bytes: &[u8], start: usize) -> usize {
    let mut p = start;
    let mut last_newline = None;
    while p < bytes.len() {
        let ch = char_at::<MARKS>(bytes, p);
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
fn window<const ISA: u8, const DIGITS3: bool, const MARKS: bool>(
    bytes: &[u8],
    pos: usize,
) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    if masks.non_ascii != 0 {
        return window_unicode::<DIGITS3, MARKS>(bytes, pos, masks);
    }
    bits::<DIGITS3, MARKS>(bytes, pos, masks, Uni::EMPTY, 64, false)
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
/// the bit run. Groups of three count characters, which the byte masks cannot.
/// A single number is one piece, so the preceding digits stay where they are.
#[inline(never)]
fn window_unicode<const DIGITS3: bool, const MARKS: bool>(
    bytes: &[u8],
    pos: usize,
    masks: Masks,
) -> Option<Window> {
    let found = absorb::<MARKS>(bytes, pos, masks.non_ascii);
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
    bits::<DIGITS3, MARKS>(bytes, pos, masks, found.uni, stop, found.tail_nonws)
}

struct Absorbed {
    uni: Uni,
    stop: u32,
    /// A Unicode number ended the run, so groups of three rewind into the
    /// ASCII digits in front of it.
    number: bool,
    /// The byte at the original stop is not whitespace. After a rewind this
    /// bit sits on the first digit, which is also not whitespace.
    tail_nonws: bool,
}

fn absorb<const MARKS: bool>(bytes: &[u8], pos: usize, mut bits: u64) -> Absorbed {
    let mut uni = Uni::EMPTY;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        let b = bytes[pos + i];
        if b & 0xC0 == 0x80 || pos + i + utf8_width(b) > bytes.len() {
            return Absorbed {
                uni,
                stop: i as u32,
                number: false,
                tail_nonws: false,
            };
        }
        let (ch, n) = decode_char(bytes, pos + i);
        let class = kind::<MARKS>(ch);
        // The character starts in this window and continues past it. Its
        // class still decides whether the whitespace run in front splits.
        if i + n > 64 {
            return Absorbed {
                uni,
                stop: i as u32,
                number: class == Class::Number,
                tail_nonws: class != Class::Whitespace,
            };
        }
        if class == Class::Number {
            return Absorbed {
                uni,
                stop: i as u32,
                number: true,
                tail_nonws: true,
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
        match class {
            Class::Letter => uni.letter |= char_bits,
            Class::Whitespace => uni.ws |= char_bits,
            Class::Number | Class::Other => {}
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

/// Piece starts for one 64-byte window. `stop` below 64 resumes the scalar
/// walker there. While more text follows, the last start is resumed rather
/// than treated as a finished piece.
#[inline(always)]
fn bits<const DIGITS3: bool, const MARKS: bool>(
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
    let digit_start = if DIGITS3 { splits3(digit) } else { digit };

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
            if advance::<DIGITS3, MARKS>(bytes, pos + start_rel as usize) > pos + cut as usize {
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
        stops: None,
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

    #[test]
    fn bounds_match_fill() {
        let samples = [
            "",
            "Hello world",
            "it's a test\n",
            "123 456789",
            "café naïve 東京",
            "a12b34",
        ];
        let long = "Hello world! It's a test.\n\tDon't  stop.\n".repeat(40);
        let words = "word ".repeat(300);
        for text in samples {
            crate::walk::bounds_match_fill::<Cl100k>(text);
            crate::walk::bounds_match_fill::<Olmo>(text);
            crate::walk::bounds_match_fill::<Qwen>(text);
            crate::walk::bounds_match_fill::<Voyage>(text);
        }
        for text in [&long, &words] {
            crate::walk::bounds_match_fill::<Cl100k>(text);
            crate::walk::bounds_match_fill::<Olmo>(text);
            crate::walk::bounds_match_fill::<Qwen>(text);
            crate::walk::bounds_match_fill::<Voyage>(text);
        }
    }


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

    fn drain(text: &str, mut next: impl FnMut(&mut [Span]) -> usize) -> Vec<String> {
        let mut buf = [Span { start: 0, end: 0 }; 32];
        let mut out = Vec::new();
        loop {
            let n = next(&mut buf);
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
    fn voyage_splits_every_number() {
        let text = format!("id 12345 end {}", "word ".repeat(16));
        let mut walk = Voyage::walk(&text);
        let got = drain(&text, |buf| walk.fill(buf));
        assert!(got.join("|").contains("1|2|3|4|5"), "{got:?}");
        let mut walk = Voyage::walk("1¹23");
        assert_eq!(drain("1¹23", |buf| walk.fill(buf)), ["1", "¹", "2", "3"]);
    }

    #[test]
    fn olmo_groups_numbers_like_cl100k() {
        let mut olmo = Olmo::walk("1¹23");
        let mut cl = Cl100k::walk("1¹23");
        assert_eq!(
            drain("1¹23", |buf| olmo.fill(buf)),
            drain("1¹23", |buf| cl.fill(buf))
        );
        let mut walk = Olmo::walk("12345");
        assert_eq!(drain("12345", |buf| walk.fill(buf)), ["123", "45"]);
    }

    #[test]
    fn qwen_counts_a_mark_as_a_letter() {
        let word = format!("e{}", "\u{0301}");
        let mut walk = Qwen::walk(&word);
        assert_eq!(drain(&word, |buf| walk.fill(buf)), [word.clone()]);
        let bang = format!("!!{}", "\u{0301}");
        let mut walk = Qwen::walk(&bang);
        assert_eq!(
            drain(&bang, |buf| walk.fill(buf)),
            ["!!".to_string(), "\u{0301}".to_string()]
        );
        let text = format!("cafe{} {}", "\u{0301}", "word ".repeat(16));
        let mut walk = Qwen::walk(&text);
        let got = drain(&text, |buf| walk.fill(buf));
        assert_eq!(got[0], format!("cafe{}", "\u{0301}"));
        let mut walk = Qwen::walk("12345");
        assert_eq!(
            drain("12345", |buf| walk.fill(buf)),
            ["1", "2", "3", "4", "5"]
        );
    }
}
