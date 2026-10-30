//! GPT-2 split rules.
//!
//! The pattern is `'s|'t|'re|'ve|'m|'ll|'d`, an optional space plus letters,
//! an optional space plus numbers, an optional space plus other characters,
//! then a whitespace run. Only a single ASCII space prefixes the next piece.
//! A tab or a newline does not.

use crate::walk::classify::{class_of, masks_for, Class, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;

/// GPT-2 pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Gpt2;

/// Pulls pieces in batches. Create one with [`Gpt2::walk`].
pub struct Gpt2Walk<'a> {
    inner: Walk<'a, Gpt2>,
}

impl Gpt2 {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<Gpt2>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> Gpt2Walk<'_> {
        Gpt2Walk {
            inner: Walk::new(text),
        }
    }

}

impl Gpt2Walk<'_> {
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

impl Scheme for Gpt2 {
    #[inline]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance::<false>(bytes, pos)
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        window::<ISA, false>(bytes, pos)
    }

}

#[inline]
fn is_ascii_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline]
fn is_ascii_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline]
fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

/// `bytes[at]` is whitespace, including a continuation byte of a whitespace character.
pub(crate) fn byte_is_whitespace(bytes: &[u8], at: usize) -> bool {
    if at >= bytes.len() {
        return false;
    }
    let b = bytes[at];
    if b < 0x80 {
        is_ascii_ws(b)
    } else if b & 0xC0 == 0x80 {
        true
    } else {
        let (ch, _) = decode_char(bytes, at);
        class_of(ch) == Class::Whitespace
    }
}

/// `bytes[at]` starts a numeric character. A continuation byte does not.
pub(crate) fn byte_is_number(bytes: &[u8], at: usize) -> bool {
    if at >= bytes.len() {
        return false;
    }
    let b = bytes[at];
    if b < 0x80 {
        return is_ascii_digit(b);
    }
    if b & 0xC0 == 0x80 {
        return false;
    }
    let (ch, _) = decode_char(bytes, at);
    class_of(ch) == Class::Number
}

/// `bytes[i..]` is valid UTF-8 and `i` is a character boundary.
///
/// Only the bytes of this character are checked. Validating the rest of
/// the text here would rescan the suffix on every non-ASCII character.
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

/// End of the piece at `start`.
///
/// `SINGLE_DIGITS` keeps each numeric character as its own piece and leaves
/// a whitespace run whole when a number follows it.
pub(crate) fn advance<const SINGLE_DIGITS: bool>(bytes: &[u8], start: usize) -> usize {
    let b0 = bytes[start];
    if is_ascii_letter(b0) {
        return scan_letters(bytes, start + 1);
    }
    if b0 == b' ' {
        return advance_space::<SINGLE_DIGITS>(bytes, start);
    }
    if b0 >= 0x80 {
        let (ch, n) = decode_char(bytes, start);
        let next = start + n;
        return match class_of(ch) {
            Class::Letter => scan_letters(bytes, next),
            Class::Number if SINGLE_DIGITS => next,
            Class::Number => scan_digits(bytes, next),
            Class::Whitespace => advance_ws::<SINGLE_DIGITS>(bytes, next, start),
            Class::Other => scan_other(bytes, next),
        };
    }
    if is_ascii_digit(b0) {
        if SINGLE_DIGITS {
            return start + 1;
        }
        return scan_digits(bytes, start + 1);
    }
    if b0 == b'\'' {
        return advance_apostrophe(bytes, start);
    }
    if is_ascii_ws(b0) {
        return advance_ws::<SINGLE_DIGITS>(bytes, start + 1, start);
    }
    scan_other(bytes, start + 1)
}

fn advance_space<const SINGLE_DIGITS: bool>(bytes: &[u8], start: usize) -> usize {
    if start + 1 >= bytes.len() {
        return start + 1;
    }
    let b1 = bytes[start + 1];
    if is_ascii_letter(b1) {
        return scan_letters(bytes, start + 2);
    }
    if is_ascii_digit(b1) {
        if SINGLE_DIGITS {
            return start + 1;
        }
        return scan_digits(bytes, start + 2);
    }
    if b1 >= 0x80 {
        let (ch, n) = decode_char(bytes, start + 1);
        let next = start + 1 + n;
        return match class_of(ch) {
            Class::Letter => scan_letters(bytes, next),
            Class::Number if SINGLE_DIGITS => start + 1,
            Class::Number => scan_digits(bytes, next),
            Class::Whitespace => advance_ws::<SINGLE_DIGITS>(bytes, next, start),
            Class::Other => scan_other(bytes, next),
        };
    }
    if is_ascii_ws(b1) {
        return advance_ws::<SINGLE_DIGITS>(bytes, start + 1, start);
    }
    scan_other(bytes, start + 2)
}

fn advance_apostrophe(bytes: &[u8], start: usize) -> usize {
    match bytes.get(start + 1).copied() {
        Some(b's' | b'd' | b'm' | b't') => start + 2,
        Some(b'l') if bytes.get(start + 2) == Some(&b'l') => start + 3,
        Some(b'v') if bytes.get(start + 2) == Some(&b'e') => start + 3,
        Some(b'r') if bytes.get(start + 2) == Some(&b'e') => start + 3,
        _ => scan_other(bytes, start + 1),
    }
}

fn scan_letters(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() {
        let b = bytes[p];
        if is_ascii_letter(b) {
            p += 1;
            continue;
        }
        if b < 0x80 {
            break;
        }
        let (ch, n) = decode_char(bytes, p);
        if class_of(ch) != Class::Letter {
            break;
        }
        p += n;
    }
    p
}

fn scan_digits(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() {
        let b = bytes[p];
        if is_ascii_digit(b) {
            p += 1;
            continue;
        }
        if b < 0x80 {
            break;
        }
        let (ch, n) = decode_char(bytes, p);
        if class_of(ch) != Class::Number {
            break;
        }
        p += n;
    }
    p
}

fn scan_other(bytes: &[u8], mut p: usize) -> usize {
    while p < bytes.len() {
        let b = bytes[p];
        if b < 0x80 {
            if is_ascii_letter(b) || is_ascii_digit(b) || is_ascii_ws(b) {
                break;
            }
            p += 1;
            continue;
        }
        let (ch, n) = decode_char(bytes, p);
        if class_of(ch) != Class::Other {
            break;
        }
        p += n;
    }
    p
}

/// Consume a whitespace run that started at `token_start`.
///
/// `scan` is the first byte not yet classified. When content follows a run
/// of two or more whitespace characters, the last of those characters is
/// left for the next piece. A trailing run is consumed whole.
fn advance_ws<const SINGLE_DIGITS: bool>(bytes: &[u8], scan: usize, token_start: usize) -> usize {
    let mut p = scan;
    while p < bytes.len() {
        let b = bytes[p];
        if b < 0x80 {
            if !is_ascii_ws(b) {
                break;
            }
            p += 1;
            continue;
        }
        let (ch, n) = decode_char(bytes, p);
        if class_of(ch) != Class::Whitespace {
            break;
        }
        p += n;
    }
    if p < bytes.len() && !(SINGLE_DIGITS && byte_is_number(bytes, p)) {
        let mut last = p - 1;
        while last > token_start && bytes[last] & 0xC0 == 0x80 {
            last -= 1;
        }
        if last > token_start {
            return last;
        }
    }
    p
}

/// Leads of multibyte whitespace characters that end a run because content follows.
///
/// A multibyte whitespace character occupies several whitespace bytes. The
/// split belongs on its lead, not on its last byte. ASCII whitespace is
/// handled in the kernel.
///
/// The bits match a walk of each whitespace lead. A character's width is
/// how many continuation bits follow its lead, and the bit is set on the
/// lead when the next character is not whitespace. Outlined so the walk
/// stays out of the ASCII window.
#[inline(never)]
fn ws_multibyte_splits<const SINGLE_DIGITS: bool>(
    whitespace: u64,
    cont: u64,
    digit: u64,
    bytes: &[u8],
    pos: usize,
) -> u64 {
    let leads = whitespace & !cont;
    let c1 = cont >> 1;
    let c2 = cont >> 2;
    let c3 = cont >> 3;
    let w1 = leads & !c1;
    let w2 = leads & c1 & !c2;
    let w3 = leads & c1 & c2 & !c3;
    let w4 = leads & c1 & c2 & c3;
    let not_ws = !whitespace;
    let mut split = (w1 & (not_ws >> 1))
        | (w2 & (not_ws >> 2))
        | (w3 & (not_ws >> 3))
        | (w4 & (not_ws >> 4));
    if SINGLE_DIGITS {
        split &= !((w1 & (digit >> 1))
            | (w2 & (digit >> 2))
            | (w3 & (digit >> 3))
            | (w4 & (digit >> 4)));
    }
    // A lead whose next byte sits past the window looks at `pos + 64`.
    // At the end of the text there is no such byte. A run that reaches
    // the end stays whole; a run with content after it already split above.
    let edge = (w1 & (1u64 << 63)) | (w2 & (1u64 << 62)) | (w3 & (1u64 << 61)) | (w4 & (1u64 << 60));
    if edge != 0 && pos + 64 < bytes.len() {
        let next_is_ws = byte_is_whitespace(bytes, pos + 64);
        let next_is_number = SINGLE_DIGITS && byte_is_number(bytes, pos + 64);
        if !next_is_ws && !next_is_number {
            split |= edge;
        } else {
            split &= !edge;
        }
    }
    split
}

/// Fold non-ASCII characters into the class masks.
///
/// Every in-window byte of a character gets that character's class, so the
/// masks treat the character as one run. Continuation bytes are
/// returned separately and are never piece starts. The stop index is the
/// first byte that cannot be classified, or 64 when the window is complete.
fn absorb_non_ascii(bytes: &[u8], pos: usize, masks: &Masks) -> (u64, u64, u64, u64, u32) {
    let mut letter = masks.letter;
    let mut digit = masks.digit;
    let mut whitespace = 0u64;
    let mut cont = 0u64;
    let mut bits = masks.non_ascii;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        let b = bytes[pos + i];
        if b & 0xC0 == 0x80 {
            return (letter, digit, whitespace, cont, i as u32);
        }
        let width = if b < 0xE0 {
            2
        } else if b < 0xF0 {
            3
        } else {
            4
        };
        if pos + i + width > bytes.len() {
            return (letter, digit, whitespace, cont, i as u32);
        }
        let (ch, n) = decode_char(bytes, pos + i);
        let marked = n.min(64 - i);
        let char_bits = ((1u64 << marked) - 1) << i;
        bits &= !char_bits;
        if marked > 1 {
            cont |= char_bits ^ (1u64 << i);
        }
        match class_of(ch) {
            Class::Letter => letter |= char_bits,
            Class::Number => digit |= char_bits,
            Class::Whitespace => whitespace |= char_bits,
            Class::Other => {}
        }
    }
    (letter, digit, whitespace, cont, 64)
}

/// Piece starts for the 64-byte window at `pos`, classified once.
///
/// Non-ASCII characters are classified in a second pass and stay in the
/// window. `None` means a tail shorter than 64 bytes, a piece longer than
/// the window, or a byte the second pass cannot classify. The highest start
/// is unfinished unless the window reaches the end of the text, or a
/// contraction consumes the rest of it.
#[inline(always)]
pub(crate) fn window<const ISA: u8, const SINGLE_DIGITS: bool>(
    bytes: &[u8],
    pos: usize,
) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    let (letter, digit, uni_ws, cont, stop) = if masks.non_ascii == 0 {
        (masks.letter, masks.digit, 0, 0, 64)
    } else {
        absorb_non_ascii(bytes, pos, &masks)
    };
    if stop == 0 {
        return None;
    }
    let eof = pos + 64 == bytes.len();
    let whitespace = masks.space | masks.newline | masks.other_ws | uni_ws;
    let other = !(letter | digit | whitespace);
    let digit_run = if SINGLE_DIGITS {
        0
    } else {
        digit & (digit << 1)
    };
    let cont_same = (letter & (letter << 1)) | digit_run | (other & (other << 1));
    let after_space = masks.space << 1;
    let prefixed = if SINGLE_DIGITS {
        after_space & !digit
    } else {
        after_space
    };
    let content_start = !whitespace & !cont_same & !prefixed;

    // Peel the last byte of a whitespace run when content follows it. That
    // includes the final window: the end of the text only means there is
    // no byte past the window to consult. A run that reaches the end stays
    // whole, because nothing inside the window follows it.
    let split_ok = if whitespace == 0 {
        0
    } else if cont == 0 {
        let mut split = whitespace & (!whitespace >> 1);
        if SINGLE_DIGITS {
            split &= !(digit >> 1);
        }
        if !eof {
            // SAFETY: `eof` is false, so `pos + 64` is inside `bytes`.
            let next = unsafe { *bytes.get_unchecked(pos + 64) };
            let next_is_ws = if next < 0x80 {
                is_ascii_ws(next)
            } else {
                byte_is_whitespace(bytes, pos + 64)
            };
            let next_is_number = SINGLE_DIGITS && byte_is_number(bytes, pos + 64);
            if !next_is_ws && !next_is_number {
                split |= whitespace & (1u64 << 63);
            }
        }
        split
    } else {
        ws_multibyte_splits::<SINGLE_DIGITS>(whitespace, cont, digit, bytes, pos)
    };
    let run_start = !(whitespace << 1);
    let ws_start = whitespace & (run_start | split_ok);
    let mut boundary = content_start | ws_start;
    let mut finishes = eof;
    if stop < 64 {
        boundary &= (1u64 << stop) - 1;
        finishes = false;
    }

    let mut apostrophes = masks.apostrophe & boundary;
    while apostrophes != 0 {
        let i = apostrophes.trailing_zeros() as usize;
        apostrophes &= apostrophes - 1;
        let Some(k) = contraction_len(bytes, pos + i) else {
            continue;
        };
        let end = i + k;
        if end > 63 {
            if i < 63 {
                boundary &= (1u64 << (i + 1)) - 1;
            }
            finishes = eof && end == 64;
            break;
        }
        boundary &= !(1u64 << (i + 1));
        boundary |= 1u64 << end;
    }
    boundary &= !cont;

    if boundary & 1 == 0 || (!finishes && boundary == 1) {
        return None;
    }
    Some(Window {
        starts: boundary,
        stops: None,
        finishes,
    })
}

/// Length of a lowercase contraction starting at an apostrophe, including
/// the apostrophe. `None` when the following bytes are not `'s`, `'t`,
/// `'m`, `'d`, `'ll`, `'ve`, or `'re`.
fn contraction_len(bytes: &[u8], at: usize) -> Option<usize> {
    match bytes.get(at + 1).copied() {
        Some(b's' | b'd' | b'm' | b't') => Some(2),
        Some(b'l') if bytes.get(at + 2) == Some(&b'l') => Some(3),
        Some(b'v') if bytes.get(at + 2) == Some(&b'e') => Some(3),
        Some(b'r') if bytes.get(at + 2) == Some(&b'e') => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<&str> {
        let mut buf = [Span { start: 0, end: 0 }; 8];
        let mut walk = Gpt2::walk(text);
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
        assert_eq!(Gpt2::count(text), expected.len());
    }

    #[test]
    fn basic_english() {
        assert_pieces("Hello world", &["Hello", " world"]);
        assert_pieces(
            "Hello world! It's a test.",
            &["Hello", " world", "!", " It", "'s", " a", " test", "."],
        );
        assert_pieces("don't", &["don", "'t"]);
        assert_pieces("It's", &["It", "'s"]);
    }

    #[test]
    fn contractions_are_lowercase_only() {
        assert_pieces("'s", &["'s"]);
        assert_pieces("'t", &["'t"]);
        assert_pieces("'m", &["'m"]);
        assert_pieces("'d", &["'d"]);
        assert_pieces("'ll", &["'ll"]);
        assert_pieces("'ve", &["'ve"]);
        assert_pieces("'re", &["'re"]);
        assert_pieces("'S", &["'", "S"]);
        assert_pieces("'hello", &["'", "hello"]);
        assert_pieces("'l", &["'", "l"]);
        assert_pieces(" 's", &[" '", "s"]);
    }

    #[test]
    fn whitespace() {
        assert_pieces("\thello", &["\t", "hello"]);
        assert_pieces("\t\thello", &["\t", "\t", "hello"]);
        assert_pieces("  hello", &[" ", " hello"]);
        assert_pieces("   hello", &["  ", " hello"]);
        assert_pieces("hello\n\n", &["hello", "\n\n"]);
        assert_pieces(".\n\nSessions", &[".", "\n", "\n", "Sessions"]);
        assert_pieces("hello  ", &["hello", "  "]);
        assert_pieces("a  b", &["a", " ", " b"]);
        assert_pieces("a\tb", &["a", "\t", "b"]);
        assert_pieces("hello\n world", &["hello", "\n", " world"]);
        assert_pieces("hello\nworld", &["hello", "\n", "world"]);
        assert_pieces(" \thello", &[" ", "\t", "hello"]);
        assert_pieces("", &[]);
        assert_pieces(" ", &[" "]);
        assert_pieces("\n", &["\n"]);
    }

    #[test]
    fn punctuation_and_digits() {
        assert_pieces("a!!b", &["a", "!!", "b"]);
        assert_pieces("you?!?", &["you", "?!?"]);
        assert_pieces("123 456", &["123", " 456"]);
        assert_pieces("a !", &["a", " !"]);
        assert_pieces("123abc", &["123", "abc"]);
        assert_pieces("abc123", &["abc", "123"]);
    }

    #[test]
    fn unicode_inside_a_long_window() {
        let text = format!("{}café \u{00A0}中文 e\u{0301}", "Hello world! ".repeat(6));
        assert!(text.len() > 64);
        let mut expected = Vec::new();
        let mut pos = 0;
        while pos < text.len() {
            let end = advance::<false>(text.as_bytes(), pos);
            expected.push(&text[pos..end]);
            pos = end;
        }
        assert_pieces(&text, &expected);
    }

    #[test]
    fn unicode() {
        assert_pieces("café", &["café"]);
        assert_pieces("中文", &["中文"]);
        assert_pieces("a\u{00A0}b", &["a", "\u{00A0}", "b"]);
        assert_pieces("e\u{0301}", &["e", "\u{0301}"]);
    }

    #[test]
    fn count_upto_stops_past_the_limit() {
        let text = "Hello world! It's a test.";
        assert_eq!(Gpt2::count(text), 8);
        assert_eq!(Gpt2::count_upto(text, 3), 4);
        assert_eq!(Gpt2::count_upto(text, 8), 8);
        assert_eq!(Gpt2::count_upto(text, 100), 8);
        assert_eq!(Gpt2::count_upto("", 0), 0);
    }

    #[test]
    fn one_span_at_a_time_matches_a_full_batch() {
        let text = "Hello world! It's a test.\n\tDon't  stop.\n".repeat(8);
        let mut one = [Span { start: 0, end: 0 }];
        let mut many = [Span { start: 0, end: 0 }; 32];
        let mut a = Gpt2::walk(&text);
        let mut b = Gpt2::walk(&text);
        let mut n = 0;
        loop {
            let n2 = b.fill(&mut many);
            if n2 == 0 {
                assert_eq!(a.fill(&mut one), 0);
                break;
            }
            for span in &many[..n2] {
                assert_eq!(a.fill(&mut one), 1);
                assert_eq!(one[0], *span);
                n += 1;
            }
        }
        assert_eq!(n, Gpt2::count(&text));
    }

    #[test]
    fn a_newline_run_in_the_final_window_still_peels() {
        let tail = "Justice.\n\nSessions and his aides";
        let pad = 64 - (tail.len() % 64);
        let text = format!("{}{tail}", "a".repeat(pad % 64));
        assert_eq!(text.len() % 64, 0);
        let mut expected = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            let end = advance::<false>(bytes, pos);
            expected.push(&text[pos..end]);
            pos = end;
        }
        assert_pieces(&text, &expected);
        assert!(
            expected.windows(2).any(|pair| pair == ["\n", "\n"]),
            "the newline run must stay two pieces: {expected:?}"
        );
    }

    #[test]
    fn long_ascii_uses_the_class_masks() {
        let text = "Hello world! It's a test.\n\tDon't  stop.\n".repeat(8);
        assert!(text.len() > 64);
        assert_pieces(&text, &{
            let mut expected = Vec::new();
            let mut pos = 0;
            while pos < text.len() {
                let end = advance::<false>(text.as_bytes(), pos);
                expected.push(&text[pos..end]);
                pos = end;
            }
            expected
        });
    }

    #[test]
    fn fill_from_the_start_is_one_batch() {
        let text = "Hello world";
        let mut out = [Span { start: 0, end: 0 }; 1];
        assert_eq!(Gpt2::fill(text, &mut out), 1);
        assert_eq!(&text[out[0].start as usize..out[0].end as usize], "Hello");
    }

    #[test]
    fn bounds_match_fill() {
        let mut samples = vec![
            String::new(),
            "Hello world".to_string(),
            "it's a test\n".to_string(),
            "123 456".to_string(),
            "café naïve 東京".to_string(),
            "a".repeat(80),
            "Hello world! It's a test.\n\tDon't  stop.\n".repeat(40),
        ];
        samples.push("word ".repeat(300));
        for text in samples {
            let mut spans = vec![Span { start: 0, end: 0 }; text.len() + 1];
            let n = {
                let mut walk = Gpt2::walk(&text);
                let mut at = 0;
                let mut buf = [Span { start: 0, end: 0 }; 64];
                loop {
                    let k = walk.fill(&mut buf);
                    if k == 0 {
                        break;
                    }
                    spans[at..at + k].copy_from_slice(&buf[..k]);
                    at += k;
                }
                at
            };
            for limit in [1usize, 7, 64, 256] {
                let mut walk = Gpt2::walk(&text);
                let mut buf = vec![0u32; limit + 80];
                let mut got = Vec::new();
                loop {
                    let k = walk.bounds(&mut buf, limit);
                    if k == 0 {
                        break;
                    }
                    for i in 0..k {
                        got.push(Span {
                            start: buf[i],
                            end: buf[i + 1],
                        });
                    }
                    if k < limit {
                        break;
                    }
                }
                assert_eq!(got.len(), n, "limit {limit} text len {}", text.len());
                assert_eq!(&got[..], &spans[..n], "limit {limit}");
            }
        }
    }

}
