//! T5 split rules. Drops whitespace and leaves punctuation on the word.
//!
//! A piece is a run of every character that is not whitespace. ALBERT and
//! XLM-RoBERTa use this same cut beside SentencePiece metaspace. This is
//! the cut Hugging Face `WhitespaceSplit` runs.

use crate::walk::classify::{masks_for, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;

/// T5 pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct T5;

/// Pulls pieces in batches. Create one with [`T5::walk`].
pub struct T5Walk<'a> {
    inner: Walk<'a, T5>,
}

impl T5 {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<T5>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> T5Walk<'_> {
        T5Walk {
            inner: Walk::new(text),
        }
    }
}

impl T5Walk<'_> {
    /// Write the next pieces into `out`. Returns how many were written.
    /// A short count means the text is finished.
    pub fn fill(&mut self, out: &mut [Span]) -> usize {
        self.inner.fill(out)
    }
}

impl Scheme for T5 {
    #[inline]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance(bytes, pos)
    }

    #[inline(always)]
    fn align(bytes: &[u8], pos: usize) -> usize {
        align(bytes, pos)
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        window::<ISA>(bytes, pos)
    }
}

fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

/// Next piece start, skipping whitespace.
fn align(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 0x80 {
            if is_ascii_ws(b) {
                pos += 1;
                continue;
            }
            return pos;
        }
        let (ch, n) = decode(bytes, pos);
        if !ch.is_whitespace() {
            return pos;
        }
        pos += n;
    }
    pos
}

fn advance(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 0x80 {
            if is_ascii_ws(b) {
                return pos;
            }
            pos += 1;
            continue;
        }
        let (ch, n) = decode(bytes, pos);
        if ch.is_whitespace() {
            return pos;
        }
        pos += n;
    }
    pos
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

/// The byte at `at` is part of the current run. End of text and whitespace are not.
fn continues_run(bytes: &[u8], at: usize) -> bool {
    if at >= bytes.len() {
        return false;
    }
    let b = bytes[at];
    if b < 0x80 {
        return !is_ascii_ws(b);
    }
    let (ch, _) = decode(bytes, at);
    !ch.is_whitespace()
}

#[inline(always)]
fn window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    let masks = masks_for::<ISA>(bytes, pos);
    let stop = if masks.non_ascii == 0 {
        64
    } else {
        masks.non_ascii.trailing_zeros()
    };
    if stop == 0 {
        return None;
    }
    bits(bytes, pos, masks, stop)
}

#[inline(always)]
fn bits(bytes: &[u8], pos: usize, masks: Masks, stop: u32) -> Option<Window> {
    let keep = if stop >= 64 {
        u64::MAX
    } else {
        (1u64 << stop) - 1
    };
    let ws = (masks.space | masks.newline | masks.other_ws) & keep;
    let word = keep & !ws;
    let touches = stop > 0 && word & (1u64 << (stop - 1)) != 0;
    let continues = touches && continues_run(bytes, pos + stop as usize);

    let mut starts = word & !(word << 1);
    let mut stops = (word & !(word >> 1)) << 1;
    if continues && stop < 64 {
        stops &= !(1u64 << stop);
    }
    if stop < 64 && !continues {
        starts |= 1u64 << stop;
    }
    let finishes = stop >= 64 && !continues;
    if starts & 1 == 0 {
        return None;
    }
    let complete = if finishes {
        starts.count_ones()
    } else {
        starts.count_ones().saturating_sub(1)
    };
    if complete == 0 {
        return None;
    }
    Some(Window {
        starts,
        stops: Some(stops),
        finishes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<String> {
        let mut walk = T5::walk(text);
        let mut buf = [Span { start: 0, end: 0 }; 1];
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
    fn punctuation_stays_on_the_word() {
        assert_eq!(pieces(""), [] as [&str; 0]);
        assert_eq!(pieces("   "), [] as [&str; 0]);
        assert_eq!(pieces("Hello, world!"), ["Hello,", "world!"]);
        assert_eq!(pieces("don't"), ["don't"]);
        assert_eq!(pieces("a-b"), ["a-b"]);
        assert_eq!(pieces("..."), ["..."]);
        assert_eq!(pieces("$5"), ["$5"]);
        assert_eq!(pieces("a\0b"), ["a\0b"]);
        assert_eq!(pieces("中 文"), ["中", "文"]);
    }

    #[test]
    fn rules_hold_inside_a_long_window() {
        let text = format!("Hello, world! don't {}", "word ".repeat(20));
        let got = pieces(&text);
        assert_eq!(&got[..3], ["Hello,", "world!", "don't"]);
        assert_eq!(got.last().map(String::as_str), Some("word"));
    }
}
