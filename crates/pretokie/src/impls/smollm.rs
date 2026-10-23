//! SmolLM split rules.
//!
//! Hugging Face takes each numeric character out first, then runs the GPT-2
//! pattern on what remains. A number is one piece per character, a space does
//! not prefix it, and a whitespace run that ends at a number stays whole.

use super::gpt2::{advance, window};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;

/// SmolLM pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmolLM;

/// Pulls pieces in batches. Create one with [`SmolLM::walk`].
pub struct SmolLMWalk<'a> {
    inner: Walk<'a, SmolLM>,
}

impl SmolLM {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<SmolLM>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> SmolLMWalk<'_> {
        SmolLMWalk {
            inner: Walk::new(text),
        }
    }
}

impl SmolLMWalk<'_> {
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

impl Scheme for SmolLM {
    #[inline]
    fn advance(bytes: &[u8], pos: usize) -> usize {
        advance::<true>(bytes, pos)
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        window::<ISA, true>(bytes, pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_match_fill() {
        let samples = ["", "Hello world", "123 456", "café naïve 東京", "a12b"];
        let long = "Hello world! It's a test.\n\tDon't  stop.\n".repeat(40);
        for text in samples {
            crate::walk::bounds_match_fill::<SmolLM>(text);
        }
        crate::walk::bounds_match_fill::<SmolLM>(&long);
        crate::walk::bounds_match_fill::<SmolLM>(&"word ".repeat(300));
    }


    fn pieces(text: &str) -> Vec<&str> {
        let mut buf = [Span { start: 0, end: 0 }; 16];
        let mut walk = SmolLM::walk(text);
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
        assert_eq!(SmolLM::count(text), expected.len());
    }

    #[test]
    fn letters_match_gpt2() {
        assert_pieces("Hello world", &["Hello", " world"]);
        assert_pieces("don't", &["don", "'t"]);
        assert_pieces("a\n\nb", &["a", "\n", "\n", "b"]);
        assert_pieces("a  b", &["a", " ", " b"]);
    }

    #[test]
    fn each_number_is_its_own_piece() {
        assert_pieces("12345", &["1", "2", "3", "4", "5"]);
        assert_pieces("test 123", &["test", " ", "1", "2", "3"]);
        assert_pieces("a 1 b", &["a", " ", "1", " b"]);
        assert_pieces("abc\n123", &["abc", "\n", "1", "2", "3"]);
        assert_pieces("abc123", &["abc", "1", "2", "3"]);
        assert_pieces("123abc", &["1", "2", "3", "abc"]);
    }

    #[test]
    fn a_run_before_a_number_stays_whole() {
        assert_pieces("a\n\n½ cup", &["a", "\n\n", "½", " cup"]);
        assert_pieces("x  ½", &["x", "  ", "½"]);
        assert_pieces("a\n\n123", &["a", "\n\n", "1", "2", "3"]);
        assert_pieces("  9", &["  ", "9"]);
    }
}
