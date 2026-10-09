//! BERT pretokenizer — single-pass, zero allocation.
//!
//! Matches HuggingFace's `BertPreTokenizer`:
//! - Whitespace (`char::is_whitespace`) is a delimiter only (stripped, never part of a piece)
//! - Each punctuation character (ASCII punct or Unicode P) is a separate piece:
//!   "..." → [".", ".", "."]
//! - Everything else stays together in words: "18th" → ["18th"], "€100" → ["€100"]
//! - CJK ideographs are NOT split here: HF isolates them in `BertNormalizer`
//!   (`handle_chinese_chars`) before pretokenization, and so does tokie's normalizer
//! - No contractions, no space prefix

use crate::util::decode_utf8;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Fast ASCII punctuation check (same ranges as HuggingFace's `_is_punctuation`).
#[inline(always)]
fn is_ascii_punct(b: u8) -> bool {
    matches!(b, 33..=47 | 58..=64 | 91..=96 | 123..=126)
}

/// ASCII `char::is_whitespace` (includes VT and FF).
#[inline(always)]
fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

/// Check if a Unicode character is punctuation.
///
/// Matches HuggingFace's `_is_punctuation`: ASCII punct ranges + Unicode P* categories.
#[inline]
fn is_unicode_punct(c: char) -> bool {
    let cp = c as u32;
    if cp < 0x80 {
        return is_ascii_punct(cp as u8);
    }
    matches!(
        get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::OtherPunctuation
            | GeneralCategory::OpenPunctuation
    )
}

pub struct Bert<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Bert<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    /// Skip whitespace (delimiter in BERT).
    #[inline(always)]
    fn skip_whitespace(&mut self) {
        while self.pos < self.len {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if is_ascii_ws(b) {
                self.pos += 1;
            } else if b < 0x80 {
                return;
            } else {
                let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
                if c.is_whitespace() { self.pos += cl; } else { return; }
            }
        }
    }

    /// Scan word characters: anything that is neither whitespace nor punctuation.
    #[inline(always)]
    fn scan_word(&mut self) {
        while self.pos < self.len {
            let b = unsafe { *self.bytes.get_unchecked(self.pos) };
            if b.is_ascii_alphanumeric() {
                self.pos += 1;
            } else if b < 0x80 {
                if is_ascii_ws(b) || is_ascii_punct(b) {
                    return;
                }
                self.pos += 1; // ASCII control: part of the word, as in HF
            } else {
                let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
                if c.is_whitespace() || is_unicode_punct(c) {
                    return;
                }
                // Letter, number, mark, symbol, CJK, … → continues word
                self.pos += cl;
            }
        }
    }
}

impl<'a> Iterator for Bert<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        // Skip whitespace
        self.skip_whitespace();
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = unsafe { *self.bytes.get_unchecked(self.pos) };

        if b < 0x80 {
            if is_ascii_punct(b) {
                // Single punctuation character
                self.pos += 1;
            } else {
                // Any other ASCII char (letter, digit, control) — scan word
                self.pos += 1;
                self.scan_word();
            }
        } else {
            let (c, cl) = decode_utf8(unsafe { self.bytes.get_unchecked(self.pos..) });
            if is_unicode_punct(c) {
                // Single Unicode punctuation character
                self.pos += cl;
            } else {
                // Non-ASCII letter, number, mark, or other → start word
                self.pos += cl;
                self.scan_word();
            }
        }

        Some(unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) })
    }
}
