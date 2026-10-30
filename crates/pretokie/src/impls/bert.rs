//! BERT split rules.
//!
//! Whitespace is dropped. Each punctuation character is its own piece.
//! Every other character, including letters, digits, marks, symbols, and
//! CJK, stays in the run. Punctuation is the ASCII punctuation ranges plus
//! any character whose Unicode category starts with `P`. This is the cut
//! Hugging Face `BertPreTokenizer` runs.

use crate::walk::classify::{masks_for, Masks};
use crate::walk::{Scheme, Walk, Window};
use crate::Span;
use unicode_general_category::{get_general_category, GeneralCategory};

/// BERT pretokenizer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bert;

/// Pulls pieces in batches. Create one with [`Bert::walk`].
pub struct BertWalk<'a> {
    inner: Walk<'a, Bert>,
}

impl Bert {
    /// Number of pieces in `text`.
    pub fn count(text: &str) -> usize {
        Self::count_upto(text, usize::MAX)
    }

    /// Number of pieces, stopping once the count passes `limit`.
    ///
    /// Returns `limit + 1` when the text has more pieces than `limit`.
    pub fn count_upto(text: &str, limit: usize) -> usize {
        Walk::<Bert>::new(text).count_upto(limit)
    }

    /// Write every piece of `text` that fits in `out`. Returns how many were written.
    pub fn fill(text: &str, out: &mut [Span]) -> usize {
        Self::walk(text).fill(out)
    }

    /// Cursor for pulling pieces in batches.
    pub fn walk(text: &str) -> BertWalk<'_> {
        BertWalk {
            inner: Walk::new(text),
        }
    }
}

impl BertWalk<'_> {
    /// Write the next pieces into `out`. Returns how many were written.
    /// A short count means the text is finished.
    pub fn fill(&mut self, out: &mut [Span]) -> usize {
        self.inner.fill(out)
    }
}

impl Scheme for Bert {
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

fn is_ascii_punct(b: u8) -> bool {
    b.is_ascii_punctuation()
}

fn is_unicode_punct(ch: char) -> bool {
    matches!(
        get_general_category(ch),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
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

fn advance(bytes: &[u8], pos: usize) -> usize {
    let b0 = bytes[pos];
    if b0 < 0x80 {
        if is_ascii_punct(b0) {
            return pos + 1;
        }
        return scan_word(bytes, pos + 1);
    }
    // Kana and unified ideographs are not punctuation, so they stay in the
    // word and never consult the category table.
    if let Some(n) = kana_span(bytes, pos).or_else(|| han_span(bytes, pos)) {
        return scan_word(bytes, pos + n);
    }
    let (ch, n) = decode(bytes, pos);
    if is_unicode_punct(ch) {
        return pos + n;
    }
    scan_word(bytes, pos + n)
}

fn scan_word(bytes: &[u8], mut pos: usize) -> usize {
    while pos < bytes.len() {
        let b = bytes[pos];
        if b < 0x80 {
            if is_ascii_ws(b) || is_ascii_punct(b) {
                return pos;
            }
            pos += 1;
            continue;
        }
        if let Some(n) = kana_span(bytes, pos).or_else(|| han_span(bytes, pos)) {
            pos += n;
            continue;
        }
        let (ch, n) = decode(bytes, pos);
        if ch.is_whitespace() || is_unicode_punct(ch) {
            return pos;
        }
        pos += n;
    }
    pos
}

/// Three bytes of hiragana or katakana, excluding the two punctuation marks.
/// U+30A0 is a dash and U+30FB is a middle dot. Everything else in
/// U+3040..=U+30FF stays in the word, including the combining voicing marks.
fn kana_span(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 3 > bytes.len() {
        return None;
    }
    let b0 = bytes[pos];
    let b1 = bytes[pos + 1];
    let b2 = bytes[pos + 2];
    if b0 != 0xE3 || b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
        return None;
    }
    let word = match b1 {
        0x81 => true,
        0x82 => b2 != 0xA0,
        0x83 => b2 != 0xBB,
        _ => false,
    };
    word.then_some(3)
}

/// Three bytes of a unified ideograph, U+4E00..=U+9FFF. Those are
/// Other_Letter: not whitespace and not punctuation.
fn han_span(bytes: &[u8], pos: usize) -> Option<usize> {
    if pos + 3 > bytes.len() {
        return None;
    }
    let b0 = bytes[pos];
    let b1 = bytes[pos + 1];
    let b2 = bytes[pos + 2];
    if b1 & 0xC0 != 0x80 || b2 & 0xC0 != 0x80 {
        return None;
    }
    let han = matches!(b0, 0xE5..=0xE8) || (b0 == 0xE4 && b1 >= 0xB8) || b0 == 0xE9;
    han.then_some(3)
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

/// The byte at `at` continues a word. End of text, whitespace, and
/// punctuation do not.
fn continues_word(bytes: &[u8], at: usize) -> bool {
    if at >= bytes.len() {
        return false;
    }
    let b = bytes[at];
    if b < 0x80 {
        return !is_ascii_ws(b) && !is_ascii_punct(b);
    }
    if kana_span(bytes, at).is_some() || han_span(bytes, at).is_some() {
        return true;
    }
    let (ch, _) = decode(bytes, at);
    !ch.is_whitespace() && !is_unicode_punct(ch)
}

#[inline(always)]
fn window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
    if pos + 64 > bytes.len() {
        return None;
    }
    // A non-ASCII lead makes the window empty: the classifier stops at byte 0.
    // Chinese text starts almost every piece on an ideograph, so the 64-byte
    // classify would be discarded.
    if bytes[pos] >= 0x80 {
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
    let punct = punct & keep;
    let ws = (masks.space | masks.newline | masks.other_ws) & keep;
    let word = keep & !ws & !punct;
    let touches = stop > 0 && word & (1u64 << (stop - 1)) != 0;
    let continues = touches && continues_word(bytes, pos + stop as usize);

    let mut starts = (word & !(word << 1)) | punct;
    let mut stops = ((word & !(word >> 1)) << 1) | (punct << 1);
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
            (_mm256_movemask_epi8(m0) as u32 as u64) | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };
        let le = |v: __m256i, lim: __m256i| -> __m256i { _mm256_cmpeq_epi8(_mm256_min_epu8(v, lim), v) };
        let p = bytes.as_ptr().add(pos);
        let v0 = _mm256_loadu_si256(p as *const _);
        let v1 = _mm256_loadu_si256(p.add(32) as *const _);
        let range = |lo: u8, hi: u8| {
            let lim = _mm256_set1_epi8((hi - lo) as i8);
            let a = le(_mm256_sub_epi8(v0, _mm256_set1_epi8(lo as i8)), lim);
            let b = le(_mm256_sub_epi8(v1, _mm256_set1_epi8(lo as i8)), lim);
            pack(a, b)
        };
        range(0x21, 0x2F) | range(0x3A, 0x40) | range(0x5B, 0x60) | range(0x7B, 0x7E)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(text: &str) -> Vec<String> {
        let mut walk = Bert::walk(text);
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
    fn whitespace_drops_and_punctuation_is_alone() {
        assert_eq!(pieces(""), [] as [&str; 0]);
        assert_eq!(pieces("   "), [] as [&str; 0]);
        assert_eq!(pieces("Hello, world!"), ["Hello", ",", "world", "!"]);
        assert_eq!(pieces("don't"), ["don", "'", "t"]);
        assert_eq!(pieces("18th"), ["18th"]);
        assert_eq!(pieces("..."), [".", ".", "."]);
        assert_eq!(pieces("a\0b"), ["a\0b"]);
        assert_eq!(pieces("€5"), ["€5"]);
        assert_eq!(pieces("$5"), ["$", "5"]);
    }

    #[test]
    fn rules_hold_inside_a_long_window() {
        let text = format!("Hello, world! don't 18th {}", "word ".repeat(20));
        let got = pieces(&text);
        assert_eq!(&got[..7], ["Hello", ",", "world", "!", "don", "'", "t"]);
        assert!(got.iter().any(|p| p == "18th"), "{got:?}");
        assert_eq!(got.last().map(String::as_str), Some("word"));
    }
}
