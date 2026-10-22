//! A parsed SentencePiece charsmap.
//!
//! Printable ASCII is copied when the map leaves it alone. A control byte
//! uses the ASCII cache. A non-ASCII span, plus one preceding byte a
//! combining mark can absorb, goes through the trie.

use unicode_segmentation::UnicodeSegmentation;

use crate::charsmap::{Charsmap, CharsmapError};
use crate::formula::{Formula, ISA_SCALAR};
use crate::scan::printable_prefix;

pub struct Precompiled {
    map: Charsmap,
}

impl Precompiled {
    /// Parse a `precompiled_charsmap` blob.
    pub fn parse(blob: &[u8]) -> Result<Self, CharsmapError> {
        Ok(Self {
            map: Charsmap::parse(blob)?,
        })
    }

    pub fn write(&self, text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(self, text, at_start, out)
    }
}

impl Formula for Precompiled {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        if !self.map.fast_scan {
            return self.drive_slow(text, out);
        }
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
            {
                // SAFETY: AVX-512F and AVX-512BW were just detected.
                return unsafe { self.drive_avx512(text, out) };
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 was just detected.
                return unsafe { self.drive_avx2(text, out) };
            }
        }
        self.drive_isa::<ISA_SCALAR>(text, out)
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        let _ = text;
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        self.map.normalize_graphemes(text, out);
        text.len().max(1)
    }
}

impl Precompiled {
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn drive_avx512(&self, text: &str, out: &mut String) -> bool {
        self.drive_isa::<{ crate::formula::ISA_AVX512 }>(text, out)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn drive_avx2(&self, text: &str, out: &mut String) -> bool {
        self.drive_isa::<{ crate::formula::ISA_AVX2 }>(text, out)
    }

    #[inline(always)]
    fn drive_isa<const ISA: u8>(&self, text: &str, out: &mut String) -> bool {
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut started = false;
        let mut tmp = String::new();
        while i < bytes.len() {
            let run = printable_prefix::<ISA>(&bytes[i..]);
            let j = i + run;
            if j >= bytes.len() {
                if started {
                    out.push_str(&text[i..]);
                }
                return started;
            }
            let b = bytes[j];
            if b < 0x80 {
                let (repl, n, same) = self.ascii_unit(bytes, j);
                if same && !started {
                    i = j + n;
                    continue;
                }
                if !started {
                    out.reserve(text.len());
                    out.push_str(&text[..j]);
                    started = true;
                } else {
                    out.push_str(&text[i..j]);
                }
                if let Some(repl) = repl {
                    out.push_str(repl);
                } else {
                    out.push(b as char);
                }
                i = j + n;
                continue;
            }
            let span_start = if j > i { j - 1 } else { j };
            let span_end = non_ascii_end(bytes, j);
            let region = &text[span_start..span_end];
            if let Some((piece, tail, changed)) = one_scalar(&self.map, region) {
                if !changed && !started {
                    i = span_end;
                    continue;
                }
                if !started {
                    out.reserve(text.len());
                    out.push_str(&text[..span_start]);
                    started = true;
                } else {
                    out.push_str(&text[i..span_start]);
                }
                out.push_str(piece);
                out.push_str(tail);
                i = span_end;
                continue;
            }
            tmp.clear();
            self.map.normalize_graphemes(region, &mut tmp);
            if !started && tmp.as_str() == region {
                i = span_end;
                continue;
            }
            if !started {
                out.reserve(text.len());
                out.push_str(&text[..span_start]);
                started = true;
            } else {
                out.push_str(&text[i..span_start]);
            }
            out.push_str(&tmp);
            i = span_end;
        }
        started
    }

    fn ascii_unit<'a>(&'a self, bytes: &[u8], j: usize) -> (Option<&'a str>, usize, bool) {
        if bytes[j] == b'\r' && bytes.get(j + 1) == Some(&b'\n') {
            return match self.map.crlf() {
                Some(repl) => (Some(repl), 2, false),
                None => (None, 2, true),
            };
        }
        match self.map.ascii(bytes[j]) {
            Some(repl) => (Some(repl), 1, false),
            None => (None, 1, true),
        }
    }

    fn drive_slow(&self, text: &str, out: &mut String) -> bool {
        let mut started = false;
        for (g_idx, grapheme) in text.grapheme_indices(true) {
            if grapheme.len() < 6
                && let Some(repl) = self.map.transform(grapheme)
            {
                if !started {
                    out.push_str(&text[..g_idx]);
                    started = true;
                }
                out.push_str(repl);
                continue;
            }
            let g_bytes = g_idx;
            for (c_idx, c) in grapheme.char_indices() {
                let part = &grapheme[c_idx..c_idx + c.len_utf8()];
                if let Some(repl) = self.map.transform(part) {
                    if !started {
                        out.push_str(&text[..g_bytes + c_idx]);
                        started = true;
                    }
                    out.push_str(repl);
                } else if started {
                    out.push(c);
                }
            }
        }
        started
    }
}

/// One scalar, plus an optional trailing ASCII byte a cluster can absorb.
///
/// `None` means the span needs the grapheme walk. The piece and the tail
/// together are the normalized span.
fn one_scalar<'a>(map: &'a Charsmap, region: &'a str) -> Option<(&'a str, &'a str, bool)> {
    let bytes = region.as_bytes();
    let mut core_end = region.len();
    if bytes.len() > 1 && bytes[bytes.len() - 1] < 0x80 {
        core_end -= 1;
    }
    let core = &region[..core_end];
    let core_bytes = core.as_bytes();
    if core_bytes.is_empty() {
        return None;
    }
    let width = match core_bytes[0] {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    };
    if core_bytes.len() != width {
        return None;
    }
    let tail = &region[core_end..];
    let c = core.chars().next().unwrap();
    match map.rewrite_char(c) {
        Some(norm) => Some((norm, tail, true)),
        None => Some((region, "", false)),
    }
}

/// Exclusive end of a non-ASCII span that starts at `j`.
fn non_ascii_end(bytes: &[u8], j: usize) -> usize {
    let mut k = j;
    while k < bytes.len() {
        let c = bytes[k];
        if c < 0x20 || c == 0x7F {
            return k;
        }
        if c < 0x80 && bytes.get(k + 1).is_none_or(|n| *n < 0x80) {
            return k + 1;
        }
        k += 1;
    }
    bytes.len()
}
