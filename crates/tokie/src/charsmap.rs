//! SentencePiece `precompiled_charsmap` normalization.
//!
//! SentencePiece models embed their normalizer as a binary `precompiled_charsmap`
//! blob: a darts-clone double-array trie mapping source byte sequences to offsets
//! into a NUL-separated replacement string. HuggingFace `tokenizers` applies this
//! blob per grapheme cluster (via the `spm_precompiled` crate); this module
//! reimplements that exact behavior so models like XLM-RoBERTa match HF
//! per-character (a category-level approximation cannot: e.g. the xlm-roberta
//! charsmap maps ZWNJ to space but keeps U+00AD and C1 controls verbatim).
//!
//! Blob layout: `[trie_byte_len: u32 le][trie: [u32]][replacements: str]`.

use unicode_segmentation::UnicodeSegmentation;

/// Error parsing a `precompiled_charsmap` blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharsmapError {
    /// Blob too short or trie section out of bounds.
    Truncated,
    /// Replacement section is not valid UTF-8.
    InvalidUtf8,
}

impl std::fmt::Display for CharsmapError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "precompiled_charsmap blob truncated"),
            Self::InvalidUtf8 => write!(f, "precompiled_charsmap replacements not UTF-8"),
        }
    }
}

impl std::error::Error for CharsmapError {}

// Double-array unit accessors (darts-clone layout, identical to spm_precompiled).
#[inline]
fn has_leaf(unit: usize) -> bool {
    (unit >> 8) & 1 == 1
}

#[inline]
fn value(unit: usize) -> usize {
    unit & ((1usize << 31) - 1)
}

#[inline]
fn label(unit: usize) -> usize {
    unit & ((1usize << 31) | 0xFF)
}

#[inline]
fn offset(unit: usize) -> usize {
    (unit >> 10) << ((unit & (1usize << 9)) >> 6)
}

/// A parsed SentencePiece `precompiled_charsmap`.
///
/// Keeps the raw blob for `.tkz` serialization and precomputes single-ASCII-char
/// replacements so ASCII-heavy text skips grapheme segmentation and trie walks.
pub struct PrecompiledCharsmap {
    blob: Vec<u8>,
    trie: Vec<u32>,
    replacements: String,
    /// Replacement for each single ASCII char; `None` = kept verbatim.
    ascii: [Option<Box<str>>; 128],
    /// Replacement for the "\r\n" grapheme cluster; `None` = kept verbatim.
    crlf: Option<Box<str>>,
}

impl PrecompiledCharsmap {
    /// Parse a raw `precompiled_charsmap` blob.
    pub fn from_blob(blob: &[u8]) -> Result<Self, CharsmapError> {
        if blob.len() < 4 {
            return Err(CharsmapError::Truncated);
        }
        let trie_bytes = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        let trie_end = 4usize.checked_add(trie_bytes).ok_or(CharsmapError::Truncated)?;
        if trie_end > blob.len() || trie_bytes % 4 != 0 {
            return Err(CharsmapError::Truncated);
        }
        let trie: Vec<u32> = blob[4..trie_end]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let replacements = std::str::from_utf8(&blob[trie_end..])
            .map_err(|_| CharsmapError::InvalidUtf8)?
            .to_string();

        let mut this = Self {
            blob: blob.to_vec(),
            trie,
            replacements,
            ascii: std::array::from_fn(|_| None),
            crlf: None,
        };
        // A replacement equal to its input is stored as `None` (kept
        // verbatim): the same output, and the char stays on the bulk-copy
        // path of `normalize_with`.
        let mut buf = [0u8; 1];
        this.ascii = std::array::from_fn(|b| {
            let c: &str = (b as u8 as char).encode_utf8(&mut buf);
            this.transform(c).filter(|r| *r != c).map(Into::into)
        });
        this.crlf = this.transform("\r\n").filter(|r| *r != "\r\n").map(Into::into);
        Ok(this)
    }

    /// The raw blob this charsmap was parsed from.
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// Shortest-prefix trie match for `key`, returning the replacement offset.
    ///
    /// spm_precompiled collects all prefix matches and uses the first (shortest);
    /// walking the same order lets us return at the first leaf.
    #[inline]
    fn first_prefix_match(&self, key: &[u8]) -> Option<usize> {
        let mut node_pos = 0usize;
        let mut unit = *self.trie.first()? as usize;
        node_pos ^= offset(unit);
        for &c in key {
            if c == 0 {
                break;
            }
            node_pos ^= c as usize;
            unit = *self.trie.get(node_pos)? as usize;
            if label(unit) != c as usize {
                return None;
            }
            node_pos ^= offset(unit);
            if has_leaf(unit) {
                return Some(value(*self.trie.get(node_pos)? as usize));
            }
        }
        None
    }

    /// Look up the replacement for `chunk`. `None` means keep it verbatim.
    #[inline]
    fn transform(&self, chunk: &str) -> Option<&str> {
        let start = self.first_prefix_match(chunk.as_bytes())?;
        let bytes = self.replacements.as_bytes();
        let end = memchr::memchr(0, bytes.get(start..)?).map_or(bytes.len(), |p| start + p);
        self.replacements.get(start..end)
    }

    /// Apply the charsmap to `text`, appending the result to `out`.
    ///
    /// Matches HF `tokenizers` semantics exactly: per grapheme cluster, try the
    /// whole cluster if under 6 bytes, otherwise transform each char separately.
    pub fn normalize_into(&self, text: &str, out: &mut String) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        while i < len {
            let b = bytes[i];
            // Fast path: an ASCII char whose neighbors are ASCII is always its
            // own grapheme cluster, except CR+LF which clusters as a pair.
            // (ASCII never combines with ASCII otherwise, and the loop only
            // reaches here from a verified boundary, so the previous char
            // cannot extend into this one.)
            if b < 0x80 && (i + 1 >= len || bytes[i + 1] < 0x80) {
                if b == b'\r' && i + 1 < len && bytes[i + 1] == b'\n' {
                    match &self.crlf {
                        Some(r) => out.push_str(r),
                        None => out.push_str("\r\n"),
                    }
                    i += 2;
                    continue;
                }
                match &self.ascii[b as usize] {
                    Some(r) => out.push_str(r),
                    None => out.push(b as char),
                }
                i += 1;
            } else {
                let j = region_end(bytes, i);
                self.normalize_graphemes(&text[i..j], out);
                i = j;
            }
        }
    }

    /// [`Self::normalize_into`] streamed into `sink` instead of a string, so
    /// a following whitespace/metaspace step runs in the same pass. Runs of
    /// ASCII bytes the charsmap keeps (and `sink` accepts in bulk) go to
    /// [`CharsmapSink::run`] whole; every other output char goes to
    /// [`CharsmapSink::char`], in order. `scratch` holds the output of one
    /// non-ASCII region at a time.
    pub(crate) fn normalize_with(&self, text: &str, sink: &mut impl CharsmapSink, scratch: &mut String) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        // Bytes that go out in runs: ASCII the charsmap keeps, outside a
        // CR LF pair, that `sink` accepts in bulk.
        let mut bulk = [false; 256];
        for b in 0..128u8 {
            bulk[b as usize] = b != b'\r' && self.ascii[b as usize].is_none() && sink.bulk(b);
        }
        let mut i = 0;
        while i < len {
            let b = bytes[i];
            // The fast path of `normalize_into`: an ASCII char followed by
            // ASCII (or the end) is its own grapheme cluster.
            if b < 0x80 && (i + 1 >= len || bytes[i + 1] < 0x80) {
                if bulk[b as usize] {
                    let mut k = i + 1;
                    while k < len && bulk[bytes[k] as usize] {
                        k += 1;
                    }
                    // An ASCII char before a non-ASCII one belongs to the
                    // grapheme region that starts with it. Then `k > i + 1`
                    // (`bytes[i + 1]` is ASCII), so the run is never empty.
                    let end = if k < len && bytes[k] >= 0x80 { k - 1 } else { k };
                    sink.run(&text[i..end]);
                    i = end;
                    continue;
                }
                if b == b'\r' && i + 1 < len && bytes[i + 1] == b'\n' {
                    match &self.crlf {
                        Some(r) => r.chars().for_each(|c| sink.char(c)),
                        None => {
                            sink.char('\r');
                            sink.char('\n');
                        }
                    }
                    i += 2;
                    continue;
                }
                match &self.ascii[b as usize] {
                    Some(r) => r.chars().for_each(|c| sink.char(c)),
                    None => sink.char(b as char),
                }
                i += 1;
            } else {
                let j = region_end(bytes, i);
                scratch.clear();
                self.normalize_graphemes(&text[i..j], scratch);
                scratch.chars().for_each(|c| sink.char(c));
                i = j;
            }
        }
    }

    /// Grapheme-by-grapheme transform, mirroring HF's Precompiled normalizer.
    fn normalize_graphemes(&self, region: &str, out: &mut String) {
        for grapheme in region.graphemes(true) {
            if grapheme.len() < 6 {
                if let Some(norm) = self.transform(grapheme) {
                    out.push_str(norm);
                    continue;
                }
            }
            for (ci, c) in grapheme.char_indices() {
                let part = &grapheme[ci..ci + c.len_utf8()];
                match self.transform(part) {
                    Some(norm) => out.push_str(norm),
                    None => out.push(c),
                }
            }
        }
    }
}

/// End of the slow region starting at `i`: extend until a safe boundary (an
/// ASCII char whose predecessor is also ASCII, not splitting a CR LF pair),
/// which full grapheme segmentation then covers.
fn region_end(bytes: &[u8], i: usize) -> usize {
    let mut j = i + 1;
    while j < bytes.len() {
        if bytes[j] < 0x80 && bytes[j - 1] < 0x80 && !(bytes[j - 1] == b'\r' && bytes[j] == b'\n') {
            break;
        }
        j += 1;
    }
    j
}

/// Receives a charsmap transform's output as it is produced (see
/// [`PrecompiledCharsmap::normalize_with`]).
pub(crate) trait CharsmapSink {
    /// Whether a kept ASCII byte may arrive inside a [`Self::run`]; bytes it
    /// refuses come one at a time through [`Self::char`].
    fn bulk(&self, b: u8) -> bool;
    /// A run of kept ASCII bytes, each accepted by [`Self::bulk`].
    fn run(&mut self, run: &str);
    /// The next output char.
    fn char(&mut self, c: char);
}

impl PartialEq for PrecompiledCharsmap {
    fn eq(&self, other: &Self) -> bool {
        self.blob == other.blob
    }
}

impl Eq for PrecompiledCharsmap {}

impl std::fmt::Debug for PrecompiledCharsmap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrecompiledCharsmap")
            .field("blob_len", &self.blob.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal charsmap blob from (source, replacement) pairs using a
    /// naive double-array builder sufficient for tests.
    ///
    /// We cheat: rather than building a real darts trie, tests below only use
    /// blobs captured from real models (see accuracy suite). Here we just check
    /// the parser rejects malformed input.
    #[test]
    fn rejects_truncated_blob() {
        assert_eq!(PrecompiledCharsmap::from_blob(&[]), Err(CharsmapError::Truncated));
        assert_eq!(
            PrecompiledCharsmap::from_blob(&[16, 0, 0, 0, 1, 2]),
            Err(CharsmapError::Truncated)
        );
    }

    #[test]
    fn empty_trie_is_identity() {
        // trie_size = 0, empty replacements: every lookup misses.
        let cm = PrecompiledCharsmap::from_blob(&[0, 0, 0, 0]).unwrap();
        let mut out = String::new();
        cm.normalize_into("Hello, wörld!\r\nnext", &mut out);
        assert_eq!(out, "Hello, wörld!\r\nnext");
    }
}
