//! SentencePiece `precompiled_charsmap`.
//!
//! Blob layout: `[trie_byte_len: u32 le][trie: [u32]][replacements: str]`.
//! A grapheme shorter than 6 bytes is looked up whole. A longer grapheme is
//! looked up one character at a time. `None` means the chunk is kept.

use unicode_segmentation::UnicodeSegmentation;

/// A charsmap blob that could not be parsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CharsmapError {
    /// The trie length runs past the blob, or the blob is shorter than 4 bytes.
    Truncated,
    /// The replacement table is not UTF-8.
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

/// Parsed trie plus the ASCII and `\r\n` answers cached off it.
pub(crate) struct Charsmap {
    trie: Vec<u32>,
    replacements: String,
    /// `Some` when this ASCII byte is rewritten. `None` keeps the byte.
    ascii: [Option<Box<str>>; 128],
    /// Replacement span in `replacements` for each BMP scalar.
    /// `(u32::MAX, 0)` keeps the scalar.
    bmp: Vec<(u32, u32)>,
    /// `Some` when the `\r\n` grapheme is rewritten.
    crlf: Option<Box<str>>,
    /// Printable ASCII (`0x20..0x7F`) is kept, so a scan can copy those runs.
    pub(crate) fast_scan: bool,
}

impl Charsmap {
    pub(crate) fn parse(blob: &[u8]) -> Result<Self, CharsmapError> {
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

        let mut map = Self {
            trie,
            replacements,
            ascii: std::array::from_fn(|_| None),
            bmp: vec![(u32::MAX, 0); 0x10000],
            crlf: None,
            fast_scan: true,
        };
        let mut fast_scan = true;
        let mut buf = [0u8; 4];
        for cp in 0..0x10000u32 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let s = c.encode_utf8(&mut buf);
            let recorded = {
                let Some(norm) = map.transform(s) else {
                    continue;
                };
                if norm == s {
                    continue;
                }
                let start = norm.as_ptr() as usize - map.replacements.as_ptr() as usize;
                (start, norm.len())
            };
            if cp < 128 && (0x20..0x7F).contains(&(cp as u8)) {
                fast_scan = false;
            }
            map.bmp[cp as usize] = (recorded.0 as u32, (recorded.0 + recorded.1) as u32);
            if cp < 128 {
                let norm = &map.replacements[recorded.0..recorded.0 + recorded.1];
                map.ascii[cp as usize] = Some(norm.to_string().into());
            }
        }
        if let Some(norm) = map.transform("\r\n")
            && norm != "\r\n"
        {
            map.crlf = Some(norm.into());
        }
        map.fast_scan = fast_scan;
        Ok(map)
    }

    pub(crate) fn parts(&self) -> (&[u32], &str, &[(u32, u32)]) {
        (&self.trie, &self.replacements, &self.bmp)
    }

    /// Assemble a map from a stored trie and BMP table.
    ///
    /// The ASCII cache and the `\r\n` answer come from the trie. The BMP rows
    /// are stored as given.
    pub(crate) fn from_parts(
        trie: Vec<u32>,
        replacements: String,
        bmp: Vec<(u32, u32)>,
    ) -> Result<Self, CharsmapError> {
        if bmp.len() != 0x10000 {
            return Err(CharsmapError::Truncated);
        }
        let mut map = Self {
            trie,
            replacements,
            ascii: std::array::from_fn(|_| None),
            bmp,
            crlf: None,
            fast_scan: true,
        };
        let mut fast_scan = true;
        for cp in 0..128u32 {
            let (start, end) = map.bmp[cp as usize];
            if start == u32::MAX {
                continue;
            }
            let start = start as usize;
            let end = end as usize;
            if start > end || end > map.replacements.len() {
                return Err(CharsmapError::Truncated);
            }
            let norm = map
                .replacements
                .get(start..end)
                .ok_or(CharsmapError::InvalidUtf8)?;
            if (0x20..0x7F).contains(&(cp as u8)) {
                fast_scan = false;
            }
            map.ascii[cp as usize] = Some(norm.to_string().into());
        }
        for (start, end) in &map.bmp[128..] {
            if *start == u32::MAX {
                continue;
            }
            if *start > *end || *end as usize > map.replacements.len() {
                return Err(CharsmapError::Truncated);
            }
        }
        if let Some(norm) = map.transform("\r\n")
            && norm != "\r\n"
        {
            map.crlf = Some(norm.to_string().into());
        }
        map.fast_scan = fast_scan;
        Ok(map)
    }

    pub(crate) fn split_blob(blob: &[u8]) -> Result<(Vec<u32>, String), CharsmapError> {
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
        Ok((trie, replacements))
    }

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

    /// Replacement for one scalar, or `None` to keep it.
    ///
    /// BMP scalars hit a table filled at parse time.
    #[inline]
    pub(crate) fn rewrite_char(&self, c: char) -> Option<&str> {
        let cp = c as u32;
        if cp >= 0x10000 {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            return self.transform(s).filter(|norm| *norm != s);
        }
        let (start, end) = self.bmp[cp as usize];
        if start == u32::MAX {
            None
        } else {
            Some(&self.replacements[start as usize..end as usize])
        }
    }

    /// Replacement for `chunk`, or `None` to keep it.
    #[inline]
    pub(crate) fn transform<'a>(&'a self, chunk: &str) -> Option<&'a str> {
        let start = self.first_prefix_match(chunk.as_bytes())?;
        let bytes = self.replacements.as_bytes();
        let end = memchr::memchr(0, bytes.get(start..)?).map_or(bytes.len(), |p| start + p);
        self.replacements.get(start..end)
    }

    /// Cached ASCII rewrite, if this byte is not kept.
    #[inline]
    pub(crate) fn ascii(&self, b: u8) -> Option<&str> {
        self.ascii.get(b as usize)?.as_deref()
    }

    #[inline]
    pub(crate) fn crlf(&self) -> Option<&str> {
        self.crlf.as_deref()
    }

    /// Grapheme walk used for a non-ASCII span and for maps that rewrite ASCII.
    pub(crate) fn normalize_graphemes(&self, region: &str, out: &mut String) {
        for grapheme in region.graphemes(true) {
            if grapheme.len() < 6
                && let Some(norm) = self.transform(grapheme)
            {
                out.push_str(norm);
                continue;
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
