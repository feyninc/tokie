//! Text normalization for tokenizers.
//!
//! Normalizers transform text before tokenization. Common operations include:
//! - Lowercasing (BERT uncased models)
//! - Unicode normalization (NFC/NFD)
//! - Cleaning control characters
//!
//! # Example
//!
//! ```
//! use tokie::Normalizer;
//!
//! let normalizer = Normalizer::BertUncased;
//! let text = "Hello World!";
//! let normalized = normalizer.normalize(text);
//! assert_eq!(normalized, "hello world!");
//! ```

use std::borrow::Cow;
use std::sync::Arc;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::charsmap::PrecompiledCharsmap;

/// Lookup table for ASCII bytes that need cleaning in clean_text.
/// true = problematic byte (control char, DEL, or high bit set)
/// false = safe ASCII byte (printable or tab/newline/CR)
static NEEDS_CLEANING: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = 0u16;
    while i < 256 {
        let b = i as u8;
        // High bit set (non-ASCII) - will trigger Unicode path
        // Control chars (< 0x20) except tab (0x09), newline (0x0A), CR (0x0D)
        // DEL (0x7F)
        table[i as usize] = b >= 0x80
            || (b < 0x20 && b != 0x09 && b != 0x0A && b != 0x0D)
            || b == 0x7F;
        i += 1;
    }
    table
};

/// Text normalizer configuration.
///
/// Normalizers transform input text before pre-tokenization and encoding.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Normalizer {
    /// No normalization (GPT-2, RoBERTa, Llama, etc.)
    #[default]
    None,

    /// BERT uncased: clean_text + strip_accents + lowercase.
    ///
    /// Used by: bert-base-uncased, GTE, BGE, E5, MiniLM, etc.
    ///
    /// This implements:
    /// - `clean_text`: removes control characters (BOM, null, etc.) and normalizes whitespace
    /// - `strip_accents`: removes diacritical marks (é → e, ñ → n)
    /// - `lowercase`: converts all text to lowercase
    BertUncased,

    /// BERT cased: clean_text only (no lowercasing).
    ///
    /// Used by: bert-base-cased, bert-multilingual-cased, etc.
    ///
    /// This implements:
    /// - `clean_text`: removes control characters (BOM, null, etc.) and normalizes whitespace
    /// - NO lowercasing (preserves case)
    BertCased,

    /// Unicode NFC normalization.
    ///
    /// Used by: ModernBERT, Qwen, etc.
    Nfc,

    /// SentencePiece Metaspace normalization.
    ///
    /// Used by: Llama, Mistral, Gemma, etc.
    ///
    /// This implements:
    /// - Prepends `▁` (U+2581) at the start of the text
    /// - Replaces all spaces with `▁`
    ///
    /// Example: "Hello world" → "▁Hello▁world"
    Metaspace,

    /// SentencePiece normalization with NFKC.
    ///
    /// Used by: T5, XLM-RoBERTa, mT5, etc.
    ///
    /// This implements:
    /// - NFKC Unicode normalization
    /// - Collapse multiple whitespace to single space
    /// - Strip leading/trailing whitespace
    /// - Prepend `▁` and replace spaces with `▁`
    ///
    /// Example: "  Hello   world  " → "▁Hello▁world"
    SentencePiece,

    /// SentencePiece normalization with NFKC + lowercase.
    ///
    /// Used by: ALBERT, etc.
    ///
    /// Same as SentencePiece but also lowercases.
    ///
    /// Example: "  Hello   World  " → "▁hello▁world"
    SentencePieceLowercase,

    /// Replace space with metaspace only (no prepend).
    ///
    /// Used by: Gemma (Replace normalizer)
    ///
    /// This only replaces spaces with `▁`, without prepending at start.
    ///
    /// Example: "Hello world" → "Hello▁world"
    MetaspaceReplace,

    /// SentencePiece normalization driven by the model's exact
    /// `precompiled_charsmap` blob (XLM-RoBERTa, T5, bge-m3, ...).
    ///
    /// This implements the model's own charsmap transform (NFKC-style
    /// rewrites, exact per-character keep/map/drop decisions) followed by the
    /// metaspace step matching the model's pre-tokenizer chain:
    ///
    /// - `whitespace_split: true` (XLM-R, T5 — WhitespaceSplit + Metaspace):
    ///   collapse whitespace, strip leading/trailing, prepend `▁`, spaces → `▁`
    /// - `whitespace_split: false` (bge-m3 family — Replace `" {2,}"` +
    ///   Metaspace only): collapse space runs, no strip (a trailing space
    ///   becomes a real `▁` token), prepend `▁`, spaces → `▁`
    ///
    /// Unlike [`Normalizer::SentencePiece`], which approximates the charsmap
    /// with Unicode-category rules, this reproduces HF byte-for-byte (the real
    /// xlm-roberta charsmap keeps U+00AD and C1 controls that category rules
    /// would strip).
    SentencePiecePrecompiled {
        charsmap: Arc<PrecompiledCharsmap>,
        whitespace_split: bool,
    },
}

impl Normalizer {
    /// Normalize text according to this normalizer's rules.
    ///
    /// Returns `Cow::Borrowed` when no changes are needed (zero allocation).
    /// Returns `Cow::Owned` when text was modified.
    #[inline]
    pub fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        match self {
            Normalizer::None => Cow::Borrowed(text),
            Normalizer::BertCased => clean_text(text),
            Normalizer::BertUncased => bert_uncased_normalize(text),
            Normalizer::Nfc => normalize_nfc(text),
            Normalizer::Metaspace => metaspace_normalize(text),
            Normalizer::SentencePiece => sentencepiece_normalize(text),
            Normalizer::SentencePieceLowercase => sentencepiece_lowercase_normalize(text),
            Normalizer::MetaspaceReplace => metaspace_replace_normalize(text),
            Normalizer::SentencePiecePrecompiled { charsmap, whitespace_split } => {
                sentencepiece_precompiled_normalize(charsmap, *whitespace_split, text)
            }
        }
    }

    /// Check if this normalizer modifies text.
    #[inline]
    pub fn is_identity(&self) -> bool {
        matches!(self, Normalizer::None)
    }
}

/// NFC Unicode normalization with early exit.
///
/// Returns borrowed text if already NFC normalized, avoiding allocation.
/// Uses ICU4X for fast normalization (~22x faster than unicode-normalization).
#[inline]
fn normalize_nfc<'a>(text: &'a str) -> Cow<'a, str> {
    let nfc = icu_normalizer::ComposingNormalizer::new_nfc();

    if nfc.is_normalized(text) {
        return Cow::Borrowed(text);
    }
    Cow::Owned(nfc.normalize(text))
}

/// Metaspace normalization for SentencePiece tokenizers.
///
/// Transforms text for SentencePiece-style tokenization:
/// - Prepends `▁` (U+2581) at the start IF text doesn't start with whitespace
/// - Replaces all spaces with `▁`
///
/// This matches HuggingFace's Metaspace with `prepend_scheme: "first"`:
/// - "Hello world" → "▁Hello▁world"
/// - "  spaces" → "▁▁spaces" (no extra prepend, spaces become ▁)
///
/// # Performance
///
/// Uses SIMD-accelerated `fnr` for fast space replacement.
#[inline]
pub fn metaspace_normalize(text: &str) -> Cow<'_, str> {
    // Check if text starts with whitespace
    let starts_with_space = text.starts_with(' ') || text.starts_with('\t');

    // Use fnr for efficient space -> ▁ replacement
    let replaced = fnr(text, " ", "▁");

    if starts_with_space {
        // Don't prepend when text starts with space (space→▁ handles it)
        match replaced {
            Cow::Borrowed(_) => {
                // No spaces were replaced, but we checked starts_with_space
                // This shouldn't happen if starts_with_space is true
                replaced
            }
            Cow::Owned(s) => Cow::Owned(s),
        }
    } else {
        // Prepend ▁ to the result
        let mut result = String::with_capacity(replaced.len() + 3);
        result.push('▁');
        result.push_str(&replaced);
        Cow::Owned(result)
    }
}

/// Metaspace replace normalization (no prepend).
///
/// Unlike `metaspace_normalize`, this only replaces spaces with `▁`
/// without prepending at the start.
///
/// Used by: Gemma (Replace normalizer in HF tokenizer.json)
///
/// Example: "Hello world" → "Hello▁world"
#[inline]
pub fn metaspace_replace_normalize(text: &str) -> Cow<'_, str> {
    // Simply replace spaces with metaspace character
    fnr(text, " ", "▁")
}

/// SentencePiece normalization with NFKC.
///
/// Used by T5, XLM-RoBERTa, mT5, etc.
///
/// Applies the following transformations:
/// 1. NFKC Unicode normalization
/// 2. Strip control characters (BOM, null, format chars)
/// 3. Collapse all whitespace (space, tab, newline, etc.) to single space
/// 4. Strip leading and trailing whitespace
/// 5. Prepend `▁` and replace internal spaces with `▁`
///
/// Example: "  Hello   world\n" → "▁Hello▁world"
#[inline]
pub fn sentencepiece_normalize(text: &str) -> Cow<'_, str> {
    if text.is_empty() {
        return Cow::Borrowed(text);
    }

    // Step 1: NFKC normalization
    let nfkc = icu_normalizer::ComposingNormalizer::new_nfkc();
    let normalized = if nfkc.is_normalized(text) {
        Cow::Borrowed(text)
    } else {
        Cow::Owned(nfkc.normalize(text))
    };

    // Step 2: Strip control characters (BOM, null, format chars, etc.)
    // Step 3 & 4: Collapse whitespace and strip
    let collapsed = collapse_strip_whitespace_and_controls(&normalized);

    // Step 5: Apply metaspace (prepend ▁ and replace spaces)
    let mut result = String::with_capacity(collapsed.len() + 3);
    result.push('▁');
    for c in collapsed.chars() {
        if c == ' ' {
            result.push('▁');
        } else {
            result.push(c);
        }
    }

    Cow::Owned(result)
}

/// SentencePiece normalization using the model's exact `precompiled_charsmap`.
///
/// Applies the charsmap transform (grapheme-wise, HF-identical — see
/// [`PrecompiledCharsmap`]) followed by the metaspace step matching the
/// model's pre-tokenizer chain:
///
/// - `whitespace_split: true` — WhitespaceSplit + Metaspace (XLM-R, T5):
///   whitespace runs of any kind separate words and are dropped, edges
///   stripped, every word prefixed with `▁`.
/// - `whitespace_split: false` — Replace `" {2,}"` → `" "` + Metaspace
///   (bge-m3, snowflake-arctic-v2): only space runs collapse, nothing is
///   stripped (e.g. a trailing space survives as a lone `▁`), and `▁` is
///   prepended unless the text already starts with a space or `▁`.
pub fn sentencepiece_precompiled_normalize<'a>(
    charsmap: &PrecompiledCharsmap,
    whitespace_split: bool,
    text: &'a str,
) -> Cow<'a, str> {
    if text.is_empty() {
        return Cow::Borrowed(text);
    }

    // Step 1: exact charsmap transform
    let mut transformed = String::with_capacity(text.len());
    charsmap.normalize_into(text, &mut transformed);

    if whitespace_split {
        // Step 2: collapse whitespace and strip (WhitespaceSplit)
        let collapsed = collapse_and_strip_whitespace(&transformed);

        // Step 3: apply metaspace (prepend ▁ and replace spaces)
        let mut result = String::with_capacity(collapsed.len() + 3);
        result.push('▁');
        for c in collapsed.chars() {
            if c == ' ' {
                result.push('▁');
            } else {
                result.push(c);
            }
        }
        Cow::Owned(result)
    } else {
        // Step 2-3 fused: collapse space runs (Replace " {2,}" → " ") and
        // replace with ▁; prepend ▁ unless the text starts with space/▁
        // (HF Metaspace checks starts_with(replacement) after replacing).
        let mut result = String::with_capacity(transformed.len() + 3);
        if !(transformed.starts_with(' ') || transformed.starts_with('▁')) {
            result.push('▁');
        }
        let mut prev_space = false;
        for c in transformed.chars() {
            if c == ' ' {
                if !prev_space {
                    result.push('▁');
                }
                prev_space = true;
            } else {
                result.push(c);
                prev_space = false;
            }
        }
        Cow::Owned(result)
    }
}

/// SentencePiece normalization with NFKD + StripAccents + lowercase.
///
/// Used by ALBERT, etc.
///
/// Applies the following transformations:
/// 1. NFKD Unicode normalization (decompose characters)
/// 2. Strip accents (remove combining marks) and control characters
/// 3. Lowercase
/// 4. Collapse whitespace and strip
/// 5. Prepend ▁ and replace spaces with ▁
///
/// Example: "  Héllo   Wörld\n" → "▁hello▁world"
#[inline]
pub fn sentencepiece_lowercase_normalize(text: &str) -> Cow<'_, str> {
    if text.is_empty() {
        return Cow::Borrowed(text);
    }

    // Step 1: NFKD normalization (decompose to separate base chars from combining marks)
    let nfkd = icu_normalizer::DecomposingNormalizer::new_nfkd();
    let normalized = nfkd.normalize(text);

    // Step 2: Strip accents (combining marks) and control characters (BOM, null, etc.)
    let stripped: String = normalized
        .chars()
        .filter(|&c| !is_combining_mark(c) && !is_control(c) && c != '\0' && c != '\u{FFFD}')
        .collect();

    // Step 3 & 4: Collapse whitespace and strip
    let collapsed = collapse_and_strip_whitespace(&stripped);

    // Step 5: Lowercase + Apply metaspace (prepend ▁ and replace spaces)
    let mut result = String::with_capacity(collapsed.len() + 3);
    result.push('▁');
    for c in collapsed.chars() {
        if c == ' ' {
            result.push('▁');
        } else {
            // Lowercase each character
            for lc in c.to_lowercase() {
                result.push(lc);
            }
        }
    }

    Cow::Owned(result)
}

/// Collapse multiple whitespace chars to single space and strip leading/trailing.
///
/// Handles: space, tab, newline, carriage return, and Unicode whitespace.
fn collapse_and_strip_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = true; // Start true to strip leading whitespace

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_was_space {
                result.push(' ');
                prev_was_space = true;
            }
            // Skip if prev was already space (collapse)
        } else {
            result.push(c);
            prev_was_space = false;
        }
    }

    // Strip trailing space
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

/// Collapse whitespace, strip leading/trailing, and handle control/format characters.
///
/// Matches SentencePiece's Precompiled charsmap behavior:
/// - Control chars (Cc category, except tab/newline/CR): **removed**
/// - Format chars (Cf category: ZWNJ, ZWJ, directional marks, BOM): **mapped to space**
/// - Replacement char (U+FFFD): **mapped to space**
/// - Whitespace: collapsed to single space, stripped from edges
///
/// The key difference from stripping format chars: ZWNJ (U+200C) between
/// non-space chars becomes a space, creating a word boundary that affects
/// tokenization (e.g., Persian "دولت‌زدائی" → "دولت زدائی").
fn collapse_strip_whitespace_and_controls(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_was_space = true; // Start true to strip leading whitespace

    for c in text.chars() {
        if c == '\0' {
            continue;
        }

        // Classify the character
        let cat = get_general_category(c);
        match cat {
            // Control chars (Cc): tab/newline/CR → space, others → removed
            GeneralCategory::Control => {
                if c == '\t' || c == '\n' || c == '\r' || c == '\x0C' {
                    // Map to space (whitespace collapsing)
                    if !prev_was_space {
                        result.push(' ');
                        prev_was_space = true;
                    }
                }
                // Other control chars: silently removed
            }
            // Format chars (Cf): ZWNJ, ZWJ, directional marks, BOM → space
            GeneralCategory::Format => {
                if !prev_was_space {
                    result.push(' ');
                    prev_was_space = true;
                }
            }
            _ => {
                // Replacement char → space (matching charsmap)
                if c == '\u{FFFD}' {
                    if !prev_was_space {
                        result.push(' ');
                        prev_was_space = true;
                    }
                } else if c.is_whitespace() {
                    if !prev_was_space {
                        result.push(' ');
                        prev_was_space = true;
                    }
                } else {
                    result.push(c);
                    prev_was_space = false;
                }
            }
        }
    }

    // Strip trailing space
    if result.ends_with(' ') {
        result.pop();
    }

    result
}

/// Check if a character is a Unicode control character.
///
/// Returns true for Unicode categories Cc, Cf, Cn, Co (control, format, unassigned, private-use),
/// EXCEPT for tab, newline, and carriage return which are treated as whitespace.
#[inline]
fn is_control(c: char) -> bool {
    match c {
        '\t' | '\n' | '\r' => false,
        _ => matches!(
            get_general_category(c),
            GeneralCategory::Control
                | GeneralCategory::Format
                | GeneralCategory::Unassigned
                | GeneralCategory::PrivateUse
        ),
    }
}

/// Check if a character is whitespace (for BERT normalization).
///
/// Includes standard ASCII whitespace plus Unicode space characters.
#[inline]
fn is_bert_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
        || get_general_category(c) == GeneralCategory::SpaceSeparator
}

/// Clean text by removing control characters and normalizing whitespace.
///
/// This matches HuggingFace's BertNormalizer `clean_text` behavior:
/// - Removes null (U+0000), replacement char (U+FFFD), and control characters
/// - Normalizes all whitespace variants to standard space ' '
///
/// Returns `Cow::Borrowed` when no changes needed (zero allocation).
///
/// Performance: Uses byte-level fast path for ASCII text (~2.7 GB/s vs ~475 MB/s).
///
/// # Example
///
/// ```
/// use tokie::normalizer::clean_text;
///
/// // BOM and control chars are removed
/// let text = "\u{FEFF}Hello\u{0000}World";
/// let cleaned = clean_text(text);
/// assert_eq!(cleaned, "HelloWorld");
///
/// // Various whitespace becomes regular space
/// let text = "hello\u{00A0}world"; // non-breaking space
/// let cleaned = clean_text(text);
/// assert_eq!(cleaned, "hello world");
/// ```
pub fn clean_text<'a>(text: &'a str) -> Cow<'a, str> {
    let bytes = text.as_bytes();

    // Fast lookup-table scan (2x faster than complex condition)
    // Find first byte that needs attention
    let first_problem = bytes.iter().position(|&b| NEEDS_CLEANING[b as usize]);

    let first_pos = match first_problem {
        None => return Cow::Borrowed(text), // No cleaning needed
        Some(pos) => pos,
    };

    // Check if we hit a non-ASCII byte (high bit set)
    if bytes[first_pos] >= 0x80 {
        return clean_text_unicode(text, first_pos);
    }

    // Pure ASCII with control chars - check rest for non-ASCII
    if bytes[first_pos..].iter().any(|&b| b >= 0x80) {
        return clean_text_unicode(text, first_pos);
    }

    // Pure ASCII path: process byte-by-byte without Unicode overhead
    let mut result = Vec::with_capacity(text.len());
    result.extend_from_slice(&bytes[..first_pos]);

    for &b in &bytes[first_pos..] {
        if b < 0x20 {
            if b == b'\t' || b == b'\n' || b == b'\r' {
                result.push(b' '); // Normalize whitespace to space
            }
            // else: skip control character
        } else if b == 0x7F {
            // Skip DEL
        } else {
            result.push(b);
        }
    }

    // Safe: we only added ASCII bytes
    Cow::Owned(unsafe { String::from_utf8_unchecked(result) })
}

/// Unicode path for clean_text - handles text with non-ASCII characters.
/// `first_problem` is the byte position where we found the first problematic byte.
fn clean_text_unicode<'a>(text: &'a str, first_problem: usize) -> Cow<'a, str> {
    // Scan the prefix (pure ASCII) to find if it needs cleaning
    let prefix_bytes = &text.as_bytes()[..first_problem];
    let prefix_needs_cleaning = prefix_bytes.iter().any(|&b| NEEDS_CLEANING[b as usize]);

    // Check if the Unicode portion needs cleaning
    let suffix = &text[first_problem..];
    let suffix_needs_cleaning = suffix.chars().any(|c| {
        c == '\0' || c == '\u{FFFD}' || is_control(c) || (is_bert_whitespace(c) && c != ' ')
    });

    if !prefix_needs_cleaning && !suffix_needs_cleaning {
        return Cow::Borrowed(text);
    }

    // Build result: clean prefix + clean suffix
    let mut result = String::with_capacity(text.len());

    // Add cleaned prefix
    for &b in prefix_bytes {
        if b < 0x20 {
            if b == b'\t' || b == b'\n' || b == b'\r' {
                result.push(' ');
            }
        } else if b != 0x7F {
            result.push(b as char);
        }
    }

    // Add cleaned suffix
    for c in suffix.chars() {
        if c == '\0' || c == '\u{FFFD}' || is_control(c) {
            continue;
        }
        if is_bert_whitespace(c) {
            result.push(' ');
        } else {
            result.push(c);
        }
    }

    Cow::Owned(result)
}

/// Strip accents (diacritical marks) from text.
///
/// This matches HuggingFace's BertNormalizer `strip_accents` behavior.
/// Uses NFD normalization to decompose characters, then filters out
/// combining marks (Unicode category Mn - NonspacingMark).
///
/// Performance: ~3.1 GB/s for pure ASCII text, ~469 MB/s for accented text (using ICU4X).
///
/// # Example
///
/// ```
/// use tokie::normalizer::strip_accents;
///
/// let text = "Pávlovna café résumé";
/// let stripped = strip_accents(text);
/// assert_eq!(stripped, "Pavlovna cafe resume");
/// ```
pub fn strip_accents<'a>(text: &'a str) -> Cow<'a, str> {
    // Fast path: if all ASCII, no accents possible
    if text.bytes().all(|b| b < 0x80) {
        return Cow::Borrowed(text);
    }

    // Use ICU4X for fast NFD normalization
    let nfd = icu_normalizer::DecomposingNormalizer::new_nfd();
    let normalized = nfd.normalize(text);

    // Filter out combining marks (accents) using fast inline check
    // Most combining marks are in range U+0300-U+036F (Combining Diacritical Marks)
    // and U+0080-U+00FF doesn't contain combining marks
    let result: String = normalized
        .chars()
        .filter(|&c| !is_combining_mark(c))
        .collect();

    Cow::Owned(result)
}

/// Fast check for combining marks (Unicode category Mn - NonspacingMark).
/// Uses inline range checks for common ranges before falling back to full lookup.
#[inline]
fn is_combining_mark(c: char) -> bool {
    let cp = c as u32;

    // Fast path: ASCII and Latin-1 Supplement never have combining marks
    if cp < 0x0300 {
        return false;
    }

    // Common combining mark ranges (covers ~95% of combining marks in real text)
    // U+0300-U+036F: Combining Diacritical Marks
    // U+0483-U+0489: Cyrillic combining marks
    // U+0591-U+05BD, U+05BF, U+05C1-U+05C2, U+05C4-U+05C5, U+05C7: Hebrew
    // U+0610-U+061A, U+064B-U+065F, U+0670: Arabic
    if cp <= 0x036F {
        return true; // U+0300-U+036F
    }

    // For other ranges, use full Unicode lookup
    // This is rare in typical text
    get_general_category(c) == GeneralCategory::NonspacingMark
}

/// Fused BERT uncased normalization: clean_text + strip_accents + lowercase in one pass.
///
/// This is significantly faster than applying each transformation separately because:
/// 1. Single pass over the text instead of 3 passes
/// 2. Single allocation instead of up to 3 allocations
/// 3. Better cache locality
/// 4. Uses ICU4X for fast NFD normalization (~12x faster than unicode-normalization)
///
/// For pure ASCII lowercase text, returns `Cow::Borrowed` (zero allocation).
/// Lookup table for bytes that need processing in bert_uncased_normalize.
/// true = needs work (non-ASCII, uppercase, control char)
static NEEDS_BERT_NORMALIZE: [bool; 256] = {
    let mut table = [false; 256];
    let mut i = 0u16;
    while i < 256 {
        let b = i as u8;
        // Non-ASCII (high bit set)
        // Uppercase A-Z
        // Control chars (< 0x20) except tab (0x09), newline (0x0A), CR (0x0D)
        // Null byte
        table[i as usize] = b >= 0x80
            || (b >= b'A' && b <= b'Z')
            || (b < 0x20 && b != 0x09 && b != 0x0A && b != 0x0D)
            || b == 0;
        i += 1;
    }
    table
};

pub fn bert_uncased_normalize(text: &str) -> Cow<'_, str> {
    let bytes = text.as_bytes();

    // Fast lookup-table scan to find first byte needing work
    let first_problem = bytes.iter().position(|&b| NEEDS_BERT_NORMALIZE[b as usize]);

    // Fast path: no work needed
    let first_pos = match first_problem {
        None => return Cow::Borrowed(text),
        Some(pos) => pos,
    };

    // Check if we have non-ASCII (need Unicode processing)
    let has_non_ascii = bytes[first_pos] >= 0x80
        || bytes[first_pos..].iter().any(|&b| b >= 0x80);

    // Pre-allocate result
    let mut result = String::with_capacity(text.len());

    if has_non_ascii {
        // Use ICU4X for fast NFD normalization
        let nfd = icu_normalizer::DecomposingNormalizer::new_nfd();
        let normalized = nfd.normalize(text);

        // Process NFD-normalized text: filter accents, clean, lowercase
        for c in normalized.chars() {
            // Skip combining marks (accents) - use fast inline check
            if is_combining_mark(c) {
                continue;
            }
            // Skip control characters
            if is_control(c) || c == '\0' || c == '\u{FFFD}' {
                continue;
            }
            // Normalize whitespace to space
            if is_bert_whitespace(c) {
                result.push(' ');
            } else if c.is_ascii() {
                // Inline ASCII lowercase (faster than c.to_lowercase())
                if c >= 'A' && c <= 'Z' {
                    result.push((c as u8 | 0x20) as char);
                } else {
                    result.push(c);
                }
            } else {
                // Unicode lowercase
                for lc in c.to_lowercase() {
                    result.push(lc);
                }
            }
        }
    } else {
        // ASCII-only fast path: copy prefix unchanged, then process
        // SAFETY: first_pos is at an ASCII byte boundary
        result.push_str(&text[..first_pos]);

        for &b in &bytes[first_pos..] {
            // Skip control characters (except tab/newline/cr)
            if b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r' {
                continue;
            }
            if b == 0 {
                continue;
            }
            // Normalize whitespace
            if b == b'\t' || b == b'\n' || b == b'\r' {
                result.push(' ');
            } else if b >= b'A' && b <= b'Z' {
                // Lowercase ASCII
                result.push((b + 32) as char);
            } else {
                result.push(b as char);
            }
        }
    }

    Cow::Owned(result)
}

/// Fast find-and-replace using SIMD-accelerated memchr/memmem.
///
/// Replaces all occurrences of `needle` with `replacement` in `text`.
/// Returns `Cow::Borrowed` when no matches are found (zero allocation).
///
/// # Example
///
/// ```
/// use tokie::normalizer::fnr;
/// use std::borrow::Cow;
///
/// let result = fnr("hello world", "world", "rust");
/// assert_eq!(result, "hello rust");
///
/// // No match - returns borrowed
/// let result = fnr("hello", "xyz", "abc");
/// assert!(matches!(result, Cow::Borrowed(_)));
/// ```
#[inline]
pub fn fnr<'a>(text: &'a str, needle: &str, replacement: &str) -> Cow<'a, str> {
    use memchr::memmem;

    // Edge cases
    if needle.is_empty() || text.is_empty() {
        return Cow::Borrowed(text);
    }

    let finder = memmem::Finder::new(needle.as_bytes());
    let text_bytes = text.as_bytes();

    // Fast path: check if needle exists at all
    if finder.find(text_bytes).is_none() {
        return Cow::Borrowed(text);
    }

    // Estimate capacity: if replacement is shorter, we save space; if longer, we need more
    let size_diff = replacement.len() as isize - needle.len() as isize;
    let estimated_cap = if size_diff > 0 {
        // Rough estimate: assume ~2 matches on average
        text.len() + (size_diff as usize * 2)
    } else {
        text.len()
    };

    let mut result = String::with_capacity(estimated_cap);
    let mut last_end = 0;

    // Iterate over all matches (positions are absolute to the full text)
    for pos in finder.find_iter(text_bytes) {
        result.push_str(&text[last_end..pos]);
        result.push_str(replacement);
        last_end = pos + needle.len();
    }

    // Add remaining text
    result.push_str(&text[last_end..]);

    Cow::Owned(result)
}

/// Fast find-and-replace with precomputed finder for repeated searches.
///
/// Use this when searching for the same needle across multiple texts.
///
/// # Example
///
/// ```
/// use tokie::normalizer::FnrFinder;
///
/// let finder = FnrFinder::new("foo");
/// let result1 = finder.replace("foo bar foo", "baz");
/// let result2 = finder.replace("no match here", "baz");
///
/// assert_eq!(result1, "baz bar baz");
/// assert_eq!(result2, "no match here");
/// ```
pub struct FnrFinder<'n> {
    needle: &'n str,
    finder: memchr::memmem::Finder<'n>,
}

impl<'n> FnrFinder<'n> {
    /// Create a new finder for the given needle.
    #[inline]
    pub fn new(needle: &'n str) -> Self {
        Self {
            needle,
            finder: memchr::memmem::Finder::new(needle.as_bytes()),
        }
    }

    /// Replace all occurrences of the needle with replacement.
    #[inline]
    pub fn replace<'a>(&self, text: &'a str, replacement: &str) -> Cow<'a, str> {
        if self.needle.is_empty() || text.is_empty() {
            return Cow::Borrowed(text);
        }

        let text_bytes = text.as_bytes();

        // Fast path: check if needle exists
        if self.finder.find(text_bytes).is_none() {
            return Cow::Borrowed(text);
        }

        let size_diff = replacement.len() as isize - self.needle.len() as isize;
        let estimated_cap = if size_diff > 0 {
            text.len() + (size_diff as usize * 2)
        } else {
            text.len()
        };

        let mut result = String::with_capacity(estimated_cap);
        let mut last_end = 0;

        for pos in self.finder.find_iter(text_bytes) {
            result.push_str(&text[last_end..pos]);
            result.push_str(replacement);
            last_end = pos + self.needle.len();
        }

        result.push_str(&text[last_end..]);

        Cow::Owned(result)
    }

    /// Check if the needle exists in the text without allocating.
    #[inline]
    pub fn contains(&self, text: &str) -> bool {
        self.finder.find(text.as_bytes()).is_some()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnr_basic() {
        // Single replacement
        let result = fnr("hello world", "world", "rust");
        assert_eq!(result, "hello rust");
        assert!(matches!(result, Cow::Owned(_)));
    }

    #[test]
    fn test_fnr_multiple() {
        // Multiple replacements
        let result = fnr("foo bar foo baz foo", "foo", "x");
        assert_eq!(result, "x bar x baz x");
    }

    #[test]
    fn test_fnr_no_match() {
        // No match - should borrow
        let result = fnr("hello world", "xyz", "abc");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_fnr_empty_needle() {
        // Empty needle - should borrow
        let result = fnr("hello", "", "x");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_fnr_empty_text() {
        // Empty text - should borrow
        let result = fnr("", "foo", "bar");
        assert!(matches!(result, Cow::Borrowed(_)));
    }

    #[test]
    fn test_fnr_empty_replacement() {
        // Empty replacement (deletion)
        let result = fnr("hello world", " ", "");
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn test_fnr_longer_replacement() {
        // Replacement longer than needle
        let result = fnr("a b c", " ", "---");
        assert_eq!(result, "a---b---c");
    }

    #[test]
    fn test_fnr_at_boundaries() {
        // At start
        let result = fnr("foo bar", "foo", "baz");
        assert_eq!(result, "baz bar");

        // At end
        let result = fnr("bar foo", "foo", "baz");
        assert_eq!(result, "bar baz");

        // Entire string
        let result = fnr("foo", "foo", "bar");
        assert_eq!(result, "bar");
    }

    #[test]
    fn test_fnr_unicode() {
        let result = fnr("héllo wörld", "ö", "o");
        assert_eq!(result, "héllo world");
    }

    #[test]
    fn test_fnr_finder_reuse() {
        let finder = FnrFinder::new("foo");

        let r1 = finder.replace("foo bar foo", "baz");
        assert_eq!(r1, "baz bar baz");

        let r2 = finder.replace("no match", "baz");
        assert!(matches!(r2, Cow::Borrowed(_)));
        assert_eq!(r2, "no match");

        assert!(finder.contains("has foo"));
        assert!(!finder.contains("no match"));
    }

    #[test]
    fn test_none_normalizer() {
        let norm = Normalizer::None;
        let text = "Hello World!";
        let result = norm.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_bert_uncased_lowercase() {
        let norm = Normalizer::BertUncased;

        // Has uppercase - should allocate
        let result = norm.normalize("Hello World!");
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result, "hello world!");

        // Already lowercase - should borrow
        let result = norm.normalize("hello world!");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello world!");
    }

    #[test]
    fn test_bert_uncased_unicode() {
        let norm = Normalizer::BertUncased;

        // Unicode uppercase - accents are stripped for uncased
        let result = norm.normalize("HÉLLO");
        assert_eq!(result, "hello"); // é → e (accent stripped)

        // German sharp S stays as is (lowercase of ß is ß)
        let result = norm.normalize("straße");
        assert_eq!(result, "straße"); // ß has no accent to strip

        // Test accent stripping
        let result = norm.normalize("café résumé naïve");
        assert_eq!(result, "cafe resume naive");
    }

    #[test]
    fn test_nfc_normalizer() {
        let norm = Normalizer::Nfc;

        // Already NFC - should borrow
        let result = norm.normalize("hello");
        assert!(matches!(result, Cow::Borrowed(_)));

        // NFD decomposed é (e + combining acute) -> NFC é
        let nfd = "e\u{0301}"; // e + combining acute accent
        let result = norm.normalize(nfd);
        assert!(matches!(result, Cow::Owned(_)));
        assert_eq!(result, "é");
    }

    #[test]
    fn test_bert_cased() {
        let norm = Normalizer::BertCased;
        let text = "Hello World!";
        let result = norm.normalize(text);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "Hello World!"); // No change
    }

}
