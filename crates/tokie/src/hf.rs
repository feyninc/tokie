//! HuggingFace tokenizer.json loading support.

use std::path::Path;

use crate::encoder::{BacktrackingBytePairEncoder, BytePairEncoder, Encoder, EncoderType, SentencePieceBPE, UnigramEncoder, WordPieceEncoder};
use crate::decoder::Decoder;
use crate::normalizer::Normalizer;
use crate::postprocessor::PostProcessor;
use crate::pretok::{PretokType, Pretokenizer};
use crate::tokenizer::Tokenizer;
use crate::types::TokenId;

/// Error loading from HuggingFace JSON format.
#[derive(Debug)]
pub enum JsonLoadError {
    Io(std::io::Error),
    Json(serde_json::Error),
    InvalidFormat(&'static str),
}

impl std::fmt::Display for JsonLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Json(e) => write!(f, "JSON error: {}", e),
            Self::InvalidFormat(msg) => write!(f, "Invalid format: {}", msg),
        }
    }
}

impl std::error::Error for JsonLoadError {}

impl From<std::io::Error> for JsonLoadError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for JsonLoadError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Load a tokenizer from a HuggingFace tokenizer.json file.
///
/// Only GPT-2 style tokenizers (with `pre_tokenizer.type == "ByteLevel"`) are
/// auto-detected. For cl100k/o200k tokenizers that use Sequence pretokenizers,
/// use [`from_json_with_pretokenizer`] to explicitly specify the type.
///
/// # Example
/// ```ignore
/// use tokie::hf;
///
/// // GPT-2 (auto-detected)
/// let gpt2 = hf::from_json("gpt2_tokenizer.json")?;
///
/// // cl100k (requires explicit type)
/// use tokie::PretokenizerType;
/// let cl100k = hf::from_json_with_pretokenizer("cl100k_tokenizer.json", PretokenizerType::Cl100k)?;
/// ```
pub fn from_json(path: impl AsRef<Path>) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    from_json_str(&json_str)
}

/// Load a tokenizer with a specific pretokenizer type (overriding auto-detection).
///
/// Use this when you want to use a different pretokenizer than what the JSON specifies.
pub fn from_json_with_pretokenizer(
    path: impl AsRef<Path>,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    from_json_str_with_pretokenizer(&json_str, pretokenizer_type)
}

/// Load a tokenizer with a specific encoder type.
///
/// Use this when you want to use a specific encoder algorithm:
/// - `EncoderType::Backtracking`: Fast O(n) with streaming support (default)
/// - `EncoderType::Simple`: Fast O(n²) for small pieces, better parallel throughput
pub fn from_json_with_encoder(
    path: impl AsRef<Path>,
    encoder_type: EncoderType,
) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;
    let detected = detect_pretokenizer_type(&data);
    let mut tok = load_from_json_value_with_encoder(&data, detected.pretok_type, encoder_type)?;
    apply_regex_fallback(&mut tok, &detected);
    Ok(tok)
}

/// Load a tokenizer with both encoder type and pretokenizer type specified.
pub fn from_json_with_options(
    path: impl AsRef<Path>,
    encoder_type: EncoderType,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let json_str = std::fs::read_to_string(path)?;
    let data: serde_json::Value = serde_json::from_str(&json_str)?;
    load_from_json_value_with_encoder(&data, pretokenizer_type, encoder_type)
}

/// Load a tokenizer from a HuggingFace tokenizer.json string.
pub fn from_json_str(json_str: &str) -> Result<Tokenizer, JsonLoadError> {
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    let detected = detect_pretokenizer_type(&data);
    let mut tok = load_from_json_value(&data, detected.pretok_type)?;
    apply_regex_fallback(&mut tok, &detected);
    Ok(tok)
}

/// Load a tokenizer from JSON string with a specific pretokenizer type.
pub fn from_json_str_with_pretokenizer(
    json_str: &str,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    let data: serde_json::Value = serde_json::from_str(json_str)?;
    load_from_json_value(&data, pretokenizer_type)
}

/// Internal: load tokenizer from parsed JSON value.
///
/// Detects whether this is a byte-level BPE tokenizer (GPT-2, cl100k, o200k, p50k)
/// or a vocab-defined BPE tokenizer (SentencePiece-style, some LLaMA variants).
fn load_from_json_value(
    data: &serde_json::Value,
    pretokenizer_type: PretokType,
) -> Result<Tokenizer, JsonLoadError> {
    // Default to Backtracking encoder
    load_from_json_value_with_encoder(data, pretokenizer_type, EncoderType::Backtracking)
}

/// Internal: load tokenizer from parsed JSON value with specific encoder type.
fn load_from_json_value_with_encoder(
    data: &serde_json::Value,
    pretokenizer_type: PretokType,
    encoder_type: EncoderType,
) -> Result<Tokenizer, JsonLoadError> {
    let model = &data["model"];
    let normalizer = detect_normalizer(data);

    // Check if this is a Unigram tokenizer
    // Unigram can be detected by:
    // 1. model.type == "Unigram"
    // 2. model.vocab is an array (not object) and has [token, score] pairs
    //    (T5, ALBERT, XLM-RoBERTa don't set model.type but have array vocab)
    let is_unigram = model["type"].as_str() == Some("Unigram")
        || (model["vocab"].is_array() && model["merges"].is_null());
    if is_unigram {
        return load_unigram(data, pretokenizer_type, normalizer);
    }

    // Check if this is a WordPiece tokenizer
    // WordPiece can be detected by:
    // 1. model.type == "WordPiece"
    // 2. decoder.type == "WordPiece" (HuggingFace BERT format)
    // 3. No merges array + continuing_subword_prefix exists
    let is_wordpiece = model["type"].as_str() == Some("WordPiece")
        || data["decoder"]["type"].as_str() == Some("WordPiece")
        || (model["merges"].is_null() && model["continuing_subword_prefix"].is_string());

    if is_wordpiece {
        return load_wordpiece(data, pretokenizer_type, normalizer);
    }

    let vocab_map = model["vocab"]
        .as_object()
        .ok_or(JsonLoadError::InvalidFormat("vocab should be object"))?;
    let merges_arr = model["merges"]
        .as_array()
        .ok_or(JsonLoadError::InvalidFormat("merges should be array"))?;

    // Build vocabulary mapping sorted by id
    let mut vocab: Vec<(String, u32)> = vocab_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);

    // Detect tokenizer style based on merge order.
    // - Sequential merges (tiktoken): Merge N only references tokens 0 to 255+N-1
    // - Vocab-defined (LLaMA 3, etc.): Merges may reference "future" tokens from vocab
    //
    // We check if merges are in topological order (can be processed sequentially).
    // If not, we need to use vocab-first loading where all token bytes are pre-built.
    let num_base_tokens = detect_num_base_tokens(vocab_map, merges_arr);

    if are_merges_topological(vocab_map, merges_arr, num_base_tokens) {
        load_byte_level_bpe(data, &vocab, vocab_map, merges_arr, pretokenizer_type, encoder_type, normalizer)
    } else {
        // SentencePiece-style tokenizers: use Backtracking encoder which finds
        // vocab tokens directly via Aho-Corasick, better suited than Simple encoder
        // which relies on byte-level merge rules.
        load_vocab_defined_bpe(data, &vocab, vocab_map, merges_arr, pretokenizer_type, encoder_type, normalizer)
    }
}

/// Detect the number of base tokens by finding the first merge result's vocab ID.
///
/// In byte-level BPE, the base tokens are all vocab entries before the first merge result.
/// Standard GPT-2 has 256 base tokens (one per byte), but some models (e.g., ModernBERT)
/// have fewer byte tokens or added tokens at low IDs, shifting the merge start.
fn detect_num_base_tokens(
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
) -> usize {
    // The first merge result's vocab ID tells us where base tokens end.
    if let Some(first_merge) = merges_arr.first() {
        let (left_str, right_str) = if let Some(arr) = first_merge.as_array() {
            if arr.len() >= 2 {
                match (arr[0].as_str(), arr[1].as_str()) {
                    (Some(l), Some(r)) => (l, r),
                    _ => return 256,
                }
            } else {
                return 256;
            }
        } else if let Some(s) = first_merge.as_str() {
            let mut parts = s.split(' ');
            match (parts.next(), parts.next()) {
                (Some(l), Some(r)) => (l, r),
                _ => return 256,
            }
        } else {
            return 256;
        };

        let merged = format!("{}{}", left_str, right_str);
        if let Some(id) = vocab_map.get(&merged).and_then(|v| v.as_u64()) {
            return id as usize;
        }
    }
    256 // fallback for standard GPT-2
}

/// Check if merges are in topological order (can be processed sequentially).
///
/// In tiktoken-style tokenizers, merge N only references tokens 0 to 255+N-1.
/// In vocab-defined tokenizers (like LLaMA 3), merges may reference "future" tokens
/// that exist in the vocab but would be created by later merges.
fn are_merges_topological(
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    num_base_tokens: usize,
) -> bool {
    for (merge_idx, merge) in merges_arr.iter().enumerate() {
        // Extract left and right token strings from merge
        // Handle both string format ("Ġ Ġ") and array format (["Ġ", "Ġ"])
        let (left_str, right_str) = if let Some(arr) = merge.as_array() {
            if arr.len() >= 2 {
                match (arr[0].as_str(), arr[1].as_str()) {
                    (Some(l), Some(r)) => (l, r),
                    _ => continue,
                }
            } else {
                continue;
            }
        } else if let Some(s) = merge.as_str() {
            let mut parts = s.split(' ');
            match (parts.next(), parts.next()) {
                (Some(l), Some(r)) => (l, r),
                _ => continue,
            }
        } else {
            continue;
        };

        let left_id = vocab_map
            .get(left_str)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let right_id = vocab_map
            .get(right_str)
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // At merge N, we have tokens 0 to (num_base_tokens + N - 1)
        let max_available = num_base_tokens + merge_idx;

        if left_id >= max_available || right_id >= max_available {
            // This merge references a token that hasn't been created yet
            return false;
        }
    }
    true
}

/// Load byte-level BPE tokenizer (GPT-2, cl100k, o200k, p50k, ModernBERT).
///
/// These tokenizers have:
/// - Base tokens (byte tokens + any low-ID added tokens) before the first merge
/// - Each merge creates the next sequential token ID
/// - Vocab IDs match: base_tokens + merge_order
fn load_byte_level_bpe(
    data: &serde_json::Value,
    vocab: &[(String, u32)],
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    pretokenizer_type: PretokType,
    encoder_type: EncoderType,
    normalizer: Normalizer,
) -> Result<Tokenizer, JsonLoadError> {
    // Detect the actual number of base tokens (may differ from 256 if added tokens
    // occupy low IDs, e.g., ModernBERT has 245 base tokens instead of 256)
    let num_base_tokens = detect_num_base_tokens(vocab_map, merges_arr);

    // Build full vocab with byte-level decoding (includes all vocab entries, not just
    // merge-derived ones, so the Aho-Corasick automaton covers the entire vocabulary)
    let full_vocab: Vec<(u32, Vec<u8>)> = vocab
        .iter()
        .map(|(s, id)| (*id, decode_bytelevel_token(s)))
        .collect();

    // Build merges with proper ID mapping
    // Handle both string format ("Ġ Ġ") and array format (["Ġ", "Ġ"])
    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            // Try array format first (Llama 4 style): ["left", "right"]
            if let Some(arr) = m.as_array() {
                if arr.len() >= 2 {
                    let left_str = arr[0].as_str()?;
                    let right_str = arr[1].as_str()?;
                    let left = vocab_map.get(left_str)?.as_u64()? as u32;
                    let right = vocab_map.get(right_str)?.as_u64()? as u32;
                    return Some((left, right));
                }
            }
            // Fall back to string format (GPT-2/Llama 3 style): "left right"
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            let right = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    let (encoder, token_bytes) = match encoder_type {
        EncoderType::Backtracking | EncoderType::WordPiece | EncoderType::SentencePiece | EncoderType::Unigram => {
            let (enc, bytes) = BacktrackingBytePairEncoder::from_vocab_and_merges(
                &full_vocab, &merges, num_base_tokens,
            );
            (Encoder::Backtracking(enc), bytes)
        }
        EncoderType::Simple => {
            let (enc, bytes) =
                BytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, num_base_tokens);
            (Encoder::Simple(enc), bytes)
        }
    };
    let decoder = Decoder::for_encoder(token_bytes, encoder.encoder_type());
    let post_processor = detect_post_processor(data);

    let mut tokenizer = Tokenizer::new(encoder, decoder, pretokenizer_type, normalizer, post_processor);
    if let Some(pad_id) = extract_pad_token_id(data) {
        tokenizer.set_pad_token_id(pad_id);
    }
    setup_added_tokens(&mut tokenizer, data);
    Ok(tokenizer)
}

/// Load vocab-defined BPE tokenizer (LLaMA 3, Mistral, SentencePiece-style, etc.).
///
/// These tokenizers have:
/// - Vocab with pre-assigned IDs (not necessarily sequential with merges)
/// - Merges may reference "future" tokens that exist in vocab
/// - Need to pre-build all token bytes from vocab before processing merges
fn load_vocab_defined_bpe(
    data: &serde_json::Value,
    vocab: &[(String, u32)],
    vocab_map: &serde_json::Map<String, serde_json::Value>,
    merges_arr: &[serde_json::Value],
    pretokenizer_type: PretokType,
    encoder_type: EncoderType,
    normalizer: Normalizer,
) -> Result<Tokenizer, JsonLoadError> {
    // Detect token encoding style from the decoder configuration
    let uses_bytelevel = is_bytelevel_decoder(data);

    // Build full vocab with proper byte handling
    let mut byte_fallback_ids = foldhash::HashSet::default();
    let full_vocab: Vec<(u32, Vec<u8>)> = vocab
        .iter()
        .map(|(s, id)| {
            let bytes = if uses_bytelevel {
                decode_bytelevel_token(s)
            } else if let Some(byte_val) = parse_byte_fallback_token(s) {
                byte_fallback_ids.insert(*id);
                vec![byte_val]
            } else {
                decode_sentencepiece_token(s)
            };
            (*id, bytes)
        })
        .collect();

    // Find number of base tokens (single-byte tokens at the start)
    // For SentencePiece, this varies but we need at least byte coverage
    let num_base_tokens = full_vocab
        .iter()
        .take_while(|(_, bytes)| bytes.len() == 1)
        .count()
        .max(256);

    // Build merges with proper ID mapping
    // Handle both string format ("Ġ Ġ") and array format (["Ġ", "Ġ"])
    let merges: Vec<(u32, u32)> = merges_arr
        .iter()
        .filter_map(|m| {
            // Try array format first (Llama 4 style): ["left", "right"]
            if let Some(arr) = m.as_array() {
                if arr.len() >= 2 {
                    let left_str = arr[0].as_str()?;
                    let right_str = arr[1].as_str()?;
                    let left = vocab_map.get(left_str)?.as_u64()? as u32;
                    let right = vocab_map.get(right_str)?.as_u64()? as u32;
                    return Some((left, right));
                }
            }
            // Fall back to string format (GPT-2/Llama 3 style): "left right"
            let s = m.as_str()?;
            let mut parts = s.split(' ');
            let left = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            let right = vocab_map.get(parts.next()?)?.as_u64()? as u32;
            Some((left, right))
        })
        .collect();

    // Use SentencePiece encoder if explicitly requested or if Metaspace-style normalizer detected
    // Both Metaspace (Mistral) and MetaspaceReplace (Gemma) indicate SentencePiece tokenizers
    let use_sentencepiece = encoder_type == EncoderType::SentencePiece
        || matches!(normalizer, Normalizer::Metaspace | Normalizer::MetaspaceReplace);

    // For ByteLevel vocab-defined BPE (Llama 3, Qwen), use Simple encoder
    // because Backtracking's is_valid_pair doesn't work with non-sequential IDs
    let use_simple = uses_bytelevel && encoder_type != EncoderType::SentencePiece;

    let (encoder, token_bytes) = if use_sentencepiece {
        // SentencePiece encoder with radix heap
        let (enc, bytes) =
            SentencePieceBPE::from_vocab_and_merges(&full_vocab, &merges, num_base_tokens, &byte_fallback_ids);
        (Encoder::SentencePiece(enc), bytes)
    } else if use_simple || encoder_type == EncoderType::Simple {
        // Simple encoder for ByteLevel vocab-defined BPE or explicit request
        let (enc, bytes) =
            BytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, num_base_tokens);
        (Encoder::Simple(enc), bytes)
    } else {
        // Backtracking for sequential-ID BPE (GPT-2, cl100k, o200k)
        let (enc, bytes) =
            BacktrackingBytePairEncoder::from_vocab_and_merges(&full_vocab, &merges, num_base_tokens);
        (Encoder::Backtracking(enc), bytes)
    };
    let decoder = Decoder::for_encoder(token_bytes, encoder.encoder_type());
    let post_processor = detect_post_processor(data);

    let mut tokenizer = Tokenizer::new(encoder, decoder, pretokenizer_type, normalizer, post_processor);
    if let Some(pad_id) = extract_pad_token_id(data) {
        tokenizer.set_pad_token_id(pad_id);
    }
    setup_added_tokens(&mut tokenizer, data);
    Ok(tokenizer)
}

/// Parse a byte fallback token like `<0x0A>` and return the byte value.
/// Returns None if the token doesn't match the pattern.
#[inline]
fn parse_byte_fallback_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Detect the pretokenizer type from HuggingFace JSON.
///
/// Auto-detects based on:
/// - Pre-tokenizer type (ByteLevel = GPT-2)
/// - Vocabulary size (~100K = cl100k, ~200K = o200k)
///
/// For edge cases, use `from_json_with_pretokenizer` to explicitly specify.
/// Apply regex fallback pretokenizer if the pattern was unrecognized.
fn apply_regex_fallback(tok: &mut Tokenizer, detected: &DetectedPretokenizer) {
    if tok.pretokenizer().is_none() {
        if let Some(pattern) = &detected.fallback_pattern {
            // Handle the common \s+(?!\S) negative lookahead pattern:
            // Split into \s+\s (with lookahead trim) and \s+ (without)
            let patterns = if pattern.contains("(?!\\S)") || pattern.contains("(?!\\s)") {
                let main = pattern.replace("\\s+(?!\\S)", "\\s+$")
                                  .replace("\\s+(?!\\s)", "\\s+$");
                vec![
                    (main, false),
                    ("\\s+\\s".to_string(), true),
                    ("\\s+".to_string(), false),
                ]
            } else {
                vec![(pattern.clone(), false)]
            };

            let pat_refs: Vec<(&str, bool)> = patterns.iter()
                .map(|(p, l)| (p.as_str(), *l))
                .collect();

            if let Ok(regex) = pretokie::Regex::new(&pat_refs) {
                tok.set_pretokenizer(Some(Pretokenizer::from_regex(regex)));
            }
        }
    }
}

/// Result of pretokenizer detection from JSON.
struct DetectedPretokenizer {
    pretok_type: PretokType,
    /// Raw regex pattern for fallback when pretok_type is None but a pattern was found.
    fallback_pattern: Option<String>,
}

fn detect_pretokenizer_type(data: &serde_json::Value) -> DetectedPretokenizer {
    let pre_tokenizer = &data["pre_tokenizer"];

    // Check pre_tokenizer type first
    if let Some(typ) = pre_tokenizer["type"].as_str() {
        // ByteLevel pre-tokenizer (GPT-2 style)
        if typ == "ByteLevel" {
            return DetectedPretokenizer { pretok_type: PretokType::Gpt2, fallback_pattern: None };
        }

        // Sequence pretokenizers - check for ByteLevel inside
        // This handles Llama 3, Qwen, and similar models that use Split + ByteLevel
        if typ == "Sequence" {
            if let Some(pretokenizers) = pre_tokenizer["pretokenizers"].as_array() {
                let has_byte_level = pretokenizers
                    .iter()
                    .any(|p| p["type"].as_str() == Some("ByteLevel"));

                if has_byte_level {
                    // Check for Digits pretokenizer (SmolLM2 style)
                    let has_digits = pretokenizers
                        .iter()
                        .any(|p| p["type"].as_str() == Some("Digits"));

                    // Collect all Split regex patterns for potential fallback
                    let mut split_patterns: Vec<String> = Vec::new();

                    // Digit chunking may live in its own Split stage (DeepSeek puts
                    // \p{N}{1,3} before the letter pattern), so scan all stages up front
                    let has_chunked_digits = pretokenizers.iter().any(|p| {
                        p["type"].as_str() == Some("Split")
                            && p["pattern"]["Regex"].as_str().is_some_and(|pat| pat.contains("\\p{N}{"))
                    });

                    // Check for Split with regex pattern to determine exact type
                    for p in pretokenizers {
                        if p["type"].as_str() == Some("Split") {
                            if let Some(pattern) = p["pattern"]["Regex"].as_str() {
                                split_patterns.push(pattern.to_string());

                                // O200K has case-aware letter patterns like [\p{Lu}\p{Lt}...]
                                // This splits CamelCase words
                                let is_case_aware = pattern.contains("\\p{Lu}")
                                    || pattern.contains("\\p{Lt}")
                                    || pattern.contains("\\p{Ll}");

                                if is_case_aware {
                                    return DetectedPretokenizer { pretok_type: PretokType::O200k, fallback_pattern: None };
                                }

                                // Patterns with [\p{L}\p{M}]+ include combining marks
                                if pattern.contains("[\\p{L}\\p{M}]+") {
                                    // DeepSeek: multi-stage splits, no contractions in main pattern,
                                    // digit groups \p{N}{1,3} (in an earlier Split stage)
                                    // Qwen3.5: single split, has contractions, single digits \p{N}
                                    if has_chunked_digits {
                                        return DetectedPretokenizer { pretok_type: PretokType::DeepSeek, fallback_pattern: None };
                                    }
                                    // Single-digit pattern with marks = Voyage + marks (Qwen3.5)
                                    return DetectedPretokenizer { pretok_type: PretokType::Qwen35, fallback_pattern: None };
                                }

                                // Simple \p{L}+ pattern = CL100K style (no CamelCase split)
                                // This includes Llama 3, Qwen, and similar models
                                if pattern.contains("\\p{L}+") || pattern.contains("(?i:'s|'t|'re") {
                                    // Check number chunking: \p{N}{1,3} = CL100K, \p{N}| = Voyage
                                    // Voyage uses single digits, CL100K uses groups of 3
                                    if pattern.contains("\\p{N}|") && !pattern.contains("\\p{N}{") {
                                        return DetectedPretokenizer { pretok_type: PretokType::Voyage, fallback_pattern: None };
                                    }
                                    return DetectedPretokenizer { pretok_type: PretokType::Cl100k, fallback_pattern: None };
                                }
                            }
                        }
                    }
                    // Default ByteLevel sequence to GPT-2
                    // If Digits pretokenizer is present, use GPT-2 with individual digit isolation
                    if has_digits {
                        return DetectedPretokenizer { pretok_type: PretokType::SmolLM, fallback_pattern: None };
                    }
                    // If we found Split patterns but couldn't classify them, return as fallback
                    if !split_patterns.is_empty() {
                        return DetectedPretokenizer {
                            pretok_type: PretokType::Gpt2, // default ByteLevel
                            fallback_pattern: Some(split_patterns.join("|")),
                        };
                    }
                    return DetectedPretokenizer { pretok_type: PretokType::Gpt2, fallback_pattern: None };
                }

                // No ByteLevel but has Split patterns — try regex fallback
                let mut split_patterns: Vec<String> = Vec::new();
                for p in pretokenizers {
                    if p["type"].as_str() == Some("Split") {
                        if let Some(pattern) = p["pattern"]["Regex"].as_str() {
                            split_patterns.push(pattern.to_string());
                        }
                    }
                }
                if !split_patterns.is_empty() {
                    return DetectedPretokenizer {
                        pretok_type: PretokType::None,
                        fallback_pattern: Some(split_patterns.join("|")),
                    };
                }
            }
        }

        // Metaspace pre-tokenizer (SentencePiece style - Mistral, Llama 2)
        // These work with PretokType::None + Metaspace normalizer
        if typ == "Metaspace" {
            return DetectedPretokenizer { pretok_type: PretokType::None, fallback_pattern: None };
        }

        // Single Split pretokenizer with regex
        if typ == "Split" {
            if let Some(pattern) = pre_tokenizer["pattern"]["Regex"].as_str() {
                return DetectedPretokenizer {
                    pretok_type: PretokType::None,
                    fallback_pattern: Some(pattern.to_string()),
                };
            }
        }
    }

    // Unknown - return None
    DetectedPretokenizer { pretok_type: PretokType::None, fallback_pattern: None }
}

/// Detect the normalizer from HuggingFace JSON.
///
/// Auto-detects based on normalizer and pre_tokenizer configuration:
/// - Precompiled + WhitespaceSplit + Metaspace + Lowercase → SentencePieceLowercase (ALBERT)
/// - Precompiled + WhitespaceSplit + Metaspace → SentencePiece (T5, XLM-RoBERTa)
/// - BertNormalizer with lowercase=true → BertUncased
/// - BertNormalizer with lowercase=false → BertCased
/// - NFC → Nfc
/// - Sequence containing NFC → Nfc
/// - Metaspace pre_tokenizer only → Metaspace (Llama, Mistral)
/// - None or null → None
fn detect_normalizer(data: &serde_json::Value) -> Normalizer {
    let normalizer = &data["normalizer"];
    let has_metaspace = is_metaspace_pretokenizer(data);
    let has_whitespace_split = is_whitespace_split_pretokenizer(data);

    // Handle null/missing normalizer - check for Metaspace pre_tokenizer
    if normalizer.is_null() {
        if has_metaspace {
            return Normalizer::Metaspace;
        }
        return Normalizer::None;
    }

    // Check normalizer type
    if let Some(typ) = normalizer["type"].as_str() {
        match typ {
            "Precompiled" => {
                // Precompiled charsmap = SentencePiece normalization
                // Models: T5 (with WhitespaceSplit), bge-m3 (without WhitespaceSplit)
                if has_metaspace {
                    // Prefer the model's exact charsmap blob; fall back to the
                    // category-rule approximation if it's missing or malformed.
                    return parse_precompiled_charsmap(normalizer, has_whitespace_split)
                        .unwrap_or(Normalizer::SentencePiece);
                }
            }
            "BertNormalizer" => {
                // BertNormalizer with lowercase option
                let lowercase = normalizer["lowercase"].as_bool().unwrap_or(false);
                if lowercase {
                    return Normalizer::BertUncased;
                } else {
                    return Normalizer::BertCased;
                }
            }
            "NFC" => {
                return Normalizer::Nfc;
            }
            "Sequence" => {
                // Check sequence normalizers for specific patterns
                if let Some(normalizers) = normalizer["normalizers"].as_array() {
                    let has_lowercase = normalizers.iter().any(|n| {
                        n["type"].as_str() == Some("Lowercase")
                    });
                    let has_precompiled = normalizers.iter().any(|n| {
                        n["type"].as_str() == Some("Precompiled")
                    });

                    // Check for Prepend+Replace pattern (Mixtral style metaspace)
                    // Prepend "▁" + Replace " " → "▁" = same as Metaspace
                    let has_prepend_metaspace = normalizers.iter().any(|n| {
                        n["type"].as_str() == Some("Prepend")
                            && n["prepend"].as_str() == Some("▁")
                    });
                    let has_replace_space_metaspace = normalizers.iter().any(|n| {
                        n["type"].as_str() == Some("Replace")
                            && n["pattern"]["String"].as_str() == Some(" ")
                            && n["content"].as_str() == Some("▁")
                    });

                    // ALBERT pattern: Sequence with Lowercase + Precompiled + WhitespaceSplit + Metaspace
                    if has_precompiled && has_lowercase && has_metaspace && has_whitespace_split {
                        return Normalizer::SentencePieceLowercase;
                    }

                    // General SentencePiece: Sequence with Precompiled + Metaspace
                    // (with or without WhitespaceSplit — bge-m3 omits it)
                    if has_precompiled && has_metaspace {
                        return normalizers
                            .iter()
                            .find(|n| n["type"].as_str() == Some("Precompiled"))
                            .and_then(|n| parse_precompiled_charsmap(n, has_whitespace_split))
                            .unwrap_or(Normalizer::SentencePiece);
                    }

                    // Mixtral pattern: Prepend "▁" + Replace " " → "▁"
                    if has_prepend_metaspace && has_replace_space_metaspace {
                        return Normalizer::Metaspace;
                    }

                    // Check for NFC or BertNormalizer in sequence
                    for n in normalizers {
                        if let Some(n_type) = n["type"].as_str() {
                            if n_type == "NFC" {
                                return Normalizer::Nfc;
                            }
                            if n_type == "BertNormalizer" {
                                let lowercase = n["lowercase"].as_bool().unwrap_or(false);
                                if lowercase {
                                    return Normalizer::BertUncased;
                                } else {
                                    return Normalizer::BertCased;
                                }
                            }
                        }
                    }
                }
            }
            "Lowercase" => {
                // Simple lowercase normalizer - check if also has Metaspace
                if has_metaspace && has_whitespace_split {
                    return Normalizer::SentencePieceLowercase;
                }
                return Normalizer::BertUncased;
            }
            "Replace" => {
                // Replace normalizer - check if it converts space to metaspace (Gemma style)
                // Unlike Metaspace, this does NOT prepend ▁ at the start
                let pattern = normalizer["pattern"]["String"].as_str();
                let content = normalizer["content"].as_str();
                if pattern == Some(" ") && content == Some("▁") {
                    return Normalizer::MetaspaceReplace;
                }
            }
            _ => {}
        }
    }

    // Even if normalizer exists, check for Metaspace pre_tokenizer
    if has_metaspace {
        return Normalizer::Metaspace;
    }

    Normalizer::None
}

/// Parse the base64 `precompiled_charsmap` blob from a Precompiled normalizer
/// JSON node. Returns `None` if absent or malformed (caller falls back to the
/// approximate SentencePiece normalizer).
///
/// `whitespace_split` selects the metaspace shape: WhitespaceSplit + Metaspace
/// (XLM-R, T5) vs Replace `" {2,}"` + Metaspace only (bge-m3 family).
fn parse_precompiled_charsmap(
    normalizer: &serde_json::Value,
    whitespace_split: bool,
) -> Option<Normalizer> {
    use base64::Engine as _;
    let b64 = normalizer["precompiled_charsmap"].as_str()?;
    let blob = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
    let charsmap = crate::charsmap::PrecompiledCharsmap::from_blob(&blob).ok()?;
    Some(Normalizer::SentencePiecePrecompiled {
        charsmap: std::sync::Arc::new(charsmap),
        whitespace_split,
    })
}

/// Check if the tokenizer uses a WhitespaceSplit pre_tokenizer.
fn is_whitespace_split_pretokenizer(data: &serde_json::Value) -> bool {
    let pre_tokenizer = &data["pre_tokenizer"];

    // Direct WhitespaceSplit type
    if let Some(typ) = pre_tokenizer["type"].as_str() {
        if typ == "WhitespaceSplit" {
            return true;
        }
    }

    // Sequence containing WhitespaceSplit
    if let Some(pretokenizers) = pre_tokenizer["pretokenizers"].as_array() {
        for p in pretokenizers {
            if let Some(typ) = p["type"].as_str() {
                if typ == "WhitespaceSplit" {
                    return true;
                }
            }
        }
    }

    false
}
/// Check if the tokenizer uses a Metaspace pre_tokenizer.
///
/// Metaspace is used by SentencePiece models (Llama, Mistral, Gemma, etc.)
/// and replaces spaces with ▁ (U+2581).
fn is_metaspace_pretokenizer(data: &serde_json::Value) -> bool {
    let pre_tokenizer = &data["pre_tokenizer"];

    // Direct Metaspace type
    if let Some(typ) = pre_tokenizer["type"].as_str() {
        if typ == "Metaspace" {
            return true;
        }
    }

    // Sequence containing Metaspace
    if let Some(pretokenizers) = pre_tokenizer["pretokenizers"].as_array() {
        for p in pretokenizers {
            if let Some(typ) = p["type"].as_str() {
                if typ == "Metaspace" {
                    return true;
                }
            }
        }
    }

    false
}

/// Decode a HuggingFace ByteLevel token string to bytes.
///
/// HuggingFace's ByteLevel encoding maps bytes to visible Unicode characters:
/// - Printable bytes (33-126, 161-172, 174-255) map to their character
/// - Non-printable bytes (0-32, 127-160, 173) are encoded as U+0100+n
///
/// This encoding is used by GPT-2, LLaMA 3, and most modern BPE tokenizers.
fn decode_bytelevel_token(s: &str) -> Vec<u8> {
    // Non-printable bytes in the order GPT-2 encodes them (mapped to U+0100+)
    // This is: 0-32 (33 bytes), 127-160 (34 bytes), 173 (1 byte) = 68 bytes total
    static NON_PRINTABLE: [u8; 68] = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, // 0-32
        127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
        145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, // 127-160
        173, // 173
    ];

    let mut bytes = Vec::with_capacity(s.len());
    for c in s.chars() {
        let code = c as u32;

        let b = if code >= 256 && code < 256 + NON_PRINTABLE.len() as u32 {
            // Non-printable byte encoded as U+0100+n
            NON_PRINTABLE[(code - 256) as usize]
        } else if code <= 255 {
            // Direct byte mapping (printable characters)
            code as u8
        } else {
            // Outside GPT-2 encoding range - shouldn't happen in valid tokens
            bytes.extend(c.to_string().as_bytes());
            continue;
        };
        bytes.push(b);
    }

    bytes
}

/// Check if the tokenizer uses ByteLevel decoding (vs ByteFallback/SentencePiece).
///
/// ByteLevel: Uses Ġ (U+0120) for space, characters map to bytes
/// ByteFallback: Uses ▁ (U+2581) for space, <0xXX> for raw bytes
fn is_bytelevel_decoder(data: &serde_json::Value) -> bool {
    let decoder = &data["decoder"];

    // Check decoder type directly
    if let Some(typ) = decoder["type"].as_str() {
        if typ == "ByteLevel" {
            return true;
        }
    }

    // Check for Sequence decoder containing ByteLevel
    if let Some(decoders) = decoder["decoders"].as_array() {
        for d in decoders {
            if let Some(typ) = d["type"].as_str() {
                if typ == "ByteLevel" {
                    return true;
                }
            }
        }
    }

    // Check pre_tokenizer for ByteLevel (often correlates)
    if let Some(pretoks) = data["pre_tokenizer"]["pretokenizers"].as_array() {
        for p in pretoks {
            if let Some(typ) = p["type"].as_str() {
                if typ == "ByteLevel" {
                    return true;
                }
            }
        }
    }

    false
}

/// Decode a SentencePiece token string to bytes.
///
/// SentencePiece uses different encoding than ByteLevel:
/// - ▁ (U+2581) is kept as-is (UTF-8: \xe2\x96\x81) for matching during encoding
/// - <0xXX> patterns represent raw bytes (e.g., <0x0A> for newline)
/// - Other characters are their UTF-8 representation
///
/// Note: ▁ → space conversion happens during decoding output, not vocab building.
fn decode_sentencepiece_token(s: &str) -> Vec<u8> {
    // Handle <0xXX> byte patterns (e.g., "<0x0A>" for newline)
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        if let Ok(byte) = u8::from_str_radix(&s[3..5], 16) {
            return vec![byte];
        }
    }

    // Keep ▁ as UTF-8 bytes (\xe2\x96\x81) - don't replace with space
    // The Metaspace normalizer adds ▁ to input, so tokens must match
    s.as_bytes().to_vec()
}

/// Load a WordPiece tokenizer (BERT-style).
///
/// WordPiece tokenizers have:
/// - `model.type`: "WordPiece"
/// - `model.vocab`: vocabulary mapping token string → ID
/// - `model.unk_token`: the unknown token string (e.g., "[UNK]")
/// - `model.continuing_subword_prefix`: continuation prefix (e.g., "##")
fn load_wordpiece(
    data: &serde_json::Value,
    pretokenizer_type: PretokType,
    normalizer: Normalizer,
) -> Result<Tokenizer, JsonLoadError> {
    let model = &data["model"];
    let vocab_map = model["vocab"]
        .as_object()
        .ok_or(JsonLoadError::InvalidFormat("vocab should be object"))?;

    // Get WordPiece-specific config
    let unk_token_str = model["unk_token"]
        .as_str()
        .unwrap_or("[UNK]");
    let continuation_prefix = model["continuing_subword_prefix"]
        .as_str()
        .unwrap_or("##");
    let max_input_chars_per_word = model["max_input_chars_per_word"]
        .as_u64()
        .unwrap_or(100) as usize;

    // Build vocabulary mapping sorted by id
    let mut vocab: Vec<(String, u32)> = vocab_map
        .iter()
        .map(|(k, v)| (k.clone(), v.as_u64().unwrap_or(0) as u32))
        .collect();
    vocab.sort_by_key(|(_, id)| *id);

    // Find the unk_token ID
    let unk_token = vocab_map
        .get(unk_token_str)
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    // Build token bytes from vocab (tokens are UTF-8 strings)
    let token_bytes: Vec<Vec<u8>> = vocab
        .iter()
        .map(|(s, _)| s.as_bytes().to_vec())
        .collect();

    // Build vocab pairs for WordPieceEncoder
    let vocab_pairs: Vec<(Vec<u8>, u32)> = token_bytes
        .iter()
        .enumerate()
        .map(|(i, bytes)| (bytes.clone(), i as u32))
        .collect();

    // Create WordPiece encoder
    let encoder = WordPieceEncoder::from_vocab(
        &vocab_pairs,
        unk_token,
        continuation_prefix.as_bytes(),
        max_input_chars_per_word,
    );

    // Build decoder
    let decoder = Decoder::for_encoder(token_bytes, EncoderType::WordPiece);

    // Use BERT pretokenizer if not specified
    let pretok = if pretokenizer_type == PretokType::None {
        // Auto-detect: WordPiece typically uses BERT-style pretokenization
        PretokType::Bert
    } else {
        pretokenizer_type
    };

    let post_processor = detect_post_processor(data);

    let mut tokenizer = Tokenizer::new(Encoder::WordPiece(encoder), decoder, pretok, normalizer, post_processor);
    if let Some(pad_id) = extract_pad_token_id(data) {
        tokenizer.set_pad_token_id(pad_id);
    }
    setup_added_tokens(&mut tokenizer, data);
    Ok(tokenizer)
}

/// Load a Unigram tokenizer (SentencePiece Unigram models like T5, XLM-RoBERTa, ALBERT).
///
/// Unigram tokenizers have:
/// - `model.type`: "Unigram"
/// - `model.vocab`: Array of `[token_string, score]` pairs
/// - `model.unk_id`: The unknown token ID (index in vocab)
fn load_unigram(
    data: &serde_json::Value,
    pretokenizer_type: PretokType,
    normalizer: Normalizer,
) -> Result<Tokenizer, JsonLoadError> {
    let model = &data["model"];

    // Unigram vocab is an array of [token_string, score] pairs
    let vocab_arr = model["vocab"]
        .as_array()
        .ok_or(JsonLoadError::InvalidFormat("Unigram vocab should be array"))?;

    // Get unk_id (default to 0 if not present)
    let unk_id = model["unk_id"].as_u64().unwrap_or(0) as u32;

    // Parse vocab with scores
    let vocab: Vec<(u32, Vec<u8>, f64)> = vocab_arr
        .iter()
        .enumerate()
        .filter_map(|(id, entry)| {
            let arr = entry.as_array()?;
            if arr.len() < 2 {
                return None;
            }
            let token_str = arr[0].as_str()?;
            let score = arr[1].as_f64()?;
            let bytes = decode_sentencepiece_token(token_str);
            Some((id as u32, bytes, score))
        })
        .collect();

    if vocab.is_empty() {
        return Err(JsonLoadError::InvalidFormat("Unigram vocab is empty"));
    }

    // Create Unigram encoder
    let (encoder, token_bytes) = UnigramEncoder::from_vocab_with_scores(&vocab, unk_id);

    // Build decoder
    let decoder = Decoder::for_encoder(token_bytes, EncoderType::Unigram);

    // Determine pretokenizer - Unigram models often use Metaspace pretokenizer
    let pretok = if pretokenizer_type == PretokType::None {
        // Check if there's a Metaspace pre_tokenizer
        if is_metaspace_pretokenizer(data) {
            PretokType::None // Let normalizer handle metaspace
        } else {
            PretokType::None
        }
    } else {
        pretokenizer_type
    };

    let post_processor = detect_post_processor(data);

    let mut tokenizer = Tokenizer::new(Encoder::Unigram(encoder), decoder, pretok, normalizer, post_processor);
    if let Some(pad_id) = extract_pad_token_id(data) {
        tokenizer.set_pad_token_id(pad_id);
    }
    setup_added_tokens(&mut tokenizer, data);
    Ok(tokenizer)
}

/// Detect the post-processor type from HuggingFace JSON.
///
/// Looks at the `post_processor` field to determine what special tokens
/// to add during encoding.
fn detect_post_processor(data: &serde_json::Value) -> PostProcessor {
    let pp = &data["post_processor"];

    let pp_type = pp["type"].as_str().unwrap_or("");

    match pp_type {
        "TemplateProcessing" => {
            // Parse template processing for BERT-style tokenizers
            // Look for patterns like "[CLS]:0 $A:0 [SEP]:0" in single template
            if let Some(single) = pp["single"].as_array() {
                parse_template_post_processor(data, single)
            } else {
                PostProcessor::None
            }
        }
        "Sequence" => {
            // Check for LLaMA 3 style: Sequence of [ByteLevel, TemplateProcessing]
            if let Some(processors) = pp["processors"].as_array() {
                for processor in processors {
                    if processor["type"].as_str() == Some("TemplateProcessing") {
                        if let Some(single) = processor["single"].as_array() {
                            return parse_template_post_processor(data, single);
                        }
                    }
                }
            }
            PostProcessor::None
        }
        _ => PostProcessor::None,
    }
}

/// Parse a TemplateProcessing post-processor's single template.
fn parse_template_post_processor(data: &serde_json::Value, single: &[serde_json::Value]) -> PostProcessor {
    // Check for BERT pattern: [CLS] $A [SEP]
    // In HF format: [{"SpecialToken": {"id": "[CLS]", ...}}, {"Sequence": ...}, {"SpecialToken": {"id": "[SEP]", ...}}]

    let mut cls_token = None;
    let mut sep_token = None;
    let mut bos_token = None;

    for item in single {
        if let Some(special) = item.get("SpecialToken") {
            if let Some(id) = special["id"].as_str() {
                let token_id = lookup_special_token_id(data, id);

                match id {
                    "[CLS]" => cls_token = token_id,
                    "[SEP]" => sep_token = token_id,
                    "<|begin_of_text|>" | "<s>" | "<bos>" => bos_token = token_id,
                    _ => {}
                }
            }
        }
    }

    // Determine the post-processor type
    if let (Some(cls), Some(sep)) = (cls_token, sep_token) {
        PostProcessor::Bert { cls_token: cls, sep_token: sep }
    } else if let Some(bos) = bos_token {
        PostProcessor::Prefix { bos_token: bos }
    } else {
        PostProcessor::None
    }
}

/// Extract pad_token_id from HuggingFace tokenizer.json.
///
/// Checks (in order):
/// 1. `padding.pad_id` — HuggingFace tokenizer.json padding config
/// 2. `added_tokens` — looks for tokens named `[PAD]`, `<pad>`, or `<|pad|>`
fn extract_pad_token_id(data: &serde_json::Value) -> Option<TokenId> {
    // Check padding config
    if let Some(pad_id) = data["padding"]["pad_id"].as_u64() {
        return Some(pad_id as TokenId);
    }

    // Check added_tokens for common pad token names
    if let Some(added) = data["added_tokens"].as_array() {
        for token in added {
            if let Some(content) = token["content"].as_str() {
                match content {
                    "[PAD]" | "<pad>" | "<|pad|>" => {
                        return token["id"].as_u64().map(|id| id as TokenId);
                    }
                    _ => {}
                }
            }
        }
    }

    None
}

/// Extract added tokens from HuggingFace tokenizer.json.
///
/// HuggingFace tokenizers match ALL added tokens (both special and non-special)
/// before pretokenization. This ensures tokens like `<|im_start|>`, `<think>`,
/// and multi-space sequences are recognized as single tokens.
/// Returns (token_id, bytes) pairs.
fn extract_added_tokens(data: &serde_json::Value) -> Vec<(TokenId, Vec<u8>)> {
    let Some(added) = data["added_tokens"].as_array() else {
        return Vec::new();
    };

    added.iter().filter_map(|token| {
        let id = token["id"].as_u64()? as TokenId;
        let content = token["content"].as_str()?;
        if content.is_empty() {
            return None;
        }
        // Skip single-byte tokens — they're already in the BPE vocab and
        // matching them in the DAAC would add overhead with no benefit.
        if content.len() == 1 {
            return None;
        }
        Some((id, content.as_bytes().to_vec()))
    }).collect()
}

/// Set up added tokens and special token metadata on a tokenizer from HF JSON data.
fn setup_added_tokens(tokenizer: &mut Tokenizer, data: &serde_json::Value) {
    let added = extract_added_tokens(data);
    if !added.is_empty() {
        tokenizer.set_added_tokens(&added);
    }
    // Extract special token metadata (string -> ID mapping)
    if let Some(added_arr) = data["added_tokens"].as_array() {
        let special: Vec<(String, TokenId)> = added_arr.iter().filter_map(|token| {
            let special = token["special"].as_bool().unwrap_or(false);
            if !special { return None; }
            let id = token["id"].as_u64()? as TokenId;
            let content = token["content"].as_str()?;
            Some((content.to_string(), id))
        }).collect();
        if !special.is_empty() {
            tokenizer.set_special_tokens(special);
        }
    }
}

/// Look up a special token's ID from added_tokens or vocab.
fn lookup_special_token_id(data: &serde_json::Value, token_str: &str) -> Option<u32> {
    // First check added_tokens
    if let Some(added) = data["added_tokens"].as_array() {
        for token in added {
            if token["content"].as_str() == Some(token_str) {
                return token["id"].as_u64().map(|id| id as u32);
            }
        }
    }

    // Then check vocab
    if let Some(vocab) = data["model"]["vocab"].as_object() {
        if let Some(id) = vocab.get(token_str) {
            return id.as_u64().map(|id| id as u32);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_deepseek_multi_stage_splits() {
        // DeepSeek-V3 puts \p{N}{1,3} in its own Split stage, separate from the
        // [\p{L}\p{M}]+ letter pattern — detection must consider all stages
        let data = serde_json::json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": "\\p{N}{1,3}"}, "behavior": "Isolated", "invert": false},
                    {"type": "Split", "pattern": {"Regex": "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+"}, "behavior": "Isolated", "invert": false},
                    {"type": "Split", "pattern": {"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
                ]
            }
        });
        assert_eq!(detect_pretokenizer_type(&data).pretok_type, PretokType::DeepSeek);
    }

    #[test]
    fn test_detect_qwen35_single_stage_single_digits() {
        // Qwen3.5-style: one Split with [\p{L}\p{M}]+ and single \p{N} — must stay Qwen35
        let data = serde_json::json!({
            "pre_tokenizer": {
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
                    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
                ]
            }
        });
        assert_eq!(detect_pretokenizer_type(&data).pretok_type, PretokType::Qwen35);
    }

    #[test]
    fn test_decode_bytelevel_token_ascii() {
        // Simple ASCII characters should decode to themselves
        assert_eq!(decode_bytelevel_token("Hello"), b"Hello".to_vec());
        assert_eq!(decode_bytelevel_token("world"), b"world".to_vec());
    }

    #[test]
    fn test_decode_bytelevel_token_space() {
        // Space (byte 32) is encoded as U+0120 (Ġ)
        assert_eq!(decode_bytelevel_token("Ġ"), vec![32]);
        assert_eq!(decode_bytelevel_token("Ġhello"), vec![32, 104, 101, 108, 108, 111]);
    }

    #[test]
    fn test_decode_bytelevel_token_newline() {
        // Newline (byte 10) is encoded as U+010A (Ċ)
        assert_eq!(decode_bytelevel_token("Ċ"), vec![10]);
    }

    #[test]
    fn test_decode_bytelevel_token_tab() {
        // Tab (byte 9) is encoded as U+0109 (ĉ)
        assert_eq!(decode_bytelevel_token("ĉ"), vec![9]);
    }

    #[test]
    fn test_decode_bytelevel_token_punctuation() {
        // Punctuation in printable ASCII range should decode directly
        assert_eq!(decode_bytelevel_token(","), vec![44]);
        assert_eq!(decode_bytelevel_token("."), vec![46]);
        assert_eq!(decode_bytelevel_token("!"), vec![33]);
    }
}
