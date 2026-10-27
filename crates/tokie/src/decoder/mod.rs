//! Decoders for converting token IDs back to text.
//!
//! - [`Decoder`] — high-level decoder combining vocab lookup with text post-processing
//! - [`VocabDecoder`] — maps token IDs to raw byte sequences (flat buffer, O(1) lookup)
//! - [`DecoderType`] — text-level post-processing strategy (WordPiece `##` stripping, metaspace `▁` → space, etc.)

mod vocab;

pub use vocab::VocabDecoder;

use crate::encoder::EncoderType;
use crate::postprocessor::PostProcessor;
use crate::types::TokenId;

/// How far past the vocab table [`Decoder::set_added`] lays out added ids.
const MAX_ADDED_PAST_TABLE: usize = 1 << 20;

/// Text-level decoder type, mirroring [`EncoderType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum DecoderType {
    /// Direct byte concatenation. Token bytes are already correct text.
    /// Used by byte-level BPE (GPT-2, Llama, Mistral, Phi, Qwen).
    #[default]
    ByteLevel = 0,

    /// WordPiece decoding (BERT, MiniLM, BGE, GTE).
    /// Strips `##` continuation prefixes and joins tokens with spaces.
    WordPiece = 1,

    /// Metaspace decoding (T5, XLM-R, Gemma, SentencePiece models).
    /// Replaces `▁` (U+2581) with spaces and strips the leading space.
    Metaspace = 2,
}

impl DecoderType {
    /// Infer the decoder type from the encoder type.
    pub fn from_encoder_type(encoder_type: EncoderType) -> Self {
        match encoder_type {
            EncoderType::WordPiece => DecoderType::WordPiece,
            EncoderType::SentencePiece | EncoderType::Unigram => DecoderType::Metaspace,
            _ => DecoderType::ByteLevel,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::ByteLevel),
            1 => Some(Self::WordPiece),
            2 => Some(Self::Metaspace),
            _ => None,
        }
    }
}

/// High-level decoder wrapping [`VocabDecoder`] (byte lookup) and [`DecoderType`] (text post-processing).
///
/// This is the primary decoder type used by [`Tokenizer`]. It provides:
/// - `decode()` — token IDs → String with text-level post-processing
/// - `decode_bytes()` — token IDs → raw bytes (no post-processing)
/// - `token_to_bytes()` — single token → byte slice
///
/// [`Tokenizer`]: crate::Tokenizer
#[derive(Clone)]
pub struct Decoder {
    vocab: VocabDecoder,
    decoder_type: DecoderType,
    /// Bytes of the added tokens past the vocab table, for ids
    /// `vocab_size()..` (empty where no token has the id). Set by
    /// [`Self::set_added`]; not serialized (the added tokens are).
    added: Vec<Box<[u8]>>,
}

impl Decoder {
    /// Create a new decoder from token byte sequences.
    ///
    /// Decoder type defaults to [`DecoderType::ByteLevel`].
    pub fn new(token_bytes: Vec<Vec<u8>>) -> Self {
        Self {
            vocab: VocabDecoder::new(token_bytes),
            decoder_type: DecoderType::ByteLevel,
            added: Vec::new(),
        }
    }

    /// Create a new decoder, inferring the decoder type from the encoder type.
    pub fn for_encoder(token_bytes: Vec<Vec<u8>>, encoder_type: EncoderType) -> Self {
        Self {
            vocab: VocabDecoder::new(token_bytes),
            decoder_type: DecoderType::from_encoder_type(encoder_type),
            added: Vec::new(),
        }
    }

    /// Create a decoder with a specific decoder type.
    pub fn with_type(vocab: VocabDecoder, decoder_type: DecoderType) -> Self {
        Self { vocab, decoder_type, added: Vec::new() }
    }

    /// Get the decoder type.
    pub fn decoder_type(&self) -> DecoderType {
        self.decoder_type
    }

    /// Get a reference to the underlying vocab decoder.
    pub fn vocab(&self) -> &VocabDecoder {
        &self.vocab
    }

    /// Consume self and return the inner VocabDecoder.
    pub fn into_vocab(self) -> VocabDecoder {
        self.vocab
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.vocab.vocab_size()
    }

    /// Ids past the vocab table that have an added token: `vocab_size()..`
    /// up to this.
    pub fn len_with_added(&self) -> usize {
        self.vocab.vocab_size() + self.added.len()
    }

    /// Give the added tokens whose ids lie past the vocab table (Llama 3
    /// keeps its special tokens only in the added vocabulary) their bytes,
    /// so they decode instead of indexing off the table.
    ///
    /// A special token whose table entry disagrees with its content takes
    /// the content, as HF's added vocabulary wins over the model's: tables
    /// built before they were laid out by id had every token after an id
    /// hole shifted by one (gpt-4 decoded `<|endoftext|>` as
    /// `<|fim_prefix|>`). Other added tokens keep the table entry, which
    /// holds decoded bytes: a byte-fallback `<0x0A>` or byte-level `ĠĠ`
    /// spells its content differently from the bytes it stands for.
    pub fn set_added<'a>(&mut self, tokens: impl IntoIterator<Item = (TokenId, &'a [u8], bool)>) {
        let base = self.vocab.vocab_size();
        let mut fixes: Vec<(usize, &[u8])> = Vec::new();
        for (id, bytes, special) in tokens {
            let id = id as usize;
            let Some(i) = id.checked_sub(base) else {
                if special && self.vocab.token_to_bytes(id as TokenId) != bytes {
                    fixes.push((id, bytes));
                }
                continue;
            };
            // A declared id far past the table is ignored rather than
            // allocating up to it.
            if i >= MAX_ADDED_PAST_TABLE {
                continue;
            }
            if self.added.len() <= i {
                self.added.resize(i + 1, Box::default());
            }
            self.added[i] = bytes.into();
        }
        if !fixes.is_empty() {
            let mut table = self.vocab.token_bytes();
            for (id, bytes) in fixes {
                table[id] = bytes.to_vec();
            }
            self.vocab = VocabDecoder::new(table);
        }
    }

    /// Get the byte sequence for a token (empty for an id no token has).
    #[inline]
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        let base = self.vocab.vocab_size();
        match (token as usize).checked_sub(base) {
            None => self.vocab.token_to_bytes(token),
            Some(i) => self.added.get(i).map_or(&[], |b| &**b),
        }
    }

    /// Get the length of a token in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.token_to_bytes(token).len()
    }

    /// Whether every id is in the vocab table (the fast decode paths).
    #[inline]
    fn in_table(&self, tokens: &[TokenId]) -> bool {
        let base = self.vocab.vocab_size();
        tokens.iter().all(|&t| (t as usize) < base)
    }

    /// Decode token IDs back to raw bytes (no text post-processing).
    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        if self.in_table(tokens) {
            return self.vocab.decode(tokens);
        }
        tokens.iter().flat_map(|&t| self.token_to_bytes(t)).copied().collect()
    }

    /// Decode token IDs to a string with text-level post-processing.
    ///
    /// Behavior depends on [`DecoderType`]:
    /// - **ByteLevel**: Direct byte concatenation (already correct)
    /// - **WordPiece**: Strips `##` prefixes, joins with spaces, skips special tokens
    /// - **Metaspace**: Replaces `▁` with spaces, strips leading space, skips special tokens
    ///
    /// Returns `None` if the result is not valid UTF-8.
    pub fn decode(&self, tokens: &[TokenId], post_processor: &PostProcessor) -> Option<String> {
        match self.decoder_type {
            DecoderType::ByteLevel if self.in_table(tokens) => self.vocab.decode_to_string(tokens),
            DecoderType::ByteLevel => String::from_utf8(self.decode_bytes(tokens)).ok(),
            DecoderType::WordPiece => decode_wordpiece(tokens, self, post_processor),
            DecoderType::Metaspace => decode_metaspace(tokens, self, post_processor),
        }
    }

    /// Decode token IDs to a UTF-8 string (raw, no text post-processing).
    ///
    /// Returns `None` if the decoded bytes are not valid UTF-8.
    pub fn decode_to_string(&self, tokens: &[TokenId]) -> Option<String> {
        self.vocab.decode_to_string(tokens)
    }

    // --- Delegated methods for serialization compatibility ---

    /// Create a decoder from pre-built parts (used for deserialization).
    pub fn from_parts(data: Vec<u8>, offsets: Vec<u32>, decoder_type: DecoderType) -> Self {
        Self {
            vocab: VocabDecoder::from_parts(data, offsets),
            decoder_type,
            added: Vec::new(),
        }
    }

    /// Get references to the internal data and offsets.
    pub fn as_parts(&self) -> (&[u8], &[u32]) {
        self.vocab.as_parts()
    }

    /// Get token bytes as a Vec for compatibility.
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.vocab.token_bytes()
    }
}

fn decode_wordpiece(
    tokens: &[TokenId],
    vocab: &Decoder,
    post_processor: &PostProcessor,
) -> Option<String> {
    let mut result = String::new();
    for &id in tokens {
        if post_processor.is_special_token(id) {
            continue;
        }
        let bytes = vocab.token_to_bytes(id);
        let token_str = std::str::from_utf8(bytes).ok()?;
        if let Some(stripped) = token_str.strip_prefix("##") {
            result.push_str(stripped);
        } else {
            if !result.is_empty() {
                result.push(' ');
            }
            result.push_str(token_str);
        }
    }
    Some(result)
}

fn decode_metaspace(
    tokens: &[TokenId],
    vocab: &Decoder,
    post_processor: &PostProcessor,
) -> Option<String> {
    let mut result = String::new();
    for &id in tokens {
        if post_processor.is_special_token(id) {
            continue;
        }
        let bytes = vocab.token_to_bytes(id);
        let token_str = std::str::from_utf8(bytes).ok()?;
        result.push_str(&token_str.replace('\u{2581}', " "));
    }
    if result.starts_with(' ') {
        result.remove(0);
    }
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_added_keeps_byte_tokens_and_repairs_shifted_specials() {
        let mut table: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        table.push(b"<|wrong|>".to_vec()); // 256: a special shifted by a hole
        let mut d = Decoder::new(table);
        d.set_added([
            (10, &b"<0x0A>"[..], false), // byte fallback: the table holds the byte
            (256, &b"<|endoftext|>"[..], true),
            (300, &b"<|past|>"[..], true),
        ]);
        assert_eq!(d.token_to_bytes(10), b"\n");
        assert_eq!(d.token_to_bytes(256), b"<|endoftext|>");
        assert_eq!(d.token_to_bytes(300), b"<|past|>");
        assert_eq!(d.token_to_bytes(299), b"");
        assert_eq!(d.decode_bytes(&[10, 256, 300]), b"\n<|endoftext|><|past|>");
    }
}
