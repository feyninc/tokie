//! High-level Tokenizer that combines pre-tokenization with BPE encoding.

use std::borrow::Cow;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::path::Path;
use std::sync::OnceLock;
use std::thread;

use foldhash::HashMap as FoldHashMap;

use chunk::chunk;

use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};

use crate::encoder::{Encoder, EncoderIter, EncoderType, PretokenCache};

/// Minimum bytes a thread's work chunk must contain before it pays for a
/// per-thread `PretokenCache` (4 MiB table; zeroing it costs ~100µs).
const PRETOKEN_CACHE_MIN_BYTES: usize = 256 * 1024;
use crate::decoder::{Decoder, DecoderType};
use crate::hf::{self, JsonLoadError};
use crate::normalizer::Normalizer;
use crate::padding::{Encoding, PaddingParams, TruncationParams, pad_batch, pad_encoding, truncate_ids, truncate_pair};
use crate::postprocessor::PostProcessor;
use crate::pretok::{PretokType, Pretokenizer};
use crate::types::TokenId;

/// Backward-compatible alias for [`Encoding`].
pub type EncodingPair = Encoding;

/// Cached number of available CPU cores.
fn num_cpus() -> usize {
    static CPUS: OnceLock<usize> = OnceLock::new();
    *CPUS.get_or_init(|| {
        thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1)
    })
}

/// High-level tokenizer combining pre-tokenization, encoding, and decoding.
///
/// # Example
/// ```ignore
/// use tokie::Tokenizer;
///
/// let tokenizer = Tokenizer::from_json("tokenizer.json")?;
/// let enc = tokenizer.encode("Hello, world!", false);
/// let text = tokenizer.decode(&enc.ids);
/// ```
pub struct Tokenizer {
    encoder: Encoder,
    decoder: Decoder,
    pretokenizer: Option<Pretokenizer>,
    pretokenizer_type: PretokType,
    normalizer: Normalizer,
    post_processor: PostProcessor,
    /// Persisted in .tkz format.
    pad_token_id: Option<TokenId>,
    /// Runtime config, not serialized.
    padding: Option<PaddingParams>,
    /// Runtime config, not serialized.
    truncation: Option<TruncationParams>,
    reverse_vocab: OnceLock<FoldHashMap<String, TokenId>>,
    /// DAAC matcher for added tokens (special and non-special).
    /// HF scans for these BEFORE pretokenization, splitting text at their boundaries.
    added_tokens_matcher: Option<DoubleArrayAhoCorasick>,
    /// Special token metadata: maps token string -> token ID.
    /// Populated from the `added_tokens` array in tokenizer.json where `special: true`.
    special_tokens: Vec<(String, TokenId)>,
}

impl Tokenizer {
    pub fn new(
        encoder: Encoder,
        decoder: Decoder,
        pretokenizer_type: PretokType,
        normalizer: Normalizer,
        post_processor: PostProcessor,
    ) -> Self {
        let pretokenizer = pretokenizer_type.to_pretokenizer();
        Self {
            encoder,
            decoder,
            pretokenizer,
            pretokenizer_type,
            normalizer,
            post_processor,
            pad_token_id: None,
            padding: None,
            truncation: None,
            reverse_vocab: OnceLock::new(),
            added_tokens_matcher: None,
            special_tokens: Vec::new(),
        }
    }

    /// Set added tokens matcher. These are matched BEFORE pretokenization, like HuggingFace does.
    pub fn set_added_tokens(&mut self, tokens: &[(TokenId, Vec<u8>)]) {
        if tokens.is_empty() {
            return;
        }
        let mut trie = Trie::new();
        for (id, bytes) in tokens {
            if !bytes.is_empty() {
                trie.add(bytes, *id);
            }
        }
        trie.build(MatchKind::LeftmostLongest);
        self.added_tokens_matcher = Some(trie.compile());
    }

    /// Set special token metadata (token string -> ID mapping).
    pub fn set_special_tokens(&mut self, tokens: Vec<(String, TokenId)>) {
        self.special_tokens = tokens;
    }

    /// Get special token metadata as (token_string, token_id) pairs.
    pub fn special_tokens(&self) -> &[(String, TokenId)] {
        &self.special_tokens
    }

    pub fn pretokenizer_type(&self) -> PretokType { self.pretokenizer_type }
    pub fn normalizer(&self) -> &Normalizer { &self.normalizer }
    pub fn post_processor(&self) -> &PostProcessor { &self.post_processor }
    pub fn encoder_type(&self) -> EncoderType { self.encoder.encoder_type() }
    pub fn decoder_type(&self) -> DecoderType { self.decoder.decoder_type() }
    pub fn encoder(&self) -> &Encoder { &self.encoder }
    pub fn decoder(&self) -> &Decoder { &self.decoder }
    pub fn pretokenizer(&self) -> Option<&Pretokenizer> { self.pretokenizer.as_ref() }
    pub fn set_pretokenizer(&mut self, pretok: Option<Pretokenizer>) { self.pretokenizer = pretok; }
    pub fn vocab_size(&self) -> usize { self.decoder.vocab_size() }
    pub fn pad_token_id(&self) -> Option<TokenId> { self.pad_token_id }
    pub fn padding(&self) -> Option<&PaddingParams> { self.padding.as_ref() }
    pub fn truncation(&self) -> Option<&TruncationParams> { self.truncation.as_ref() }

    /// Number of special tokens added for a single sequence.
    pub fn num_special_tokens_to_add(&self, is_pair: bool) -> usize {
        if is_pair {
            self.post_processor.num_special_tokens_pair()
        } else {
            self.post_processor.num_special_tokens_single()
        }
    }

    /// Minimum text size (in bytes) to trigger chunked parallel encoding.
    const PARALLEL_CHUNK_THRESHOLD: usize = 10_000;

    // --- Loading ---

    /// Load from a HuggingFace tokenizer.json file.
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self, JsonLoadError> {
        hf::from_json(path)
    }

    /// Load from a HuggingFace tokenizer.json with a specific encoder type.
    pub fn from_json_with_encoder(
        path: impl AsRef<Path>,
        encoder_type: EncoderType,
    ) -> Result<Self, JsonLoadError> {
        hf::from_json_with_encoder(path, encoder_type)
    }

    // --- Configuration ---

    pub fn enable_padding(&mut self, params: PaddingParams) -> &mut Self {
        self.padding = Some(params);
        self
    }

    pub fn enable_truncation(&mut self, params: TruncationParams) -> &mut Self {
        self.truncation = Some(params);
        self
    }

    pub fn no_padding(&mut self) -> &mut Self {
        self.padding = None;
        self
    }

    pub fn no_truncation(&mut self) -> &mut Self {
        self.truncation = None;
        self
    }

    pub fn set_pad_token_id(&mut self, id: TokenId) -> &mut Self {
        self.pad_token_id = Some(id);
        self
    }

    // --- Vocabulary access ---

    /// Get the token string for a given token ID.
    /// Returns lossy UTF-8 for byte-level tokens that aren't valid UTF-8.
    pub fn id_to_token(&self, id: TokenId) -> Option<Cow<'_, str>> {
        if (id as usize) >= self.vocab_size() {
            return None;
        }
        Some(String::from_utf8_lossy(self.decoder.token_to_bytes(id)))
    }

    /// Look up a token string and return its token ID (O(1) after first call).
    pub fn token_to_id(&self, token: &str) -> Option<TokenId> {
        self.reverse_vocab().get(token).copied()
    }

    /// Get the full vocabulary as a map from token strings to token IDs.
    pub fn get_vocab(&self) -> std::collections::HashMap<String, TokenId> {
        self.reverse_vocab().iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    /// Get the byte sequence for a token.
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        self.decoder.token_to_bytes(token)
    }

    fn reverse_vocab(&self) -> &FoldHashMap<String, TokenId> {
        self.reverse_vocab.get_or_init(|| {
            let n = self.vocab_size();
            let mut map = FoldHashMap::with_capacity_and_hasher(n, Default::default());
            for id in 0..n {
                let bytes = self.decoder.token_to_bytes(id as TokenId);
                let s = match std::str::from_utf8(bytes) {
                    Ok(s) => s.to_owned(),
                    Err(_) => String::from_utf8_lossy(bytes).into_owned(),
                };
                map.insert(s, id as TokenId);
            }
            map
        })
    }

    // --- Encoding ---

    /// Encode text into an [`Encoding`] with token IDs, attention mask, and type IDs.
    ///
    /// # Example
    /// ```ignore
    /// let enc = tokenizer.encode("Hello, world!", true);
    /// println!("{:?}", enc.ids);
    /// ```
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Encoding {
        self.encode_inner(text, add_special_tokens, None)
    }

    /// Encode with an optional per-thread pretoken cache (batch hot path).
    fn encode_inner(
        &self,
        text: &str,
        add_special_tokens: bool,
        cache: Option<&mut PretokenCache>,
    ) -> Encoding {
        let mut tokens = self.encode_raw_ctx(text, cache);

        if let Some(ref trunc) = self.truncation {
            let special = if add_special_tokens {
                self.post_processor.num_special_tokens_single()
            } else {
                0
            };
            let max_content = trunc.max_length.saturating_sub(special);
            truncate_ids(&mut tokens, max_content, trunc.direction);
        }

        let ids = if add_special_tokens {
            self.post_processor.process(&tokens)
        } else {
            tokens
        };

        let mut encoding = Encoding::from_ids(ids);

        if let Some(ref pad) = self.padding {
            if let crate::padding::PaddingStrategy::Fixed(n) = pad.strategy {
                pad_encoding(&mut encoding, n, pad);
            }
        }

        encoding
    }

    /// Encode text with byte offsets for each token.
    ///
    /// Returns an [`Encoding`] with `offsets` populated — each entry is a `(start, end)`
    /// byte range in the (normalized) input text corresponding to that token.
    ///
    /// Special tokens (CLS, SEP, BOS) get offset `(0, 0)`.
    ///
    /// # Example
    /// ```ignore
    /// let enc = tokenizer.encode_with_offsets("Hello, world!", true);
    /// for (id, (start, end)) in enc.ids.iter().zip(&enc.offsets) {
    ///     println!("token {} -> bytes {}..{}", id, start, end);
    /// }
    /// ```
    pub fn encode_with_offsets(&self, text: &str, add_special_tokens: bool) -> Encoding {
        let (mut tokens, mut offsets) = self.encode_raw_with_offsets(text);

        if let Some(ref trunc) = self.truncation {
            let special = if add_special_tokens {
                self.post_processor.num_special_tokens_single()
            } else {
                0
            };
            let max_content = trunc.max_length.saturating_sub(special);
            if tokens.len() > max_content {
                match trunc.direction {
                    crate::padding::TruncationDirection::Right => {
                        tokens.truncate(max_content);
                        offsets.truncate(max_content);
                    }
                    crate::padding::TruncationDirection::Left => {
                        let start = tokens.len() - max_content;
                        tokens.drain(..start);
                        offsets.drain(..start);
                    }
                }
            }
        }

        let (ids, final_offsets) = if add_special_tokens {
            let processed = self.post_processor.process(&tokens);
            // Build offsets for the processed sequence (special tokens get (0,0))
            let mut new_offsets = Vec::with_capacity(processed.len());
            let mut content_idx = 0;
            for &id in &processed {
                if self.post_processor.is_special_token(id) {
                    new_offsets.push((0, 0));
                } else if content_idx < offsets.len() {
                    new_offsets.push(offsets[content_idx]);
                    content_idx += 1;
                } else {
                    new_offsets.push((0, 0));
                }
            }
            (processed, new_offsets)
        } else {
            (tokens, offsets)
        };

        let mut encoding = Encoding::from_ids_with_offsets(ids, final_offsets);

        if let Some(ref pad) = self.padding {
            if let crate::padding::PaddingStrategy::Fixed(n) = pad.strategy {
                pad_encoding(&mut encoding, n, pad);
            }
        }

        encoding
    }

    /// Encode a pair of texts (e.g. for cross-encoder models).
    ///
    /// # Example
    /// ```ignore
    /// let enc = tokenizer.encode_pair("What is Berlin?", "Berlin is the capital.", true);
    /// ```
    pub fn encode_pair(&self, text_a: &str, text_b: &str, add_special_tokens: bool) -> Encoding {
        let mut tokens_a = self.encode_raw(text_a);
        let mut tokens_b = self.encode_raw(text_b);

        if let Some(ref trunc) = self.truncation {
            let special = if add_special_tokens {
                self.post_processor.num_special_tokens_pair()
            } else {
                0
            };
            let max_content = trunc.max_length.saturating_sub(special);
            truncate_pair(&mut tokens_a, &mut tokens_b, max_content, trunc.strategy, trunc.direction);
        }

        let (ids, type_ids) = if add_special_tokens {
            self.post_processor.process_pair(&tokens_a, &tokens_b)
        } else {
            let mut ids = Vec::with_capacity(tokens_a.len() + tokens_b.len());
            ids.extend_from_slice(&tokens_a);
            ids.extend_from_slice(&tokens_b);
            let mut type_ids = vec![0u8; tokens_a.len()];
            type_ids.resize(tokens_a.len() + tokens_b.len(), 1u8);
            (ids, type_ids)
        };

        Encoding::from_pair(ids, type_ids)
    }

    /// Core encoding path: normalize + pretokenize + encode. No special tokens.
    fn encode_raw(&self, text: &str) -> Vec<TokenId> {
        self.encode_raw_ctx(text, None)
    }

    fn encode_raw_ctx(&self, text: &str, cache: Option<&mut PretokenCache>) -> Vec<TokenId> {
        // If there are non-special added tokens, split the text at their boundaries first.
        // HuggingFace scans for added tokens BEFORE pretokenization.
        if let Some(ref matcher) = self.added_tokens_matcher {
            return self.encode_with_added_tokens(text, matcher, cache);
        }

        self.encode_raw_inner(text, cache)
    }

    /// Encode text after splitting at added token boundaries.
    fn encode_with_added_tokens(
        &self,
        text: &str,
        matcher: &DoubleArrayAhoCorasick,
        mut cache: Option<&mut PretokenCache>,
    ) -> Vec<TokenId> {
        let bytes = text.as_bytes();
        let mut result = Vec::new();
        let mut pos = 0;

        for m in matcher.find_iter(bytes) {
            // Encode text before this added token
            if m.start > pos {
                let segment = &text[pos..m.start];
                result.extend(self.encode_raw_inner(segment, cache.as_deref_mut()));
            }
            // Insert the added token directly
            result.push(m.pattern_id);
            pos = m.end;
        }

        // Encode remaining text after last added token
        if pos < text.len() {
            let segment = &text[pos..];
            result.extend(self.encode_raw_inner(segment, cache));
        }

        result
    }

    /// Inner encoding without added token splitting.
    fn encode_raw_inner(&self, text: &str, cache: Option<&mut PretokenCache>) -> Vec<TokenId> {
        // For models without pretokenizer (SentencePiece, Unigram), normalize the full
        // text first and pass directly to the encoder. The encoder handles its own
        // chunking at safe boundaries (metaspace). We must NOT use encode_parallel here
        // because it splits raw text at spaces before normalization, which breaks:
        // - Whitespace collapsing in SentencePiece normalizer (T5, XLM-R)
        // - Metaspace sequence merging (Voyage-code-2, Voyage-law-2)
        if self.pretokenizer.is_none() {
            let normalized = self.normalizer.normalize(text);
            return self.encoder.encode(normalized.as_ref().as_bytes());
        }

        if text.len() >= Self::PARALLEL_CHUNK_THRESHOLD {
            self.encode_parallel(text)
        } else {
            let normalized = self.normalizer.normalize(text);
            self.encode_sequential(normalized.as_ref(), cache)
        }
    }

    /// Core encoding path with byte offset tracking.
    /// Returns (token_ids, offsets) where offsets are byte ranges in the normalized text.
    fn encode_raw_with_offsets(&self, text: &str) -> (Vec<TokenId>, Vec<(usize, usize)>) {
        let normalized = self.normalizer.normalize(text);
        let normalized_ref = normalized.as_ref();

        match &self.pretokenizer {
            Some(pretok) => {
                let base_ptr = normalized_ref.as_ptr() as usize;
                // Collect pieces with their byte start positions
                let pieces: Vec<(&str, usize)> = pretok.split(normalized_ref)
                    .map(|piece| {
                        let start = piece.as_ptr() as usize - base_ptr;
                        (piece, start)
                    })
                    .collect();

                let cpus = num_cpus();
                if pieces.len() > cpus * 2 && normalized_ref.len() >= Self::PARALLEL_CHUNK_THRESHOLD {
                    // Parallel path: distribute pieces across threads
                    let chunk_size = (pieces.len() + cpus - 1) / cpus;
                    let encoder = &self.encoder;
                    let decoder = &self.decoder;

                    let results: Vec<(Vec<TokenId>, Vec<(usize, usize)>)> = thread::scope(|s| {
                        pieces.chunks(chunk_size)
                            .map(|chunk| {
                                s.spawn(move || {
                                    let mut tokens = Vec::new();
                                    let mut offsets = Vec::new();
                                    for &(piece, piece_start) in chunk {
                                        let toks = encoder.encode(piece.as_bytes());
                                        let mut pos = piece_start;
                                        for &token_id in &toks {
                                            let len = decoder.token_len(token_id);
                                            offsets.push((pos, pos + len));
                                            pos += len;
                                        }
                                        tokens.extend(toks);
                                    }
                                    (tokens, offsets)
                                })
                            })
                            .collect::<Vec<_>>()
                            .into_iter()
                            .map(|h| h.join().unwrap())
                            .collect()
                    });

                    let total: usize = results.iter().map(|(t, _)| t.len()).sum();
                    let mut all_tokens = Vec::with_capacity(total);
                    let mut all_offsets = Vec::with_capacity(total);
                    for (t, o) in results {
                        all_tokens.extend(t);
                        all_offsets.extend(o);
                    }
                    (all_tokens, all_offsets)
                } else {
                    // Sequential path
                    let mut all_tokens = Vec::new();
                    let mut all_offsets = Vec::new();
                    for (piece, piece_start) in pieces {
                        let tokens = self.encoder.encode(piece.as_bytes());
                        let mut pos = piece_start;
                        for &token_id in &tokens {
                            let len = self.decoder.token_len(token_id);
                            all_offsets.push((pos, pos + len));
                            pos += len;
                        }
                        all_tokens.extend(tokens);
                    }
                    (all_tokens, all_offsets)
                }
            }
            None => {
                let tokens = self.encoder.encode(normalized_ref.as_bytes());
                let mut pos = 0;
                let mut all_offsets = Vec::with_capacity(tokens.len());
                for &token_id in &tokens {
                    let len = self.decoder.token_len(token_id);
                    all_offsets.push((pos, pos + len));
                    pos += len;
                }
                (tokens, all_offsets)
            }
        }
    }

    #[inline]
    fn encode_sequential(&self, text: &str, mut cache: Option<&mut PretokenCache>) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(text.len() / 3);
        for piece in self.pretokenizer.as_ref().unwrap().split(text) {
            self.encoder.encode_into(piece.as_bytes(), cache.as_deref_mut(), &mut out);
        }
        out
    }

    /// Split text into chunks at whitespace, encode each in parallel.
    fn encode_parallel(&self, text: &str) -> Vec<TokenId> {
        let bytes = text.as_bytes();
        let cpus = num_cpus();
        let target_size = bytes.len() / cpus;

        let chunks: Vec<&[u8]> = chunk(bytes)
            .size(target_size)
            .delimiters(b" ")
            .prefix()
            .collect();

        if chunks.len() <= 1 {
            let normalized = self.normalizer.normalize(text);
            return self.encode_sequential(normalized.as_ref(), None);
        }

        let encoder = &self.encoder;
        let normalizer = &self.normalizer;
        let pretok = self.pretokenizer.as_ref().unwrap();
        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            chunks
                .iter()
                .map(|chunk_bytes| {
                    s.spawn(move || {
                        // SAFETY: Input was valid UTF-8, split at ASCII whitespace.
                        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                        let normalized = normalizer.normalize(chunk_str);
                        let mut cache = (chunk_bytes.len() >= PRETOKEN_CACHE_MIN_BYTES)
                            .then(PretokenCache::new);
                        let mut out = Vec::with_capacity(chunk_bytes.len() / 3);
                        for piece in pretok.split(normalized.as_ref()) {
                            encoder.encode_into(piece.as_bytes(), cache.as_mut(), &mut out);
                        }
                        out
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        });

        let total: usize = results.iter().map(|v| v.len()).sum();
        let mut output = Vec::with_capacity(total);
        for chunk_tokens in results {
            output.extend(chunk_tokens);
        }
        output
    }

    /// Encode raw bytes directly (bypasses pretokenizer and normalizer).
    pub fn encode_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.encoder.encode(bytes)
    }

    /// Streaming iterator over encoded tokens.
    pub fn encode_iter<'a>(&'a self, text: &'a str) -> TokenizeIter<'a> {
        TokenizeIter::new(self, text)
    }

    /// Streaming iterator over encoded tokens from bytes (bypasses pretokenizer).
    pub fn encode_bytes_iter<'a>(&'a self, bytes: &'a [u8]) -> EncoderIter<'a> {
        self.encoder.encode_iter(bytes)
    }

    // --- Decoding ---

    /// Decode token IDs back to a string, applying text-level post-processing.
    ///
    /// Behavior depends on the [`DecoderType`]:
    /// - **WordPiece**: Strips `##` continuation prefixes, joins tokens with spaces,
    ///   and skips special tokens (CLS, SEP, etc.)
    /// - **Metaspace** (SentencePiece/Unigram): Replaces `▁` with spaces, strips leading space
    /// - **ByteLevel** (BPE): Direct byte concatenation (already correct)
    ///
    /// Returns `None` if the result is not valid UTF-8.
    pub fn decode(&self, tokens: &[TokenId]) -> Option<String> {
        self.decoder.decode(tokens, &self.post_processor)
    }

    /// Raw byte-level decode without text post-processing.
    pub fn decode_bytes(&self, tokens: &[TokenId]) -> Vec<u8> {
        self.decoder.decode_bytes(tokens)
    }

    /// Decode multiple token sequences in parallel.
    pub fn decode_batch(&self, sequences: &[&[TokenId]]) -> Vec<Option<String>> {
        let cpus = num_cpus();
        if sequences.len() <= cpus || cpus == 1 {
            return sequences.iter().map(|tokens| self.decode(tokens)).collect();
        }

        let chunk_size = (sequences.len() + cpus - 1) / cpus;
        thread::scope(|s| {
            sequences.chunks(chunk_size)
                .map(|chunk| s.spawn(|| {
                    chunk.iter().map(|tokens| self.decode(tokens)).collect::<Vec<_>>()
                }))
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    // --- Batch encoding ---

    /// Encode multiple texts in parallel, with optional padding.
    ///
    /// # Example
    /// ```ignore
    /// let encodings = tokenizer.encode_batch(&["Hello!", "World"], true);
    /// ```
    pub fn encode_batch(&self, texts: &[&str], add_special_tokens: bool) -> Vec<Encoding> {
        let cpus = num_cpus();

        let mut encodings: Vec<Encoding> = if texts.len() > cpus && cpus > 1 {
            let chunk_size = (texts.len() + cpus - 1) / cpus;
            thread::scope(|s| {
                texts.chunks(chunk_size)
                    .map(|text_chunk| {
                        s.spawn(|| {
                            let chunk_bytes: usize = text_chunk.iter().map(|t| t.len()).sum();
                            let mut cache = (chunk_bytes >= PRETOKEN_CACHE_MIN_BYTES)
                                .then(PretokenCache::new);
                            text_chunk.iter()
                                .map(|t| self.encode_inner(t, add_special_tokens, cache.as_mut()))
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .flat_map(|h| h.join().unwrap())
                    .collect()
            })
        } else {
            texts.iter().map(|t| self.encode(t, add_special_tokens)).collect()
        };

        if let Some(ref pad) = self.padding {
            pad_batch(&mut encodings, pad);
        }

        encodings
    }

    /// Count tokens for multiple texts in parallel.
    pub fn count_tokens_batch(&self, texts: &[&str]) -> Vec<usize> {
        let cpus = num_cpus();
        if texts.is_empty() || cpus == 1 || texts.len() <= cpus {
            return texts.iter().map(|t| self.count_tokens(t)).collect();
        }

        let chunk_size = (texts.len() + cpus - 1) / cpus;
        thread::scope(|s| {
            texts.chunks(chunk_size)
                .map(|text_chunk| {
                    s.spawn(|| {
                        let chunk_bytes: usize = text_chunk.iter().map(|t| t.len()).sum();
                        let mut cache = (chunk_bytes >= PRETOKEN_CACHE_MIN_BYTES)
                            .then(PretokenCache::new);
                        text_chunk.iter()
                            .map(|t| self.encode_raw_ctx(t, cache.as_mut()).len())
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    /// Count tokens without storing them (no special tokens).
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encode_raw(text).len()
    }

    /// Lazy token count with early termination for comparisons.
    ///
    /// # Example
    /// ```ignore
    /// if tokenizer.token_count(text) > 8192 {
    ///     println!("text exceeds context window");
    /// }
    /// ```
    pub fn token_count<'a>(&'a self, text: &'a str) -> TokenCount<'a> {
        TokenCount {
            iter: RefCell::new(Some(self.encoder.encode_iter(text.as_bytes()))),
        }
    }
}

/// Lazy token count that supports comparison with `usize`.
/// Each `TokenCount` can only be compared once (the iterator is consumed).
pub struct TokenCount<'a> {
    iter: RefCell<Option<EncoderIter<'a>>>,
}

impl PartialEq<usize> for TokenCount<'_> {
    fn eq(&self, other: &usize) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

impl PartialOrd<usize> for TokenCount<'_> {
    fn partial_cmp(&self, limit: &usize) -> Option<Ordering> {
        let iter = self.iter.borrow_mut().take()?;
        let count = iter.take(*limit + 1).count();
        Some(count.cmp(limit))
    }
}

/// Iterator over tokens from the high-level Tokenizer.
pub struct TokenizeIter<'a> {
    tokenizer: &'a Tokenizer,
    pretokens: Option<Box<dyn Iterator<Item = &'a str> + 'a>>,
    current_encoder_iter: Option<EncoderIter<'a>>,
    bytes_iter: Option<EncoderIter<'a>>,
}

impl<'a> TokenizeIter<'a> {
    fn new(tokenizer: &'a Tokenizer, text: &'a str) -> Self {
        if tokenizer.pretokenizer.is_some() {
            let pretokens = tokenizer.pretokenizer.as_ref().unwrap().split(text);
            Self {
                tokenizer,
                pretokens: Some(Box::new(pretokens)),
                current_encoder_iter: None,
                bytes_iter: None,
            }
        } else {
            Self {
                tokenizer,
                pretokens: None,
                current_encoder_iter: None,
                bytes_iter: Some(tokenizer.encoder.encode_iter(text.as_bytes())),
            }
        }
    }
}

impl<'a> Iterator for TokenizeIter<'a> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        if let Some(ref mut iter) = self.bytes_iter {
            return iter.next();
        }

        loop {
            if let Some(ref mut encoder_iter) = self.current_encoder_iter {
                if let Some(token) = encoder_iter.next() {
                    return Some(token);
                }
            }

            if let Some(ref mut pretokens) = self.pretokens {
                if let Some(piece) = pretokens.next() {
                    self.current_encoder_iter =
                        Some(self.tokenizer.encoder.encode_iter(piece.as_bytes()));
                    continue;
                }
            }

            return None;
        }
    }
}

impl std::iter::FusedIterator for TokenizeIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::BacktrackingBytePairEncoder;
    use crate::padding::{PaddingStrategy, PaddingDirection};

    fn make_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)];
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::None, Normalizer::None, PostProcessor::None)
    }

    fn make_pretok_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)];
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::Gpt2, Normalizer::None, PostProcessor::None)
    }

    fn make_bert_tokenizer() -> Tokenizer {
        let base_tokens: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let merges = vec![(b'a' as u32, b'b' as u32)];
        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = Decoder::new(token_bytes);
        Tokenizer::new(Encoder::Backtracking(encoder), decoder, PretokType::None, Normalizer::None, PostProcessor::bert(101, 102))
    }

    #[test]
    fn test_no_pretokenizer() {
        let tokenizer = make_tokenizer();
        let enc = tokenizer.encode("abc", false);
        assert_eq!(enc.ids.len(), 2);
    }

    #[test]
    fn test_encode_returns_encoding() {
        let tokenizer = make_tokenizer();
        let enc = tokenizer.encode("abc", false);
        assert_eq!(enc.ids.len(), enc.attention_mask.len());
        assert_eq!(enc.ids.len(), enc.type_ids.len());
        assert!(enc.attention_mask.iter().all(|&m| m == 1));
        assert!(enc.type_ids.iter().all(|&t| t == 0));
    }

    #[test]
    fn test_with_pretokenizer() {
        let tokenizer = make_pretok_tokenizer();
        let enc = tokenizer.encode("Hello world", false);
        assert!(!enc.ids.is_empty());
        let decoded = tokenizer.decode(&enc.ids).unwrap();
        assert_eq!(decoded, "Hello world");
    }

    #[test]
    fn test_count_tokens() {
        let tokenizer = make_pretok_tokenizer();
        let text = "Hello world";
        let count = tokenizer.count_tokens(text);
        let enc = tokenizer.encode(text, false);
        assert_eq!(count, enc.ids.len());
    }

    #[test]
    fn test_token_count_comparisons() {
        let tokenizer = make_pretok_tokenizer();
        let text = "Hello world test";
        let total = tokenizer.count_tokens(text);
        assert!(tokenizer.token_count(text) > total - 1);
        assert!(!(tokenizer.token_count(text) > total));
        assert!(tokenizer.token_count(text) < total + 1);
        assert!(!(tokenizer.token_count(text) < total));
        assert!(tokenizer.token_count(text) == total);
    }

    #[test]
    fn test_encode_iter() {
        let tokenizer = make_pretok_tokenizer();
        let text = "Hello world";
        let tokens: Vec<_> = tokenizer.encode_iter(text).collect();
        let expected = tokenizer.encode_raw(text);
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_decode_bytes() {
        let tokenizer = make_tokenizer();
        let text = b"abc";
        let tokens = tokenizer.encode_bytes(text);
        let decoded = tokenizer.decode_bytes(&tokens);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_batch_empty() {
        let tokenizer = make_tokenizer();
        let result = tokenizer.encode_batch(&[], false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_encode_batch_single() {
        let tokenizer = make_pretok_tokenizer();
        let single = tokenizer.encode("Hello world", false);
        let batch = tokenizer.encode_batch(&["Hello world"], false);
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0], single);
    }

    #[test]
    fn test_encode_batch_multiple() {
        let tokenizer = make_pretok_tokenizer();
        let texts = vec!["Hello world", "abc def", "test"];
        let batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch.len(), 3);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tokenizer.encode(text, false));
        }
    }

    #[test]
    fn test_encode_batch_preserves_order() {
        let tokenizer = make_pretok_tokenizer();
        let texts: Vec<&str> = (0..20).map(|i| match i % 4 {
            0 => "alpha",
            1 => "beta gamma",
            2 => "delta epsilon zeta",
            _ => "x",
        }).collect();
        let batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch.len(), texts.len());
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(batch[i], tokenizer.encode(text, false));
        }
    }

    #[test]
    fn test_encode_batch_with_special_tokens() {
        let tokenizer = make_pretok_tokenizer();
        let texts = vec!["Hello", "world"];
        let batch_with = tokenizer.encode_batch(&texts, true);
        let batch_without = tokenizer.encode_batch(&texts, false);
        assert_eq!(batch_with, batch_without);
    }

    #[test]
    fn test_count_tokens_batch() {
        let tokenizer = make_pretok_tokenizer();
        let texts = vec!["Hello world", "abc", "test one two"];
        let counts = tokenizer.count_tokens_batch(&texts);
        assert_eq!(counts.len(), 3);
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(counts[i], tokenizer.count_tokens(text));
        }
    }

    #[test]
    fn test_count_tokens_batch_empty() {
        let tokenizer = make_tokenizer();
        let result = tokenizer.count_tokens_batch(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_vocab_access() {
        let tokenizer = make_tokenizer();
        // Token 97 = 'a', Token 98 = 'b', Token 256 = 'ab'
        assert_eq!(tokenizer.id_to_token(97).unwrap(), "a");
        assert_eq!(tokenizer.id_to_token(98).unwrap(), "b");
        assert_eq!(tokenizer.token_to_id("a"), Some(97));
        assert_eq!(tokenizer.token_to_id("b"), Some(98));
        assert!(tokenizer.id_to_token(999999).is_none());
        assert!(tokenizer.token_to_id("nonexistent_token_xyz").is_none());

        let vocab = tokenizer.get_vocab();
        // Vocab may have fewer entries than vocab_size due to lossy UTF-8 collisions
        assert!(vocab.len() <= tokenizer.vocab_size());
        assert!(vocab.len() > 0);
        assert_eq!(vocab["a"], 97);
    }

    // --- Truncation tests ---

    #[test]
    fn test_encode_with_truncation() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_truncation(TruncationParams {
            max_length: 3,
            ..Default::default()
        });
        let enc = tokenizer.encode("abcde", false);
        assert!(enc.ids.len() <= 3);
    }

    #[test]
    fn test_encode_truncation_preserves_special_tokens() {
        let mut tokenizer = make_bert_tokenizer();
        tokenizer.enable_truncation(TruncationParams {
            max_length: 4,
            ..Default::default()
        });
        let enc = tokenizer.encode("abcde", true);
        assert!(enc.ids.len() <= 4);
        assert_eq!(enc.ids[0], 101);
        assert_eq!(*enc.ids.last().unwrap(), 102);
    }

    #[test]
    fn test_encode_pair_with_truncation() {
        let mut tokenizer = make_bert_tokenizer();
        tokenizer.enable_truncation(TruncationParams {
            max_length: 7,
            ..Default::default()
        });
        let enc = tokenizer.encode_pair("abcde", "fghij", true);
        assert!(enc.ids.len() <= 7);
        assert_eq!(enc.ids[0], 101);
    }

    // --- Padding tests ---

    #[test]
    fn test_encode_batch_with_padding() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            pad_id: 0,
            ..Default::default()
        });
        let batch = tokenizer.encode_batch(&["ab", "abcde"], false);
        assert_eq!(batch[0].ids.len(), batch[1].ids.len());
        assert!(batch[0].attention_mask.iter().any(|&m| m == 0));
        assert!(batch[1].attention_mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn test_encode_with_fixed_padding() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(10),
            pad_id: 0,
            ..Default::default()
        });
        let enc = tokenizer.encode("ab", false);
        assert_eq!(enc.ids.len(), 10);
        assert_eq!(enc.attention_mask.iter().filter(|&&m| m == 0).count(), 10 - 1);
    }

    #[test]
    fn test_encode_batch_with_fixed_padding() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(8),
            pad_id: 0,
            ..Default::default()
        });
        let batch = tokenizer.encode_batch(&["ab", "cd", "e"], false);
        assert!(batch.iter().all(|e| e.ids.len() == 8));
    }

    #[test]
    fn test_encode_batch_left_padding() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_padding(PaddingParams {
            strategy: PaddingStrategy::Fixed(5),
            direction: PaddingDirection::Left,
            pad_id: 0,
            ..Default::default()
        });
        let enc = tokenizer.encode("ab", false);
        assert_eq!(enc.ids.len(), 5);
        assert_eq!(enc.attention_mask[0], 0);
        assert_eq!(*enc.attention_mask.last().unwrap(), 1);
    }

    #[test]
    fn test_no_padding_no_truncation_defaults() {
        let tokenizer = make_tokenizer();
        assert!(tokenizer.padding().is_none());
        assert!(tokenizer.truncation().is_none());
        assert!(tokenizer.pad_token_id().is_none());
    }

    #[test]
    fn test_config_methods() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_padding(PaddingParams::default());
        assert!(tokenizer.padding().is_some());
        tokenizer.no_padding();
        assert!(tokenizer.padding().is_none());

        tokenizer.enable_truncation(TruncationParams::default());
        assert!(tokenizer.truncation().is_some());
        tokenizer.no_truncation();
        assert!(tokenizer.truncation().is_none());

        tokenizer.set_pad_token_id(0);
        assert_eq!(tokenizer.pad_token_id(), Some(0));
    }

    // --- Offset tests ---

    #[test]
    fn test_encode_with_offsets_basic() {
        let tokenizer = make_tokenizer();
        let enc = tokenizer.encode_with_offsets("abc", false);
        // "abc" with merges a+b -> ab: tokens are [ab, c]
        assert_eq!(enc.ids.len(), 2);
        assert_eq!(enc.offsets.len(), 2);
        // "ab" covers bytes 0..2, "c" covers bytes 2..3
        assert_eq!(enc.offsets[0], (0, 2));
        assert_eq!(enc.offsets[1], (2, 3));
    }

    #[test]
    fn test_encode_with_offsets_single_byte() {
        let tokenizer = make_tokenizer();
        let enc = tokenizer.encode_with_offsets("x", false);
        assert_eq!(enc.ids.len(), 1);
        assert_eq!(enc.offsets, vec![(0, 1)]);
    }

    #[test]
    fn test_encode_with_offsets_contiguous() {
        // Verify offsets are contiguous (end of one = start of next)
        let tokenizer = make_pretok_tokenizer();
        let text = "Hello world";
        let enc = tokenizer.encode_with_offsets(text, false);
        assert_eq!(enc.ids.len(), enc.offsets.len());
        // Each offset should be valid byte range
        for &(start, end) in &enc.offsets {
            assert!(start <= end);
            assert!(end <= text.len());
        }
    }

    #[test]
    fn test_encode_with_offsets_roundtrip() {
        // Verify reconstructing text from offsets gives the original
        let tokenizer = make_tokenizer();
        let text = "abcde";
        let enc = tokenizer.encode_with_offsets(text, false);
        let mut reconstructed = String::new();
        for &(start, end) in &enc.offsets {
            reconstructed.push_str(&text[start..end]);
        }
        assert_eq!(reconstructed, text);
    }

    #[test]
    fn test_encode_with_offsets_special_tokens() {
        let tokenizer = make_bert_tokenizer();
        let enc = tokenizer.encode_with_offsets("ab", true);
        // Should have [CLS] ab [SEP]
        assert_eq!(enc.ids[0], 101); // CLS
        assert_eq!(*enc.ids.last().unwrap(), 102); // SEP
        // Special tokens get (0, 0) offsets
        assert_eq!(enc.offsets[0], (0, 0));
        assert_eq!(*enc.offsets.last().unwrap(), (0, 0));
    }

    #[test]
    fn test_encode_with_offsets_empty() {
        let tokenizer = make_tokenizer();
        let enc = tokenizer.encode_with_offsets("", false);
        assert!(enc.ids.is_empty());
        assert!(enc.offsets.is_empty());
    }

    #[test]
    fn test_encode_with_offsets_truncation() {
        let mut tokenizer = make_tokenizer();
        tokenizer.enable_truncation(TruncationParams {
            max_length: 2,
            ..Default::default()
        });
        let enc = tokenizer.encode_with_offsets("abcde", false);
        assert!(enc.ids.len() <= 2);
        assert_eq!(enc.ids.len(), enc.offsets.len());
    }
}
