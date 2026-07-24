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

/// Byte-length of a batch element: lets the work-stealing scaffold serve
/// both `&str` batches (Python-boundary path) and `&[u8]` document slices
/// (byte-source file path).
trait ByteLen {
    fn byte_len(&self) -> usize;
}
impl ByteLen for str {
    #[inline]
    fn byte_len(&self) -> usize {
        self.len()
    }
}
impl ByteLen for [u8] {
    #[inline]
    fn byte_len(&self) -> usize {
        self.len()
    }
}

/// Split `texts` into up to `parts` contiguous runs of roughly equal total
/// bytes. Count-based chunking lets one oversized document serialize a whole
/// thread; byte-balancing keeps workers evenly loaded.
fn byte_balanced_chunks<'a, 'b, T: ByteLen + ?Sized>(
    texts: &'b [&'a T],
    parts: usize,
) -> Vec<&'b [&'a T]> {
    let total: usize = texts.iter().map(|t| t.byte_len()).sum();
    let target = total / parts + 1;
    let mut chunks = Vec::with_capacity(parts);
    let mut start = 0;
    let mut acc = 0usize;
    for (i, t) in texts.iter().enumerate() {
        acc += t.byte_len();
        if acc >= target && chunks.len() + 1 < parts {
            chunks.push(&texts[start..=i]);
            start = i + 1;
            acc = 0;
        }
    }
    if start < texts.len() {
        chunks.push(&texts[start..]);
    }
    chunks
}
/// Split raw file buffers into document byte-slices on `separator`,
/// dropping empty documents (Python-`if d`-filter semantics). Documents
/// never span files; an empty separator means one document per file.
/// UTF-8 validation is deliberately NOT done here — the encode workers
/// validate per document in parallel.
fn split_file_docs<'a, B: AsRef<[u8]>>(buffers: &'a [B], separator: &[u8]) -> Vec<&'a [u8]> {
    let mut docs: Vec<&'a [u8]> = Vec::new();
    for buf in buffers {
        let buf = buf.as_ref();
        if separator.is_empty() {
            if !buf.is_empty() {
                docs.push(buf);
            }
            continue;
        }
        let mut start = 0usize;
        for pos in memchr::memmem::find_iter(buf, separator) {
            if pos > start {
                docs.push(&buf[start..pos]);
            }
            start = pos + separator.len();
        }
        if start < buf.len() {
            docs.push(&buf[start..]);
        }
    }
    docs
}

/// File contents for the byte-source bulk path: small files are read
/// into memory, large files are memory-mapped read-only. Mapping a
/// page-cache-warm corpus costs microseconds where `fs::read` pays an
/// allocation, a full copy, and a free — all inside the caller's timed
/// region. Standard mmap caveat: truncating the file while it is mapped
/// is undefined (SIGBUS), like every mmap-based reader.
enum FileBytes {
    Owned(Vec<u8>),
    #[cfg(unix)]
    Mapped(MmapFile),
}

impl AsRef<[u8]> for FileBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        match self {
            FileBytes::Owned(v) => v,
            #[cfg(unix)]
            FileBytes::Mapped(m) => m.as_slice(),
        }
    }
}

/// Minimal read-only `mmap` wrapper (unmapped on drop).
#[cfg(unix)]
struct MmapFile {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is immutable (PROT_READ, MAP_PRIVATE) and owned.
#[cfg(unix)]
unsafe impl Send for MmapFile {}
#[cfg(unix)]
unsafe impl Sync for MmapFile {}

#[cfg(unix)]
impl MmapFile {
    fn map(file: &std::fs::File, len: usize) -> std::io::Result<Self> {
        use std::os::unix::io::AsRawFd;
        // SAFETY: fd is valid for the duration of the call; a MAP_FAILED
        // result is checked before the pointer is used.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self { ptr, len })
    }

    #[inline]
    fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr..ptr+len is a live PROT_READ mapping owned by self.
        unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
    }
}

#[cfg(unix)]
impl Drop for MmapFile {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap and are unmapped once.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}

/// Touch every page of a fresh mapping from all cores. Serial demand
/// paging during the separator scan costs ~15ms on a warm 191MB corpus;
/// faulting in parallel first cuts that to ~2ms.
#[cfg(unix)]
fn prefault(slice: &[u8]) {
    const STRIDE: usize = 4096;
    let cpus = num_cpus();
    if cpus <= 1 || slice.len() < (4 << 20) {
        return;
    }
    let chunk = slice.len().div_ceil(cpus).next_multiple_of(STRIDE);
    thread::scope(|s| {
        for part in slice.chunks(chunk) {
            s.spawn(move || {
                let mut acc = 0u8;
                let mut i = 0;
                while i < part.len() {
                    acc ^= part[i];
                    i += STRIDE;
                }
                std::hint::black_box(acc);
            });
        }
    });
}

/// Open one corpus file as [`FileBytes`], mmap-ing above a small
/// threshold on unix.
fn read_file_bytes(path: &Path) -> std::io::Result<FileBytes> {
    #[cfg(unix)]
    {
        const MMAP_MIN: u64 = 1 << 20;
        let file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        if len >= MMAP_MIN {
            let map = MmapFile::map(&file, len as usize)?;
            prefault(map.as_slice());
            return Ok(FileBytes::Mapped(map));
        }
        drop(file);
    }
    Ok(FileBytes::Owned(std::fs::read(path)?))
}

use crate::decoder::{Decoder, DecoderType};
use crate::hf::{self, JsonLoadError};
use crate::normalizer::Normalizer;
use crate::padding::{Encoding, PaddingParams, TruncationParams, pad_batch, pad_encoding, truncate_ids, truncate_pair};
use crate::postprocessor::PostProcessor;
use crate::pretok::{PretokType, Pretokenizer};
use crate::types::TokenId;

/// Backward-compatible alias for [`Encoding`].
pub type EncodingPair = Encoding;

/// One added-token entry with its HuggingFace matching flags.
///
/// HF's `AddedVocabulary` honors per-token flags when scanning for added
/// tokens (`tokenizers/src/tokenizer/added_vocabulary.rs`):
/// - `lstrip`/`rstrip`: the match extends over adjacent whitespace, which is
///   consumed (e.g. roberta's `<mask>` has `lstrip` and swallows the space
///   before it).
/// - `normalized`: the token is matched *after* normalization, against the
///   normalizer-transformed pattern (voyage-2's `</s>` matches as `▁</s>`).
///   Non-normalized tokens are matched on the raw input first.
/// - `single_word`: the match is dropped when adjacent to a word character.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddedTokenSpec {
    pub id: TokenId,
    pub bytes: Vec<u8>,
    pub special: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub single_word: bool,
}

impl AddedTokenSpec {
    /// A plain added token with all flags off (pre-flag behavior).
    pub fn plain(id: TokenId, bytes: Vec<u8>) -> Self {
        Self { id, bytes, special: false, lstrip: false, rstrip: false, normalized: false, single_word: false }
    }
}

/// Split `text` at added-token matches, honoring per-token flags.
///
/// Port of HF's `AddedVocabulary::find_matches`: the DAAC yields
/// leftmost-longest matches; `single_word` drops matches adjacent to word
/// characters, `lstrip`/`rstrip` extend the match over neighboring
/// whitespace (which is consumed — the emitted token is just the id).
/// Returns `(Some(spec_index), byte_range)` for added tokens and
/// `(None, byte_range)` for the text in between, covering all of `text`.
fn split_on_added(
    text: &str,
    matcher: &DoubleArrayAhoCorasick,
    specs: &[AddedTokenSpec],
) -> Vec<(Option<usize>, std::ops::Range<usize>)> {
    let bytes = text.as_bytes();
    let mut splits = Vec::new();
    let mut pos = 0usize;

    for m in matcher.find_iter(bytes) {
        let spec = &specs[m.pattern_id as usize];
        let mut start = m.start;
        let mut stop = m.end;

        if spec.single_word {
            let start_ok = start == 0
                || !text[..start].chars().next_back().is_some_and(is_word_char);
            let stop_ok = stop == text.len()
                || !text[stop..].chars().next().is_some_and(is_word_char);
            if !(start_ok && stop_ok) {
                continue;
            }
        }
        if spec.lstrip {
            // Leftmost byte of the whitespace run ending at `start`, clamped
            // so a previous match's consumed whitespace isn't re-consumed.
            let ws_start = text[..start]
                .char_indices()
                .rev()
                .take_while(|(_, c)| c.is_whitespace())
                .last()
                .map_or(start, |(i, _)| i);
            start = ws_start.max(pos);
        }
        if spec.rstrip {
            let ws_len = text[stop..]
                .char_indices()
                .take_while(|(_, c)| c.is_whitespace())
                .last()
                .map_or(0, |(i, c)| i + c.len_utf8());
            stop += ws_len;
        }

        if pos < start {
            splits.push((None, pos..start));
        }
        splits.push((Some(m.pattern_id as usize), start..stop));
        pos = stop;
    }

    if pos < text.len() {
        splits.push((None, pos..text.len()));
    }
    splits
}

/// Approximation of the regex `\w` class HF uses for `single_word`
/// boundaries: alphanumerics plus underscore.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

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
    /// DAAC matcher for non-normalized added tokens, matched on the raw
    /// input BEFORE normalization/pretokenization, like HF's `split_trie`.
    /// Pattern values index into `added_tokens_raw`.
    raw_added_matcher: Option<DoubleArrayAhoCorasick>,
    /// DAAC matcher for `normalized: true` added tokens, matched on each
    /// normalized segment like HF's `split_normalized_trie`. Patterns are the
    /// normalizer-transformed token contents; values index `added_tokens_raw`.
    norm_added_matcher: Option<DoubleArrayAhoCorasick>,
    /// Special token metadata: maps token string -> token ID.
    /// Populated from the `added_tokens` array in tokenizer.json where `special: true`.
    special_tokens: Vec<(String, TokenId)>,
    /// Added-token list backing the matchers, kept for serialization
    /// (.tkz v13+ stores added tokens in the file).
    added_tokens_raw: Vec<AddedTokenSpec>,
    /// True when this tokenizer was loaded from a .tkz that carries the
    /// added-tokens section — loaders can skip the tokenizer.json fetch.
    added_tokens_serialized: bool,
    /// Process-unique id tagging pooled pretoken-cache contents (see
    /// `crate::pool`).
    cache_generation: u64,
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
            raw_added_matcher: None,
            norm_added_matcher: None,
            special_tokens: Vec::new(),
            added_tokens_raw: Vec::new(),
            added_tokens_serialized: false,
            cache_generation: crate::pool::next_generation(),
        }
    }

    /// Byte-balanced work-stealing scaffold for batch calls: split `texts`
    /// into fine-grained chunks, run one worker per CPU with a leased
    /// long-lived pretoken cache, workers claim chunks as they finish
    /// (fast P-cores keep working instead of idling on the slowest
    /// E-core's tail), and return per-chunk results in input order.
    fn steal_batches<'a, 'b, T, R, F>(&self, texts: &'b [&'a T], work: F) -> Vec<R>
    where
        T: ByteLen + ?Sized + Sync,
        R: Send,
        F: Fn(&'b [&'a T], &mut PretokenCache) -> R + Sync,
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let cpus = num_cpus();
        let chunks = byte_balanced_chunks(texts, cpus * 4);
        let next = AtomicUsize::new(0);
        let mut results: Vec<Option<R>> = Vec::new();
        results.resize_with(chunks.len(), || None);
        let generation = self.cache_generation;
        thread::scope(|s| {
            let handles: Vec<_> = (0..cpus.min(chunks.len()))
                .map(|_| {
                    let chunks = &chunks;
                    let next = &next;
                    let work = &work;
                    s.spawn(move || {
                        let mut lease = crate::pool::CacheLease::checkout(generation);
                        let mut out: Vec<(usize, R)> = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= chunks.len() {
                                break;
                            }
                            out.push((i, work(chunks[i], lease.cache())));
                        }
                        out
                    })
                })
                .collect();
            for h in handles {
                for (i, r) in h.join().unwrap() {
                    results[i] = Some(r);
                }
            }
        });
        results.into_iter().map(|r| r.unwrap()).collect()
    }

    /// Set added tokens. Non-normalized tokens are matched on the raw input
    /// before pretokenization; `normalized: true` tokens are matched on each
    /// normalized segment against their normalizer-transformed pattern, both
    /// like HuggingFace. Call this after the normalizer is in place — the
    /// normalized patterns are computed with `self.normalizer`.
    pub fn set_added_tokens(&mut self, tokens: &[AddedTokenSpec]) {
        if tokens.is_empty() {
            return;
        }
        self.added_tokens_raw = tokens.to_vec();
        let mut raw_trie = Trie::new();
        let mut norm_trie = Trie::new();
        let (mut raw_count, mut norm_count) = (0usize, 0usize);
        for (idx, tok) in tokens.iter().enumerate() {
            if tok.bytes.is_empty() {
                continue;
            }
            // Skip single-byte tokens that plain encoding already maps to
            // the same id — matching them in the DAAC would add overhead
            // with no benefit. Single-byte tokens that encode differently
            // (out-of-vocab remaps) must stay in the matcher.
            if tok.bytes.len() == 1 && self.encoder.encode(&tok.bytes) == [tok.id] {
                continue;
            }
            if tok.normalized {
                // HF builds the normalized trie from normalizer(content).
                // Non-UTF-8 contents can't be normalized; match them raw.
                match std::str::from_utf8(&tok.bytes) {
                    Ok(s) => {
                        let pattern = self.normalizer.normalize(s);
                        if !pattern.is_empty() {
                            norm_trie.add(pattern.as_ref().as_bytes(), idx as u32);
                            norm_count += 1;
                        }
                    }
                    Err(_) => {
                        raw_trie.add(&tok.bytes, idx as u32);
                        raw_count += 1;
                    }
                }
            } else {
                raw_trie.add(&tok.bytes, idx as u32);
                raw_count += 1;
            }
        }
        self.raw_added_matcher = (raw_count > 0).then(|| {
            raw_trie.build(MatchKind::LeftmostLongest);
            raw_trie.compile()
        });
        self.norm_added_matcher = (norm_count > 0).then(|| {
            norm_trie.build(MatchKind::LeftmostLongest);
            norm_trie.compile()
        });
    }

    /// The added-token list backing the matchers.
    pub fn added_tokens_raw(&self) -> &[AddedTokenSpec] {
        &self.added_tokens_raw
    }

    /// Whether this tokenizer came from a .tkz that stores added tokens (v13+).
    pub fn added_tokens_serialized(&self) -> bool {
        self.added_tokens_serialized
    }

    pub(crate) fn mark_added_tokens_serialized(&mut self) {
        self.added_tokens_serialized = true;
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

    /// Minimum text size (in bytes) to trigger chunked parallel encoding
    /// on standalone (non-batch) calls.
    ///
    /// Phase-1 measurement (profile_spawn_cost, M3): a `thread::scope`
    /// spawn+join of 8 workers costs ~100us, while the fused cache-first
    /// sequential loop runs ~300 MB/s — so parallel chunking only breaks
    /// even near `100us * 300MB/s * 8/7 ~ 34KB` even with warm worker
    /// caches. 64 KiB adds margin for spawn-cost variance; below it the
    /// old 10 KB threshold made 10-50KB documents ~2x SLOWER than the
    /// sequential loop (140 vs 267 MB/s on OWT).
    const PARALLEL_CHUNK_THRESHOLD: usize = 64 * 1024;

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

    /// Encode to bare token ids (truncation + special tokens applied, no
    /// Encoding struct, no attention/type-id buffers). The low-latency path
    /// for callers that only consume ids.
    pub fn encode_ids(&self, text: &str, add_special_tokens: bool) -> Vec<TokenId> {
        self.encode_ids_ctx(text, add_special_tokens, None)
    }

    /// [`Self::encode_ids`] with an optional per-thread pretoken cache
    /// (batch hot path).
    fn encode_ids_ctx(
        &self,
        text: &str,
        add_special_tokens: bool,
        cache: Option<&mut PretokenCache>,
    ) -> Vec<TokenId> {
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
        if add_special_tokens {
            self.post_processor.process(&tokens)
        } else {
            tokens
        }
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

    /// Cache-first: when no per-thread cache is supplied and the model is
    /// BPE-with-pretokenizer, check out a pooled, process-lived
    /// [`PretokenCache`], so repeated pieces resolve to a single table probe
    /// even across single-document calls.
    /// A caller-supplied cache marks a batch-worker context: the batch is
    /// already running one worker per CPU, so per-document parallel
    /// chunking would only oversubscribe (a `thread::scope` spawn+join
    /// costs ~100us on macOS — phase-1 profile_spawn_cost) and is
    /// disabled. Standalone calls check out a pooled lease and keep the
    /// parallel path for large documents.
    fn encode_raw_ctx(&self, text: &str, cache: Option<&mut PretokenCache>) -> Vec<TokenId> {
        let allow_parallel = cache.is_none();
        if cache.is_none() && self.encoder.as_backtracking().is_some() && self.pretokenizer.is_some() {
            let mut lease = crate::pool::CacheLease::checkout(self.cache_generation);
            return self.encode_raw_dispatch(text, Some(lease.cache()), allow_parallel);
        }
        self.encode_raw_dispatch(text, cache, allow_parallel)
    }

    fn encode_raw_dispatch(&self, text: &str, cache: Option<&mut PretokenCache>, allow_parallel: bool) -> Vec<TokenId> {
        // If there are added tokens, split the text at their boundaries first.
        // HuggingFace scans for added tokens BEFORE pretokenization.
        if self.raw_added_matcher.is_some() || self.norm_added_matcher.is_some() {
            return self.encode_with_added_tokens(text, cache, allow_parallel);
        }

        self.encode_raw_inner(text, cache, allow_parallel)
    }

    /// Encode text after splitting at added token boundaries.
    ///
    /// Mirrors HF's `AddedVocabulary::extract_and_normalize` two-stage split:
    /// 1. the raw input is split on non-normalized added tokens;
    /// 2. each remaining segment is normalized (position-aware for the
    ///    metaspace prepend) and split on the normalized-token patterns;
    ///    the leftover pieces are encoded without being normalized again.
    fn encode_with_added_tokens(
        &self,
        text: &str,
        mut cache: Option<&mut PretokenCache>,
        allow_parallel: bool,
    ) -> Vec<TokenId> {
        let mut result = Vec::new();

        let raw_splits = match &self.raw_added_matcher {
            Some(matcher) => split_on_added(text, matcher, &self.added_tokens_raw),
            None => vec![(None, 0..text.len())],
        };

        for (spec_idx, range) in raw_splits {
            if let Some(idx) = spec_idx {
                result.push(self.added_tokens_raw[idx].id);
                continue;
            }
            let segment = &text[range.clone()];
            if segment.is_empty() {
                continue;
            }
            // Only the segment at byte 0 of the original input counts as
            // "first" (HF Metaspace prepend_scheme=first checks the
            // original offset).
            let first_segment = range.start == 0;

            match &self.norm_added_matcher {
                None => {
                    result.extend(self.encode_segment(segment, first_segment, cache.as_deref_mut(), allow_parallel));
                }
                Some(matcher) => {
                    let normalized = self.normalizer.normalize_segment(segment, first_segment);
                    for (nidx, nrange) in
                        split_on_added(normalized.as_ref(), matcher, &self.added_tokens_raw)
                    {
                        if let Some(idx) = nidx {
                            result.push(self.added_tokens_raw[idx].id);
                            continue;
                        }
                        let piece = &normalized[nrange];
                        if !piece.is_empty() {
                            result.extend(self.encode_prenormalized(piece, cache.as_deref_mut(), allow_parallel));
                        }
                    }
                }
            }
        }

        result
    }

    /// Encode one raw segment of an added-token split, normalizing it with
    /// segment-position awareness.
    fn encode_segment(
        &self,
        segment: &str,
        first_segment: bool,
        cache: Option<&mut PretokenCache>,
        allow_parallel: bool,
    ) -> Vec<TokenId> {
        if self.pretokenizer.is_none() {
            let normalized = self.normalizer.normalize_segment(segment, first_segment);
            return self.encoder.encode(normalized.as_ref().as_bytes());
        }
        // Models with a pretokenizer never use position-aware metaspace
        // prepending, so the standard path (with its parallel branch for
        // large segments) is equivalent.
        self.encode_raw_inner(segment, cache, allow_parallel)
    }

    /// Encode an already-normalized piece (stage 2 of the added-token split).
    fn encode_prenormalized(&self, piece: &str, cache: Option<&mut PretokenCache>, allow_parallel: bool) -> Vec<TokenId> {
        if self.pretokenizer.is_none() {
            return self.encoder.encode(piece.as_bytes());
        }
        // Pretokenizer models pair with idempotent normalizers (None, NFC,
        // Bert clean-text), so re-normalizing in the standard path is a
        // no-op and keeps the parallel branch for large pieces.
        self.encode_raw_inner(piece, cache, allow_parallel)
    }

    /// Inner encoding without added token splitting.
    fn encode_raw_inner(&self, text: &str, cache: Option<&mut PretokenCache>, allow_parallel: bool) -> Vec<TokenId> {
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

        if allow_parallel && text.len() >= Self::PARALLEL_CHUNK_THRESHOLD {
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

    /// Sequential fused pretokenize+encode of one normalized text.
    ///
    /// For the Backtracking encoder both per-piece enum dispatches (the
    /// `PretokenizerIter` match and the `Encoder` match) are hoisted out of
    /// the loop: `for_each_piece` monomorphizes the walker per config and
    /// the consumer closure calls the concrete encoder directly. Phase-1
    /// profiling (profile_glue_costs, gpt2/OWT) put the two enum taxes at
    /// ~1 ns/piece of a ~16 ns/piece loop.
    #[inline]
    fn encode_sequential(&self, text: &str, cache: Option<&mut PretokenCache>) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(text.len() / 3);
        self.encode_sequential_into(text, cache, &mut out);
        out
    }

    /// [`Self::encode_sequential`] appending into a caller-owned buffer —
    /// the byte-source bulk path reuses one buffer per worker chunk so no
    /// per-document vector is ever allocated.
    #[inline]
    fn encode_sequential_into(
        &self,
        text: &str,
        mut cache: Option<&mut PretokenCache>,
        out: &mut Vec<TokenId>,
    ) {
        let pretok = self.pretokenizer.as_ref().unwrap();
        let db = text.as_bytes();
        if let Some(bt) = self.encoder.as_backtracking() {
            match cache {
                Some(c) => pretok.for_each_piece(text, |p| {
                    bt.encode_piece_into(db, p.as_bytes(), Some(&mut *c), out)
                }),
                None => pretok.for_each_piece(text, |p| {
                    bt.encode_piece_into(db, p.as_bytes(), None, out)
                }),
            }
        } else {
            for piece in pretok.split(text) {
                self.encoder.encode_piece_into(db, piece.as_bytes(), cache.as_deref_mut(), out);
            }
        }
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

        let normalizer = &self.normalizer;
        let generation = self.cache_generation;
        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            chunks
                .iter()
                .map(|chunk_bytes| {
                    s.spawn(move || {
                        // SAFETY: Input was valid UTF-8, split at ASCII whitespace.
                        let chunk_str = unsafe { std::str::from_utf8_unchecked(chunk_bytes) };
                        let normalized = normalizer.normalize(chunk_str);
                        // Pooled process-lived cache: warm across calls and
                        // free of the 2 MiB alloc+zero a fresh table costs
                        // (which the old code paid per chunk, and only for
                        // chunks over 256 KiB — smaller chunks ran fully
                        // uncached).
                        let mut lease = crate::pool::CacheLease::checkout(generation);
                        self.encode_sequential(normalized.as_ref(), Some(lease.cache()))
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

    /// Split `text` at added-token matches, honoring per-token flags.
    /// Exposed for tests; see [`split_on_added`].
    #[doc(hidden)]
    pub fn debug_split_added(&self, text: &str) -> Vec<(Option<TokenId>, std::ops::Range<usize>)> {
        match &self.raw_added_matcher {
            Some(m) => split_on_added(text, m, &self.added_tokens_raw)
                .into_iter()
                .map(|(idx, r)| (idx.map(|i| self.added_tokens_raw[i].id), r))
                .collect(),
            None => vec![(None, 0..text.len())],
        }
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
            self.steal_batches(texts, |chunk, cache| {
                chunk
                    .iter()
                    .map(|t| self.encode_inner(t, add_special_tokens, Some(cache)))
                    .collect::<Vec<_>>()
            })
            .into_iter()
            .flatten()
            .collect()
        } else {
            texts.iter().map(|t| self.encode(t, add_special_tokens)).collect()
        };

        if let Some(ref pad) = self.padding {
            pad_batch(&mut encodings, pad);
        }

        encodings
    }

    /// Encode multiple texts in parallel into one contiguous id buffer.
    ///
    /// Returns `(ids, lens)`: every document's token ids concatenated in
    /// order, and per-document id counts. This is the zero-materialization
    /// bulk contract — no per-document `Encoding` objects or vectors reach
    /// the caller, so bindings can hand the buffers over as flat arrays.
    /// Truncation and special tokens apply as in [`Self::encode_ids`];
    /// padding does not (bulk consumers reconstruct boundaries from
    /// `lens`).
    pub fn encode_batch_flat(
        &self,
        texts: &[&str],
        add_special_tokens: bool,
    ) -> (Vec<TokenId>, Vec<u64>) {
        let cpus = num_cpus();
        if texts.is_empty() {
            return (Vec::new(), Vec::new());
        }
        if texts.len() <= cpus || cpus == 1 {
            let total_bytes: usize = texts.iter().map(|t| t.len()).sum();
            let mut ids = Vec::with_capacity(total_bytes / 3);
            let mut lens = Vec::with_capacity(texts.len());
            for t in texts {
                let v = self.encode_ids(t, add_special_tokens);
                lens.push(v.len() as u64);
                ids.extend_from_slice(&v);
            }
            return (ids, lens);
        }

        let results: Vec<(Vec<TokenId>, Vec<u64>)> =
            self.steal_batches(texts, |chunk, cache| {
                let chunk_bytes: usize = chunk.iter().map(|t| t.len()).sum();
                let mut ids = Vec::with_capacity(chunk_bytes / 3);
                let mut lens = Vec::with_capacity(chunk.len());
                for t in chunk {
                    let v = self.encode_ids_ctx(t, add_special_tokens, Some(cache));
                    lens.push(v.len() as u64);
                    ids.extend_from_slice(&v);
                }
                (ids, lens)
            });

        let total_ids: usize = results.iter().map(|(i, _)| i.len()).sum();
        let mut ids = Vec::with_capacity(total_ids);
        let mut lens = Vec::with_capacity(texts.len());
        for (i, l) in results {
            ids.extend_from_slice(&i);
            lens.extend_from_slice(&l);
        }
        (ids, lens)
    }

    /// Encode corpus files in bulk into one contiguous id buffer.
    ///
    /// Reads each file's bytes in Rust, splits every file on the
    /// `separator` byte sequence (documents never span files; an empty
    /// separator treats each file as a single document), drops empty
    /// documents — matching the usual Python
    /// `[d for d in text.split(sep) if d]` pre-split — and encodes all
    /// documents with the parallel bulk pipeline. No text ever crosses a
    /// binding boundary, so this is the fastest way to tokenize corpora
    /// from disk.
    ///
    /// Each document is UTF-8-validated once; documents containing invalid
    /// UTF-8 fall back to lossy conversion (invalid sequences become
    /// U+FFFD) instead of failing, so arbitrary bytes are safe. Valid
    /// documents are borrowed straight from the read buffer — no copies.
    ///
    /// Returns `(ids, offsets)`: every document's token ids concatenated
    /// in order, plus document boundaries with `offsets.len() == ndocs + 1`
    /// — document `i` is `ids[offsets[i] as usize..offsets[i + 1] as usize]`.
    /// Truncation and special tokens apply as in [`Self::encode_batch_flat`];
    /// padding does not.
    pub fn encode_files_flat<P: AsRef<Path>>(
        &self,
        paths: &[P],
        separator: &[u8],
        add_special_tokens: bool,
    ) -> std::io::Result<(Vec<TokenId>, Vec<u64>)> {
        let profile = std::env::var_os("TOKIE_PROFILE_FILES").is_some();
        let t0 = std::time::Instant::now();
        let buffers: Vec<FileBytes> = paths
            .iter()
            .map(|p| read_file_bytes(p.as_ref()))
            .collect::<std::io::Result<_>>()?;
        if profile {
            eprintln!("[files] read : {:6.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let t0 = std::time::Instant::now();
        let docs = split_file_docs(&buffers, separator);
        if profile {
            eprintln!("[files] split: {:6.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        let t0 = std::time::Instant::now();
        let (ids, lens) = self.encode_docs_bytes_flat(&docs, add_special_tokens);
        if profile {
            eprintln!("[files] encode: {:6.1} ms", t0.elapsed().as_secs_f64() * 1e3);
        }
        // Unmapping/freeing the corpus buffers is off the hot path too.
        drop(docs);
        std::thread::spawn(move || drop(buffers));
        let mut offsets = Vec::with_capacity(lens.len() + 1);
        let mut acc = 0u64;
        offsets.push(0);
        for l in lens {
            acc += l;
            offsets.push(acc);
        }
        Ok((ids, offsets))
    }

    /// Bulk-encode document byte-slices into one contiguous id buffer.
    ///
    /// Worker chunks validate UTF-8 per document (in parallel — a serial
    /// prepass over a 191MB corpus costs ~60ms) and append ids straight
    /// into one buffer per chunk: on the fused fast path (pretokenizer
    /// model, no added tokens, no truncation, no special tokens) no
    /// per-document vector is allocated at all. Chunk buffers are then
    /// copied into the final flat buffer in parallel.
    fn encode_docs_bytes_flat(
        &self,
        docs: &[&[u8]],
        add_special_tokens: bool,
    ) -> (Vec<TokenId>, Vec<u64>) {
        if docs.is_empty() {
            return (Vec::new(), Vec::new());
        }
        // Mirrors the exact conditions under which `encode_ids_ctx` with a
        // caller-supplied cache reduces to normalize + `encode_sequential`
        // (see `encode_raw_dispatch` / `encode_raw_inner`).
        let fused = self.pretokenizer.is_some()
            && self.raw_added_matcher.is_none()
            && self.norm_added_matcher.is_none()
            && self.truncation.is_none()
            && !add_special_tokens;

        let profile = std::env::var_os("TOKIE_PROFILE_FILES").is_some();
        let t_workers = std::time::Instant::now();

        // Allocate and prefault the final id buffer concurrently with the
        // encode workers: zeroing ~180MB of fresh pages on first touch
        // costs ~8ms serial, which would otherwise land in the concat
        // phase. bytes/3 over-estimates every BPE corpus we ship (OWT/gpt2
        // is ~bytes/4.4); the rare under-estimate falls back to a plain
        // allocation below.
        let est: usize = docs.iter().map(|d| d.len()).sum::<usize>() / 3;
        let prealloc = std::thread::spawn(move || {
            let mut v: Vec<TokenId> = Vec::with_capacity(est);
            let spare = v.spare_capacity_mut();
            let mut i = 0;
            while i < spare.len() {
                spare[i] = std::mem::MaybeUninit::new(0);
                i += 1024;
            }
            v
        });

        let results: Vec<(Vec<TokenId>, Vec<u64>)> =
            self.steal_batches(docs, |chunk, cache| {
                let chunk_bytes: usize = chunk.iter().map(|d| d.len()).sum();
                let mut ids: Vec<TokenId> = Vec::with_capacity(chunk_bytes / 3);
                let mut lens: Vec<u64> = Vec::with_capacity(chunk.len());
                for d in chunk {
                    // One SIMD validation pass (`str::from_utf8` beats
                    // `from_utf8_lossy`'s chunk walker on valid input);
                    // documents with invalid bytes take the lossy path
                    // (U+FFFD) so arbitrary bytes can never panic.
                    let text: Cow<str> = match std::str::from_utf8(d) {
                        Ok(s) => Cow::Borrowed(s),
                        Err(_) => String::from_utf8_lossy(d),
                    };
                    let before = ids.len();
                    if fused {
                        let normalized = self.normalizer.normalize(&text);
                        self.encode_sequential_into(normalized.as_ref(), Some(cache), &mut ids);
                    } else {
                        let v = self.encode_ids_ctx(&text, add_special_tokens, Some(cache));
                        ids.extend_from_slice(&v);
                    }
                    lens.push((ids.len() - before) as u64);
                }
                (ids, lens)
            });

        if profile {
            eprintln!("[files] workers: {:6.1} ms", t_workers.elapsed().as_secs_f64() * 1e3);
        }
        let t_concat = std::time::Instant::now();
        let total_ids: usize = results.iter().map(|(i, _)| i.len()).sum();
        let mut lens = Vec::with_capacity(docs.len());
        for (_, l) in &results {
            lens.extend_from_slice(l);
        }

        // Concatenate chunk id buffers in parallel: a serial memcpy of a
        // large corpus' ids (~180MB for 191MB of OWT) costs ~15-20ms.
        let mut ids: Vec<TokenId> = match prealloc.join() {
            Ok(v) if v.capacity() >= total_ids => v,
            _ => Vec::with_capacity(total_ids),
        };
        {
            let mut spare = &mut ids.spare_capacity_mut()[..total_ids];
            let mut jobs: Vec<(&mut [std::mem::MaybeUninit<TokenId>], &[TokenId])> =
                Vec::with_capacity(results.len());
            for (chunk_ids, _) in &results {
                let (dst, rest) = spare.split_at_mut(chunk_ids.len());
                spare = rest;
                jobs.push((dst, chunk_ids));
            }
            let workers = num_cpus().min(jobs.len()).max(1);
            let mut per_worker: Vec<Vec<(&mut [std::mem::MaybeUninit<TokenId>], &[TokenId])>> =
                (0..workers).map(|_| Vec::new()).collect();
            // Chunks are byte-balanced, so round-robin keeps copies even.
            for (i, job) in jobs.into_iter().enumerate() {
                per_worker[i % workers].push(job);
            }
            thread::scope(|s| {
                for work in per_worker {
                    s.spawn(move || {
                        for (dst, src) in work {
                            // SAFETY: dst and src have equal length; the
                            // uninitialized destination is only written.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    src.as_ptr(),
                                    dst.as_mut_ptr().cast::<TokenId>(),
                                    src.len(),
                                );
                            }
                        }
                    });
                }
            });
        }
        // SAFETY: the jobs above covered ..total_ids exactly.
        unsafe { ids.set_len(total_ids) };
        // Freeing ~180MB of chunk buffers costs several ms; hand them to a
        // detached thread so the caller gets its result first.
        std::thread::spawn(move || drop(results));
        if profile {
            eprintln!("[files] concat : {:6.1} ms", t_concat.elapsed().as_secs_f64() * 1e3);
        }
        (ids, lens)
    }

    /// Count tokens across corpus files without materializing ids.
    ///
    /// Same file reading, separator splitting, empty-document filtering,
    /// and lossy UTF-8 handling as [`Self::encode_files_flat`]; returns the
    /// total token count over all documents (no special tokens, as in
    /// [`Self::count_tokens`]).
    pub fn count_tokens_files<P: AsRef<Path>>(
        &self,
        paths: &[P],
        separator: &[u8],
    ) -> std::io::Result<usize> {
        let buffers: Vec<FileBytes> = paths
            .iter()
            .map(|p| read_file_bytes(p.as_ref()))
            .collect::<std::io::Result<_>>()?;
        let docs = split_file_docs(&buffers, separator);
        if docs.is_empty() {
            return Ok(0);
        }
        let counts = self.steal_batches(&docs, |chunk, cache| {
            let mut n = 0usize;
            for d in chunk {
                let text: Cow<str> = match std::str::from_utf8(d) {
                    Ok(s) => Cow::Borrowed(s),
                    Err(_) => String::from_utf8_lossy(d),
                };
                n += self.encode_raw_ctx(&text, Some(cache)).len();
            }
            n
        });
        Ok(counts.iter().sum())
    }

    /// Count tokens for multiple texts in parallel.
    pub fn count_tokens_batch(&self, texts: &[&str]) -> Vec<usize> {
        let cpus = num_cpus();
        if texts.is_empty() || cpus == 1 || texts.len() <= cpus {
            return texts.iter().map(|t| self.count_tokens(t)).collect();
        }

        self.steal_batches(texts, |chunk, cache| {
            chunk
                .iter()
                .map(|t| self.encode_raw_ctx(t, Some(cache)).len())
                .collect::<Vec<_>>()
        })
        .into_iter()
        .flatten()
        .collect()
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
    fn test_encode_batch_flat_matches_encode_batch() {
        let tokenizer = make_pretok_tokenizer();
        // Enough texts to cross the parallel threshold (texts.len() > cpus)
        let texts: Vec<&str> = (0..64).map(|i| match i % 5 {
            0 => "Hello world, this is a somewhat longer document to encode.",
            1 => "abc def ghi",
            2 => "",
            3 => "short",
            _ => "the quick brown fox jumps over the lazy dog 0123456789",
        }).collect();
        let (flat, lens) = tokenizer.encode_batch_flat(&texts, false);
        let batch = tokenizer.encode_batch(&texts, false);
        assert_eq!(lens.len(), texts.len());
        assert_eq!(flat.len() as u64, lens.iter().sum::<u64>());
        let mut off = 0usize;
        for (i, enc) in batch.iter().enumerate() {
            let n = lens[i] as usize;
            assert_eq!(&flat[off..off + n], enc.ids.as_slice(), "doc {i}");
            off += n;
        }
        assert_eq!(off, flat.len());
    }

    #[test]
    fn test_encode_batch_flat_empty() {
        let tokenizer = make_pretok_tokenizer();
        let (flat, lens) = tokenizer.encode_batch_flat(&[], false);
        assert!(flat.is_empty() && lens.is_empty());
    }

    /// Unfused per-piece reference: full-text pretokenization, each piece
    /// encoded independently through the ground-truth `Encoder::encode`.
    /// No caches, no chunking, no fused loop.
    fn reference_encode(tokenizer: &Tokenizer, text: &str) -> Vec<TokenId> {
        let pretok = tokenizer.pretokenizer().expect("pretokenizer");
        let mut out = Vec::new();
        for piece in pretok.split(text) {
            out.extend(tokenizer.encoder.encode(piece.as_bytes()));
        }
        out
    }

    fn tricky_texts() -> Vec<String> {
        vec![
            String::new(),
            " ".to_string(),
            "Hello world".to_string(),
            "don't we're I'll O'Toole don'ts".to_string(),
            "a\n\nb  c   d\te".to_string(),
            "日本語のテキスト and English, русский текст".to_string(),
            "money $100.99, 50% off! e.g. Dr. Smith's co-op".to_string(),
            // Pieces over the 15-byte cache key limit (long letter runs)
            "Supercalifragilisticexpialidocious antidisestablishmentarianism".to_string(),
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab".to_string(),
            "«ab» ¹²³ ½ cup \u{200B}école".to_string(),
        ]
    }

    #[test]
    fn test_fused_sequential_matches_reference() {
        let tokenizer = make_pretok_tokenizer();
        for text in tricky_texts() {
            let expect = reference_encode(&tokenizer, &text);
            // Twice: second pass reads pooled-cache entries the first inserted.
            for pass in 0..2 {
                let got = tokenizer.encode(&text, false).ids;
                assert_eq!(got, expect, "pass {pass}, text {:?}", text);
                assert_eq!(tokenizer.count_tokens(&text), expect.len(), "count, text {:?}", text);
            }
        }
    }

    #[test]
    fn test_parallel_path_matches_reference() {
        // A document over PARALLEL_CHUNK_THRESHOLD exercises encode_parallel
        // (chunked, pooled per-thread caches); output must equal the
        // unchunked per-piece reference.
        let tokenizer = make_pretok_tokenizer();
        let atom = "The quick brown fox! Ate 1234 grapes, don't ask — «why?» \u{200B}école 日本語 ";
        let big: String = atom.repeat(2 * Tokenizer::PARALLEL_CHUNK_THRESHOLD / atom.len());
        assert!(big.len() > Tokenizer::PARALLEL_CHUNK_THRESHOLD);
        let expect = reference_encode(&tokenizer, &big);
        assert_eq!(tokenizer.encode(&big, false).ids, expect);
        assert_eq!(tokenizer.count_tokens(&big), expect.len());
    }

    #[test]
    fn test_batch_with_large_doc_matches_reference() {
        // Batch workers never nest parallel chunking; a large doc inside a
        // batch takes the fused sequential path and must still match both
        // the reference and the standalone (parallel) result.
        let tokenizer = make_pretok_tokenizer();
        let atom = "Words, numbers 42 and unicode — ½ cup of \u{AD}soft hyphens. ";
        let big: String = atom.repeat(2 * Tokenizer::PARALLEL_CHUNK_THRESHOLD / atom.len());
        let mut texts: Vec<&str> = vec!["Hello world", "", "don't", &big, "tail piece"];
        // Enough docs to force the steal_batches worker path.
        for _ in 0..32 {
            texts.push("filler doc with some text 123");
        }
        let counts = tokenizer.count_tokens_batch(&texts);
        let encs = tokenizer.encode_batch(&texts, false);
        for (i, t) in texts.iter().enumerate() {
            let expect = reference_encode(&tokenizer, t);
            assert_eq!(encs[i].ids, expect, "doc {i}");
            assert_eq!(counts[i], expect.len(), "doc {i}");
        }
        let (flat, lens) = tokenizer.encode_batch_flat(&texts, false);
        let mut off = 0usize;
        for (i, t) in texts.iter().enumerate() {
            let expect = reference_encode(&tokenizer, t);
            assert_eq!(&flat[off..off + lens[i] as usize], &expect[..], "flat doc {i}");
            off += lens[i] as usize;
        }
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

    // --- encode_files_flat tests ---

    /// Write bytes to a unique temp file and return its path.
    fn tmp_file(name: &str, contents: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir()
            .join(format!("tokie_files_test_{}_{}", std::process::id(), name));
        std::fs::write(&path, contents).unwrap();
        path
    }

    /// Reference result: split on `sep`, drop empties, lossy-convert,
    /// encode each doc with `encode_batch` and concatenate.
    fn reference_files(
        tokenizer: &Tokenizer,
        contents: &[&[u8]],
        sep: &[u8],
    ) -> (Vec<TokenId>, Vec<u64>) {
        let mut docs: Vec<String> = Vec::new();
        for buf in contents {
            let pieces: Vec<&[u8]> = if sep.is_empty() {
                vec![&buf[..]]
            } else {
                let mut out = Vec::new();
                let mut start = 0;
                for pos in memchr::memmem::find_iter(buf, sep) {
                    out.push(&buf[start..pos]);
                    start = pos + sep.len();
                }
                out.push(&buf[start..]);
                out
            };
            for p in pieces {
                if !p.is_empty() {
                    docs.push(String::from_utf8_lossy(p).into_owned());
                }
            }
        }
        let refs: Vec<&str> = docs.iter().map(|s| s.as_str()).collect();
        let encs = tokenizer.encode_batch(&refs, false);
        let mut ids = Vec::new();
        let mut offsets = vec![0u64];
        for e in encs {
            ids.extend_from_slice(&e.ids);
            offsets.push(ids.len() as u64);
        }
        (ids, offsets)
    }

    fn assert_files_match(name: &str, contents: &[&[u8]], sep: &[u8]) {
        let tokenizer = make_pretok_tokenizer();
        let paths: Vec<std::path::PathBuf> = contents
            .iter()
            .enumerate()
            .map(|(i, c)| tmp_file(&format!("{name}_{i}"), c))
            .collect();
        let (ids, offsets) = tokenizer.encode_files_flat(&paths, sep, false).unwrap();
        let (want_ids, want_offsets) = reference_files(&tokenizer, contents, sep);
        for p in &paths {
            let _ = std::fs::remove_file(p);
        }
        assert_eq!(ids, want_ids, "{name}: ids mismatch");
        assert_eq!(offsets, want_offsets, "{name}: offsets mismatch");
    }

    #[test]
    fn test_encode_files_multi_doc() {
        assert_files_match(
            "multi",
            &[b"Hello world<SEP>second doc here<SEP>and a third"],
            b"<SEP>",
        );
    }

    #[test]
    fn test_encode_files_separator_at_edges() {
        // Separator at file start and end: leading/trailing empty docs are
        // dropped, matching Python's `[d for d in text.split(sep) if d]`.
        assert_files_match("edges", &[b"<SEP>middle doc<SEP>"], b"<SEP>");
    }

    #[test]
    fn test_encode_files_consecutive_separators() {
        assert_files_match("consecutive", &[b"one<SEP><SEP><SEP>two"], b"<SEP>");
    }

    #[test]
    fn test_encode_files_no_separator() {
        assert_files_match("nosep", &[b"just one document, no separator"], b"<SEP>");
    }

    #[test]
    fn test_encode_files_non_utf8() {
        // Invalid UTF-8 must take the lossy path (U+FFFD), never panic.
        assert_files_match(
            "nonutf8",
            &[b"good doc<SEP>bad \xff\xfe bytes<SEP>trailing ok"],
            b"<SEP>",
        );
    }

    #[test]
    fn test_encode_files_multiple_files() {
        assert_files_match(
            "multifile",
            &[
                b"file one doc a<SEP>file one doc b<SEP>",
                b"file two doc a",
                b"<SEP>file three doc a<SEP>file three doc b",
            ],
            b"<SEP>",
        );
    }

    #[test]
    fn test_encode_files_empty_file() {
        assert_files_match("emptyfile", &[b"", b"only doc"], b"<SEP>");
    }

    #[test]
    fn test_encode_files_empty_separator() {
        // Empty separator: each file is a single doc.
        assert_files_match("emptysep", &[b"whole file is one doc"], b"");
    }

    #[test]
    fn test_encode_files_many_docs_parallel() {
        // Enough docs to force the steal_batches worker path.
        let mut content = Vec::new();
        for i in 0..200 {
            content.extend_from_slice(
                format!("document number {i} with some text abc").as_bytes(),
            );
            content.extend_from_slice(b"<SEP>");
        }
        assert_files_match("parallel", &[&content], b"<SEP>");
    }

    #[test]
    fn test_encode_files_special_tokens_fallback() {
        // add_special_tokens=true leaves the fused fast path; the
        // per-document fallback must still match encode_batch exactly.
        let tokenizer = make_bert_tokenizer();
        let path = tmp_file("special", b"abc<SEP>ab ab<SEP>zz");
        let (ids, offsets) = tokenizer.encode_files_flat(&[&path], b"<SEP>", true).unwrap();
        let _ = std::fs::remove_file(&path);
        let encs = tokenizer.encode_batch(&["abc", "ab ab", "zz"], true);
        let mut want = Vec::new();
        let mut want_offsets = vec![0u64];
        for e in encs {
            want.extend_from_slice(&e.ids);
            want_offsets.push(want.len() as u64);
        }
        assert_eq!(ids, want);
        assert_eq!(offsets, want_offsets);
    }

    #[test]
    fn test_encode_files_truncation_fallback() {
        // Truncation also leaves the fused fast path.
        let mut tokenizer = make_pretok_tokenizer();
        tokenizer.enable_truncation(TruncationParams { max_length: 2, ..Default::default() });
        let path = tmp_file("trunc", b"Hello world again<SEP>ab");
        let (ids, offsets) = tokenizer.encode_files_flat(&[&path], b"<SEP>", false).unwrap();
        let _ = std::fs::remove_file(&path);
        let encs = tokenizer.encode_batch(&["Hello world again", "ab"], false);
        let mut want = Vec::new();
        for e in encs {
            want.extend_from_slice(&e.ids);
        }
        assert_eq!(ids, want);
        assert_eq!(offsets.len(), 3);
    }

    #[test]
    fn test_encode_files_missing_file_errors() {
        let tokenizer = make_pretok_tokenizer();
        let missing = std::env::temp_dir().join("tokie_files_test_does_not_exist_xyz");
        assert!(tokenizer.encode_files_flat(&[missing], b"<SEP>", false).is_err());
    }

    #[test]
    fn test_count_tokens_files() {
        let tokenizer = make_pretok_tokenizer();
        let content: &[u8] = b"Hello world<SEP>second doc<SEP>third one here";
        let path = tmp_file("count", content);
        let total = tokenizer.count_tokens_files(&[&path], b"<SEP>").unwrap();
        let (ids, _) = tokenizer.encode_files_flat(&[&path], b"<SEP>", false).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(total, ids.len());
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
