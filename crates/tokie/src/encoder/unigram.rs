//! Unigram encoder using Viterbi dynamic programming.
//!
//! Implements the Unigram language model tokenization algorithm as used by
//! SentencePiece Unigram models (T5, XLM-RoBERTa, ALBERT, mT5, etc.).
//!
//! Unlike BPE (deterministic merges), Unigram uses probabilistic tokenization:
//! - Each token has a **log probability score**
//! - Find segmentation that **maximizes total score** using Viterbi DP
//! - Time: O(n × L) where L = max token length
//!
//! Hot path: normalized text is split at every metaspace `▁` (same boundaries
//! as HF Metaspace `MergedWithNext`), each short unit is Viterbi'd independently
//! (fresh f64 score accumulator), and Zipf-hot units are memoized in
//! [`UnigramPieceCache`].

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};
use foldhash::HashMap as FoldHashMap;
use smallvec::SmallVec;

use crate::types::TokenId;

/// Process-unique id so thread-local unit caches never alias after an
/// encoder is dropped and another is allocated at the same address.
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

/// Get the length of a UTF-8 character from its first byte.
#[inline]
fn utf8_char_len(b: u8) -> usize {
    match b {
        0..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xFF => 4,
        _ => 1,
    }
}

/// Metaspace character (▁) in UTF-8: E2 96 81.
const METASPACE: [u8; 3] = [0xE2, 0x96, 0x81];

/// Maximum token length to cache for early exit lookup (unk-bridging only).
const MAX_CACHED_TOKEN_LEN: usize = 16;

/// log2 of the direct-mapped front-cache size. 2^18 ≈ 4 MiB of keys + 2 MiB
/// of (offset,len) values — enough for Zipf-hot `▁word` units without the
/// 16 MiB front table gigatoken uses for SP BPE.
const FRONT_BITS: u32 = 18;

/// Units of ≤ this many bytes use the packed `u128` short/front path.
const SHORT_KEY_MAX: usize = 15;

/// Scan a vocabulary's token bytes for the per-`▁`-unit split guard.
///
/// Returns `true` (safe to split) when **no** token contains the metaspace
/// sequence at an interior offset (> 0). A leading `▁` (offset 0, the normal
/// `▁word` shape) is fine; an interior `▁` marks a multi-word token that a
/// per-unit split could never reassemble, so its presence forces the
/// whole-string Viterbi fallback in [`UnigramEncoder::encode_into`].
#[inline]
fn compute_unit_split_safe(token_bytes: &[Vec<u8>]) -> bool {
    !token_bytes
        .iter()
        .any(|bytes| memchr::memmem::find_iter(bytes, &METASPACE).any(|pos| pos > 0))
}

thread_local! {
    /// Thread-local unit cache tagged by encoder identity so switching
    /// tokenizers on the same thread cannot return another model's ids.
    static THREAD_UNIT_CACHE: RefCell<Option<(usize, UnigramPieceCache)>> =
        const { RefCell::new(None) };
}

/// Per-thread / pooled memoization of Unigram `▁`-unit encodings.
///
/// Units repeat heavily under Zipf (e.g. `▁the`, `▁of`). Cache entries store
/// `(offset, len)` into an append-only token arena so variable-length
/// segmentations aren't capped like [`crate::encoder::PretokenCache`]'s
/// 15B / 3-token inline slots.
pub struct UnigramPieceCache {
    arena: Vec<TokenId>,
    front_keys: Box<[u128]>,
    front_vals: Box<[(u32, u32)]>,
    short: FoldHashMap<u128, (u32, u32)>,
    long: FoldHashMap<Box<[u8]>, (u32, u32)>,
}

impl UnigramPieceCache {
    pub fn new() -> Self {
        let n = 1usize << FRONT_BITS;
        Self {
            arena: Vec::new(),
            front_keys: vec![0u128; n].into_boxed_slice(),
            front_vals: vec![(0u32, 0u32); n].into_boxed_slice(),
            short: FoldHashMap::default(),
            long: FoldHashMap::default(),
        }
    }

    /// Drop all cached entries (tokenizer switch / pool generation change).
    pub fn clear(&mut self) {
        self.arena.clear();
        self.front_keys.fill(0);
        self.front_vals.fill((0, 0));
        self.short.clear();
        self.long.clear();
    }

    #[inline]
    fn front_index(key: u128) -> usize {
        let folded = (key as u64) ^ ((key >> 64) as u64);
        let h = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> (64 - FRONT_BITS)) as usize
    }

    /// Pack ≤15-byte units into a length-tagged `u128` (nonzero length in
    /// the top nibble so key 0 remains the empty-slot sentinel).
    #[inline]
    fn pack_key(bytes: &[u8]) -> Option<u128> {
        let n = bytes.len();
        if n == 0 || n > SHORT_KEY_MAX {
            return None;
        }
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        Some(u128::from_le_bytes(lanes) | ((n as u128) << 120))
    }

    #[inline]
    fn lookup(&mut self, unit: &[u8], out: &mut Vec<TokenId>) -> bool {
        if let Some(key) = Self::pack_key(unit) {
            let idx = Self::front_index(key);
            if self.front_keys[idx] == key {
                let (offset, len) = self.front_vals[idx];
                let start = offset as usize;
                out.extend_from_slice(&self.arena[start..start + len as usize]);
                return true;
            }
            if let Some(&(offset, len)) = self.short.get(&key) {
                self.front_keys[idx] = key;
                self.front_vals[idx] = (offset, len);
                let start = offset as usize;
                out.extend_from_slice(&self.arena[start..start + len as usize]);
                return true;
            }
            false
        } else if let Some(&(offset, len)) = self.long.get(unit) {
            let start = offset as usize;
            out.extend_from_slice(&self.arena[start..start + len as usize]);
            true
        } else {
            false
        }
    }

    #[inline]
    fn insert(&mut self, unit: &[u8], toks: &[TokenId]) {
        if unit.is_empty() || toks.is_empty() {
            return;
        }
        let offset = self.arena.len() as u32;
        let len = toks.len() as u32;
        self.arena.extend_from_slice(toks);
        if let Some(key) = Self::pack_key(unit) {
            self.short.insert(key, (offset, len));
            let idx = Self::front_index(key);
            self.front_keys[idx] = key;
            self.front_vals[idx] = (offset, len);
        } else {
            self.long.insert(unit.to_vec().into_boxed_slice(), (offset, len));
        }
    }
}

impl Default for UnigramPieceCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Unigram encoder using Viterbi dynamic programming.
#[derive(Clone)]
pub struct UnigramEncoder {
    /// Aho-Corasick automaton for efficient token matching.
    matcher: DoubleArrayAhoCorasick,

    /// Token scores (log probabilities), indexed by token ID.
    ///
    /// Stored as f64 to match HF exactly: near-tie Viterbi paths (e.g. equal
    /// token multisets summed in different orders) resolve by the last ulp of
    /// these sums, so rounding through f32 flips segmentations vs HF.
    scores: Vec<f64>,

    /// Unknown token ID.
    unk_token: TokenId,

    /// Byte fallback: byte value -> token ID for <0xXX> tokens.
    /// u32::MAX means no byte fallback token exists.
    byte_tokens: [TokenId; 256],

    /// Token lengths in bytes.
    token_lengths: Vec<u16>,

    /// Vocabulary size.
    vocab_size: usize,

    /// Maps byte sequence -> token ID for early exit.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// Whether the model has <0xXX> byte fallback tokens.
    /// When false, <unk> gets a heavy penalty in Viterbi to prefer real tokens.
    has_byte_fallback: bool,

    /// Identity for thread-local unit-cache tagging.
    cache_id: u64,

    /// Whether it is safe to split the input at every metaspace `▁` and
    /// Viterbi each unit independently. False when the vocab contains any
    /// multi-word token (metaspace at an interior offset), in which case
    /// encoding falls back to whole-string Viterbi to stay exact. See
    /// [`compute_unit_split_safe`].
    unit_split_safe: bool,
}

impl std::fmt::Debug for UnigramEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnigramEncoder")
            .field("vocab_size", &self.vocab_size)
            .field("unk_token", &self.unk_token)
            .field("unit_split_safe", &self.unit_split_safe)
            .finish()
    }
}

impl UnigramEncoder {
    /// Create encoder from vocabulary with scores.
    ///
    /// # Arguments
    /// * `vocab` - List of (token_id, token_bytes, score) tuples, sorted by ID
    /// * `unk_token` - Token ID to use for unknown sequences
    pub fn from_vocab_with_scores(
        vocab: &[(u32, Vec<u8>, f64)],
        unk_token: TokenId,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes, _)| bytes.clone()).collect();
        let scores: Vec<f64> = vocab.iter().map(|(_, _, score)| *score).collect();

        // Build byte fallback table (<0xXX> tokens)
        let mut byte_tokens = [u32::MAX; 256];
        for (id, bytes, _) in vocab {
            // <0xXX> tokens are 6 bytes: '<', '0', 'x', hex, hex, '>'
            if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes.ends_with(b">") {
                if let Ok(byte_val) = u8::from_str_radix(
                    std::str::from_utf8(&bytes[3..5]).unwrap_or(""),
                    16,
                ) {
                    if byte_tokens[byte_val as usize] == u32::MAX {
                        byte_tokens[byte_val as usize] = *id;
                    }
                }
            }
        }

        // Build Aho-Corasick matcher for all tokens
        // Use Overlapping to get ALL matches at each position (required for Viterbi)
        let mut trie = Trie::new();
        for (id, bytes, _) in vocab {
            if !bytes.is_empty() {
                trie.add(bytes, *id);
            }
        }
        trie.build(MatchKind::Overlapping);
        let matcher = trie.compile();

        let token_lengths: Vec<u16> = token_bytes.iter().map(|b| b.len() as u16).collect();

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (id, bytes, _) in vocab {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), *id);
            }
        }

        let has_byte_fallback = byte_tokens.iter().any(|&t| t != u32::MAX);

        let encoder = Self {
            matcher,
            scores,
            unk_token,
            byte_tokens,
            token_lengths,
            vocab_size: vocab.len(),
            token_cache,
            has_byte_fallback,
            cache_id: NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            unit_split_safe: compute_unit_split_safe(&token_bytes),
        };

        (encoder, token_bytes)
    }

    /// Create encoder from pre-built components (for deserialization).
    pub fn from_parts(
        matcher: DoubleArrayAhoCorasick,
        scores: Vec<f64>,
        unk_token: TokenId,
        byte_tokens: [TokenId; 256],
        token_lengths: Vec<u16>,
        token_bytes: &[Vec<u8>],
    ) -> Self {
        let vocab_size = scores.len();

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), id as TokenId);
            }
        }

        let has_byte_fallback = byte_tokens.iter().any(|&t| t != u32::MAX);

        Self {
            matcher,
            scores,
            unk_token,
            byte_tokens,
            token_lengths,
            vocab_size,
            token_cache,
            has_byte_fallback,
            cache_id: NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed),
            unit_split_safe: compute_unit_split_safe(token_bytes),
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get number of base tokens (for Unigram, this is vocab_size since there are no merges).
    pub fn num_base_tokens(&self) -> usize {
        self.vocab_size
    }

    /// Get unknown token ID.
    pub fn unk_token(&self) -> TokenId {
        self.unk_token
    }

    /// Get token scores.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// Get byte fallback tokens.
    pub fn byte_tokens(&self) -> &[TokenId; 256] {
        &self.byte_tokens
    }

    /// Get token lengths.
    pub fn token_lengths(&self) -> &[u16] {
        &self.token_lengths
    }

    /// Get reference to the DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Whether encoding uses the fast per-`▁`-unit split path (`true`) or the
    /// whole-string Viterbi fallback for vocabs with interior-`▁` tokens.
    #[inline]
    pub fn unit_split_safe(&self) -> bool {
        self.unit_split_safe
    }

    /// Get token length in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.token_lengths[token as usize] as usize
    }

    /// Check if two tokens can appear adjacent in valid Unigram encoding.
    /// For Unigram, any pair is valid since there are no merge constraints.
    #[inline]
    pub fn is_valid_pair(&self, _token1: TokenId, _token2: TokenId) -> bool {
        true
    }

    /// Encode text to token IDs using per-`▁` unit Viterbi + thread-local memoization.
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(text.len() / 3);
        self.encode_into(text, None, &mut out);
        out
    }

    /// Append the encoding of `text` to `out`, memoizing `▁` units in `cache`.
    ///
    /// When `cache` is `None`, a process thread-local cache keyed by this
    /// encoder's address is used so repeated single-threaded encodes still
    /// benefit from Zipf hits without leaking entries across models.
    ///
    /// When [`Self::unit_split_safe`] is false the vocab has a multi-word
    /// (interior-`▁`) token, so the input is Viterbi'd as a single whole
    /// string with no metaspace splitting — exact, at the cost of the unit
    /// memoization. `viterbi_unit` treats `▁` as an ordinary byte, so running
    /// it over the entire text is precisely whole-string Viterbi.
    pub fn encode_into(
        &self,
        text: &[u8],
        cache: Option<&mut UnigramPieceCache>,
        out: &mut Vec<TokenId>,
    ) {
        if text.is_empty() {
            return;
        }
        if !self.unit_split_safe {
            // Whole-string Viterbi fallback: no `▁` split, cache bypassed.
            let toks = self.viterbi_unit(text);
            self.append_collapsing_unk(out, &toks);
            return;
        }
        match cache {
            Some(cache) => self.encode_units_into(text, cache, out),
            None => {
                let key = self.cache_id as usize;
                THREAD_UNIT_CACHE.with(|slot| {
                    let mut slot = slot.borrow_mut();
                    let needs_new = match slot.as_ref() {
                        Some((k, _)) => *k != key,
                        None => true,
                    };
                    if needs_new {
                        *slot = Some((key, UnigramPieceCache::new()));
                    }
                    self.encode_units_into(text, &mut slot.as_mut().unwrap().1, out);
                });
            }
        }
    }

    /// Split at every metaspace `▁` and encode each unit through the cache.
    fn encode_units_into(
        &self,
        text: &[u8],
        cache: &mut UnigramPieceCache,
        out: &mut Vec<TokenId>,
    ) {
        let mut unit_start = 0usize;
        for pos in memchr::memmem::find_iter(text, &METASPACE) {
            if pos > unit_start {
                self.encode_unit_cached(&text[unit_start..pos], cache, out);
                unit_start = pos;
            }
        }
        if unit_start < text.len() {
            self.encode_unit_cached(&text[unit_start..], cache, out);
        }
    }

    #[inline]
    fn encode_unit_cached(
        &self,
        unit: &[u8],
        cache: &mut UnigramPieceCache,
        out: &mut Vec<TokenId>,
    ) {
        if unit.is_empty() {
            return;
        }
        let start_len = out.len();
        if cache.lookup(unit, out) {
            self.collapse_unk_join(out, start_len);
            return;
        }
        let tokens = self.viterbi_unit(unit);
        cache.insert(unit, &tokens);
        self.append_collapsing_unk(out, &tokens);
    }

    /// If `out` already ends with `<unk>` and `tokens` starts with `<unk>`,
    /// drop the leading run so consecutive unks stay collapsed across units.
    #[inline]
    fn append_collapsing_unk(&self, out: &mut Vec<TokenId>, tokens: &[TokenId]) {
        if tokens.is_empty() {
            return;
        }
        let mut start = 0;
        if out.last() == Some(&self.unk_token) {
            while start < tokens.len() && tokens[start] == self.unk_token {
                start += 1;
            }
        }
        out.extend_from_slice(&tokens[start..]);
    }

    /// After a cache hit append, collapse a leading unk against the prior token.
    #[inline]
    fn collapse_unk_join(&self, out: &mut Vec<TokenId>, appended_at: usize) {
        if appended_at == 0 || appended_at >= out.len() {
            return;
        }
        if out[appended_at - 1] == self.unk_token && out[appended_at] == self.unk_token {
            let mut end = appended_at;
            while end < out.len() && out[end] == self.unk_token {
                end += 1;
            }
            out.drain(appended_at..end);
        }
    }

    /// Viterbi on a single `▁` unit (no interior metaspace barriers).
    fn viterbi_unit(&self, text: &[u8]) -> Vec<TokenId> {
        // NOTE: Unlike BPE, Unigram cannot use early exit for single-token matches.
        // Even if the input matches a single token, Viterbi might find a better
        // segmentation (e.g., "ab" as [a, b] scores -0.2 vs "ab" scores -10.0).
        let n = text.len();
        if n == 0 {
            return Vec::new();
        }

        let mut best_score = vec![f64::NEG_INFINITY; n + 1];
        let mut backptr: Vec<(TokenId, usize)> = vec![(0, 0); n + 1];
        best_score[0] = 0.0;

        let unk_penalty = if self.has_byte_fallback {
            self.scores[self.unk_token as usize]
        } else {
            -100.0
        };

        type MatchList = SmallVec<[(usize, TokenId); 8]>;
        let mut matches_at: Vec<MatchList> = vec![SmallVec::new(); n];

        for m in self.matcher.find_iter(text) {
            matches_at[m.start].push((m.end, m.pattern_id));
        }

        for pos in 0..n {
            if best_score[pos] == f64::NEG_INFINITY {
                continue;
            }

            let current_score = best_score[pos];
            let has_match = !matches_at[pos].is_empty();

            for &(end, token_id) in &matches_at[pos] {
                let token_score = self.scores[token_id as usize];
                let new_score = current_score + token_score;
                if new_score > best_score[end] {
                    best_score[end] = new_score;
                    backptr[end] = (token_id, pos);
                }
            }

            let byte_val = text[pos];
            let byte_token = self.byte_tokens[byte_val as usize];
            if byte_token != u32::MAX {
                let token_score = self.scores[byte_token as usize];
                let new_score = current_score + token_score;

                if new_score > best_score[pos + 1] {
                    best_score[pos + 1] = new_score;
                    backptr[pos + 1] = (byte_token, pos);
                }
            } else if !has_match {
                let char_len = utf8_char_len(text[pos]);
                let end = (pos + char_len).min(n);
                let new_score = current_score + unk_penalty;
                if new_score > best_score[end] {
                    best_score[end] = new_score;
                    backptr[end] = (self.unk_token, pos);
                }
            }
        }

        if best_score[n] == f64::NEG_INFINITY {
            return self.encode_with_unk_bridging(text);
        }

        self.collect_tokens_from_backptr(&backptr, n)
    }

    /// Collect tokens from backpointer array (backward pass of Viterbi).
    /// Consecutive `<unk>` tokens are collapsed into a single `<unk>`,
    /// matching SentencePiece behavior where unknown character runs
    /// produce exactly one `<unk>` token.
    #[inline]
    fn collect_tokens_from_backptr(&self, backptr: &[(TokenId, usize)], end: usize) -> Vec<TokenId> {
        let mut tokens = Vec::new();
        let mut pos = end;
        while pos > 0 {
            let (token_id, start_pos) = backptr[pos];
            tokens.push(token_id);
            pos = start_pos;
        }
        tokens.reverse();

        if tokens.contains(&self.unk_token) {
            tokens.dedup_by(|a, b| *a == self.unk_token && *b == self.unk_token);
        }

        tokens
    }

    /// Encode with <unk> bridging for positions that couldn't be reached.
    fn encode_with_unk_bridging(&self, text: &[u8]) -> Vec<TokenId> {
        let n = text.len();
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < n {
            let max_len = (n - pos).min(MAX_CACHED_TOKEN_LEN);
            let mut best_match: Option<(usize, TokenId)> = None;

            for len in (1..=max_len).rev() {
                let substr = &text[pos..pos + len];
                if let Some(&token_id) = self.token_cache.get(substr) {
                    best_match = Some((len, token_id));
                    break;
                }
            }

            let remaining = &text[pos..];
            if let Some(m) = self.matcher.find_iter(remaining).next() {
                if m.start == 0 && (best_match.is_none() || m.end > best_match.unwrap().0) {
                    best_match = Some((m.end, m.pattern_id));
                }
            }

            if let Some((len, token_id)) = best_match {
                tokens.push(token_id);
                pos += len;
            } else {
                let byte_val = text[pos];
                let byte_token = self.byte_tokens[byte_val as usize];
                if byte_token != u32::MAX {
                    tokens.push(byte_token);
                } else {
                    tokens.push(self.unk_token);
                }
                pos += 1;
            }
        }

        tokens
    }

    /// Encode text using chunked processing for better memory efficiency.
    ///
    /// With per-unit Viterbi, DP memory is O(unit length); this remains as a
    /// thin wrapper that walks the same `▁` units.
    pub fn encode_chunked(&self, text: &[u8], _chunk_size: usize) -> Vec<TokenId> {
        self.encode(text)
    }

    /// Encode using the historical default chunk size (unused; units are small).
    #[inline]
    pub fn encode_chunked_default(&self, text: &[u8]) -> Vec<TokenId> {
        self.encode(text)
    }

    /// Encode a single buffer with Viterbi (no unit cache). Kept for tests /
    /// callers that want a cold DP over an already-split piece.
    ///
    /// This always splits at every `▁` regardless of [`Self::unit_split_safe`],
    /// so on an unsafe vocab it deliberately reproduces the *naive* per-unit
    /// segmentation — the guard tests use it as the "without the guard" oracle.
    pub fn encode_single(&self, text: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut unit_start = 0usize;
        for pos in memchr::memmem::find_iter(text, &METASPACE) {
            if pos > unit_start {
                let toks = self.viterbi_unit(&text[unit_start..pos]);
                self.append_collapsing_unk(&mut out, &toks);
                unit_start = pos;
            }
        }
        if unit_start < text.len() {
            let toks = self.viterbi_unit(&text[unit_start..]);
            self.append_collapsing_unk(&mut out, &toks);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_unigram() {
        // Simple vocab: "hello" (high score), "hell", "o", "h", "e", "l"
        let vocab = vec![
            (0, b"h".to_vec(), -1.0),
            (1, b"e".to_vec(), -1.0),
            (2, b"l".to_vec(), -1.0),
            (3, b"o".to_vec(), -1.0),
            (4, b"hell".to_vec(), -2.0),
            (5, b"hello".to_vec(), -3.0), // Best: one token for "hello"
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // "hello" should be tokenized as single token [5] due to Viterbi
        // Score -3.0 vs h+e+l+l+o = -5.0
        assert_eq!(encoder.encode(b"hello"), vec![5]);

        // "h" should be single token [0]
        assert_eq!(encoder.encode(b"h"), vec![0]);
    }

    #[test]
    fn test_viterbi_chooses_best_path() {
        // Vocab where individual chars score better than combined
        let vocab = vec![
            (0, b"a".to_vec(), -0.1),    // Very good score
            (1, b"b".to_vec(), -0.1),    // Very good score
            (2, b"ab".to_vec(), -10.0),  // Bad score - should not be chosen
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // "ab" should be tokenized as [0, 1] (score -0.2) not [2] (score -10.0)
        assert_eq!(encoder.encode(b"ab"), vec![0, 1]);
    }

    #[test]
    fn test_byte_fallback() {
        // Vocab with byte fallback tokens
        let vocab = vec![
            (0, b"<0x00>".to_vec(), -5.0),
            (1, b"<0x01>".to_vec(), -5.0),
            (2, b"<0xFF>".to_vec(), -5.0),
            (3, b"hello".to_vec(), -1.0),
        ];

        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);

        // Verify byte tokens were parsed correctly
        assert_eq!(encoder.byte_tokens[0x00], 0);
        assert_eq!(encoder.byte_tokens[0x01], 1);
        assert_eq!(encoder.byte_tokens[0xFF], 2);
    }

    #[test]
    fn test_empty_input() {
        let vocab = vec![(0, b"a".to_vec(), -1.0)];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        let empty: Vec<TokenId> = vec![];
        assert_eq!(encoder.encode(b""), empty);
    }

    #[test]
    fn test_vocab_size() {
        let vocab = vec![
            (0, b"a".to_vec(), -1.0),
            (1, b"b".to_vec(), -1.0),
            (2, b"c".to_vec(), -1.0),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        assert_eq!(encoder.vocab_size(), 3);
        assert_eq!(encoder.num_base_tokens(), 3);
    }

    #[test]
    fn test_unit_cache_matches_cold() {
        let mark = "▁".as_bytes();
        let vocab = vec![
            (0, mark.to_vec(), -1.0),
            (1, [mark, b"a"].concat(), -0.5),
            (2, b"a".to_vec(), -1.0),
            (3, b"b".to_vec(), -1.0),
            (4, [mark, b"b"].concat(), -0.5),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        let text = [mark, b"a", mark, b"b", mark, b"a"].concat();
        let cold = encoder.encode_single(&text);
        let mut cache = UnigramPieceCache::new();
        let mut warm = Vec::new();
        encoder.encode_into(&text, Some(&mut cache), &mut warm);
        assert_eq!(cold, warm);
        // Second pass should hit cache and match
        let mut warm2 = Vec::new();
        encoder.encode_into(&text, Some(&mut cache), &mut warm2);
        assert_eq!(cold, warm2);
    }

    // --- per-`▁`-unit split correctness guard -----------------------------

    /// Metaspace bytes for `▁`.
    fn mark() -> &'static [u8] {
        "▁".as_bytes()
    }

    /// A vocab with a multi-word (interior-`▁`) token `▁a▁b` scored so high
    /// that whole-string Viterbi selects it, plus the single-`▁`-unit pieces
    /// so the naive per-unit split resolves `▁a▁b` to `[▁a, ▁b]` instead.
    fn interior_metaspace_vocab() -> (UnigramEncoder, Vec<TokenId>) {
        // ids: 0=▁a▁b (best), 1=▁a, 2=▁b, 3=a, 4=b, 5=▁ (also <unk>)
        let vocab = vec![
            (0u32, [mark(), b"a", mark(), b"b"].concat(), -0.5),
            (1u32, [mark(), b"a"].concat(), -3.0),
            (2u32, [mark(), b"b"].concat(), -3.0),
            (3u32, b"a".to_vec(), -5.0),
            (4u32, b"b".to_vec(), -5.0),
            (5u32, mark().to_vec(), -5.0),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 5);
        // Whole-string best path over "▁a▁b" is the single token [0]
        // (-0.5) vs the split [1, 2] (-6.0).
        (encoder, vec![0])
    }

    #[test]
    fn test_interior_metaspace_forces_whole_string_viterbi() {
        let (encoder, whole_string_expected) = interior_metaspace_vocab();
        let text = [mark(), b"a", mark(), b"b"].concat();

        // The guard must detect the multi-word token and flip to unsafe.
        assert!(
            !encoder.unit_split_safe(),
            "interior-`▁` token must make the vocab unit-split-unsafe"
        );

        // With the guard, `encode` runs whole-string Viterbi and uses the
        // multi-word token `▁a▁b` = [0].
        assert_eq!(
            encoder.encode(&text),
            whole_string_expected,
            "guarded encode must match whole-string Viterbi"
        );

        // Sanity: the naive per-unit split (encode_single, always splitting)
        // resolves the SAME input differently — proving the guard changes the
        // result and is load-bearing.
        let naive_split = encoder.encode_single(&text);
        assert_eq!(
            naive_split,
            vec![1, 2],
            "naive per-unit split picks [▁a, ▁b] — the wrong segmentation"
        );
        assert_ne!(
            encoder.encode(&text),
            naive_split,
            "guard must produce a different (correct) result than the naive split"
        );

        // The pooled/worker-cache path must also honor the guard.
        let mut cache = UnigramPieceCache::new();
        let mut warm = Vec::new();
        encoder.encode_into(&text, Some(&mut cache), &mut warm);
        assert_eq!(warm, whole_string_expected, "cache path must honor the guard");
    }

    #[test]
    fn test_normal_vocab_is_unit_split_safe() {
        // No interior `▁` anywhere → fast split path stays enabled.
        let vocab = vec![
            (0u32, mark().to_vec(), -1.0),
            (1u32, [mark(), b"the"].concat(), -0.5),
            (2u32, [mark(), b"a"].concat(), -0.5),
            (3u32, b"t".to_vec(), -2.0),
            (4u32, b"h".to_vec(), -2.0),
            (5u32, b"e".to_vec(), -2.0),
            (6u32, b"a".to_vec(), -2.0),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        assert!(
            encoder.unit_split_safe(),
            "vocab without interior `▁` must be unit-split-safe"
        );

        // Fast split path must still equal the cold per-unit reference.
        let text = [mark(), b"the", mark(), b"a"].concat();
        let cold = encoder.encode_single(&text);
        assert_eq!(encoder.encode(&text), cold);
        assert_eq!(encoder.encode(&text), vec![1, 2]);
    }

    #[test]
    fn test_leading_metaspace_is_safe() {
        // A token that merely STARTS with `▁` (offset 0) is the normal shape
        // and must NOT trip the guard.
        let vocab = vec![
            (0u32, mark().to_vec(), -1.0),
            (1u32, [mark(), b"word"].concat(), -0.5),
            (2u32, b"w".to_vec(), -2.0),
        ];
        let (encoder, _) = UnigramEncoder::from_vocab_with_scores(&vocab, 0);
        assert!(encoder.unit_split_safe());
    }
}
