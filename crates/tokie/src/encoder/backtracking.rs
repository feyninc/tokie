//! Backtracking BPE encoder with early exit optimization.
//!
//! Key optimizations:
//! 1. Early exit for single-token pieces (88.9% of pretokenized pieces)
//! 2. foldhash + packed u64 keys for fast hash lookups
//! 3. SmallVec to avoid heap allocation for small pieces

use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};
use foldhash::HashMap as FoldHashMap;
use chunk::chunk;
use smallvec::SmallVec;
use std::collections::VecDeque;
use std::thread;

use crate::types::{Split, TokenId};

/// Minimum text size to use parallel processing (10KB).
const PARALLEL_THRESHOLD: usize = 10_000;

/// Maximum token length to cache for early exit lookup.
const MAX_CACHED_TOKEN_LEN: usize = 16;

/// Buffer size for streaming iterator.
const ENCODE_ITER_BUFFER_SIZE: usize = 8;

/// Pack two u32 token IDs into a single u64 key for faster hashing.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Split text into chunks at boundary characters (space/newline).
#[inline]
fn split_at_boundaries(text: &[u8]) -> Vec<&[u8]> {
    let num_cpus = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1);
    let target_size = text.len() / num_cpus;
    chunk(text)
        .size(target_size)
        .delimiters(b" \n")
        .prefix()
        .collect()
}

/// Streaming iterator over encoded tokens.
///
/// Created by [`BacktrackingBytePairEncoder::encode_iter`]. Uses a small buffer (8 tokens)
/// to enable true streaming - tokens are yielded as they're confirmed safe,
/// without pre-computing the entire encoding.
pub struct EncodeIter<'a> {
    encoder: &'a BacktrackingBytePairEncoder,
    text: &'a [u8],
    pos: usize,
    buffer: VecDeque<TokenId>,
    bitfield: Bitfield,
    next_token: Option<TokenId>,
    done: bool,
}

impl<'a> EncodeIter<'a> {
    pub(crate) fn new(encoder: &'a BacktrackingBytePairEncoder, text: &'a [u8]) -> Self {
        let n = text.len();
        let next_token = if text.is_empty() {
            None
        } else {
            encoder.next_match(text)
        };

        Self {
            encoder,
            text,
            pos: 0,
            buffer: VecDeque::with_capacity(ENCODE_ITER_BUFFER_SIZE + 1),
            bitfield: Bitfield::new(n + 1),
            next_token,
            done: text.is_empty(),
        }
    }

    fn encode_one_token(&mut self) -> bool {
        let Some(mut token) = self.next_token else {
            return false;
        };

        let last = self.buffer.back().copied();

        loop {
            let token_len = self.encoder.token_len(token);
            let end_pos = self.pos + token_len;

            let is_reachable = self.bitfield.is_set(end_pos);
            let is_compatible = last
                .map(|last_token| self.encoder.is_valid_pair(last_token, token))
                .unwrap_or(true);

            if is_reachable && is_compatible {
                self.buffer.push_back(token);
                self.pos = end_pos;
                self.next_token = self.encoder.next_match(&self.text[self.pos..]);
                return true;
            } else if let Some(shorter) = self.encoder.next_prefix(token) {
                token = shorter;
            } else {
                self.bitfield.clear(self.pos);
                if let Some(last_token) = self.buffer.pop_back() {
                    self.pos -= self.encoder.token_len(last_token);
                    self.next_token = Some(last_token);
                    return false;
                } else {
                    self.next_token = None;
                    return false;
                }
            }
        }
    }
}

impl Iterator for EncodeIter<'_> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        if self.done {
            return self.buffer.pop_front();
        }

        while self.buffer.len() < ENCODE_ITER_BUFFER_SIZE {
            if !self.encode_one_token() {
                if self.next_token.is_none() {
                    self.done = true;
                    break;
                }
            }
        }

        self.buffer.pop_front()
    }
}

impl std::iter::FusedIterator for EncodeIter<'_> {}

/// BPE encoder using greedy matching with backtracking + early exit.
///
/// Optimized version that checks if input is already a single token
/// before running the full backtracking algorithm.
#[derive(Clone)]
pub struct BacktrackingBytePairEncoder {
    split_table: Vec<Split>,
    /// Maps packed (left, right) u64 -> merged TokenId.
    pair_lookup: FoldHashMap<u64, TokenId>,
    token_lengths: Vec<u8>,
    num_base_tokens: usize,
    matcher: DoubleArrayAhoCorasick,
    next_prefix_match: Vec<TokenId>,
    /// Maps byte sequence -> token ID for early exit.
    /// Uses foldhash for fast lookups.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,
}

impl BacktrackingBytePairEncoder {
    /// Create a new BPE encoder from merge rules.
    pub fn from_merges(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
    ) -> (Self, Vec<Vec<u8>>) {
        Self::from_merges_with_added(merges, base_tokens, &[])
    }

    /// Create a BPE encoder from a complete vocabulary and merge rules.
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes)| bytes.clone()).collect();

        let bytes_to_id: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        let mut pair_lookup = FoldHashMap::default();
        let mut merge_creates: FoldHashMap<TokenId, (TokenId, TokenId)> = FoldHashMap::default();

        for &(left, right) in merges.iter() {
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend_from_slice(&token_bytes[right as usize]);

            if let Some(&merged_id) = bytes_to_id.get(&merged_bytes) {
                pair_lookup.insert(pack_pair(left, right), merged_id);
                merge_creates.entry(merged_id).or_insert((left, right));
            }
        }

        let mut split_table: Vec<Split> = Vec::with_capacity(vocab.len());
        for (id, _) in vocab.iter() {
            let id = *id as TokenId;
            if let Some(&(left, right)) = merge_creates.get(&id) {
                split_table.push(Split::merge(left, right));
            } else {
                split_table.push(Split::base(id));
            }
        }

        let (matcher, next_prefix_match) = Self::build_matcher_and_prefixes(&token_bytes);
        let token_lengths = Self::build_token_lengths(&token_bytes);

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let encoder = Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
        };

        (encoder, token_bytes)
    }

    /// Create a BPE encoder from merge rules, handling added/special tokens.
    pub fn from_merges_with_added(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
        added_tokens: &[(u32, Vec<u8>)],
    ) -> (Self, Vec<Vec<u8>>) {
        let num_base_tokens = base_tokens.len();

        let mut split_table: Vec<Split> = (0..num_base_tokens as TokenId)
            .map(Split::base)
            .collect();

        let mut token_bytes: Vec<Vec<u8>> = base_tokens.to_vec();
        let mut pair_lookup = FoldHashMap::default();

        let mut added_sorted: Vec<_> = added_tokens.to_vec();
        added_sorted.sort_by_key(|(id, _)| *id);
        let mut added_iter = added_sorted.into_iter().peekable();

        for &(left, right) in merges.iter() {
            let next_id = split_table.len() as TokenId;

            // Insert any added tokens that come before this merge
            while let Some(&(added_id, _)) = added_iter.peek() {
                if added_id <= next_id {
                    let (_, bytes) = added_iter.next().unwrap();
                    split_table.push(Split::base(split_table.len() as TokenId));
                    token_bytes.push(bytes);
                } else {
                    break;
                }
            }

            let new_id = split_table.len() as TokenId;
            split_table.push(Split::merge(left, right));
            pair_lookup.insert(pack_pair(left, right), new_id);

            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }

        // Append remaining added tokens
        for (_, bytes) in added_iter {
            split_table.push(Split::base(split_table.len() as TokenId));
            token_bytes.push(bytes);
        }

        let (matcher, next_prefix_match) = Self::build_matcher_and_prefixes(&token_bytes);
        let token_lengths = Self::build_token_lengths(&token_bytes);

        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let encoder = Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
        };

        (encoder, token_bytes)
    }

    /// Create a BPE encoder from pre-built components (for deserialization).
    pub fn from_parts(
        split_table: Vec<Split>,
        pair_lookup: FoldHashMap<u64, TokenId>,
        token_lengths: Vec<u8>,
        num_base_tokens: usize,
        matcher: DoubleArrayAhoCorasick,
        next_prefix_match: Vec<TokenId>,
        token_bytes: &[Vec<u8>],
    ) -> Self {
        // Build token_cache for early exit
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
        }
    }

    // === Builder Helpers ===

    /// Build the Aho-Corasick matcher and prefix lookup table.
    fn build_matcher_and_prefixes(token_bytes: &[Vec<u8>]) -> (DoubleArrayAhoCorasick, Vec<TokenId>) {
        let mut trie = Trie::new();
        for (id, bytes) in token_bytes.iter().enumerate() {
            trie.add(bytes, id as TokenId);
        }
        trie.build(MatchKind::LeftmostLongest);
        let matcher = trie.compile();

        let next_prefix_match: Vec<TokenId> = token_bytes
            .iter()
            .map(|token| {
                if token.len() <= 1 {
                    u32::MAX
                } else {
                    let prefix = &token[..token.len() - 1];
                    matcher
                        .find_iter(prefix)
                        .next()
                        .map(|m| m.pattern_id)
                        .unwrap_or(u32::MAX)
                }
            })
            .collect();

        (matcher, next_prefix_match)
    }

    /// Build the token lengths table.
    fn build_token_lengths(token_bytes: &[Vec<u8>]) -> Vec<u8> {
        token_bytes
            .iter()
            .map(|t| t.len().min(255) as u8)
            .collect()
    }

    /// Get a reference to the split table.
    pub fn split_table(&self) -> &[Split] {
        &self.split_table
    }

    /// Get a reference to the DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Get a reference to the next_prefix_match table.
    pub fn next_prefix_match_table(&self) -> &[TokenId] {
        &self.next_prefix_match
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    #[inline]
    pub fn is_valid_pair(&self, mut token1: TokenId, mut token2: TokenId) -> bool {
        let mut limit = u32::MAX;

        loop {
            if let Some(&combined) = self.pair_lookup.get(&pack_pair(token1, token2)) {
                if combined < limit {
                    return false;
                }
            }

            if token1 > token2 {
                limit = token1;
                let right = self.split_table[token1 as usize].right;
                if right == token1 {
                    limit = token2 + 1;
                    let left = self.split_table[token2 as usize].left;
                    if left + 1 == limit {
                        return true;
                    }
                    token2 = left;
                } else {
                    token1 = right;
                }
            } else {
                limit = token2 + 1;
                let left = self.split_table[token2 as usize].left;
                if left + 1 == limit {
                    limit = token1;
                    let right = self.split_table[token1 as usize].right;
                    if right == limit {
                        return true;
                    }
                    token1 = right;
                } else {
                    token2 = left;
                }
            }
        }
    }

    /// Get the length of a token in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        self.token_lengths[token as usize] as usize
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.token_lengths.len()
    }

    /// Get the number of base tokens.
    pub fn num_base_tokens(&self) -> usize {
        self.num_base_tokens
    }

    /// Append the encoding of one pretokenized piece to `out`.
    ///
    /// The hot path for corpus encoding: no per-piece Vec, and with a
    /// `PretokenCache` most pieces resolve to a single 32-byte table probe
    /// (pretoken frequency is Zipfian — on web text the vast majority of
    /// pieces repeat, and ~90% encode to a single token).
    pub fn encode_into(&self, text: &[u8], mut cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        if text.is_empty() {
            return;
        }
        if let Some(c) = cache.as_deref_mut() {
            if text.len() <= CACHE_KEY_MAX && c.get(text, out) {
                return;
            }
        }
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                if let Some(c) = cache {
                    c.insert(text, &[token_id]);
                }
                out.push(token_id);
                return;
            }
        }
        if text.len() >= PARALLEL_THRESHOLD {
            // Degenerate giant piece: fall back to the chunk-parallel path
            out.extend(self.encode(text));
            return;
        }
        let start = out.len();
        self.encode_sequential_into(text, out);
        if let Some(c) = cache {
            c.insert(text, &out[start..]); // self-guards key/value size limits
        }
    }

    /// Encode text into BPE tokens.
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION: Early exit if input is already a single token
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
        }

        if text.len() < PARALLEL_THRESHOLD {
            return self.encode_sequential(text);
        }

        let chunks = split_at_boundaries(text);

        if chunks.len() == 1 {
            return self.encode_sequential(chunks[0]);
        }

        let results: Vec<Vec<TokenId>> = thread::scope(|s| {
            let handles: Vec<_> = chunks
                .iter()
                .map(|chunk| s.spawn(|| self.encode_sequential(chunk)))
                .collect();

            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });

        let total: usize = results.iter().map(|v| v.len()).sum();
        let mut output = Vec::with_capacity(total);
        for chunk in results {
            output.extend(chunk);
        }
        output
    }

    /// Returns a streaming iterator over encoded tokens.
    pub fn encode_iter<'a>(&'a self, text: &'a [u8]) -> EncodeIter<'a> {
        EncodeIter::new(self, text)
    }

    /// Encode multiple texts in parallel.
    pub fn encode_batch(&self, texts: &[&[u8]]) -> Vec<Vec<TokenId>> {
        if texts.is_empty() {
            return Vec::new();
        }

        let num_cpus = thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);

        if texts.len() <= num_cpus || num_cpus == 1 {
            if num_cpus == 1 {
                return texts.iter().map(|t| self.encode_sequential(t)).collect();
            }

            return thread::scope(|s| {
                let handles: Vec<_> = texts
                    .iter()
                    .map(|text| s.spawn(|| self.encode_sequential(text)))
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
        }

        let chunk_size = (texts.len() + num_cpus - 1) / num_cpus;

        thread::scope(|s| {
            let handles: Vec<_> = texts
                .chunks(chunk_size)
                .map(|chunk| {
                    s.spawn(|| {
                        chunk
                            .iter()
                            .map(|t| self.encode_sequential(t))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();

            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        })
    }

    fn encode_sequential(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION: Early exit if input is already a single token
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
        }

        let mut out = Vec::new();
        self.encode_sequential_into(text, &mut out);
        out
    }

    fn encode_sequential_into(&self, text: &[u8], out: &mut Vec<TokenId>) {
        let n = text.len();
        // Use SmallVec to avoid heap allocation for small pieces
        let mut tokens: SmallVec<[TokenId; 16]> = SmallVec::new();
        let mut bitfield = Bitfield::new(n + 1);

        let mut pos = 0;
        let mut next_token = self.next_match(&text[pos..]);

        while let Some(mut token) = next_token {
            let last = tokens.last().copied();

            loop {
                let token_len = self.token_len(token);
                let end_pos = pos + token_len;

                let is_reachable = bitfield.is_set(end_pos);
                let is_compatible = last
                    .map(|last_token| self.is_valid_pair(last_token, token))
                    .unwrap_or(true);

                if is_reachable && is_compatible {
                    tokens.push(token);
                    pos = end_pos;
                    next_token = self.next_match(&text[pos..]);
                    break;
                } else if let Some(shorter) = self.next_prefix(token) {
                    token = shorter;
                } else {
                    bitfield.clear(pos);
                    if let Some(last_token) = tokens.pop() {
                        pos -= self.token_len(last_token);
                    }
                    next_token = last;
                    break;
                }
            }
        }

        out.extend_from_slice(&tokens);
    }

    #[inline]
    fn next_match(&self, text: &[u8]) -> Option<TokenId> {
        self.matcher.find_iter(text).next().map(|m| m.pattern_id)
    }

    #[inline]
    fn next_prefix(&self, token: TokenId) -> Option<TokenId> {
        let prefix = self.next_prefix_match[token as usize];
        if prefix == u32::MAX {
            None
        } else {
            Some(prefix)
        }
    }
}

/// Per-thread cache of pretoken bytes → encoded token sequence.
///
/// Open-addressing table of 32-byte entries: a 16-byte key block
/// (`[len, bytes...]`, compared as one 16-byte memcmp) plus up to 3 inline
/// token ids. Sized so a warm chunk's working set stays resident; collisions
/// beyond the probe window overwrite the home slot, which Zipfian pretoken
/// frequency makes self-correcting (hot keys win back their slot).
pub struct PretokenCache {
    entries: Box<[CacheEntry]>,
    mask: usize,
}

const CACHE_KEY_MAX: usize = 15;
const CACHE_BITS_DEFAULT: usize = 16; // 65536 entries * 32 B = 2 MiB (fits M-series shared L2 alongside 8 workers)
const CACHE_PROBES: usize = 4;
const CACHE_MAX_TOKENS: usize = 3;

/// Table size exponent, overridable for tuning via TOKIE_CACHE_BITS.
fn cache_bits() -> usize {
    static BITS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BITS.get_or_init(|| {
        std::env::var("TOKIE_CACHE_BITS").ok()
            .and_then(|v| v.parse().ok())
            .filter(|&b| (10..=24).contains(&b))
            .unwrap_or(CACHE_BITS_DEFAULT)
    })
}

#[derive(Clone, Copy)]
#[repr(C)]
struct CacheEntry {
    /// key[0] = piece length (0 = empty slot), key[1..1+len] = piece bytes.
    key: [u8; 16],
    toks: [TokenId; CACHE_MAX_TOKENS],
    ntok: u32,
}

impl PretokenCache {
    pub fn new() -> Self {
        let empty = CacheEntry { key: [0; 16], toks: [0; CACHE_MAX_TOKENS], ntok: 0 };
        let n = 1usize << cache_bits();
        Self { entries: vec![empty; n].into_boxed_slice(), mask: n - 1 }
    }

    /// Reset every entry to empty (for reuse under a different tokenizer).
    pub fn clear(&mut self) {
        let empty = CacheEntry { key: [0; 16], toks: [0; CACHE_MAX_TOKENS], ntok: 0 };
        self.entries.fill(empty);
    }

    #[inline(always)]
    fn key_block(bytes: &[u8]) -> [u8; 16] {
        debug_assert!(!bytes.is_empty() && bytes.len() <= CACHE_KEY_MAX);
        let mut k = [0u8; 16];
        k[0] = bytes.len() as u8;
        k[1..1 + bytes.len()].copy_from_slice(bytes);
        k
    }

    #[inline(always)]
    fn slot(&self, k: &[u8; 16]) -> usize {
        let a = u64::from_le_bytes(k[..8].try_into().unwrap());
        let b = u64::from_le_bytes(k[8..].try_into().unwrap());
        let h = (a ^ 0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xA076_1D64_78BD_642F)
            ^ b.wrapping_mul(0xE703_7ED1_A0B4_28DB);
        ((h ^ (h >> 32)) as usize) & self.mask
    }

    /// Look up a piece; on hit, append its tokens to `out` and return true.
    #[inline(always)]
    fn get(&self, bytes: &[u8], out: &mut Vec<TokenId>) -> bool {
        let k = Self::key_block(bytes);
        let mut i = self.slot(&k);
        for _ in 0..CACHE_PROBES {
            let e = &self.entries[i];
            if e.key == k {
                out.extend_from_slice(&e.toks[..e.ntok as usize]);
                return true;
            }
            if e.key[0] == 0 {
                return false;
            }
            i = (i + 1) & self.mask;
        }
        false
    }

    #[inline]
    fn insert(&mut self, bytes: &[u8], toks: &[TokenId]) {
        if bytes.is_empty() || bytes.len() > CACHE_KEY_MAX || toks.is_empty() || toks.len() > CACHE_MAX_TOKENS {
            return;
        }
        let k = Self::key_block(bytes);
        let home = self.slot(&k);
        let mut i = home;
        let mut target = home;
        for _ in 0..CACHE_PROBES {
            let e = &self.entries[i];
            if e.key[0] == 0 || e.key == k {
                target = i;
                break;
            }
            i = (i + 1) & self.mask;
        }
        let e = &mut self.entries[target];
        e.key = k;
        e.ntok = toks.len() as u32;
        e.toks[..toks.len()].copy_from_slice(toks);
    }
}

impl Default for PretokenCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Bitfield for tracking reachable positions.
///
/// Inline storage for pieces up to 255 bytes (the overwhelmingly common
/// case) — no heap allocation on the per-piece hot path.
struct Bitfield {
    bits: SmallVec<[u64; 4]>,
}

impl Bitfield {
    fn new(size: usize) -> Self {
        let num_words = (size + 63) / 64;
        let mut bits = SmallVec::new();
        bits.resize(num_words, u64::MAX);
        Self { bits }
    }

    #[inline]
    fn clear(&mut self, pos: usize) {
        let word = pos / 64;
        let bit = pos % 64;
        self.bits[word] &= !(1 << bit);
    }

    #[inline]
    fn is_set(&self, pos: usize) -> bool {
        let word = pos / 64;
        let bit = pos % 64;
        (self.bits[word] >> bit) & 1 != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::VocabDecoder;

    #[test]
    fn test_encode_into_with_cache_matches_encode() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // ab, abc
        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let pieces: Vec<&[u8]> = vec![
            b"abc", b"ab", b"ba", b"cab", b"abcabcabc", b"a", b"",
            b"cccab", // 4 tokens: too long to cache, must still be correct
            b"abcabcabcabcabca", // 16 bytes: over the cache key limit
        ];
        let mut cache = PretokenCache::new();
        // Two passes: the second pass reads entries the first pass inserted
        for pass in 0..2 {
            for &p in &pieces {
                let expect = encoder.encode(p);
                let mut got = Vec::new();
                encoder.encode_into(p, Some(&mut cache), &mut got);
                assert_eq!(got, expect, "pass {pass}, piece {:?}", p);
            }
        }
        // And without a cache at all
        for &p in &pieces {
            let mut got = Vec::new();
            encoder.encode_into(p, None, &mut got);
            assert_eq!(got, encoder.encode(p), "no-cache piece {:?}", p);
        }
    }

    #[test]
    fn test_from_merges() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)];

        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = VocabDecoder::new(token_bytes);

        assert_eq!(encoder.vocab_size(), 5);
        assert_eq!(encoder.num_base_tokens(), 3);
        assert_eq!(decoder.token_to_bytes(0), b"a");
        assert_eq!(decoder.token_to_bytes(3), b"ab");
        assert_eq!(decoder.token_to_bytes(4), b"abc");
    }

    #[test]
    fn test_is_valid_pair() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);

        assert!(!encoder.is_valid_pair(0, 1));
        assert!(encoder.is_valid_pair(3, 2));
        assert!(encoder.is_valid_pair(1, 2));
    }

    #[test]
    fn test_encode_merged_token() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);

        assert_eq!(encoder.encode(b"ab"), vec![3]);
        assert_eq!(encoder.encode(b"abc"), vec![3, 2]);
    }

    #[test]
    fn test_early_exit() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)];

        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);

        // Single byte - early exit
        assert_eq!(encoder.encode(b"a"), vec![0]);

        // "ab" is token 3 - early exit
        assert_eq!(encoder.encode(b"ab"), vec![3]);

        // "abc" is token 4 - early exit
        assert_eq!(encoder.encode(b"abc"), vec![4]);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![(0, 1), (2, 3), (4, 5)];

        let (encoder, token_bytes) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = VocabDecoder::new(token_bytes);

        for text in [b"abcd".as_slice(), b"ab", b"cd", b"abcdabcd", b"a", b""] {
            let encoded = encoder.encode(text);
            let decoded = decoder.decode(&encoded);
            assert_eq!(decoded, text);
        }
    }

    #[test]
    fn test_encode_iter_matches_encode() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![(0, 1), (2, 3), (4, 5)];

        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);

        for text in [b"".as_slice(), b"a", b"ab", b"abcd", b"abcdabcdabcdabcdabcd"] {
            let encoded = encoder.encode(text);
            let iter_encoded: Vec<_> = encoder.encode_iter(text).collect();
            assert_eq!(encoded, iter_encoded);
        }
    }
}
