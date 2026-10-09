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

use super::simple::MergeCore;
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

/// Inline capacity of the rank-merge core's token buffer (it is now only a
/// fallback for vocabs the merge core rejects, on pieces of 15 bytes or less).
const RANK_MERGE_MAX_LEN: usize = 32;

/// Direct-index dense sub-table bound: pairs with both ids below this use a
/// flat array lookup instead of the hash probe. 512*512*4 B = 1 MiB.
const DENSE_PAIR_BOUND: u32 = 512;

/// Empty-slot sentinel for the open-addressed pair table. No valid packed
/// pair can equal this (both halves would need to be u32::MAX, which is
/// never a token id).
const PAIR_EMPTY_KEY: u64 = u64::MAX;

/// Rank/merge lookup for BPE pairs, gigatoken-style.
///
/// Open-addressed flat table of inline 16-byte entries mapping a packed
/// (left, right) u64 to the merged token id, plus a dense direct-indexed
/// sub-table for pairs where both ids are below [`DENSE_PAIR_BOUND`]
/// (which covers every first-round byte-pair lookup).
///
/// The merge *rank* is the merged token id itself: tokie's backtracking
/// encoder already defines canonical order by merged id (`is_valid_pair`
/// compares `combined < limit`), so using the id keeps the rank-merge loop
/// consistent with the DAAC walk by construction. A construction-time check
/// verifies every merge produces an id greater than both parts (true for
/// well-formed BPE vocabs); otherwise the rank-merge path is disabled.
#[derive(Clone)]
struct RankPairTable {
    /// Open-addressed table; slot count is a power of two.
    entries: Box<[PairEntry]>,
    mask: usize,
    /// Dense sub-table: `dense[(l << 9) | r]` = merged id or u32::MAX.
    dense: Box<[u32]>,
    /// byte value -> base token id for that single byte.
    byte_to_base: [TokenId; 256],
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PairEntry {
    key: u64,
    merged: TokenId,
    _pad: u32,
}

impl RankPairTable {
    /// Build the table, or return None when the vocab lacks the required
    /// structure (missing single-byte base tokens, or a merge whose id is
    /// not greater than both parts).
    fn build(pair_lookup: &FoldHashMap<u64, TokenId>, token_bytes: &[Vec<u8>]) -> Option<Self> {
        if pair_lookup.is_empty() {
            return None;
        }

        // Byte -> base token map: lowest id single-byte token wins.
        let mut byte_to_base = [u32::MAX; 256];
        for (id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() == 1 && byte_to_base[bytes[0] as usize] == u32::MAX {
                byte_to_base[bytes[0] as usize] = id as TokenId;
            }
        }
        if byte_to_base.iter().any(|&id| id == u32::MAX) {
            return None; // not a byte-complete vocab; keep DAAC everywhere
        }

        // Merge-monotonicity check: rank-merge merges the lowest merged id
        // first and assumes newly created pairs always rank later.
        for (&key, &merged) in pair_lookup.iter() {
            let left = (key >> 32) as u32;
            let right = key as u32;
            if merged <= left || merged <= right {
                return None;
            }
        }

        let slots = (pair_lookup.len() * 2).next_power_of_two();
        let mut entries = vec![
            PairEntry { key: PAIR_EMPTY_KEY, merged: 0, _pad: 0 };
            slots
        ]
        .into_boxed_slice();
        let mask = slots - 1;

        let dense_len = (DENSE_PAIR_BOUND * DENSE_PAIR_BOUND) as usize;
        let mut dense = vec![u32::MAX; dense_len].into_boxed_slice();

        for (&key, &merged) in pair_lookup.iter() {
            let left = (key >> 32) as u32;
            let right = key as u32;
            if left < DENSE_PAIR_BOUND && right < DENSE_PAIR_BOUND {
                let idx = ((left << 9) | right) as usize;
                // Duplicate pairs (same bytes, several ids) keep the lowest id,
                // matching the min-rank selection the merge loop performs.
                if merged < dense[idx] {
                    dense[idx] = merged;
                }
            }
            let mut i = Self::home_slot(key, mask);
            loop {
                let e = &mut entries[i];
                if e.key == PAIR_EMPTY_KEY {
                    e.key = key;
                    e.merged = merged;
                    break;
                }
                if e.key == key {
                    if merged < e.merged {
                        e.merged = merged;
                    }
                    break;
                }
                i = (i + 1) & mask;
            }
        }

        Some(Self { entries, mask, dense, byte_to_base })
    }

    #[inline(always)]
    fn home_slot(key: u64, mask: usize) -> usize {
        // Fibonacci multiplicative hash on the packed pair; the pair ids are
        // small so the high bits need the multiply to get mixed.
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> 32) as usize) & mask
    }

    /// Merged id for (left, right), or u32::MAX when the pair never merges.
    #[inline(always)]
    fn merged_id(&self, left: TokenId, right: TokenId) -> u32 {
        if left < DENSE_PAIR_BOUND && right < DENSE_PAIR_BOUND {
            return self.dense[((left << 9) | right) as usize];
        }
        self.merged_id_flat(left, right)
    }

    /// Flat-table probe only (used to measure whether the dense table pays).
    #[inline(always)]
    fn merged_id_flat(&self, left: TokenId, right: TokenId) -> u32 {
        let key = pack_pair(left, right);
        let mut i = Self::home_slot(key, self.mask);
        loop {
            let e = self.entries[i];
            if e.key == key {
                return e.merged;
            }
            if e.key == PAIR_EMPTY_KEY {
                return u32::MAX;
            }
            i = (i + 1) & self.mask;
        }
    }
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
    /// Rank-based merge table for the short-piece cache-miss path.
    /// None when the vocab lacks byte-complete base tokens or monotone
    /// merge ids; those vocabs keep the DAAC walk everywhere.
    rank_table: Option<RankPairTable>,
    /// Carried-code multipass/queue merge over the same pairs, ranked by
    /// merged id. Built whenever `rank_table` is (same preconditions) and
    /// merged ids are unique per pair.
    merge_core: Option<MergeCore>,
    /// Longest uncached piece routed to `merge_core` instead of the DAAC
    /// walk (see [`default_core_max_len`]).
    core_max_len: usize,
}

/// Vocab size up to which the merge core beats the DAAC walk at every
/// piece length (gpt2-sized vocabs). On larger vocabs (cl100k, o200k,
/// DeepSeek, Qwen) the core only wins on short pieces: their pair table
/// spills out of cache and long pieces take many merge rounds, while the
/// DAAC walk stays linear in output tokens.
const CORE_ANY_LEN_MAX_VOCAB: usize = 65_536;

fn default_core_max_len(vocab_len: usize) -> usize {
    if let Some(v) = std::env::var("TOKIE_CORE_MAX_LEN").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    if vocab_len <= CORE_ANY_LEN_MAX_VOCAB { usize::MAX } else { CACHE_KEY_MAX }
}

/// Build the flat merge core for a vocab the rank table accepted: rank =
/// merged id, which must identify the pair (the core's queue drops stale
/// entries by rank, and its SAFE pass merges every pair with the winning
/// rank). Every pair is SAFE because merges are id-monotone.
fn build_merge_core(rank_table: &Option<RankPairTable>, pair_lookup: &FoldHashMap<u64, TokenId>) -> Option<MergeCore> {
    let table = rank_table.as_ref()?;
    let mut ranked: FoldHashMap<u64, (TokenId, u32)> = FoldHashMap::default();
    let mut seen: FoldHashMap<TokenId, ()> = FoldHashMap::default();
    for (&k, &m) in pair_lookup {
        if seen.insert(m, ()).is_some() {
            return None;
        }
        ranked.insert(k, (m, m));
    }
    MergeCore::build(&ranked, table.byte_to_base)
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

        let rank_table = RankPairTable::build(&pair_lookup, &token_bytes);
        let merge_core = build_merge_core(&rank_table, &pair_lookup);
        let core_max_len = default_core_max_len(token_bytes.len());
        let encoder = Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
            rank_table,
            merge_core,
            core_max_len,
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

        let rank_table = RankPairTable::build(&pair_lookup, &token_bytes);
        let merge_core = build_merge_core(&rank_table, &pair_lookup);
        let core_max_len = default_core_max_len(token_bytes.len());
        let encoder = Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
            rank_table,
            merge_core,
            core_max_len,
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

        let rank_table = RankPairTable::build(&pair_lookup, token_bytes);
        let merge_core = build_merge_core(&rank_table, &pair_lookup);
        let core_max_len = default_core_max_len(token_bytes.len());
        Self {
            split_table,
            pair_lookup,
            token_lengths,
            num_base_tokens,
            matcher,
            next_prefix_match,
            token_cache,
            rank_table,
            merge_core,
            core_max_len,
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
    #[inline]
    pub fn encode_into(&self, text: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        self.encode_piece_into(text, text, cache, out)
    }

    /// Like [`Self::encode_into`], but `piece` is known to be a subslice of
    /// `doc` (e.g. a pretokenizer split of the document being encoded). The
    /// surrounding document lets the cache key be built with one masked
    /// 16-byte load instead of length-dependent partial loads. Passing a
    /// `piece` that is not inside `doc` is safe — it just loses that fast
    /// path (and `encode_into` does exactly that with `doc == piece`).
    #[inline]
    pub fn encode_piece_into(&self, doc: &[u8], piece: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        if piece.is_empty() {
            return;
        }
        if let Some(c) = cache {
            if piece.len() <= CACHE_KEY_MAX {
                let (lo, hi) = key_words_within(doc, piece);
                if c.get_with_key(lo, hi, out) {
                    return;
                }
                return self.encode_cache_miss(piece, lo, hi, c, out);
            }
            if piece.len() <= LONG_KEY_MAX {
                return self.encode_long_cached(piece, c, out);
            }
        }
        self.encode_uncached(piece, out)
    }

    /// Cache-miss slow path: outlined so the hit path stays tight.
    /// `lo`/`hi` are the piece's already-computed key words. Pieces with
    /// more tokens than an inline entry holds go to the long-piece cache.
    #[inline(never)]
    fn encode_cache_miss(&self, text: &[u8], lo: u64, hi: u64, cache: &mut PretokenCache, out: &mut Vec<TokenId>) {
        debug_assert!(text.len() <= CACHE_KEY_MAX);
        if let Some(&token_id) = self.token_cache.get(text) {
            cache.insert_with_key(lo, hi, &[token_id]);
            out.push(token_id);
            return;
        }
        if cache.get_long(text, out) {
            return;
        }
        let start = out.len();
        self.encode_short(text, out);
        let toks = &out[start..];
        if toks.len() <= CACHE_MAX_TOKENS {
            if !toks.is_empty() {
                cache.insert_with_key(lo, hi, toks);
            }
        } else {
            cache.insert_long(text, toks);
        }
    }

    /// Pieces of 16..=LONG_KEY_MAX bytes: long-piece cache, then the
    /// uncached path.
    #[inline(never)]
    fn encode_long_cached(&self, text: &[u8], cache: &mut PretokenCache, out: &mut Vec<TokenId>) {
        if cache.get_long(text, out) {
            return;
        }
        let start = out.len();
        self.encode_uncached(text, out);
        cache.insert_long(text, &out[start..]);
    }

    /// Miss path for a piece of at most CACHE_KEY_MAX bytes that is not a
    /// single token: the merge core wins at this length on every vocab
    /// measured; the older rank-merge core and the DAAC walk are fallbacks.
    #[inline]
    fn encode_short(&self, text: &[u8], out: &mut Vec<TokenId>) {
        if let Some(core) = &self.merge_core {
            core.encode(text, out);
        } else if self.rank_table.is_some() {
            self.encode_rank_merge(text, out);
        } else {
            self.encode_sequential_into(text, out);
        }
    }

    /// No-cache / long-piece path.
    #[inline(never)]
    fn encode_uncached(&self, text: &[u8], out: &mut Vec<TokenId>) {
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                out.push(token_id);
                return;
            }
        }
        if text.len() <= CACHE_KEY_MAX {
            return self.encode_short(text, out);
        }
        if text.len() >= PARALLEL_THRESHOLD {
            // Degenerate giant piece: fall back to the chunk-parallel path
            out.extend(self.encode(text));
            return;
        }
        match &self.merge_core {
            Some(core) if text.len() <= self.core_max_len => core.encode(text, out),
            _ => self.encode_sequential_into(text, out),
        }
    }

    /// Whether the flat merge core is available for this vocab.
    pub fn has_merge_core(&self) -> bool {
        self.merge_core.is_some()
    }

    /// Encode one piece with the flat merge core (multipass for short
    /// pieces, queue for long ones), bypassing caches.
    ///
    /// Panics if the core is unavailable; check [`Self::has_merge_core`].
    #[doc(hidden)]
    pub fn encode_merge_core(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.merge_core.as_ref().expect("merge core unavailable").encode(text, out)
    }

    /// Whether the rank-merge core is available for this vocab.
    pub fn has_rank_merge(&self) -> bool {
        self.rank_table.is_some()
    }

    /// Encode one piece with the rank-based BPE merge loop.
    ///
    /// Starts from per-byte base tokens and repeatedly merges the
    /// lowest-ranked adjacent pair (rank = merged token id, see
    /// [`RankPairTable`]) until no pair in the table applies. All
    /// occurrences of the winning pair are merged left-to-right in one
    /// pass, which is equivalent to one-at-a-time lowest-rank merging
    /// because merges are id-monotone (checked at construction).
    ///
    /// Panics if the rank table is unavailable; callers must check
    /// [`Self::has_rank_merge`] first.
    #[doc(hidden)]
    pub fn encode_rank_merge(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.encode_rank_merge_impl::<true>(text, out)
    }

    /// Flat-probe-only variant, used to measure whether the dense
    /// direct-index sub-table pays for itself.
    #[doc(hidden)]
    pub fn encode_rank_merge_flat(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.encode_rank_merge_impl::<false>(text, out)
    }

    #[inline(always)]
    fn encode_rank_merge_impl<const DENSE: bool>(&self, text: &[u8], out: &mut Vec<TokenId>) {
        let table = self.rank_table.as_ref().expect("rank table unavailable");

        let mut toks: SmallVec<[TokenId; RANK_MERGE_MAX_LEN]> = text
            .iter()
            .map(|&b| table.byte_to_base[b as usize])
            .collect();

        while toks.len() > 1 {
            // Find the lowest-rank adjacent pair (leftmost on ties).
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            for i in 0..toks.len() - 1 {
                let m = if DENSE {
                    table.merged_id(toks[i], toks[i + 1])
                } else {
                    table.merged_id_flat(toks[i], toks[i + 1])
                };
                if m < best_rank {
                    best_rank = m;
                    best_i = i;
                }
            }
            if best_i == usize::MAX {
                break;
            }

            // Merge every occurrence of that exact pair, left to right.
            let left = toks[best_i];
            let right = toks[best_i + 1];
            let mut w = best_i;
            let mut i = best_i;
            let n = toks.len();
            while i < n {
                if i + 1 < n && toks[i] == left && toks[i + 1] == right {
                    toks[w] = best_rank;
                    i += 2;
                } else {
                    toks[w] = toks[i];
                    i += 1;
                }
                w += 1;
            }
            toks.truncate(w);
        }

        out.extend_from_slice(&toks);
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

    /// DAAC greedy-longest-match walk with validity backtracking.
    /// Public (hidden) so differential tests can compare it against
    /// [`Self::encode_rank_merge`] directly, bypassing caches.
    #[doc(hidden)]
    pub fn encode_sequential_into(&self, text: &[u8], out: &mut Vec<TokenId>) {
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

    /// Profiling hook: direct single-token lookup in the token byte cache.
    #[doc(hidden)]
    #[inline]
    pub fn token_cache_get(&self, text: &[u8]) -> Option<TokenId> {
        self.token_cache.get(text).copied()
    }

    /// Profiling hook: run the full backtracking path, bypassing all caches.
    #[doc(hidden)]
    #[inline]
    pub fn encode_backtrack_into(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.encode_sequential_into(text, out);
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
/// Open-addressing table of 32-byte entries: a 16-byte inline key held as
/// two u64 words (piece bytes zero-padded, length in the top byte — built
/// with overlapping loads, no memcpy) plus up to 3 inline token ids. Sized
/// so a warm chunk's working set stays resident; collisions
/// beyond the probe window overwrite the home slot, which Zipfian pretoken
/// frequency makes self-correcting (hot keys win back their slot).
pub struct PretokenCache {
    entries: Box<[CacheEntry]>,
    mask: usize,
    /// Pieces the inline table can't hold (longer than 15 bytes, or more
    /// than 3 tokens), allocated on first use.
    long: Option<Box<LongCache>>,
}

/// Longest piece the long-piece cache keys on. Longer pieces are rare and
/// (outside code) almost never repeat.
const LONG_KEY_MAX: usize = 256;
const LONG_SLOTS: usize = 1 << 13;
const LONG_PROBES: usize = 4;
/// Arena budgets; hitting either (or 3/4 slot occupancy) flushes the whole
/// cache, so memory stays bounded and there is no per-entry eviction.
const LONG_ARENA_BYTES: usize = 512 << 10;
const LONG_ARENA_TOKS: usize = 128 << 10;

/// Overflow cache for long / many-token pieces: open-addressed slots over
/// key and token arenas, flushed wholesale when full (HF v1's generation
/// flush). Repeated long pieces are common in code (identifiers, paths,
/// indentation runs) and cost 10x+ per byte on the miss path.
struct LongCache {
    slots: Box<[LongSlot]>,
    used: usize,
    bytes: Vec<u8>,
    toks: Vec<TokenId>,
}

#[derive(Clone, Copy, Default)]
struct LongSlot {
    /// Piece hash with the low bit forced on; 0 marks an empty slot.
    hash: u64,
    key_off: u32,
    tok_off: u32,
    key_len: u16,
    ntok: u16,
}

/// Hash of a 1..=LONG_KEY_MAX byte piece: 8-byte words with an
/// overlapping tail word, multiply-mixed.
#[inline]
fn long_hash(p: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let n = p.len();
    let mut h = (n as u64).wrapping_mul(K);
    if n >= 8 {
        let mut i = 0;
        while i + 8 <= n {
            let w = u64::from_le_bytes(p[i..i + 8].try_into().unwrap());
            h = (h ^ w).wrapping_mul(K).rotate_left(29);
            i += 8;
        }
        if i < n {
            let w = u64::from_le_bytes(p[n - 8..].try_into().unwrap());
            h = (h ^ w).wrapping_mul(K).rotate_left(29);
        }
    } else {
        h = (h ^ load_le_partial(p)).wrapping_mul(K);
    }
    (h ^ (h >> 31)) | 1
}

impl LongCache {
    fn new() -> Box<Self> {
        Box::new(Self {
            slots: vec![LongSlot::default(); LONG_SLOTS].into_boxed_slice(),
            used: 0,
            bytes: Vec::with_capacity(LONG_ARENA_BYTES),
            toks: Vec::with_capacity(LONG_ARENA_TOKS),
        })
    }

    fn clear(&mut self) {
        if self.used > 0 {
            self.slots.fill(LongSlot::default());
        }
        self.used = 0;
        self.bytes.clear();
        self.toks.clear();
    }

    #[inline]
    fn get(&self, piece: &[u8], out: &mut Vec<TokenId>) -> bool {
        let h = long_hash(piece);
        let mut i = (h >> 32) as usize & (LONG_SLOTS - 1);
        for _ in 0..LONG_PROBES {
            let s = self.slots[i];
            if s.hash == 0 {
                return false;
            }
            if s.hash == h && s.key_len as usize == piece.len() {
                let k = s.key_off as usize;
                if &self.bytes[k..k + piece.len()] == piece {
                    let t = s.tok_off as usize;
                    out.extend_from_slice(&self.toks[t..t + s.ntok as usize]);
                    return true;
                }
            }
            i = (i + 1) & (LONG_SLOTS - 1);
        }
        false
    }

    fn insert(&mut self, piece: &[u8], toks: &[TokenId]) {
        if self.bytes.len() + piece.len() > LONG_ARENA_BYTES
            || self.toks.len() + toks.len() > LONG_ARENA_TOKS
            || self.used >= LONG_SLOTS / 4 * 3
        {
            self.clear();
        }
        let h = long_hash(piece);
        let home = (h >> 32) as usize & (LONG_SLOTS - 1);
        let mut i = home;
        let mut target = home;
        for _ in 0..LONG_PROBES {
            if self.slots[i].hash == 0 {
                target = i;
                self.used += 1;
                break;
            }
            i = (i + 1) & (LONG_SLOTS - 1);
        }
        // Probe window full: overwrite the home slot (its arena bytes leak
        // until the next flush).
        self.slots[target] = LongSlot {
            hash: h,
            key_off: self.bytes.len() as u32,
            tok_off: self.toks.len() as u32,
            key_len: piece.len() as u16,
            ntok: toks.len() as u16,
        };
        self.bytes.extend_from_slice(piece);
        self.toks.extend_from_slice(toks);
    }
}

const CACHE_KEY_MAX: usize = 15;
const CACHE_BITS_DEFAULT: usize = 16; // 65536 entries * 32 B = 2 MiB (fits M-series shared L2 alongside 8 workers)
const CACHE_PROBES: usize = 4;
const CACHE_MAX_TOKENS: usize = 3;

/// TOKIE_NO_LONG_CACHE=1 disables the long-piece cache (A/B switch).
#[inline]
fn long_cache_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !matches!(std::env::var("TOKIE_NO_LONG_CACHE").as_deref(), Ok(v) if !v.is_empty() && v != "0"))
}

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
    /// Canonical key: `lo` = first 8 piece bytes (LE, zero-padded), `hi` =
    /// remaining bytes (LE, zero-padded) with the piece length in the top
    /// byte. A real key always has `hi != 0` (len >= 1), so `hi == 0` marks
    /// an empty slot.
    key_lo: u64,
    key_hi: u64,
    toks: [TokenId; CACHE_MAX_TOKENS],
    ntok: u32,
}

/// Zero-padded little-endian load of 1..=7 bytes, branch-light.
///
/// Uses overlapping reads: each byte lands at its own bit position, and
/// overlapped bytes OR with themselves, so the result is exact.
#[inline(always)]
fn load_le_partial(p: &[u8]) -> u64 {
    let len = p.len();
    debug_assert!((1..=7).contains(&len));
    if len >= 4 {
        let a = u32::from_le_bytes(p[..4].try_into().unwrap()) as u64;
        let b = u32::from_le_bytes(p[len - 4..].try_into().unwrap()) as u64;
        a | (b << ((len - 4) * 8))
    } else {
        let a = p[0] as u64;
        let b = (p[len / 2] as u64) << ((len / 2) * 8);
        let c = (p[len - 1] as u64) << ((len - 1) * 8);
        a | b | c
    }
}

/// Build the canonical (lo, hi) key words for `piece` (1..=15 bytes) when
/// `piece` is a subslice of `doc`: a single unconditional 16-byte load from
/// `doc` masked down to `len` bytes — no length-dependent branches. Falls
/// back to [`key_words`] near the end of `doc` or when `piece` is not
/// inside `doc` (detected by the bounds check; distinct live allocations
/// are disjoint, so an in-bounds offset proves the bytes are the piece's).
#[inline(always)]
pub(crate) fn key_words_within(doc: &[u8], piece: &[u8]) -> (u64, u64) {
    let len = piece.len();
    debug_assert!((1..=CACHE_KEY_MAX).contains(&len));
    let start = (piece.as_ptr() as usize).wrapping_sub(doc.as_ptr() as usize);
    if start <= doc.len() && doc.len() - start >= 16 {
        let raw = u128::from_le_bytes(doc[start..start + 16].try_into().unwrap());
        let masked = raw & (u128::MAX >> (128 - 8 * len));
        (masked as u64, (masked >> 64) as u64 | ((len as u64) << 56))
    } else {
        key_words(piece)
    }
}

/// Build the canonical (lo, hi) key words for a piece of 1..=15 bytes.
#[inline(always)]
fn key_words(bytes: &[u8]) -> (u64, u64) {
    let len = bytes.len();
    debug_assert!((1..=CACHE_KEY_MAX).contains(&len));
    let (lo, mut hi) = if len >= 8 {
        let lo = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let hi = if len > 8 { load_le_partial(&bytes[8..]) } else { 0 };
        (lo, hi)
    } else {
        (load_le_partial(bytes), 0)
    };
    hi |= (len as u64) << 56;
    (lo, hi)
}

impl PretokenCache {
    /// Longest piece (in bytes) the cache can key on.
    pub const KEY_MAX: usize = CACHE_KEY_MAX;
    /// Most tokens a cached entry can hold.
    pub const MAX_TOKENS: usize = CACHE_MAX_TOKENS;

    pub fn new() -> Self {
        let empty = CacheEntry { key_lo: 0, key_hi: 0, toks: [0; CACHE_MAX_TOKENS], ntok: 0 };
        let n = 1usize << cache_bits();
        let entries = vec![empty; n].into_boxed_slice();
        // Advise transparent huge pages for the table on Linux: the default
        // table is exactly one 2 MiB page, and probes are uniform-random, so
        // THP removes almost all TLB misses on the probe path. Advisory only
        // (errors ignored). NOTE: wired but not benchmarked locally — the
        // development machine is macOS, which has no madvise(MADV_HUGEPAGE).
        #[cfg(target_os = "linux")]
        {
            const MADV_HUGEPAGE: i32 = 14;
            unsafe extern "C" {
                fn madvise(addr: *mut core::ffi::c_void, length: usize, advice: i32) -> i32;
            }
            // SAFETY: the pointer/length describe the live `entries`
            // allocation; madvise(MADV_HUGEPAGE) does not alter contents.
            unsafe {
                madvise(
                    entries.as_ptr() as *mut core::ffi::c_void,
                    n * std::mem::size_of::<CacheEntry>(),
                    MADV_HUGEPAGE,
                );
            }
        }
        Self { entries, mask: n - 1, long: None }
    }

    /// Reset every entry to empty (for reuse under a different tokenizer).
    pub fn clear(&mut self) {
        let empty = CacheEntry { key_lo: 0, key_hi: 0, toks: [0; CACHE_MAX_TOKENS], ntok: 0 };
        self.entries.fill(empty);
        if let Some(l) = &mut self.long {
            l.clear();
        }
    }

    /// Longest piece the long-piece cache keys on.
    pub const LONG_KEY_MAX: usize = LONG_KEY_MAX;

    /// Look up a piece the inline table can't hold (see
    /// [`Self::insert_long`]); on hit, append its tokens and return true.
    #[inline]
    pub fn get_long(&self, piece: &[u8], out: &mut Vec<TokenId>) -> bool {
        match &self.long {
            Some(l) if long_cache_enabled() => l.get(piece, out),
            _ => false,
        }
    }

    /// Remember a piece of 1..=[`Self::LONG_KEY_MAX`] bytes that the inline
    /// table can't hold (too long, or too many tokens).
    pub fn insert_long(&mut self, piece: &[u8], toks: &[TokenId]) {
        debug_assert!(!piece.is_empty() && piece.len() <= LONG_KEY_MAX);
        if toks.is_empty() || toks.len() > u16::MAX as usize || !long_cache_enabled() {
            return;
        }
        self.long.get_or_insert_with(LongCache::new).insert(piece, toks);
    }

    #[inline(always)]
    fn slot(&self, lo: u64, hi: u64) -> usize {
        let h = (lo ^ 0x9E37_79B9_7F4A_7C15)
            .wrapping_mul(0xA076_1D64_78BD_642F)
            ^ hi.wrapping_mul(0xE703_7ED1_A0B4_28DB);
        ((h ^ (h >> 32)) as usize) & self.mask
    }

    /// Look up a piece; on hit, append its tokens to `out` and return true.
    ///
    /// Public for profiling harnesses; `encode_into` is the normal entry point.
    #[inline(always)]
    pub fn get(&self, bytes: &[u8], out: &mut Vec<TokenId>) -> bool {
        let (lo, hi) = key_words(bytes);
        self.get_with_key(lo, hi, out)
    }

    /// Profiling hook: build the canonical key words for a piece.
    #[doc(hidden)]
    #[inline(always)]
    pub fn key_of(bytes: &[u8]) -> (u64, u64) {
        key_words(bytes)
    }

    /// Probe with precomputed key words (profiling hook + batch path).
    #[doc(hidden)]
    #[inline(always)]
    pub fn get_with_key(&self, lo: u64, hi: u64, out: &mut Vec<TokenId>) -> bool {
        let mut i = self.slot(lo, hi);
        for _ in 0..CACHE_PROBES {
            let e = &self.entries[i];
            if e.key_lo == lo && e.key_hi == hi {
                // Emit via a fixed-width store: always copy all 3 slots, then
                // advance len by the real count. Avoids the variable-length
                // memcpy branch on the hottest path in the crate.
                out.reserve(CACHE_MAX_TOKENS);
                // SAFETY: reserve guarantees capacity for CACHE_MAX_TOKENS
                // more elements; ntok <= CACHE_MAX_TOKENS by construction,
                // and the first ntok slots are initialized token ids.
                unsafe {
                    let len = out.len();
                    std::ptr::copy_nonoverlapping(e.toks.as_ptr(), out.as_mut_ptr().add(len), CACHE_MAX_TOKENS);
                    out.set_len(len + e.ntok as usize);
                }
                return true;
            }
            if e.key_hi == 0 {
                return false;
            }
            i = (i + 1) & self.mask;
        }
        false
    }

    /// Insert a piece → token mapping (self-guards key/value size limits).
    ///
    /// Public for profiling harnesses; `encode_into` is the normal entry point.
    #[inline]
    pub fn insert(&mut self, bytes: &[u8], toks: &[TokenId]) {
        if bytes.is_empty() || bytes.len() > CACHE_KEY_MAX || toks.is_empty() || toks.len() > CACHE_MAX_TOKENS {
            return;
        }
        let (lo, hi) = key_words(bytes);
        self.insert_with_key(lo, hi, toks);
    }

    /// Insert with precomputed key words. `toks` must be non-empty and at
    /// most [`Self::MAX_TOKENS`] long (checked in debug builds).
    #[inline]
    pub fn insert_with_key(&mut self, lo: u64, hi: u64, toks: &[TokenId]) {
        debug_assert!(!toks.is_empty() && toks.len() <= CACHE_MAX_TOKENS);
        let home = self.slot(lo, hi);
        let mut i = home;
        let mut target = home;
        for _ in 0..CACHE_PROBES {
            let e = &self.entries[i];
            if e.key_hi == 0 || (e.key_lo == lo && e.key_hi == hi) {
                target = i;
                break;
            }
            i = (i + 1) & self.mask;
        }
        let e = &mut self.entries[target];
        e.key_lo = lo;
        e.key_hi = hi;
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
    fn test_key_words_within_matches_standalone() {
        // Every length 1..=15, at every offset of a small doc, including the
        // tail (< 16 bytes left, fallback path) — the contextual masked-load
        // key must equal the standalone key.
        let doc: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
        for start in 0..doc.len() {
            for len in 1..=CACHE_KEY_MAX {
                if start + len > doc.len() {
                    break;
                }
                let piece = &doc[start..start + len];
                assert_eq!(
                    key_words_within(&doc, piece),
                    key_words(piece),
                    "start {start} len {len}"
                );
            }
        }
        // A piece that is not a subslice of doc must fall back safely.
        let outside = vec![0xABu8; 7];
        assert_eq!(key_words_within(&doc, &outside), key_words(&outside));
        // All-0xFF piece: masking must not leak neighboring bytes.
        let doc2 = [0xFFu8; 32];
        for len in 1..=CACHE_KEY_MAX {
            assert_eq!(key_words_within(&doc2, &doc2[3..3 + len]), key_words(&doc2[3..3 + len]));
        }
    }

    #[test]
    fn test_encode_piece_into_doc_context_matches_encode() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // ab, abc
        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        // Build a doc and carve pieces out of it: single-token, multi-token,
        // repeats, a >15-byte piece, and pieces at the very end of the doc.
        let doc: Vec<u8> = b"abcabbacababcabcabcabcabcababcacab".to_vec();
        let ranges: Vec<(usize, usize)> = vec![
            (0, 3),   // abc — single token
            (3, 5),   // ab — single token
            (5, 8),   // bac — multi token
            (8, 11),  // aba
            (11, 14), (14, 17), (11, 14), // repeats
            (5, 25),  // 20 bytes — over the cache key limit
            (doc.len() - 2, doc.len()),   // tail: fallback key path
            (doc.len() - 1, doc.len()),   // last byte
            (7, 7),   // empty
        ];
        let mut cache = PretokenCache::new();
        for pass in 0..2 {
            for &(s, e) in &ranges {
                let piece = &doc[s..e];
                let expect = encoder.encode(piece);
                let mut got = Vec::new();
                encoder.encode_piece_into(&doc, piece, Some(&mut cache), &mut got);
                assert_eq!(got, expect, "pass {pass}, range {s}..{e}");
                // And uncached
                let mut got2 = Vec::new();
                encoder.encode_piece_into(&doc, piece, None, &mut got2);
                assert_eq!(got2, expect, "no-cache, range {s}..{e}");
            }
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

    /// Byte-complete vocab (all 256 single-byte base tokens) with a few
    /// merges. The merge list is self-consistent (every merge is reachable
    /// under classic lowest-rank-first merging), as real trained BPE vocabs
    /// are — an inconsistent list makes greedy-longest-match and canonical
    /// merge order legitimately diverge.
    fn byte_complete_encoder() -> BacktrackingBytePairEncoder {
        let base_tokens: Vec<Vec<u8>> = (0u16..256).map(|b| vec![b as u8]).collect();
        let a = b'a' as TokenId;
        let merges = vec![
            (a, a + 1),        // 256 "ab"
            (256, a + 2),      // 257 "abc"
            (b'l' as u32, b'l' as u32), // 258 "ll"
            (b'h' as u32, b'e' as u32), // 259 "he"
            (258, b'o' as u32), // 260 "llo"
            (259, 260),        // 261 "hello"
            (b' ' as u32, b't' as u32), // 262 " t"
            (262, 259),         // 263 " the" (" t" + "he")
        ];
        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        encoder
    }

    #[test]
    fn test_rank_merge_matches_backtracking() {
        let encoder = byte_complete_encoder();
        assert!(encoder.has_rank_merge());

        let cases: Vec<&[u8]> = vec![
            b"abc", b"ab", b"hello", b"hhello", b"llllll", b"aaabbb",
            b" the", b" the the", b"abcabcabc", b"xyz", b"\x00\xff\xfe",
        ];
        for text in cases {
            let mut daac = Vec::new();
            encoder.encode_sequential_into(text, &mut daac);
            let mut rank = Vec::new();
            encoder.encode_rank_merge(text, &mut rank);
            assert_eq!(rank, daac, "piece {:?}", text);
            let mut flat = Vec::new();
            encoder.encode_rank_merge_flat(text, &mut flat);
            assert_eq!(flat, daac, "flat probe, piece {:?}", text);
        }
    }

    #[test]
    fn test_rank_merge_fuzz_matches_backtracking() {
        let encoder = byte_complete_encoder();
        let mut state = 0x853C_49E6_748F_EA9Bu64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..5000 {
            let len = 1 + (next() as usize) % 40;
            let bytes: Vec<u8> = (0..len).map(|_| next() as u8).collect();
            let mut daac = Vec::new();
            encoder.encode_sequential_into(&bytes, &mut daac);
            let mut rank = Vec::new();
            encoder.encode_rank_merge(&bytes, &mut rank);
            assert_eq!(rank, daac, "fuzz bytes {:?}", bytes);
        }
    }

    #[test]
    fn test_merge_core_fuzz_matches_backtracking() {
        let encoder = byte_complete_encoder();
        assert!(encoder.has_merge_core());
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // Past SHORT_MAX (64) so both the multipass and queue loops run;
        // a small alphabet makes merges (and repeats) frequent.
        for _ in 0..5000 {
            let len = 1 + (next() as usize) % 160;
            let bytes: Vec<u8> = (0..len).map(|_| b"ab ehlot"[(next() % 8) as usize]).collect();
            let mut daac = Vec::new();
            encoder.encode_sequential_into(&bytes, &mut daac);
            let mut core = Vec::new();
            encoder.encode_merge_core(&bytes, &mut core);
            assert_eq!(core, daac, "fuzz bytes {:?}", String::from_utf8_lossy(&bytes));
        }
    }

    #[test]
    fn test_long_cache_matches_uncached() {
        // Pieces over 15 bytes and many-token short pieces go through the
        // long-piece cache; a warm cache must reproduce the uncached output.
        let encoder = byte_complete_encoder();
        let mut cache = PretokenCache::new();
        let pieces: Vec<Vec<u8>> = (0..400)
            .map(|i| {
                let n = 2 + (i * 7) % 300;
                (0..n).map(|j| b"ab ehlot\xff"[(i + j * j) % 9]).collect()
            })
            .collect();
        for round in 0..3 {
            for p in &pieces {
                let mut want = Vec::new();
                encoder.encode_into(p, None, &mut want);
                let mut got = vec![7u32]; // appends after existing content
                encoder.encode_into(p, Some(&mut cache), &mut got);
                assert_eq!(&got[1..], &want[..], "round {round} piece len {}", p.len());
            }
        }
    }

    #[test]
    fn test_long_cache_distinguishes_lengths_and_flushes() {
        let mut cache = PretokenCache::new();
        let a = vec![b'x'; 40];
        let b = vec![b'x'; 41];
        cache.insert_long(&a, &[1, 2]);
        cache.insert_long(&b, &[3]);
        let mut out = Vec::new();
        assert!(cache.get_long(&a, &mut out));
        assert!(cache.get_long(&b, &mut out));
        assert_eq!(out, vec![1, 2, 3]);
        // Fill past the arena budget: the cache flushes instead of growing,
        // and later inserts stay retrievable.
        let big = vec![b'y'; LONG_KEY_MAX];
        for i in 0..(LONG_ARENA_BYTES / LONG_KEY_MAX + 10) {
            let mut k = big.clone();
            k[..8].copy_from_slice(&(i as u64).to_le_bytes());
            cache.insert_long(&k, &[i as u32]);
            let l = cache.long.as_ref().unwrap();
            assert!(l.bytes.len() <= LONG_ARENA_BYTES);
            let mut o = Vec::new();
            assert!(cache.get_long(&k, &mut o));
            assert_eq!(o, vec![i as u32]);
        }
        cache.clear();
        assert!(!cache.get_long(&b, &mut Vec::new()));
    }

    #[test]
    fn test_rank_merge_disabled_for_non_byte_vocab() {
        // 3-letter vocab: not byte-complete, rank merge must be disabled
        // and encode_into must still produce correct output via DAAC.
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)];
        let (encoder, _) = BacktrackingBytePairEncoder::from_merges(&merges, &base_tokens);
        assert!(!encoder.has_rank_merge());
        let mut out = Vec::new();
        encoder.encode_into(b"abcab", None, &mut out);
        assert_eq!(out, encoder.encode(b"abcab"));
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
