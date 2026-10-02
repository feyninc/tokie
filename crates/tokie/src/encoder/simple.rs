//! Optimized Simple BPE encoder with single-byte fast path.
//!
//! Key optimizations:
//! 1. Single-byte fast path: 21.5% of pieces are 1 byte - use direct array lookup
//! 2. Early exit for single tokens (88.9% of pieces)
//! 3. foldhash + packed u64 keys for fast hash lookups
//! 4. Pretoken cache (shared with Backtracking) in front of the merge loop
//! 5. Merge loop over a pair table of u32 codes (`rank << 1 | SAFE`; dense
//!    grid for low ids, 8-byte open-addressed entries otherwise, merged ids
//!    looked up by rank), with per-pair codes carried between rounds and
//!    all-occurrence merging for SAFE pairs; long pieces use a two-tier
//!    (sorted cold + heap hot) queue over a linked list.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use foldhash::HashMap as FoldHashMap;
use smallvec::SmallVec;

use super::backtracking::{key_words_within, PretokenCache};
use crate::types::TokenId;

/// Maximum token length for the whole-piece early exit. Unbounded: a piece
/// that is itself a vocab token encodes as that token (HF `ignore_merges`,
/// set by Llama-3, whose tiktoken-derived vocab has long tokens the merges
/// alone never rebuild, e.g. " принимать" (19 B) -> one token, not 3).
const MAX_CACHED_TOKEN_LEN: usize = usize::MAX;

/// Pack two u32 token IDs into a single u64 key for faster hashing.
#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

/// Pair code: `rank << 1 | SAFE`. Codes order by rank (ranks are unique
/// per pair, so equal codes mean the same pair); `NO_MERGE` sorts last.
const NO_MERGE: u32 = u32::MAX;
/// Set when every merge that takes the merged token as an operand has a
/// larger rank. Merging all non-overlapping occurrences of such a pair
/// left-to-right in one pass then equals merging them one at a time
/// (no pair the merges create can outrank it).
const SAFE_BIT: u32 = 1;
/// Pairs with both ids below this hit a dense grid instead of the hash.
const DENSE_BOUND: usize = 512;
/// Pieces up to this many bytes use the stack multipass merge loop;
/// longer ones use the queue.
const SHORT_MAX: usize = 64;
/// Dead-symbol marker in the long-piece linked list.
const DEAD: u32 = u32::MAX;

/// Flat pair table: a dense code grid for small ids, and 8-byte
/// open-addressed entries `key << code_bits | code` (key = `left <<
/// id_bits | right`, never 0 outside the grid, so 0 marks empty) for the
/// rest. The merged id lives in a rank-indexed side array, read only when a
/// pair actually merges. Keeping entries to 8 bytes halves the table's
/// cache footprint (the merge loop is bound on these lookups).
#[derive(Clone)]
struct PairTable {
    dense: Box<[u32]>,
    slots: Box<[u64]>,
    shift: u32,
    id_bits: u32,
    code_bits: u32,
    merged_by_rank: Box<[TokenId]>,
}

impl PairTable {
    /// `None` when ids/ranks are too wide to pack (callers then use the
    /// reference loop).
    fn build(pair_lookup: &FoldHashMap<u64, (TokenId, u32)>) -> Option<Self> {
        let mut max_id = 0u32;
        let mut max_rank = 0u32;
        for (&k, &(m, rank)) in pair_lookup {
            max_id = max_id.max((k >> 32) as u32).max(k as u32).max(m);
            max_rank = max_rank.max(rank);
        }
        let bits = |v: u64| 64 - v.leading_zeros();
        let id_bits = bits(max_id as u64).max(1);
        let code_bits = bits(((max_rank as u64) << 1) | 1);
        if max_rank >= (1 << 30) || 2 * id_bits + code_bits > 64 {
            return None;
        }

        // Lowest rank of any merge taking each token as an operand.
        let mut min_use = vec![u32::MAX; max_id as usize + 1];
        for (&k, &(_, rank)) in pair_lookup {
            for id in [(k >> 32) as usize, k as u32 as usize] {
                min_use[id] = min_use[id].min(rank);
            }
        }
        let mut merged_by_rank = vec![0 as TokenId; max_rank as usize + 1].into_boxed_slice();
        let mut dense = vec![NO_MERGE; DENSE_BOUND * DENSE_BOUND].into_boxed_slice();
        let cap = (pair_lookup.len() * 2).next_power_of_two().max(16);
        let mut slots = vec![0u64; cap].into_boxed_slice();
        let shift = 64 - cap.trailing_zeros();
        for (&k, &(merged, rank)) in pair_lookup {
            merged_by_rank[rank as usize] = merged;
            let code = (rank << 1) | if min_use[merged as usize] > rank { SAFE_BIT } else { 0 };
            let (l, r) = ((k >> 32) as usize, k as u32 as usize);
            if l < DENSE_BOUND && r < DENSE_BOUND {
                dense[l * DENSE_BOUND + r] = code;
            } else {
                let key = ((l as u64) << id_bits) | r as u64;
                let mut i = Self::home(key, shift);
                while slots[i] != 0 {
                    i = (i + 1) & (cap - 1);
                }
                slots[i] = (key << code_bits) | code as u64;
            }
        }
        Some(Self { dense, slots, shift, id_bits, code_bits, merged_by_rank })
    }

    #[inline(always)]
    fn home(key: u64, shift: u32) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> shift) as usize
    }

    /// Pair code for `(left, right)`, or `NO_MERGE`.
    #[inline(always)]
    fn get(&self, left: TokenId, right: TokenId) -> u32 {
        if ((left | right) as usize) < DENSE_BOUND {
            return self.dense[left as usize * DENSE_BOUND + right as usize];
        }
        if (left | right) >> self.id_bits != 0 {
            return NO_MERGE;
        }
        let key = ((left as u64) << self.id_bits) | right as u64;
        let mask = self.slots.len() - 1;
        let mut i = Self::home(key, self.shift);
        loop {
            let e = self.slots[i];
            if e >> self.code_bits == key {
                return (e & ((1u64 << self.code_bits) - 1)) as u32;
            }
            if e == 0 {
                return NO_MERGE;
            }
            i = (i + 1) & mask;
        }
    }

    /// Merged token for a (real) pair code.
    #[inline(always)]
    fn merged(&self, code: u32) -> TokenId {
        self.merged_by_rank[(code >> 1) as usize]
    }
}

/// Reused per-thread scratch for the long-piece queue.
#[derive(Default)]
struct LongScratch {
    toks: Vec<TokenId>,
    prev: Vec<u32>,
    next: Vec<u32>,
    cold: Vec<u64>,
    hot: BinaryHeap<Reverse<u64>>,
}

thread_local! {
    static LONG_SCRATCH: RefCell<LongScratch> = RefCell::new(LongScratch::default());
}

/// Optimized BPE encoder with single-byte fast path.
///
/// Adds direct array lookup for single-byte inputs (21.5% of pieces)
/// before falling back to hash-based early exit lookup.
#[derive(Clone)]
pub struct BytePairEncoder {
    /// Maps (left_token, right_token) -> (merged_token, merge_rank).
    /// Uses foldhash for fast lookups.
    pair_lookup: FoldHashMap<u64, (TokenId, u32)>,

    /// Maps byte value -> token ID (for base tokens).
    byte_lut: [TokenId; 256],

    /// Maps byte sequence -> token ID for early exit.
    /// Includes tokens up to MAX_CACHED_TOKEN_LEN bytes (all of them).
    /// Uses foldhash for fast lookups.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// `pair_lookup` flattened for the merge loop.
    /// `None` for vocabularies too wide to pack (reference loop).
    pairs: Option<PairTable>,

    /// Total vocabulary size.
    vocab_size: usize,

    /// Number of base (single-byte) tokens.
    num_base_tokens: usize,
}

impl BytePairEncoder {
    /// Create a new encoder from merge rules.
    pub fn from_merges(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
    ) -> (Self, Vec<Vec<u8>>) {
        Self::from_merges_with_added(merges, base_tokens, &[])
    }

    /// Create encoder from merge rules with added tokens.
    pub fn from_merges_with_added(
        merges: &[(TokenId, TokenId)],
        base_tokens: &[Vec<u8>],
        added_tokens: &[(u32, Vec<u8>)],
    ) -> (Self, Vec<Vec<u8>>) {
        let mut token_bytes: Vec<Vec<u8>> = base_tokens.to_vec();
        let mut pair_lookup = FoldHashMap::default();

        // Build byte -> token mapping
        let mut byte_lut = [0u32; 256];
        for (token_id, bytes) in base_tokens.iter().enumerate() {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = token_id as TokenId;
            }
        }
        // Fallback for unmapped bytes
        for (i, token) in byte_lut.iter_mut().enumerate() {
            if *token == 0 && i < base_tokens.len() {
                if base_tokens.get(i).is_some_and(|b| b.len() == 1 && b[0] == i as u8) {
                    *token = i as TokenId;
                }
            }
        }

        // Handle added tokens interleaved with merges
        let mut added_sorted: Vec<_> = added_tokens.to_vec();
        added_sorted.sort_by_key(|(id, _)| *id);
        let mut added_iter = added_sorted.into_iter().peekable();

        for (merge_index, &(left, right)) in merges.iter().enumerate() {
            let next_id = token_bytes.len() as TokenId;

            // Insert any added tokens that come before this merge
            while let Some(&(added_id, _)) = added_iter.peek() {
                if added_id <= next_id {
                    let (_, bytes) = added_iter.next().unwrap();
                    token_bytes.push(bytes);
                } else {
                    break;
                }
            }

            let new_id = token_bytes.len() as TokenId;
            pair_lookup.insert(pack_pair(left, right), (new_id, merge_index as u32));

            let mut bytes = token_bytes[left as usize].clone();
            bytes.extend_from_slice(&token_bytes[right as usize]);
            token_bytes.push(bytes);
        }

        // Append remaining added tokens
        for (_, bytes) in added_iter {
            token_bytes.push(bytes);
        }

        // Build token_cache for early exit optimization (foldhash for speed)
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let vocab_size = token_bytes.len();
        let num_base_tokens = base_tokens.len();

        let encoder = Self {
            pairs: PairTable::build(&pair_lookup),
            pair_lookup,
            byte_lut,
            token_cache,
            vocab_size,
            num_base_tokens,
        };

        (encoder, token_bytes)
    }

    /// Create encoder from a complete vocabulary and merge rules.
    ///
    /// For tokenizers (like LLaMA 3) where vocab has pre-assigned IDs.
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
    ) -> (Self, Vec<Vec<u8>>) {
        let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, bytes)| bytes.clone()).collect();

        // Build byte -> token mapping by scanning ALL tokens.
        // For SentencePiece models, byte tokens (<0x00>, etc.) can be scattered
        // throughout the vocab, not just at the beginning.
        let mut byte_lut = [u32::MAX; 256];
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() == 1 {
                let byte_val = bytes[0] as usize;
                // Only set if not already mapped (prefer earlier token IDs)
                if byte_lut[byte_val] == u32::MAX {
                    byte_lut[byte_val] = token_id as TokenId;
                }
            }
        }

        // Build bytes -> token ID lookup for ALL tokens (used in construction)
        let all_bytes_to_id: FoldHashMap<Vec<u8>, TokenId> = vocab
            .iter()
            .map(|(id, bytes)| (bytes.clone(), *id))
            .collect();

        // Build pair_lookup with merge ranks (pack (u32, u32) into u64 for faster hashing)
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right)) in merges.iter().enumerate() {
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend_from_slice(&token_bytes[right as usize]);

            if let Some(&merged_id) = all_bytes_to_id.get(&merged_bytes) {
                pair_lookup
                    .entry(pack_pair(left, right))
                    .or_insert((merged_id, merge_index as u32));
            }
        }

        // Build token_cache for early exit (foldhash for speed)
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        let encoder = Self {
            pairs: PairTable::build(&pair_lookup),
            pair_lookup,
            byte_lut,
            token_cache,
            vocab_size: vocab.len(),
            num_base_tokens,
        };

        (encoder, token_bytes)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the number of base tokens.
    pub fn num_base_tokens(&self) -> usize {
        self.num_base_tokens
    }

    /// Get a reference to the pair lookup table.
    pub fn pair_lookup(&self) -> &FoldHashMap<u64, (TokenId, u32)> {
        &self.pair_lookup
    }

    /// Check if two tokens can appear adjacent in a valid BPE encoding.
    ///
    /// Returns false if there exists a merge that would combine these tokens.
    /// Note: This is a simplified check compared to BacktrackingBytePairEncoder.
    #[inline]
    pub fn is_valid_pair(&self, token1: TokenId, token2: TokenId) -> bool {
        // If there's a merge for this pair, they shouldn't appear adjacent
        !self.pair_lookup.contains_key(&pack_pair(token1, token2))
    }

    /// Reconstruct encoder from serialized parts with pre-computed merged IDs and lookups.
    ///
    /// Used during deserialization to rebuild the encoder from saved data.
    /// - `merged_id` is pre-computed during serialization
    /// - `byte_lut` and `token_cache` are pre-built from decoder data (single copy)
    pub fn from_parts(
        merges: &[(TokenId, TokenId, TokenId)], // (left, right, merged_id)
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        vocab_size: usize,
        num_base_tokens: usize,
    ) -> Self {
        // Build pair_lookup directly from pre-computed merged IDs - O(num_merges)
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right, merged_id)) in merges.iter().enumerate() {
            pair_lookup.insert(pack_pair(left, right), (merged_id, merge_index as u32));
        }

        Self {
            pairs: PairTable::build(&pair_lookup),
            pair_lookup,
            byte_lut,
            token_cache,
            vocab_size,
            num_base_tokens,
        }
    }

    /// Encode bytes into BPE tokens.
    ///
    /// Optimizations:
    /// 1. Single-byte fast path: direct array lookup (no hashing)
    /// 2. Early exit: hash lookup for known tokens
    /// 3. Flat-table merge loop (see [`Self::merge_short`])
    #[inline]
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encode_piece_into(text, text, None, &mut out);
        out
    }

    /// Append the encoding of one pretokenized piece to `out`, resolving
    /// repeated pieces through `cache` when given.
    #[inline]
    pub fn encode_into(&self, text: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        self.encode_piece_into(text, text, cache, out)
    }

    /// Like [`Self::encode_into`] for a `piece` that is a subslice of `doc`
    /// (lets the cache key be built with one masked load).
    #[inline]
    pub fn encode_piece_into(&self, doc: &[u8], piece: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        match piece.len() {
            0 => return,
            1 => return out.push(self.byte_lut[piece[0] as usize]),
            _ => {}
        }
        if let Some(c) = cache {
            if piece.len() <= PretokenCache::KEY_MAX {
                let (lo, hi) = key_words_within(doc, piece);
                if c.get_with_key(lo, hi, out) {
                    return;
                }
                return self.encode_cache_miss(piece, lo, hi, c, out);
            }
            if piece.len() <= PretokenCache::LONG_KEY_MAX {
                return self.encode_long_cached(piece, c, out);
            }
        }
        self.encode_uncached(piece, out)
    }

    /// Short-piece miss; results too many tokens for an inline entry go to
    /// the long-piece cache.
    #[inline(never)]
    fn encode_cache_miss(&self, text: &[u8], lo: u64, hi: u64, cache: &mut PretokenCache, out: &mut Vec<TokenId>) {
        if cache.get_long(text, out) {
            return;
        }
        let start = out.len();
        self.encode_uncached(text, out);
        let toks = &out[start..];
        if toks.len() <= PretokenCache::MAX_TOKENS {
            if !toks.is_empty() {
                cache.insert_with_key(lo, hi, toks);
            }
        } else {
            cache.insert_long(text, toks);
        }
    }

    #[inline(never)]
    fn encode_long_cached(&self, text: &[u8], cache: &mut PretokenCache, out: &mut Vec<TokenId>) {
        if cache.get_long(text, out) {
            return;
        }
        let start = out.len();
        self.encode_uncached(text, out);
        cache.insert_long(text, &out[start..]);
    }

    /// Encode a piece of 2+ bytes without the pretoken cache.
    #[inline(never)]
    fn encode_uncached(&self, text: &[u8], out: &mut Vec<TokenId>) {
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                return out.push(token_id);
            }
        }
        match &self.pairs {
            Some(pairs) if text.len() <= SHORT_MAX => merge_short(pairs, &self.byte_lut, text, out),
            Some(pairs) => merge_long(pairs, &self.byte_lut, text, out),
            None => out.extend(self.encode_reference(text)),
        }
    }

    /// The original O(n²) merge loop, kept as the equivalence oracle for
    /// the flat-table paths.
    #[doc(hidden)]
    pub fn encode_reference(&self, text: &[u8]) -> Vec<TokenId> {
        if text.is_empty() {
            return Vec::new();
        }

        // OPTIMIZATION 1: Single-byte fast path (21.5% of pieces)
        // Direct array lookup - no hashing needed!
        if text.len() == 1 {
            return vec![self.byte_lut[text[0] as usize]];
        }

        // OPTIMIZATION 2: Early exit if input is already a single token
        // This handles most of the remaining 88.9% of pretokenized pieces
        if text.len() <= MAX_CACHED_TOKEN_LEN {
            if let Some(&token_id) = self.token_cache.get(text) {
                return vec![token_id];
            }
        }

        // Initialize with byte tokens using SmallVec (stack allocation for ≤16 tokens)
        let mut tokens: SmallVec<[TokenId; 16]> = text
            .iter()
            .map(|&b| self.byte_lut[b as usize])
            .collect();

        let mut len = tokens.len();

        // Merge until no more merges possible
        while len > 1 {
            // Find the lowest-rank merge
            let mut best_rank = u32::MAX;
            let mut best_pos = usize::MAX;
            let mut best_merged = 0;

            for i in 0..len - 1 {
                if let Some(&(merged, rank)) = self.pair_lookup.get(&pack_pair(tokens[i], tokens[i + 1])) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = i;
                        best_merged = merged;
                    }
                }
            }

            if best_pos == usize::MAX {
                break; // No more merges
            }

            // Apply the merge: replace pair with merged token
            tokens[best_pos] = best_merged;
            // Shift remaining tokens left by one
            tokens.copy_within(best_pos + 2..len, best_pos + 1);
            len -= 1;
        }

        tokens.truncate(len);
        tokens.into_vec()
    }
}

/// The flat-table merge loops over a byte-complete vocab, for encoders
/// whose merge order is known up front. The Backtracking encoder uses it on
/// its cache-miss path with rank = merged id (its canonical order).
#[derive(Clone)]
pub(crate) struct MergeCore {
    pairs: PairTable,
    byte_lut: [TokenId; 256],
}

impl MergeCore {
    /// `pair_lookup` maps packed pairs to `(merged, rank)`; ranks must be
    /// unique per pair. `None` when the table can't be packed.
    pub(crate) fn build(pair_lookup: &FoldHashMap<u64, (TokenId, u32)>, byte_lut: [TokenId; 256]) -> Option<Self> {
        Some(Self { pairs: PairTable::build(pair_lookup)?, byte_lut })
    }

    /// Encode a piece of 1+ bytes starting from per-byte base tokens.
    #[inline]
    pub(crate) fn encode(&self, text: &[u8], out: &mut Vec<TokenId>) {
        match text.len() {
            0 => {}
            1 => out.push(self.byte_lut[text[0] as usize]),
            n if n <= SHORT_MAX => merge_short(&self.pairs, &self.byte_lut, text, out),
            _ => merge_long(&self.pairs, &self.byte_lut, text, out),
        }
    }
}

/// Multipass merge for short pieces: each round takes the lowest pair
/// code (lowest rank, leftmost on ties; equal ranks are the same
/// pair). Per-pair codes are carried between rounds so only pairs next
/// to a merge are looked up again. A SAFE winner merges every
/// non-overlapping occurrence in the same round.
fn merge_short(pairs: &PairTable, byte_lut: &[TokenId; 256], text: &[u8], out: &mut Vec<TokenId>) {
    let n = text.len();
    debug_assert!((2..=SHORT_MAX).contains(&n));
    let mut toks = [0 as TokenId; SHORT_MAX];
    let mut pv = [NO_MERGE; SHORT_MAX];
    for (t, &b) in toks.iter_mut().zip(text) {
        *t = byte_lut[b as usize];
    }
    for i in 0..n - 1 {
        pv[i] = pairs.get(toks[i], toks[i + 1]);
    }
    let mut len = n;
    while len > 1 {
        let mut best = NO_MERGE;
        let mut pos = 0;
        for (i, &v) in pv[..len - 1].iter().enumerate() {
            if v < best {
                best = v;
                pos = i;
            }
        }
        if best == NO_MERGE {
            break;
        }
        let merged = pairs.merged(best);
        if best & SAFE_BIT != 0 {
            // In-place compaction from `pos` (the leftmost occurrence).
            // Unchanged neighbours keep their pair value; pairs touching
            // a merged symbol are looked up. Writes to pv[w - 1] never
            // pass the read cursor (w <= r).
            let mut w = pos;
            let mut r = pos;
            let mut prev_merged = false;
            while r < len {
                if r + 1 < len && pv[r] == best {
                    toks[w] = merged;
                    if w > 0 {
                        pv[w - 1] = pairs.get(toks[w - 1], merged);
                    }
                    r += 2;
                    prev_merged = true;
                } else {
                    let t = toks[r];
                    toks[w] = t;
                    if prev_merged {
                        pv[w - 1] = pairs.get(toks[w - 1], t);
                    } else if w > 0 {
                        pv[w - 1] = pv[r - 1];
                    }
                    r += 1;
                    prev_merged = false;
                }
                w += 1;
            }
            len = w;
        } else {
            toks[pos] = merged;
            if pos + 2 < len {
                toks.copy_within(pos + 2..len, pos + 1);
                pv.copy_within(pos + 2..len - 1, pos + 1);
            }
            len -= 1;
            if pos > 0 {
                pv[pos - 1] = pairs.get(toks[pos - 1], merged);
            }
            if pos + 1 < len {
                pv[pos] = pairs.get(merged, toks[pos + 1]);
            }
        }
    }
    out.extend_from_slice(&toks[..len]);
}

/// Queue merge for long pieces: symbols in a linked list, candidate
/// pairs keyed `code << 32 | left position`. Initial pairs sit in a
/// sorted cold array; pairs created by merges go to a hot heap; the
/// smaller head wins. Stale entries are dropped on pop (left symbol
/// dead, no right neighbour, or the current pair's rank differs; ranks
/// identify pairs uniquely).
fn merge_long(pairs: &PairTable, byte_lut: &[TokenId; 256], text: &[u8], out: &mut Vec<TokenId>) {
    LONG_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let LongScratch { toks, prev, next, cold, hot } = &mut *s;
        let n = text.len();
        let n32 = n as u32;
        toks.clear();
        toks.extend(text.iter().map(|&b| byte_lut[b as usize]));
        prev.clear();
        prev.extend((0..n32).map(|i| i.wrapping_sub(1)));
        next.clear();
        next.extend(1..=n32);
        cold.clear();
        hot.clear();
        for i in 0..n - 1 {
            let v = pairs.get(toks[i], toks[i + 1]);
            if v != NO_MERGE {
                cold.push(((v as u64) << 32) | i as u64);
            }
        }
        cold.sort_unstable();
        let mut ci = 0;
        loop {
            let key = match (cold.get(ci), hot.peek()) {
                (Some(&c), Some(&Reverse(h))) if h < c => hot.pop().unwrap().0,
                (Some(&c), _) => {
                    ci += 1;
                    c
                }
                (None, Some(_)) => hot.pop().unwrap().0,
                (None, None) => break,
            };
            let pos = key as u32 as usize;
            let nx = next[pos];
            // Dead symbols have next == DEAD; the last one has next == n.
            if nx >= n32 {
                continue;
            }
            let code = (key >> 32) as u32;
            if pairs.get(toks[pos], toks[nx as usize]) != code {
                continue;
            }
            let merged = pairs.merged(code);
            toks[pos] = merged;
            let nn = next[nx as usize];
            next[nx as usize] = DEAD;
            next[pos] = nn;
            if nn < n32 {
                prev[nn as usize] = pos as u32;
                let v = pairs.get(merged, toks[nn as usize]);
                if v != NO_MERGE {
                    hot.push(Reverse(((v as u64) << 32) | pos as u64));
                }
            }
            let pp = prev[pos];
            if pp < n32 {
                let v = pairs.get(toks[pp as usize], merged);
                if v != NO_MERGE {
                    hot.push(Reverse(((v as u64) << 32) | pp as u64));
                }
            }
        }
        let mut i = 0u32;
        while i < n32 {
            out.push(toks[i as usize]);
            i = next[i as usize];
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::VocabDecoder;

    #[test]
    fn test_encode_basic() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = VocabDecoder::new(token_bytes);

        let encoded = encoder.encode(b"abc");
        assert_eq!(encoded, vec![4]); // abc

        assert_eq!(decoder.decode(&encoded), b"abc");
    }

    #[test]
    fn test_single_byte_fast_path() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1)];

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // Single byte should use fast path (direct array lookup)
        assert_eq!(encoder.encode(b"a"), vec![0]);
        assert_eq!(encoder.encode(b"b"), vec![1]);
        assert_eq!(encoder.encode(b"c"), vec![2]);
    }

    #[test]
    fn test_early_exit_multi_byte_token() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c']];
        let merges = vec![(0, 1), (3, 2)]; // a+b->ab(3), ab+c->abc(4)

        let (encoder, _) = BytePairEncoder::from_merges(&merges, &base_tokens);

        // "ab" is token 3, should use early exit
        assert_eq!(encoder.encode(b"ab"), vec![3]);

        // "abc" is token 4, should use early exit
        assert_eq!(encoder.encode(b"abc"), vec![4]);
    }

    #[test]
    fn test_early_exit_long_vocab_token() {
        // Llama-3 (`ignore_merges`): a piece that is itself a vocab token is
        // that token even when longer than 16 bytes and unreachable by merges
        // (" принимать", 19 B -> 127246, not 3 tokens).
        let long = vec![b'a'; 20];
        let vocab = vec![(0, vec![b'a']), (1, vec![b'a', b'a']), (2, long.clone())];
        let (encoder, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &[(0, 0)], 1);
        assert_eq!(encoder.encode(&long), vec![2]);
        assert_eq!(encoder.encode_reference(&long), vec![2]);
        assert_eq!(encoder.encode(&long[..19]), encoder.encode_reference(&long[..19]));
    }

    #[test]
    fn test_encode_roundtrip() {
        let base_tokens = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'd']];
        let merges = vec![(0, 1), (2, 3), (4, 5)];

        let (encoder, token_bytes) = BytePairEncoder::from_merges(&merges, &base_tokens);
        let decoder = VocabDecoder::new(token_bytes);

        for text in [b"abcd".as_slice(), b"ab", b"cd", b"abcdabcd", b"a", b""] {
            let encoded = encoder.encode(text);
            let decoded = decoder.decode(&encoded);
            assert_eq!(decoded, text);
        }
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Random vocab over a small alphabet. Merges are shuffled when
    /// `shuffle`, so merged tokens can be operands of lower-ranked merges
    /// (non-SAFE pairs) and one token can come from several pairs.
    fn random_vocab(rng: &mut Rng, alphabet: usize, n_merges: usize, shuffle: bool) -> (Vec<(u32, Vec<u8>)>, Vec<(u32, u32)>) {
        let mut vocab: Vec<(u32, Vec<u8>)> = (0..256u32).map(|b| (b, vec![b as u8])).collect();
        let mut merges = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let base = b'a' as usize;
        for _ in 0..n_merges * 4 {
            if merges.len() == n_merges {
                break;
            }
            let pick = |rng: &mut Rng, vocab: &Vec<(u32, Vec<u8>)>| -> u32 {
                if vocab.len() == 256 || rng.below(3) == 0 {
                    (base + rng.below(alphabet)) as u32
                } else {
                    256 + rng.below(vocab.len() - 256) as u32
                }
            };
            let (l, r) = (pick(rng, &vocab), pick(rng, &vocab));
            if !seen.insert((l, r)) {
                continue;
            }
            let mut bytes = vocab[l as usize].1.clone();
            bytes.extend_from_slice(&vocab[r as usize].1);
            if bytes.len() > 24 {
                continue;
            }
            if !vocab.iter().any(|(_, b)| *b == bytes) {
                vocab.push((vocab.len() as u32, bytes));
            }
            merges.push((l, r));
        }
        if shuffle {
            for i in (1..merges.len()).rev() {
                merges.swap(i, rng.below(i + 1));
            }
        }
        (vocab, merges)
    }

    #[test]
    fn test_fast_paths_match_reference_on_random_vocabs() {
        let mut rng = Rng(0x2545_F491_4F6C_DD1D);
        for round in 0..60 {
            let alphabet = 2 + round % 4;
            let (vocab, merges) = random_vocab(&mut rng, alphabet, 40 + round * 5, round % 2 == 0);
            let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256);
            let (enc2, _) = BytePairEncoder::from_merges(&merges.iter().copied().filter(|&(l, r)| l < 256 && r < 256).collect::<Vec<_>>(), &vocab[..256].iter().map(|(_, b)| b.clone()).collect::<Vec<_>>());
            let mut caches = [PretokenCache::new(), PretokenCache::new()];
            for _ in 0..400 {
                let len = if rng.below(8) == 0 { rng.below(400) } else { rng.below(40) };
                let text: Vec<u8> = (0..len).map(|_| b'a' + rng.below(alphabet) as u8).collect();
                for (e, cache) in [&enc, &enc2].into_iter().zip(caches.iter_mut()) {
                    let want = e.encode_reference(&text);
                    assert_eq!(e.encode(&text), want, "round {round} text {:?}", String::from_utf8_lossy(&text));
                    for _ in 0..2 {
                        let mut got = Vec::new();
                        e.encode_into(&text, Some(&mut *cache), &mut got);
                        assert_eq!(got, want);
                    }
                }
            }
        }
    }

    #[test]
    fn test_from_parts_matches() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let (vocab, merges) = random_vocab(&mut rng, 3, 80, true);
        let (enc, token_bytes) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256);
        let parts: Vec<(u32, u32, u32)> = {
            let mut v: Vec<_> = enc.pair_lookup().iter().map(|(&k, &(m, r))| (r, (k >> 32) as u32, k as u32, m)).collect();
            v.sort();
            v.into_iter().map(|(_, l, r, m)| (l, r, m)).collect()
        };
        let mut token_cache = FoldHashMap::default();
        for (id, b) in token_bytes.iter().enumerate() {
            if b.len() <= MAX_CACHED_TOKEN_LEN {
                token_cache.insert(b.clone(), id as TokenId);
            }
        }
        let rebuilt = BytePairEncoder::from_parts(&parts, enc.byte_lut, token_cache, enc.vocab_size(), 256);
        for _ in 0..500 {
            let text: Vec<u8> = (0..rng.below(120)).map(|_| b'a' + rng.below(3) as u8).collect();
            assert_eq!(rebuilt.encode(&text), rebuilt.encode_reference(&text));
        }
    }
}
