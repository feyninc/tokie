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
//! 6. Character fold (`fold.rs`): complete UTF-8 characters whose byte
//!    assembly is provably unaffected by their neighbours start the merge
//!    loop as one symbol.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::sync::OnceLock;
use std::collections::BinaryHeap;

use foldhash::HashMap as FoldHashMap;
use smallvec::SmallVec;

use super::backtracking::{cache_seeds, key_words_within, CacheSeed, PretokenCache, WholeTable};
use super::fold::{self, CharFold, FoldStats};
use crate::types::TokenId;

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
/// Pieces of up to this many *symbols* use the stack multipass merge loop;
/// longer ones use the queue. Without a character fold, symbols = bytes.
const SHORT_MAX: usize = 64;
/// [fold] Longest piece (bytes) whose folded symbols are seeded into a
/// stack buffer, so pieces of `SHORT_MAX+1..=` this many bytes that fold to
/// at most `SHORT_MAX` symbols still take the multipass loop.
const SHORT_FOLD_MAX_BYTES: usize = 3 * SHORT_MAX;
/// Most symbols of a non-ASCII piece merged by `merge_short` (O(n²) scan);
/// longer ones take `merge_long`'s queue. Non-ASCII pieces need many more
/// merge rounds per symbol than English words.
const NONASCII_SHORT_MAX: usize = 32;
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
        // Load 0.25..0.5: a 1.3x table (load ~0.77) measured 21-29% slower
        // on CJK, whose many non-merging pairs walk long probe chains.
        let cap = (pair_lookup.len() * 2).next_power_of_two().max(16);
        let mut slots = vec![0u64; cap].into_boxed_slice();
        for (&k, &(merged, rank)) in pair_lookup {
            merged_by_rank[rank as usize] = merged;
            let code = (rank << 1) | if min_use[merged as usize] > rank { SAFE_BIT } else { 0 };
            let (l, r) = ((k >> 32) as usize, k as u32 as usize);
            if l < DENSE_BOUND && r < DENSE_BOUND {
                dense[l * DENSE_BOUND + r] = code;
            } else {
                let key = ((l as u64) << id_bits) | r as u64;
                let mut i = Self::home(key, cap);
                while slots[i] != 0 {
                    i += 1;
                    if i == cap {
                        i = 0;
                    }
                }
                slots[i] = (key << code_bits) | code as u64;
            }
        }
        Some(Self { dense, slots, id_bits, code_bits, merged_by_rank })
    }

    /// Home slot: multiply-high of the mixed key onto `0..cap`.
    #[inline(always)]
    fn home(key: u64, cap: usize) -> usize {
        ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u128 * cap as u128) >> 64) as usize
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
        let cap = self.slots.len();
        let mut i = Self::home(key, cap);
        loop {
            let e = self.slots[i];
            if e >> self.code_bits == key {
                return (e & ((1u64 << self.code_bits) - 1)) as u32;
            }
            if e == 0 {
                return NO_MERGE;
            }
            i += 1;
            if i == cap {
                i = 0;
            }
        }
    }

    /// Merged token for a (real) pair code. `ID`: rank == merged id for
    /// every pair (Backtracking cores), so the side array is skipped.
    #[inline(always)]
    fn merged<const ID: bool>(&self, code: u32) -> TokenId {
        if ID {
            code >> 1
        } else {
            self.merged_by_rank[(code >> 1) as usize]
        }
    }
}

/// Reused per-thread scratch for the long-piece queue.
#[derive(Default)]
struct LongScratch {
    toks: Vec<TokenId>,
    prev: Vec<u32>,
    next: Vec<u32>,
    /// Current pair code of each symbol with its right neighbour
    /// (`NO_MERGE` for dead/last symbols).
    code: Vec<u32>,
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

    /// Maps byte sequence -> token ID for the whole-piece early exit (any
    /// length). With `ignore_merges` it holds every token: a piece that is
    /// itself a vocab token encodes as that token (HF `ignore_merges`, set by
    /// Llama-3 / Mistral-Nemo, whose vocabs have long tokens the merges alone
    /// never rebuild, e.g. " принимать" (19 B) -> one token, not 3). Without
    /// it, only tokens the merge loop itself rebuilds from their bytes, so the
    /// shortcut never changes the result (Qwen3.5 has 201 unreachable tokens).
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// HF `model.ignore_merges`; see `token_cache`.
    ignore_merges: bool,

    /// [`PretokenCache::seed`] entries (from `token_cache`), built on first
    /// use.
    seeds: OnceLock<Box<[CacheSeed]>>,

    /// [`WholeTable`] of `token_cache`, built on first use.
    whole: OnceLock<WholeTable>,

    /// `pair_lookup` flattened for the merge loop (plus the character fold).
    /// `None` for vocabularies too wide to pack (reference loop).
    core: Option<MergeCore>,

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
    ///
    /// Uses plain BPE semantics (`ignore_merges = false`): only tokens the
    /// merges rebuild short-circuit, so added tokens encode via merges.
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
            token_cache.insert(bytes.clone(), token_id as TokenId);
        }

        let vocab_size = token_bytes.len();
        let num_base_tokens = base_tokens.len();

        // No HF config here: plain BPE semantics (`ignore_merges = false`).
        let encoder = Self::with_cache(pair_lookup, byte_lut, token_cache, false, vocab_size, num_base_tokens, Some(&token_bytes));

        (encoder, token_bytes)
    }

    /// Create encoder from a complete vocabulary and merge rules.
    ///
    /// For tokenizers (like LLaMA 3) where vocab has pre-assigned IDs.
    /// `ignore_merges` is HF `model.ignore_merges` (see `token_cache`).
    pub fn from_vocab_and_merges(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
        num_base_tokens: usize,
        ignore_merges: bool,
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
            token_cache.insert(bytes.clone(), token_id as TokenId);
        }

        let encoder =
            Self::with_cache(pair_lookup, byte_lut, token_cache, ignore_merges, vocab.len(), num_base_tokens, Some(&token_bytes));

        (encoder, token_bytes)
    }

    /// Assemble the encoder from a cache of every token, pruning the cache
    /// to merge-reachable tokens unless `ignore_merges`. `token_bytes`
    /// (by id) feeds the character fold; `None` reads `token_cache`, which
    /// then must hold every token.
    fn with_cache(
        pair_lookup: FoldHashMap<u64, (TokenId, u32)>,
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        ignore_merges: bool,
        vocab_size: usize,
        num_base_tokens: usize,
        token_bytes: Option<&[Vec<u8>]>,
    ) -> Self {
        let core = match token_bytes {
            Some(tb) => MergeCore::build(&pair_lookup, byte_lut, tb.iter().enumerate().map(|(i, b)| (i as TokenId, b.as_slice()))),
            None => MergeCore::build(&pair_lookup, byte_lut, token_cache.iter().map(|(b, &id)| (id, b.as_slice()))),
        };
        let mut encoder = Self {
            core,
            pair_lookup,
            byte_lut,
            token_cache,
            ignore_merges,
            seeds: OnceLock::new(),
            whole: OnceLock::new(),
            vocab_size,
            num_base_tokens,
        };
        if !ignore_merges {
            encoder.prune_token_cache();
        }
        encoder
    }

    /// Drop multi-byte cache entries whose bytes the merge loop does not
    /// encode as exactly that token. Single-byte entries are left alone
    /// (1-byte pieces go through `byte_lut`). One merge run per token
    /// (~70 ms single-threaded for a 248k vocab, ~13 ms on 16 threads), so
    /// large vocabs are checked across threads.
    fn prune_token_cache(&mut self) {
        let mut cache = std::mem::take(&mut self.token_cache);
        let this = &*self;
        let entries: Vec<(&Vec<u8>, TokenId)> =
            cache.iter().filter(|(b, _)| b.len() >= 2).map(|(b, &id)| (b, id)).collect();
        let unreachable = |chunk: &[(&Vec<u8>, TokenId)]| -> Vec<Vec<u8>> {
            let mut out = Vec::new();
            chunk
                .iter()
                .filter(|&&(bytes, id)| {
                    out.clear();
                    this.merge_only(bytes, &mut out);
                    !(out.len() == 1 && out[0] == id)
                })
                .map(|&(bytes, _)| bytes.clone())
                .collect()
        };
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(entries.len() / 16_384).max(1);
        let dead: Vec<Vec<u8>> = if threads == 1 {
            unreachable(&entries)
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = entries
                    .chunks(entries.len().div_ceil(threads))
                    .map(|c| s.spawn(move || unreachable(c)))
                    .collect();
                handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
            })
        };
        drop(entries);
        for bytes in &dead {
            cache.remove(bytes);
        }
        self.token_cache = cache;
    }

    /// HF `ignore_merges`: whether every vocab token short-circuits.
    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
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
    /// - `token_cache` holds every token; it is pruned unless `ignore_merges`
    pub fn from_parts(
        merges: &[(TokenId, TokenId, TokenId)], // (left, right, merged_id)
        byte_lut: [TokenId; 256],
        token_cache: FoldHashMap<Vec<u8>, TokenId>,
        ignore_merges: bool,
        vocab_size: usize,
        num_base_tokens: usize,
    ) -> Self {
        // Build pair_lookup directly from pre-computed merged IDs - O(num_merges)
        let mut pair_lookup = FoldHashMap::default();
        for (merge_index, &(left, right, merged_id)) in merges.iter().enumerate() {
            pair_lookup.insert(pack_pair(left, right), (merged_id, merge_index as u32));
        }

        Self::with_cache(pair_lookup, byte_lut, token_cache, ignore_merges, vocab_size, num_base_tokens, None)
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

    /// Whole-token [`PretokenCache::seed`] entries of this vocab.
    pub(crate) fn cache_seeds(&self) -> &[CacheSeed] {
        self.seeds.get_or_init(|| cache_seeds(&self.token_cache))
    }

    /// Short-piece miss; results too many tokens for an inline entry go to
    /// the long-piece cache.
    #[inline(never)]
    fn encode_cache_miss(&self, text: &[u8], lo: u64, hi: u64, cache: &mut PretokenCache, out: &mut Vec<TokenId>) {
        if cache.get_long(text, out) {
            return;
        }
        // `encode_uncached` would check the whole-piece map first; for a
        // 2..=15-byte piece (1-byte pieces never get here) the flat table
        // answers the same question.
        debug_assert!(text.len() >= 2);
        if let Some(t) = self.whole.get_or_init(|| WholeTable::build(&self.token_cache)).get(lo, hi) {
            cache.insert_with_key(lo, hi, &[t]);
            return out.push(t);
        }
        let start = out.len();
        self.merge_only(text, out);
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
        if let Some(&token_id) = self.token_cache.get(text) {
            return out.push(token_id);
        }
        self.merge_only(text, out)
    }

    /// The merge loop alone (no whole-piece shortcut) on a piece of 2+ bytes.
    fn merge_only(&self, text: &[u8], out: &mut Vec<TokenId>) {
        match &self.core {
            Some(core) => core.encode(text, out),
            None => out.extend(self.merge_reference(text)),
        }
    }

    /// [fold] Like [`Self::encode`] but the merge loop starts from bytes
    /// (no character fold). Equivalence oracle for the fold.
    #[doc(hidden)]
    pub fn encode_unfolded(&self, text: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::new();
        match text.len() {
            0 => {}
            1 => out.push(self.byte_lut[text[0] as usize]),
            _ => match (self.token_cache.get(text), &self.core) {
                (Some(&t), _) => out.push(t),
                (None, Some(core)) => core.encode_unfolded(text, &mut out),
                (None, None) => out.extend(self.merge_reference(text)),
            },
        }
        out
    }

    /// [fold] Build-time fold verdicts (`None` without a merge core).
    #[doc(hidden)]
    pub fn fold_stats(&self) -> Option<FoldStats> {
        self.core.as_ref().map(|c| c.fold_stats())
    }

    /// [fold] Number of symbols the merge loop starts from for `text`.
    #[doc(hidden)]
    pub fn seed_len(&self, text: &[u8]) -> usize {
        self.core.as_ref().map_or(text.len(), |c| c.seed_len(text))
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
        if let Some(&token_id) = self.token_cache.get(text) {
            return vec![token_id];
        }

        self.merge_reference(text)
    }

    /// The O(n²) merge loop on a piece of 2+ bytes, without the shortcut.
    fn merge_reference(&self, text: &[u8]) -> Vec<TokenId> {
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
    /// [fold] Characters seeded as one symbol (see `fold.rs`); `None` when
    /// none fold or the switch is off.
    fold: Option<CharFold>,
    fold_stats: FoldStats,
    /// Every pair's rank is its merged id (Backtracking cores): merged ids
    /// come from the pair code itself.
    rank_is_id: bool,
}

impl MergeCore {
    /// `pair_lookup` maps packed pairs to `(merged, rank)`; ranks must be
    /// unique per pair. `tokens` lists every vocab token's bytes (used to
    /// prove the character fold). `None` when the table can't be packed.
    pub(crate) fn build<'a>(
        pair_lookup: &FoldHashMap<u64, (TokenId, u32)>,
        byte_lut: [TokenId; 256],
        tokens: impl Iterator<Item = (TokenId, &'a [u8])>,
    ) -> Option<Self> {
        let pairs = PairTable::build(pair_lookup)?;
        let rank_is_id = pair_lookup.values().all(|&(m, r)| m == r);
        let (fold, fold_stats) = CharFold::build(pair_lookup, &byte_lut, tokens);
        Some(Self { pairs, byte_lut, fold, fold_stats, rank_is_id })
    }

    /// Whether any character folds.
    #[inline]
    pub(crate) fn fold_stats(&self) -> FoldStats {
        self.fold_stats
    }

    /// Whether every pair's rank is its merged id.
    #[cfg(test)]
    pub(crate) fn rank_is_id(&self) -> bool {
        self.rank_is_id
    }

    /// Encode a piece of 1+ bytes (folded characters as one symbol, other
    /// bytes as their base tokens).
    #[inline]
    pub(crate) fn encode(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.encode_with(text, self.fold.as_ref(), out)
    }

    /// [fold] Encode starting from per-byte base tokens only.
    pub(crate) fn encode_unfolded(&self, text: &[u8], out: &mut Vec<TokenId>) {
        self.encode_with(text, None, out)
    }

    /// [fold] Symbols the merge loop starts from.
    pub(crate) fn seed_len(&self, text: &[u8]) -> usize {
        match &self.fold {
            Some(f) => {
                let mut n = 0;
                fold::seed(f, &self.byte_lut, text, |_| n += 1);
                n
            }
            None => text.len(),
        }
    }

    #[inline]
    fn encode_with(&self, text: &[u8], fold: Option<&CharFold>, out: &mut Vec<TokenId>) {
        match text.len() {
            0 => {}
            1 => out.push(self.byte_lut[text[0] as usize]),
            _ if self.rank_is_id => self.encode_impl::<true>(text, fold, out),
            _ => self.encode_impl::<false>(text, fold, out),
        }
    }

    #[inline(always)]
    fn encode_impl<const ID: bool>(&self, text: &[u8], fold: Option<&CharFold>, out: &mut Vec<TokenId>) {
        let n = text.len();
        let lut = &self.byte_lut;
        let Some(f) = fold else {
            if n <= SHORT_MAX {
                let mut toks = [0 as TokenId; SHORT_MAX];
                for (t, &b) in toks.iter_mut().zip(text) {
                    *t = lut[b as usize];
                }
                return merge_short::<ID>(&self.pairs, &mut toks, n, out);
            }
            return merge_long::<ID>(&self.pairs, |t| t.extend(text.iter().map(|&b| lut[b as usize])), out);
        };
        let short_max = if n <= NONASCII_SHORT_MAX || text.is_ascii() { SHORT_MAX } else { NONASCII_SHORT_MAX };
        if n <= SHORT_MAX {
            let mut toks = [0 as TokenId; SHORT_MAX];
            let mut w = 0;
            fold::seed(f, lut, text, |t| {
                toks[w] = t;
                w += 1;
            });
            if w <= short_max {
                return merge_short::<ID>(&self.pairs, &mut toks, w, out);
            }
            return merge_long::<ID>(&self.pairs, |t| t.extend_from_slice(&toks[..w]), out);
        }
        if n <= SHORT_FOLD_MAX_BYTES {
            // [fold] Gate on symbol count: seed once into a byte-sized
            // buffer, then pick the loop.
            let mut toks = [0 as TokenId; SHORT_FOLD_MAX_BYTES];
            let mut w = 0;
            fold::seed(f, lut, text, |t| {
                toks[w] = t;
                w += 1;
            });
            if w <= short_max {
                return merge_short::<ID>(&self.pairs, &mut toks, w, out);
            }
            return merge_long::<ID>(&self.pairs, |t| t.extend_from_slice(&toks[..w]), out);
        }
        merge_long::<ID>(&self.pairs, |t| fold::seed(f, lut, text, |x| t.push(x)), out)
    }
}

/// Multipass merge for short pieces over the `len` seeded symbols in
/// `toks` (`1..=SHORT_MAX`): each round takes the lowest pair code (lowest
/// rank, leftmost on ties; equal ranks are the same pair). Per-pair codes
/// are carried between rounds so only pairs next to a merge are looked up
/// again. A SAFE winner merges every non-overlapping occurrence in the
/// same round.
#[inline(always)]
fn merge_short<const ID: bool>(pairs: &PairTable, toks: &mut [TokenId], n: usize, out: &mut Vec<TokenId>) {
    debug_assert!((1..=SHORT_MAX).contains(&n) && n <= toks.len());
    let toks = &mut toks[..n];
    let mut pv = [NO_MERGE; SHORT_MAX];
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
        let merged = pairs.merged::<ID>(best);
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

/// Queue merge for long pieces: `fill` seeds the symbols into a linked
/// list, candidate pairs keyed `code << 32 | left position`. Initial pairs
/// sit in a sorted cold array; pairs created by merges go to a hot heap;
/// the smaller head wins. Each position carries its current pair code
/// (`NO_MERGE` once dead or last), so a popped entry is stale exactly when
/// its code differs from the carried one (ranks identify pairs uniquely):
/// no pair-table probe on pop.
fn merge_long<const ID: bool>(pairs: &PairTable, fill: impl FnOnce(&mut Vec<TokenId>), out: &mut Vec<TokenId>) {
    LONG_SCRATCH.with(|s| {
        let mut s = s.borrow_mut();
        let LongScratch { toks, prev, next, code, cold, hot } = &mut *s;
        toks.clear();
        fill(toks);
        let n = toks.len();
        if n <= 1 {
            out.extend_from_slice(toks);
            return;
        }
        let n32 = n as u32;
        prev.clear();
        prev.extend((0..n32).map(|i| i.wrapping_sub(1)));
        next.clear();
        next.extend(1..=n32);
        code.clear();
        cold.clear();
        hot.clear();
        for i in 0..n - 1 {
            let v = pairs.get(toks[i], toks[i + 1]);
            code.push(v);
            if v != NO_MERGE {
                cold.push(((v as u64) << 32) | i as u64);
            }
        }
        code.push(NO_MERGE);
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
            let c = (key >> 32) as u32;
            if code[pos] != c {
                continue;
            }
            let nx = next[pos] as usize;
            debug_assert!(nx < n);
            let merged = pairs.merged::<ID>(c);
            toks[pos] = merged;
            let nn = next[nx];
            next[nx] = DEAD;
            code[nx] = NO_MERGE;
            next[pos] = nn;
            if nn < n32 {
                prev[nn as usize] = pos as u32;
                let v = pairs.get(merged, toks[nn as usize]);
                code[pos] = v;
                if v != NO_MERGE {
                    hot.push(Reverse(((v as u64) << 32) | pos as u64));
                }
            } else {
                code[pos] = NO_MERGE;
            }
            let pp = prev[pos];
            if pp < n32 {
                let v = pairs.get(toks[pp as usize], merged);
                code[pp as usize] = v;
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
        let (encoder, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &[(0, 0)], 1, true);
        assert_eq!(encoder.encode(&long), vec![2]);
        assert_eq!(encoder.encode_reference(&long), vec![2]);
        assert_eq!(encoder.encode(&long[..19]), encoder.encode_reference(&long[..19]));
    }

    #[test]
    fn test_ignore_merges_gates_unreachable_tokens() {
        // "abc" exists as a vocab token, but the merges build "ab" first and
        // never merge "ab"+"c" (only "a"+"bc", which "ab" pre-empts).
        let mut vocab: Vec<(u32, Vec<u8>)> = (0..256u32).map(|b| (b, vec![b as u8])).collect();
        vocab.push((256, b"ab".to_vec()));
        vocab.push((257, b"bc".to_vec()));
        vocab.push((258, b"abc".to_vec()));
        let merges = [(b'a' as u32, b'b' as u32), (b'b' as u32, b'c' as u32), (b'a' as u32, 257)];
        for (ignore, want) in [(false, vec![256, b'c' as u32]), (true, vec![258])] {
            let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, ignore);
            assert_eq!(enc.ignore_merges(), ignore);
            assert_eq!(enc.encode(b"abc"), want);
            assert_eq!(enc.encode_reference(b"abc"), want);
            let mut got = Vec::new();
            enc.encode_into(b"abc", Some(&mut PretokenCache::new()), &mut got);
            assert_eq!(got, want);
            // Reachable tokens still short-circuit either way.
            assert_eq!(enc.encode(b"ab"), vec![256]);
            assert_eq!(enc.encode(b"bc"), vec![257]);
        }
        // Plain BPE oracle: "abc" merges as "ab" + "c".
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        assert_eq!(enc.merge_reference(b"abc"), vec![256, b'c' as u32]);
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
            let ignore_merges = round % 3 == 0;
            let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, ignore_merges);
            let (enc2, _) = BytePairEncoder::from_merges(&merges.iter().copied().filter(|&(l, r)| l < 256 && r < 256).collect::<Vec<_>>(), &vocab[..256].iter().map(|(_, b)| b.clone()).collect::<Vec<_>>());
            let mut caches = [PretokenCache::new(), PretokenCache::new()];
            for _ in 0..400 {
                let len = if rng.below(8) == 0 { rng.below(400) } else { rng.below(40) };
                let text: Vec<u8> = (0..len).map(|_| b'a' + rng.below(alphabet) as u8).collect();
                for (e, cache) in [&enc, &enc2].into_iter().zip(caches.iter_mut()) {
                    let want = e.encode_reference(&text);
                    if !e.ignore_merges() && text.len() >= 2 {
                        // Plain BPE: the shortcut never changes the result.
                        assert_eq!(e.merge_reference(&text), want);
                    }
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
        let (enc, token_bytes) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        let parts: Vec<(u32, u32, u32)> = {
            let mut v: Vec<_> = enc.pair_lookup().iter().map(|(&k, &(m, r))| (r, (k >> 32) as u32, k as u32, m)).collect();
            v.sort();
            v.into_iter().map(|(_, l, r, m)| (l, r, m)).collect()
        };
        let mut token_cache = FoldHashMap::default();
        for (id, b) in token_bytes.iter().enumerate() {
            token_cache.insert(b.clone(), id as TokenId);
        }
        let rebuilt = BytePairEncoder::from_parts(&parts, enc.byte_lut, token_cache, false, enc.vocab_size(), 256);
        assert_eq!(rebuilt.token_cache, enc.token_cache);
        for _ in 0..500 {
            let text: Vec<u8> = (0..rng.below(120)).map(|_| b'a' + rng.below(3) as u8).collect();
            assert_eq!(rebuilt.encode(&text), rebuilt.encode_reference(&text));
        }
    }

    // ---- [fold] character fold ------------------------------------------

    const FOLD_CHARS: &[&str] = &["é", "ж", "中", "文", "朝", "日", "😀", "ñ", "語"];

    /// Byte-level vocab whose merges assemble some multi-byte characters
    /// (in random association order) plus random merges across tokens,
    /// including ones straddling character boundaries. `monotone`: every
    /// merged id is new and above its operands (rank = merged id usable).
    fn random_utf8_vocab(rng: &mut Rng, n_random: usize, monotone: bool) -> (Vec<(u32, Vec<u8>)>, Vec<(u32, u32)>) {
        let mut vocab: Vec<(u32, Vec<u8>)> = (0..256u32).map(|b| (b, vec![b as u8])).collect();
        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut add = |vocab: &mut Vec<(u32, Vec<u8>)>, merges: &mut Vec<(u32, u32)>, l: u32, r: u32| -> Option<u32> {
            let mut bytes = vocab[l as usize].1.clone();
            bytes.extend_from_slice(&vocab[r as usize].1);
            if bytes.len() > 12 || !seen.insert((l, r)) {
                return None;
            }
            let id = match vocab.iter().position(|(_, b)| *b == bytes) {
                Some(_) if monotone => return None,
                Some(i) => i as u32,
                None => {
                    vocab.push((vocab.len() as u32, bytes));
                    vocab.len() as u32 - 1
                }
            };
            merges.push((l, r));
            Some(id)
        };
        let ascii = b"xyz ";
        for _ in 0..n_random {
            match rng.below(3) {
                0 => {
                    // Assemble a character, left- or right-associated.
                    let c = FOLD_CHARS[rng.below(FOLD_CHARS.len())].as_bytes();
                    let mut toks: Vec<u32> = c.iter().map(|&b| b as u32).collect();
                    while toks.len() > 1 {
                        let i = rng.below(toks.len() - 1);
                        let Some(m) = add(&mut vocab, &mut merges, toks[i], toks[i + 1]) else { break };
                        toks[i] = m;
                        toks.remove(i + 1);
                    }
                }
                _ => {
                    // Random pair of existing tokens / ascii / char bytes.
                    let pick = |rng: &mut Rng, vocab: &Vec<(u32, Vec<u8>)>| -> u32 {
                        match rng.below(3) {
                            0 => ascii[rng.below(ascii.len())] as u32,
                            1 => {
                                let c = FOLD_CHARS[rng.below(FOLD_CHARS.len())].as_bytes();
                                c[rng.below(c.len())] as u32
                            }
                            _ if vocab.len() > 256 => 256 + rng.below(vocab.len() - 256) as u32,
                            _ => ascii[0] as u32,
                        }
                    };
                    let (l, r) = (pick(rng, &vocab), pick(rng, &vocab));
                    add(&mut vocab, &mut merges, l, r);
                }
            }
        }
        if !monotone {
            for i in (1..merges.len()).rev() {
                merges.swap(i, rng.below(i + 1));
            }
        }
        (vocab, merges)
    }

    /// Text of random chars, ascii, stray bytes and truncated characters.
    fn random_utf8_text(rng: &mut Rng, max_parts: usize) -> Vec<u8> {
        let mut t = Vec::new();
        for _ in 0..rng.below(max_parts + 1) {
            match rng.below(10) {
                0..=4 => t.extend_from_slice(FOLD_CHARS[rng.below(FOLD_CHARS.len())].as_bytes()),
                5..=6 => t.push(b"xyz "[rng.below(4)]),
                7 => t.push(rng.below(256) as u8),
                8 => {
                    let c = FOLD_CHARS[rng.below(FOLD_CHARS.len())].as_bytes();
                    t.extend_from_slice(&c[..1 + rng.below(c.len())]);
                }
                _ => t.push([0x80, 0xBF, 0xC0, 0xC1, 0xE6, 0xF0, 0xFF][rng.below(7)]),
            }
        }
        t
    }

    /// Plain BPE (lowest rank, leftmost) from bytes over `(merged, rank)`.
    fn ref_bpe(pairs: &FoldHashMap<u64, (TokenId, u32)>, text: &[u8]) -> Vec<TokenId> {
        let mut t: Vec<TokenId> = text.iter().map(|&b| b as TokenId).collect();
        loop {
            let best = (0..t.len().saturating_sub(1))
                .filter_map(|i| pairs.get(&pack_pair(t[i], t[i + 1])).map(|&(m, r)| (r, i, m)))
                .min();
            let Some((_, i, m)) = best else { return t };
            t[i] = m;
            t.remove(i + 1);
        }
    }

    #[test]
    fn test_fold_matches_unfolded_on_random_utf8_vocabs() {
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let (mut folded_rounds, mut step_rejects, mut final_rejects) = (0, 0, 0);
        for round in 0..200 {
            let monotone = round % 2 == 1;
            let (vocab, merges) = random_utf8_vocab(&mut rng, 20 + round % 60, monotone);
            let token_bytes: Vec<Vec<u8>> = vocab.iter().map(|(_, b)| b.clone()).collect();
            let lut: [TokenId; 256] = std::array::from_fn(|b| b as TokenId);
            // Simple (merge-index ranks) and Backtracking-style (rank = id).
            let mut pl: FoldHashMap<u64, (TokenId, u32)> = FoldHashMap::default();
            let ids: FoldHashMap<&[u8], u32> = token_bytes.iter().enumerate().map(|(i, b)| (b.as_slice(), i as u32)).collect();
            for (rank, &(l, r)) in merges.iter().enumerate() {
                let mut b = token_bytes[l as usize].clone();
                b.extend_from_slice(&token_bytes[r as usize]);
                let m = ids[b.as_slice()];
                pl.entry(pack_pair(l, r)).or_insert((m, if monotone { m } else { rank as u32 }));
            }
            let tokens = token_bytes.iter().enumerate().map(|(i, b)| (i as TokenId, b.as_slice()));
            let core = MergeCore::build(&pl, lut, tokens).expect("packable");
            assert_eq!(core.rank_is_id(), monotone);
            let st = core.fold_stats();
            folded_rounds += (st.folded > 0) as usize;
            step_rejects += st.unsafe_step;
            final_rejects += st.unsafe_final;
            if monotone {
                assert_eq!(st.unsafe_final, 0, "id-monotone vocabs satisfy the final guard");
            }
            let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
            for i in 0..300 {
                let text = random_utf8_text(&mut rng, if i % 20 == 0 { 120 } else { 12 });
                let want = ref_bpe(&pl, &text);
                let (mut got, mut unf) = (Vec::new(), Vec::new());
                core.encode(&text, &mut got);
                core.encode_unfolded(&text, &mut unf);
                assert_eq!(unf, want, "round {round} unfolded {text:x?}");
                assert_eq!(got, want, "round {round} folded {text:x?} (seed {} syms)", core.seed_len(&text));
                if !monotone {
                    assert_eq!(enc.encode(&text), enc.encode_reference(&text), "round {round} simple {text:x?}");
                }
            }
        }
        assert!(folded_rounds > 50, "fold too rarely admitted: {folded_rounds}");
        assert!(step_rejects > 0 && final_rejects > 0, "guards never exercised: {step_rejects} {final_rejects}");
    }

    /// Byte-level vocab over all 256 bytes plus `extra` tokens and merges
    /// (`(left bytes, right bytes)` in rank order).
    fn byte_vocab(merges: &[(&[u8], &[u8])]) -> (Vec<(u32, Vec<u8>)>, Vec<(u32, u32)>) {
        let mut vocab: Vec<(u32, Vec<u8>)> = (0..256u32).map(|b| (b, vec![b as u8])).collect();
        let mut ids = Vec::new();
        for &(l, r) in merges {
            let id_of = |vocab: &mut Vec<(u32, Vec<u8>)>, b: &[u8]| -> u32 {
                match vocab.iter().position(|(_, x)| x == b) {
                    Some(i) => i as u32,
                    None => {
                        vocab.push((vocab.len() as u32, b.to_vec()));
                        vocab.len() as u32 - 1
                    }
                }
            };
            let (li, ri) = (id_of(&mut vocab, l), id_of(&mut vocab, r));
            id_of(&mut vocab, &[l, r].concat());
            ids.push((li, ri));
        }
        (vocab, ids)
    }

    #[test]
    fn test_fold_rejects_boundary_steal() {
        // 朝 = E6 9C 9D assembles via (E6,9C) then (E6 9C, 9D); the thief
        // (9D, E6) at rank 0 straddles a boundary in "朝朝": HF's guard.
        let thief: &[(&[u8], &[u8])] = &[(b"\x9d", b"\xe6"), (b"\xe6", b"\x9c"), (b"\xe6\x9c", b"\x9d")];
        let (vocab, merges) = byte_vocab(thief);
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        let st = enc.fold_stats().unwrap();
        assert_eq!((st.folded, st.unsafe_step), (0, 1));
        let t = "朝朝".as_bytes();
        assert_eq!(enc.encode(t), enc.merge_reference(t));
        // Without the thief it folds.
        let (vocab, merges) = byte_vocab(&thief[1..]);
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        assert_eq!(enc.fold_stats().unwrap().folded, 1);
        assert_eq!(enc.seed_len(t), 2);
        assert_eq!(enc.encode(t), enc.merge_reference(t));
    }

    #[test]
    fn test_fold_rejects_low_rank_merge_into_folded_token() {
        // HF's step guard passes for é (C3 A9, nothing takes C3 from the
        // left or A9 from the right), but ("y", "é") outranks é's own
        // assembly: folding "xyé" would give [x, yé], BPE gives [xy, é].
        let (vocab, merges) = byte_vocab(&[(b"y", "é".as_bytes()), (b"x", b"y"), (b"\xc3", b"\xa9")]);
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        let st = enc.fold_stats().unwrap();
        assert_eq!((st.folded, st.unsafe_step, st.unsafe_final), (0, 0, 1));
        let t = "xyé".as_bytes();
        let want = enc.merge_reference(t);
        assert_eq!(want.len(), 2);
        assert_eq!(vocab[want[0] as usize].1, b"xy");
        assert_eq!(enc.encode(t), want);
    }

    #[test]
    fn test_fold_skips_invalid_utf8_neighbours() {
        // 朝 folds (no valid-UTF-8 neighbour can steal), but a stray lead
        // byte E6 before it can: (E6, E6) is not a boundary merge, so the
        // encoder must not fold there.
        let (vocab, merges) = byte_vocab(&[(b"\xe6", b"\xe6"), (b"\xe6", b"\x9c"), (b"\xe6\x9c", b"\x9d"), (b"\x9d", b"\x80")]);
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        assert_eq!(enc.fold_stats().unwrap().folded, 1);
        for t in [&b"\xe6\xe6\x9c\x9d"[..], b"\xe6\x9c\x9d\x80", b"a\xe6\x9c\x9d", b"\xe6\x9c\x9d\xe6\x9c"] {
            assert_eq!(enc.encode(t), enc.merge_reference(t), "{t:x?}");
            assert_eq!(enc.encode(t), enc.encode_unfolded(t), "{t:x?}");
        }
        assert_eq!(enc.seed_len(b"\xe6\xe6\x9c\x9d"), 4);
        assert_eq!(enc.seed_len(b"a\xe6\x9c\x9d"), 2);
    }

    #[test]
    fn test_fold_short_long_gate() {
        // Pieces past SHORT_MAX bytes that fold to <= SHORT_MAX symbols, and
        // longer ones, agree with the unfolded queue.
        let (vocab, merges) = byte_vocab(&[(b"\xe4\xb8", b"\xad"), (b"\xe4", b"\xb8"), (b"\xe6\x96", b"\x87"), (b"\xe6", b"\x96")]);
        let (enc, _) = BytePairEncoder::from_vocab_and_merges(&vocab, &merges, 256, false);
        assert_eq!(enc.fold_stats().unwrap().folded, 2);
        for n in [20, 21, 22, 64, 65, 100, 300] {
            let t: Vec<u8> = (0..n).flat_map(|i| if i % 3 == 0 { "文".as_bytes() } else { "中".as_bytes() }).copied().collect();
            assert_eq!(enc.seed_len(&t), n);
            assert_eq!(enc.encode(&t), enc.merge_reference(&t));
            assert_eq!(enc.encode(&t), enc.encode_unfolded(&t));
        }
    }
}
