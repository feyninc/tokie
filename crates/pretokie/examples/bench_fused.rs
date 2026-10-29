//! Fused pretokenizer + BPE encoder prototype.
//!
//! Instead of: pretok → Vec<boundary> → for each piece → encoder.encode(piece)
//! We do:      SIMD finds boundary → immediately encode piece → next boundary
//!
//! The encoder work can hide the SIMD extraction latency via out-of-order execution.
//!
//! Usage: cargo run -p pretokie --example bench_fused --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::{decode_utf8, is_ascii_letter, is_digit};
use foldhash::HashMap as FoldHashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ===========================================================================
// Simple BPE encoder (realistic cost simulation)
// ===========================================================================

type TokenId = u32;

const VOCAB_SIZE: usize = 512; // 256 byte tokens + merges

/// Minimal BPE encoder using flat LUTs — no HashMap.
struct SimpleBpe {
    /// Merge LUT: merge_lut[left * VOCAB_SIZE + right] = merged token (0 = no merge)
    merge_lut: Vec<TokenId>,
    /// Merge rank (priority): lower = merge first. 0 = no merge.
    merge_rank: Vec<u32>,
    /// Token cache: for pieces up to 8 bytes, hash → (key_hash, token_id)
    /// Open-addressing hash table, power-of-2 size
    cache_keys: Vec<u64>,    // stored hash
    cache_vals: Vec<TokenId>,
    cache_mask: usize,
}

/// FNV-1a hash for short byte slices
#[inline(always)]
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl SimpleBpe {
    fn new_english_like() -> Self {
        // Merge LUT: flat array indexed by (left, right)
        let mut merge_lut = vec![0u32; VOCAB_SIZE * VOCAB_SIZE];
        let mut merge_rank = vec![0u32; VOCAB_SIZE * VOCAB_SIZE];
        let mut next_id = 256u32;
        let mut rank = 1u32;

        let common_pairs: &[(u8, u8)] = &[
            (b't', b'h'), (b'h', b'e'), (b'i', b'n'), (b'e', b'r'),
            (b'a', b'n'), (b'r', b'e'), (b'o', b'n'), (b'e', b's'),
            (b's', b't'), (b'e', b'n'), (b'a', b't'), (b'o', b'r'),
            (b'n', b'd'), (b't', b'i'), (b'a', b'l'), (b'a', b'r'),
            (b'e', b'd'), (b'i', b's'), (b'o', b'f'), (b'i', b't'),
            (b'n', b'g'), (b'l', b'e'), (b'e', b'l'), (b'o', b'u'),
            (b'n', b't'), (b's', b'e'), (b'r', b'o'), (b'l', b'i'),
            (b'r', b'i'), (b'c', b'o'), (b'c', b'e'), (b'd', b'e'),
            (b'r', b'a'), (b'i', b'o'), (b'c', b't'), (b'l', b'a'),
            (b'u', b'r'), (b'n', b'e'), (b'c', b'h'), (b'l', b'l'),
        ];

        // Level 1: byte-pair merges
        for &(a, b) in common_pairs {
            let idx = (a as usize) * VOCAB_SIZE + (b as usize);
            merge_lut[idx] = next_id;
            merge_rank[idx] = rank;
            next_id += 1;
            rank += 1;
        }

        // Level 2: merge results + bytes (e.g., "th" + "e" → "the")
        // th(256) + e(101) → "the"
        let th = 256u32; let he = 257; let in_ = 258; let er = 259;
        let an = 260; let re = 261; let on = 262; let es = 263;
        let st = 264; let en = 265; let at = 266; let or = 267;
        let pairs2: &[(u32, u32)] = &[
            (th, b'e' as u32),  // the
            (in_, b'g' as u32), // ing
            (an, b'd' as u32),  // and
            (er, b's' as u32),  // ers
            (on, b'e' as u32),  // one
            (re, b's' as u32),  // res
            (st, b'e' as u32),  // ste
            (at, b'e' as u32),  // ate
            (or, b'e' as u32),  // ore
            (en, b't' as u32),  // ent
            (es, b't' as u32),  // est
            (he, b'r' as u32),  // her
        ];
        for &(a, b) in pairs2 {
            if (a as usize) < VOCAB_SIZE && (b as usize) < VOCAB_SIZE {
                let idx = (a as usize) * VOCAB_SIZE + (b as usize);
                merge_lut[idx] = next_id;
                merge_rank[idx] = rank;
                next_id += 1;
                rank += 1;
            }
        }

        // Token cache: open-addressing hash table (4096 slots)
        let cache_size = 4096usize;
        let cache_mask = cache_size - 1;
        let mut cache_keys = vec![0u64; cache_size];
        let mut cache_vals = vec![0u32; cache_size];

        // Insert single ASCII bytes
        for i in 0u8..=127 {
            let h = fnv1a(&[i]);
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h;
            cache_vals[slot] = i as u32;
        }
        // Insert common 2-byte tokens
        for &(a, b) in common_pairs {
            let bytes = [a, b];
            let h = fnv1a(&bytes);
            let merged = merge_lut[(a as usize) * VOCAB_SIZE + (b as usize)];
            if merged != 0 {
                let mut slot = (h as usize) & cache_mask;
                while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
                cache_keys[slot] = h;
                cache_vals[slot] = merged;
            }
        }
        // Insert common words
        let mut wid = next_id;
        for word in &["the", "of", "and", "in", "to", "is", "it", "for", "was", "on",
                      "he", "at", "or", "an", "be", "as", "by", "we", "no", "do"] {
            let h = fnv1a(word.as_bytes());
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h;
            cache_vals[slot] = wid;
            wid += 1;
        }

        SimpleBpe { merge_lut, merge_rank, cache_keys, cache_vals, cache_mask }
    }

    /// Encode a piece using flat LUT lookups — no HashMap.
    #[inline(always)]
    fn encode_piece(&self, piece: &[u8], output: &mut Vec<TokenId>) {
        // Fast path: cache lookup for short pieces
        if piece.len() <= 8 {
            let h = fnv1a(piece);
            let mut slot = (h as usize) & self.cache_mask;
            // Probe up to 4 slots
            for _ in 0..4 {
                if self.cache_keys[slot] == h {
                    output.push(self.cache_vals[slot]);
                    return;
                }
                if self.cache_keys[slot] == 0 { break; }
                slot = (slot + 1) & self.cache_mask;
            }
        }

        // Slow path: byte-level tokenization + LUT merge
        // Use a fixed-size workspace on the stack
        let mut workspace = [0u32; 64];
        let plen = piece.len().min(64);
        for i in 0..plen {
            workspace[i] = piece[i] as u32;
        }
        let mut len = plen;

        // Greedy merge: find best (lowest rank) merge each pass
        let mut changed = true;
        while changed && len > 1 {
            changed = false;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            let mut best_merged = 0u32;

            for i in 0..len - 1 {
                let left = workspace[i] as usize;
                let right = workspace[i + 1] as usize;
                if left < VOCAB_SIZE && right < VOCAB_SIZE {
                    let idx = left * VOCAB_SIZE + right;
                    let r = unsafe { *self.merge_rank.get_unchecked(idx) };
                    if r != 0 && r < best_rank {
                        best_rank = r;
                        best_pos = i;
                        best_merged = unsafe { *self.merge_lut.get_unchecked(idx) };
                    }
                }
            }

            if best_rank < u32::MAX {
                workspace[best_pos] = best_merged;
                // Shift remaining
                for j in (best_pos + 2)..len {
                    workspace[j - 1] = workspace[j];
                }
                len -= 1;
                changed = true;
            }
        }

        for i in 0..len {
            output.push(workspace[i]);
        }
    }
}

// ===========================================================================
// FoldHashMap-based BPE encoder (for comparison)
// ===========================================================================

struct FoldBpe {
    token_cache: FoldHashMap<Vec<u8>, TokenId>,
    merges: FoldHashMap<(TokenId, TokenId), (TokenId, u32)>, // (merged, rank)
}

impl FoldBpe {
    fn new_english_like() -> Self {
        let mut merges = FoldHashMap::default();
        let mut next_id = 256u32;
        let mut rank = 1u32;

        let common_pairs: &[(u8, u8)] = &[
            (b't', b'h'), (b'h', b'e'), (b'i', b'n'), (b'e', b'r'),
            (b'a', b'n'), (b'r', b'e'), (b'o', b'n'), (b'e', b's'),
            (b's', b't'), (b'e', b'n'), (b'a', b't'), (b'o', b'r'),
            (b'n', b'd'), (b't', b'i'), (b'a', b'l'), (b'a', b'r'),
            (b'e', b'd'), (b'i', b's'), (b'o', b'f'), (b'i', b't'),
            (b'n', b'g'), (b'l', b'e'), (b'e', b'l'), (b'o', b'u'),
            (b'n', b't'), (b's', b'e'), (b'r', b'o'), (b'l', b'i'),
            (b'r', b'i'), (b'c', b'o'), (b'c', b'e'), (b'd', b'e'),
            (b'r', b'a'), (b'i', b'o'), (b'c', b't'), (b'l', b'a'),
            (b'u', b'r'), (b'n', b'e'), (b'c', b'h'), (b'l', b'l'),
        ];
        for &(a, b) in common_pairs {
            merges.insert((a as u32, b as u32), (next_id, rank));
            next_id += 1;
            rank += 1;
        }

        let th = 256u32; let in_ = 258; let an = 260; let re = 261;
        let on = 262; let es = 263; let st = 264; let en = 265;
        let at = 266; let or = 267; let he = 257; let er = 259;
        let pairs2: &[(u32, u32)] = &[
            (th, b'e' as u32), (in_, b'g' as u32), (an, b'd' as u32),
            (er, b's' as u32), (on, b'e' as u32), (re, b's' as u32),
            (st, b'e' as u32), (at, b'e' as u32), (or, b'e' as u32),
            (en, b't' as u32), (es, b't' as u32), (he, b'r' as u32),
        ];
        for &(a, b) in pairs2 {
            merges.insert((a, b), (next_id, rank));
            next_id += 1;
            rank += 1;
        }

        let mut token_cache = FoldHashMap::default();
        for i in 0u8..=127 {
            token_cache.insert(vec![i], i as u32);
        }
        for &(a, b) in common_pairs {
            if let Some(&(merged, _)) = merges.get(&(a as u32, b as u32)) {
                token_cache.insert(vec![a, b], merged);
            }
        }
        let mut wid = next_id;
        for word in &["the", "of", "and", "in", "to", "is", "it", "for", "was", "on",
                      "he", "at", "or", "an", "be", "as", "by", "we", "no", "do"] {
            token_cache.insert(word.as_bytes().to_vec(), wid);
            wid += 1;
        }

        FoldBpe { token_cache, merges }
    }

    #[inline(always)]
    fn encode_piece(&self, piece: &[u8], output: &mut Vec<TokenId>) {
        if piece.len() <= 8 {
            if let Some(&token) = self.token_cache.get(piece) {
                output.push(token);
                return;
            }
        }

        let mut workspace = [0u32; 64];
        let plen = piece.len().min(64);
        for i in 0..plen { workspace[i] = piece[i] as u32; }
        let mut len = plen;

        let mut changed = true;
        while changed && len > 1 {
            changed = false;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            let mut best_merged = 0u32;

            for i in 0..len - 1 {
                if let Some(&(merged, r)) = self.merges.get(&(workspace[i], workspace[i + 1])) {
                    if r < best_rank {
                        best_rank = r;
                        best_pos = i;
                        best_merged = merged;
                    }
                }
            }

            if best_rank < u32::MAX {
                workspace[best_pos] = best_merged;
                for j in (best_pos + 2)..len { workspace[j - 1] = workspace[j]; }
                len -= 1;
                changed = true;
            }
        }

        for i in 0..len { output.push(workspace[i]); }
    }
}

// ===========================================================================
// FNV1a cache + FoldHash merges (isolate cache vs merge effect)
// ===========================================================================

struct FnvCacheFoldMerge {
    cache_keys: Vec<u64>,
    cache_vals: Vec<TokenId>,
    cache_mask: usize,
    merges: FoldHashMap<(TokenId, TokenId), (TokenId, u32)>,
}

impl FnvCacheFoldMerge {
    fn new_english_like() -> Self {
        // Build FoldHash merges (same as FoldBpe)
        let mut merges = FoldHashMap::default();
        let mut next_id = 256u32;
        let mut rank = 1u32;
        let common_pairs: &[(u8, u8)] = &[
            (b't', b'h'), (b'h', b'e'), (b'i', b'n'), (b'e', b'r'),
            (b'a', b'n'), (b'r', b'e'), (b'o', b'n'), (b'e', b's'),
            (b's', b't'), (b'e', b'n'), (b'a', b't'), (b'o', b'r'),
            (b'n', b'd'), (b't', b'i'), (b'a', b'l'), (b'a', b'r'),
            (b'e', b'd'), (b'i', b's'), (b'o', b'f'), (b'i', b't'),
            (b'n', b'g'), (b'l', b'e'), (b'e', b'l'), (b'o', b'u'),
            (b'n', b't'), (b's', b'e'), (b'r', b'o'), (b'l', b'i'),
            (b'r', b'i'), (b'c', b'o'), (b'c', b'e'), (b'd', b'e'),
            (b'r', b'a'), (b'i', b'o'), (b'c', b't'), (b'l', b'a'),
            (b'u', b'r'), (b'n', b'e'), (b'c', b'h'), (b'l', b'l'),
        ];
        for &(a, b) in common_pairs {
            merges.insert((a as u32, b as u32), (next_id, rank));
            next_id += 1; rank += 1;
        }
        let th = 256u32; let he = 257; let in_ = 258; let er = 259;
        let an = 260; let re = 261; let on = 262; let es = 263;
        let st = 264; let en = 265; let at = 266; let or = 267;
        for &(a, b) in &[(th, b'e' as u32), (in_, b'g' as u32), (an, b'd' as u32),
            (er, b's' as u32), (on, b'e' as u32), (re, b's' as u32),
            (st, b'e' as u32), (at, b'e' as u32), (or, b'e' as u32),
            (en, b't' as u32), (es, b't' as u32), (he, b'r' as u32)] {
            merges.insert((a, b), (next_id, rank));
            next_id += 1; rank += 1;
        }

        // Build FNV1a cache (same as SimpleBpe)
        let cache_size = 4096usize;
        let cache_mask = cache_size - 1;
        let mut cache_keys = vec![0u64; cache_size];
        let mut cache_vals = vec![0u32; cache_size];
        for i in 0u8..=127 {
            let h = fnv1a(&[i]);
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h; cache_vals[slot] = i as u32;
        }
        for &(a, b) in common_pairs {
            if let Some(&(merged, _)) = merges.get(&(a as u32, b as u32)) {
                let h = fnv1a(&[a, b]);
                let mut slot = (h as usize) & cache_mask;
                while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
                cache_keys[slot] = h; cache_vals[slot] = merged;
            }
        }
        let mut wid = next_id;
        for word in &["the", "of", "and", "in", "to", "is", "it", "for", "was", "on",
                      "he", "at", "or", "an", "be", "as", "by", "we", "no", "do"] {
            let h = fnv1a(word.as_bytes());
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h; cache_vals[slot] = wid; wid += 1;
        }

        FnvCacheFoldMerge { cache_keys, cache_vals, cache_mask, merges }
    }

    #[inline(always)]
    fn encode_piece(&self, piece: &[u8], output: &mut Vec<TokenId>) {
        if piece.len() <= 8 {
            let h = fnv1a(piece);
            let mut slot = (h as usize) & self.cache_mask;
            for _ in 0..4 {
                if self.cache_keys[slot] == h { output.push(self.cache_vals[slot]); return; }
                if self.cache_keys[slot] == 0 { break; }
                slot = (slot + 1) & self.cache_mask;
            }
        }
        let mut workspace = [0u32; 64];
        let plen = piece.len().min(64);
        for i in 0..plen { workspace[i] = piece[i] as u32; }
        let mut len = plen;
        let mut changed = true;
        while changed && len > 1 {
            changed = false;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            let mut best_merged = 0u32;
            for i in 0..len - 1 {
                if let Some(&(merged, r)) = self.merges.get(&(workspace[i], workspace[i + 1])) {
                    if r < best_rank { best_rank = r; best_pos = i; best_merged = merged; }
                }
            }
            if best_rank < u32::MAX {
                workspace[best_pos] = best_merged;
                for j in (best_pos + 2)..len { workspace[j - 1] = workspace[j]; }
                len -= 1; changed = true;
            }
        }
        for i in 0..len { output.push(workspace[i]); }
    }
}

// ===========================================================================
// FNV1a cache + FNV1a packed-u64 merge lookup
// ===========================================================================

#[inline(always)]
fn pack_pair(left: u32, right: u32) -> u64 {
    (left as u64) << 32 | right as u64
}

#[inline(always)]
fn fnv1a_u64(key: u64) -> u64 {
    let b = key.to_le_bytes();
    let mut h: u64 = 0xcbf29ce484222325;
    h = (h ^ b[0] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[1] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[2] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[3] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[4] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[5] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[6] as u64).wrapping_mul(0x100000001b3);
    h = (h ^ b[7] as u64).wrapping_mul(0x100000001b3);
    h
}

struct FnvBpe {
    // Token cache: FNV1a open-addressing
    cache_keys: Vec<u64>,
    cache_vals: Vec<TokenId>,
    cache_mask: usize,
    // Merge lookup: FNV1a on packed u64
    merge_keys: Vec<u64>,       // packed (left, right)
    merge_vals: Vec<TokenId>,   // merged token
    merge_ranks: Vec<u32>,      // merge priority
    merge_mask: usize,
}

impl FnvBpe {
    fn new_english_like() -> Self {
        let common_pairs: &[(u8, u8)] = &[
            (b't', b'h'), (b'h', b'e'), (b'i', b'n'), (b'e', b'r'),
            (b'a', b'n'), (b'r', b'e'), (b'o', b'n'), (b'e', b's'),
            (b's', b't'), (b'e', b'n'), (b'a', b't'), (b'o', b'r'),
            (b'n', b'd'), (b't', b'i'), (b'a', b'l'), (b'a', b'r'),
            (b'e', b'd'), (b'i', b's'), (b'o', b'f'), (b'i', b't'),
            (b'n', b'g'), (b'l', b'e'), (b'e', b'l'), (b'o', b'u'),
            (b'n', b't'), (b's', b'e'), (b'r', b'o'), (b'l', b'i'),
            (b'r', b'i'), (b'c', b'o'), (b'c', b'e'), (b'd', b'e'),
            (b'r', b'a'), (b'i', b'o'), (b'c', b't'), (b'l', b'a'),
            (b'u', b'r'), (b'n', b'e'), (b'c', b'h'), (b'l', b'l'),
        ];

        // Merge table: FNV1a open-addressing on packed u64
        let merge_size = 256usize; // power of 2, >2x entries
        let merge_mask = merge_size - 1;
        let mut merge_keys = vec![0u64; merge_size];
        let mut merge_vals = vec![0u32; merge_size];
        let mut merge_ranks = vec![0u32; merge_size];
        let mut next_id = 256u32;
        let mut rank = 1u32;

        let mut insert_merge = |left: u32, right: u32, id: u32, r: u32,
                                 keys: &mut Vec<u64>, vals: &mut Vec<u32>, ranks: &mut Vec<u32>, mask: usize| {
            let key = pack_pair(left, right);
            let h = fnv1a_u64(key);
            let mut slot = (h as usize) & mask;
            while keys[slot] != 0 { slot = (slot + 1) & mask; }
            keys[slot] = key;
            vals[slot] = id;
            ranks[slot] = r;
        };

        for &(a, b) in common_pairs {
            insert_merge(a as u32, b as u32, next_id, rank,
                        &mut merge_keys, &mut merge_vals, &mut merge_ranks, merge_mask);
            next_id += 1; rank += 1;
        }
        let th = 256u32; let he = 257; let in_ = 258; let er = 259;
        let an = 260; let re = 261; let on = 262; let es = 263;
        let st = 264; let en = 265; let at = 266; let or = 267;
        for &(a, b) in &[(th, b'e' as u32), (in_, b'g' as u32), (an, b'd' as u32),
            (er, b's' as u32), (on, b'e' as u32), (re, b's' as u32),
            (st, b'e' as u32), (at, b'e' as u32), (or, b'e' as u32),
            (en, b't' as u32), (es, b't' as u32), (he, b'r' as u32)] {
            insert_merge(a, b, next_id, rank,
                        &mut merge_keys, &mut merge_vals, &mut merge_ranks, merge_mask);
            next_id += 1; rank += 1;
        }

        // Token cache: FNV1a (same as SimpleBpe)
        let cache_size = 4096usize;
        let cache_mask = cache_size - 1;
        let mut cache_keys = vec![0u64; cache_size];
        let mut cache_vals = vec![0u32; cache_size];
        for i in 0u8..=127 {
            let h = fnv1a(&[i]);
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h; cache_vals[slot] = i as u32;
        }
        for &(a, b) in common_pairs {
            let h = fnv1a(&[a, b]);
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            // Look up merged ID from merge table
            let key = pack_pair(a as u32, b as u32);
            let mh = fnv1a_u64(key);
            let mut ms = (mh as usize) & merge_mask;
            let mut merged = 0u32;
            for _ in 0..4 {
                if merge_keys[ms] == key { merged = merge_vals[ms]; break; }
                if merge_keys[ms] == 0 { break; }
                ms = (ms + 1) & merge_mask;
            }
            if merged != 0 { cache_keys[slot] = h; cache_vals[slot] = merged; }
        }
        let mut wid = next_id;
        for word in &["the", "of", "and", "in", "to", "is", "it", "for", "was", "on",
                      "he", "at", "or", "an", "be", "as", "by", "we", "no", "do"] {
            let h = fnv1a(word.as_bytes());
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h; cache_vals[slot] = wid; wid += 1;
        }

        FnvBpe { cache_keys, cache_vals, cache_mask,
                 merge_keys, merge_vals, merge_ranks, merge_mask }
    }

    #[inline(always)]
    fn merge_get(&self, left: u32, right: u32) -> Option<(TokenId, u32)> {
        let key = pack_pair(left, right);
        let h = fnv1a_u64(key);
        let mut slot = (h as usize) & self.merge_mask;
        for _ in 0..4 {
            let k = unsafe { *self.merge_keys.get_unchecked(slot) };
            if k == key {
                return Some((
                    unsafe { *self.merge_vals.get_unchecked(slot) },
                    unsafe { *self.merge_ranks.get_unchecked(slot) },
                ));
            }
            if k == 0 { return None; }
            slot = (slot + 1) & self.merge_mask;
        }
        None
    }

    #[inline(always)]
    fn encode_piece(&self, piece: &[u8], output: &mut Vec<TokenId>) {
        if piece.len() <= 8 {
            let h = fnv1a(piece);
            let mut slot = (h as usize) & self.cache_mask;
            for _ in 0..4 {
                if self.cache_keys[slot] == h { output.push(self.cache_vals[slot]); return; }
                if self.cache_keys[slot] == 0 { break; }
                slot = (slot + 1) & self.cache_mask;
            }
        }
        let mut workspace = [0u32; 64];
        let plen = piece.len().min(64);
        for i in 0..plen { workspace[i] = piece[i] as u32; }
        let mut len = plen;
        let mut changed = true;
        while changed && len > 1 {
            changed = false;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            let mut best_merged = 0u32;
            for i in 0..len - 1 {
                if let Some((merged, r)) = self.merge_get(workspace[i], workspace[i + 1]) {
                    if r < best_rank { best_rank = r; best_pos = i; best_merged = merged; }
                }
            }
            if best_rank < u32::MAX {
                workspace[best_pos] = best_merged;
                for j in (best_pos + 2)..len { workspace[j - 1] = workspace[j]; }
                len -= 1; changed = true;
            }
        }
        for i in 0..len { output.push(workspace[i]); }
    }
}

// ===========================================================================
// Tiered LUT BPE encoder — direct array index for 1-3 byte pieces
// ===========================================================================

struct TieredBpe {
    /// Tier 0: 1-byte pieces → token ID. tier0[byte] = token_id (0 = miss)
    tier0: [TokenId; 256],
    /// Tier 1: 2-byte pieces → token ID. tier1[b0 * 256 + b1] = token_id
    tier1: Vec<TokenId>,  // 65536 entries = 256 KB
    /// Tier 2: 3-byte pieces → token ID. tier2[b0<<16 | b1<<8 | b2] = token_id
    tier2: Vec<TokenId>,  // 16M entries = 64 MB
    /// Fallback: FNV1a hash cache for 4+ byte pieces
    cache_keys: Vec<u64>,
    cache_vals: Vec<TokenId>,
    cache_mask: usize,
    /// Merge LUT (same as SimpleBpe)
    merge_lut: Vec<TokenId>,
    merge_rank: Vec<u32>,
}

impl TieredBpe {
    fn new_english_like() -> Self {
        // Build merge tables (same as SimpleBpe)
        let mut merge_lut = vec![0u32; VOCAB_SIZE * VOCAB_SIZE];
        let mut merge_rank = vec![0u32; VOCAB_SIZE * VOCAB_SIZE];
        let mut next_id = 256u32;
        let mut rank = 1u32;

        let common_pairs: &[(u8, u8)] = &[
            (b't', b'h'), (b'h', b'e'), (b'i', b'n'), (b'e', b'r'),
            (b'a', b'n'), (b'r', b'e'), (b'o', b'n'), (b'e', b's'),
            (b's', b't'), (b'e', b'n'), (b'a', b't'), (b'o', b'r'),
            (b'n', b'd'), (b't', b'i'), (b'a', b'l'), (b'a', b'r'),
            (b'e', b'd'), (b'i', b's'), (b'o', b'f'), (b'i', b't'),
            (b'n', b'g'), (b'l', b'e'), (b'e', b'l'), (b'o', b'u'),
            (b'n', b't'), (b's', b'e'), (b'r', b'o'), (b'l', b'i'),
            (b'r', b'i'), (b'c', b'o'), (b'c', b'e'), (b'd', b'e'),
            (b'r', b'a'), (b'i', b'o'), (b'c', b't'), (b'l', b'a'),
            (b'u', b'r'), (b'n', b'e'), (b'c', b'h'), (b'l', b'l'),
        ];
        for &(a, b) in common_pairs {
            let idx = (a as usize) * VOCAB_SIZE + (b as usize);
            merge_lut[idx] = next_id;
            merge_rank[idx] = rank;
            next_id += 1;
            rank += 1;
        }
        let th = 256u32; let he = 257; let in_ = 258; let er = 259;
        let an = 260; let re = 261; let on = 262; let es = 263;
        let st = 264; let en = 265; let at = 266; let or = 267;
        let pairs2: &[(u32, u32)] = &[
            (th, b'e' as u32), (in_, b'g' as u32), (an, b'd' as u32),
            (er, b's' as u32), (on, b'e' as u32), (re, b's' as u32),
            (st, b'e' as u32), (at, b'e' as u32), (or, b'e' as u32),
            (en, b't' as u32), (es, b't' as u32), (he, b'r' as u32),
        ];
        for &(a, b) in pairs2 {
            if (a as usize) < VOCAB_SIZE && (b as usize) < VOCAB_SIZE {
                let idx = (a as usize) * VOCAB_SIZE + (b as usize);
                merge_lut[idx] = next_id;
                merge_rank[idx] = rank;
                next_id += 1;
                rank += 1;
            }
        }

        // Tier 0: single bytes
        let mut tier0 = [0u32; 256];
        for i in 0u8..=255 {
            tier0[i as usize] = i as u32;
        }

        // Tier 1: 2-byte pieces (65K entries)
        let mut tier1 = vec![0u32; 65536];
        for &(a, b) in common_pairs {
            let merged = merge_lut[(a as usize) * VOCAB_SIZE + (b as usize)];
            if merged != 0 {
                tier1[(a as usize) * 256 + (b as usize)] = merged;
            }
        }

        // Tier 2: 3-byte pieces (16M entries = 64 MB)
        let mut tier2 = vec![0u32; 16 * 1024 * 1024];
        // Populate with known 3-byte tokens (common words)
        let mut wid = next_id;
        let words3: &[&[u8]] = &[
            b"the", b"and", b"for", b"was", b"are", b"but", b"not", b"you",
            b"all", b"can", b"had", b"her", b"one", b"our", b"out", b"has",
            b"his", b"how", b"its", b"may", b"new", b"now", b"old", b"see",
            b"way", b"who", b"did", b"get", b"let", b"say", b"she", b"too",
            b"use", b"ing", b"ent", b"ion", b"est", b"ous", b"ble", b"ful",
        ];
        for word in words3 {
            if word.len() == 3 {
                let idx = (word[0] as usize) << 16 | (word[1] as usize) << 8 | word[2] as usize;
                tier2[idx] = wid;
                wid += 1;
            }
        }

        // Fallback hash cache for 4+ byte pieces
        let cache_size = 4096usize;
        let cache_mask = cache_size - 1;
        let mut cache_keys = vec![0u64; cache_size];
        let mut cache_vals = vec![0u32; cache_size];
        let words4plus: &[&str] = &[
            "that", "with", "have", "this", "will", "your", "from", "they",
            "been", "said", "each", "make", "like", "long", "look", "many",
            "some", "them", "than", "time", "very", "when", "what", "were",
        ];
        for word in words4plus {
            let h = fnv1a(word.as_bytes());
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h;
            cache_vals[slot] = wid;
            wid += 1;
        }
        // Also add 2-letter words to hash cache as fallback
        for word in &["of", "in", "to", "is", "it", "he", "at", "or", "an",
                      "be", "as", "by", "we", "no", "do"] {
            let h = fnv1a(word.as_bytes());
            let mut slot = (h as usize) & cache_mask;
            while cache_keys[slot] != 0 { slot = (slot + 1) & cache_mask; }
            cache_keys[slot] = h;
            // For 2-byte words, use tier1 value if available, else a new id
            let b = word.as_bytes();
            let t1 = tier1[(b[0] as usize) * 256 + b[1] as usize];
            cache_vals[slot] = if t1 != 0 { t1 } else { wid };
            if t1 == 0 { wid += 1; }
        }

        TieredBpe { tier0, tier1, tier2, cache_keys, cache_vals, cache_mask,
                    merge_lut, merge_rank }
    }

    #[inline(always)]
    fn encode_piece(&self, piece: &[u8], output: &mut Vec<TokenId>) {
        // Tiered cache lookup — no hashing for 1-3 byte pieces
        match piece.len() {
            0 => return,
            1 => {
                let id = self.tier0[piece[0] as usize];
                if id != 0 || piece[0] == 0 { // byte 0 maps to token 0
                    output.push(id);
                    return;
                }
            }
            2 => {
                let idx = (piece[0] as usize) * 256 + piece[1] as usize;
                let id = unsafe { *self.tier1.get_unchecked(idx) };
                if id != 0 {
                    output.push(id);
                    return;
                }
            }
            3 => {
                let idx = (piece[0] as usize) << 16
                        | (piece[1] as usize) << 8
                        | piece[2] as usize;
                let id = unsafe { *self.tier2.get_unchecked(idx) };
                if id != 0 {
                    output.push(id);
                    return;
                }
            }
            _ if piece.len() <= 8 => {
                // Hash fallback for 4-8 byte pieces
                let h = fnv1a(piece);
                let mut slot = (h as usize) & self.cache_mask;
                for _ in 0..4 {
                    if self.cache_keys[slot] == h {
                        output.push(self.cache_vals[slot]);
                        return;
                    }
                    if self.cache_keys[slot] == 0 { break; }
                    slot = (slot + 1) & self.cache_mask;
                }
            }
            _ => {}
        }

        // Slow path: byte-level + merge
        let mut workspace = [0u32; 64];
        let plen = piece.len().min(64);
        for i in 0..plen { workspace[i] = piece[i] as u32; }
        let mut len = plen;

        let mut changed = true;
        while changed && len > 1 {
            changed = false;
            let mut best_rank = u32::MAX;
            let mut best_pos = 0;
            let mut best_merged = 0u32;
            for i in 0..len - 1 {
                let left = workspace[i] as usize;
                let right = workspace[i + 1] as usize;
                if left < VOCAB_SIZE && right < VOCAB_SIZE {
                    let idx = left * VOCAB_SIZE + right;
                    let r = unsafe { *self.merge_rank.get_unchecked(idx) };
                    if r != 0 && r < best_rank {
                        best_rank = r;
                        best_pos = i;
                        best_merged = unsafe { *self.merge_lut.get_unchecked(idx) };
                    }
                }
            }
            if best_rank < u32::MAX {
                workspace[best_pos] = best_merged;
                for j in (best_pos + 2)..len { workspace[j - 1] = workspace[j]; }
                len -= 1;
                changed = true;
            }
        }
        for i in 0..len { output.push(workspace[i]); }
    }
}

// ===========================================================================
// SIMD pretokenizer helpers
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn first_non_letter_16(ptr: *const u8) -> usize {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(offset, vdupq_n_u8(26));
    if vminvq_u8(is_letter) == 0xFF { return 16; }
    let not_letter = vmvnq_u8(is_letter);
    static POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powers = vld1q_u8(POWERS.as_ptr());
    let bits = vandq_u8(not_letter, powers);
    let lo = vaddv_u8(vget_low_u8(bits)) as u16;
    let hi = vaddv_u8(vget_high_u8(bits)) as u16;
    (lo | (hi << 8)).trailing_zeros() as usize
}

// ===========================================================================
// Approach 1: Separate pretok → encode (current architecture)
// ===========================================================================

fn encode_separate(text: &str, bpe: &SimpleBpe) -> Vec<TokenId> {
    let mut tokens = Vec::with_capacity(text.len() / 3);
    for piece in Gpt2::new(text) {
        bpe.encode_piece(piece.as_bytes(), &mut tokens);
    }
    tokens
}

// ===========================================================================
// Approach 2: Fused callback pretok + inline encode
// ===========================================================================

fn encode_fused_callback(text: &str, bpe: &SimpleBpe) -> Vec<TokenId> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 3);
    let mut pos = 0;

    while pos < len {
        let start = pos;
        let b = unsafe { *bytes.get_unchecked(pos) };

        if is_ascii_letter(b) {
            pos += 1;
            // SIMD scan letters
            #[cfg(target_arch = "aarch64")]
            unsafe {
                while pos + 16 <= len {
                    let n = first_non_letter_16(bytes.as_ptr().add(pos));
                    pos += n;
                    if n < 16 {
                        if pos < len && *bytes.get_unchecked(pos) >= 0x80 {
                            let (ch, cl) = decode_utf8(&bytes[pos..]);
                            if ch.is_alphabetic() { pos += cl; continue; }
                        }
                        break;
                    }
                }
            }
            while pos < len {
                let b2 = unsafe { *bytes.get_unchecked(pos) };
                if is_ascii_letter(b2) { pos += 1; }
                else if b2 >= 0x80 {
                    let (ch, cl) = decode_utf8(&bytes[pos..]);
                    if ch.is_alphabetic() { pos += cl; } else { break; }
                } else { break; }
            }
            // Check contraction
            if pos < len && bytes[pos] == b'\'' {
                let rem = len - pos;
                if rem >= 2 {
                    let b1 = bytes[pos + 1];
                    let is_c2 = matches!(b1, b's'|b't'|b'd'|b'm')
                        && (rem == 2 || !is_ascii_letter(bytes[pos + 2]));
                    let is_c3 = rem >= 3 && {
                        let b2 = bytes[pos + 2];
                        (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e')
                    };
                    if is_c2 || is_c3 {
                        // Emit letter piece, contraction is next
                        bpe.encode_piece(&bytes[start..pos], &mut tokens);
                        continue;
                    }
                }
            }
        } else if b == b' ' {
            pos += 1;
            if pos < len {
                let next = unsafe { *bytes.get_unchecked(pos) };
                if is_ascii_letter(next) {
                    pos += 1;
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        while pos + 16 <= len {
                            let n = first_non_letter_16(bytes.as_ptr().add(pos));
                            pos += n;
                            if n < 16 {
                                if pos < len && *bytes.get_unchecked(pos) >= 0x80 {
                                    let (ch, cl) = decode_utf8(&bytes[pos..]);
                                    if ch.is_alphabetic() { pos += cl; continue; }
                                }
                                break;
                            }
                        }
                    }
                    while pos < len {
                        let b2 = unsafe { *bytes.get_unchecked(pos) };
                        if is_ascii_letter(b2) { pos += 1; }
                        else if b2 >= 0x80 {
                            let (ch, cl) = decode_utf8(&bytes[pos..]);
                            if ch.is_alphabetic() { pos += cl; } else { break; }
                        } else { break; }
                    }
                    // Check contraction
                    if pos < len && bytes[pos] == b'\'' {
                        let rem = len - pos;
                        if rem >= 2 {
                            let b1 = bytes[pos + 1];
                            let is_c2 = matches!(b1, b's'|b't'|b'd'|b'm')
                                && (rem == 2 || !is_ascii_letter(bytes[pos + 2]));
                            let is_c3 = rem >= 3 && {
                                let b2 = bytes[pos + 2];
                                (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e')
                            };
                            if is_c2 || is_c3 {
                                bpe.encode_piece(&bytes[start..pos], &mut tokens);
                                continue;
                            }
                        }
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&bytes[pos..]);
                    if ch.is_alphabetic() {
                        while pos < len {
                            let b2 = unsafe { *bytes.get_unchecked(pos) };
                            if is_ascii_letter(b2) { pos += 1; }
                            else if b2 >= 0x80 {
                                let (ch2, cl) = decode_utf8(&bytes[pos..]);
                                if ch2.is_alphabetic() { pos += cl; } else { break; }
                            } else { break; }
                        }
                    }
                } else if is_digit(next) {
                    pos += 1;
                    while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) {
                        pos += 1;
                    }
                }
            }
        } else if b == b'\'' {
            let rem = len - pos;
            let mut clen = 0;
            if rem >= 2 {
                let b1 = bytes[pos + 1];
                if matches!(b1, b's'|b't'|b'd'|b'm')
                    && (rem == 2 || !is_ascii_letter(bytes[pos + 2])) {
                    clen = 2;
                } else if rem >= 3 {
                    let b2 = bytes[pos + 2];
                    if (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e') {
                        clen = 3;
                    }
                }
            }
            if clen > 0 { pos += clen; }
            else {
                pos += 1;
                while pos < len {
                    let b2 = unsafe { *bytes.get_unchecked(pos) };
                    if is_ascii_letter(b2)||is_digit(b2)||b2==b' '||b2==b'\n'||b2==b'\r'||b2>=0x80 { break; }
                    pos += 1;
                }
            }
        } else if is_digit(b) {
            pos += 1;
            while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) { pos += 1; }
        } else if b == b'\n' || b == b'\r' {
            pos += 1;
            while pos < len {
                let c = unsafe { *bytes.get_unchecked(pos) };
                if c == b'\n' || c == b'\r' || c == b' ' { pos += 1; } else { break; }
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&bytes[pos..]);
            pos += cl;
            if ch.is_alphabetic() {
                while pos < len {
                    let b2 = unsafe { *bytes.get_unchecked(pos) };
                    if is_ascii_letter(b2) { pos += 1; }
                    else if b2 >= 0x80 {
                        let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                        if ch2.is_alphabetic() { pos += cl2; } else { break; }
                    } else { break; }
                }
            } else if ch.is_whitespace() {
                while pos < len {
                    let c = unsafe { *bytes.get_unchecked(pos) };
                    if c == b' ' || c == b'\n' || c == b'\r' { pos += 1; }
                    else if c >= 0x80 {
                        let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                        if ch2.is_whitespace() { pos += cl2; } else { break; }
                    } else { break; }
                }
            }
        } else {
            pos += 1;
            while pos < len {
                let b2 = unsafe { *bytes.get_unchecked(pos) };
                if is_ascii_letter(b2)||is_digit(b2)||b2==b' '||b2==b'\n'||b2==b'\r'||b2>=0x80 { break; }
                pos += 1;
            }
        }

        // Encode piece inline
        bpe.encode_piece(&bytes[start..pos], &mut tokens);
    }

    tokens
}

// ===========================================================================
// Approach 3: SIMD boundary detection + inline encode
// ===========================================================================

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn classify_and_detect(ptr: *const u8, prev_last: u8) -> (u16, u8) {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let is_letter = vcltq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(26));
    let is_digit = vcltq_u8(vsubq_u8(chunk, vdupq_n_u8(b'0')), vdupq_n_u8(10));
    let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
    let is_nl = vorrq_u8(vceqq_u8(chunk, vdupq_n_u8(b'\n')), vceqq_u8(chunk, vdupq_n_u8(b'\r')));
    let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));
    let is_apos = vceqq_u8(chunk, vdupq_n_u8(b'\''));

    let mut cls = vdupq_n_u8(6);
    cls = vbslq_u8(is_high, vdupq_n_u8(5), cls);
    cls = vbslq_u8(is_apos, vdupq_n_u8(4), cls);
    cls = vbslq_u8(is_nl, vdupq_n_u8(3), cls);
    cls = vbslq_u8(is_space, vdupq_n_u8(2), cls);
    cls = vbslq_u8(is_digit, vdupq_n_u8(1), cls);
    cls = vbslq_u8(is_letter, vdupq_n_u8(0), cls);

    let prev_vec = vdupq_n_u8(prev_last);
    let shifted = vextq_u8(prev_vec, cls, 15);
    let transitions = vmvnq_u8(vceqq_u8(cls, shifted));

    let prev_is_space = vceqq_u8(shifted, vdupq_n_u8(2));
    let prev_is_nl = vceqq_u8(shifted, vdupq_n_u8(3));
    let curr_is_letter = vceqq_u8(cls, vdupq_n_u8(0));
    let curr_is_digit = vceqq_u8(cls, vdupq_n_u8(1));
    let curr_is_nl = vceqq_u8(cls, vdupq_n_u8(3));
    let curr_is_space = vceqq_u8(cls, vdupq_n_u8(2));

    let suppress = vorrq_u8(
        vandq_u8(prev_is_space, vorrq_u8(curr_is_letter, vorrq_u8(curr_is_digit, curr_is_nl))),
        vandq_u8(prev_is_nl, vorrq_u8(curr_is_space, curr_is_nl)),
    );

    let real = vandq_u8(transitions, vmvnq_u8(suppress));

    static POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powers = vld1q_u8(POWERS.as_ptr());
    let bits = vandq_u8(real, powers);
    let lo = vaddv_u8(vget_low_u8(bits)) as u16;
    let hi = vaddv_u8(vget_high_u8(bits)) as u16;

    (lo | (hi << 8), vgetq_lane_u8(cls, 15))
}

#[cfg(target_arch = "aarch64")]
fn encode_fused_simd(text: &str, bpe: &SimpleBpe) -> Vec<TokenId> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::with_capacity(len / 3);

    if len == 0 { return tokens; }

    static CLASS: [u8; 256] = {
        let mut t = [6u8; 256];
        let mut i = b'a'; while i <= b'z' { t[i as usize] = 0; i += 1; }
        i = b'A'; while i <= b'Z' { t[i as usize] = 0; i += 1; }
        i = b'0'; while i <= b'9' { t[i as usize] = 1; i += 1; }
        t[b' ' as usize] = 2; t[b'\n' as usize] = 3; t[b'\r' as usize] = 3;
        t[b'\'' as usize] = 4;
        i = 0x80; loop { t[i as usize] = 5; if i == 0xFF { break; } i += 1; }
        t
    };

    let mut prev_boundary = 0usize;
    let mut pos = 1usize;
    let mut prev_last = CLASS[bytes[0] as usize];

    unsafe {
        while pos + 16 <= len {
            let (mask, last) = classify_and_detect(bytes.as_ptr().add(pos), prev_last);
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                let boundary = pos + bit;
                // Encode the piece from prev_boundary..boundary inline
                bpe.encode_piece(&bytes[prev_boundary..boundary], &mut tokens);
                prev_boundary = boundary;
                m &= m - 1;
            }
            prev_last = last;
            pos += 16;
        }
    }

    // Scalar tail
    for i in pos..len {
        let c = CLASS[bytes[i] as usize];
        if c != prev_last {
            let suppress = matches!((prev_last, c), (2,0)|(2,1)|(2,3)|(3,2)|(3,3));
            if !suppress {
                bpe.encode_piece(&bytes[prev_boundary..i], &mut tokens);
                prev_boundary = i;
            }
        }
        prev_last = c;
    }

    // Final piece
    if prev_boundary < len {
        bpe.encode_piece(&bytes[prev_boundary..len], &mut tokens);
    }

    tokens
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    let bpe = SimpleBpe::new_english_like();

    // Correctness: compare token counts
    let sep_tokens = encode_separate(&text, &bpe);
    let fused_tokens = encode_fused_callback(&text, &bpe);
    println!("Separate:       {} tokens ({} pieces)", sep_tokens.len(),
        Gpt2::new(&text).count());
    println!("Fused callback: {} tokens", fused_tokens.len());

    #[cfg(target_arch = "aarch64")]
    {
        let simd_tokens = encode_fused_simd(&text, &bpe);
        println!("Fused SIMD:     {} tokens (approx — simplified boundary rules)", simd_tokens.len());
    }
    println!();

    let iters = 10;
    let fold_bpe = FoldBpe::new_english_like();

    println!("--- Pretok only ---");
    {
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Pretok:         {mbs:>8.1} MB/s  ({c} pieces)");
    }

    let fnv_cache_fold_merge = FnvCacheFoldMerge::new_english_like();
    let fnv_bpe = FnvBpe::new_english_like();

    println!("\n--- Encode only (pieces pre-collected, 24.5M pieces) ---");
    println!("  cache type    | merge type    | label");
    println!("  --------------|---------------|------");
    let pieces: Vec<&str> = Gpt2::new(&text).collect();

    // 1. FoldHash cache + FoldHash merges (tokie's current architecture)
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for p in &pieces { fold_bpe.encode_piece(p.as_bytes(), &mut tokens); }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("  FoldHash      | FoldHash      | {mbs:>8.1} MB/s  ({n} tokens)  ← tokie now");
    }
    // 2. FNV1a cache + FoldHash merges (isolate cache effect)
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for p in &pieces { fnv_cache_fold_merge.encode_piece(p.as_bytes(), &mut tokens); }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("  FNV1a         | FoldHash      | {mbs:>8.1} MB/s  ({n} tokens)");
    }
    // 3. FNV1a cache + FNV1a packed-u64 merges
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for p in &pieces { fnv_bpe.encode_piece(p.as_bytes(), &mut tokens); }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("  FNV1a         | FNV1a packed  | {mbs:>8.1} MB/s  ({n} tokens)");
    }
    // 4. FNV1a cache + flat LUT merges (best case, only works for small vocab)
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for p in &pieces { bpe.encode_piece(p.as_bytes(), &mut tokens); }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("  FNV1a         | flat LUT      | {mbs:>8.1} MB/s  ({n} tokens)  ← small vocab only");
    }

    println!("\n--- Full pipeline: pretok + encode ---");
    // FoldHash + FoldHash (tokie-like)
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for piece in Gpt2::new(&text) {
                fold_bpe.encode_piece(piece.as_bytes(), &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Fold+Fold:      {mbs:>8.1} MB/s  ({n} tokens)  ← tokie now");
    }
    // FNV1a cache + FoldHash merges
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for piece in Gpt2::new(&text) {
                fnv_cache_fold_merge.encode_piece(piece.as_bytes(), &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("FNV1a+Fold:     {mbs:>8.1} MB/s  ({n} tokens)");
    }
    // FNV1a cache + FNV1a packed merges
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for piece in Gpt2::new(&text) {
                fnv_bpe.encode_piece(piece.as_bytes(), &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("FNV1a+FNV1a:    {mbs:>8.1} MB/s  ({n} tokens)");
    }
    // FNV1a cache + flat LUT merges
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let t = encode_separate(&text, &bpe);
            n = t.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("FNV1a+LUT:      {mbs:>8.1} MB/s  ({n} tokens)  ← small vocab only");
    }

    println!("\n--- Prefetch pipeline: collect boundaries → batch encode ---");

    // Helper: collect all boundaries using the pretokenizer
    fn collect_boundaries(text: &str) -> Vec<u32> {
        let mut bounds = Vec::with_capacity(text.len() / 3);
        for piece in Gpt2::new(text) {
            bounds.push(piece.as_ptr() as u32);
        }
        // Store end of last piece
        bounds.push((text.as_ptr() as usize + text.len()) as u32);
        bounds
    }

    // Better: store (start, end) offsets from callback pretokenizer
    fn collect_offsets(text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut offsets = Vec::with_capacity(len / 3 + 1);
        let mut pos = 0usize;

        while pos < len {
            let start = pos;
            let b = unsafe { *bytes.get_unchecked(pos) };

            if is_ascii_letter(b) {
                pos += 1;
                while pos < len {
                    let b2 = unsafe { *bytes.get_unchecked(pos) };
                    if is_ascii_letter(b2) { pos += 1; }
                    else if b2 >= 0x80 {
                        let (ch, cl) = decode_utf8(&bytes[pos..]);
                        if ch.is_alphabetic() { pos += cl; } else { break; }
                    } else { break; }
                }
                // Check contraction
                if pos < len && bytes[pos] == b'\'' {
                    let rem = len - pos;
                    if rem >= 2 {
                        let b1 = bytes[pos + 1];
                        let is_c2 = matches!(b1, b's'|b't'|b'd'|b'm')
                            && (rem == 2 || !is_ascii_letter(bytes[pos + 2]));
                        let is_c3 = rem >= 3 && {
                            let b2 = bytes[pos + 2];
                            (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e')
                        };
                        if is_c2 || is_c3 {
                            offsets.push(start as u32);
                            continue;
                        }
                    }
                }
            } else if b == b' ' {
                pos += 1;
                if pos < len {
                    let next = unsafe { *bytes.get_unchecked(pos) };
                    if is_ascii_letter(next) {
                        pos += 1;
                        while pos < len {
                            let b2 = unsafe { *bytes.get_unchecked(pos) };
                            if is_ascii_letter(b2) { pos += 1; }
                            else if b2 >= 0x80 {
                                let (ch, cl) = decode_utf8(&bytes[pos..]);
                                if ch.is_alphabetic() { pos += cl; } else { break; }
                            } else { break; }
                        }
                        if pos < len && bytes[pos] == b'\'' {
                            let rem = len - pos;
                            if rem >= 2 {
                                let b1 = bytes[pos + 1];
                                let is_c2 = matches!(b1, b's'|b't'|b'd'|b'm')
                                    && (rem == 2 || !is_ascii_letter(bytes[pos + 2]));
                                let is_c3 = rem >= 3 && {
                                    let b2 = bytes[pos + 2];
                                    (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e')
                                };
                                if is_c2 || is_c3 {
                                    offsets.push(start as u32);
                                    continue;
                                }
                            }
                        }
                    } else if next >= 0x80 {
                        let (ch, _) = decode_utf8(&bytes[pos..]);
                        if ch.is_alphabetic() {
                            while pos < len {
                                let b2 = unsafe { *bytes.get_unchecked(pos) };
                                if is_ascii_letter(b2) { pos += 1; }
                                else if b2 >= 0x80 {
                                    let (ch2, cl) = decode_utf8(&bytes[pos..]);
                                    if ch2.is_alphabetic() { pos += cl; } else { break; }
                                } else { break; }
                            }
                        }
                    } else if is_digit(next) {
                        pos += 1;
                        while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) { pos += 1; }
                    }
                }
            } else if b == b'\'' {
                let rem = len - pos;
                let mut clen = 0;
                if rem >= 2 {
                    let b1 = bytes[pos + 1];
                    if matches!(b1, b's'|b't'|b'd'|b'm')
                        && (rem == 2 || !is_ascii_letter(bytes[pos + 2])) { clen = 2; }
                    else if rem >= 3 {
                        let b2 = bytes[pos + 2];
                        if (b1==b'l'&&b2==b'l')||(b1==b'v'&&b2==b'e')||(b1==b'r'&&b2==b'e') { clen = 3; }
                    }
                }
                if clen > 0 { pos += clen; }
                else {
                    pos += 1;
                    while pos < len {
                        let b2 = unsafe { *bytes.get_unchecked(pos) };
                        if is_ascii_letter(b2)||is_digit(b2)||b2==b' '||b2==b'\n'||b2==b'\r'||b2>=0x80 { break; }
                        pos += 1;
                    }
                }
            } else if is_digit(b) {
                pos += 1;
                while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) { pos += 1; }
            } else if b == b'\n' || b == b'\r' {
                pos += 1;
                while pos < len {
                    let c = unsafe { *bytes.get_unchecked(pos) };
                    if c == b'\n' || c == b'\r' || c == b' ' { pos += 1; } else { break; }
                }
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&bytes[pos..]);
                pos += cl;
                if ch.is_alphabetic() {
                    while pos < len {
                        let b2 = unsafe { *bytes.get_unchecked(pos) };
                        if is_ascii_letter(b2) { pos += 1; }
                        else if b2 >= 0x80 {
                            let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                            if ch2.is_alphabetic() { pos += cl2; } else { break; }
                        } else { break; }
                    }
                } else if ch.is_whitespace() {
                    while pos < len {
                        let c = unsafe { *bytes.get_unchecked(pos) };
                        if c == b' ' || c == b'\n' || c == b'\r' { pos += 1; }
                        else if c >= 0x80 {
                            let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                            if ch2.is_whitespace() { pos += cl2; } else { break; }
                        } else { break; }
                    }
                }
            } else {
                pos += 1;
                while pos < len {
                    let b2 = unsafe { *bytes.get_unchecked(pos) };
                    if is_ascii_letter(b2)||is_digit(b2)||b2==b' '||b2==b'\n'||b2==b'\r'||b2>=0x80 { break; }
                    pos += 1;
                }
            }

            offsets.push(start as u32);
        }
        offsets.push(len as u32); // sentinel end
        offsets
    }

    let bytes = text.as_bytes();

    // Verify collect_offsets matches iterator
    {
        let offsets = collect_offsets(&text);
        let iter_pieces: Vec<&str> = Gpt2::new(&text).collect();
        println!("Offsets: {} boundaries ({} pieces)  Iterator: {} pieces",
            offsets.len(), offsets.len() - 1, iter_pieces.len());
        let mut mismatches = 0;
        for i in 0..iter_pieces.len().min(offsets.len() - 1) {
            let off_piece = &text[offsets[i] as usize..offsets[i + 1] as usize];
            if off_piece != iter_pieces[i] {
                if mismatches < 3 {
                    println!("  mismatch at {i}: offset={off_piece:?} iter={:?}", iter_pieces[i]);
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 { println!("  total mismatches: {mismatches}"); }
        else { println!("  Correctness: OK"); }
    }

    // Baseline: sequential encode from offsets (no prefetch)
    {
        let offsets = collect_offsets(&text);
        let num_pieces = offsets.len() - 1;

        // Warmup
        {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for i in 0..num_pieces {
                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                bpe.encode_piece(piece, &mut tokens);
            }
        }

        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for i in 0..num_pieces {
                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                bpe.encode_piece(piece, &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Sequential:     {mbs:>8.1} MB/s  ({n} tokens)  (offsets → encode, no prefetch)");
    }

    // Prefetch: compute hash for piece i+AHEAD, prefetch cache slot, encode piece i
    {
        let offsets = collect_offsets(&text);
        let num_pieces = offsets.len() - 1;
        const AHEAD: usize = 4; // prefetch distance

        // Warmup
        {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for i in 0..num_pieces {
                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                bpe.encode_piece(piece, &mut tokens);
            }
        }

        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for i in 0..num_pieces {
                // Prefetch cache slot for piece i+AHEAD
                if i + AHEAD < num_pieces {
                    let future_piece = &bytes[offsets[i + AHEAD] as usize..offsets[i + AHEAD + 1] as usize];
                    if future_piece.len() <= 8 {
                        let h = fnv1a(future_piece);
                        let slot = (h as usize) & bpe.cache_mask;
                        unsafe {
                            let ptr = bpe.cache_keys.as_ptr().add(slot) as *const u8;
                            #[cfg(target_arch = "aarch64")]
                            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr);
                            #[cfg(target_arch = "x86_64")]
                            std::arch::asm!("prefetcht0 [{}]", in(reg) ptr);
                        }
                    }
                }
                // Encode current piece
                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                bpe.encode_piece(piece, &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Prefetch({}):    {mbs:>8.1} MB/s  ({n} tokens)  (hash ahead, prefetch slot)", AHEAD);
    }

    // Prefetch distance sweep
    for ahead in [1usize, 2, 4, 8, 16] {
        let offsets = collect_offsets(&text);
        let num_pieces = offsets.len() - 1;

        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for i in 0..num_pieces {
                if i + ahead < num_pieces {
                    let fp = &bytes[offsets[i + ahead] as usize..offsets[i + ahead + 1] as usize];
                    if fp.len() <= 8 {
                        let h = fnv1a(fp);
                        let slot = (h as usize) & bpe.cache_mask;
                        unsafe {
                            let ptr = bpe.cache_keys.as_ptr().add(slot) as *const u8;
                            #[cfg(target_arch = "aarch64")]
                            std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr);
                            #[cfg(target_arch = "x86_64")]
                            std::arch::asm!("prefetcht0 [{}]", in(reg) ptr);
                        }
                    }
                }
                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                bpe.encode_piece(piece, &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Prefetch({:>2}):   {mbs:>8.1} MB/s  ({n} tokens)", ahead);
    }

    // For reference: total time breakdown
    {
        // Phase 1: collect offsets only
        let start = Instant::now();
        let mut num = 0;
        for _ in 0..iters {
            let o = collect_offsets(&text);
            num = o.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("\nPhase 1 (collect offsets): {mbs:>8.1} MB/s  ({} boundaries)", num);
    }

    // =====================================================================
    // Parallel chunking: split text → pretok+encode per thread → merge
    // =====================================================================
    println!("\n--- Parallel chunking: split → pretok+encode per thread → merge ---");

    /// Find a safe chunk boundary near `target` — scan forward for a newline or space.
    /// These are always piece boundaries for GPT-2 pretokenizer.
    fn find_safe_boundary(bytes: &[u8], target: usize) -> usize {
        let mut pos = target;
        while pos < bytes.len() {
            if bytes[pos] == b'\n' || bytes[pos] == b'\r' {
                return pos; // split before the newline
            }
            pos += 1;
        }
        bytes.len()
    }

    /// Split text into `n` chunks at safe boundaries.
    fn chunk_text(text: &str, n: usize) -> Vec<&str> {
        let bytes = text.as_bytes();
        let chunk_size = bytes.len() / n;
        let mut chunks = Vec::with_capacity(n);
        let mut start = 0;
        for i in 0..n {
            let end = if i == n - 1 {
                bytes.len()
            } else {
                find_safe_boundary(bytes, start + chunk_size)
            };
            if start < end {
                // Safety: we're splitting at ASCII boundaries
                chunks.push(unsafe { std::str::from_utf8_unchecked(&bytes[start..end]) });
            }
            start = end;
        }
        chunks
    }

    // Verify: parallel result should match sequential
    {
        let chunks = chunk_text(&text, 4);
        let mut parallel_tokens = Vec::new();
        for chunk in &chunks {
            for piece in Gpt2::new(chunk) {
                bpe.encode_piece(piece.as_bytes(), &mut parallel_tokens);
            }
        }
        let mut seq_tokens = Vec::new();
        for piece in Gpt2::new(&text) {
            bpe.encode_piece(piece.as_bytes(), &mut seq_tokens);
        }
        if parallel_tokens.len() == seq_tokens.len() && parallel_tokens == seq_tokens {
            println!("Correctness: OK (4 chunks, {} tokens)", parallel_tokens.len());
        } else {
            println!("MISMATCH: seq={} parallel={}", seq_tokens.len(), parallel_tokens.len());
            // Find where they diverge
            for i in 0..seq_tokens.len().min(parallel_tokens.len()) {
                if seq_tokens[i] != parallel_tokens[i] {
                    println!("  first diff at token {i}: seq={} par={}", seq_tokens[i], parallel_tokens[i]);
                    break;
                }
            }
        }
    }

    // Single-threaded baseline (for fair comparison)
    {
        let start = Instant::now();
        let mut n = 0;
        for _ in 0..iters {
            let mut tokens = Vec::with_capacity(text.len() / 3);
            for piece in Gpt2::new(&text) {
                bpe.encode_piece(piece.as_bytes(), &mut tokens);
            }
            n = tokens.len();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("1 thread:       {mbs:>8.1} MB/s  ({n} tokens)");
    }

    // Parallel: sweep thread counts
    let bpe_ref = &bpe;
    for num_threads in [2, 4, 6, 8] {
        let chunks = chunk_text(&text, num_threads);
        let start = Instant::now();
        let mut total_tokens = 0usize;
        for _ in 0..iters {
            let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                let handles: Vec<_> = chunks.iter().map(|chunk| {
                    s.spawn(|| {
                        let mut tokens = Vec::with_capacity(chunk.len() / 3);
                        for piece in Gpt2::new(chunk) {
                            bpe_ref.encode_piece(piece.as_bytes(), &mut tokens);
                        }
                        tokens
                    })
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            total_tokens = results.iter().map(|v| v.len()).sum();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let speedup = mbs / 82.0; // approx single-thread baseline
        println!("{num_threads} threads:      {mbs:>8.1} MB/s  ({total_tokens} tokens)  ({speedup:.1}x)");
    }

    // =====================================================================
    // Work-stealing: pretok first → shared piece list → threads grab batches
    // =====================================================================
    println!("\n--- Work-stealing: pretok → shared offsets → atomic batch encode ---");

    // Approach A: Single-thread pretok → work-stealing encode
    {
        let offsets = collect_offsets(&text);
        let num_pieces = offsets.len() - 1;
        println!("Pieces: {num_pieces}  (pretok'd single-thread)");

        for num_threads in [1, 2, 4, 6, 8] {
            for batch_size in [64, 256, 1024, 4096] {
                if num_threads == 1 && batch_size != 256 { continue; } // skip sweeps for 1T

                let start = Instant::now();
                let mut total_tokens = 0usize;
                for _ in 0..iters {
                    let counter = AtomicUsize::new(0);
                    let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                        let handles: Vec<_> = (0..num_threads).map(|_| {
                            let counter = &counter;
                            let offsets = &offsets;
                            s.spawn(move || {
                                let mut tokens = Vec::with_capacity(num_pieces / num_threads * 4);
                                loop {
                                    let batch_start = counter.fetch_add(batch_size, Ordering::Relaxed);
                                    if batch_start >= num_pieces { break; }
                                    let batch_end = (batch_start + batch_size).min(num_pieces);
                                    for i in batch_start..batch_end {
                                        let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                                        bpe_ref.encode_piece(piece, &mut tokens);
                                    }
                                }
                                tokens
                            })
                        }).collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    total_tokens = results.iter().map(|v| v.len()).sum();
                }
                let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
                let speedup = mbs / 84.0;
                if num_threads == 1 {
                    println!("  1T steal:     {mbs:>8.1} MB/s  ({total_tokens} tokens)  (baseline)");
                } else {
                    println!("  {num_threads}T batch={batch_size:<4}  {mbs:>8.1} MB/s  ({total_tokens} tokens)  ({speedup:.1}x)");
                }
            }
        }
    }

    // Approach B: Parallel pretok (chunked) → merge offsets → work-stealing encode
    println!("\n--- Parallel pretok → merge offsets → work-stealing encode ---");
    {
        for num_threads in [2, 4, 8] {
            let chunks = chunk_text(&text, num_threads);
            let batch_size = 256;

            let start = Instant::now();
            let mut total_tokens = 0usize;
            for _ in 0..iters {
                // Phase 1: parallel pretok → collect offsets per chunk
                let chunk_offsets: Vec<(usize, Vec<u32>)> = std::thread::scope(|s| {
                    let handles: Vec<_> = chunks.iter().map(|chunk| {
                        let base = chunk.as_ptr() as usize - text.as_ptr() as usize;
                        s.spawn(move || {
                            let mut offsets = collect_offsets(chunk);
                            // Convert chunk-local offsets to global offsets
                            for o in offsets.iter_mut() {
                                *o += base as u32;
                            }
                            (base, offsets)
                        })
                    }).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

                // Phase 1.5: merge all offsets into a single flat array
                let mut all_offsets: Vec<u32> = Vec::with_capacity(text.len() / 3);
                for (_, offsets) in &chunk_offsets {
                    // Each chunk's offsets include a sentinel end; skip it except for the last chunk
                    let piece_starts = &offsets[..offsets.len() - 1];
                    all_offsets.extend_from_slice(piece_starts);
                }
                all_offsets.push(text.len() as u32); // final sentinel
                let num_pieces = all_offsets.len() - 1;

                // Phase 2: work-stealing encode
                let counter = AtomicUsize::new(0);
                let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                    let handles: Vec<_> = (0..num_threads).map(|_| {
                        let counter = &counter;
                        let all_offsets = &all_offsets;
                        s.spawn(move || {
                            let mut tokens = Vec::with_capacity(num_pieces / num_threads * 4);
                            loop {
                                let batch_start = counter.fetch_add(batch_size, Ordering::Relaxed);
                                if batch_start >= num_pieces { break; }
                                let batch_end = (batch_start + batch_size).min(num_pieces);
                                for i in batch_start..batch_end {
                                    let piece = &bytes[all_offsets[i] as usize..all_offsets[i + 1] as usize];
                                    bpe_ref.encode_piece(piece, &mut tokens);
                                }
                            }
                            tokens
                        })
                    }).collect();
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });
                total_tokens = results.iter().map(|v| v.len()).sum();
            }
            let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
            let speedup = mbs / 84.0;
            println!("  {num_threads}T par-pretok+steal: {mbs:>6.1} MB/s  ({total_tokens} tokens)  ({speedup:.1}x)");
        }
    }

    // Recap: chunk-per-thread (from above) for comparison
    println!("\n--- Comparison: chunk-per-thread vs work-stealing (best configs) ---");
    for num_threads in [2, 4, 8] {
        // Chunk-per-thread
        let chunks = chunk_text(&text, num_threads);
        let start = Instant::now();
        let mut n_chunk = 0usize;
        for _ in 0..iters {
            let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                let handles: Vec<_> = chunks.iter().map(|chunk| {
                    s.spawn(|| {
                        let mut tokens = Vec::with_capacity(chunk.len() / 3);
                        for piece in Gpt2::new(chunk) {
                            bpe_ref.encode_piece(piece.as_bytes(), &mut tokens);
                        }
                        tokens
                    })
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            n_chunk = results.iter().map(|v| v.len()).sum();
        }
        let mbs_chunk = mb * iters as f64 / start.elapsed().as_secs_f64();

        // Work-stealing (single-thread pretok, batch=256)
        let offsets = collect_offsets(&text);
        let num_pieces = offsets.len() - 1;
        let batch_size = 256;
        let start = Instant::now();
        let mut n_steal = 0usize;
        for _ in 0..iters {
            let counter = AtomicUsize::new(0);
            let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..num_threads).map(|_| {
                    let counter = &counter;
                    let offsets = &offsets;
                    s.spawn(move || {
                        let mut tokens = Vec::with_capacity(num_pieces / num_threads * 4);
                        loop {
                            let bs = counter.fetch_add(batch_size, Ordering::Relaxed);
                            if bs >= num_pieces { break; }
                            let be = (bs + batch_size).min(num_pieces);
                            for i in bs..be {
                                let piece = &bytes[offsets[i] as usize..offsets[i + 1] as usize];
                                bpe_ref.encode_piece(piece, &mut tokens);
                            }
                        }
                        tokens
                    })
                }).collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            n_steal = results.iter().map(|v| v.len()).sum();
        }
        let mbs_steal = mb * iters as f64 / start.elapsed().as_secs_f64();

        println!("  {num_threads}T chunk: {mbs_chunk:>6.1} MB/s ({n_chunk})  steal: {mbs_steal:>6.1} MB/s ({n_steal})  (Δ{:+.0}%)",
            (mbs_steal - mbs_chunk) / mbs_chunk * 100.0);
    }

    // =====================================================================
    // Pipelined: producer pretok → shared buffer → consumer encode threads
    // =====================================================================
    println!("\n--- Pipeline: producer pretok → buffer → consumer encode threads ---");

    // Shared buffer: pre-allocated offset array with atomic cursors.
    // Producer writes offsets, advances write_cursor.
    // Consumers grab batches via atomic read_cursor, spin-wait on write_cursor.

    // Wrapper to send raw pointers across threads (safe because we control access)
    #[derive(Clone, Copy)]
    struct SendPtr(usize);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    impl SendPtr {
        #[inline(always)]
        fn write(self, idx: usize, val: u32) { unsafe { *(self.0 as *mut u32).add(idx) = val; } }
        #[inline(always)]
        fn read(self, idx: usize) -> u32 { unsafe { *(self.0 as *const u32).add(idx) } }
        #[inline(always)]
        fn as_mut_slice(self, len: usize) -> &'static mut [u32] { unsafe { std::slice::from_raw_parts_mut(self.0 as *mut u32, len) } }
    }

    for (n_producers, n_consumers) in [(1, 1), (1, 2), (1, 3), (1, 5), (1, 7)] {
        assert_eq!(n_producers, 1, "only single-producer implemented");
        let batch_size: usize = 4096;

        let start = Instant::now();
        let mut total_tokens_final = 0usize;

        for _ in 0..iters {
            // Pre-allocate offset buffer (worst case: 1 piece per byte)
            let capacity = text.len() / 2;
            let mut offset_buf: Vec<u32> = Vec::with_capacity(capacity + 1);
            unsafe { offset_buf.set_len(capacity + 1); }

            let write_cursor = AtomicUsize::new(0);
            let read_cursor = AtomicUsize::new(0);
            let producer_done = AtomicUsize::new(0); // 0 = running, >0 = final piece count

            let results: Vec<Vec<TokenId>> = std::thread::scope(|s| {
                let write_cursor = &write_cursor;
                let read_cursor = &read_cursor;
                let producer_done = &producer_done;
                let buf_ptr = SendPtr(offset_buf.as_mut_ptr() as usize);
                let text_bytes = text.as_bytes();
                let text_ptr = text.as_ptr() as usize;
                let text_len = text.len();

                // Single producer thread: pretokenize and write offsets
                let p = buf_ptr;
                let producer = s.spawn(move || -> usize {
                    let mut wpos = 0usize;
                    let t = unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(text_ptr as *const u8, text_len)) };
                    for piece in Gpt2::new(t) {
                        let piece_start = piece.as_ptr() as usize - text_ptr;
                        p.write(wpos, piece_start as u32);
                        wpos += 1;
                        if wpos & 255 == 0 {
                            write_cursor.store(wpos, Ordering::Release);
                        }
                    }
                    p.write(wpos, text_len as u32);
                    write_cursor.store(wpos, Ordering::Release);
                    producer_done.store(wpos, Ordering::Release);
                    wpos
                });

                // Consumer threads: grab batches and encode
                let consumer_handles: Vec<_> = (0..n_consumers).map(|_| {
                    let cp = buf_ptr;
                    s.spawn(move || {
                        let mut tokens = Vec::with_capacity(capacity / n_consumers * 4);
                        loop {
                            let batch_start = read_cursor.fetch_add(batch_size, Ordering::Relaxed);

                            // Wait for producer to write these offsets
                            loop {
                                let written = write_cursor.load(Ordering::Acquire);
                                if written > batch_start { break; }
                                let done = producer_done.load(Ordering::Acquire);
                                if done > 0 && batch_start >= done { return tokens; }
                                std::hint::spin_loop();
                            }

                            let total_pieces = {
                                let done = producer_done.load(Ordering::Acquire);
                                if done > 0 { done } else { write_cursor.load(Ordering::Acquire) }
                            };

                            if batch_start >= total_pieces { return tokens; }
                            let batch_end = (batch_start + batch_size).min(total_pieces);

                            // Wait for all pieces in this batch to be written
                            loop {
                                let written = write_cursor.load(Ordering::Acquire);
                                if written >= batch_end { break; }
                                let done = producer_done.load(Ordering::Acquire);
                                if done > 0 { break; }
                                std::hint::spin_loop();
                            }

                            for i in batch_start..batch_end {
                                let ps = cp.read(i) as usize;
                                let pe = cp.read(i + 1) as usize;
                                if ps < pe && pe <= text_bytes.len() {
                                    bpe_ref.encode_piece(&text_bytes[ps..pe], &mut tokens);
                                }
                            }
                        }
                    })
                }).collect();

                // Wait for producer
                let _ = producer.join().unwrap();

                // Collect consumer results
                consumer_handles.into_iter().map(|h| h.join().unwrap()).collect()
            });

            total_tokens_final = results.iter().map(|v| v.len()).sum();
        }

        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let total_threads = n_producers + n_consumers;
        let speedup = mbs / 84.0;
        println!("  {n_producers}P+{n_consumers}C ({total_threads}T): {mbs:>6.1} MB/s  ({total_tokens_final} tokens)  ({speedup:.1}x)");
    }

    // Final summary
    println!("\n=== SUMMARY (8-core Apple Silicon, FNV1a+LUT, enwik8 95MB) ===");
    println!("Single-thread baseline:    ~84 MB/s");
    println!("Chunk-per-thread 8T:      ~400 MB/s  (4.8x)");
    println!("Work-stealing 8T:         ~466 MB/s  (5.5x)");
    println!("Pipeline results above.");
}
