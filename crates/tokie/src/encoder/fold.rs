//! Character fold for the byte-level merge core.
//!
//! A byte-level BPE piece reaches the merge loop as bytes, so a CJK
//! character is three symbols that the merges then reassemble. When that
//! reassembly is *predetermined* the loop can start from the character's
//! token instead (HF tokenizers v1 does the same), skipping its merges and
//! their pair lookups.
//!
//! # Proof obligation
//!
//! Folding character `c` (bytes `b0..bk`) into token `T` must not change
//! the BPE result. Replay `c`'s own assembly in isolation (lowest rank,
//! leftmost on ties, exactly as the merge loop picks). Admit the fold only if
//!
//! 1. the bytes collapse to exactly one symbol `T`;
//! 2. (HF's boundary guard) at every step, with current edge symbols
//!    `s_first`/`s_last` and step rank `r`, no merge `(L, s_first)` whose `L`
//!    can end at a character boundary, and no merge `(s_last, R)` whose `R`
//!    can start at one, has rank `<= r`. Then cross-boundary pairs never
//!    pre-empt the assembly: interior pairs only involve `c`'s symbols, and
//!    the lowest of them is always lower than any cross pair;
//! 3. (extra, not in HF) no merge `(L, T)` / `(T, R)` with boundary-capable
//!    `L` / `R` has rank `<= R_max`, the largest rank in `c`'s assembly.
//!    Without it, a seeded `T` could merge with its neighbour *before* an
//!    outside merge of rank in `(rank(L,T), R_max)` that the byte run applies
//!    while `c` is still partly assembled. With it, every merge the byte run
//!    applies outside `c` before `c` is assembled is also the folded run's
//!    next merge (lockstep), and once `c` is assembled the two states are
//!    identical. Id-monotone vocabs (Backtracking: rank = merged id) satisfy
//!    it automatically; merge-index vocabs with "unsafe" merges may not.
//!
//! The argument treats the rest of the sequence as arbitrary symbols, so
//! folding several characters composes by induction (fold one at a time).
//!
//! "Boundary-capable" uses UTF-8 structure: the symbol left of a complete
//! character ends with the previous character's last byte (ASCII or a
//! continuation, `< 0xC0`), the symbol right of it starts with the next
//! character's first byte (ASCII or a lead `>= 0xC2`). Input here is `&[u8]`
//! and may be invalid UTF-8, so the encoder re-checks those two bytes per
//! occurrence and falls back to bytes when they don't hold. Tokens whose
//! bytes are unknown count as boundary-capable on both sides (conservative).

use foldhash::HashMap as FoldHashMap;

use crate::types::TokenId;

/// No fold for this character (emit its bytes).
pub(crate) const NO_FOLD: TokenId = TokenId::MAX;

/// Codepoint -> folded token: a BMP bitmap (1024 rows of 64 bits) with a
/// popcount index into the packed symbols, plus a map for 4-byte chars.
#[derive(Clone)]
pub(crate) struct CharFold {
    rows: Box<[u64]>,
    row_start: Box<[u32]>,
    symbols: Box<[TokenId]>,
    non_bmp: FoldHashMap<u32, TokenId>,
}

/// Build-time verdict counts, for diagnostics.
#[derive(Debug, Default, Clone, Copy)]
pub struct FoldStats {
    /// Characters that fold.
    pub folded: usize,
    /// Assemble to one symbol, but rejected by HF's step guard.
    pub unsafe_step: usize,
    /// Pass HF's guard, rejected by the extra `(L, T)` / `(T, R)` guard.
    pub unsafe_final: usize,
}

impl CharFold {
    /// Build from the pair map (`(left << 32 | right) -> (merged, rank)`,
    /// ranks unique per pair), the encoder's byte -> token table, and every
    /// vocab token's bytes. `None` when nothing folds or `byte_lut` does not
    /// map each byte to a token spelling exactly that byte.
    pub(crate) fn build<'a>(
        pair_lookup: &FoldHashMap<u64, (TokenId, u32)>,
        byte_lut: &[TokenId; 256],
        tokens: impl Iterator<Item = (TokenId, &'a [u8])>,
    ) -> (Option<Self>, FoldStats) {
        let mut stats = FoldStats::default();
        let mut max_id = 0u32;
        for (&k, &(m, _)) in pair_lookup {
            max_id = max_id.max((k >> 32) as u32).max(k as u32).max(m);
        }
        let n_ids = max_id as usize + 1;
        // (first byte, last byte) per id; ids without known bytes stay
        // `None` and count as boundary-capable on both sides.
        let mut edges: Vec<Option<(u8, u8)>> = vec![None; n_ids];
        let mut single: Vec<Option<u8>> = vec![None; n_ids];
        let mut chars: Vec<(char, Vec<u8>)> = Vec::new();
        for (id, bytes) in tokens {
            if let (Some(&f), Some(&l)) = (bytes.first(), bytes.last()) {
                if (id as usize) < n_ids {
                    edges[id as usize] = Some((f, l));
                    if bytes.len() == 1 {
                        single[id as usize] = Some(f);
                    }
                }
            }
            if bytes.len() >= 2 && bytes.len() <= 4 {
                if let Ok(s) = std::str::from_utf8(bytes) {
                    let mut it = s.chars();
                    if let (Some(c), None) = (it.next(), it.next()) {
                        chars.push((c, bytes.to_vec()));
                    }
                }
            }
        }
        // The fold reasons about the bytes each symbol covers, so every base
        // symbol that takes part in merges must spell exactly its byte
        // (base symbols outside every merge are inert).
        for (b, &t) in byte_lut.iter().enumerate() {
            if (t as usize) < n_ids && single[t as usize] != Some(b as u8) {
                return (None, stats);
            }
        }
        let ends_at_boundary = |id: TokenId| edges.get(id as usize).copied().flatten().is_none_or(|(_, l)| l < 0xC0);
        let starts_at_boundary = |id: TokenId| {
            edges.get(id as usize).copied().flatten().is_none_or(|(f, _)| !(0x80..0xC2).contains(&f))
        };
        // Lowest rank at which a symbol can be taken across a character
        // boundary from its right (`left_min`, it is the left operand) or
        // from its left (`right_min`, it is the right operand).
        let mut left_min = vec![u32::MAX; n_ids];
        let mut right_min = vec![u32::MAX; n_ids];
        for (&k, &(_, rank)) in pair_lookup {
            let (a, b) = ((k >> 32) as u32, k as u32);
            if starts_at_boundary(b) {
                left_min[a as usize] = left_min[a as usize].min(rank);
            }
            if ends_at_boundary(a) {
                right_min[b as usize] = right_min[b as usize].min(rank);
            }
        }
        let rank_of = |l: TokenId, r: TokenId| pair_lookup.get(&(((l as u64) << 32) | r as u64)).copied();
        let at = |v: &Vec<u32>, id: TokenId| v.get(id as usize).copied().unwrap_or(u32::MAX);

        let mut bmp = vec![NO_FOLD; 0x10000];
        let mut non_bmp: FoldHashMap<u32, TokenId> = FoldHashMap::default();
        'chars: for (c, bytes) in &chars {
            let mut running: Vec<TokenId> = bytes.iter().map(|&b| byte_lut[b as usize]).collect();
            if running.contains(&TokenId::MAX) {
                continue;
            }
            let mut r_max = 0u32;
            while running.len() > 1 {
                let mut best: Option<(usize, u32, TokenId)> = None;
                for i in 0..running.len() - 1 {
                    if let Some((m, rank)) = rank_of(running[i], running[i + 1]) {
                        if best.is_none_or(|(_, br, _)| rank < br) {
                            best = Some((i, rank, m));
                        }
                    }
                }
                let Some((i, rank, m)) = best else { continue 'chars };
                if rank >= at(&right_min, running[0]) || rank >= at(&left_min, *running.last().unwrap()) {
                    stats.unsafe_step += 1;
                    continue 'chars;
                }
                r_max = r_max.max(rank);
                running[i] = m;
                running.remove(i + 1);
            }
            let t = running[0];
            if at(&right_min, t) <= r_max || at(&left_min, t) <= r_max {
                stats.unsafe_final += 1;
                continue;
            }
            let cp = *c as u32;
            if cp < 0x10000 {
                bmp[cp as usize] = t;
            } else {
                non_bmp.insert(cp, t);
            }
            stats.folded += 1;
        }
        if stats.folded == 0 {
            return (None, stats);
        }
        let mut rows = vec![0u64; 1024];
        for (cp, &t) in bmp.iter().enumerate() {
            if t != NO_FOLD {
                rows[cp >> 6] |= 1u64 << (cp & 63);
            }
        }
        let mut row_start = vec![0u32; 1024];
        let mut seen = 0u32;
        for (row, bits) in rows.iter().enumerate() {
            row_start[row] = seen;
            seen += bits.count_ones();
        }
        let symbols: Vec<TokenId> = bmp.into_iter().filter(|&t| t != NO_FOLD).collect();
        let fold = Self {
            rows: rows.into_boxed_slice(),
            row_start: row_start.into_boxed_slice(),
            symbols: symbols.into_boxed_slice(),
            non_bmp,
        };
        (Some(fold), stats)
    }

    #[inline(always)]
    fn bmp(&self, cp: u32) -> TokenId {
        let (row, col) = ((cp >> 6) as usize, cp & 63);
        let bits = self.rows[row];
        if (bits >> col) & 1 == 0 {
            return NO_FOLD;
        }
        self.symbols[self.row_start[row] as usize + (bits & ((1u64 << col) - 1)).count_ones() as usize]
    }

    /// Folded token of the complete, valid UTF-8 character starting at
    /// `text[i]` (a byte `>= 0xC2`) and its byte length, or `NO_FOLD`.
    /// Truncated, overlong, surrogate and out-of-range sequences never fold
    /// (no folded codepoint decodes from them).
    #[inline(always)]
    pub(crate) fn get(&self, text: &[u8], i: usize) -> (TokenId, usize) {
        let b0 = text[i] as u32;
        let cont = |j: usize| text.get(j).map_or(u32::MAX, |&b| if b & 0xC0 == 0x80 { (b & 0x3F) as u32 } else { u32::MAX });
        if b0 < 0xE0 {
            // C2..DF: 2 bytes, cp >= 0x80 by the lead.
            let c1 = cont(i + 1);
            if c1 == u32::MAX {
                return (NO_FOLD, 0);
            }
            (self.bmp(((b0 & 0x1F) << 6) | c1), 2)
        } else if b0 < 0xF0 {
            let (c1, c2) = (cont(i + 1), cont(i + 2));
            if (c1 | c2) == u32::MAX {
                return (NO_FOLD, 0);
            }
            let cp = ((b0 & 0x0F) << 12) | (c1 << 6) | c2;
            if cp < 0x800 {
                return (NO_FOLD, 0); // overlong; surrogates have no folded bit
            }
            (self.bmp(cp), 3)
        } else if b0 < 0xF5 && !self.non_bmp.is_empty() {
            let (c1, c2, c3) = (cont(i + 1), cont(i + 2), cont(i + 3));
            if (c1 | c2 | c3) == u32::MAX {
                return (NO_FOLD, 0);
            }
            let cp = ((b0 & 0x07) << 18) | (c1 << 12) | (c2 << 6) | c3;
            if !(0x10000..=0x10FFFF).contains(&cp) {
                return (NO_FOLD, 0);
            }
            (self.non_bmp.get(&cp).copied().unwrap_or(NO_FOLD), 4)
        } else {
            (NO_FOLD, 0)
        }
    }
}

/// Seed the merge loop's symbols for `text`: folded characters as one
/// symbol each (only when the bytes on both sides are at a valid character
/// boundary, see the module docs), every other byte as its base token.
#[inline(always)]
pub(crate) fn seed(fold: &CharFold, byte_lut: &[TokenId; 256], text: &[u8], mut put: impl FnMut(TokenId)) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let b = text[i];
        if b >= 0xC2 {
            let (t, l) = fold.get(text, i);
            if t != NO_FOLD {
                let j = i + l;
                let left_ok = i == 0 || text[i - 1] < 0xC0;
                let right_ok = j == n || text[j] < 0x80 || text[j] >= 0xC2;
                if left_ok && right_ok {
                    put(t);
                    i = j;
                    continue;
                }
            }
        }
        put(byte_lut[b as usize]);
        i += 1;
    }
}
