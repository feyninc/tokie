//! Character fold for the byte-level merge.
//!
//! A byte-level piece reaches the merge loop as bytes, so a CJK character is
//! three symbols that the merges then reassemble. When that reassembly is
//! predetermined the loop starts from the character's token instead, which is
//! what tokenizers v1 does. The merge still runs on whatever is left.
//!
//! Folding character `c` (raw bytes `b0..bk`) into token `T` keeps the same
//! ids when all three of these hold. They are checked once, at load.
//!
//! 1. Merging `c` by itself, lowest rank and leftmost on a tie, ends at one
//!    symbol `T`.
//! 2. At every step of that assembly, with edge symbols `s_first` / `s_last`
//!    and step rank `r`, no merge `(L, s_first)` whose `L` can end at a
//!    character boundary, and no merge `(s_last, R)` whose `R` can start at
//!    one, has rank `<= r`. A cross-boundary pair therefore cannot fire
//!    before `c` has assembled.
//! 3. No merge `(L, T)` or `(T, R)` with a boundary-capable `L` or `R` has
//!    rank `<=` the largest rank in `c`'s assembly. A seeded `T` then cannot
//!    merge with a neighbour before an outside merge that the byte run would
//!    still apply while `c` is only partly assembled.
//!
//! The rest of the sequence is arbitrary, so folding several characters
//! composes. "Boundary-capable" uses UTF-8: the symbol left of a complete
//! character ends in a byte `< 0xC0`, and the symbol right of it starts with
//! ASCII or a lead `>= 0xC2`. Tokens whose raw bytes are unknown count as
//! boundary-capable on both sides. The encoder also checks the two
//! neighbouring bytes of each occurrence and falls back to bytes when they
//! are not a character boundary.

use crate::bpe::Pairs;

/// No fold for this character.
pub(crate) const NO_FOLD: u32 = u32::MAX;

/// Codepoint to folded token. BMP characters sit in a bitmap. The rare
/// 4-byte characters sit in a map.
pub(crate) struct CharFold {
    rows: Box<[u64]>,
    row_start: Box<[u32]>,
    symbols: Box<[u32]>,
    non_bmp: Vec<(u32, u32)>,
    /// Folds whose scalar is a unified ideograph. A few hundred do not shrink
    /// a Chinese piece; a few thousand do.
    han_folds: u16,
}

impl CharFold {
    /// `tokens` are `(id, raw bytes)` for vocab entries that decode through
    /// the byte-level alphabet. `None` when nothing folds, or when some byte
    /// has no single-byte token.
    pub(crate) fn build<'a>(
        byte_ids: &[u32; 256],
        pairs: &Pairs,
        tokens: impl Iterator<Item = (u32, &'a [u8])>,
    ) -> Option<Self> {
        let mut max_id = 0u32;
        for &(left, right, _, id) in pairs.rows() {
            max_id = max_id.max(left).max(right).max(id);
        }
        let n_ids = max_id as usize + 1;
        let mut edges: Vec<Option<(u8, u8)>> = vec![None; n_ids];
        let mut single: Vec<Option<u8>> = vec![None; n_ids];
        let mut chars: Vec<(char, Vec<u8>)> = Vec::new();
        for (id, bytes) in tokens {
            if let (Some(&first), Some(&last)) = (bytes.first(), bytes.last())
                && (id as usize) < n_ids
            {
                edges[id as usize] = Some((first, last));
                if bytes.len() == 1 {
                    single[id as usize] = Some(first);
                }
            }
            if (2..=4).contains(&bytes.len())
                && let Ok(text) = std::str::from_utf8(bytes)
            {
                let mut iter = text.chars();
                if let (Some(ch), None) = (iter.next(), iter.next()) {
                    chars.push((ch, bytes.to_vec()));
                }
            }
        }
        for (byte, &id) in byte_ids.iter().enumerate() {
            if (id as usize) >= n_ids || single[id as usize] != Some(byte as u8) {
                return None;
            }
        }
        let ends_at_boundary = |id: u32| {
            edges
                .get(id as usize)
                .copied()
                .flatten()
                .is_none_or(|(_, last)| last < 0xC0)
        };
        let starts_at_boundary = |id: u32| {
            edges
                .get(id as usize)
                .copied()
                .flatten()
                .is_none_or(|(first, _)| !(0x80..0xC2).contains(&first))
        };
        let mut left_min = vec![u32::MAX; n_ids];
        let mut right_min = vec![u32::MAX; n_ids];
        for &(left, right, rank, _) in pairs.rows() {
            if starts_at_boundary(right) {
                left_min[left as usize] = left_min[left as usize].min(rank);
            }
            if ends_at_boundary(left) {
                right_min[right as usize] = right_min[right as usize].min(rank);
            }
        }
        let at = |mins: &[u32], id: u32| mins.get(id as usize).copied().unwrap_or(u32::MAX);

        let mut bmp = vec![NO_FOLD; 0x10000];
        let mut non_bmp: Vec<(u32, u32)> = Vec::new();
        let mut folded = 0usize;
        'chars: for (ch, bytes) in &chars {
            let mut running: Vec<u32> = bytes.iter().map(|&byte| byte_ids[byte as usize]).collect();
            if running.contains(&u32::MAX) {
                continue;
            }
            let mut r_max = 0u32;
            while running.len() > 1 {
                let mut best: Option<(usize, u32, u32)> = None;
                for i in 0..running.len() - 1 {
                    if let Some(pair) = pairs.lookup(running[i], running[i + 1])
                        && best.is_none_or(|(_, rank, _)| pair.rank < rank)
                    {
                        best = Some((i, pair.rank, pair.id));
                    }
                }
                let Some((i, rank, id)) = best else {
                    continue 'chars;
                };
                if rank >= at(&right_min, running[0]) || rank >= at(&left_min, *running.last().unwrap()) {
                    continue 'chars;
                }
                r_max = r_max.max(rank);
                running[i] = id;
                running.remove(i + 1);
            }
            let id = running[0];
            if at(&right_min, id) <= r_max || at(&left_min, id) <= r_max {
                continue;
            }
            let cp = *ch as u32;
            if cp < 0x10000 {
                bmp[cp as usize] = id;
            } else {
                non_bmp.push((cp, id));
            }
            folded += 1;
        }
        if folded == 0 {
            return None;
        }
        non_bmp.sort_by_key(|(cp, _)| *cp);
        let mut rows = vec![0u64; 1024];
        let mut han_folds = 0u16;
        for (cp, &id) in bmp.iter().enumerate() {
            if id != NO_FOLD {
                rows[cp >> 6] |= 1u64 << (cp & 63);
                if (0x4E00..=0x9FFF).contains(&cp) {
                    han_folds = han_folds.saturating_add(1);
                }
            }
        }
        let mut row_start = vec![0u32; 1024];
        let mut seen = 0u32;
        for (row, bits) in rows.iter().enumerate() {
            row_start[row] = seen;
            seen += bits.count_ones();
        }
        let symbols: Vec<u32> = bmp.into_iter().filter(|&id| id != NO_FOLD).collect();
        Some(Self {
            rows: rows.into_boxed_slice(),
            row_start: row_start.into_boxed_slice(),
            symbols: symbols.into_boxed_slice(),
            non_bmp,
            han_folds,
        })
    }

    /// The Han block barely folds, so a medium piece is cheaper to walk as
    /// bytes than to fold and merge. A dense fold still shrinks the piece.
    #[inline(always)]
    pub(crate) fn sparse_han(&self) -> bool {
        self.han_folds < 1000
    }

    #[inline(always)]
    fn bmp(&self, cp: u32) -> u32 {
        let (row, col) = ((cp >> 6) as usize, cp & 63);
        let bits = self.rows[row];
        if (bits >> col) & 1 == 0 {
            return NO_FOLD;
        }
        self.symbols[self.row_start[row] as usize + (bits & ((1u64 << col) - 1)).count_ones() as usize]
    }

    /// Folded token of the complete UTF-8 character at `text[i]`, and its
    /// byte length. Truncated, overlong, and out-of-range sequences do not fold.
    #[inline(always)]
    pub(crate) fn get(&self, text: &[u8], i: usize) -> (u32, usize) {
        let b0 = text[i] as u32;
        let cont = |j: usize| {
            text.get(j).map_or(u32::MAX, |&byte| {
                if byte & 0xC0 == 0x80 {
                    (byte & 0x3F) as u32
                } else {
                    u32::MAX
                }
            })
        };
        if b0 < 0xE0 {
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
                return (NO_FOLD, 0);
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
            let id = self
                .non_bmp
                .binary_search_by_key(&cp, |(key, _)| *key)
                .ok()
                .map(|at| self.non_bmp[at].1)
                .unwrap_or(NO_FOLD);
            (id, 4)
        } else {
            (NO_FOLD, 0)
        }
    }
}

/// Starting symbols for `text`. A folded character is one symbol when both
/// neighbours sit on a character boundary. Every other byte is its base token.
#[inline(always)]
pub(crate) fn seed(fold: &CharFold, byte_ids: &[u32; 256], text: &[u8], mut put: impl FnMut(u32)) {
    let n = text.len();
    let mut i = 0;
    while i < n {
        let byte = text[i];
        if byte >= 0xC2 {
            let (id, len) = fold.get(text, i);
            if id != NO_FOLD {
                let end = i + len;
                let left_ok = i == 0 || text[i - 1] < 0xC0;
                let right_ok = end == n || text[end] < 0x80 || text[end] >= 0xC2;
                if left_ok && right_ok {
                    put(id);
                    i = end;
                    continue;
                }
            }
        }
        put(byte_ids[byte as usize]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::seed;
    use crate::bpe::{Pairs, merge_ids};

    /// Three bytes of U+4F60 (你) collapse to one token, and a later ASCII
    /// merge cannot overtake that assembly. The folded start must match the
    /// byte-wise merge.
    #[test]
    fn a_closed_character_folds_without_changing_ids() {
        let bytes = "你ab".as_bytes();
        assert_eq!(&bytes[..3], &[0xE4, 0xBD, 0xA0]);
        let mut byte_ids = [u32::MAX; 256];
        for byte in 0..256 {
            byte_ids[byte] = byte as u32;
        }
        let rows = [
            (0xE4, 0xBD, 0, 300),
            (300, 0xA0, 1, 301),
            (u32::from(b'a'), u32::from(b'b'), 5, 302),
        ];
        let pairs = Pairs::from_list(&rows);
        let tokens: Vec<(u32, Vec<u8>)> = (0..256)
            .map(|byte| (byte as u32, vec![byte as u8]))
            .chain([(300, vec![0xE4, 0xBD]), (301, "你".as_bytes().to_vec()), (302, b"ab".to_vec())])
            .collect();
        let fold = super::CharFold::build(&byte_ids, &pairs, tokens.iter().map(|(id, raw)| (*id, raw.as_slice())))
            .expect("the character is closed");
        let mut folded = Vec::new();
        seed(&fold, &byte_ids, bytes, |id| folded.push(id));
        assert_eq!(folded[0], 301, "你 starts as its token");
        merge_ids(&mut folded, &pairs);
        let mut plain: Vec<u32> = bytes.iter().map(|byte| byte_ids[*byte as usize]).collect();
        merge_ids(&mut plain, &pairs);
        assert_eq!(folded, plain);
    }
}
