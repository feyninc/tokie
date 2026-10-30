//! Merge loop for one piece.
//!
//! The lowest rank adjacent pair merges, and that repeats. `merge` scans the
//! rule list. `merge_ids` looks the pair up in [`Pairs`]: a flat table for
//! the byte-token range, and an open-addressed table after that. The first
//! merges of a piece are byte pairs, so that probe is one indexed load.

/// Both ids below this use the flat table. 512 covers every byte token in
/// the byte-level vocabs (they sit at 0..=258) and the first merges of those
/// bytes. 512 × 512 × 4 bytes is 1 MiB.
const DENSE: u32 = 512;
const NO_RANK: u32 = u32::MAX;

/// One merge. A lower `rank` runs first.
#[derive(Clone, Copy, Debug)]
pub struct Merge {
    pub left: u32,
    pub right: u32,
    pub id: u32,
    pub rank: u32,
}

/// Append the ids for `piece`. Each input byte is its own starting symbol.
pub fn merge(piece: &[u8], merges: &[Merge], out: &mut Vec<u32>) {
    let mut syms: Vec<u32> = piece.iter().map(|byte| u32::from(*byte)).collect();
    while syms.len() > 1 {
        let mut best: Option<(usize, u32, u32)> = None;
        for i in 0..syms.len() - 1 {
            for rule in merges {
                if rule.left == syms[i] && rule.right == syms[i + 1] {
                    let better = best.is_none_or(|(_, rank, _)| rule.rank < rank);
                    if better {
                        best = Some((i, rule.rank, rule.id));
                    }
                }
            }
        }
        let Some((at, _, id)) = best else {
            break;
        };
        syms[at] = id;
        syms.remove(at + 1);
    }
    out.extend_from_slice(&syms);
}

/// One merge already resolved to ids. A lower `rank` runs first.
#[derive(Clone, Copy, Debug)]
pub struct Pair {
    pub rank: u32,
    pub id: u32,
}

/// Merge rules. `rows` keeps the file order for `save`.
///
/// A long piece looks up many pairs that are not rules. The miss stops at an
/// empty slot, so that walk is the merge. Dense ranks are 4 bytes. The other
/// pairs are one `u64` each (`(key + 1) << code_bits | rank`), at load at
/// most one half, so a miss is a short run of single loads. A filter in
/// front of that table rejects a wide miss before the probe. The merged id
/// is read from [`Self::merged_by_rank`] only when a pair actually merges.
/// Ranks are unique: they are the merge's place in the list.
pub struct Pairs {
    /// Rank, or [`NO_RANK`]. Index is `(left << 9) | right`.
    dense: Box<[u32]>,
    slots: Box<[u64]>,
    /// Wide pairs only. A clear bit is not a rule, so `rank_of` does not
    /// probe `slots`. Byte pairs stay on `dense` and never touch this.
    wide: WideBits,
    id_bits: u32,
    code_bits: u32,
    merged_by_rank: Box<[u32]>,
    rows: Vec<(u32, u32, u32, u32)>,
    /// Every use of a merged id has a worse rank than the merge that made it.
    /// Long pieces can then apply every copy of the winning pair in one pass.
    batch: bool,
}

/// Bloom filter of wide merge pairs. False positives still probe `slots`.
struct WideBits {
    words: Box<[u64]>,
    mask: usize,
}

impl WideBits {
    fn from_rows(rows: &[(u32, u32, u32, u32)]) -> Self {
        let wide = rows
            .iter()
            .filter(|&&(left, right, _, _)| left >= DENSE || right >= DENSE)
            .count()
            .max(1);
        let bits = (wide * 16).next_power_of_two().clamp(1 << 16, 1 << 22);
        let mut words = vec![0u64; bits / 64];
        let mask = bits - 1;
        for &(left, right, _, _) in rows {
            if left < DENSE && right < DENSE {
                continue;
            }
            let (one, two) = wide_mix(left, right, mask);
            wide_set(&mut words, one);
            wide_set(&mut words, two);
        }
        Self { words: words.into_boxed_slice(), mask }
    }

    #[inline(always)]
    fn may_hold(&self, left: u32, right: u32) -> bool {
        let (one, two) = wide_mix(left, right, self.mask);
        wide_on(&self.words, one) && wide_on(&self.words, two)
    }
}

/// The low bits of `(left << 32 | right) * odd` depend only on `right`, so a
/// mask of those bits stays set and the probe still runs. The high bits mix
/// both ids. Insert and query share this mix, so a real pair is never rejected.
fn wide_mix(left: u32, right: u32, mask: usize) -> (usize, usize) {
    let packed = ((left as u64) << 32) | right as u64;
    let one = packed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let two = packed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let shift = 64 - mask.trailing_ones();
    ((one >> shift) as usize, (two >> shift) as usize)
}

fn wide_set(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

fn wide_on(words: &[u64], bit: usize) -> bool {
    (words[bit / 64] >> (bit % 64)) & 1 == 1
}

impl Pairs {
    pub fn from_list(rows: &[(u32, u32, u32, u32)]) -> Self {
        let mut max_id = 0u32;
        let mut max_rank = 0u32;
        let mut wide = 0usize;
        for &(left, right, rank, id) in rows {
            max_id = max_id.max(left).max(right).max(id);
            max_rank = max_rank.max(rank);
            if left >= DENSE || right >= DENSE {
                wide += 1;
            }
        }
        let bits = |value: u32| 32 - value.leading_zeros();
        let id_bits = bits(max_id).max(1);
        let code_bits = bits(max_rank).max(1);
        assert!(
            2 * id_bits + 1 + code_bits <= 64,
            "pair ids do not pack into one word"
        );
        // Two slots per wide pair. A fuller table makes a miss walk a longer
        // chain, and a long piece misses far more often than it hits.
        let cap = (wide.max(1) * 2).next_power_of_two();
        let mut pairs = Self {
            dense: vec![NO_RANK; (DENSE as usize) * (DENSE as usize)].into_boxed_slice(),
            slots: vec![0; cap].into_boxed_slice(),
            wide: WideBits::from_rows(rows),
            id_bits,
            code_bits,
            merged_by_rank: vec![0; max_rank as usize + 1].into_boxed_slice(),
            rows: rows.to_vec(),
            batch: batches(rows),
        };
        for &(left, right, rank, id) in rows {
            pairs.place(left, right, rank, id);
        }
        pairs
    }

    /// `(left, right, rank, id)` in insertion order.
    pub fn rows(&self) -> &[(u32, u32, u32, u32)] {
        &self.rows
    }

    #[inline(always)]
    pub fn lookup(&self, left: u32, right: u32) -> Option<Pair> {
        let rank = self.rank_of(left, right);
        if rank == NO_RANK {
            None
        } else {
            Some(Pair {
                rank,
                id: self.merged_by_rank[rank as usize],
            })
        }
    }

    /// Rank of `(left, right)`, or [`NO_RANK`].
    #[inline(always)]
    fn rank_of(&self, left: u32, right: u32) -> u32 {
        if left < DENSE && right < DENSE {
            return self.dense[((left as usize) << 9) | right as usize];
        }
        if (left | right) >> self.id_bits != 0 {
            return NO_RANK;
        }
        if !self.wide.may_hold(left, right) {
            return NO_RANK;
        }
        let packed = (((left as u64) << self.id_bits) | right as u64) + 1;
        let bits = self.code_bits;
        let cap = self.slots.len();
        let mut index = home(packed, cap);
        loop {
            let entry = self.slots[index];
            if entry >> bits == packed {
                return (entry & ((1u64 << bits) - 1)) as u32;
            }
            if entry == 0 {
                return NO_RANK;
            }
            index += 1;
            if index == cap {
                index = 0;
            }
        }
    }

    fn place(&mut self, left: u32, right: u32, rank: u32, id: u32) {
        self.merged_by_rank[rank as usize] = id;
        if left < DENSE && right < DENSE {
            self.dense[((left as usize) << 9) | right as usize] = rank;
            return;
        }
        let packed = (((left as u64) << self.id_bits) | right as u64) + 1;
        let bits = self.code_bits;
        let cap = self.slots.len();
        let mut index = home(packed, cap);
        loop {
            let entry = self.slots[index];
            if entry == 0 || entry >> bits == packed {
                self.slots[index] = (packed << bits) | rank as u64;
                return;
            }
            index += 1;
            if index == cap {
                index = 0;
            }
        }
    }
}

/// Home slot. Multiply-high spreads keys without a second mix.
fn home(key: u64, cap: usize) -> usize {
    ((key.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u128 * cap as u128) >> 64) as usize
}

/// A merged id is only used by later merges. Base symbols have no birth rank.
fn batches(rows: &[(u32, u32, u32, u32)]) -> bool {
    let Some(max) = rows.iter().map(|row| row.3).max() else {
        return true;
    };
    let mut born = vec![u32::MAX; max as usize + 1];
    for &(_, _, rank, id) in rows {
        let slot = &mut born[id as usize];
        if *slot == u32::MAX {
            *slot = rank;
        }
    }
    for &(left, right, rank, _) in rows {
        if left as usize <= max as usize && born[left as usize] != u32::MAX && rank <= born[left as usize] {
            return false;
        }
        if right as usize <= max as usize && born[right as usize] != u32::MAX && rank <= born[right as usize] {
            return false;
        }
    }
    true
}

const SMALL: usize = 32;

/// Merge `syms` in place. `pairs` maps an adjacent pair to its replacement.
///
/// A piece of 32 symbols or fewer keeps its ranks on the stack and only
/// refreshes the pairs a merge touches. A live scan of a longer piece repeats
/// a lookup per symbol per round, and that measured slower than the heap on
/// Thai. Longer pieces use the heap.
pub fn merge_ids(syms: &mut Vec<u32>, pairs: &Pairs) {
    let n = syms.len();
    if n < 2 {
        return;
    }
    if n <= SMALL {
        merge_small(syms, pairs);
    } else {
        merge_long(syms, pairs);
    }
}

/// Long pieces go to the queue. A repeated pair can still merge in one pass,
/// but only when that pass deletes a large share of the piece. A scan that
/// removes a handful of copies costs as much as building the queue, and the
/// queue would still have to run.
fn merge_long(syms: &mut Vec<u32>, pairs: &Pairs) {
    if !merge_queue(syms, pairs) && syms.len() > 1 {
        merge_small(syms, pairs);
    }
}

#[inline(always)]
fn lookup(pairs: &Pairs, left: u32, right: u32) -> Option<Pair> {
    pairs.lookup(left, right)
}

fn merge_small(syms: &mut Vec<u32>, pairs: &Pairs) {
    let n = syms.len();
    let mut next = [0u8; SMALL];
    let mut prev = [0u8; SMALL];
    let mut ranks = [u32::MAX; SMALL];
    let mut merged = [0u32; SMALL];
    for i in 0..n {
        next[i] = (i + 1) as u8;
        prev[i] = (i as u8).wrapping_sub(1);
    }
    for i in 0..n - 1 {
        if let Some(pair) = lookup(pairs, syms[i], syms[i + 1]) {
            ranks[i] = pair.rank;
            merged[i] = pair.id;
        }
    }
    loop {
        let mut best = u32::MAX;
        let mut at = 0usize;
        let mut i = 0usize;
        loop {
            if ranks[i] < best {
                best = ranks[i];
                at = i;
            }
            let step = next[i] as usize;
            if step >= n {
                break;
            }
            i = step;
        }
        if best == u32::MAX {
            break;
        }
        syms[at] = merged[at];
        let dead = next[at] as usize;
        let right = next[dead] as usize;
        next[at] = right as u8;
        ranks[dead] = u32::MAX;
        if right < n {
            prev[right] = at as u8;
            store(&mut ranks, &mut merged, at, lookup(pairs, syms[at], syms[right]));
        } else {
            ranks[at] = u32::MAX;
        }
        let left = prev[at];
        if left != u8::MAX {
            let left = left as usize;
            store(&mut ranks, &mut merged, left, lookup(pairs, syms[left], syms[at]));
        }
    }
    compact(syms, &next, n);
}

fn compact(syms: &mut Vec<u32>, next: &[u8], n: usize) {
    let mut write = 0;
    let mut i = 0;
    loop {
        syms[write] = syms[i];
        write += 1;
        let step = next[i] as usize;
        if step >= n {
            break;
        }
        i = step;
    }
    syms.truncate(write);
}

fn store(ranks: &mut [u32], merged: &mut [u32], at: usize, pair: Option<Pair>) {
    match pair {
        Some(pair) => {
            ranks[at] = pair.rank;
            merged[at] = pair.id;
        }
        None => ranks[at] = u32::MAX,
    }
}

struct QueueScratch {
    next: Vec<u32>,
    prev: Vec<u32>,
    code: Vec<u32>,
    cold: Vec<u64>,
    hot: std::collections::BinaryHeap<std::cmp::Reverse<u64>>,
}

thread_local! {
    static QUEUE: std::cell::RefCell<QueueScratch> = std::cell::RefCell::new(QueueScratch {
        next: Vec::new(),
        prev: Vec::new(),
        code: Vec::new(),
        cold: Vec::new(),
        hot: std::collections::BinaryHeap::new(),
    });
}

/// Linked merge that does not allocate per piece.
///
/// The pairs present at the start sit in a sorted array of `rank << 32 |
/// position`. A merge only creates pairs next to the symbol it wrote, and
/// those go on a small heap. Ranks are unique, so the rank stored at a
/// position is the pair: a popped entry that lost is one integer compare,
/// and the merged id is read only for the pair that wins.
///
/// Returns false when a repeated pair has already shrunk the piece into the
/// stack merge. The heap path returns true and leaves `syms` fully merged.
fn merge_queue(syms: &mut Vec<u32>, pairs: &Pairs) -> bool {
    use std::cmp::Reverse;

    QUEUE.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        loop {
            let n = syms.len();
            if n <= SMALL {
                return false;
            }
            let QueueScratch { next, prev, code, cold, hot } = &mut *scratch;
            let n32 = n as u32;
            next.clear();
            prev.clear();
            code.clear();
            cold.clear();
            hot.clear();
            prev.extend((0..n32).map(|i| i.wrapping_sub(1)));
            next.extend(1..=n32);
            let mut best = NO_RANK;
            let mut copies = 0u32;
            let mut live = 0u32;
            for i in 0..n - 1 {
                let rank = pairs.rank_of(syms[i], syms[i + 1]);
                code.push(rank);
                if rank == NO_RANK {
                    continue;
                }
                live += 1;
                cold.push(((rank as u64) << 32) | i as u64);
                if rank < best {
                    best = rank;
                    copies = 1;
                } else if rank == best {
                    copies += 1;
                }
            }
            code.push(NO_RANK);
            // One pass has to delete at least an eighth of the live pairs.
            // Ranks are unique, so every copy is the same pair, and applying
            // them together is safe only when no later merge outranks it.
            if pairs.batch && copies >= 2 && copies.saturating_mul(8) >= live {
                let merged = pairs.merged_by_rank[best as usize];
                let mut write = 0;
                let mut i = 0;
                while i < n {
                    if i + 1 < n && code[i] == best {
                        syms[write] = merged;
                        i += 2;
                    } else {
                        syms[write] = syms[i];
                        i += 1;
                    }
                    write += 1;
                }
                syms.truncate(write);
                continue;
            }
            cold.sort_unstable();
            let mut ci = 0;
            loop {
                let cold_best = cold.get(ci).copied();
                let hot_best = hot.peek().map(|item| item.0);
                let key = match (cold_best, hot_best) {
                    (Some(cold), Some(hot_key)) if hot_key < cold => hot.pop().unwrap().0,
                    (Some(cold), _) => {
                        ci += 1;
                        cold
                    }
                    (None, Some(_)) => hot.pop().unwrap().0,
                    (None, None) => break,
                };
                let pos = key as u32 as usize;
                let rank = (key >> 32) as u32;
                if code[pos] != rank {
                    continue;
                }
                let right = next[pos] as usize;
                let merged = pairs.merged_by_rank[rank as usize];
                syms[pos] = merged;
                let far = next[right];
                next[right] = NO_RANK;
                code[right] = NO_RANK;
                next[pos] = far;
                if far < n32 {
                    prev[far as usize] = pos as u32;
                    let rank = pairs.rank_of(merged, syms[far as usize]);
                    code[pos] = rank;
                    if rank != NO_RANK {
                        hot.push(Reverse(((rank as u64) << 32) | pos as u64));
                    }
                } else {
                    code[pos] = NO_RANK;
                }
                let left = prev[pos];
                if left < n32 {
                    let rank = pairs.rank_of(syms[left as usize], merged);
                    code[left as usize] = rank;
                    if rank != NO_RANK {
                        hot.push(Reverse(((rank as u64) << 32) | left as u64));
                    }
                }
            }
            let mut write = 0;
            let mut i = 0u32;
            while i < n32 {
                syms[write] = syms[i as usize];
                write += 1;
                i = next[i as usize];
            }
            syms.truncate(write);
            return true;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{Merge, Pairs, merge, merge_ids};

    fn pairs(rules: &[Merge]) -> Pairs {
        let rows: Vec<(u32, u32, u32, u32)> = rules
            .iter()
            .map(|rule| (rule.left, rule.right, rule.rank, rule.id))
            .collect();
        Pairs::from_list(&rows)
    }

    fn assert_same(piece: &[u8], rules: &[Merge]) {
        let mut slow = Vec::new();
        merge(piece, rules, &mut slow);
        let mut fast: Vec<u32> = piece.iter().map(|byte| u32::from(*byte)).collect();
        merge_ids(&mut fast, &pairs(rules));
        assert_eq!(fast, slow, "{}", String::from_utf8_lossy(piece));
    }

    #[test]
    fn fast_merge_matches_the_scan() {
        let rules = [
            Merge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                id: 256,
                rank: 2,
            },
            Merge {
                left: 256,
                right: u32::from(b'c'),
                id: 257,
                rank: 0,
            },
            Merge {
                left: u32::from(b'a'),
                right: u32::from(b'a'),
                id: 258,
                rank: 1,
            },
        ];
        assert_same(b"abc", &rules);
        assert_same(b"aaab", &rules);
        assert_same(b"ababc", &rules);
        let mut long = vec![b'a'; 40];
        long.extend_from_slice(b"bc");
        assert_same(&long, &rules);
        assert_same(b"abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz", &rules);
        let wide = [
            Merge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                id: 600,
                rank: 0,
            },
            Merge {
                left: 600,
                right: u32::from(b'a'),
                id: 700,
                rank: 1,
            },
        ];
        assert_same(b"abababababababababababababababababababab", &wide);
        let repeated = [
            Merge {
                left: u32::from(b'a'),
                right: u32::from(b'a'),
                id: 256,
                rank: 0,
            },
            Merge {
                left: 256,
                right: 256,
                id: 257,
                rank: 1,
            },
        ];
        assert_same(&vec![b'a'; 80], &repeated);
        assert_same(b"abababababababababababababababab", &wide);
        assert_same(b"aaa", &repeated);
        assert_same(&vec![b'a'; 31], &repeated);
        // The second merge outranks the pair that created its left symbol,
        // so the copies of `ab` must merge one at a time.
        let ordered = [
            Merge {
                left: u32::from(b'a'),
                right: u32::from(b'b'),
                id: 256,
                rank: 5,
            },
            Merge {
                left: 256,
                right: u32::from(b'a'),
                id: 257,
                rank: 1,
            },
        ];
        assert_same(b"abab", &ordered);
        assert_same(b"abababababababab", &ordered);
    }
}
