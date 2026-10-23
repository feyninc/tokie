//! Merge loop for one piece.
//!
//! The lowest rank adjacent pair merges, and that repeats. `merge` scans the
//! rule list. `merge_ids` looks the pair up.

use std::collections::HashMap;

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

const SMALL: usize = 32;

/// Merge `syms` in place. `pairs` maps an adjacent pair to its replacement.
///
/// A piece of 32 symbols or fewer keeps its ranks on the stack and only
/// refreshes the pairs a merge touches. Longer pieces use a heap.
pub fn merge_ids<S: std::hash::BuildHasher>(syms: &mut Vec<u32>, pairs: &HashMap<(u32, u32), Pair, S>) {
    let n = syms.len();
    if n < 2 {
        return;
    }
    if n <= SMALL {
        merge_small(syms, pairs);
    } else {
        merge_heap(syms, pairs);
    }
}

fn lookup<S: std::hash::BuildHasher>(pairs: &HashMap<(u32, u32), Pair, S>, left: u32, right: u32) -> Option<Pair> {
    pairs.get(&(left, right)).copied()
}

fn merge_small<S: std::hash::BuildHasher>(syms: &mut Vec<u32>, pairs: &HashMap<(u32, u32), Pair, S>) {
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
        for i in 0..n - 1 {
            if ranks[i] < best {
                best = ranks[i];
                at = i;
            }
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

fn store(ranks: &mut [u32], merged: &mut [u32], at: usize, pair: Option<Pair>) {
    match pair {
        Some(pair) => {
            ranks[at] = pair.rank;
            merged[at] = pair.id;
        }
        None => ranks[at] = u32::MAX,
    }
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

fn merge_heap<S: std::hash::BuildHasher>(syms: &mut Vec<u32>, pairs: &HashMap<(u32, u32), Pair, S>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let n = syms.len();
    const NONE: usize = usize::MAX;
    let mut next: Vec<usize> = (1..n).chain(std::iter::once(NONE)).collect();
    let mut prev: Vec<usize> = std::iter::once(NONE).chain(0..n - 1).collect();
    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
    for i in 0..n - 1 {
        if let Some(pair) = lookup(pairs, syms[i], syms[i + 1]) {
            heap.push(Reverse((pair.rank, i)));
        }
    }
    while let Some(Reverse((rank, pos))) = heap.pop() {
        let right = next[pos];
        if right == NONE {
            continue;
        }
        let Some(pair) = lookup(pairs, syms[pos], syms[right]) else {
            continue;
        };
        if pair.rank != rank {
            continue;
        }
        syms[pos] = pair.id;
        let far = next[right];
        next[pos] = far;
        if far != NONE {
            prev[far] = pos;
        }
        next[right] = NONE;
        let left = prev[pos];
        if left != NONE
            && let Some(pair) = lookup(pairs, syms[left], syms[pos])
        {
            heap.push(Reverse((pair.rank, left)));
        }
        if next[pos] != NONE
            && let Some(pair) = lookup(pairs, syms[pos], syms[next[pos]])
        {
            heap.push(Reverse((pair.rank, pos)));
        }
    }
    let mut write = 0;
    let mut i = 0;
    loop {
        syms[write] = syms[i];
        write += 1;
        if next[i] == NONE {
            break;
        }
        i = next[i];
    }
    syms.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::{Merge, Pair, merge, merge_ids};
    use std::collections::HashMap;

    fn pairs(rules: &[Merge]) -> HashMap<(u32, u32), Pair> {
        rules
            .iter()
            .map(|rule| {
                (
                    (rule.left, rule.right),
                    Pair {
                        rank: rule.rank,
                        id: rule.id,
                    },
                )
            })
            .collect()
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
    }
}
