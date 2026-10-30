//! Longest-match walk for a long byte-level piece.
//!
//! The merge heap is linear in the number of symbols. A piece that did not
//! fold is one symbol per byte, and Thai and Japanese runs are hundreds of
//! bytes. The walk takes the longest vocab token that is still a valid BPE
//! pair, which is what the heap would emit, in one pass over the bytes.
//!
//! It is built on the first piece that needs it. English and OpenWebText
//! never ask: their longest piece is under 64 bytes. A model at or below
//! 65536 tokens keeps the heap at every length, which is the faster one there.

use crate::bpe::{Pairs, merge_ids};
use daggrs::{DoubleArrayAhoCorasick, MatchKind, Trie};
use std::cell::RefCell;

const MIN_SYMBOLS: usize = 64;
/// `merge_small` keeps its ranks on the stack up to this many symbols.
const STACK_SYMBOLS: usize = 32;

struct Split {
    left: u32,
    right: u32,
}

pub(crate) struct Scan {
    matcher: DoubleArrayAhoCorasick,
    /// Longest token that is a prefix of this token minus its last byte.
    next_prefix: Box<[u32]>,
    lengths: Box<[u16]>,
    split: Box<[Split]>,
    /// Merges that exist. A clear bit is a pair with no rule, so the walk
    /// does not probe the pair table. The table is several megabytes and
    /// knocks the automaton out of the cache.
    merges: MergeBits,
}

/// Bloom filter of real merge pairs. False positives still check the table.
struct MergeBits {
    words: Box<[u64]>,
    mask: usize,
}

struct Scratch {
    tokens: Vec<u32>,
    bits: Vec<u64>,
}

/// Answers of `valid_pair`, kept across pieces on this thread. A long Thai
/// run asks the same token boundary many times, and each miss walks the
/// split tree through the pair table. The walk is never built for English
/// or OpenWebText, so this table stays empty there.
struct Memo {
    key: Vec<u64>,
    bit: Vec<u64>,
}

const MEMO_BITS: u32 = 16;
const MEMO_LEN: usize = 1 << MEMO_BITS;

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch {
        tokens: Vec::new(),
        bits: Vec::new(),
    });
    static PAIR_MEMO: RefCell<Memo> = RefCell::new(Memo {
        key: Vec::new(),
        bit: Vec::new(),
    });
}

impl Scan {
    /// `None` when the merge list is not id-monotone, or when no token is
    /// worth a pattern. The caller keeps the heap in that case.
    pub(crate) fn build(tokens: &[Option<String>], pairs: &Pairs, byte_ids: &[u32; 256]) -> Option<Self> {
        if !monotone(pairs) {
            return None;
        }
        let raw = raw_bytes(tokens)?;
        let mut keep = vec![false; raw.len()];
        for (id, bytes) in raw.iter().enumerate() {
            if bytes.is_empty() || bytes.len() > u16::MAX as usize {
                continue;
            }
            let mut syms = Vec::with_capacity(bytes.len());
            let mut ok = true;
            for &byte in bytes {
                let id = byte_ids[byte as usize];
                if id == u32::MAX {
                    ok = false;
                    break;
                }
                syms.push(id);
            }
            if !ok {
                continue;
            }
            merge_ids(&mut syms, pairs);
            keep[id] = syms.len() == 1 && syms[0] == id as u32;
        }
        if !keep.iter().any(|flag| *flag) {
            return None;
        }

        let mut trie = Trie::new();
        for (id, bytes) in raw.iter().enumerate() {
            if keep[id] {
                trie.add(bytes, id as u32);
            }
        }
        trie.build(MatchKind::LeftmostLongest);
        let matcher = trie.compile();

        let mut next_prefix = vec![u32::MAX; raw.len()].into_boxed_slice();
        let mut lengths = vec![0u16; raw.len()].into_boxed_slice();
        for (id, bytes) in raw.iter().enumerate() {
            if !keep[id] {
                continue;
            }
            lengths[id] = bytes.len() as u16;
            if bytes.len() <= 1 {
                continue;
            }
            if let Some(shorter) = next_match(&matcher, &bytes[..bytes.len() - 1]) {
                next_prefix[id] = shorter;
            }
        }
        let mut split = Vec::with_capacity(raw.len());
        for id in 0..raw.len() as u32 {
            split.push(Split { left: id, right: id });
        }
        for &(left, right, _, id) in pairs.rows() {
            if (id as usize) < split.len() {
                split[id as usize] = Split { left, right };
            }
        }
        Some(Self {
            matcher,
            next_prefix,
            lengths,
            split: split.into_boxed_slice(),
            merges: MergeBits::from_pairs(pairs),
        })
    }

    /// Append the ids of `text` when the walk consumes it. Otherwise leave
    /// `out` alone so the caller can merge.
    pub(crate) fn try_encode(&self, pairs: &Pairs, text: &[u8], out: &mut Vec<u32>) -> bool {
        let n = text.len();
        if n == 0 {
            return true;
        }
        // One borrow for the piece. A Thai run checks the same boundary many
        // times, and taking the memo on each check cost more than the hit.
        PAIR_MEMO.with(|memo| {
            let memo = &mut *memo.borrow_mut();
            SCRATCH.with(|scratch| {
                let mut scratch = scratch.borrow_mut();
                let Scratch { tokens, bits } = &mut *scratch;
                tokens.clear();
                let words = (n + 1 + 63) / 64;
                bits.clear();
                bits.resize(words, u64::MAX);
                let mut pos = 0usize;
                let mut next_token = next_match(&self.matcher, text);
                let mut steps = 0usize;
                let limit = n.saturating_mul(64);
                while let Some(mut token) = next_token {
                    steps += 1;
                    if steps > limit {
                        return false;
                    }
                    let last = tokens.last().copied();
                    loop {
                        steps += 1;
                        if steps > limit {
                            return false;
                        }
                        let token_len = self.lengths[token as usize] as usize;
                        let end_pos = pos + token_len;
                        let reachable = end_pos <= n && bit_set(bits, end_pos);
                        let compatible = last.is_none_or(|prev| {
                            memo_probe(memo, pairs, &self.split, &self.merges, prev, token)
                        });
                        if reachable && compatible && token_len > 0 {
                            tokens.push(token);
                            pos = end_pos;
                            next_token = if pos == n {
                                None
                            } else {
                                next_match(&self.matcher, &text[pos..])
                            };
                            break;
                        } else if let Some(shorter) = self.prefix(token) {
                            token = shorter;
                        } else {
                            bit_clear(bits, pos);
                            if let Some(last_token) = tokens.pop() {
                                pos -= self.lengths[last_token as usize] as usize;
                            }
                            next_token = last;
                            break;
                        }
                    }
                }
                if pos != n {
                    return false;
                }
                out.extend_from_slice(tokens);
                true
            })
        })
    }

    /// Same walk as [`Self::try_encode`] for a piece of at most 64 bytes.
    /// The tokens and the reachability bits stay on the stack. The pair memo
    /// is borrowed once for the piece: the same two tokens recur across a
    /// Chinese run, and the borrow per check cost more than the hit saved.
    pub(crate) fn try_short(&self, pairs: &Pairs, text: &[u8], out: &mut Vec<u32>) -> bool {
        PAIR_MEMO.with(|memo| {
            let memo = &mut *memo.borrow_mut();
            self.try_short_memo(pairs, text, out, memo)
        })
    }

    fn try_short_memo(
        &self,
        pairs: &Pairs,
        text: &[u8],
        out: &mut Vec<u32>,
        memo: &mut Memo,
    ) -> bool {
        let n = text.len();
        let mut tokens = [0u32; 64];
        let mut ntok = 0usize;
        let mut bits = [u64::MAX; 2];
        let mut pos = 0usize;
        let mut next_token = next_match(&self.matcher, text);
        let mut steps = 0usize;
        let limit = n.saturating_mul(64);
        while let Some(mut token) = next_token {
            steps += 1;
            if steps > limit {
                return false;
            }
            let last = if ntok == 0 { None } else { Some(tokens[ntok - 1]) };
            loop {
                steps += 1;
                if steps > limit {
                    return false;
                }
                let token_len = self.lengths[token as usize] as usize;
                let end_pos = pos + token_len;
                let reachable = end_pos <= n && ((bits[end_pos / 64] >> (end_pos % 64)) & 1) == 1;
                let compatible = last.is_none_or(|prev| {
                    memo_probe(memo, pairs, &self.split, &self.merges, prev, token)
                });
                if reachable && compatible && token_len > 0 {
                    tokens[ntok] = token;
                    ntok += 1;
                    pos = end_pos;
                    next_token = if pos == n {
                        None
                    } else {
                        next_match(&self.matcher, &text[pos..])
                    };
                    break;
                } else if let Some(shorter) = self.prefix(token) {
                    token = shorter;
                } else {
                    bits[pos / 64] &= !(1u64 << (pos % 64));
                    if ntok > 0 {
                        ntok -= 1;
                        pos -= self.lengths[tokens[ntok] as usize] as usize;
                    }
                    next_token = last;
                    break;
                }
            }
        }
        if pos != n {
            return false;
        }
        out.extend_from_slice(&tokens[..ntok]);
        true
    }

    fn prefix(&self, token: u32) -> Option<u32> {
        let next = *self.next_prefix.get(token as usize)?;
        if next == u32::MAX { None } else { Some(next) }
    }
}

/// A long piece the character fold did not collapse. English and OpenWebText
/// stay under 64 symbols, and a folded Chinese piece is closer to one symbol
/// per three bytes, so neither builds this.
pub(crate) fn worth_walking(symbols: usize, bytes: usize) -> bool {
    unfolded(symbols, bytes) && symbols > MIN_SYMBOLS
}

/// The walk already exists. The stack merge still handles a short piece.
/// Anything longer that the fold did not collapse reuses the walk instead of
/// the heap. OpenWebText never builds the walk, so its 53-byte piece stays
/// on the heap.
pub(crate) fn worth_reusing(symbols: usize, bytes: usize) -> bool {
    unfolded(symbols, bytes) && symbols > STACK_SYMBOLS
}

/// The walk already exists, and a sparse model folded this piece without
/// bringing it under the stack merge. The queue that then runs is the slow
/// part of a Japanese encode. A dense Han fold stays on that queue: it is
/// already faster than the walk. This does not build the walk.
pub(crate) fn worth_folded_walk(symbols: usize, bytes: usize, ready: bool, sparse: bool) -> bool {
    ready && sparse && symbols > STACK_SYMBOLS && !unfolded(symbols, bytes)
}

/// Walk this piece without first folding it.
///
/// The fold's symbol vector is discarded when the walk succeeds, and on Thai
/// that vector is almost the raw bytes. A Thai piece of 16 bytes is therefore
/// still a stack merge of raw bytes, which is slower than the walk. ASCII
/// stays at the stack-merge limit. This never builds the walk: Chinese and
/// Japanese stay on the fold, and a short ASCII piece never reaches it.
/// English is at most 27 bytes. OpenWebText never builds the walk, so its
/// 53-byte piece stays on the heap.
pub(crate) fn worth_direct(bytes: &[u8], ready: bool) -> bool {
    if !ready {
        return false;
    }
    if bytes.len() > STACK_SYMBOLS && thai_or_ascii(bytes) {
        return true;
    }
    (16..=STACK_SYMBOLS).contains(&bytes.len()) && thai_piece(bytes)
}

/// Thai (U+0E00, lead `0xE0`) or ASCII. Any other lead, including CJK, kana,
/// and fullwidth punctuation, keeps the fold.
/// A medium piece that contains a unified ideograph, on a model whose Han
/// fold is sparse. The fold does not shrink it, and the stack walk avoids
/// the short merge. A punctuation byte in front of the ideograph still
/// counts: the piece is the whole run. This does not build the walk. Kana
/// with no ideograph stays on the fold.
pub(crate) fn worth_sparse_han(bytes: &[u8], ready: bool, sparse: bool) -> bool {
    ready && sparse && (16..=64).contains(&bytes.len()) && han_lead(bytes)
}

fn han_lead(bytes: &[u8]) -> bool {
    bytes.iter().any(|&byte| (0xE4..=0xE9).contains(&byte))
}

fn thai_or_ascii(bytes: &[u8]) -> bool {
    for &byte in bytes {
        if byte < 0x80 {
            continue;
        }
        return byte == 0xE0;
    }
    true
}

fn thai_piece(bytes: &[u8]) -> bool {
    for &byte in bytes {
        if byte < 0x80 {
            continue;
        }
        return byte == 0xE0;
    }
    false
}

fn unfolded(symbols: usize, bytes: usize) -> bool {
    symbols * 4 >= bytes * 3
}

fn monotone(pairs: &Pairs) -> bool {
    let mut prev = 0u32;
    for &(left, right, _, id) in pairs.rows() {
        if id <= left || id <= right || id < prev {
            return false;
        }
        prev = id;
    }
    true
}

fn raw_bytes(tokens: &[Option<String>]) -> Option<Vec<Vec<u8>>> {
    let map = crate::tokenizer::unicode_to_byte();
    let mut raw = Vec::with_capacity(tokens.len());
    for token in tokens {
        let Some(token) = token else {
            raw.push(Vec::new());
            continue;
        };
        let mut bytes = Vec::with_capacity(token.len());
        for ch in token.chars() {
            let Some(&byte) = map.get(&ch) else {
                bytes.clear();
                break;
            };
            bytes.push(byte);
        }
        raw.push(bytes);
    }
    Some(raw)
}

fn next_match(matcher: &DoubleArrayAhoCorasick, text: &[u8]) -> Option<u32> {
    // Every byte of a byte-level vocab is its own token, so a leftmost-longest
    // failure link is dead once the first byte has matched. The longest token
    // that starts at offset 0 is the last output on the direct transitions.
    // The iterator also follows those dead links and builds a match object.
    let states = matcher.states.as_slice();
    let outputs = matcher.outputs.as_slice();
    let mut state = matcher.start_state();
    let mut found = None;
    for &byte in text {
        let Some(current) = states.get(state as usize) else {
            break;
        };
        let child = current.base ^ u32::from(byte);
        let Some(next) = states.get(child as usize) else {
            break;
        };
        if next.check != state {
            break;
        }
        state = child;
        if next.outpos != u32::MAX {
            found = Some(outputs[next.outpos as usize].pattern_id);
        }
    }
    found
}

fn bit_set(bits: &[u64], pos: usize) -> bool {
    (bits[pos / 64] >> (pos % 64)) & 1 == 1
}

fn bit_clear(bits: &mut [u64], pos: usize) {
    let word = pos / 64;
    if word < bits.len() {
        bits[word] &= !(1u64 << (pos % 64));
    }
}

impl MergeBits {
    fn from_pairs(pairs: &Pairs) -> Self {
        let rules = pairs.rows().len().max(1);
        let bits = (rules * 16).next_power_of_two().clamp(1 << 16, 1 << 22);
        let mut words = vec![0u64; bits / 64];
        let mask = bits - 1;
        for &(left, right, _, _) in pairs.rows() {
            let (one, two) = mix(left, right, mask);
            set_bit(&mut words, one);
            set_bit(&mut words, two);
        }
        Self { words: words.into_boxed_slice(), mask }
    }

    #[inline(always)]
    fn may_hold(&self, left: u32, right: u32) -> bool {
        let (one, two) = mix(left, right, self.mask);
        bit_on(&self.words, one) && bit_on(&self.words, two)
    }
}

fn mix(left: u32, right: u32, mask: usize) -> (usize, usize) {
    // The low bits of `(left << 32 | right) * odd` depend only on `right`,
    // so a mask of those bits stays set and the walk probes the pair table.
    // The high bits mix both ids.
    let packed = ((left as u64) << 32) | right as u64;
    let one = packed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let two = packed.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let shift = 64 - mask.trailing_ones();
    ((one >> shift) as usize, (two >> shift) as usize)
}

fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

fn bit_on(words: &[u64], bit: usize) -> bool {
    (words[bit / 64] >> (bit % 64)) & 1 == 1
}

fn memo_probe(
    memo: &mut Memo,
    pairs: &Pairs,
    split: &[Split],
    merges: &MergeBits,
    left: u32,
    right: u32,
) -> bool {
    if memo.key.len() != MEMO_LEN {
        memo.key.resize(MEMO_LEN, 0);
        memo.bit.resize(MEMO_LEN / 64, 0);
    }
    let packed = ((left as u64) << 32) | right as u64;
    let stored = packed.wrapping_add(1);
    if stored == 0 {
        return valid_pair(pairs, split, merges, left, right);
    }
    let slot = (packed.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - MEMO_BITS)) as usize;
    if memo.key[slot] == stored {
        return (memo.bit[slot / 64] >> (slot % 64)) & 1 == 1;
    }
    let ok = valid_pair(pairs, split, merges, left, right);
    memo.key[slot] = stored;
    let word = slot / 64;
    let mask = 1u64 << (slot % 64);
    if ok {
        memo.bit[word] |= mask;
    } else {
        memo.bit[word] &= !mask;
    }
    ok
}

fn valid_pair(
    pairs: &Pairs,
    split: &[Split],
    merges: &MergeBits,
    mut left: u32,
    mut right: u32,
) -> bool {
    let mut limit = u32::MAX;
    for _ in 0..128 {
        if merges.may_hold(left, right)
            && let Some(pair) = pairs.lookup(left, right)
            && pair.id < limit
        {
            return false;
        }
        if left > right {
            limit = left;
            let side = split.get(left as usize).map(|row| row.right).unwrap_or(left);
            if side == left {
                limit = right + 1;
                let side = split.get(right as usize).map(|row| row.left).unwrap_or(right);
                if side + 1 == limit {
                    return true;
                }
                right = side;
            } else {
                left = side;
            }
        } else {
            limit = right + 1;
            let side = split.get(right as usize).map(|row| row.left).unwrap_or(right);
            if side + 1 == limit {
                limit = left;
                let side = split.get(left as usize).map(|row| row.right).unwrap_or(left);
                if side == limit {
                    return true;
                }
                left = side;
            } else {
                right = side;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::Scan;
    use crate::bpe::{Merge, Pairs, merge_ids};

    fn rules() -> Pairs {
        let rows = [
            Merge { left: 0, right: 1, id: 3, rank: 0 },
            Merge { left: 3, right: 2, id: 4, rank: 1 },
            Merge { left: 0, right: 0, id: 5, rank: 2 },
        ];
        let list: Vec<_> = rows.iter().map(|rule| (rule.left, rule.right, rule.rank, rule.id)).collect();
        Pairs::from_list(&list)
    }

    #[test]
    fn a_long_piece_matches_the_merge() {
        let pairs = rules();
        let mut byte_ids = [u32::MAX; 256];
        byte_ids[b'a' as usize] = 0;
        byte_ids[b'b' as usize] = 1;
        byte_ids[b'c' as usize] = 2;
        let tokens: Vec<Option<String>> = ["a", "b", "c", "ab", "abc", "aa"]
            .into_iter()
            .map(|text| Some(text.to_string()))
            .collect();
        let scan = Scan::build(&tokens, &pairs, &byte_ids).expect("monotone vocab walks");
        let piece = b"abc".repeat(30);
        let mut walked = Vec::new();
        assert!(scan.try_encode(&pairs, &piece, &mut walked));
        let short = b"abc".repeat(10);
        let mut short_walked = Vec::new();
        assert!(scan.try_short(&pairs, &short, &mut short_walked));
        let mut short_syms: Vec<u32> = short.iter().map(|byte| byte_ids[*byte as usize]).collect();
        merge_ids(&mut short_syms, &pairs);
        assert_eq!(short_walked, short_syms);
        let mut syms: Vec<u32> = piece.iter().map(|byte| byte_ids[*byte as usize]).collect();
        merge_ids(&mut syms, &pairs);
        assert_eq!(walked, syms);
    }

    #[test]
    fn a_suffix_pattern_does_not_replace_the_prefix() {
        let rows = [(1u32, 2u32, 0u32, 3u32)];
        let pairs = Pairs::from_list(&rows);
        let mut byte_ids = [u32::MAX; 256];
        byte_ids[b'a' as usize] = 0;
        byte_ids[b'b' as usize] = 1;
        byte_ids[b'c' as usize] = 2;
        let tokens: Vec<Option<String>> = ["a", "b", "c", "bc"]
            .into_iter()
            .map(|text| Some(text.to_string()))
            .collect();
        let scan = Scan::build(&tokens, &pairs, &byte_ids).expect("monotone vocab walks");
        let piece = b"abc".repeat(40);
        let mut walked = Vec::new();
        assert!(scan.try_encode(&pairs, &piece, &mut walked));
        let mut syms: Vec<u32> = piece.iter().map(|byte| byte_ids[*byte as usize]).collect();
        merge_ids(&mut syms, &pairs);
        assert_eq!(walked, syms);
    }

    #[test]
    fn reuse_starts_where_the_stack_merge_stops() {
        assert!(!super::worth_reusing(32, 32));
        assert!(super::worth_reusing(33, 33));
        assert!(!super::worth_reusing(40, 120));
        assert!(!super::worth_walking(53, 53));
        assert!(super::worth_walking(65, 65));
        assert!(super::worth_folded_walk(40, 160, true, true));
        assert!(!super::worth_folded_walk(40, 40, true, true));
        assert!(!super::worth_folded_walk(40, 160, true, false));
        assert!(!super::worth_folded_walk(40, 160, false, true));
        assert!(!super::worth_folded_walk(32, 160, true, true));
    }

    #[test]
    fn direct_walk_skips_the_fold_only_for_a_long_non_cjk_piece() {
        let thai = [0xE0u8, 0xB8, 0x81].repeat(30);
        assert!(!super::worth_direct(&thai, false));
        assert!(super::worth_direct(&thai[..40], true));
        assert!(super::worth_direct(&thai[..30], true));
        assert!(!super::worth_direct(&thai[..15], true));
        assert!(!super::worth_direct(&[b'a'; 53], false));
        assert!(!super::worth_direct(&[b'a'; 30], true));
        assert!(super::worth_direct(&[b'a'; 40], true));
        let han = [0xE4u8, 0xBD, 0xA0].repeat(30);
        assert!(!super::worth_direct(&han, true));
        let kana = [0xE3u8, 0x81, 0x82].repeat(30);
        assert!(!super::worth_direct(&kana, true));
        let fullwidth = [0xEFu8, 0xBC, 0x8C].repeat(30);
        assert!(!super::worth_direct(&fullwidth, true));
    }

    #[test]
    fn sparse_han_walks_a_medium_piece_without_building() {
        let han = [0xE4u8, 0xBD, 0xA0].repeat(8);
        assert!(!super::worth_sparse_han(&han, false, true));
        assert!(!super::worth_sparse_han(&han, true, false));
        assert!(super::worth_sparse_han(&han, true, true));
        assert!(!super::worth_sparse_han(&han[..15], true, true));
        assert!(!super::worth_sparse_han(&[0xE4u8, 0xBD, 0xA0].repeat(22), true, true));
        let kana = [0xE3u8, 0x81, 0x82].repeat(8);
        assert!(!super::worth_sparse_han(&kana, true, true));
        assert!(!super::worth_sparse_han(&[b'a'; 24], true, true));
        let mut prefixed = vec![0xEF, 0xBC, 0x8C];
        prefixed.extend(han[..24].iter().copied());
        assert!(super::worth_sparse_han(&prefixed, true, true));
    }
}
