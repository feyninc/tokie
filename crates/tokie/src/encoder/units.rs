//! Metaspace (`▁`) unit splitting shared by the SentencePiece BPE and
//! Unigram encoders.
//!
//! Both encoders see normalized text with spaces replaced by `▁` and no
//! pretokenizer, so a whole document arrives as one string. Splitting it
//! into short units at `▁` boundaries and memoizing each unit turns a
//! document-sized merge loop / lattice into mostly cache hits, but is only
//! exact when no vocab token can span a split point.
//!
//! # Split safety
//!
//! Let `p` be a split point (a `▁` at byte `p` of the text) and suppose some
//! segment — a merged BPE symbol or a Unigram lattice edge — covers bytes
//! `s..e` with `s < p < e`.
//!
//! * Every segment is either a single byte or char (byte fallback, `<unk>`)
//!   or a vocab token. A single byte or char cannot straddle `p`, which is a
//!   char boundary.
//! * If every multi-byte vocab token is valid UTF-8, the token's first byte
//!   is a lead byte, so `s` is a char boundary of the (valid UTF-8) text and
//!   the token's chars align with the text's. `e > p` is then a char
//!   boundary after `p`, so the token contains the whole `▁` at offset
//!   `p - s > 0`, preceded by the complete char that ends at `p`.
//!
//! Hence:
//!
//! * [`UnitSplit::EveryMetaspace`] (split before every `▁`) is exact iff no
//!   token contains `▁` at an offset > 0.
//! * [`UnitSplit::WordStart`] (split only before a `▁` whose preceding char
//!   `c` is not `▁`, so `▁▁▁word` stays one unit) is exact iff no token
//!   contains `c▁`. This admits vocabs with whitespace-run tokens (`▁▁`,
//!   `▁▁▁▁`) such as Llama's. The rule is per preceding char: a vocab whose
//!   only offenders are e.g. Gemma-3's `>▁</` just also skips split points
//!   after `>` (the `blocked` ASCII set).
//!
//! Given no segment spans a split point, the encoders are unit-local:
//! * BPE: no cross-boundary pair has a merge, and the merge queue pops the
//!   global (rank, leftmost) pair, whose restriction to one unit is exactly
//!   the order that unit's own queue would pop — so per-unit output equals
//!   whole-text output.
//! * Unigram: every whole-text path decomposes at `p`, so the best path is
//!   the concatenation of per-unit best paths (up to f64 near-tie order,
//!   same as the existing per-unit design).

use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

use super::UnigramPieceCache;

/// Metaspace character (▁) in UTF-8: E2 96 81.
pub(crate) const METASPACE: [u8; 3] = [0xE2, 0x96, 0x81];

/// Which `▁` boundaries an encoder may split its input at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnitSplit {
    /// Some token spans a word boundary: encode the whole text at once.
    None,
    /// Split before every `▁` (units: `▁`, `▁`, `▁word`).
    EveryMetaspace,
    /// Split before a `▁` preceded by a char that is neither `▁` nor an
    /// ASCII byte in the `blocked` bitmask (units: `▁▁word`).
    WordStart { blocked: u128 },
}

impl UnitSplit {
    /// Classify a vocabulary (see the module docs for the proof).
    pub fn classify<'a, I: IntoIterator<Item = &'a [u8]>>(tokens: I) -> Self {
        let mut every = true;
        let mut blocked = 0u128;
        for bytes in tokens {
            if bytes.len() > 1 && std::str::from_utf8(bytes).is_err() {
                return UnitSplit::None;
            }
            for pos in memchr::memmem::find_iter(bytes, &METASPACE) {
                if pos == 0 {
                    continue;
                }
                every = false;
                if pos >= 3 && bytes[pos - 3..pos] == METASPACE {
                    continue;
                }
                // `c▁` with `c != ▁`: split points after `c` are unsafe.
                // Only ASCII `c` is tracked (one byte identifies the char).
                match bytes[pos - 1] {
                    c @ 0..=0x7F => blocked |= 1u128 << c,
                    _ => return UnitSplit::None,
                }
            }
        }
        if every { UnitSplit::EveryMetaspace } else { UnitSplit::WordStart { blocked } }
    }

    /// Call `f` on each unit of `text` in order. Must not be called with
    /// [`UnitSplit::None`] (the caller encodes the whole text instead).
    #[inline]
    pub(crate) fn for_each_unit(self, text: &[u8], mut f: impl FnMut(&[u8])) {
        debug_assert!(self != UnitSplit::None);
        let (word_start, blocked) = match self {
            UnitSplit::WordStart { blocked } => (true, blocked),
            _ => (false, 0),
        };
        let mut unit_start = 0usize;
        for pos in memchr::memmem::find_iter(text, &METASPACE) {
            if pos <= unit_start {
                continue;
            }
            if word_start {
                // `pos > unit_start >= 0`, so there is a preceding char.
                let prev = text[pos - 1];
                if pos >= 3 && text[pos - 3..pos] == METASPACE {
                    continue;
                }
                if prev < 0x80 && blocked & (1u128 << prev) != 0 {
                    continue;
                }
            }
            f(&text[unit_start..pos]);
            unit_start = pos;
        }
        if unit_start < text.len() {
            f(&text[unit_start..]);
        }
    }
}

/// Process-unique id so thread-local unit caches never alias after an
/// encoder is dropped and another is allocated at the same address.
static NEXT_CACHE_ID: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_cache_id() -> u64 {
    NEXT_CACHE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Unit caches kept per thread, one per recently used encoder. Two lets a
/// thread alternate between two `▁`-unit models (e.g. a Unigram and a
/// SentencePiece BPE tokenizer) without clearing on every call; a third
/// model evicts the least recently used. Slots are allocated on first use,
/// so a single-model thread holds one cache (6 MiB of front tables plus at
/// most 32 MiB of arena/keys, see [`UnigramPieceCache`]).
const THREAD_UNIT_SLOTS: usize = 2;

type UnitSlots = [Option<(u64, UnigramPieceCache)>; THREAD_UNIT_SLOTS];

thread_local! {
    /// Thread-local unit caches tagged by encoder identity so switching
    /// tokenizers on the same thread cannot return another model's ids.
    /// Ordered most recently used first.
    static THREAD_UNIT_CACHE: RefCell<UnitSlots> = const { RefCell::new([const { None }; THREAD_UNIT_SLOTS]) };
}

/// Make `slots[0]` the cache for `cache_id`: promote a matching slot, or
/// recycle (clear + re-tag) the least recently used one.
#[cold]
fn promote_thread_cache(slots: &mut UnitSlots, cache_id: u64) {
    let pos = slots
        .iter()
        .position(|s| matches!(s, Some((id, _)) if *id == cache_id))
        .unwrap_or_else(|| {
            let lru = slots.iter().position(Option::is_none).unwrap_or(THREAD_UNIT_SLOTS - 1);
            match &mut slots[lru] {
                Some((id, cache)) => {
                    cache.clear();
                    *id = cache_id;
                }
                empty => *empty = Some((cache_id, UnigramPieceCache::new())),
            }
            lru
        });
    slots[..=pos].rotate_right(1);
}

/// Run `f` with this thread's unit cache for encoder `cache_id`.
#[inline]
pub(crate) fn with_thread_cache<R>(cache_id: u64, f: impl FnOnce(&mut UnigramPieceCache) -> R) -> R {
    THREAD_UNIT_CACHE.with(|slots| {
        let mut slots = slots.borrow_mut();
        if !matches!(&slots[0], Some((id, _)) if *id == cache_id) {
            promote_thread_cache(&mut slots, cache_id);
        }
        f(&mut slots[0].as_mut().unwrap().1)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(tokens: &[&str]) -> UnitSplit {
        UnitSplit::classify(tokens.iter().map(|t| t.as_bytes()))
    }

    fn units(split: UnitSplit, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        split.for_each_unit(text.as_bytes(), |u| out.push(String::from_utf8(u.to_vec()).unwrap()));
        out
    }

    #[test]
    fn classify_rules() {
        assert_eq!(classify(&["a", "▁a", "▁", "ab"]), UnitSplit::EveryMetaspace);
        assert_eq!(classify(&["▁▁", "▁▁▁▁", "▁a"]), UnitSplit::WordStart { blocked: 0 });
        assert_eq!(
            classify(&["▁a", ">▁</"]),
            UnitSplit::WordStart { blocked: 1u128 << b'>' }
        );
        // Non-ASCII char before an interior `▁`: not tracked -> None.
        assert_eq!(classify(&["é▁"]), UnitSplit::None);
        // Invalid UTF-8 multi-byte token -> None.
        assert_eq!(UnitSplit::classify([&[0xE2u8, 0x96][..]]), UnitSplit::None);
        // Single raw bytes (byte fallback) are fine.
        assert_eq!(UnitSplit::classify([&[0xE2u8][..], "▁".as_bytes()]), UnitSplit::EveryMetaspace);
    }

    fn thread_hit(id: u64, unit: &[u8]) -> Option<Vec<crate::types::TokenId>> {
        with_thread_cache(id, |c| {
            let mut out = Vec::new();
            c.lookup(unit, &mut out).then_some(out)
        })
    }

    #[test]
    fn thread_cache_alternation_keeps_both_models() {
        let (a, b) = (next_cache_id(), next_cache_id());
        let long = "▁a-unit-longer-than-fifteen-bytes".as_bytes();
        with_thread_cache(a, |c| {
            c.insert(b"\xe2\x96\x81the", &[1]);
            c.insert(long, &[1, 2, 3]);
        });
        with_thread_cache(b, |c| c.insert(b"\xe2\x96\x81the", &[2]));
        for _ in 0..3 {
            assert_eq!(thread_hit(a, b"\xe2\x96\x81the"), Some(vec![1]));
            assert_eq!(thread_hit(a, long), Some(vec![1, 2, 3]));
            assert_eq!(thread_hit(b, b"\xe2\x96\x81the"), Some(vec![2]));
            assert_eq!(thread_hit(b, long), None);
        }
    }

    #[test]
    fn thread_cache_evicts_least_recently_used() {
        let (a, b, c) = (next_cache_id(), next_cache_id(), next_cache_id());
        with_thread_cache(a, |k| k.insert(b"x", &[1]));
        with_thread_cache(b, |k| k.insert(b"x", &[2]));
        assert_eq!(thread_hit(a, b"x"), Some(vec![1])); // b is now LRU
        assert_eq!(thread_hit(c, b"x"), None); // recycles b's slot
        assert_eq!(thread_hit(a, b"x"), Some(vec![1]));
        assert_eq!(thread_hit(b, b"x"), None);
    }

    #[test]
    fn unit_boundaries() {
        let t = "▁▁▁a▁b>▁c▁▁d";
        assert_eq!(units(UnitSplit::EveryMetaspace, t), ["▁", "▁", "▁a", "▁b>", "▁c", "▁", "▁d"]);
        assert_eq!(units(UnitSplit::WordStart { blocked: 0 }, t), ["▁▁▁a", "▁b>", "▁c", "▁▁d"]);
        assert_eq!(
            units(UnitSplit::WordStart { blocked: 1u128 << b'>' }, t),
            ["▁▁▁a", "▁b>▁c", "▁▁d"]
        );
        assert_eq!(units(UnitSplit::EveryMetaspace, "ab"), ["ab"]);
        assert_eq!(units(UnitSplit::EveryMetaspace, "x▁y"), ["x", "▁y"]);
    }
}
