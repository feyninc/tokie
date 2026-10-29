//! Long-lived worker caches for batch encoding.
//!
//! Batch calls used to build a fresh multi-MiB [`PretokenCache`] per
//! worker thread per call — allocating, zeroing, and faulting in the
//! table every time, and discarding the warm entries at the end. This
//! module keeps one cache per CPU alive for the process and leases them
//! to batch workers (and to single-document encodes) for the duration of
//! one call.
//!
//! Cache entries map piece/unit bytes to token ids for a specific
//! tokenizer, so each lease is tagged with the owning tokenizer's
//! generation id. Checkout prefers a slot already warm for the caller's
//! generation; otherwise it takes an empty slot, and only then recycles
//! the least recently used stale slot (clearing it — the same cost as the
//! old fresh build, paid only when more tokenizers are live than slots).
//!
//! Memory bound: at most `max(CPUs, MIN_SLOTS)` [`WorkerCaches`], each
//! allocated on first use (a 2 MiB pretoken table that grows to at most
//! 8 MiB once half full or once seeded, plus the `▁`-unit cache
//! whose zeroed tables are only faulted in by Unigram / SentencePiece
//! models).
//!
//! [`PretokenCache`]: crate::encoder::PretokenCache

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::encoder::WorkerCaches;

/// Monotonic tokenizer generation source (one id per constructed
/// `Tokenizer`).
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Floor on the slot count so a few tokenizers alternating on a
/// low-core machine each keep their own warm caches.
const MIN_SLOTS: usize = 4;

/// Monotonic checkout clock for LRU recycling of stale slots.
static CLOCK: AtomicU64 = AtomicU64::new(1);

pub(crate) struct Slot {
    caches: Option<WorkerCaches>,
    /// Generation of the tokenizer whose entries the caches hold.
    generation: u64,
    /// [`CLOCK`] value at the last checkout.
    last_used: u64,
}

impl Slot {
    const fn empty() -> Self {
        Self { caches: None, generation: 0, last_used: 0 }
    }
}

fn slots() -> &'static [Mutex<Slot>] {
    static SLOTS: OnceLock<Vec<Mutex<Slot>>> = OnceLock::new();
    SLOTS.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        (0..n.max(MIN_SLOTS)).map(|_| Mutex::new(Slot::empty())).collect()
    })
}

/// A leased long-lived set of worker caches; returns to the pool on drop.
/// If every slot is busy (concurrent batch calls), holds private fresh
/// caches instead.
pub(crate) struct CacheLease {
    guard: Option<MutexGuard<'static, Slot>>,
    private: Option<WorkerCaches>,
}

impl CacheLease {
    /// Check out caches valid for `generation`, building or clearing them
    /// as needed (see the module docs for the slot preference order).
    pub(crate) fn checkout(generation: u64) -> Self {
        Self::checkout_in(slots(), generation)
    }

    fn checkout_in(slots: &'static [Mutex<Slot>], generation: u64) -> Self {
        let now = CLOCK.fetch_add(1, Ordering::Relaxed);
        // First pass: a slot whose entries are already this tokenizer's.
        for slot in slots {
            if let Ok(mut guard) = slot.try_lock() {
                if guard.generation == generation && guard.caches.is_some() {
                    guard.last_used = now;
                    return Self { guard: Some(guard), private: None };
                }
            }
        }
        // Second pass: an empty slot, else the least recently used stale one.
        let mut stale: Option<MutexGuard<'static, Slot>> = None;
        for slot in slots {
            if let Ok(mut guard) = slot.try_lock() {
                if guard.caches.is_none() {
                    guard.caches = Some(WorkerCaches::new());
                    guard.generation = generation;
                    guard.last_used = now;
                    return Self { guard: Some(guard), private: None };
                }
                if stale.as_ref().is_none_or(|s| guard.last_used < s.last_used) {
                    stale = Some(guard);
                }
            }
        }
        if let Some(mut guard) = stale {
            guard.caches.as_mut().unwrap().clear();
            guard.generation = generation;
            guard.last_used = now;
            return Self { guard: Some(guard), private: None };
        }
        Self { guard: None, private: Some(WorkerCaches::private()) }
    }

    #[inline]
    pub(crate) fn caches(&mut self) -> &mut WorkerCaches {
        match self.guard {
            Some(ref mut g) => g.caches.as_mut().unwrap(),
            None => self.private.as_mut().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(n: usize) -> &'static [Mutex<Slot>] {
        Box::leak((0..n).map(|_| Mutex::new(Slot::empty())).collect::<Vec<_>>().into_boxed_slice())
    }

    /// Mark the leased pretoken cache with `id` for `piece`.
    fn mark(slots: &'static [Mutex<Slot>], generation: u64, piece: &[u8], id: u32) {
        CacheLease::checkout_in(slots, generation).caches().pretok.insert(piece, &[id]);
    }

    fn hit(slots: &'static [Mutex<Slot>], generation: u64, piece: &[u8]) -> Option<u32> {
        let mut lease = CacheLease::checkout_in(slots, generation);
        let mut out = Vec::new();
        lease.caches().pretok.get(piece, &mut out).then(|| out[0])
    }

    #[test]
    fn busy_pool_falls_back_to_unseeded_private_caches() {
        let slots = pool(1);
        let mut held = CacheLease::checkout_in(slots, 1);
        assert!(held.caches().pretok.wants_seed());
        let mut fallback = CacheLease::checkout_in(slots, 1);
        assert!(fallback.guard.is_none());
        assert!(!fallback.caches().pretok.wants_seed());
    }

    #[test]
    fn alternating_generations_keep_their_caches() {
        let slots = pool(2);
        mark(slots, 1, b"a", 10);
        mark(slots, 2, b"a", 20);
        for _ in 0..3 {
            assert_eq!(hit(slots, 1, b"a"), Some(10));
            assert_eq!(hit(slots, 2, b"a"), Some(20));
        }
    }

    #[test]
    fn third_generation_recycles_least_recently_used() {
        let slots = pool(2);
        mark(slots, 1, b"a", 10);
        mark(slots, 2, b"a", 20);
        assert_eq!(hit(slots, 1, b"a"), Some(10)); // 2 is now LRU
        assert_eq!(hit(slots, 3, b"a"), None); // recycles 2's slot, cleared
        assert_eq!(hit(slots, 1, b"a"), Some(10));
        assert_eq!(hit(slots, 2, b"a"), None);
    }
}
