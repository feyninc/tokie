//! Long-lived worker caches for batch encoding.
//!
//! Batch calls used to build a fresh multi-MiB [`PretokenCache`] per
//! worker thread per call — allocating, zeroing, and faulting in the
//! table every time, and discarding the warm entries at the end. This
//! module keeps one cache per CPU alive for the process and leases them
//! to batch workers for the duration of one call.
//!
//! Cache entries map piece bytes to token ids for a specific tokenizer,
//! so each lease is tagged with the owning tokenizer's generation id; a
//! lease checked out under a different generation clears the table first
//! (the same cost as the old fresh build, paid only on tokenizer switch).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::encoder::PretokenCache;

/// Monotonic tokenizer generation source (one id per constructed
/// `Tokenizer`).
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

pub(crate) fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

struct Slot {
    cache: Option<PretokenCache>,
    /// Generation of the tokenizer whose entries the cache holds.
    generation: u64,
}

fn slots() -> &'static [Mutex<Slot>] {
    static SLOTS: OnceLock<Vec<Mutex<Slot>>> = OnceLock::new();
    SLOTS.get_or_init(|| {
        let n = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
        (0..n)
            .map(|_| Mutex::new(Slot { cache: None, generation: 0 }))
            .collect()
    })
}

/// A leased long-lived cache; returns to the pool on drop. If every slot
/// is busy (concurrent batch calls), holds a private fresh cache instead.
pub(crate) struct CacheLease {
    guard: Option<MutexGuard<'static, Slot>>,
    private: Option<PretokenCache>,
}

impl CacheLease {
    /// Check out a cache valid for `generation`, building or clearing it
    /// as needed. Prefers a slot already warm for this generation so that
    /// tokenizers alternating in one process each keep their own warm
    /// table instead of clearing each other's on every checkout.
    pub(crate) fn checkout(generation: u64) -> Self {
        // First pass: a slot whose entries are already this tokenizer's.
        for slot in slots() {
            if let Ok(guard) = slot.try_lock() {
                if guard.generation == generation && guard.cache.is_some() {
                    return Self { guard: Some(guard), private: None };
                }
            }
        }
        // Second pass: any free slot, preferring empty over stale ones.
        let mut stale_guard = None;
        for slot in slots() {
            if let Ok(mut guard) = slot.try_lock() {
                if guard.cache.is_none() {
                    guard.cache = Some(PretokenCache::new());
                    guard.generation = generation;
                    return Self { guard: Some(guard), private: None };
                }
                if stale_guard.is_none() {
                    stale_guard = Some(guard);
                }
            }
        }
        if let Some(mut guard) = stale_guard {
            guard.cache.as_mut().unwrap().clear();
            guard.generation = generation;
            return Self { guard: Some(guard), private: None };
        }
        Self { guard: None, private: Some(PretokenCache::new()) }
    }

    #[inline]
    pub(crate) fn cache(&mut self) -> &mut PretokenCache {
        match self.guard {
            Some(ref mut g) => g.cache.as_mut().unwrap(),
            None => self.private.as_mut().unwrap(),
        }
    }
}
