//! Piece to token ids.
//!
//! A piece of 15 bytes or fewer packs into one `u128`. The short table is
//! open addressing over 32-byte entries, two to a cache line. Encodings of
//! four ids or fewer sit in the entry when the first id fits in 24 bits.
//! The rest sit in an arena. A piece of 16 to 32 bytes uses a second table.
//! One that contains a unified ideograph is stored on its second sight:
//! the first sight is almost always the last, and keeping it grows that
//! table until the encode rehashes it. A longer piece is not stored.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use rustc_hash::FxHashMap;
use std::ptr::NonNull;

/// Bit 7 of the packed value. The low 7 bits are the token count.
const SPILL: u64 = 0x80;
/// Bits in the first-sight filter for 16 to 32 byte pieces.
const SEEN_BITS: usize = 1 << 18;
const SEEN_WORDS: usize = SEEN_BITS / 64;

#[derive(Clone, Copy)]
#[repr(C)]
struct Entry {
    key: u128,
    lo: u64,
    hi: u64,
}

const _: () = assert!(std::mem::size_of::<Entry>() == 32);

struct Table {
    ptr: NonNull<Entry>,
    cap: usize,
}

// The pointer owns the block `Drop` frees. Moving a table moves that block.
unsafe impl Send for Table {}

impl Table {
    fn new(cap: usize) -> Self {
        let layout = table_layout(cap);
        let raw = unsafe { alloc(layout) };
        let Some(ptr) = NonNull::new(raw) else {
            handle_alloc_error(layout);
        };
        // Hint huge pages before the first touch. Zeroing first faults the
        // mapping in as 4 KiB pages, and Zen then drops prefetches that miss
        // the dTLB.
        hugepage(ptr.as_ptr(), layout.size());
        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, layout.size()) };
        Self {
            ptr: ptr.cast(),
            cap,
        }
    }

    fn get(&self, index: usize) -> Entry {
        unsafe { *self.ptr.as_ptr().add(index) }
    }

    fn set(&mut self, index: usize, entry: Entry) {
        unsafe { *self.ptr.as_ptr().add(index) = entry };
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr().cast(), table_layout(self.cap)) };
    }
}

/// Cache of encoded pieces. The caller reuses one of these across calls.
pub struct Cache {
    table: Table,
    mask: usize,
    len: usize,
    /// Hardware CRC32C is the hash when the CPU has SSE4.2. Inserts and
    /// lookups in one process must agree, so this is fixed at construction.
    crc: bool,
    arena: Vec<u32>,
    /// Pieces of 16 to 32 bytes. Longer pieces are not stored.
    long: FxHashMap<Vec<u8>, Vec<u32>>,
    /// First sights of a 16 to 32 byte piece. The second sight stores it.
    seen: Box<[u64]>,
}

impl Cache {
    pub fn new() -> Self {
        let cap = 1024;
        Self {
            table: Table::new(cap),
            mask: cap - 1,
            len: 0,
            crc: crc_available(),
            arena: Vec::new(),
            long: FxHashMap::default(),
            seen: vec![0; SEEN_WORDS].into_boxed_slice(),
        }
    }

    pub fn crc(&self) -> bool {
        self.crc
    }

    /// A second cache with the same entries. `fork` uses this for a loaded piece list.
    pub(crate) fn duplicate(&self) -> Self {
        let table = Table::new(self.table.cap);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.table.ptr.as_ptr(),
                table.ptr.as_ptr(),
                self.table.cap,
            );
        }
        Self {
            table,
            mask: self.mask,
            len: self.len,
            crc: self.crc,
            arena: self.arena.clone(),
            long: self.long.clone(),
            seen: self.seen.clone(),
        }
    }

    /// `(filled slots, capacity, table address)`.
    pub fn occupancy(&self) -> (usize, usize, usize) {
        (self.len, self.table.cap, self.table.ptr.as_ptr() as usize)
    }

    /// Append the stored ids. `false` means the piece is not cached.
    #[inline(always)]
    pub fn lookup(&self, piece: &[u8], out: &mut Vec<u32>) -> bool {
        if self.crc {
            self.probe::<true>(piece, out)
        } else {
            self.probe::<false>(piece, out)
        }
    }

    /// Pull the line for `piece` into L2. The encode loop does this for a
    /// whole batch before it probes.
    #[inline(always)]
    pub fn prefetch<const CRC: bool>(&self, piece: &[u8]) {
        let Some(key) = pack(piece) else {
            return;
        };
        let index = (mix::<CRC>(key) as usize) & self.mask & !1;
        let ptr = unsafe { self.table.ptr.as_ptr().add(index) };
        prefetch_line::<false>(ptr);
    }

    /// Append the stored ids. `CRC` selects the hash, and it must match
    /// [`Self::crc`].
    #[inline(always)]
    pub fn probe<const CRC: bool>(&self, piece: &[u8], out: &mut Vec<u32>) -> bool {
        let Some(key) = pack(piece) else {
            // Past 32 bytes the next miss is almost always a new piece, and
            // storing it allocates a key. English stays at or under 27 bytes.
            if piece.len() > 32 {
                return false;
            }
            let Some(ids) = self.long.get(piece) else {
                return false;
            };
            out.extend_from_slice(ids);
            return true;
        };
        let mut index = (mix::<CRC>(key) as usize) & self.mask & !1;
        let arena = self.arena.as_slice();
        loop {
            let first = self.table.get(index);
            let second = self.table.get(index + 1);
            if first.key == key {
                emit(first, arena, out);
                return true;
            }
            if second.key == key {
                emit(second, arena, out);
                return true;
            }
            if first.key == 0 || second.key == 0 {
                return false;
            }
            index = (index + 2) & self.mask;
        }
    }

/// Remember `ids` for `piece`. A 16 to 32 byte piece that contains a
/// unified ideograph waits for a second sight: those pieces do not
/// repeat, and storing each one rehashes the map during encode. Thai,
/// kana, and ASCII in that length are stored immediately. A loaded or
/// warmed piece uses [`Self::insert`] and does not wait.
    pub fn admit(&mut self, piece: &[u8], ids: &[u32]) {
        if piece.is_empty() || ids.is_empty() {
            return;
        }
        if pack(piece).is_none() {
            if piece.len() > 32 || (han_piece(piece) && !self.mark(piece)) {
                return;
            }
        }
        self.insert(piece, ids);
    }

    /// True when `piece` was marked by an earlier miss.
    fn mark(&mut self, piece: &[u8]) -> bool {
        let mut hash = 0x9E37_79B9_7F4A_7C15u64;
        for &byte in piece {
            hash = hash.wrapping_mul(0x100_0000_01B3).wrapping_add(u64::from(byte));
        }
        let one = (hash as usize) & (SEEN_BITS - 1);
        let two = (hash.wrapping_mul(0xBF58_476D_1CE4_E5B9) as usize) & (SEEN_BITS - 1);
        let again = seen_bit(&self.seen, one) && seen_bit(&self.seen, two);
        set_seen(&mut self.seen, one);
        set_seen(&mut self.seen, two);
        again
    }

    /// Remember `ids` for `piece`. An empty piece is ignored.
    pub fn insert(&mut self, piece: &[u8], ids: &[u32]) {
        if piece.is_empty() || ids.is_empty() {
            return;
        }
        let Some(key) = pack(piece) else {
            if piece.len() > 32 {
                return;
            }
            self.long.insert(piece.to_vec(), ids.to_vec());
            return;
        };
        if (self.len + 1) * 4 > self.table.cap * 3 {
            self.grow();
        }
        let (lo, hi) = pack_ids(ids, &mut self.arena);
        let crc = self.crc;
        if self.place(key, lo, hi, crc) {
            self.len += 1;
        }
    }

    fn grow(&mut self) {
        let cap = self.table.cap * 2;
        let old = std::mem::replace(&mut self.table, Table::new(cap));
        self.mask = cap - 1;
        for index in 0..old.cap {
            let entry = old.get(index);
            if entry.key != 0 {
                self.place(entry.key, entry.lo, entry.hi, self.crc);
            }
        }
    }

    /// `true` when the slot was empty.
    fn place(&mut self, key: u128, lo: u64, hi: u64, crc: bool) -> bool {
        let mixed = if crc { mix::<true>(key) } else { mix::<false>(key) };
        let mut index = (mixed as usize) & self.mask & !1;
        loop {
            for slot in [index, index + 1] {
                let entry = self.table.get(slot);
                if entry.key == key || entry.key == 0 {
                    self.table.set(
                        slot,
                        Entry {
                            key,
                            lo,
                            hi,
                        },
                    );
                    return entry.key == 0;
                }
            }
            index = (index + 2) & self.mask;
        }
    }

    /// Table base and the pair mask. The encode loop keeps this in registers.
    /// A view taken before [`Self::insert`] dangles if the table grows.
    #[inline(always)]
    pub(crate) fn view(&self) -> ProbeView {
        ProbeView {
            base: self.table.ptr.as_ptr(),
            pair_mask: self.mask & !1,
        }
    }
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

fn pack_ids(ids: &[u32], arena: &mut Vec<u32>) -> (u64, u64) {
    let inline = match *ids {
        [a] if a < (1 << 24) => Some((1 | (u64::from(a) << 8), 0)),
        [a, b] if a < (1 << 24) => Some((2 | (u64::from(a) << 8) | (u64::from(b) << 32), 0)),
        [a, b, c] if a < (1 << 24) => {
            Some((3 | (u64::from(a) << 8) | (u64::from(b) << 32), u64::from(c)))
        }
        [a, b, c, d] if a < (1 << 24) => Some((
            4 | (u64::from(a) << 8) | (u64::from(b) << 32),
            u64::from(c) | (u64::from(d) << 32),
        )),
        _ => None,
    };
    if let Some(packed) = inline {
        return packed;
    }
    let off = arena.len() as u64;
    arena.extend_from_slice(ids);
    (SPILL | (ids.len() as u64) | (off << 32), 0)
}

fn emit(entry: Entry, arena: &[u32], out: &mut Vec<u32>) {
    let n = (entry.lo & 0x7f) as usize;
    if entry.lo & SPILL != 0 {
        let off = (entry.lo >> 32) as usize;
        out.extend_from_slice(&arena[off..off + n]);
        return;
    }
    let lanes = [
        (entry.lo >> 8) as u32 & 0xff_ffff,
        (entry.lo >> 32) as u32,
        entry.hi as u32,
        (entry.hi >> 32) as u32,
    ];
    out.extend_from_slice(&lanes[..n]);
}

/// One prepared piece. `key == 0` is a long piece or an empty one; the slow
/// path tells those apart from the bytes. `hash` is the unmasked mix.
#[derive(Clone, Copy)]
pub(crate) struct Prep {
    pub key: u128,
    pub hash: u64,
}

/// Borrowed table pointer. Copy it into the emit loop; do not hold it across
/// an insert that might grow the table.
#[derive(Clone, Copy)]
pub(crate) struct ProbeView {
    base: *const Entry,
    pair_mask: usize,
}

impl ProbeView {
    #[inline(always)]
    fn pair_ptr(self, hash: u64) -> *const Entry {
        unsafe { self.base.add((hash as usize) & self.pair_mask) }
    }

    /// `L1` pulls the line into L1. Otherwise it stops in L2.
    #[inline(always)]
    pub(crate) fn prefetch<const L1: bool>(self, hash: u64) {
        prefetch_line::<L1>(self.pair_ptr(hash));
    }

    /// Home-pair probe. On a miss the returned lanes belong to some other
    /// entry and the caller ignores them. A key displaced past this pair
    /// also misses; the slow path walks the rest. Key 0 compares equal to
    /// an empty slot, so the caller rejects key 0 itself.
    #[inline(always)]
    pub(crate) fn probe_pair(self, key: u128, hash: u64) -> (u64, u64, bool) {
        let p = self.pair_ptr(hash);
        let (e0, e1) = unsafe { (&*p, &*p.add(1)) };
        let m0 = e0.key == key;
        let m1 = e1.key == key;
        #[cfg(target_arch = "x86_64")]
        let (val, ext) = {
            let (mut val, mut ext) = (e1.lo, e1.hi);
            unsafe {
                std::arch::asm!(
                    "test {m}, {m}",
                    "cmovne {val}, {v0}",
                    "cmovne {ext}, {x0}",
                    m = in(reg) m0 as u64,
                    val = inout(reg) val,
                    ext = inout(reg) ext,
                    v0 = in(reg) e0.lo,
                    x0 = in(reg) e0.hi,
                    options(pure, nomem, nostack),
                );
            }
            (val, ext)
        };
        #[cfg(not(target_arch = "x86_64"))]
        let (val, ext) = {
            let sel = (m0 as u64).wrapping_neg();
            ((e0.lo & sel) | (e1.lo & !sel), (e0.hi & sel) | (e1.hi & !sel))
        };
        (val, ext, m0 | m1)
    }
}

/// Length in the high byte, then the bytes. `None` when `piece` is empty or
/// longer than 15 bytes. A real key is never zero.
#[inline(always)]
pub fn pack(piece: &[u8]) -> Option<u128> {
    pretokie::pack_piece(piece)
}

#[inline(always)]
pub(crate) fn mix<const CRC: bool>(key: u128) -> u64 {
    pretokie::mix_piece::<CRC>(key)
}

fn table_layout(cap: usize) -> Layout {
    let size = cap * std::mem::size_of::<Entry>();
    // 2 MiB once the table is that big. Smaller tables stay power-of-two
    // aligned so a pair never straddles a line (floor 64).
    let align = (2 * 1024 * 1024).min(size.next_power_of_two()).max(64);
    Layout::from_size_align(size, align).expect("cache layout")
}

/// Mask halves for a packed length of 1..=15. Index 0 is unused.
const PACK_MASKS: [[u64; 2]; 16] = {
    let mut table = [[0u64; 2]; 16];
    let mut n = 1usize;
    while n <= 15 {
        let s = (n * 8) as u32;
        let lo = if n < 8 {
            u64::MAX >> (64u32.wrapping_sub(s) & 63)
        } else {
            u64::MAX
        };
        let hi = if n > 8 {
            u64::MAX >> (128u32.wrapping_sub(s) & 63)
        } else {
            0
        };
        table[n] = [lo, hi];
        n += 1;
    }
    table
};

/// Pack `spans` into `prep`, then pull each line into L2.
///
/// Spans are in order. When the last span starts at least 16 bytes before
/// the end of `bytes`, every key is one unaligned load and a mask, with no
/// per-span branch. A span longer than 15 bytes stores key 0. `prep` must
/// hold 16 slots past the batch so the emit loop can prefetch ahead.
#[inline(never)]
pub(crate) fn pack_spans<const CRC: bool>(
    bytes: &[u8],
    spans: &[pretokie::Span],
    prep: &mut [Prep],
    table: ProbeView,
) {
    let n = spans.len();
    debug_assert!(prep.len() >= n + 16);
    for slot in &mut prep[n..n + 16] {
        *slot = Prep { key: 0, hash: 0 };
    }
    if n == 0 {
        return;
    }
    let last = spans[n - 1].start as usize;
    if last + 16 <= bytes.len() {
        let masks: *const [u64; 2] = std::hint::black_box(PACK_MASKS.as_ptr());
        let base = bytes.as_ptr();
        for (i, span) in spans.iter().enumerate() {
            let start = span.start as usize;
            let tok_len = (span.end - span.start) as usize;
            let raw = unsafe { (base.add(start) as *const u128).read_unaligned() };
            let m = tok_len.min(15);
            let [mask_lo, mask_hi] = unsafe { *masks.add(m) };
            let keep = ((tok_len <= 15) as u64).wrapping_neg();
            let lo = (raw as u64) & mask_lo & keep;
            let hi = (((raw >> 64) as u64 & mask_hi) | ((m as u64) << 56)) & keep;
            let key = (lo as u128) | ((hi as u128) << 64);
            let hash = mix::<CRC>(key);
            prep[i] = Prep { key, hash };
            table.prefetch::<false>(hash);
        }
        return;
    }
    for (i, span) in spans.iter().enumerate() {
        let piece = &bytes[span.start as usize..span.end as usize];
        let key = pack(piece).unwrap_or(0);
        let hash = if key == 0 { 0 } else { mix::<CRC>(key) };
        prep[i] = Prep { key, hash };
        if key != 0 {
            table.prefetch::<false>(hash);
        }
    }
}

/// Pack `n` pieces whose boundaries are `bounds[0..=n]`.
///
/// Same key and prefetch as [`pack_spans`]. `bounds[i]` is the absolute
/// start of piece `i`.
#[inline(never)]
pub(crate) fn pack_bounds<const CRC: bool>(
    bytes: &[u8],
    bounds: &[u32],
    n: usize,
    prep: &mut [Prep],
    table: ProbeView,
) {
    debug_assert!(prep.len() >= n + 16);
    debug_assert!(bounds.len() > n);
    for slot in &mut prep[n..n + 16] {
        *slot = Prep { key: 0, hash: 0 };
    }
    if n == 0 {
        return;
    }
    let last = bounds[n - 1] as usize;
    if last + 16 <= bytes.len() {
        let masks: *const [u64; 2] = std::hint::black_box(PACK_MASKS.as_ptr());
        let base = bytes.as_ptr();
        let bounds = bounds.as_ptr();
        let out = prep.as_mut_ptr();
        for i in 0..n {
            let start = unsafe { *bounds.add(i) } as usize;
            let end = unsafe { *bounds.add(i + 1) } as usize;
            let tok_len = end - start;
            let raw = unsafe { (base.add(start) as *const u128).read_unaligned() };
            let m = tok_len.min(15);
            let [mask_lo, mask_hi] = unsafe { *masks.add(m) };
            let keep = ((tok_len <= 15) as u64).wrapping_neg();
            let lo = (raw as u64) & mask_lo & keep;
            let hi = (((raw >> 64) as u64 & mask_hi) | ((m as u64) << 56)) & keep;
            let key = (lo as u128) | ((hi as u128) << 64);
            let hash = mix::<CRC>(key);
            unsafe { out.add(i).write(Prep { key, hash }) };
            table.prefetch::<false>(hash);
        }
        return;
    }
    let out = prep.as_mut_ptr();
    for i in 0..n {
        let start = bounds[i] as usize;
        let end = bounds[i + 1] as usize;
        let piece = &bytes[start..end];
        let key = pack(piece).unwrap_or(0);
        let hash = if key == 0 { 0 } else { mix::<CRC>(key) };
        unsafe { out.add(i).write(Prep { key, hash }) };
        if key != 0 {
            table.prefetch::<false>(hash);
        }
    }
}

fn crc_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("sse4.2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn prefetch_line<const L1: bool>(ptr: *const Entry) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if L1 {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
        } else {
            std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T1);
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

fn hugepage(ptr: *mut u8, bytes: usize) {
    // 14 is MADV_HUGEPAGE.
    advise(ptr, bytes, 14);
}

fn advise(ptr: *mut u8, bytes: usize, advice: i32) {
    #[cfg(target_os = "linux")]
    unsafe {
        const PAGE: usize = 4096;
        unsafe extern "C" {
            fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
        }
        let start = (ptr as usize + PAGE - 1) & !(PAGE - 1);
        let end = (ptr as usize).saturating_add(bytes);
        if end > start {
            madvise(start as *mut u8, end - start, advice);
        }
    }
    #[cfg(not(target_os = "linux"))]
    let _ = (ptr, bytes, advice);
}

fn han_piece(piece: &[u8]) -> bool {
    piece.iter().any(|&byte| (0xE4..=0xE9).contains(&byte))
}

fn seen_bit(words: &[u64], bit: usize) -> bool {
    (words[bit / 64] >> (bit % 64)) & 1 == 1
}

fn set_seen(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

#[cfg(test)]
mod tests {
    use super::{Cache, Prep, Table, mix, pack, pack_spans};

    fn is_mapping(line: &str) -> bool {
        let Some((range, _)) = line.split_once(' ') else {
            return false;
        };
        let Some((start, end)) = range.split_once('-') else {
            return false;
        };
        !start.is_empty()
            && !end.is_empty()
            && start.chars().all(|c| c.is_ascii_hexdigit())
            && end.chars().all(|c| c.is_ascii_hexdigit())
    }
    use pretokie::Span;

    #[test]
    fn a_full_file_table_is_backed_by_huge_pages() {
        let table = Table::new(1 << 20);
        let addr = table.ptr.as_ptr() as usize;
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut block = String::new();
        let mut hit = false;
        for line in smaps.lines() {
            if is_mapping(line) {
                if hit {
                    break;
                }
                if let Some((range, _)) = line.split_once(' ') {
                    if let Some((start, end)) = range.split_once('-') {
                        let start = usize::from_str_radix(start, 16).unwrap_or(0);
                        let end = usize::from_str_radix(end, 16).unwrap_or(0);
                        hit = addr >= start && addr < end;
                    }
                }
            }
            if hit {
                block.push_str(line);
                block.push('\n');
            }
        }
        assert!(hit, "table {addr:#x} missing from smaps");
        if !block.contains("THPeligible:") || block.contains("THPeligible:           0") {
            return;
        }
        let huge = block
            .lines()
            .find(|line| line.starts_with("AnonHugePages:"))
            .unwrap_or("AnonHugePages missing");
        assert!(
            huge.contains("32768 kB"),
            "table {addr:#x} anon huge pages: {huge}\n{block}"
        );
    }

    #[test]
    fn branchless_pack_matches_the_byte_pack() {
        let mut bytes = Vec::new();
        let mut spans = Vec::new();
        let mut at = 0u32;
        for n in 1..=20 {
            let start = bytes.len() as u32;
            bytes.extend(std::iter::repeat_n(b'a' + (n as u8 % 26), n));
            spans.push(Span {
                start,
                end: bytes.len() as u32,
            });
            at = bytes.len() as u32;
        }
        // Pad so every span in the first batch starts 16 bytes before the end.
        bytes.extend(std::iter::repeat_n(b'z', 16));
        let _ = at;
        let cache = Cache::new();
        let mut prep = [Prep { key: 0, hash: 0 }; 40];
        pack_spans::<true>(&bytes, &spans, &mut prep, cache.view());
        for (i, span) in spans.iter().enumerate() {
            let piece = &bytes[span.start as usize..span.end as usize];
            let key = pack(piece).unwrap_or(0);
            assert_eq!(prep[i].key, key, "len {}", piece.len());
            let hash = if key == 0 { 0 } else { mix::<true>(key) };
            assert_eq!(prep[i].hash, hash);
        }
        // The tail batch starts inside the last 16 bytes and takes the careful path.
        let tail = [Span {
            start: (bytes.len() - 4) as u32,
            end: bytes.len() as u32,
        }];
        pack_spans::<true>(&bytes, &tail, &mut prep, cache.view());
        let piece = &bytes[tail[0].start as usize..tail[0].end as usize];
        assert_eq!(prep[0].key, pack(piece).unwrap());
    }

    /// A lookup returns the ids last stored for that piece.
    ///
    /// Covers every one-byte piece, every two-byte piece whose bytes are
    /// below 16, every three-byte piece whose bytes are below 8, every
    /// inline id below 2^24, the first spilled id, and pieces of 16 and 32
    /// bytes. A longer piece is not stored. Insertion order is forward,
    /// reversed, and a fixed scramble. The short pieces share one table, so
    /// growth and open addressing both run.
    #[test]
    fn cache_roundtrip_returns_inserted_ids() {
        let mut samples: Vec<(Vec<u8>, Vec<u32>)> = Vec::new();
        for b in 0u8..=255 {
            samples.push((vec![b], vec![u32::from(b).wrapping_add(1)]));
        }
        for a in 0u8..16 {
            for b in 0u8..16 {
                let id = (u32::from(a) << 8) | u32::from(b) | (1 << 16);
                samples.push((vec![a, b], vec![id, id ^ 0x00ff_fffe]));
            }
        }
        for a in 0u8..8 {
            for b in 0u8..8 {
                for c in 0u8..8 {
                    let n = samples.len() as u32;
                    samples.push((vec![a, b, c], vec![n]));
                }
            }
        }
        samples.push((vec![0x20; 16], vec![1 << 24]));
        samples.push((vec![0x20; 32], vec![u32::MAX, 0, 7, 8, 9]));
        samples.push((b"bound".to_vec(), vec![(1 << 24) - 1]));
        samples.push((b"spill".to_vec(), vec![1 << 24, 256]));
        samples.push((b"four".to_vec(), vec![1, 2, 3, 4]));
        samples.push((
            b"lanes".to_vec(),
            vec![(1 << 24) - 1, u32::MAX, 0, 1 << 31],
        ));

        for order in 0..3 {
            let mut cache = Cache::new();
            let n = samples.len();
            for step in 0..n {
                let i = match order {
                    0 => step,
                    1 => n - 1 - step,
                    _ => (step.wrapping_mul(1664525).wrapping_add(1013904223)) % n,
                };
                let (piece, ids) = &samples[i];
                cache.insert(piece, ids);
            }
            // The scramble visits some pieces twice and others never. Fill
            // the gaps so every sample is present before the checks.
            for (piece, ids) in &samples {
                cache.insert(piece, ids);
            }
            check_samples(&cache, &samples);
            let piece = vec![0x20; 33];
            cache.insert(&piece, &[9]);
            let mut out = Vec::new();
            assert!(!cache.lookup(&piece, &mut out), "a piece past 32 bytes is not stored");
            assert!(out.is_empty());
        }

        let mut cache = Cache::new();
        let piece = b"id";
        for id in 0..(1 << 24) {
            cache.insert(piece, &[id]);
            let mut out = Vec::new();
            assert!(cache.lookup(piece, &mut out));
            assert_eq!(out, vec![id]);
        }

        let mut cache = Cache::new();
        let mut kept = vec![7];
        cache.insert(b"", &[1]);
        cache.insert(b"z", &[]);
        assert!(!cache.lookup(b"", &mut kept));
        assert!(!cache.lookup(b"z", &mut kept));
        assert!(!cache.lookup(b"missing", &mut kept));
        assert_eq!(kept, vec![7]);
        cache.insert(b"z", &[1]);
        cache.insert(b"z", &[2, 3]);
        let mut out = Vec::new();
        assert!(cache.lookup(b"z", &mut out));
        assert_eq!(out, vec![2, 3]);

        let mut cache = Cache::new();
        let medium = vec![0xE4, 0xBD, 0xA0].repeat(8);
        cache.admit(&medium, &[4, 5]);
        let mut out = Vec::new();
        assert!(!cache.lookup(&medium, &mut out));
        cache.admit(&medium, &[4, 5]);
        assert!(cache.lookup(&medium, &mut out));
        assert_eq!(out, vec![4, 5]);
        let thai = vec![0xE0, 0xB8, 0x81].repeat(8);
        cache.admit(&thai, &[6]);
        out.clear();
        assert!(cache.lookup(&thai, &mut out));
        assert_eq!(out, vec![6]);
        cache.admit(b"hi", &[8]);
        out.clear();
        assert!(cache.lookup(b"hi", &mut out));
        assert_eq!(out, vec![8]);
    }

    fn check_samples(cache: &Cache, samples: &[(Vec<u8>, Vec<u32>)]) {
        for (piece, ids) in samples {
            let mut out = Vec::new();
            assert!(
                cache.lookup(piece, &mut out),
                "miss for {piece:?}"
            );
            assert_eq!(out, *ids, "lookup ids for {piece:?}");
            if let Some(fast) = fast_inline(cache, piece) {
                assert_eq!(fast, *ids, "home-pair ids for {piece:?}");
            }
        }
    }

    /// Ids from the home-pair probe when it claims an inline hit.
    fn fast_inline(cache: &Cache, piece: &[u8]) -> Option<Vec<u32>> {
        let key = pack(piece)?;
        let hash = if cache.crc() {
            mix::<true>(key)
        } else {
            mix::<false>(key)
        };
        let (lo, hi, found) = cache.view().probe_pair(key, hash);
        if !found || lo & 0x80 != 0 {
            return None;
        }
        let mut out = Vec::new();
        super::emit(super::Entry { key, lo, hi }, &[], &mut out);
        Some(out)
    }
}
