//! Dependency-free SentencePiece-BPE encoder research harness.
//!
//! Variant A is a verbatim port of tokie's `crates/tokie/src/encoder/sentencepiece.rs`
//! (radix heap over the whole input). Candidates are measured against it for both
//! throughput and exact output identity.
//!
//! Build: rustc -O -C target-cpu=native -o sp_harness sp_harness.rs

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

// ============================================================================
// Fast hasher (stand-in for foldhash; identical for all variants so the
// comparison stays algorithmic rather than hash-implementation-dependent).
// ============================================================================

#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.hash = (self.hash.rotate_left(5) ^ b as u64).wrapping_mul(SEED);
        }
    }
    #[inline]
    fn write_u64(&mut self, n: u64) {
        self.hash = (self.hash.rotate_left(5) ^ n).wrapping_mul(SEED);
    }
    #[inline]
    fn write_u128(&mut self, n: u128) {
        self.write_u64(n as u64);
        self.write_u64((n >> 64) as u64);
    }
    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxMap<K, V> = HashMap<K, V, FxBuild>;

pub type TokenId = u32;

const NONE: u32 = u32::MAX;
const METASPACE: [u8; 3] = [0xE2, 0x96, 0x81];

#[inline(always)]
fn pack_pair(left: TokenId, right: TokenId) -> u64 {
    ((left as u64) << 32) | (right as u64)
}

#[inline]
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

// ============================================================================
// Baseline structures — verbatim from tokie sentencepiece.rs
// ============================================================================

#[derive(Clone, Copy)]
struct Symbol {
    token: TokenId,
    prev: u32,
    next: u32,
    len: u16,
}

#[derive(Clone, Copy)]
struct HeapEntry {
    key: u64,
    right: u32,
    size: u32,
}

impl HeapEntry {
    #[inline(always)]
    fn new(rank: u32, left: u32, right: u32, size: u32) -> Self {
        Self { key: ((rank as u64) << 32) | (left as u64), right, size }
    }
    #[inline(always)]
    fn left(&self) -> u32 {
        self.key as u32
    }
}

struct RadixHeap {
    buckets: [Vec<HeapEntry>; 65],
    last_min: u64,
    len: usize,
    overflow: Vec<HeapEntry>,
    /// Instrumentation only (not in tokie): counts redistribution passes.
    pub redistributions: u64,
    pub redistributed_entries: u64,
}

impl RadixHeap {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            last_min: 0,
            len: 0,
            overflow: Vec::new(),
            redistributions: 0,
            redistributed_entries: 0,
        }
    }

    #[inline]
    fn bucket_index(&self, key: u64) -> usize {
        if key == self.last_min {
            0
        } else {
            let diff = key ^ self.last_min;
            (64 - diff.leading_zeros()) as usize
        }
    }

    #[inline]
    fn push(&mut self, entry: HeapEntry) {
        if entry.key < self.last_min {
            self.overflow.push(entry);
        } else {
            let idx = self.bucket_index(entry.key);
            self.buckets[idx].push(entry);
        }
        self.len += 1;
    }

    fn pop(&mut self) -> Option<HeapEntry> {
        if self.len == 0 {
            return None;
        }
        if !self.overflow.is_empty() {
            let mut ov_min_idx = 0;
            let mut ov_min_key = self.overflow[0].key;
            for (i, entry) in self.overflow.iter().enumerate().skip(1) {
                if entry.key < ov_min_key {
                    ov_min_key = entry.key;
                    ov_min_idx = i;
                }
            }
            let mut normal_bucket_idx = 0;
            while normal_bucket_idx < 65 && self.buckets[normal_bucket_idx].is_empty() {
                normal_bucket_idx += 1;
            }
            let normal_min_key = if normal_bucket_idx < 65 {
                if normal_bucket_idx == 0 {
                    Some(self.last_min)
                } else {
                    self.buckets[normal_bucket_idx].iter().map(|e| e.key).min()
                }
            } else {
                None
            };
            if normal_min_key.is_none() || ov_min_key <= normal_min_key.unwrap() {
                let entry = self.overflow.swap_remove(ov_min_idx);
                self.len -= 1;
                return Some(entry);
            }
        }
        let mut bucket_idx = 0;
        while bucket_idx < 65 && self.buckets[bucket_idx].is_empty() {
            bucket_idx += 1;
        }
        if bucket_idx >= 65 {
            return None;
        }
        if bucket_idx == 0 {
            self.len -= 1;
            return self.buckets[0].pop();
        }
        let bucket = &mut self.buckets[bucket_idx];
        let mut min_idx = 0;
        let mut min_key = bucket[0].key;
        for (i, entry) in bucket.iter().enumerate().skip(1) {
            if entry.key < min_key {
                min_key = entry.key;
                min_idx = i;
            }
        }
        self.last_min = min_key;
        let min_entry = bucket.swap_remove(min_idx);
        let entries: Vec<HeapEntry> = std::mem::take(bucket);
        self.redistributions += 1;
        self.redistributed_entries += entries.len() as u64;
        for entry in entries {
            let new_idx = self.bucket_index(entry.key);
            self.buckets[new_idx].push(entry);
        }
        self.len -= 1;
        Some(min_entry)
    }

    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.last_min = 0;
        self.len = 0;
        self.overflow.clear();
    }
}

pub struct EncodeState {
    symbols: Vec<Symbol>,
    heap: RadixHeap,
    result: Vec<TokenId>,
}

impl EncodeState {
    pub fn new() -> Self {
        Self { symbols: Vec::new(), heap: RadixHeap::new(), result: Vec::new() }
    }
    fn clear(&mut self) {
        self.symbols.clear();
        self.heap.clear();
        self.result.clear();
    }
}

// ============================================================================
// Encoder
// ============================================================================

pub struct SentencePieceBPE {
    pair_lookup: FxMap<u64, (TokenId, u32)>,
    token_cache: FxMap<Vec<u8>, TokenId>,
    byte_lut: [TokenId; 256],
    token_lengths: Vec<u16>,
    vocab_size: usize,
    /// True when no vocab token contains an interior `▁` — the guard that
    /// makes per-unit splitting lossless (mirrors unigram.rs::compute_unit_split_safe).
    pub unit_split_safe: bool,
}

impl SentencePieceBPE {
    pub fn from_parts(
        vocab: &[(u32, Vec<u8>)],
        merges: &[(TokenId, TokenId)],
    ) -> Self {
        let mut token_bytes: Vec<Vec<u8>> = vec![Vec::new(); vocab.len()];
        for (id, bytes) in vocab {
            if (*id as usize) < token_bytes.len() {
                token_bytes[*id as usize] = bytes.clone();
            }
        }
        let bytes_to_id: FxMap<Vec<u8>, TokenId> =
            vocab.iter().map(|(id, b)| (b.clone(), *id)).collect();

        let mut pair_lookup: FxMap<u64, (TokenId, u32)> = FxMap::default();
        for (rank, &(l, r)) in merges.iter().enumerate() {
            let mut merged = token_bytes[l as usize].clone();
            merged.extend_from_slice(&token_bytes[r as usize]);
            if let Some(&mid) = bytes_to_id.get(&merged) {
                pair_lookup.entry(pack_pair(l, r)).or_insert((mid, rank as u32));
            }
        }

        let mut byte_lut = [u32::MAX; 256];
        for (id, bytes) in vocab {
            if bytes.len() == 1 && byte_lut[bytes[0] as usize] == u32::MAX {
                byte_lut[bytes[0] as usize] = *id;
            }
        }

        let token_lengths: Vec<u16> = token_bytes.iter().map(|b| b.len() as u16).collect();

        let mut token_cache: FxMap<Vec<u8>, TokenId> = vocab
            .iter()
            .filter(|(_, b)| b.len() > 1)
            .map(|(id, b)| (b.clone(), *id))
            .collect();
        for (bv, &tid) in byte_lut.iter().enumerate() {
            if tid != u32::MAX {
                token_cache.insert(vec![bv as u8], tid);
            }
        }

        let unit_split_safe = !token_bytes.iter().any(|b| {
            b.windows(3).enumerate().any(|(i, w)| i > 0 && w == METASPACE)
        });

        Self {
            pair_lookup,
            token_cache,
            byte_lut,
            token_lengths,
            vocab_size: vocab.len(),
            unit_split_safe,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    pub fn num_merges(&self) -> usize {
        self.pair_lookup.len()
    }

    #[inline]
    fn get_merge(&self, left: TokenId, right: TokenId) -> Option<(TokenId, u32)> {
        self.pair_lookup.get(&pack_pair(left, right)).copied()
    }

    fn init_symbols_into(&self, text: &[u8], symbols: &mut Vec<Symbol>) {
        let mut pos = 0;
        while pos < text.len() {
            let char_len = utf8_char_len(text[pos]);
            let end = (pos + char_len).min(text.len());
            let char_bytes = &text[pos..end];
            let (token, len) = if let Some(&tid) = self.token_cache.get(char_bytes) {
                (tid, char_bytes.len())
            } else {
                (self.byte_lut[text[pos] as usize], 1)
            };
            if token != u32::MAX {
                let idx = symbols.len() as u32;
                symbols.push(Symbol {
                    token,
                    prev: if idx == 0 { NONE } else { idx - 1 },
                    next: NONE,
                    len: self.token_lengths.get(token as usize).copied().unwrap_or(len as u16),
                });
                if idx > 0 {
                    symbols[(idx - 1) as usize].next = idx;
                }
            }
            pos += len;
        }
    }

    fn init_heap(&self, symbols: &[Symbol], heap: &mut RadixHeap) {
        for i in 0..symbols.len().saturating_sub(1) {
            let l = &symbols[i];
            let r = &symbols[i + 1];
            if let Some((_, rank)) = self.get_merge(l.token, r.token) {
                heap.push(HeapEntry::new(rank, i as u32, (i + 1) as u32, l.len as u32 + r.len as u32));
            }
        }
    }

    fn merge_loop(&self, symbols: &mut [Symbol], heap: &mut RadixHeap) {
        while let Some(entry) = heap.pop() {
            let li = entry.left() as usize;
            let ri = entry.right as usize;
            let left = &symbols[li];
            let right = &symbols[ri];
            if left.len == 0 || right.len == 0 {
                continue;
            }
            if left.next != entry.right {
                continue;
            }
            if (left.len as u32 + right.len as u32) != entry.size {
                continue;
            }
            let (merged_token, _) = self.get_merge(left.token, right.token).unwrap();
            let new_len = left.len + right.len;
            let right_next = right.next;
            symbols[li].token = merged_token;
            symbols[li].len = new_len;
            symbols[li].next = right_next;
            symbols[ri].len = 0;
            if right_next != NONE {
                symbols[right_next as usize].prev = entry.left();
            }
            let left_prev = symbols[li].prev;
            if left_prev != NONE {
                let prev = &symbols[left_prev as usize];
                if prev.len > 0 {
                    if let Some((_, rank)) = self.get_merge(prev.token, merged_token) {
                        heap.push(HeapEntry::new(rank, left_prev, entry.left(), prev.len as u32 + new_len as u32));
                    }
                }
            }
            if right_next != NONE {
                let next = &symbols[right_next as usize];
                if next.len > 0 {
                    if let Some((_, rank)) = self.get_merge(merged_token, next.token) {
                        heap.push(HeapEntry::new(rank, entry.left(), right_next, new_len as u32 + next.len as u32));
                    }
                }
            }
        }
    }

    fn collect_results_into(&self, symbols: &[Symbol], result: &mut Vec<TokenId>) {
        let mut idx = 0u32;
        while idx != NONE && (idx as usize) < symbols.len() {
            let sym = &symbols[idx as usize];
            if sym.len > 0 {
                result.push(sym.token);
            }
            idx = sym.next;
        }
    }

    /// VARIANT A — tokie today: whole-input radix heap, no memoization.
    pub fn encode_a(&self, text: &[u8], state: &mut EncodeState, out: &mut Vec<TokenId>) {
        state.clear();
        if text.is_empty() {
            return;
        }
        if let Some(&tid) = self.token_cache.get(text) {
            out.push(tid);
            return;
        }
        self.init_symbols_into(text, &mut state.symbols);
        if state.symbols.is_empty() {
            return;
        }
        self.init_heap(&state.symbols, &mut state.heap);
        let mut syms = std::mem::take(&mut state.symbols);
        self.merge_loop(&mut syms, &mut state.heap);
        self.collect_results_into(&syms, out);
        state.symbols = syms;
    }

    /// VARIANT B — split at `▁` unit boundaries, run the same merge loop per unit.
    pub fn encode_b(&self, text: &[u8], state: &mut EncodeState, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe {
            return self.encode_a(text, state, out);
        }
        for unit in MetaspaceUnits::new(text) {
            self.encode_a(unit, state, out);
        }
    }

    /// VARIANT C — per-unit + memoization of Zipf-hot units.
    pub fn encode_c(
        &self,
        text: &[u8],
        state: &mut EncodeState,
        cache: &mut UnitCache,
        out: &mut Vec<TokenId>,
    ) {
        if !self.unit_split_safe {
            return self.encode_a(text, state, out);
        }
        for unit in MetaspaceUnits::new(text) {
            if cache.lookup(unit, out) {
                continue;
            }
            let mark = out.len();
            self.encode_a(unit, state, out);
            cache.insert(unit, &out[mark..]);
        }
    }
}

// ============================================================================
// Metaspace unit iterator: split so each unit starts at a `▁` run.
// ============================================================================

pub struct MetaspaceUnits<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> MetaspaceUnits<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }
}

#[inline]
fn find_metaspace(hay: &[u8], from: usize) -> Option<usize> {
    let n = hay.len();
    let mut i = from;
    while i + 3 <= n {
        if hay[i] == 0xE2 && hay[i + 1] == 0x96 && hay[i + 2] == 0x81 {
            return Some(i);
        }
        i += 1;
    }
    None
}

impl<'a> Iterator for MetaspaceUnits<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let start = self.pos;
        // A unit begins with its `▁` run; find the next `▁` that starts a new unit.
        let mut scan = start;
        // Skip the leading metaspace run belonging to this unit.
        while scan + 3 <= self.bytes.len()
            && self.bytes[scan] == 0xE2
            && self.bytes[scan + 1] == 0x96
            && self.bytes[scan + 2] == 0x81
        {
            scan += 3;
        }
        let end = match find_metaspace(self.bytes, scan) {
            Some(p) => p,
            None => self.bytes.len(),
        };
        self.pos = end;
        Some(&self.bytes[start..end])
    }
}

// ============================================================================
// Unit cache — arena + direct-mapped front table + short/long maps.
// Mirrors tokie's UnigramPieceCache design.
// ============================================================================

const FRONT_BITS: u32 = 18;
const SHORT_KEY_MAX: usize = 15;

pub struct UnitCache {
    arena: Vec<TokenId>,
    front_keys: Box<[u128]>,
    front_vals: Box<[(u32, u32)]>,
    short: FxMap<u128, (u32, u32)>,
    long: FxMap<Box<[u8]>, (u32, u32)>,
    pub hits: u64,
    pub misses: u64,
}

impl UnitCache {
    pub fn new() -> Self {
        let n = 1usize << FRONT_BITS;
        Self {
            arena: Vec::new(),
            front_keys: vec![0u128; n].into_boxed_slice(),
            front_vals: vec![(0u32, 0u32); n].into_boxed_slice(),
            short: FxMap::default(),
            long: FxMap::default(),
            hits: 0,
            misses: 0,
        }
    }

    #[inline]
    fn front_index(key: u128) -> usize {
        let folded = (key as u64) ^ ((key >> 64) as u64);
        let h = folded.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> (64 - FRONT_BITS)) as usize
    }

    #[inline]
    fn pack_key(bytes: &[u8]) -> Option<u128> {
        let n = bytes.len();
        if n == 0 || n > SHORT_KEY_MAX {
            return None;
        }
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(bytes);
        Some(u128::from_le_bytes(lanes) | ((n as u128) << 120))
    }

    #[inline]
    fn lookup(&mut self, unit: &[u8], out: &mut Vec<TokenId>) -> bool {
        let found = if let Some(key) = Self::pack_key(unit) {
            let idx = Self::front_index(key);
            if self.front_keys[idx] == key {
                let (o, l) = self.front_vals[idx];
                out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]);
                true
            } else if let Some(&(o, l)) = self.short.get(&key) {
                self.front_keys[idx] = key;
                self.front_vals[idx] = (o, l);
                out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]);
                true
            } else {
                false
            }
        } else if let Some(&(o, l)) = self.long.get(unit) {
            out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]);
            true
        } else {
            false
        };
        if found {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        found
    }

    #[inline]
    fn insert(&mut self, unit: &[u8], toks: &[TokenId]) {
        if unit.is_empty() || toks.is_empty() {
            return;
        }
        let offset = self.arena.len() as u32;
        let len = toks.len() as u32;
        self.arena.extend_from_slice(toks);
        if let Some(key) = Self::pack_key(unit) {
            self.short.insert(key, (offset, len));
            let idx = Self::front_index(key);
            self.front_keys[idx] = key;
            self.front_vals[idx] = (offset, len);
        } else {
            self.long.insert(unit.to_vec().into_boxed_slice(), (offset, len));
        }
    }
}

// ============================================================================
// Loading
// ============================================================================

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap_or(0))
        .collect()
}

fn load_vocab(path: &str) -> Vec<(u32, Vec<u8>)> {
    let text = std::fs::read_to_string(path).expect("vocab");
    text.lines()
        .filter_map(|line| {
            let mut it = line.split('\t');
            let id: u32 = it.next()?.parse().ok()?;
            Some((id, hex_decode(it.next()?)))
        })
        .collect()
}

fn load_merges(path: &str) -> Vec<(TokenId, TokenId)> {
    let text = std::fs::read_to_string(path).expect("merges");
    text.lines()
        .filter_map(|line| {
            let mut it = line.split('\t');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        })
        .collect()
}

/// Apply the SentencePiece metaspace normalization the encoder sees:
/// every space becomes `▁`, and a `▁` is prepended to the text.
fn metaspace_normalize(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len() + text.len() / 4 + 3);
    out.extend_from_slice(&METASPACE);
    for ch in text.chars() {
        if ch == ' ' {
            out.extend_from_slice(&METASPACE);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

// ============================================================================
// Bench driver
// ============================================================================

fn bench<F: FnMut() -> usize>(label: &str, bytes: usize, reps: usize, mut f: F) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut ntok = 0;
    for _ in 0..reps {
        let t = Instant::now();
        ntok = f();
        let el = t.elapsed().as_secs_f64();
        if el < best {
            best = el;
        }
    }
    let mbps = (bytes as f64 / (1024.0 * 1024.0)) / best;
    println!(
        "  {:<34} {:>9.2} MB/s   {:>8.2} ms   {:>10} tokens",
        label,
        mbps,
        best * 1000.0,
        ntok
    );
    (mbps, ntok)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let vocab_path = args.get(1).cloned().unwrap_or("spbpe_hf32k.vocab.tsv".into());
    let merges_path = args.get(2).cloned().unwrap_or("spbpe_hf32k.merges.tsv".into());
    let corpus_path = args.get(3).cloned().unwrap_or("corpus/bench.txt".into());
    let reps: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let vocab = load_vocab(&vocab_path);
    let merges = load_merges(&merges_path);
    let enc = SentencePieceBPE::from_parts(&vocab, &merges);

    let raw = std::fs::read_to_string(&corpus_path).expect("corpus");
    let text = metaspace_normalize(&raw);
    let nbytes = text.len();

    let n_units = MetaspaceUnits::new(&text).count();
    println!("== SentencePiece-BPE harness ==");
    println!(
        "vocab={} merges={} unit_split_safe={} corpus={:.2} MB units={} ({:.2} B/unit)",
        enc.vocab_size(),
        enc.num_merges(),
        enc.unit_split_safe,
        nbytes as f64 / (1024.0 * 1024.0),
        n_units,
        nbytes as f64 / n_units as f64
    );
    println!();

    let mut state = EncodeState::new();

    // --- correctness: all variants must agree ---
    let mut a = Vec::new();
    enc.encode_a(&text, &mut state, &mut a);
    let mut b = Vec::new();
    enc.encode_b(&text, &mut state, &mut b);
    let mut cache = UnitCache::new();
    let mut c = Vec::new();
    enc.encode_c(&text, &mut state, &mut cache, &mut c);

    println!("identity check:");
    println!("   A tokens = {}", a.len());
    println!("   B == A   : {}", if b == a { "YES" } else { "NO  <-- DIVERGENCE" });
    println!("   C == A   : {}", if c == a { "YES" } else { "NO  <-- DIVERGENCE" });
    if b != a {
        let d = a.iter().zip(b.iter()).position(|(x, y)| x != y);
        println!("   first B diff at {:?} (A len {}, B len {})", d, a.len(), b.len());
    }
    println!(
        "   cache: hits={} misses={} hit_rate={:.2}%",
        cache.hits,
        cache.misses,
        100.0 * cache.hits as f64 / (cache.hits + cache.misses).max(1) as f64
    );
    println!();

    // --- heap instrumentation on the baseline ---
    let mut s2 = EncodeState::new();
    let mut tmp = Vec::new();
    enc.encode_a(&text, &mut s2, &mut tmp);
    println!(
        "baseline heap: redistributions={} entries_moved={} ({:.1} per redistribution)",
        s2.heap.redistributions,
        s2.heap.redistributed_entries,
        s2.heap.redistributed_entries as f64 / s2.heap.redistributions.max(1) as f64
    );
    println!();

    println!("throughput (best of {}):", reps);
    bench("A  whole-input radix heap", nbytes, reps, || {
        let mut o = Vec::with_capacity(nbytes / 3);
        enc.encode_a(&text, &mut state, &mut o);
        o.len()
    });
    bench("B  per-unit, no cache", nbytes, reps, || {
        let mut o = Vec::with_capacity(nbytes / 3);
        enc.encode_b(&text, &mut state, &mut o);
        o.len()
    });
    bench("C  per-unit + memo (cold cache)", nbytes, reps, || {
        let mut ca = UnitCache::new();
        let mut o = Vec::with_capacity(nbytes / 3);
        enc.encode_c(&text, &mut state, &mut ca, &mut o);
        o.len()
    });
    let mut warm = UnitCache::new();
    {
        let mut o = Vec::new();
        enc.encode_c(&text, &mut state, &mut warm, &mut o);
    }
    bench("C  per-unit + memo (warm cache)", nbytes, reps, || {
        let mut o = Vec::with_capacity(nbytes / 3);
        enc.encode_c(&text, &mut state, &mut warm, &mut o);
        o.len()
    });
}
