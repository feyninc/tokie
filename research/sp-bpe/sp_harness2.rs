//! SentencePiece-BPE optimization ladder.
//!
//! A = verbatim port of tokie's current encoder/sentencepiece.rs.
//! Each subsequent variant adds one isolated change so the win is attributable.
//! Every variant is checked for exact token-identity against A.
//!
//! Build: rustc -O -C target-cpu=native -o sp_harness2 sp_harness2.rs

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

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
const EMPTY_KEY: u64 = u64::MAX;

#[inline(always)]
fn pack_pair(l: TokenId, r: TokenId) -> u64 {
    ((l as u64) << 32) | (r as u64)
}
#[inline]
fn utf8_char_len(b: u8) -> usize {
    if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 }
}

// ============================ baseline structures ============================

#[derive(Clone, Copy)]
struct Symbol { token: TokenId, prev: u32, next: u32, len: u16 }

#[derive(Clone, Copy)]
struct HeapEntry { key: u64, right: u32, size: u32 }
impl HeapEntry {
    #[inline(always)]
    fn new(rank: u32, left: u32, right: u32, size: u32) -> Self {
        Self { key: ((rank as u64) << 32) | (left as u64), right, size }
    }
    #[inline(always)]
    fn left(&self) -> u32 { self.key as u32 }
}

struct RadixHeap {
    buckets: [Vec<HeapEntry>; 65],
    last_min: u64,
    len: usize,
    overflow: Vec<HeapEntry>,
    scratch: Vec<HeapEntry>,
    pub redistributions: u64,
    pub redistributed_entries: u64,
}

impl RadixHeap {
    fn new() -> Self {
        Self {
            buckets: std::array::from_fn(|_| Vec::new()),
            last_min: 0, len: 0, overflow: Vec::new(), scratch: Vec::new(),
            redistributions: 0, redistributed_entries: 0,
        }
    }
    #[inline]
    fn bucket_index(&self, key: u64) -> usize {
        if key == self.last_min { 0 } else { (64 - (key ^ self.last_min).leading_zeros()) as usize }
    }
    #[inline]
    fn push(&mut self, e: HeapEntry) {
        if e.key < self.last_min { self.overflow.push(e); }
        else { let i = self.bucket_index(e.key); self.buckets[i].push(e); }
        self.len += 1;
    }
    /// `scratch_reuse`: when true, redistribution drains through a retained
    /// scratch buffer instead of `mem::take` (which drops bucket capacity).
    fn pop(&mut self, scratch_reuse: bool) -> Option<HeapEntry> {
        if self.len == 0 { return None; }
        if !self.overflow.is_empty() {
            let mut oi = 0; let mut ok = self.overflow[0].key;
            for (i, e) in self.overflow.iter().enumerate().skip(1) {
                if e.key < ok { ok = e.key; oi = i; }
            }
            let mut nb = 0;
            while nb < 65 && self.buckets[nb].is_empty() { nb += 1; }
            let nmk = if nb < 65 {
                if nb == 0 { Some(self.last_min) } else { self.buckets[nb].iter().map(|e| e.key).min() }
            } else { None };
            if nmk.is_none() || ok <= nmk.unwrap() {
                let e = self.overflow.swap_remove(oi); self.len -= 1; return Some(e);
            }
        }
        let mut bi = 0;
        while bi < 65 && self.buckets[bi].is_empty() { bi += 1; }
        if bi >= 65 { return None; }
        if bi == 0 { self.len -= 1; return self.buckets[0].pop(); }
        let bucket = &mut self.buckets[bi];
        let mut mi = 0; let mut mk = bucket[0].key;
        for (i, e) in bucket.iter().enumerate().skip(1) {
            if e.key < mk { mk = e.key; mi = i; }
        }
        self.last_min = mk;
        let me = bucket.swap_remove(mi);
        self.redistributions += 1;
        self.redistributed_entries += bucket.len() as u64;
        if scratch_reuse {
            self.scratch.clear();
            self.scratch.append(bucket);
            let mut s = std::mem::take(&mut self.scratch);
            for e in s.drain(..) {
                let ni = self.bucket_index(e.key);
                self.buckets[ni].push(e);
            }
            self.scratch = s;
        } else {
            let entries: Vec<HeapEntry> = std::mem::take(bucket);
            for e in entries {
                let ni = self.bucket_index(e.key);
                self.buckets[ni].push(e);
            }
        }
        self.len -= 1;
        Some(me)
    }
    fn clear(&mut self) {
        for b in &mut self.buckets { b.clear(); }
        self.last_min = 0; self.len = 0; self.overflow.clear();
    }
}

pub struct EncodeState {
    symbols: Vec<Symbol>,
    heap: RadixHeap,
    lin: Vec<(TokenId, u16)>,
    ranks: Vec<u32>,
    mtoks: Vec<TokenId>,
}
impl EncodeState {
    pub fn new() -> Self {
        Self { symbols: Vec::new(), heap: RadixHeap::new(), lin: Vec::new(), ranks: Vec::new(), mtoks: Vec::new() }
    }
    fn clear(&mut self) { self.symbols.clear(); self.heap.clear(); }
}

// ============================ flat pair table (variant F) ============================

pub struct PairTable {
    keys: Box<[u64]>,
    vals: Box<[(TokenId, u32)]>,
    mask: usize,
}
impl PairTable {
    fn build(src: &FxMap<u64, (TokenId, u32)>) -> Self {
        let cap = (src.len() * 2).next_power_of_two().max(16);
        let mut keys = vec![EMPTY_KEY; cap].into_boxed_slice();
        let mut vals = vec![(0u32, 0u32); cap].into_boxed_slice();
        let mask = cap - 1;
        for (&k, &v) in src.iter() {
            let mut i = (k.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask;
            while keys[i] != EMPTY_KEY { i = (i + 1) & mask; }
            keys[i] = k; vals[i] = v;
        }
        Self { keys, vals, mask }
    }
    #[inline(always)]
    fn get(&self, key: u64) -> Option<(TokenId, u32)> {
        let mut i = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask;
        loop {
            let k = unsafe { *self.keys.get_unchecked(i) };
            if k == key { return Some(unsafe { *self.vals.get_unchecked(i) }); }
            if k == EMPTY_KEY { return None; }
            i = (i + 1) & self.mask;
        }
    }
}

// ============================ encoder ============================

pub struct SentencePieceBPE {
    pair_lookup: FxMap<u64, (TokenId, u32)>,
    pair_flat: PairTable,
    token_cache: FxMap<Vec<u8>, TokenId>,
    byte_lut: [TokenId; 256],
    token_lengths: Vec<u16>,
    vocab_size: usize,
    pub unit_split_safe: bool,
}

impl SentencePieceBPE {
    pub fn from_parts(vocab: &[(u32, Vec<u8>)], merges: &[(TokenId, TokenId)]) -> Self {
        let mut token_bytes: Vec<Vec<u8>> = vec![Vec::new(); vocab.len()];
        for (id, b) in vocab {
            if (*id as usize) < token_bytes.len() { token_bytes[*id as usize] = b.clone(); }
        }
        let bytes_to_id: FxMap<Vec<u8>, TokenId> =
            vocab.iter().map(|(id, b)| (b.clone(), *id)).collect();
        let mut pair_lookup: FxMap<u64, (TokenId, u32)> = FxMap::default();
        for (rank, &(l, r)) in merges.iter().enumerate() {
            let mut m = token_bytes[l as usize].clone();
            m.extend_from_slice(&token_bytes[r as usize]);
            if let Some(&mid) = bytes_to_id.get(&m) {
                pair_lookup.entry(pack_pair(l, r)).or_insert((mid, rank as u32));
            }
        }
        let mut byte_lut = [u32::MAX; 256];
        for (id, b) in vocab {
            if b.len() == 1 && byte_lut[b[0] as usize] == u32::MAX { byte_lut[b[0] as usize] = *id; }
        }
        let token_lengths: Vec<u16> = token_bytes.iter().map(|b| b.len() as u16).collect();
        let mut token_cache: FxMap<Vec<u8>, TokenId> = vocab.iter()
            .filter(|(_, b)| b.len() > 1).map(|(id, b)| (b.clone(), *id)).collect();
        for (bv, &t) in byte_lut.iter().enumerate() {
            if t != u32::MAX { token_cache.insert(vec![bv as u8], t); }
        }
        let unit_split_safe = !token_bytes.iter()
            .any(|b| b.windows(3).enumerate().any(|(i, w)| i > 0 && w == METASPACE));
        let pair_flat = PairTable::build(&pair_lookup);
        Self { pair_lookup, pair_flat, token_cache, byte_lut, token_lengths, vocab_size: vocab.len(), unit_split_safe }
    }

    pub fn vocab_size(&self) -> usize { self.vocab_size }
    pub fn num_merges(&self) -> usize { self.pair_lookup.len() }

    #[inline(always)]
    fn get_merge(&self, l: TokenId, r: TokenId) -> Option<(TokenId, u32)> {
        self.pair_lookup.get(&pack_pair(l, r)).copied()
    }
    #[inline(always)]
    fn get_merge_flat(&self, l: TokenId, r: TokenId) -> Option<(TokenId, u32)> {
        self.pair_flat.get(pack_pair(l, r))
    }

    // ---- symbol init: baseline (hash per character) ----
    fn init_symbols_into(&self, text: &[u8], symbols: &mut Vec<Symbol>) {
        let mut pos = 0;
        while pos < text.len() {
            let cl = utf8_char_len(text[pos]);
            let end = (pos + cl).min(text.len());
            let cb = &text[pos..end];
            let (token, len) = if let Some(&t) = self.token_cache.get(cb) {
                (t, cb.len())
            } else { (self.byte_lut[text[pos] as usize], 1) };
            if token != u32::MAX {
                let i = symbols.len() as u32;
                symbols.push(Symbol {
                    token, prev: if i == 0 { NONE } else { i - 1 }, next: NONE,
                    len: self.token_lengths.get(token as usize).copied().unwrap_or(len as u16),
                });
                if i > 0 { symbols[(i - 1) as usize].next = i; }
            }
            pos += len;
        }
    }

    // ---- symbol init: ASCII fast path (variant D) ----
    // token_cache contains every single-byte entry copied straight from
    // byte_lut, so for b < 0x80 the hash lookup is provably redundant.
    fn init_symbols_fast(&self, text: &[u8], symbols: &mut Vec<Symbol>) {
        let mut pos = 0;
        while pos < text.len() {
            let b0 = text[pos];
            let (token, len) = if b0 < 0x80 {
                (self.byte_lut[b0 as usize], 1usize)
            } else {
                let cl = utf8_char_len(b0);
                let end = (pos + cl).min(text.len());
                let cb = &text[pos..end];
                match self.token_cache.get(cb) {
                    Some(&t) => (t, cb.len()),
                    None => (self.byte_lut[b0 as usize], 1),
                }
            };
            if token != u32::MAX {
                let i = symbols.len() as u32;
                symbols.push(Symbol {
                    token, prev: if i == 0 { NONE } else { i - 1 }, next: NONE,
                    len: self.token_lengths.get(token as usize).copied().unwrap_or(len as u16),
                });
                if i > 0 { symbols[(i - 1) as usize].next = i; }
            }
            pos += len;
        }
    }

    fn init_heap(&self, symbols: &[Symbol], heap: &mut RadixHeap) {
        for i in 0..symbols.len().saturating_sub(1) {
            let l = &symbols[i]; let r = &symbols[i + 1];
            if let Some((_, rank)) = self.get_merge(l.token, r.token) {
                heap.push(HeapEntry::new(rank, i as u32, (i + 1) as u32, l.len as u32 + r.len as u32));
            }
        }
    }

    fn merge_loop(&self, symbols: &mut [Symbol], heap: &mut RadixHeap, scratch_reuse: bool) {
        while let Some(e) = heap.pop(scratch_reuse) {
            let li = e.left() as usize; let ri = e.right as usize;
            let (lt, ll, lnext) = { let s = &symbols[li]; (s.token, s.len, s.next) };
            let (rt, rl, rnext) = { let s = &symbols[ri]; (s.token, s.len, s.next) };
            if ll == 0 || rl == 0 { continue; }
            if lnext != e.right { continue; }
            if (ll as u32 + rl as u32) != e.size { continue; }
            let (mt, _) = self.get_merge(lt, rt).unwrap();
            let nl = ll + rl;
            symbols[li].token = mt; symbols[li].len = nl; symbols[li].next = rnext;
            symbols[ri].len = 0;
            if rnext != NONE { symbols[rnext as usize].prev = e.left(); }
            let lp = symbols[li].prev;
            if lp != NONE {
                let p = &symbols[lp as usize];
                if p.len > 0 {
                    if let Some((_, rank)) = self.get_merge(p.token, mt) {
                        heap.push(HeapEntry::new(rank, lp, e.left(), p.len as u32 + nl as u32));
                    }
                }
            }
            if rnext != NONE {
                let n = &symbols[rnext as usize];
                if n.len > 0 {
                    if let Some((_, rank)) = self.get_merge(mt, n.token) {
                        heap.push(HeapEntry::new(rank, e.left(), rnext, nl as u32 + n.len as u32));
                    }
                }
            }
        }
    }

    fn collect_into(&self, symbols: &[Symbol], out: &mut Vec<TokenId>) {
        let mut i = 0u32;
        while i != NONE && (i as usize) < symbols.len() {
            let s = &symbols[i as usize];
            if s.len > 0 { out.push(s.token); }
            i = s.next;
        }
    }

    /// VARIANT A — tokie today.
    pub fn encode_a(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>) {
        st.clear();
        if text.is_empty() { return; }
        if let Some(&t) = self.token_cache.get(text) { out.push(t); return; }
        self.init_symbols_into(text, &mut st.symbols);
        if st.symbols.is_empty() { return; }
        self.init_heap(&st.symbols, &mut st.heap);
        let mut syms = std::mem::take(&mut st.symbols);
        self.merge_loop(&mut syms, &mut st.heap, false);
        self.collect_into(&syms, out);
        st.symbols = syms;
    }

    /// Heap encode with the retained-scratch redistribution fix.
    fn encode_a_scratch(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>) {
        st.clear();
        if text.is_empty() { return; }
        if let Some(&t) = self.token_cache.get(text) { out.push(t); return; }
        self.init_symbols_into(text, &mut st.symbols);
        if st.symbols.is_empty() { return; }
        self.init_heap(&st.symbols, &mut st.heap);
        let mut syms = std::mem::take(&mut st.symbols);
        self.merge_loop(&mut syms, &mut st.heap, true);
        self.collect_into(&syms, out);
        st.symbols = syms;
    }

    /// VARIANT E core — O(n^2) leftmost-lowest-rank scan, no heap.
    /// Tie-break is `<` so the leftmost pair wins, matching the heap's
    /// composite (rank<<32)|left_index ordering exactly.
    fn encode_linear(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>, flat: bool) {
        if text.is_empty() { return; }
        if let Some(&t) = self.token_cache.get(text) { out.push(t); return; }
        let lin = &mut st.lin;
        lin.clear();
        let mut pos = 0;
        while pos < text.len() {
            let b0 = text[pos];
            let (token, len) = if b0 < 0x80 {
                (self.byte_lut[b0 as usize], 1usize)
            } else {
                let cl = utf8_char_len(b0);
                let end = (pos + cl).min(text.len());
                let cb = &text[pos..end];
                match self.token_cache.get(cb) { Some(&t) => (t, cb.len()), None => (self.byte_lut[b0 as usize], 1) }
            };
            if token != u32::MAX {
                let tl = self.token_lengths.get(token as usize).copied().unwrap_or(len as u16);
                lin.push((token, tl));
            }
            pos += len;
        }
        if lin.is_empty() { return; }
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            let mut best_tok = 0u32;
            for i in 0..lin.len().saturating_sub(1) {
                let m = if flat { self.get_merge_flat(lin[i].0, lin[i + 1].0) }
                        else { self.get_merge(lin[i].0, lin[i + 1].0) };
                if let Some((mt, rank)) = m {
                    if rank < best_rank { best_rank = rank; best_i = i; best_tok = mt; }
                }
            }
            if best_i == usize::MAX { break; }
            let nl = lin[best_i].1 + lin[best_i + 1].1;
            lin[best_i] = (best_tok, nl);
            lin.remove(best_i + 1);
        }
        for &(t, _) in lin.iter() { out.push(t); }
    }

    // ---------------- unit-level drivers ----------------

    pub fn encode_b(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in MetaspaceUnits::new(text) { self.encode_a(u, st, out); }
    }

    pub fn encode_c(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in MetaspaceUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_a(u, st, out);
            ca.insert(u, &out[m..]);
        }
    }

    /// D = C + ASCII fast path in symbol init.
    pub fn encode_d(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in MetaspaceUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            st.clear();
            if let Some(&t) = self.token_cache.get(u) { out.push(t); }
            else {
                self.init_symbols_fast(u, &mut st.symbols);
                if !st.symbols.is_empty() {
                    self.init_heap(&st.symbols, &mut st.heap);
                    let mut syms = std::mem::take(&mut st.symbols);
                    self.merge_loop(&mut syms, &mut st.heap, true);
                    self.collect_into(&syms, out);
                    st.symbols = syms;
                }
            }
            ca.insert(u, &out[m..]);
        }
    }

    /// E = D but the per-unit merge is the heap-free linear scan.
    pub fn encode_e(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in MetaspaceUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_linear(u, st, out, false);
            ca.insert(u, &out[m..]);
        }
    }

    /// F = E + flat open-addressed pair table.
    pub fn encode_f(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in MetaspaceUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_linear(u, st, out, true);
            ca.insert(u, &out[m..]);
        }
    }

    /// VARIANT H core — linear merge with an incremental rank array.
    ///
    /// The O(n^2) scan in `encode_linear` re-probes the pair table for every
    /// adjacent pair on every iteration. Here each pair's rank is probed once
    /// and cached in `ranks`; after a merge only the two pairs touching the
    /// merge point are re-probed. Argmin then scans a contiguous u32 array
    /// (cheap, vectorizable) instead of issuing hash probes.
    fn encode_linear_inc(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>) {
        if text.is_empty() { return; }
        if let Some(&t) = self.token_cache.get(text) { out.push(t); return; }
        let lin = &mut st.lin;
        lin.clear();
        let mut pos = 0;
        while pos < text.len() {
            let b0 = text[pos];
            let (token, len) = if b0 < 0x80 {
                (self.byte_lut[b0 as usize], 1usize)
            } else {
                let cl = utf8_char_len(b0);
                let end = (pos + cl).min(text.len());
                let cb = &text[pos..end];
                match self.token_cache.get(cb) { Some(&t) => (t, cb.len()), None => (self.byte_lut[b0 as usize], 1) }
            };
            if token != u32::MAX {
                let tl = self.token_lengths.get(token as usize).copied().unwrap_or(len as u16);
                lin.push((token, tl));
            }
            pos += len;
        }
        let n = lin.len();
        if n == 0 { return; }
        if n == 1 { out.push(lin[0].0); return; }

        let ranks = &mut st.ranks;
        let toks = &mut st.mtoks;
        ranks.clear(); toks.clear();
        ranks.reserve(n); toks.reserve(n);
        for i in 0..n - 1 {
            match self.get_merge_flat(lin[i].0, lin[i + 1].0) {
                Some((mt, r)) => { ranks.push(r); toks.push(mt); }
                None => { ranks.push(u32::MAX); toks.push(0); }
            }
        }
        loop {
            let mut best = u32::MAX;
            let mut bi = usize::MAX;
            for (i, &r) in ranks.iter().enumerate() {
                if r < best { best = r; bi = i; }
            }
            if bi == usize::MAX { break; }
            let nl = lin[bi].1 + lin[bi + 1].1;
            lin[bi] = (toks[bi], nl);
            lin.remove(bi + 1);
            ranks.remove(bi);
            toks.remove(bi);
            // Re-probe only the pairs adjacent to the merge point.
            if bi > 0 {
                match self.get_merge_flat(lin[bi - 1].0, lin[bi].0) {
                    Some((mt, r)) => { ranks[bi - 1] = r; toks[bi - 1] = mt; }
                    None => { ranks[bi - 1] = u32::MAX; }
                }
            }
            if bi < ranks.len() {
                match self.get_merge_flat(lin[bi].0, lin[bi + 1].0) {
                    Some((mt, r)) => { ranks[bi] = r; toks[bi] = mt; }
                    None => { ranks[bi] = u32::MAX; }
                }
            }
        }
        for &(t, _) in lin.iter() { out.push(t); }
    }

    /// H = G + incremental-rank linear merge.
    pub fn encode_h(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in FastUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_linear_inc(u, st, out);
            ca.insert(u, &out[m..]);
        }
    }

    /// H driver over a cache with an explicit front-table size.
    pub fn encode_h_bits(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        self.encode_h(text, st, ca, out)
    }

    /// J = H + a bigram-span cache layered over the unit cache.
    /// Tries a 2-unit span first; on a miss falls back to per-unit and
    /// records the span. Halves lookups on hot 2-unit sequences.
    pub fn encode_j(&self, text: &[u8], st: &mut EncodeState,
                    ca: &mut UnitCache, bi: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        let mut bounds: Vec<(usize, usize)> = Vec::new();
        {
            let mut it = FastUnits::new(text);
            let base = text.as_ptr() as usize;
            while let Some(u) = it.next() {
                let off = u.as_ptr() as usize - base;
                bounds.push((off, off + u.len()));
            }
        }
        let mut i = 0usize;
        while i < bounds.len() {
            if i + 1 < bounds.len() {
                let span = &text[bounds[i].0..bounds[i + 1].1];
                if span.len() <= SHORT_KEY_MAX && bi.lookup(span, out) {
                    i += 2;
                    continue;
                }
            }
            let u = &text[bounds[i].0..bounds[i].1];
            let m = out.len();
            if !ca.lookup(u, out) {
                self.encode_linear_inc(u, st, out);
                ca.insert(u, &out[m..]);
            }
            // record the span ending at i for next time
            if i + 1 < bounds.len() {
                let n = out.len();
                let u2 = &text[bounds[i + 1].0..bounds[i + 1].1];
                if !ca.lookup(u2, out) {
                    self.encode_linear_inc(u2, st, out);
                    ca.insert(u2, &out[n..]);
                }
                let span = &text[bounds[i].0..bounds[i + 1].1];
                if span.len() <= SHORT_KEY_MAX { bi.insert(span, &out[m..]); }
                i += 2;
            } else {
                i += 1;
            }
        }
    }

    /// I = H but with the inline-token front cache.
    pub fn encode_i(&self, text: &[u8], st: &mut EncodeState, ca: &mut InlineCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in FastUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_linear_inc(u, st, out);
            ca.insert(u, &out[m..]);
        }
    }

    /// G = F but units are found with a memchr-style 3-byte scan.
    pub fn encode_g(&self, text: &[u8], st: &mut EncodeState, ca: &mut UnitCache, out: &mut Vec<TokenId>) {
        if !self.unit_split_safe { return self.encode_a(text, st, out); }
        for u in FastUnits::new(text) {
            if ca.lookup(u, out) { continue; }
            let m = out.len();
            self.encode_linear(u, st, out, true);
            ca.insert(u, &out[m..]);
        }
    }

    /// Heap-only, no unit split, but with the scratch-reuse redistribution fix.
    /// Isolates how much of A's cost is the `mem::take` capacity churn.
    pub fn encode_a2(&self, text: &[u8], st: &mut EncodeState, out: &mut Vec<TokenId>) {
        self.encode_a_scratch(text, st, out)
    }
}

// ============================ unit iterators ============================

pub struct MetaspaceUnits<'a> { bytes: &'a [u8], pos: usize }
impl<'a> MetaspaceUnits<'a> {
    #[inline] pub fn new(b: &'a [u8]) -> Self { Self { bytes: b, pos: 0 } }
}
#[inline]
fn find_ms_naive(h: &[u8], from: usize) -> Option<usize> {
    let n = h.len();
    let mut i = from;
    while i + 3 <= n {
        if h[i] == 0xE2 && h[i + 1] == 0x96 && h[i + 2] == 0x81 { return Some(i); }
        i += 1;
    }
    None
}
impl<'a> Iterator for MetaspaceUnits<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.bytes.len() { return None; }
        let start = self.pos;
        let mut scan = start;
        while scan + 3 <= self.bytes.len()
            && self.bytes[scan] == 0xE2 && self.bytes[scan + 1] == 0x96 && self.bytes[scan + 2] == 0x81
        { scan += 3; }
        let end = find_ms_naive(self.bytes, scan).unwrap_or(self.bytes.len());
        self.pos = end;
        Some(&self.bytes[start..end])
    }
}

/// Scans for the 0xE2 lead byte in u64-sized strides (SWAR), then confirms.
pub struct FastUnits<'a> { bytes: &'a [u8], pos: usize }
impl<'a> FastUnits<'a> {
    #[inline] pub fn new(b: &'a [u8]) -> Self { Self { bytes: b, pos: 0 } }
}
#[inline]
fn find_e2(h: &[u8], from: usize) -> Option<usize> {
    let n = h.len();
    let mut i = from;
    // SWAR: locate a 0xE2 byte eight bytes at a time.
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    let pat = 0xE2u64.wrapping_mul(LO);
    while i + 8 <= n {
        let w = u64::from_le_bytes(h[i..i + 8].try_into().unwrap());
        let x = w ^ pat;
        let m = x.wrapping_sub(LO) & !x & HI;
        if m != 0 {
            let off = (m.trailing_zeros() / 8) as usize;
            return Some(i + off);
        }
        i += 8;
    }
    while i < n { if h[i] == 0xE2 { return Some(i); } i += 1; }
    None
}
#[inline]
fn find_ms_fast(h: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    loop {
        let p = find_e2(h, i)?;
        if p + 3 <= h.len() && h[p + 1] == 0x96 && h[p + 2] == 0x81 { return Some(p); }
        i = p + 1;
        if i >= h.len() { return None; }
    }
}
impl<'a> Iterator for FastUnits<'a> {
    type Item = &'a [u8];
    #[inline]
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.pos >= self.bytes.len() { return None; }
        let start = self.pos;
        let mut scan = start;
        while scan + 3 <= self.bytes.len()
            && self.bytes[scan] == 0xE2 && self.bytes[scan + 1] == 0x96 && self.bytes[scan + 2] == 0x81
        { scan += 3; }
        let end = find_ms_fast(self.bytes, scan).unwrap_or(self.bytes.len());
        self.pos = end;
        Some(&self.bytes[start..end])
    }
}

// ============================ unit cache ============================

const FRONT_BITS: u32 = 18;
const SHORT_KEY_MAX: usize = 15;

const INLINE_MAX: usize = 3;

/// Variant I cache: identical structure to `UnitCache` but the front table
/// stores up to `INLINE_MAX` token ids inline, removing the dependent load
/// into the arena for the common short-unit case. (tokie's PretokenCache
/// already uses inline slots; UnigramPieceCache chose arena indirection.)
pub struct InlineCache {
    arena: Vec<TokenId>,
    front_keys: Box<[u128]>,
    front_inline: Box<[[TokenId; INLINE_MAX]]>,
    front_meta: Box<[(u32, u32)]>, // (arena offset, len); len<=INLINE_MAX => use inline
    short: FxMap<u128, (u32, u32)>,
    long: FxMap<Box<[u8]>, (u32, u32)>,
    pub hits: u64,
    pub misses: u64,
}
impl InlineCache {
    pub fn new() -> Self {
        let n = 1usize << FRONT_BITS;
        Self {
            arena: Vec::new(),
            front_keys: vec![0u128; n].into_boxed_slice(),
            front_inline: vec![[0u32; INLINE_MAX]; n].into_boxed_slice(),
            front_meta: vec![(0u32, 0u32); n].into_boxed_slice(),
            short: FxMap::default(), long: FxMap::default(), hits: 0, misses: 0,
        }
    }
    #[inline]
    fn lookup(&mut self, u: &[u8], out: &mut Vec<TokenId>) -> bool {
        let found = if let Some(k) = UnitCache::pack_key(u) {
            let i = UnitCache::front_index(k);
            if self.front_keys[i] == k {
                let (o, l) = self.front_meta[i];
                if (l as usize) <= INLINE_MAX {
                    out.extend_from_slice(&self.front_inline[i][..l as usize]);
                } else {
                    out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]);
                }
                true
            } else if let Some(&(o, l)) = self.short.get(&k) {
                self.front_keys[i] = k; self.front_meta[i] = (o, l);
                if (l as usize) <= INLINE_MAX {
                    let src = &self.arena[o as usize..o as usize + l as usize];
                    let mut tmp = [0u32; INLINE_MAX];
                    tmp[..l as usize].copy_from_slice(src);
                    self.front_inline[i] = tmp;
                }
                out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]);
                true
            } else { false }
        } else if let Some(&(o, l)) = self.long.get(u) {
            out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]); true
        } else { false };
        if found { self.hits += 1; } else { self.misses += 1; }
        found
    }
    #[inline]
    fn insert(&mut self, u: &[u8], t: &[TokenId]) {
        if u.is_empty() || t.is_empty() { return; }
        let o = self.arena.len() as u32; let l = t.len() as u32;
        self.arena.extend_from_slice(t);
        if let Some(k) = UnitCache::pack_key(u) {
            self.short.insert(k, (o, l));
            let i = UnitCache::front_index(k);
            self.front_keys[i] = k; self.front_meta[i] = (o, l);
            if (l as usize) <= INLINE_MAX {
                let mut tmp = [0u32; INLINE_MAX];
                tmp[..l as usize].copy_from_slice(t);
                self.front_inline[i] = tmp;
            }
        } else { self.long.insert(u.to_vec().into_boxed_slice(), (o, l)); }
    }
}

pub struct UnitCache {
    bits: u32,
    arena: Vec<TokenId>,
    front_keys: Box<[u128]>,
    front_vals: Box<[(u32, u32)]>,
    short: FxMap<u128, (u32, u32)>,
    long: FxMap<Box<[u8]>, (u32, u32)>,
    pub hits: u64,
    pub misses: u64,
}
impl UnitCache {
    pub fn new() -> Self { Self::with_bits(FRONT_BITS) }
    pub fn with_bits(bits: u32) -> Self {
        let n = 1usize << bits;
        Self {
            bits,
            arena: Vec::new(),
            front_keys: vec![0u128; n].into_boxed_slice(),
            front_vals: vec![(0u32, 0u32); n].into_boxed_slice(),
            short: FxMap::default(), long: FxMap::default(), hits: 0, misses: 0,
        }
    }
    #[inline]
    pub fn front_index(k: u128) -> usize {
        let f = (k as u64) ^ ((k >> 64) as u64);
        ((f.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - FRONT_BITS)) as usize
    }
    #[inline]
    fn front_index_bits(k: u128, bits: u32) -> usize {
        let f = (k as u64) ^ ((k >> 64) as u64);
        ((f.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> (64 - bits)) as usize
    }
    #[inline]
    pub fn pack_key(b: &[u8]) -> Option<u128> {
        let n = b.len();
        if n == 0 || n > SHORT_KEY_MAX { return None; }
        let mut l = [0u8; 16];
        l[..n].copy_from_slice(b);
        Some(u128::from_le_bytes(l) | ((n as u128) << 120))
    }
    #[inline]
    fn lookup(&mut self, u: &[u8], out: &mut Vec<TokenId>) -> bool {
        let found = if let Some(k) = Self::pack_key(u) {
            let i = Self::front_index_bits(k, self.bits);
            if self.front_keys[i] == k {
                let (o, l) = self.front_vals[i];
                out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]); true
            } else if let Some(&(o, l)) = self.short.get(&k) {
                self.front_keys[i] = k; self.front_vals[i] = (o, l);
                out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]); true
            } else { false }
        } else if let Some(&(o, l)) = self.long.get(u) {
            out.extend_from_slice(&self.arena[o as usize..o as usize + l as usize]); true
        } else { false };
        if found { self.hits += 1; } else { self.misses += 1; }
        found
    }
    #[inline]
    fn insert(&mut self, u: &[u8], t: &[TokenId]) {
        if u.is_empty() || t.is_empty() { return; }
        let o = self.arena.len() as u32; let l = t.len() as u32;
        self.arena.extend_from_slice(t);
        if let Some(k) = Self::pack_key(u) {
            self.short.insert(k, (o, l));
            let i = Self::front_index_bits(k, self.bits);
            self.front_keys[i] = k; self.front_vals[i] = (o, l);
        } else { self.long.insert(u.to_vec().into_boxed_slice(), (o, l)); }
    }
}

// ============================ io + driver ============================

fn hex_decode(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).unwrap_or(0)).collect()
}
fn load_vocab(p: &str) -> Vec<(u32, Vec<u8>)> {
    std::fs::read_to_string(p).expect("vocab").lines().filter_map(|l| {
        let mut it = l.split('\t');
        Some((it.next()?.parse().ok()?, hex_decode(it.next()?)))
    }).collect()
}
fn load_merges(p: &str) -> Vec<(TokenId, TokenId)> {
    std::fs::read_to_string(p).expect("merges").lines().filter_map(|l| {
        let mut it = l.split('\t');
        Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
    }).collect()
}
fn metaspace_normalize(t: &str) -> Vec<u8> {
    let mut o = Vec::with_capacity(t.len() + t.len() / 4 + 3);
    o.extend_from_slice(&METASPACE);
    for c in t.chars() {
        if c == ' ' { o.extend_from_slice(&METASPACE); }
        else { let mut b = [0u8; 4]; o.extend_from_slice(c.encode_utf8(&mut b).as_bytes()); }
    }
    o
}

fn bench<F: FnMut() -> usize>(label: &str, bytes: usize, reps: usize, mut f: F) {
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        let _ = f();
        let e = t.elapsed().as_secs_f64();
        if e < best { best = e; }
    }
    println!("  {:<38} {:>9.2} MB/s   {:>9.2} ms", label, (bytes as f64 / 1048576.0) / best, best * 1000.0);
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let vp = a.get(1).cloned().unwrap_or("spbpe_hf32k.vocab.tsv".into());
    let mp = a.get(2).cloned().unwrap_or("spbpe_hf32k.merges.tsv".into());
    let cp = a.get(3).cloned().unwrap_or("corpus/bench.txt".into());
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);

    let enc = SentencePieceBPE::from_parts(&load_vocab(&vp), &load_merges(&mp));
    let raw = std::fs::read_to_string(&cp).expect("corpus");
    let text = metaspace_normalize(&raw);
    let nb = text.len();
    let nu = MetaspaceUnits::new(&text).count();

    println!("== SP-BPE optimization ladder ==");
    println!("vocab={} merges={} split_safe={} corpus={:.2} MB units={} ({:.2} B/unit)\n",
        enc.vocab_size(), enc.num_merges(), enc.unit_split_safe, nb as f64 / 1048576.0, nu, nb as f64 / nu as f64);

    let mut st = EncodeState::new();
    let mut ref_out = Vec::new();
    enc.encode_a(&text, &mut st, &mut ref_out);
    println!("reference A: {} tokens", ref_out.len());

    // Optional external ground truth (HF `tokenizers` ids, one per line).
    if let Some(hf_path) = a.get(5) {
        match std::fs::read_to_string(hf_path) {
            Ok(s) => {
                let hf: Vec<TokenId> = s.lines().filter_map(|l| l.trim().parse().ok()).collect();
                if hf == ref_out {
                    println!("HF ground truth: EXACT MATCH ({} ids)", hf.len());
                } else {
                    let d = hf.iter().zip(ref_out.iter()).position(|(x, y)| x != y);
                    println!("HF ground truth: MISMATCH at {:?} (hf={} a={})", d, hf.len(), ref_out.len());
                }
            }
            Err(e) => println!("HF ground truth: unreadable ({e})"),
        }
    }
    println!();

    // identity checks
    let mut check = |name: &str, v: &Vec<TokenId>| {
        let ok = *v == ref_out;
        println!("  {:<4} == A : {}{}", name, if ok { "YES" } else { "NO" },
            if ok { String::new() } else {
                let d = ref_out.iter().zip(v.iter()).position(|(x, y)| x != y);
                format!("  <-- DIVERGENCE at {:?} (A={} V={})", d, ref_out.len(), v.len())
            });
        ok
    };
    println!("identity:");
    let mut o = Vec::new(); enc.encode_a2(&text, &mut st, &mut o); check("A2", &o);
    o.clear(); enc.encode_b(&text, &mut st, &mut o); check("B", &o);
    let mut c1 = UnitCache::new(); o.clear(); enc.encode_c(&text, &mut st, &mut c1, &mut o); check("C", &o);
    let mut c2 = UnitCache::new(); o.clear(); enc.encode_d(&text, &mut st, &mut c2, &mut o); check("D", &o);
    let mut c3 = UnitCache::new(); o.clear(); enc.encode_e(&text, &mut st, &mut c3, &mut o); check("E", &o);
    let mut c4 = UnitCache::new(); o.clear(); enc.encode_f(&text, &mut st, &mut c4, &mut o); check("F", &o);
    let mut c5 = UnitCache::new(); o.clear(); enc.encode_g(&text, &mut st, &mut c5, &mut o); check("G", &o);
    let mut c6 = UnitCache::new(); o.clear(); enc.encode_h(&text, &mut st, &mut c6, &mut o); check("H", &o);
    let mut c7 = InlineCache::new(); o.clear(); enc.encode_i(&text, &mut st, &mut c7, &mut o); check("I", &o);
    let mut c8 = UnitCache::new(); let mut c8b = UnitCache::new();
    o.clear(); enc.encode_j(&text, &mut st, &mut c8, &mut c8b, &mut o); check("J", &o);
    println!("  bigram-span cache: hits={} misses={} ({:.1}% of spans)",
        c8b.hits, c8b.misses, 100.0*c8b.hits as f64/(c8b.hits+c8b.misses).max(1) as f64);

    // token-per-unit distribution (sizes the inline slots)
    {
        let mut hist = [0u64; 9];
        let mut st2 = EncodeState::new();
        for u in FastUnits::new(&text) {
            let mut t = Vec::new();
            enc.encode_linear_inc(u, &mut st2, &mut t);
            hist[t.len().min(8)] += 1;
        }
        let tot: u64 = hist.iter().sum();
        let cum3: u64 = hist[1] + hist[2] + hist[3];
        println!("  tokens/unit: 1={:.1}% 2={:.1}% 3={:.1}% <=3={:.1}% >3={:.1}%",
            100.0*hist[1] as f64/tot as f64, 100.0*hist[2] as f64/tot as f64,
            100.0*hist[3] as f64/tot as f64, 100.0*cum3 as f64/tot as f64,
            100.0*(tot-cum3) as f64/tot as f64);
    }
    println!("  cache hit rate (cold pass): {:.2}%\n",
        100.0 * c5.hits as f64 / (c5.hits + c5.misses).max(1) as f64);

    println!("COLD (fresh cache each rep) — best of {}:", reps);
    bench("A  whole-input heap [tokie today]", nb, reps, || { let mut o = Vec::new(); enc.encode_a(&text, &mut st, &mut o); o.len() });
    bench("A2 + scratch-reuse redistribution", nb, reps, || { let mut o = Vec::new(); enc.encode_a2(&text, &mut st, &mut o); o.len() });
    bench("B  per-unit, no cache", nb, reps, || { let mut o = Vec::new(); enc.encode_b(&text, &mut st, &mut o); o.len() });
    bench("C  per-unit + memo", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_c(&text, &mut st, &mut c, &mut o); o.len() });
    bench("D  C + ASCII symbol init", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_d(&text, &mut st, &mut c, &mut o); o.len() });
    bench("E  D + heap-free linear merge", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_e(&text, &mut st, &mut c, &mut o); o.len() });
    bench("F  E + flat pair table", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_f(&text, &mut st, &mut c, &mut o); o.len() });
    bench("G  F + SWAR unit split", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_g(&text, &mut st, &mut c, &mut o); o.len() });
    bench("H  G + incremental-rank merge", nb, reps, || { let mut c = UnitCache::new(); let mut o = Vec::new(); enc.encode_h(&text, &mut st, &mut c, &mut o); o.len() });
    bench("I  H + inline-token front cache", nb, reps, || { let mut c = InlineCache::new(); let mut o = Vec::new(); enc.encode_i(&text, &mut st, &mut c, &mut o); o.len() });

    bench("J  H + bigram-span cache", nb, reps, || {
        let mut c = UnitCache::new(); let mut b = UnitCache::new(); let mut o = Vec::new();
        enc.encode_j(&text, &mut st, &mut c, &mut b, &mut o); o.len() });

    println!("\ndiagnostics (isolating fixed costs):");
    bench("   unit split only, naive scan", nb, reps, || MetaspaceUnits::new(&text).count());
    bench("   unit split only, SWAR scan", nb, reps, || FastUnits::new(&text).count());
    {
        let mut c = UnitCache::new();
        { let mut o = Vec::new(); enc.encode_h(&text, &mut st, &mut c, &mut o); }
        bench("   split + cache lookup only", nb, reps, || {
            let mut o = Vec::with_capacity(nb / 3);
            let mut miss = 0usize;
            for u in FastUnits::new(&text) { if !c.lookup(u, &mut o) { miss += 1; } }
            miss
        });
    }

    println!("\nWARM (cache persists, steady-state batch) — best of {}:", reps);
    macro_rules! warmbench {
        ($name:expr, $m:ident) => {{
            let mut c = UnitCache::new();
            { let mut o = Vec::new(); enc.$m(&text, &mut st, &mut c, &mut o); }
            bench($name, nb, reps, || { let mut o = Vec::with_capacity(nb / 3); enc.$m(&text, &mut st, &mut c, &mut o); o.len() });
        }};
    }
    // ---- document-sized inputs: the realistic batch contract ----
    println!("\ndocument-sized inputs (corpus split into N-byte docs):");
    for docsz in [1024usize, 4096, 16384, 65536] {
        let mut docs: Vec<&[u8]> = Vec::new();
        let mut i = 0;
        while i < text.len() {
            let mut e = (i + docsz).min(text.len());
            // keep doc boundaries on a UTF-8 char boundary
            while e < text.len() && (text[e] & 0xC0) == 0x80 { e += 1; }
            docs.push(&text[i..e]);
            i = e;
        }
        let mut ba = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut o = Vec::with_capacity(nb / 3);
            for d in &docs { enc.encode_a(d, &mut st, &mut o); }
            let e = t.elapsed().as_secs_f64(); if e < ba { ba = e; }
        }
        let mut ch = UnitCache::with_bits(16);
        { let mut o = Vec::new(); for d in &docs { enc.encode_h(d, &mut st, &mut ch, &mut o); } }
        let mut bh = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut o = Vec::with_capacity(nb / 3);
            for d in &docs { enc.encode_h(d, &mut st, &mut ch, &mut o); }
            let e = t.elapsed().as_secs_f64(); if e < bh { bh = e; }
        }
        // cold-cache H (fresh cache per rep) = first-touch of a new corpus
        let mut bc = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut c = UnitCache::with_bits(16);
            let mut o = Vec::with_capacity(nb / 3);
            for d in &docs { enc.encode_h(d, &mut st, &mut c, &mut o); }
            let e = t.elapsed().as_secs_f64(); if e < bc { bc = e; }
        }
        let mb = nb as f64 / 1048576.0;
        println!("  doc={:>6}B n={:<6}  A {:>7.2}  H-cold {:>7.2}  H-warm {:>7.2} MB/s   (H-warm/A = {:.1}x)",
            docsz, docs.len(), mb/ba, mb/bc, mb/bh, ba/bh);
    }

    println!("\nfront-table size sweep (variant H):");
    for bits in [12u32, 14, 16, 18, 20] {
        let entries = 1usize << bits;
        let kib = entries * (16 + 8) / 1024;
        let mut c = UnitCache::with_bits(bits);
        { let mut o = Vec::new(); enc.encode_h_bits(&text, &mut st, &mut c, &mut o); }
        let (h0, m0) = (c.hits, c.misses);
        let mut best = f64::MAX;
        for _ in 0..reps {
            let t = Instant::now();
            let mut o = Vec::with_capacity(nb / 3);
            enc.encode_h_bits(&text, &mut st, &mut c, &mut o);
            let e = t.elapsed().as_secs_f64();
            if e < best { best = e; }
        }
        let hr = 100.0 * (c.hits - h0) as f64 / ((c.hits - h0) + (c.misses - m0)).max(1) as f64;
        println!("  bits={:<3} entries={:<8} table={:>6} KiB  warm {:>8.2} MB/s  hit {:.2}%",
            bits, entries, kib, (nb as f64 / 1048576.0) / best, hr);
    }

    {
        let mut c = UnitCache::new(); let mut b = UnitCache::new();
        { let mut o = Vec::new(); enc.encode_j(&text, &mut st, &mut c, &mut b, &mut o); }
        bench("J  H + bigram-span cache", nb, reps, || {
            let mut o = Vec::with_capacity(nb / 3);
            enc.encode_j(&text, &mut st, &mut c, &mut b, &mut o); o.len() });
    }
    warmbench!("C  per-unit + memo", encode_c);
    warmbench!("D  C + ASCII symbol init", encode_d);
    warmbench!("E  D + heap-free linear merge", encode_e);
    warmbench!("F  E + flat pair table", encode_f);
    warmbench!("G  F + SWAR unit split", encode_g);
    warmbench!("H  G + incremental-rank merge", encode_h);
    {
        let mut c = InlineCache::new();
        { let mut o = Vec::new(); enc.encode_i(&text, &mut st, &mut c, &mut o); }
        bench("I  H + inline-token front cache", nb, reps, || {
            let mut o = Vec::with_capacity(nb / 3); enc.encode_i(&text, &mut st, &mut c, &mut o); o.len() });
    }
}
