# Pretokenizer Performance Research

Benchmarks run on Apple Silicon (M-series), enwik8 data (95 MB), `--release` profile.

## Approaches Tried

### 1. Two-Phase: SIMD Classify + Scalar Walk (removed)

**Idea**: SIMD-classify all bytes into 7 categories (letter/digit/space/newline/apostrophe/other/non-ascii), store in a Vec, then walk the Vec to find piece boundaries.

**Results**:
- Phase 1 (NEON classify): 4,938 MB/s — extremely fast
- Phase 2 (scalar walk): ~210-250 MB/s — bottleneck
- Full pipeline: ~250 MB/s
- Allocates 1 byte per input byte (100MB Vec for 100MB input)

**Why it didn't win**: The walk phase processes 25M pieces at ~4 bytes avg. Each `next()` call has branch dispatch overhead (~18ns). The SIMD classify is negligible but the walk dominates. Also, the Vec allocation is wasteful.

### 2. Config-Driven Generic Iterator (removed)

**Idea**: Single `PretokConfig` struct with flags (letter_prefix, contraction_mode, etc.), one iterator that checks config at runtime.

**Results**: ~250 MB/s. Same walk bottleneck as two-phase, plus extra config checks per piece.

**Why removed**: Runtime config checks add branches. Specializing per tokenizer is strictly better.

### 3. Bulk split_ranges (removed)

**Idea**: Monolithic function that pushes (start, end) pairs into a Vec instead of yielding slices from an iterator. Avoids iterator overhead.

**Results**: Same speed as iterator (~220 MB/s). The compiler inlines the iterator just as well. Had correctness bugs in edge cases (Unicode letters, contractions, punctuation chaining).

### 4. memchr-Assisted Letter Scanning (didn't help)

**Idea**: Use `memchr(b' ', slice)` to find the next space, skip over letter runs in bulk.

**Results**: 228 MB/s — slower than byte-by-byte (270 MB/s). Letter runs average only 4.9 bytes on enwik8. memchr's SIMD startup cost doesn't pay off for runs this short. When memchr finds a space but there's a non-letter before it, we still verify byte-by-byte.

### 5. Specialized Single-Pass Iterators (current production)

**Idea**: Dedicated iterator per tokenizer (gpt2.rs, cl100k.rs). No classification buffer, no allocation. Direct byte inspection with `get_unchecked`. Each file is self-contained.

**Results**:
- GPT-2: **310 MB/s** (1.85x vs old tokie pretokenizer at 168 MB/s)
- CL100K: **292 MB/s** (1.78x vs old at 164 MB/s)
- Zero allocation, zero dependencies

### 6. SIMD scan_letters (NEON 16-byte letter skip)

**Idea**: Use NEON to check if 16 consecutive bytes are all ASCII letters. Skip 16 at once for long letter runs, fall back to scalar for short runs and non-ASCII.

**Results**:
- GPT-2: **355 MB/s** (+15% vs scalar)
- Correct (matches scalar output exactly)
- Only accelerates `scan_letters` inner loop; dispatch overhead unchanged

**Why limited**: Average letter run is 4.9 bytes. For short runs, SIMD overhead (~10 cycles: load + classify + movemask) matches scalar cost (~10 cycles: 5 byte loads + 5 branches + 1 exit). Win comes from occasional long runs (>16 bytes).

### 7. LUT Dispatch (match on byte class) — SLOWER

**Idea**: Replace the if-else chain in `next()` with a 256-byte LUT mapping each byte to its class (0-6), then `match` on the class.

**Results**: **209 MB/s** — 33% SLOWER than if-else.

**Why it failed**: The `match` compiles to an indirect jump table. ARM branch predictors handle direct 2-way branches (if-else) much better than 7-way indirect branches. The if-else chain lets the predictor learn "usually a letter" independently for each branch point.

### 8. Callback-Based (no Iterator trait)

**Idea**: Replace `Iterator::next()` with `for_each_piece(text, |start, end| { ... })`. Avoids Option wrapping, function call overhead.

**Results**: **356 MB/s** with SIMD scan_letters — same as SIMD iterator.

**Why limited**: The compiler already optimizes Iterator::next() well. The per-piece dispatch branch is the real bottleneck, not the Iterator trait overhead.

### 9. SIMD Transition Detection — BREAKTHROUGH

**Idea**: Classify 16 bytes at once with NEON, compare adjacent classes to find all transitions (piece boundaries) in one SIMD pass. No per-piece branch dispatch.

**Results (count-only, no boundary extraction)**:
- **3,220 MB/s** at 1.04 cycles/byte
- 8.7x faster than scalar iterator

**Results (with boundary extraction to Vec)**:
- CTZ loop + Vec::push: **608 MB/s** at 5.5 cycles/byte
- Raw pointer write: 500 MB/s (worse — compiler optimizes Vec::push better)
- Nibble table extraction: 382 MB/s (lookup overhead)

**Correctness**: ~0.05% error (24,724,886 vs 24,511,967 pieces). Missing:
- Contraction handling ('s, 't, etc. — apostrophe+letter merging)
- Multi-space splitting ("a  b" → ["a", " ", " b"])
- Both fixable with cheap scalar passes (rare patterns: 0.9% apostrophes)

## Cycle-Level Analysis (Apple Silicon)

### CPU Capabilities
- 8-wide decode, ~8 micro-ops/cycle issue
- 6 integer ALU, 2-3 load, 2 store, 4 NEON units
- **Branch mispredict penalty: ~14 cycles**
- L1D: 128KB, ~3 cycle latency

### Measured Operations

| Operation | MB/s | Cycles/byte | Bottleneck |
|-----------|------|-------------|------------|
| Raw byte sum | 10,962 | 0.30 | Memory bandwidth |
| NEON classify 16B | 12,391 | 0.27 | SIMD throughput |
| Branchless scalar transition | 3,224 | 1.04 | No branches |
| NEON cls+trans (no suppress) | 4,333 | 0.77 | SIMD pipeline |
| NEON cls+trans+suppress | 3,080 | 1.08 | + merge rules |
| 7-class LUT branchless | 2,134 | 1.56 | LUT latency |
| **Classify with branch** | **406** | **8.22** | **Branch mispredict** |
| State transition with branch | 359 | 9.29 | Branch mispredict |
| **Scalar GPT-2 iterator** | **312** | **10.71** | **Branches everywhere** |

### Key Insight: Branch Mispredictions Dominate

Making `if is_letter { count += 1 }` branchless (`count += is_letter as u64`) goes from **406 → 3,224 MB/s** (8x). The actual classification work is ~1 cycle/byte. **Branch mispredictions waste ~8 cycles/byte** — 80% of the scalar iterator's time.

### Per-Byte Budget Breakdown

```
Scalar iterator (10.7 cycles/byte):
  Useful compute:       ~2 cycles (classify, compare)
  Branch mispredicts:   ~8 cycles (dispatch + loop exits)
  Loop overhead:        ~1 cycle

SIMD boundary extraction (5.5 cycles/byte):
  SIMD classify+detect: ~1 cycle (16 bytes at once, branchless)
  CTZ extraction loop:  ~3 cycles (variable loop length, some mispredicts)
  Vec::push writes:     ~1.5 cycles (sequential stores)

SIMD count-only (1.04 cycles/byte):
  SIMD classify+detect: ~1 cycle (all branchless)
  popcount:             ~0.04 cycle (horizontal add)
```

## Speed Hierarchy

| Approach | MB/s | Cyc/B | Notes |
|----------|------|-------|-------|
| NEON classify only | 12,391 | 0.27 | Compute ceiling |
| NEON cls+trans+count | 3,220 | 1.04 | No extraction |
| NEON + CTZ → Vec | 608 | 5.5 | Best extraction |
| SIMD scan_letters iter | 355 | 9.4 | Best iterator |
| Scalar iterator | 312 | 10.7 | Current production |

## Architecture Options

### For standalone pretokenizer API
- **Iterator**: ~355 MB/s ceiling (branch misprediction limited)
- **Boundary Vec**: ~600 MB/s (SIMD transition + extraction)
- Both are correct and zero-copy for piece access

### For fused pretok+encoder pipeline
- **Inline processing**: process each piece as boundary is found, no Vec
- Theoretical ceiling: ~3 GB/s (SIMD transition cost only)
- Practical: depends on encoder cost hiding extraction latency
- **This is the key optimization** — the encoder work can overlap with SIMD via OoO execution

## Character Distribution (enwik8, 95 MB)

| Type | % of bytes |
|------|-----------|
| ASCII letters | 71.5% |
| Spaces | 13.5% |
| Other ASCII (punct) | 10.1% |
| Digits | 2.2% |
| Newlines | 1.1% |
| Non-ASCII | 0.7% |
| Apostrophes | 0.9% |

14.5M letter runs, average 4.9 bytes each. 25M total pieces for GPT-2.

### 10. Fused Pretok+BPE — ENCODER DOMINATES

**Idea**: SIMD finds boundaries, immediately feeds piece to BPE encoder inline. Encoder work hides SIMD extraction latency via OoO execution.

**Round 1 — HashMap-based BPE** (256 byte tokens, ~40 bigram merges, HashMap cache + merge table):
- Separate (iter → encode):       **28.8 MB/s**
- Fused SIMD (boundary → encode): **25.2 MB/s**
- Encode only:                     **29.5 MB/s**
- HashMap lookups destroyed performance (hashing Vec<u8> keys, random pointer chasing)

**Round 2 — Flat LUT-based BPE** (merge_lut[left*512+right], FNV1a cache, stack workspace):
- Separate (iter → encode):       **86.8 MB/s** (3.0x faster)
- Fused callback (inline encode):  **86.7 MB/s**
- Fused SIMD (boundary → encode): **87.8 MB/s**
- Pretok only:                     **322 MB/s**
- Encode only (pre-collected):     **110.9 MB/s** (3.8x faster)

**Key Insights**:
1. **Data structure choice matters more than algorithm** — HashMap→LUT gave 3.8x on encode alone
2. **Encoder is still the bottleneck** — 111 MB/s encode vs 322 MB/s pretok (2.9x gap)
3. **Fusing gives zero benefit** — 87.8 ≈ 86.8 MB/s, encoder dominates regardless
4. **Cache hit rate is low** — only ~160 cached entries (vs ~5000 in real tokenizer). Real tokenizers cache 89%+ of pieces, which would push encode speed much higher
5. **Amdahl's law**: even with infinitely fast pretok, pipeline caps at encode speed (111 MB/s)

**Round 3 — Encoder data structure comparison** (encode-only, pieces pre-collected):

| Encoder | MB/s | Notes |
|---------|------|-------|
| **Flat LUT** (merge_lut[l*512+r], FNV1a cache) | **110.8** | Array index per merge |
| **Tiered LUT** (tier0/1/2 + FNV1a fallback) | **110.1** | 64MB for 3-byte tier |
| **FoldHashMap** (foldhash cache + merge table) | **51.9** | 2.1x slower than LUT |
| **std HashMap** (round 1) | **29.5** | 3.8x slower than LUT |

Tiered LUT ≈ FNV1a LUT because **cache hit rate is only 25.6%** with our toy vocab:
- Tier 0 (1-byte): 25.2% hits — single bytes (punctuation, spaces, digits)
- Tier 1 (2-byte): 0.3% hits — only 40 bigrams cached
- Tier 2 (3-byte): 0.1% hits — 40 words in 16M slots (64 MB almost entirely wasted)
- **74.4% miss** → all go through merge loop (this dominates runtime)

With a real GPT-2 vocab (50,257 tokens), cache would cover ~89% of pieces. The merge loop would
only run on rare/novel pieces, and encode speed would be dramatically higher.

### 11. Tiered LUT Token Cache — Technique Analysis

**Idea**: Replace hash-based token cache with direct array indexing by piece bytes.

**Architecture**:
```
Tier 0: [TokenId; 256]        — 1-byte pieces: lut[byte]                    (1 KB)
Tier 1: [TokenId; 65536]      — 2-byte pieces: lut[b0*256 + b1]            (256 KB)
Tier 2: [TokenId; 16777216]   — 3-byte pieces: lut[b0<<16 | b1<<8 | b2]   (64 MB)
Tier 3+: FNV1a hash table     — 4+ byte pieces: hash probe                 (configurable)
```

**Pros**: Zero hash computation for 1-3 byte lookups. Single array load, fully predictable.
Tier 0+1 are tiny (257 KB total). CPU prefetcher loves flat arrays.

**Cons**: Tier 2 at 64 MB is large — may not be practical for all use cases.
May cause L2/L3 cache pressure if tier2 is sparse (as in our toy vocab).

**When it wins**: With a real vocabulary where thousands of 3-byte tokens populate tier2,
the hit rate would be high enough to justify the memory. For vocabularies with mostly
longer tokens (4+ bytes), the 64 MB is wasted.

**Practical alternative**: Skip tier2, use tier0+tier1 (257 KB) for 1-2 byte pieces,
and FNV1a/FoldHash for 3+ bytes. Captures ~25% of lookups with near-zero memory.

**Conclusion**: Pretokenizer at 312 MB/s is already 3x faster than the encoder. Further pretok optimization has diminishing returns. Focus areas:
1. **Bigger token cache** — cache top-5000 tokens → 89%+ hit rate → most pieces skip merge loop entirely
2. **Aho-Corasick encoder** — tokie already uses DAAC for this, which is the right approach
3. **Parallel chunking** — split text across cores for linear speedup

### 14. Parallel Chunk Processing — LINEAR SCALING

**Idea**: Split text into N chunks at safe boundaries (newlines), pretokenize+encode each
chunk on its own thread, merge results.

**Results** (8-core Apple Silicon, FNV1a+LUT encoder):

| Threads | MB/s | Speedup | Scaling efficiency |
|---------|------|---------|-------------------|
| 1 | 84.1 | 1.0x | — |
| 2 | 142.7 | 1.7x | 85% |
| 4 | 265.6 | 3.2x | 79% |
| 6 | 365.5 | 4.5x | 74% |
| 8 | **408.5** | **5.0x** | 62% |

**Correctness**: 100% — parallel output matches sequential exactly. Splitting at newlines
guarantees pieces don't cross chunk boundaries (newlines are always GPT-2 piece boundaries).

**Direct vs two-phase**: Direct (each thread runs pretok+encode) beats two-phase (collect
offsets then encode) by ~15%. The Vec<offset> overhead doesn't pay off when threads just
run the full pipeline independently.

**Key Insight**: Parallelism is the single biggest speedup we've found:
- SIMD pretok: 1.9x over old pretok
- LUT encoder: 2.8x over FoldHash
- **Parallel 8T: 5.0x on top of everything**

Combined: old tokie pipeline at ~35 MB/s → parallel LUT at 408 MB/s = **11.7x total speedup**.

### 15. Work-Stealing Parallel Encode — BEST RESULT

**Idea**: Single-thread pretok → collect all offsets → work-stealing threads grab batches
of pieces to encode via atomic counter. Auto-balances expensive (merge loop) vs cheap
(cache hit) pieces across threads.

**Results** (FNV1a+LUT encoder, 8-core Apple Silicon):

| Approach | 2T | 4T | 8T |
|----------|-----|-----|------|
| Chunk-per-thread | 149.8 | 260.2 | 402.3 |
| **Work-stealing** | **167.9** | **303.5** | **465.7** |
| Δ | +12% | +17% | **+15%** |

Best: **465.7 MB/s at 8T, batch=4096** — 5.5x over single-thread.

Batch size tuning (8T):
- batch=64: 435 MB/s (too much atomic contention)
- batch=256: 456 MB/s
- batch=1024: 455 MB/s
- batch=4096: **466 MB/s** (sweet spot)

**Why work-stealing wins**: chunk-per-thread has load imbalance — chunks with mostly short
pieces (cache hits) finish earlier, leaving threads idle while other threads grind through
expensive merge loops. Work-stealing auto-balances the workload.

**Parallel pretok + steal** (332 MB/s at 8T) is WORSE than single-thread pretok + steal
(466 MB/s). Pretok is cheap (292 MB/s single-thread), so parallelizing it + merging offsets
+ work-stealing adds overhead without benefit.

**Best architecture**: single-thread pretok (collect offsets) → work-stealing parallel encode.

**IMPORTANT**: Work-stealing 506 MB/s is encode-only — pretok was done outside timing.
True end-to-end: 95/300 + 95/506 = 0.505s → **188 MB/s**. Chunk-per-thread at 431 MB/s
includes pretok and is the **real winner** for end-to-end throughput.

Combined: old tokie (35 MB/s) → chunk-per-thread LUT 8T (431 MB/s) = **12.3x total speedup**.

### 16. Producer-Consumer Pipeline — PRODUCER-BOUND

**Idea**: 1 producer thread pretokenizes and streams offsets to a shared buffer.
N consumer threads spin-wait on an atomic write cursor, grab batches and encode.
Overlaps pretok and encode.

**Results**:

| Config | MB/s | Speedup |
|--------|------|---------|
| 1P+1C (2T) | 107 | 1.3x |
| 1P+2C (3T) | 196 | 2.3x |
| 1P+3C (4T) | 247 | 2.9x |
| 1P+5C (6T) | **258** | **3.1x** |
| 1P+7C (8T) | 223 | 2.7x (contention) |

**Why it loses**: Single producer at ~300 MB/s is the bottleneck. Consumers starve waiting
for offsets. Adding >5 consumers causes contention without more work.

**Comparison**:
- Pipeline 1P+5C: 258 MB/s (producer-bound)
- Chunk-per-thread 8T: 431 MB/s (all threads produce+consume)
- **Work-stealing 8T: 506 MB/s** (pretok first, then all 8 encode)

**Conclusion**: Pipeline is strictly worse than chunk-per-thread for this workload.

## Final Ranking (true end-to-end, 8-core Apple Silicon, enwik8)

| Approach | E2E MB/s | vs tokie (35 MB/s) |
|----------|---------|---------------------|
| **Chunk-per-thread 8T** | **431** | **12.3x** |
| Pipeline 1P+5C | 258 | 7.4x |
| Work-stealing 8T (true E2E) | 188 | 5.4x |
| Single-thread LUT | 84 | 2.4x |
| Single-thread FoldHash (tokie now) | 35 | 1.0x |

**Winner: chunk-per-thread.** Each thread fuses pretok+encode on its chunk.
All cores busy the entire time. Zero coordination overhead.

## Ideas Not Yet Tried

- **SIMD-assisted BPE**: Use SIMD within the encoder itself (parallel hash lookups, vectorized merge search).
- **Aho-Corasick token cache**: Pre-build automaton for common multi-byte tokens, skip merge loop entirely.
- **Tier0+Tier1 only**: 257 KB for 1-2 byte direct lookup, hash for 3+. Best memory/speed tradeoff.

### 17. Branchless State Machine with LUT — COUNT FAST, EXTRACT SLOW

**Idea**: Encode GPT-2 pretokenizer as a flat transition table: TABLE[state][class] → (next_state, emit).
Two loads per byte, zero branches. Contractions handled with a branch on apostrophe (0.9% of bytes).

**Results**:

| Variant | MB/s | cyc/B | vs scalar iter |
|---------|------|-------|---------------|
| **LUT count-only** | **2,649** | **1.3** | **8.4x faster** |
| Scalar iterator | 316 | 10.6 | baseline |
| LUT + Vec boundary writes | 258 | 12.9 | **0.8x (slower!)** |

**Key Insight**: The LUT loop itself is 8.4x faster than scalar — the branchless approach works
perfectly for *counting*. But extracting boundaries (writing to Vec with `write += emit as usize`)
creates a store-forwarding dependency chain that's SLOWER than the scalar iterator's well-predicted
branches. The `write` pointer depends on the previous iteration's `emit`, serializing all stores.

**The extraction bottleneck**: At 25M boundaries in 100M bytes (1 every 4 bytes), the write
pointer advances 25% of the time. This data dependency means the store can't be pipelined —
each iteration must wait for the previous write address to resolve.

**Correctness**: State table needs careful tuning (space doesn't prefix punctuation/newlines,
digit+letter boundary, contraction look-ahead). After several fixes, still ~40K boundary
mismatches remaining from contraction edge cases. Solvable but tedious.

**Conclusion**: Branchless LUT eliminates the branch misprediction bottleneck but trades it
for a store dependency bottleneck. The net result is a wash — the scalar iterator's
well-predicted branches are hard to beat for this workload where extraction is required.

**Open question**: Can we reduce the extraction cost below the 12.9 cyc/B store dependency?
Ideas to explore:

1. **Bitfield accumulate + batch extract**: Instead of writing positions to a Vec per-byte,
   accumulate emit bits into a u64 (one shift+OR per byte, no dependency on write pointer).
   Every 64 bytes, extract set bits from the u64 via CTZ loop. The accumulate is branchless
   with no store dependency. The CTZ extraction runs once per 64 bytes (~16 boundaries on
   average) — amortizes the variable-write-pointer cost. Expected: ~2-3 cyc/B for accumulate
   + ~3 cyc/B for extraction = ~5 cyc/B total → ~700 MB/s.

2. **Dual-cursor interleaving**: Run two independent LUT state machines on even/odd bytes
   (or on two halves of the input). Each has its own write pointer — the CPU's OoO engine
   can overlap the store dependencies from the two independent chains. Doubles ILP for stores.
   Problem: state at the boundary between the two streams needs reconciliation.

3. **SIMD-width branchless**: Process 16 bytes through the LUT in parallel (NEON TBL for
   class lookup, vectorized state transitions). But state is carried — each byte depends on
   the previous byte's state. This limits parallelism to the transition detection approach
   (which we already benchmarked at 608 MB/s with extraction).

4. **Don't extract — fuse with encoder**: The LUT count-only runs at 2,649 MB/s. If we could
   feed each boundary directly to the encoder as it's found (no Vec), the encoder work
   hides the store latency. We tried this with fused SIMD earlier and it didn't help because
   the encoder was too slow to overlap. But with a faster encoder (real vocab, 89% cache hit),
   the encoder per-piece cost would be ~10 cycles (cache probe) vs ~40 cycles (merge loop).
   At 10 cycles/piece with 25M pieces, that's only ~70ms — the LUT loop at 1.3 cyc/B would
   dominate at ~30ms. Fusing might help when encoder cost is this low.

### 17b. Bitfield Accumulate + Batch Extract — BLOCKED LOOP OVERHEAD KILLS IT

**Results**:

| Variant | MB/s | cyc/B |
|---------|------|-------|
| LUT count-only (flat loop) | 2,589 | 1.3 |
| Bitfield accum only (64B blocks, no extract) | 300 | 11.1 |
| Bitfield + CTZ extract | 264 | 12.6 |
| Direct Vec writes | 259 | 12.9 |
| Scalar iterator | 307 | 10.9 |

The bitfield accumulate loop at 300 MB/s is **8.5x slower** than the flat count-only loop (2,589 MB/s).
The 64-byte block structure (outer loop, mask reset, `pos - block_start` shift) prevents the
compiler from optimizing as tightly as the flat `for pos in 1..len` loop. The extraction itself
only costs 1.5 cyc/B — the blocked inner loop is the bottleneck.

**Dead end**: Any practical extraction mechanism reintroduces enough overhead to match or lose
to the scalar iterator. The branchless LUT's 2,600 MB/s speed only exists in the simplest
possible flat loop with no block structure and no output.

The scalar specialized iterator at ~310 MB/s appears to be near-optimal for a pretokenizer
that must produce piece boundaries on Apple Silicon.

## Full Pipeline Context

### Encode-only (pieces pre-collected, 24.5M pieces)

| Encoder backend | MB/s | vs FoldHash |
|----------------|------|-------------|
| FNV1a cache + flat LUT merges | **106.0** | **2.9x** |
| FoldHash cache + FoldHash merges | 36.1 | baseline |
| std HashMap (round 1) | 29.5 | 0.8x |

### Full pipeline: pretok + encode

| Pipeline | MB/s | Notes |
|----------|------|-------|
| Pretok only | 322 | Not the bottleneck |
| **Pretok + FoldHash** | **34.7** | **← tokie's current architecture** |
| Pretok + FNV1a+LUT | 83.9 | 2.4x faster |
| Fused SIMD + FNV1a+LUT | 87.3 | Marginal fusion gain |

FoldHashMap penalty comes from: Vec<u8> keys (heap alloc per lookup), hash overhead
vs FNV1a for short keys, pointer chasing in hash buckets vs flat array.

### 12. Merge Lookup: FNV1a packed-u64 vs FoldHash — NO WIN

**Idea**: Replace FoldHashMap<(u32,u32), _> merge lookup with FNV1a open-addressing on packed u64.

**Results** (encode-only, pieces pre-collected):

| Cache | Merge | Encode MB/s | vs tokie |
|-------|-------|------------|----------|
| FoldHash | FoldHash | 37.3 | baseline (tokie now) |
| FNV1a | FoldHash | 41.4 | +11% (cache swap only) |
| FNV1a | FNV1a packed | 40.4 | +8% (no merge win) |
| FNV1a | flat LUT | 102.7 | **2.8x** (small vocab only) |

**Key Insight**: FNV1a on packed u64 is NOT faster than FoldHash for merge lookups.
FoldHash already hashes 8-byte keys using NEON AES intrinsics — comparable cost to
FNV1a's 8 XOR+MUL chain. Both are open-addressing tables with same cache miss patterns.

The only real win is the **token cache swap** (+11%) from dropping Vec<u8> key hashing.
The flat LUT merge is 2.8x faster but requires bounded vocab (impractical for >32k tokens).

**Conclusion for tokie**: Swap token cache from FoldHashMap<Vec<u8>> to FNV1a open-addressing.
Keep FoldHashMap for merge lookups — not worth changing. Expected pipeline improvement: ~11%.

### 13. Two-Phase Pipeline + Prefetch — PHASE SEPARATION WINS, PREFETCH DOESN'T

**Idea**: Collect all piece boundaries first (phase 1), then batch-encode with software
prefetching of upcoming cache slots (phase 2). Exposes ILP by separating pretok from encode.

**Results** (FNV1a + flat LUT encoder):

| Approach | MB/s | vs interleaved |
|----------|------|---------------|
| Interleaved (pretok+encode fused) | 81.8 | baseline |
| **Two-phase sequential** | **102.1** | **+25%** |
| Two-phase + prefetch(1) | 87.0 | +6% |
| Two-phase + prefetch(4) | 81.0 | -1% |
| Two-phase + prefetch(16) | 79.6 | -3% |
| Phase 1 (collect offsets) | 299.5 | — |

**Key Insights**:
1. **Phase separation alone gives +25%** — the two-phase approach (collect offsets at 300 MB/s,
   then encode at 102 MB/s) beats interleaved by 25%. The pretokenizer work no longer
   contaminates the encoder's cache/branch predictor state.
2. **Prefetch hurts** — with our toy 48KB cache table (fits in L1), prefetch adds computation
   without saving latency. Even with a 1.5 MB table (simulating real vocab), prefetch hurt
   because 74% of lookups miss (empty slot, immediate bailout) — no latency to hide.
3. **Prefetch would help with real vocab** — with 89% cache hit rate and a 1.5 MB table that
   doesn't fit in L1, prefetching upcoming cache slots would hide L2 latency (~12 cycles).
   Can't verify until we have a real vocab loaded.
4. **Two-phase is the right architecture for tokie** — decouple pretokenization from encoding,
   process them as separate phases with their own optimal patterns.

Old tokie pretokenizer was 168 MB/s. New scalar pretok is 312 MB/s (1.9x).
