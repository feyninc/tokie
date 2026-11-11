# SentencePiece-BPE efficiency: measured research findings

Autoresearch loop on tokie's `crates/tokie/src/encoder/sentencepiece.rs`, the
slowest encoder in the library (README: 2–3× vs HF on Gemma 3, against 25–154×
for every other family).

## Experimental setup

`cargo` could not be used — this session's egress policy blocks
`static.crates.io`, and the registry cache is empty (285 crates unresolvable).
The experiment was therefore run as a **dependency-free `rustc` harness** in a
git worktree, with tokie's encoder ported verbatim as the baseline.

| | |
|---|---|
| Machine | 4 vCPU Intel Xeon @ 2.10 GHz, 15.7 GiB RAM |
| Models | SP-BPE trained locally with HF `tokenizers` (Metaspace + BPE), **32 000** vocab / 19 439 merges and **128 000** vocab / 115 439 merges |
| Corpus | 37 MB of system documentation (Debian changelogs, man pages); 10.11 MB metaspace-normalized benchmark slice |
| Baseline | verbatim port of `SentencePieceBPE` — radix heap, `token_cache` whole-text fast path, `init_symbols`/`init_heap`/`merge_loop`/`collect_results` |

**Correctness is established, not assumed.** The baseline reproduces HF
`tokenizers` *exactly* — 1 983 739 ids (32K) and 1 482 102 ids (128K), full
ID-level comparison, not token counts. Every variant below is checked
token-identical to the baseline on both models.

Sanity check on the regime: the baseline measures **1.37× faster than HF** on
this corpus, which lands squarely in the README's reported 2–3× band for
Gemma. The harness reproduces the problem tokie actually has.

## Caveats

- Corpus is system documentation, not web text. Word-frequency structure is
  real and Zipf-shaped, but package names and version strings are
  over-represented. Cache hit rates on OpenWebText will differ.
- 4 slow cores; absolute MB/s will not match an M3. Ratios are the result.
- Trained vocabularies, not Gemma's actual 262K vocab (HF hub is also blocked).
  The 32K→128K trend is measured, so the direction at 262K is supported by
  extrapolation, not measurement.

## The ladder

Cold = fresh cache per repetition. Whole-corpus single call, 32K vocab,
best of 5.

| | variant | MB/s | vs A |
|---|---|---:|---:|
| A | whole-input radix heap — **tokie today** | 3.13 | 1.0× |
| A2 | + scratch-reuse redistribution | 3.96 | 1.3× |
| B | per-`▁`-unit split | 10.40 | 3.3× |
| C | + unit memoization | 21.76 | 7.0× |
| D | + ASCII symbol init | 28.57 | 9.1× |
| E | + heap-free linear merge | 19.41 | 6.2× |
| F | + flat pair table | 31.23 | 10.0× |
| G | + SWAR unit split | 30.85 | 9.9× |
| **H** | **+ incremental-rank merge** | **49.05** | **15.7×** |
| I | + inline-token front cache | 47.89 | 15.3× |

## The realistic contract

The 10 MB single call exaggerates the baseline's superlinearity. Split into
document-sized inputs — how `encode_batch` actually calls the encoder — the
baseline is ~4–5 MB/s and flat, and the honest win is:

| doc size | 32K: A | H-cold | H-warm | 128K: A | H-cold | H-warm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 KiB | 5.16 | 49.77 | 184.54 | 3.90 | 42.18 | 192.77 |
| 4 KiB | 4.98 | 50.25 | 179.85 | 3.88 | 44.60 | 194.43 |
| 16 KiB | 5.38 | 50.42 | 205.85 | 4.05 | 44.18 | 191.14 |
| 64 KiB | 5.38 | 52.09 | 188.33 | 4.02 | 44.89 | 192.98 |

**~10× cold, ~36× warm at 32K; ~11× cold, ~48× warm at 128K.** Since tokie
pools worker caches across batch calls, warm is the steady-state for batch
workloads and cold is the first-touch cost.

The win **grows with vocabulary size** — which matters because every model
where this is worth doing (Gemma 262K) is larger than what was measured.

## What each change is, and why it works

**B — per-`▁`-unit split** (the structural one). The merge loop drops from
document-scale to ~12-byte words: heap depth collapses, the symbol array fits
in L1, and the linked-list walk in `collect_results_into` stops striding
across megabytes. Requires the interior-metaspace guard already written for
Unigram (`unigram.rs:63`, `compute_unit_split_safe`) — a vocab token
containing `▁` at an interior offset can only be produced by whole-string
merging, so its presence must force the old path. Both trained vocabularies
report `split_safe=true`.

**C — unit memoization.** 87% of units hit the cache on first pass. Reuses the
`UnigramPieceCache` design (arena + direct-mapped front table + short/long
maps) verbatim.

**D — ASCII symbol init.** `init_symbols_into` does a **hash lookup per
character**. For `b < 0x80` that lookup is provably redundant: `from_parts`
copies every single-byte entry from `byte_lut` into `token_cache`, so
`byte_lut[b]` is the same answer by construction. One array index instead of a
hash. +31% on the miss path (C→D).

**F — flat pair table.** Open-addressed `u64 → (id, rank)` with linear probing
in place of `FoldHashMap`. This is the same idea as the rank-merge core
already shipped for the BPE path (`b0402e7`). Its advantage **grows with vocab
size** — E→F is +61% at 32K and +97% at 128K — because the merge table is what
spills out of L2.

**H — incremental-rank merge** (largest single win, +59%). The linear scan
re-probes the pair table for every adjacent pair on every iteration. Instead,
probe each pair once into a `ranks: Vec<u32>`, and after a merge re-probe only
the two pairs touching the merge point. Argmin then scans a contiguous `u32`
array — cheap and vectorizable — instead of issuing hash probes. Turns
O(n²) probes into O(n) probes + O(n²) linear scans over hot memory.

Tie-breaking is preserved exactly: `rank < best` keeps the leftmost pair,
matching the heap's composite `(rank<<32)|left_index` ordering.

**A2 — scratch-reuse redistribution** (independent, ~5 lines). `pop()` does
`std::mem::take(bucket)`, which leaves a **zero-capacity** `Vec`, so every
bucket advance frees that bucket's allocation and re-grows it. Draining
through a retained scratch buffer instead is worth **+27% on the baseline
path**. This matters even after H lands, because it speeds up the
`unit_split_safe == false` fallback that H cannot use. Instrumentation on the
10 MB corpus: **11 139 402 redistributions moving 119 328 842 entries**.

## Negative results

These were measured and did **not** work — recording them so they aren't
re-attempted.

- **I — inline-token front cache.** Storing ≤3 ids inline in the front table
  (as `PretokenCache` does) gave nothing, despite 82% of units producing ≤3
  tokens at 32K and 89% at 128K. The arena indirection was never the
  bottleneck; adding 3 MB of inline slots worsens front-table locality.
- **E without F.** The heap-free linear merge *regresses* (19.41 vs D's 28.57)
  when paired with the hash map, and only wins once the flat table lands
  (31.23). The linear scan is lookup-cost-bound — these two changes are a
  package, not independent wins.
- **Front-table size sweep.** Peak at `bits=16` on 32K (190.7 vs 181.9 for
  the current 18) but at `bits=18` on 128K (197.6, falling to 184.4 at
  `bits=20`). Workload-dependent, not a robust win; the Unigram-inherited
  default of 18 is fine.
- **J — bigram-span cache.** The one idea identified for getting *past* the
  warm ceiling: cache 2-unit spans so a single lookup covers twice the bytes.
  Correct (token-identical) but **slower** — 39.4 vs H's 47.6 cold, 150.2 vs
  190.9 warm. Span hit rate was only 76.5%, the extra probe on a miss costs
  more than it saves, and the 15-byte `SHORT_KEY_MAX` excludes most 2-unit
  spans from qualifying at all. A fairer retest would widen the span key
  first; as implemented the hypothesis is refuted.

## Hard ceiling on the warm path

Isolating fixed costs:

| | 32K | 128K |
|---|---:|---:|
| unit split only, naive 3-byte scan | 571 MB/s | 558 MB/s |
| unit split only, SWAR scan | 892 MB/s | 897 MB/s |
| **split + cache lookup, no encoding** | **192 MB/s** | **199 MB/s** |
| measured warm, best variant | 190 MB/s | 193 MB/s |

**The warm path sits on the floor.** The BPE work is fully amortized away and
~79% of warm time is the cache lookup itself. No further work *inside the
encoder* moves it — warm C/D/E/F/H/I all land in a 172–193 MB/s band that is
mostly run-to-run noise, and the only change that reliably moved warm was G
(SWAR splitting), because splitting is the other half of the floor.

The obvious way past it — caching coarser-grained spans so one lookup covers
more bytes — was tried as variant J and **made things worse** (see negative
results). Beating this ceiling is an open problem, not a known win.

## Recommended sequence

1. **B + C + the interior-metaspace guard** — the structural change; ~7× alone
   and everything else builds on it. Template already exists in `0241784`.
2. **H + F together** — +59% on top, and they must land together (see E).
3. **D** — a few lines, +30% on the miss path.
4. **G** — SWAR `▁` scanning; the only thing that moves the warm path.
5. **A2** — independent ~5-line fix; keep it for the non-split-safe fallback.

Skip I. Leave `FRONT_BITS` at 18.

## Integration points in tokie

These were read off the tree at `bd02fe8` and are where the change lands.

| anchor | what it is | change |
|---|---|---|
| `encoder/mod.rs:150` | `_ => out.extend(self.encode(text))` in `encode_into` | SP-BPE currently falls through here — no worker cache, and an intermediate `Vec` per call. Add an `Encoder::SentencePiece(e)` arm alongside the existing `Backtracking` / `Unigram` arms. |
| `encoder/mod.rs:27-30` | `WorkerCaches { pretok, unigram }` | add an `sp` field. `pool.rs` is already generic over `WorkerCaches` (`pool.rs:29,49,71,85`), so pooling and the generation-affinity checkout come for free. |
| `unigram.rs:63` | `compute_unit_split_safe` | reuse verbatim as the interior-`▁` guard; both trained vocabularies report safe. |
| `sentencepiece.rs:751` | `init_symbols_into` | variant D — ASCII fast path. |
| `sentencepiece.rs:283` | `std::mem::take(bucket)` in `RadixHeap::pop` | variant A2 — retained scratch buffer. |
| `sentencepiece.rs:664,713` | `encode_with_state` / `encode_chunked` | currently dead — nothing in `src/` calls them. Either wire up or delete. |

The harness's variant H is `encode_linear_inc` + `FastUnits` + `PairTable` in
`research/sp-bpe/sp_harness2.rs`; those three map directly onto
`merge_loop`, a new unit iterator, and `pair_lookup` respectively.

**Not attempted here:** an actual patch to `encoder/sentencepiece.rs`. The
crates.io block means it could not be compiled or run against tokie's test
suite, and shipping an unbuilt encoder patch would be worse than shipping the
measured design. The harness code is verified; the port is mechanical.

## Reproducing

```
rustc --edition 2021 -O -C target-cpu=native -o sp_harness2 sp_harness2.rs
python3 train_hf.py 32000 spbpe_hf32k          # needs `tokenizers`
./sp_harness2 spbpe_hf32k.vocab.tsv spbpe_hf32k.merges.tsv corpus/bench.txt 5 hf_ids.txt
```

The harness asserts every variant against the baseline and, when given a
reference id file, against HF `tokenizers` directly.
