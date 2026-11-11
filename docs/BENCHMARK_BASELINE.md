# tokie vs gigatoken: OpenWebText baseline

Status: corrected baseline complete. Commit
`c59c8c8f04db6da4b7038a6debb235ffa680f2f3` (`bhavnick/api-014-compat`).
At capture time no tokenizer or pretokenizer source was changed; later
implementation edits are separate, and the preserved baseline binary is the
before side. This report describes the before-state.

Scope is the fast `encode` / `batch` core paths only. Legacy interfaces,
facades, and encode-plus paths are out of scope, and Python API source additions
get a separate baseline so Python builds do not run during Rust timing.

## Verdict

There is no single winner. The result depends on whether the corpus cache is
cold or warm:

- **First pass over fresh text (cold process, no warmup):** gigatoken is faster
  on 16 of 18 configurations, by 1.09x to 4.31x. The two exceptions are Qwen
  3.5 batch, where tokie is 1.7x to 4.7x faster.
- **Repeated passes over the same bytes (cache warm):** tokie is faster on 17
  of 18 configurations, median 1.19x faster, ratio range 0.985 to 1.449. The
  only configuration below parity is GPT-2 batch 1 thread (0.985); the o200k
  and cl100k 8-thread batch results are within 1% of parity.

The large cold/warm gap is the first baseline finding. tokie's first pass is
dominated by piece-cache construction: GPT-2 single goes from 136.0 MB/s cold
to 486.9 MB/s warm; o200k single goes from 45.0 to 407.0 MB/s; GPT-2 batch on
8 threads goes from 199.9 to 1645.2 MB/s. A corpus scan pays the cold number.
A service that re-encodes the same text pays the warm number.

This supersedes the first run, whose harness executed only one warmup per
engine even when asked for three or six. That defect is fixed below; the old
raw data is retained under `results/superseded-warmup1/` and is not the
baseline.

## Environment

| Item | Value |
| --- | --- |
| Host | `bnixos`, Linux 7.1.7, AMD Ryzen AI 9 HX 370, 12 physical / 24 logical CPUs, 30 GiB RAM |
| Load during the run | roughly 2 to 4 (Chrome, `next-server`, a Node process, and `paper-desktop` were active) |
| Isolation | `taskset`; no cgroup, no stopping of other work |
| Single-thread CPU | CPU 8 (`taskset -c 8`) |
| Batch 8-thread CPUs | CPUs 4-11 (eight distinct physical cores; SMT siblings are 12-23) |
| CPU profile | `powerprofilesctl set performance`; governor `performance` on CPU 8 |
| Rust | `rustc 1.100.0-nightly (bff8e12ff 2026-08-26)`, `cargo 1.100.0-nightly (e8cb624d5 2026-08-22)` |
| Build | release, `lto = "fat"`, `codegen-units = 1`; target `base-corrected-target` |
| Binary | `/tmp/opencode/tokie-compat/bin/compat-bench-base-corrected`, sha256 `3a865710e8196f3081a1dd1326c90f0d3d7cc9b77667b3d86dc8010a1bd043b5` |
| tokie | HEAD `c59c8c8f04db6da4b7038a6debb235ffa680f2f3`; only new harness/docs files untracked at measurement time |
| gigatoken | HEAD `fac0114b37120ec8a76362e9ee8e1c742aaafaef`, version 0.10.0, dirty worktree |
| gigatoken manifest path | `scripts/compat_bench/vendor/gigatoken` → `/home/bhavnick/Workspaces/gigatoken` (symlink made by `build.sh`; default is a sibling of the repo root, `GIGATOKEN_DIR` overrides) |
| gigatoken dirty files | `Cargo.toml`, `src/lib.rs`, `src/bpe/tiktoken.rs`, `src/bpe/sentencepiece.rs`, new `src/normalize.rs` |

The gigatoken BPE path was compiled from the local dirty worktree. The
`tiktoken.rs` diff only adds a public `normalize_nfc` helper; the measured BPE
encode code is not changed. The `sentencepiece.rs` changes do not apply to the
BPE models measured. Exact file hashes are in
`/tmp/opencode/tokie-compat/baseline_meta_base-corrected.json`.

## Corpus

| Item | Value |
| --- | --- |
| Dataset | `Skylion007/openwebtext` |
| Parquet | `plain_text/train-00000-of-00080.parquet` |
| Source blob sha256 | `caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e` |
| Selection | every 15th document, full documents, until 20 MiB |
| Documents | 4,187 (parquet rows 0 through 62,790) |
| Document text bytes | 20,974,828 |
| Joined text bytes | 20,979,014 (documents joined with one `\n`) |
| Joined text sha256 | `577e89123da82c3f6b7bb2333f8b1e0469e5b75f7d6ea8234189bfc912c4e67b` |
| JSONL sha256 | `20dc6601bee15f270b7a2c35cfde6cd2f073f8c15567c76b913453157d2c67db` |

`documents` mode times the 4,187 separate texts. `single` mode times the
joined 20,979,014-byte text. Both engines read the exact same bytes.

## Tokenizers

All are BPE families that both engines load from the same `tokenizer.json`.
`encode` is used on both sides, so the special-token template is off.

| Name | Token count, joined file | Source | tokenizer.json sha256 (first 16) |
| --- | ---: | --- | --- |
| gpt2 | 4,788,850 | cached `openai-community/gpt2` | `8414cab924d8b9b3` |
| cl100k | 4,534,217 | cached `Xenova/gpt-4` | `239eb2359f79c384` |
| o200k | 4,497,370 | cached `Xenova/gpt-4o` | `43a3ad4618a6a938` |
| qwen2 | 4,645,050 | cached `Qwen/Qwen2-7B` | `f7c9b2dba4a296b1` |
| qwen3.5 | 4,697,217 | cached `Qwen/Qwen3.5-4B` | `5f9e4d4901a92b99` |
| deepseek | 4,580,467 | cached `DeepSeek-V3` | `621ac2e32d0dba65` |

Exact paths and full sha256 values are in the metadata JSON.

## Method

- `--mode single`: one `encode` over the joined 20 MB string.
- `--mode batch`: 4,187 documents. Tokie uses `Tokenizer::batch`. Gigatoken
  uses `encode_docs_ragged` with its persistent `WorkerPool`, its documented
  batch path. Both use the same thread count.
- **Warm run** (same process): exactly three warmup passes per engine for
  single and six for batch, then 11 single or 15 batch alternating repeats;
  medians and min/max are over those repeats. The harness records both the
  requested `warmup` count and the executed `warmup_runs` count per engine in
  every JSON result. `test_warmup.sh` asserts that `--warmup 3` emits three
  events per engine, `--warmup 0` emits none, and `--reps 0` is rejected.
- **Cold run** (fresh process per sample): no warmup, five processes per side,
  alternating which engine runs first. Each process loads the tokenizer and
  corpus before timing, then does exactly one timed encode. Timing excludes
  load and includes cache construction.
- Timed regions call only in-memory encode APIs. Corpus and tokenizer reads
  happen before the timer; the timed closure has no file, network, or logging
  calls. Every timed result is consumed: token count and an FNV-1a checksum
  are recorded, and every repeat in a group must agree. The untimed perf
  profiles put the samples in encode/pretoken/cache functions, not in I/O.
- `--validate` checks tokie ids against gigatoken ids exactly, whole-file and
  per document, and checks tokie batch against serial encode. All six
  tokenizers passed.

Build and timing were separate. The build was done in
`/tmp/opencode/tokie-compat/base-corrected-target`; the binary was copied out,
so a later after-build cannot overwrite it. Tokie is a relative path in the
manifest (`../../crates/tokie`). Gigatoken is a relative manifest path
(`vendor/gigatoken`) resolved by `build.sh` from `GIGATOKEN_DIR`; the resolved
target is recorded in the metadata artifact.

### Superseded first run

The first full run (`base`, `base2`) used a harness whose warmup loop executed
one pass per engine even when asked for three or six. Those raw files are
retained under `/tmp/opencode/tokie-compat/results/superseded-warmup1/`, with
the old binary at `/tmp/opencode/tokie-compat/bin/compat-bench-base-warmup1`
(sha256 `46527b31bb0253e50dfe821bb6ec38b111de86534f63ab75695649be13e62d2f`)
and a `SUPERSEDED.md` explaining the defect. Their warm numbers are not cited
anywhere below. Cold used zero warmups and was semantically unaffected, but it
was rerun with the corrected binary for one clean provenance.

## Warm results (cache-warm, same process)

Throughput is input bytes / median wall time, `MB = 10^6` bytes.
`ratio` is gigatoken time / tokie time; greater than 1 means tokie is faster.

| model | mode | threads | tokie MB/s | gigatoken MB/s | ratio | tokie range ms | gigatoken range ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cl100k | batch | 1 | 402.5 | 391.5 | 1.028 | 48.7–54.4 | 49.8–56.2 |
| cl100k | batch | 8 | 1887.9 | 1882.8 | 1.003 | 10.5–16.7 | 8.4–15.2 |
| cl100k | single | 1 | 432.6 | 353.9 | 1.222 | 45.7–50.6 | 54.7–60.3 |
| deepseek | batch | 1 | 344.9 | 254.8 | 1.353 | 57.6–64.1 | 79.1–83.6 |
| deepseek | batch | 8 | 1731.0 | 1333.9 | 1.298 | 11.8–15.2 | 14.0–19.1 |
| deepseek | single | 1 | 354.0 | 244.4 | 1.449 | 57.6–62.8 | 82.5–91.5 |
| gpt2 | batch | 1 | 462.1 | 468.9 | 0.985 | 44.6–46.9 | 41.4–45.9 |
| gpt2 | batch | 8 | 1645.2 | 1586.9 | 1.037 | 8.8–17.1 | 8.2–16.2 |
| gpt2 | single | 1 | 486.9 | 416.1 | 1.170 | 40.3–43.8 | 46.9–53.4 |
| o200k | batch | 1 | 380.9 | 372.4 | 1.023 | 52.3–55.8 | 53.1–58.1 |
| o200k | batch | 8 | 1745.6 | 1733.4 | 1.007 | 11.5–19.3 | 8.6–18.1 |
| o200k | single | 1 | 407.0 | 334.1 | 1.218 | 48.8–54.9 | 57.4–66.1 |
| qwen2 | batch | 1 | 374.5 | 342.1 | 1.095 | 53.5–58.3 | 57.8–70.9 |
| qwen2 | batch | 8 | 1810.5 | 1328.1 | 1.363 | 10.8–18.0 | 11.4–20.7 |
| qwen2 | single | 1 | 334.1 | 275.9 | 1.211 | 59.8–64.9 | 70.3–78.0 |
| qwen3.5 | batch | 1 | 369.5 | 305.1 | 1.211 | 53.1–60.6 | 66.4–73.7 |
| qwen3.5 | batch | 8 | 1539.2 | 1399.2 | 1.100 | 10.9–18.7 | 10.3–19.2 |
| qwen3.5 | single | 1 | 322.4 | 250.1 | 1.289 | 63.1–66.8 | 76.2–85.8 |

Median ratio over all warm configurations: **1.190** (range 0.985 to 1.449).

## Cold results (fresh process, first pass)

| model | mode | threads | tokie MB/s | gigatoken MB/s | ratio | tokie range ms | gigatoken range ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cl100k | batch | 1 | 78.7 | 148.6 | 0.529 | 254.7–342.2 | 136.6–149.9 |
| cl100k | batch | 8 | 91.8 | 363.1 | 0.253 | 223.4–231.7 | 57.0–60.5 |
| cl100k | single | 1 | 81.4 | 192.3 | 0.423 | 246.4–282.8 | 106.1–110.9 |
| deepseek | batch | 1 | 64.9 | 111.8 | 0.580 | 319.0–327.4 | 183.3–197.4 |
| deepseek | batch | 8 | 67.8 | 276.2 | 0.246 | 291.9–321.2 | 75.0–81.3 |
| deepseek | single | 1 | 67.8 | 140.3 | 0.483 | 304.6–318.0 | 142.1–153.7 |
| gpt2 | batch | 1 | 128.9 | 194.4 | 0.663 | 156.5–163.4 | 107.0–110.9 |
| gpt2 | batch | 8 | 199.9 | 584.4 | 0.342 | 102.1–106.9 | 35.4–39.4 |
| gpt2 | single | 1 | 136.0 | 206.4 | 0.659 | 151.7–159.8 | 97.6–103.2 |
| o200k | batch | 1 | 44.3 | 112.8 | 0.393 | 459.3–500.8 | 183.8–187.7 |
| o200k | batch | 8 | 41.6 | 152.1 | 0.274 | 490.8–533.5 | 131.2–162.0 |
| o200k | single | 1 | 45.0 | 175.9 | 0.256 | 450.5–472.8 | 114.9–122.4 |
| qwen2 | batch | 1 | 60.9 | 122.5 | 0.497 | 338.5–362.8 | 167.3–175.4 |
| qwen2 | batch | 8 | 58.6 | 252.2 | 0.232 | 325.6–365.3 | 81.2–96.7 |
| qwen2 | single | 1 | 58.8 | 145.3 | 0.405 | 342.3–367.0 | 137.9–147.6 |
| qwen3.5 | batch | 1 | 160.6 | 93.0 | 1.727 | 129.4–132.2 | 216.6–243.5 |
| qwen3.5 | batch | 8 | 642.1 | 136.3 | 4.710 | 31.6–33.9 | 133.6–185.3 |
| qwen3.5 | single | 1 | 150.0 | 162.9 | 0.921 | 138.7–145.3 | 128.1–132.0 |

Median ratio over all cold configurations: **0.453** (range 0.232 to 4.710).
On the 16 configurations where gigatoken wins, it is 1.09x to 4.31x faster.

## Why (untimed profiles)

`perf record` on GPT-2, separate from all reported timings:

- tokie, single thread, warm: `emit_fast` 26.8%, `pretokie::walk::bounds_avx512<Gpt2>`
  15.6%, `cache::pack_bounds` 14.7%, `recover` 4.1%, `merge_small` 4.5%,
  `memchr` 2.0%, `ensure_scan_for` 2.0%.
- tokie, 8 threads, warm batch: `emit_fast` 23.6%, `pack_bounds` 12.9%,
  pretoken bounds 10.8%, `recover` 9.1%, `memmove` 7.5%, `memchr` 6.6%,
  `ensure_scan_for` 4.2%, `merge_small` 2.5%.
- gigatoken, single thread, warm: AVX-512 mask `fill_spans...` 28.0%,
  encode closure 26.4%, `probe_emit_slow` 4.7%, `merge_short` 4.3%,
  `memmove` 5.3%, `extended_masks` 3.7%.
- gigatoken, 8 threads, warm batch: encode closure 27.8%, mask fill 22.7%,
  `probe_emit_slow` 11.0%, `memmove` 8.4%, `merge_short` 6.3%,
  `memchr` 5.6%.

The hot path in both engines is the same shape: pretoken boundary walk, cache
probe/pack, and merge/emit. tokie's batch path additionally spends time in
`recover`, `ensure_scan_for`, and the final `memmove` gather. The cold gap is
not a different instruction mix; it is cache construction and per-worker cache
setup on the first call, which the warm numbers amortize.

## Correctness and verification

- All six tokenizers match gigatoken exactly for the whole 20 MB file and for
  each of the 4,187 documents.
- tokie `batch` equals tokie serial `encode`, and equals gigatoken
  `encode_docs_ragged`, for all six tokenizers at 1 and 8 threads.
- All 18 warm comparisons and all 18 cold comparisons have equal token counts
  and equal id checksums between tokie and gigatoken.
- The timed result is consumed on every repeat; a changed count or checksum
  aborts the run.
- The harness loads corpus and tokenizer before timing. The timed closure only
  calls in-memory `encode` / `batch` / `encode_docs_ragged`.

## Reproducibility check

The corrected baseline was run twice (`base-corrected`, `base-corrected2`)
under the same binary and affinity. The aggregate result was stable: median
warm tokie/gigatoken ratio 1.190 vs 1.201, tokie ahead in 17 vs 16 of 18 warm
configurations, and the same two Qwen 3.5 cold-batch exceptions. Per-config
warm batch-8 ratios moved by up to 30% between runs (for example GPT-2 1.037
vs 1.345), and some cold ratios moved by up to 14%. That is why the separate
runs are descriptive only. `compare.py` reports all token counts and checksums
identical between the two runs, but it is not a regression test.

## Paired before/after gate

`scripts/compat_bench/run_paired.py` is the authoritative gate. It launches the
saved before binary and the candidate after binary in the same time window,
alternating order for every pair. It runs only the fast `encode` / `batch`
paths; no facade or legacy API is measured. Run it alone: no build, Python
baseline, or other timing job should overlap it.

```bash
scripts/compat_bench/run_paired.py \
  --before-bin /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected \
  --after-bin  /tmp/opencode/tokie-compat/bin/compat-bench-after \
  --before-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json \
  --after-meta  /tmp/opencode/tokie-compat/baseline_meta_after.json \
  --out /tmp/opencode/tokie-compat/paired/after \
  --label after
```

The primary acceptance rule is **no statistically supported slowdown**, per
configuration. There is no default 5% (or any other) regression budget:

- `pairs = 7` default; at least 5 enforced. The default workload is all six
  models, single / batch-1 / batch-8, warm and cold: 36 rows.
- A warm pair is two independent processes, each with 3 or 6 warmups and 5
  timed reps; the process medians form the pair. A cold pair is two fresh
  processes with zero warmups and one timed encode each.
- `alpha = 0.05`. `p_slower` is the exact one-sided paired sign test
  `P(X >= slower_pairs)` under Binomial(n, 1/2), with ties dropped.
  `p_slower < alpha` fails at **any** observed magnitude.
- If `p_slower >= alpha`, the row passes even when the sample median is below
  1. That verdict is `no_measurable_change`: not proof of zero change, only no
  detected slowdown at this sample size. Every row reports median speedup, a
  2000-resample bootstrap 95% CI, observed delta percent, and `p_slower`.
- A failed row must be rerun focused to confirm, e.g.
  `--focus cold:gpt2:batch:8 --pairs 9`. `summary.json` records the failed
  configs and a suggested focus/pair count.
- With 7 pairs the sign test needs 7/7 slower for `p < 0.05`; use more pairs
  for the focused confirmation. This is a power limit, not a hidden threshold.
- `--noise-tolerance` is optional and **descriptive only**. It never changes
  pass/fail and never labels a percentage slowdown as accepted.
- Every pair's token count and checksum must match. A mismatch is a
  correctness failure, not a perf result.
- Before timing, the gate compares gigatoken HEAD, normalized dirty status, and
  every modified-file hash in both metadata files. If both sides also have the
  full source-tree hash, that is compared too. If the historical side lacks the
  full hash, the verified HEAD/status/file-hash fallback is allowed only when
  every dirty/untracked path is represented. Missing metadata, incomplete
  coverage, or any mismatch returns exit 2 without running. Never regenerate
  the historical metadata; use the enriched derivative.

The existing null run contains the case the policy must allow: warm GPT-2
single median speedup **0.966**, `p_slower` **0.188**. Under this rule that is
`no_measurable_change` and passes; the gate does not demand every noisy median
be at least 1.0. The null run will be re-exercised when timing is available.
Its raw data is under `/tmp/opencode/tokie-compat/paired/null-corrected/`.

`scripts/compat_bench/test_paired_policy.py` is a pure synthetic check of the
acceptance logic: a 1% slowdown with 5/5 slower pairs fails despite being under
5%; a 0.99 median with 3/5 slower pairs passes as no measurable change; the
optional descriptive noise tolerance never changes pass/fail. It was run with
no encode and passed.

## Metadata provenance

The historical baseline metadata is immutable:

```text
/tmp/opencode/tokie-compat/baseline_meta_base-corrected.json
sha256 eb60935eb67397e066840e1080bffb3f831cd1312ca7ab747eb0900c0d9a7bb4 (5731 bytes)
```

Do **not** rerun `collect_meta.py --label base-corrected`; the script now also
refuses to overwrite the existing file. That would replace historical
environment/tokie status with the current tree. The original
records gigatoken HEAD, dirty status, and hashes for every modified/untracked
file. Those together cover the tracked commit and all dirty source changes; the
file/hash set was verified complete against the current checkout.

A separate derivative was written after that verification:

```text
/tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json
sha256 fc9a7d0f7d8f83bd8224e1fdc875f9cd87844a5560946366d26bb82581afd032
```

The derivative copies the original environment, timestamps, binary identity,
tokie status, corpus, and tokenizer records unchanged. Its added
`gigatoken_source_sha256` is explicitly labeled
`computed_later_after_verification`; it is not presented as a historical build
hash. The provenance records the original metadata SHA, verified HEAD,
recorded/current status, modified-file hashes, and complete dirty-path coverage.

`run_paired.py` compares full source hashes when both sides have them. If the
historical side lacks one, it falls back to HEAD + normalized dirty status +
every modified-file hash, but only when the dirty manifest covers every status
path. Missing/incomplete coverage fails before timing.

Metadata-only checks (no encode) were run on these files:

- original → enriched: passes via the HEAD/status/file-hash fallback
- enriched → enriched: passes via full source hash
- removing `src/normalize.rs` from the modified manifest: fails as incomplete
  coverage
- a synthetic ` M Cargo.toml` first status line: normalized status still
  matches, raw status differs only by the historical leading-space strip

## Oracle tests

The oracle tests were not run concurrently with timing, and this baseline did
not modify them.

- `cargo test -p pretokie`: compiles; each pretokenizer has its own test file
  under `crates/pretokie/tests/` (gpt2, cl100k, qwen, o200k, deepseek, ...).
- `cargo test -p tokie`: compiles; `tests/pipeline.rs`, `tests/tokie_file.rs`.
  `tokie_file.rs` needs cached HF `gpt2`, `bert-base-uncased`, and `t5-small`
  tokenizers; all three are present.
- `bench/tokie/tests/oracles.rs` cannot be built from this worktree as written:
  `bench/tokie/Cargo.toml` points at sibling `../../../gigatoken` and
  `../../../tokenizers-v1`, neither of which exists under
  `/home/bhavnick/.local/share/opencode/worktree/e68f38/`. A later oracle run
  needs those paths supplied (symlink or temporary manifest patch outside the
  repo) or a different checkout.

## Weaknesses

- Shared desktop host, load 2-4, and no cgroup isolation. Affinity and
  interleaving reduce the bias but the ranges show the noise. The two
  corrected full runs agree on the aggregate direction, not on every
  configuration: per-config warm batch-8 ratios moved by up to 30%. Use
  `run_paired.py`, not two separate runs, for a regression call. The host
  reported `Discharging` after the session; AC state was not logged per sample,
  so thermal/power throttling is not fully controlled.
- This gate covers only the fast `encode` / `batch` core paths. No legacy
  interface, facade, or encode-plus path is measured.
- 20 MB is a requested sample, not the 11.9 GB OpenWebText train file. Cache
  saturation and cold-start cost on a larger corpus will differ.
- Warm numbers are repeated identical bytes. They are not a held-out corpus
  scan; use cold numbers for that.
- Only BPE families were compared. WordPiece and SentencePiece are different
  engine paths and were excluded to keep the comparison same-type. No HF
  tokenizers or tiktoken comparison was made.
- The gigatoken worktree is dirty. Its measured BPE code does not differ from
  HEAD in the relevant encode path, but a clean-HEAD rebuild should be used if
  this number is cited outside the repo.
- The measured public APIs differ in allocation policy: tokie appends into a
  caller-owned `Vec`, while gigatoken's batch returns owned buffers. That is
  the API each engine offers, not a bench-local choice.
- The local `perf` profiles include the harness's post-timing checksum in their
  sample denominator; the percentages above identify the encode hot spots, not
  an exact timed-region split.

## Rerun after implementation

Build without touching the baseline binary:

```bash
export PATH=/nix/store/z4c6k0mrlkwl3s4w9ysxc8vq1wylm3ms-gcc-wrapper-15.3.0/bin:$PATH
GIGATOKEN_DIR=/path/to/gigatoken \
  CARGO_TARGET_DIR=/tmp/opencode/tokie-compat/after-target \
  scripts/compat_bench/build.sh after

# Required build metadata. Never rerun collect_meta on base-corrected: the
# original historical metadata is immutable. The enriched derivative already
# exists here; this recipe regenerates it only in a fresh workspace and refuses
# to overwrite an existing derivative.
# scripts/compat_bench/enrich_baseline_meta.py \
#   --source-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.json \
#   --gigatoken-dir /path/to/gigatoken \
#   --out /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json
scripts/compat_bench/collect_meta.py --label after

# Descriptive whole-matrix run (optional; use the paired gate below for a call)
scripts/compat_bench/run_baseline.sh after
scripts/compat_bench/compare.py \
  /tmp/opencode/tokie-compat/results/base-corrected/summary.json \
  /tmp/opencode/tokie-compat/results/after/summary.json

# Authoritative paired gate
scripts/compat_bench/run_paired.py \
  --before-bin /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected \
  --after-bin  /tmp/opencode/tokie-compat/bin/compat-bench-after \
  --before-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json \
  --after-meta  /tmp/opencode/tokie-compat/baseline_meta_after.json \
  --out /tmp/opencode/tokie-compat/paired/after \
  --label after
```

Run the oracle suites separately, not during timing:

```bash
cargo test -p pretokie
cargo test -p tokie
```

Harness self-test (warmup count, zero warmup, positive reps):

```bash
scripts/compat_bench/test_warmup.sh \
  /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected
```

Owned harness files (new, no existing repo files changed):

- `scripts/compat_bench/make_corpus.py`
- `scripts/compat_bench/Cargo.toml`
- `scripts/compat_bench/build.sh`
- `scripts/compat_bench/vendor/.gitignore` (the `vendor/gigatoken` symlink is generated, not committed)
- `scripts/compat_bench/src/main.rs`
- `scripts/compat_bench/run_baseline.sh`
- `scripts/compat_bench/run_paired.py`
- `scripts/compat_bench/test_paired_policy.py`
- `scripts/compat_bench/summarize.py`
- `scripts/compat_bench/collect_meta.py`
- `scripts/compat_bench/enrich_baseline_meta.py`
- `scripts/compat_bench/compare.py`
- `scripts/compat_bench/test_warmup.sh`
- `docs/BENCHMARK_BASELINE.md` (this report)

Artifacts:

- `/tmp/opencode/tokie-compat/corpus/` — corpus and manifest
- `/tmp/opencode/tokie-compat/bin/compat-bench-base-corrected` — preserved corrected baseline binary
- `/tmp/opencode/tokie-compat/bin/compat-bench-base-warmup1` — preserved superseded binary, sha256 `46527b31bb0253e50dfe821bb6ec38b111de86534f63ab75695649be13e62d2f`
- `/tmp/opencode/tokie-compat/base-corrected-target/` — isolated corrected build
- `/tmp/opencode/tokie-compat/results/base-corrected/` — corrected raw JSON and summary
- `/tmp/opencode/tokie-compat/results/base-corrected2/` — corrected repeat run
- `/tmp/opencode/tokie-compat/results/superseded-warmup1/` — old raw results plus `SUPERSEDED.md`
- `/tmp/opencode/tokie-compat/paired/` — paired-runner output (null test included), plus `policy_test.txt` and `source_meta_checks.txt`
- `/tmp/opencode/tokie-compat/baseline_meta_base-corrected.json` — immutable historical identities/environment, sha256 `eb60935e…a7bb4`
- `/tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json` — separate attested derivative with full source hash, sha256 `fc9a7d0f…fd032`
- `/tmp/opencode/tokie-compat/perf_corrected_*` — corrected perf captures
- `/tmp/opencode/tokie-compat/perf_reports/` — old-profile reference (pre-fix)
- `/tmp/opencode/tokie-compat/perf_reports_corrected/` — corrected text `perf report` output
