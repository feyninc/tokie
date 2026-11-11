# Final 0.2 benchmark and gate results

Status: **Final measurement complete. This is NOT a full strict performance certification or release sign-off.** The Rust quiet-window A/A passed and A/B returned `no_slowdown_detected_at_reported_resolution` (finite precision, not equivalence). The final Python matrix is `potential_slowdown`: one raw `o200k:batch:1` flag with Holm-adjusted p=1.0 and no familywise evidence, plus two batch-8 inconclusive rows. Direct AFTER-vs-gigatoken remains a separate descriptive comparison. Historical/exploratory sections below are retained and labeled; no more benchmark passes.

Final source-frozen binaries:

```text
before: /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected
        sha256 3a865710e8196f3081a1dd1326c90f0d3d7cc9b77667b3d86dc8010a1bd043b5
after:  /tmp/opencode/tokie-compat/bin/compat-bench-after
        sha256 3007bfdc0ebe0e30c7b92e2f74bc1a9ad71e92f982679cd36ff86a4e65314359
```

Build metadata and source freeze:

```text
before enriched metadata: /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json
  sha256 fc9a7d0f7d8f83bd8224e1fdc875f9cd87844a5560946366d26bb82581afd032
after metadata:           /tmp/opencode/tokie-compat/baseline_meta_after.json
  sha256 aea6a1e7ce9b17c6a6c30f26412e0f21520361b714e866e76a29404126ad11c8
gigatoken source sha256:  50b4ce8dbaf8e607351d548f654be45ee1f0c3115a604c64954c247c8dfae2b8
source-freeze mode:       full_source_hash, passed
```

The immutable historical baseline metadata was not overwritten.
`baseline_meta_base-corrected.json` remains
`eb60935eb67397e066840e1080bffb3f831cd1312ca7ab747eb0900c0d9a7bb4`.

## Final handoff (2026-11-11)

This is the review entry point. The later sections are historical/exploratory
unless explicitly marked final. Correctness gates are complete; the strict
performance criterion remains **UNCERTIFIED/partial**, with no release
sign-off. No more benchmark passes are planned. Correctness counts: 50 Tokie
(26 lib + 12 API + 8 pipeline + 4 file), 97 Pretokie, 28 Normalie, 16 normalize
oracles, 1 CLI, 4 tokenizer oracles, 1 ids (13 checked + 2 existing
exclusions: SmolLM ref unsupported, Bert-cased missing), and 37 Python; no
source edits after tests.

### Final Rust quiet-window pair

- A/A: 64 pairs, median ratio 0.997734 (speedup 1.002271), exact 95% CI
  0.989654–1.002035; two-sided sign p=0.168643; passed.
- A/B: 64 pairs, median after/before **1.000438** (speedup 0.999562), exact
  95% CI **0.998352–1.005683**, one-sided slowdown p **0.266154**, one-sided
  upper 1.005576. Verdict: `no_slowdown_detected_at_reported_resolution`.
  This is finite precision, not equivalence or proof of zero change.
- Protocol: ASLR off, CPU 8, matched-length aliases, counterbalanced, warmup 6,
  reps 5; all IDs/checksums matched (`4644146` / `6ec969d6e4a01923`).
- Evidence: `/tmp/opencode/tokie-compat/controlled-quiet-20261111/` —
  `summary_AA.json` `e510eb4c4acbf78442f26c9e43ac3d1bfbe0be7aec3a35c46ddff2d56ed05112`,
  `summary_AB.json` `88ae8c6b7458f2773dc444032e070b0ad03104c83648f63dee9af5369840f94c`,
  `plan.json` `fc8a1502f1c460a67126ce790e0ff8e066f431c278b8a32e1801290c3ffcecd8`.

### Final Python paired matrix

All 27 configs completed: 16 pairs each, 864 samples total. IDs/checksums
matched everywhere and `source_guard_unchanged=true`. Overall verdict is
**`potential_slowdown`**: the single raw flag is **`o200k:batch:1`**, median
speedup **0.9926726** (~0.733% slower), exact CI **0.98275–1.00207**, raw
p **0.038406**, Holm p **1.0** (unconfirmed, no familywise evidence).
`gpt2:batch:8` and `bert:batch:8` are inconclusive. Raw per-row
`significant_improvement` labels below are single-row labels, not familywise
or release claims.

| config | pairs | median speedup | exact median CI | delta % | F/S/T | p_slower | Holm | verdict |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| gpt2:scalar:1 | 16 | 1.00613 | 0.99887-1.01848 | +0.613 | 12/4/0 | 0.989365 | 1.000000 | significant_improvement |
| gpt2:batch:1 | 16 | 1.00728 | 0.98718-1.01676 | +0.728 | 9/7/0 | 0.772751 | 1.000000 | no_detected_regression |
| gpt2:batch:8 | 16 | 1.00813 | 0.90324-1.19853 | +0.813 | 9/7/0 | 0.772751 | 1.000000 | inconclusive |
| gpt2:decode:1 | 16 | 0.99656 | 0.98845-1.01523 | -0.344 | 8/8/0 | 0.598190 | 1.000000 | no_detected_regression |
| cl100k:scalar:1 | 16 | 1.00701 | 1.00162-1.01564 | +0.701 | 14/2/0 | 0.999741 | 1.000000 | significant_improvement |
| cl100k:batch:1 | 16 | 1.00879 | 1.00270-1.01670 | +0.879 | 13/3/0 | 0.997910 | 1.000000 | significant_improvement |
| cl100k:batch:8 | 16 | 1.00265 | 0.92843-1.07465 | +0.265 | 8/8/0 | 0.598190 | 1.000000 | no_detected_regression |
| cl100k:decode:1 | 16 | 1.00449 | 0.99333-1.01221 | +0.449 | 9/7/0 | 0.772751 | 1.000000 | no_detected_regression |
| o200k:scalar:1 | 16 | 0.99818 | 0.98883-1.00404 | -0.182 | 6/10/0 | 0.227249 | 1.000000 | no_detected_regression |
| o200k:batch:1 | 16 | 0.99267 | 0.98275-1.00207 | -0.733 | 4/12/0 | 0.038406 | 1.000000 | potential_slowdown |
| o200k:batch:8 | 16 | 1.01206 | 0.98034-1.03790 | +1.206 | 10/6/0 | 0.894943 | 1.000000 | no_detected_regression |
| o200k:decode:1 | 16 | 1.00432 | 0.98986-1.00826 | +0.432 | 9/7/0 | 0.772751 | 1.000000 | no_detected_regression |
| qwen2:scalar:1 | 16 | 1.01045 | 0.99175-1.02543 | +1.045 | 11/5/0 | 0.961594 | 1.000000 | no_detected_regression |
| qwen2:batch:1 | 16 | 1.00007 | 0.99409-1.01397 | +0.007 | 8/8/0 | 0.598190 | 1.000000 | no_detected_regression |
| qwen2:batch:8 | 16 | 0.98526 | 0.96312-1.03682 | -1.474 | 7/9/0 | 0.401810 | 1.000000 | no_detected_regression |
| qwen2:decode:1 | 16 | 1.00092 | 0.98970-1.00858 | +0.092 | 9/7/0 | 0.772751 | 1.000000 | no_detected_regression |
| qwen3.5:scalar:1 | 16 | 1.00575 | 0.99938-1.01018 | +0.575 | 11/5/0 | 0.961594 | 1.000000 | no_detected_regression |
| qwen3.5:batch:1 | 16 | 1.00625 | 0.99697-1.01307 | +0.625 | 11/5/0 | 0.961594 | 1.000000 | no_detected_regression |
| qwen3.5:batch:8 | 16 | 1.00337 | 0.98855-1.02560 | +0.337 | 10/6/0 | 0.894943 | 1.000000 | no_detected_regression |
| qwen3.5:decode:1 | 16 | 0.99510 | 0.98927-1.00537 | -0.490 | 7/9/0 | 0.401810 | 1.000000 | no_detected_regression |
| deepseek:scalar:1 | 16 | 1.00372 | 0.99801-1.01350 | +0.372 | 11/5/0 | 0.961594 | 1.000000 | no_detected_regression |
| deepseek:batch:1 | 16 | 1.00496 | 0.99986-1.01178 | +0.496 | 12/4/0 | 0.989365 | 1.000000 | significant_improvement |
| deepseek:batch:8 | 16 | 0.99845 | 0.97599-1.00830 | -0.155 | 8/8/0 | 0.598190 | 1.000000 | no_detected_regression |
| deepseek:decode:1 | 16 | 0.99707 | 0.98901-1.01157 | -0.293 | 7/9/0 | 0.401810 | 1.000000 | no_detected_regression |
| bert:scalar:1 | 16 | 1.00821 | 1.00666-1.01366 | +0.821 | 15/1/0 | 0.999985 | 1.000000 | significant_improvement |
| bert:batch:1 | 16 | 1.00181 | 0.97876-1.01528 | +0.181 | 8/8/0 | 0.598190 | 1.000000 | no_detected_regression |
| bert:batch:8 | 16 | 1.00380 | 0.96922-1.08574 | +0.380 | 9/7/0 | 0.772751 | 1.000000 | inconclusive |

Other medians are roughly ±1.5% or less except broader batch-8 spreads. No
statistically reliable familywise regression was detected, but the strict
no-regression criterion is **not fully certified**; there is no waiver and no
0% proof. Evidence: `/tmp/opencode/tokie-compat/python-paired-quiet-20261111/summary.json`
sha256 `ca5536f5ac713f1ac9ff14cfd8cbd76161072600cbcc8704ce802c81617f1483`;
`summary.md` `34926295972c106906067f0785d4c129663c002bd34a7e3e7fb3dec5259f0db1`;
`plan.json` `ff0916117c436960c61014ab022006b39d3ab5a5a0919fc1b0248db22701ff92`.
Raw artifacts are under the same directory.

### Direct AFTER vs gigatoken (descriptive)

Separate descriptive run on the same six BPE tokenizer JSONs, not the paired
gate. Warm tokie wins **12/18**,
median ratio **1.122** (range 0.889–1.474); cold gigatoken wins **16/18**,
median ratio **0.440** (range 0.239–4.025). GPT-2: warm single **579.7 vs
515.1 MB/s**, cold single **167.6 vs 246.2 MB/s**. Every case's token count and
checksum matched. BERT is a before/after comparison only because gigatoken has
no WordPiece. Evidence: `/tmp/opencode/tokie-compat/results/after/summary.json`
sha256 `9c2043880736fcda2b4fa9289f3213bfe6994390c0e15c9696b10ee39b54fc71`.

## Environment

- AMD Ryzen AI 9 HX 370, 24 logical CPUs; shared desktop host.
- Load around timing: 2.7 before the paired run, 3.4 after direct runs; 3.73
  before the two-row confirmation and 3.26 after it.
- AC status: charging, battery 87% at primary timing, 92% at confirmation.
- `powerprofilesctl performance` during timing; restored to `balanced` after.
- No Cargo, maturin, pytest, or Python perf process was running at preflight.
  A light `python3 tools/verify.py` (~1.3% CPU) was present before the
  confirmation; no heavy worker was present. Sensor temperatures were
  unavailable on this host.
- The before/after paired run was interleaved on the same current host state;
  direct historical baseline rates are descriptive, not the gate.

## Commands

Full paired gate, 36 rows, 7 pairs:

```bash
python3 scripts/compat_bench/run_paired.py \
  --before-bin /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected \
  --after-bin  /tmp/opencode/tokie-compat/bin/compat-bench-after \
  --before-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json \
  --after-meta  /tmp/opencode/tokie-compat/baseline_meta_after.json \
  --out /tmp/opencode/tokie-compat/paired/final-rust \
  --label final-rust
```

Precautionary focused confirmation on the only large-magnitude borderline row:

```bash
python3 scripts/compat_bench/run_paired.py \
  --before-bin /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected \
  --after-bin  /tmp/opencode/tokie-compat/bin/compat-bench-after \
  --before-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json \
  --after-meta  /tmp/opencode/tokie-compat/baseline_meta_after.json \
  --out /tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek \
  --label final-rust-confirm-deepseek \
  --focus warm:deepseek:batch:8 --pairs 9
```

Combined 7+9 evidence:

```bash
python3 scripts/compat_bench/combine_paired.py \
  --initial /tmp/opencode/tokie-compat/paired/final-rust/summary.json \
  --confirm /tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/summary.json \
  --out /tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/combined_summary.json
```

Requested one-shot 9-pair confirmation for the other two negative-CI rows:

```bash
python3 scripts/compat_bench/run_paired.py \
  --before-bin /tmp/opencode/tokie-compat/bin/compat-bench-base-corrected \
  --after-bin  /tmp/opencode/tokie-compat/bin/compat-bench-after \
  --before-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json \
  --after-meta  /tmp/opencode/tokie-compat/baseline_meta_after.json \
  --out /tmp/opencode/tokie-compat/paired/final-rust-confirm-two \
  --label final-rust-confirm-two \
  --focus warm:qwen2:batch:1,cold:qwen3.5:single:1 --pairs 9

python3 scripts/compat_bench/combine_paired.py \
  --initial /tmp/opencode/tokie-compat/paired/final-rust/summary.json \
  --confirm /tmp/opencode/tokie-compat/paired/final-rust-confirm-two/summary.json \
  --out /tmp/opencode/tokie-compat/paired/final-rust-confirm-two/combined_summary.json
```

Direct AFTER vs gigatoken, same six BPE types, 36 warm/cold rows:

```bash
scripts/compat_bench/run_baseline.sh after
```

## Paired before/after gate (historical/exploratory)

- 36 configs: six BPE models, single-1, batch-1, batch-8, warm and cold.
- 7 pairs per config, strict policy: any one-sided paired sign-test slowdown at
  `p_slower < 0.05` fails at any magnitude. No 5% budget.
- Gate: **PASS**.
- `failed_configs: []`; all 36 verdicts `no_measurable_change`.
- All token counts and ID checksums matched.
- `p_slower` range: 0.0625 to 0.9922. No row reached significance.
- Source freeze: full source hash pass.

Notable non-significant rows (reported, not gate failures):

| config | median speedup | 95% CI | p(slower) | slower pairs |
| --- | ---: | ---: | ---: | ---: |
| warm qwen2 batch 1 | 0.967 | 0.925–0.992 | 0.0625 | 6/7 |
| cold qwen3.5 single 1 | 0.982 | 0.969–0.995 | 0.0625 | 6/7 |
| warm deepseek batch 8 | 0.873 | 0.818–0.927 | 0.0625 | 6/7 |

The deepseek row was large enough to check despite not crossing alpha. The
precautionary 9-pair confirmation gave median speedup **1.004**, 95% CI
**0.987–1.065**, `p_slower = 0.746`, verdict `no_measurable_change`. Combined
7+9 (16 pairs): median **0.990**, 95% CI **0.873–1.026**,
`p_slower = 0.227`, `no_measurable_change`. This was one focused confirmation,
not repeated retries.

The other two negative-CI rows got one planned 9-pair confirmation each,
sequentially with the same binaries, source metadata, corpus, affinity, and
performance profile:

| config | initial 7 pairs | confirm 9 pairs | combined 16 pairs | conclusion |
| --- | --- | --- | --- | --- |
| warm qwen2 batch-1 | 0.967, CI 0.925–0.992, p 0.0625 | 0.989, CI 0.923–1.019, p 0.254 | 0.9800, CI 0.9365–0.9985, p 0.0384, 12/16 slower | **descriptive/post-selection; not confirmed** |
| cold qwen3.5 single-1 | 0.982, CI 0.969–0.995, p 0.0625 | 1.006, CI 0.902–1.103, p 0.746 | **0.9917, CI 0.9685–1.0159**, p 0.2273, 10/16 slower | signal clears, `no_measurable_change` |

The old pooled 7+9 Qwen2 combined CI is below 1 and its sign test crosses
`alpha=0.05`, but the design reused the same initial 7 samples and was not a
fresh confirmation. The fresh 9-pair run did not replicate it (`p=0.254`).
Therefore this is **descriptive/post-selection**, not a confirmed regression;
it is not pooled with the fresh64 data.

Raw 16-pair order (ms, initial 7 then confirmation 9):

```text
warm:qwen2:batch:1
before [44.667,44.663,45.990,45.238,46.104,46.178,44.352,44.089,46.390,43.589,43.783,43.580,43.954,44.834,46.409,43.144]
after  [45.378,47.690,50.673,45.607,43.835,47.760,47.967,43.270,50.244,44.074,43.722,44.663,44.020,47.508,44.010,46.978]

cold:qwen3.5:single:1
before [117.531,119.333,120.202,123.124,119.869,117.684,124.890,127.430,122.133,114.135,113.027,136.469,113.060,114.308,125.369,126.383]
after  [121.347,119.874,124.728,125.741,121.326,119.887,117.055,141.273,149.804,118.329,113.269,116.077,112.365,112.517,113.682,122.682]
```

### Historical conclusion (superseded by the quiet-window result)

Primary 7-pair gate: PASS. Requested confirmation evidence:

- warm deepseek batch-8: resolved, `no_measurable_change`
- cold Qwen3.5 single: resolved, `no_measurable_change`
- warm Qwen2 batch-1: old 7+9 post-selection combined was below 1 with
  `p_slower=0.0384`, but fresh 9 pairs gave `p=0.254`; **not confirmed**

The earlier `controlled-fresh64` attempt was inconclusive (corrected A/A CI
0.972666–1.020168, outside the 0.98–1.02 containment; A/B not run). The new
quiet-window `controlled-quiet-20261111` experiment passed A/A and returned
`no_slowdown_detected_at_reported_resolution` for A/B: median ratio 1.000438,
95% CI 0.998352–1.005683, one-sided upper 1.005576, p_slower 0.266154.

Rust readiness is **resolved at finite precision**, not proven equivalent or
zero-change. Python paired work is reported in the final handoff. No further
acceptance retries were run.

GPT-2 paired examples:

| config | before MB/s | after MB/s | speedup | p(slower) |
| --- | ---: | ---: | ---: | ---: |
| warm single | 605.7 | 609.7 | 1.004 | 0.773 |
| cold single | 158.6 | 160.4 | 1.033 | 0.773 |
| warm batch 8 | 2309.6 | 2326.7 | 1.004 | 0.773 |
| cold batch 8 | 192.8 | 188.2 | 0.980 | 0.227 |

## Direct AFTER vs gigatoken (descriptive, historical section)

Descriptive, separate run on the same six BPE tokenizer JSONs; not the
before/after gate.

Throughput `MB/s` uses `10^6` bytes. Ratio is gigatoken time / tokie time;
greater than 1 means tokie faster.

- Warm: tokie faster in **12/18** configs, median ratio **1.122**, range
  **0.889–1.474**; gigatoken faster in 6/18.
- Cold: gigatoken faster in **16/18**, median ratio **0.440**, range
  **0.239–4.025**. Tokie is faster only on Qwen 3.5 cold batch:
  1.815x at 1 thread and 4.025x at 8 threads.
- GPT-2 example:
  - warm single: tokie **579.7** vs gigatoken **515.1**
  - warm batch 8: tokie **2368.2** vs gigatoken **2082.1**
  - cold single: tokie **167.6** vs gigatoken **246.2**
  - cold batch 8: tokie **208.7** vs gigatoken **553.6**

The direct warm tokie-win count is lower than the earlier descriptive baseline
(12/18 vs 17/18), but this is a separate run on the current host and is not a
paired regression result. The primary 7-pair gate found no significant row; the
old 7+9 Qwen2 post-selection combined was not confirmed by fresh data. Cold
first-pass behavior remains the same story: gigatoken wins most
configurations, with Qwen 3.5 batch as the exception.

## Artifacts

Paired raw data and log:

- `/tmp/opencode/tokie-compat/paired/final-rust/` — raw before/after JSON,
  `summary.json`
- `/tmp/opencode/tokie-compat/paired/final-rust.stdout.log`
- `/tmp/opencode/tokie-compat/paired/final-rust.stderr.log`
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/`
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/combined_summary.json`
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-two/` — raw 9-pair
  confirmation for Qwen2 batch-1 and Qwen3.5 cold single
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-two/combined_summary.json`
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-two.stdout.log`
- `/tmp/opencode/tokie-compat/paired/final-rust-confirm-*/combined_summary.original-derived-units-bug.json`
  — preserved pre-fix derived units; timings/checksums/sign tests unchanged
- `/tmp/opencode/tokie-compat/units_fix_verification.txt` — units-only assertion log
- `/tmp/opencode/tokie-compat/controlled-fresh64/` — predeclared plan, 128
  A/A raw samples, corrected phase/summary/coverage, original CI-bug copies,
  `ci_correction_verification.txt`, `test_ci_selection.out`, code and logs
- `/tmp/opencode/tokie-compat/controlled-quiet-20261111/` — new predeclared
  plan, 128 A/A + 128 A/B raw samples, phase/summary/coverage, code and logs

Direct AFTER vs gigatoken:

- `/tmp/opencode/tokie-compat/results/after/` — raw JSON and `summary.json`
- `/tmp/opencode/tokie-compat/results_after_final.log`

Hashes:

```text
paired summary        1fd7b9e3e952b1e894e9b352523618cc5ca13f2ee5a6432b7b81ef46491dc3f0
deepseek confirm      133bbc408761067dc661dbd01fb0e96109a5019e16de25672a9f5a10420b46b7
deepseek combined     e43ea316f12dc139ec50e85cd56d96d915055fe9641029ee22b5e3f91eef09e2
two-row confirm       cf1531b83c6966040b7556a107552ccf3c15c99d71dddb540bd8470ab1596475
two-row combined      797684dc4a136609632df3fa66253efc7387b77c7db572a751a074ee8c30fa81
confirm-two stdout    05a7d474ca6156a3c577985fead597945cb4a8a667a5be0d735f78f327c31981
direct after summary  9c2043880736fcda2b4fa9289f3213bfe6994390c0e15c9696b10ee39b54fc71
paired stdout log    f3dcb4f21f541e6120566aebe8aae3113483a68de270c02832f6f207d0b723db
paired stderr log    4787fb7248f5be74e4b233f3eebaec8942e096dc3229e3a6d441aed526812eed
```

## Derived units correction (historical)

`combine_paired.py` initially used a placeholder input size of 1 byte, so the
combined derived `before_mb_per_s` / `after_mb_per_s` fields were invalid
(~2e-05). The timing arrays, checksums, speedups, sign tests, and verdicts were
valid and are unchanged.

Corrected derived values use the actual corpus input:
**20,974,828** bytes for batch and **20,979,014** for single.

- deepseek combined: 1898.4 / 1908.1 MB/s
- Qwen2 batch-1 combined: 469.6 / 461.1 MB/s
- Qwen3.5 cold single combined: 174.8 / 175.0 MB/s

Original bugged derived files are preserved as
`combined_summary.original-derived-units-bug.json`:
deepseek `92335588b6d3f3cd85b76073477aa5135c24ec84b5095b120920c70440814213`,
two-row `daf9b05340b3e130856827019e7bfeb44c647f01c2ee06737dd8785cdd4bfe0d`.
The active corrected `combined_summary.json` files are linked above. The
assertion log is `/tmp/opencode/tokie-compat/units_fix_verification.txt`.

## Controlled fresh64 calibration (historical/exploratory)

Predeclared plan written before any sample:

```text
/tmp/opencode/tokie-compat/controlled-fresh64/plan.json
sha256 f6625c80eebef08a2cf5a4a9d3325bf0df9000f76770a7c48334b7ddff7df550
created 2026-11-10T18:30:24-0800
```

Protocol actually run: A/A first, 64 pairs, both sides byte-identical before
binary `3a865...`, 32 A-first / 32 B-first, CPU 8, threads 1, warmup 6, reps 5,
outer `setarch -R` (`personality=0x40000`), matched-length aliases and labels,
same corpus/model. All 128 samples kept. IDs/checksum matched:
`4644146` / `6ec969d6e4a01923`.

A/A result (corrected tightest exact CI):

- median ratio **0.997279** (speedup 1.002728)
- exact 95% median-ratio CI **0.972666–1.020168**, width **0.047502**,
  coverage **0.967234** (indices 24, 41)
- exact two-sided sign p **0.707981**; 30 slower / 34 faster / 0 ties
- one-sided 95% upper bound **1.019921** (index 40, coverage 0.9700)
- raw ratio range **0.710461–1.396181** (distinct from the CI)

The earlier `0.710461–1.396181` CI was an order-statistic selection bug:
it chose the minimum k/max-coverage interval instead of the tightest interval
with coverage >=95%. The corrected interval is **still outside the required
`[0.98,1.02]` containment** because low = 0.972666; A/A remains **FAILED**.
The real noise interval is about **−2.73% to +2.02%**, not 68.6%. Per protocol,
**A/B was not run** and no retry was attempted. This is a host-noise/instrument
resolution stop, not a code regression finding.

`plan.json` is immutable. Its code hash refers to the original analyzer
`analyze_fresh64.original-ci-selection-bug.py` (`52cbde37…`). The corrected
derivative is `analyze_fresh64.corrected.py` (`df9bb3df…`); only CI index
selection changed. Raw/plan/sign/inference fields are unchanged, asserted in
`ci_correction_verification.txt`.

Exact CI selection tests (no encoded data):

```text
n=7  -> k=1,  coverage 0.9843750000000000
n=9  -> k=2,  coverage 0.9609375000000000
n=16 -> k=4,  coverage 0.9787292480468750
n=64 -> k=24, coverage 0.9672342409011768
PASS: tightest exact >=95% order-statistic indices
```

Environment during A/A: load 5.72 → 5.66, CPU 8 frequency 3.308 → 3.280 GHz,
SMT siblings `8,20`, performance profile, battery charging 97%;
`environment_actual.json` records AC details.

Raw and analysis:

```text
plan.json                               f6625c80eebef08a2cf5a4a9d3325bf0df9000f76770a7c48334b7ddff7df550
phase_AA.json                           2b72c265409219df00a1644a404c85990c5e92d42c42475359290e244ae9b9aa
summary_AA.json                         547925a4c464c85866db7450215b18fa82b7109a942f3b80fd17ba010b7ccf9c
coverage_AA.json                        625ceadc57decf85995083b8dde3d759812c8217bb6139dcc2aa40eb56339113
summary_AA.original-ci-selection-bug     f0756080472ef8e23c88ed49e9b1671518546a45ab5521b2ef3cf24abdb543a0
coverage_AA.original-ci-selection-bug   9c3eb7f7f725c58b95f817e0ad6f822006853dd467167a64ce030753e89b22ff
analyze_fresh64.corrected.py            df9bb3df9d8b6589c6d08f3bcd0c0838c1d330b5ed05d65c4394fceb4860d5de
test_ci_selection.py                    eddd069e394397cc3e203860829b987eabc2a5d65be954f035bc56af2fb25bfe
raw/AA/*.json                           128 files (unchanged)
```

## Quiet-window Rust paired result (final; finite resolution)

This is a new, predeclared, quiet-window experiment. Old `controlled-fresh64`
raw/plan/derivatives were not reused or modified.

```text
plan.json  fc8a1502f1c460a67126ce790e0ff8e066f431c278b8a32e1801290c3ffcecd8
created    2026-11-11T11:50:13-0800
analyzer   analyze_fresh64.corrected.py df9bb3df9d8b6589c6d08f3bcd0c0838c1d330b5ed05d65c4394fceb4860d5de
AA raw     128 files
AB raw     128 files
```

Protocol: A/A first with both sides the same `3a865...` before binary; exactly
64 pairs, 32 A-first / 32 B-first; matched-length aliases and labels; outer
`setarch -R` (`personality=0x40000`); CPU 8, threads 1, warmup 6, reps 5.
A/A passed before A/B was allowed. A/B then used before `3a865...` vs after
`3007...`, 64 fresh pairs, same protocol. No old samples were pooled. All
IDs/checksums matched: `4644146` / `6ec969d6e4a01923`.

A/A calibration (passed):

- median ratio **0.997734** (speedup 1.002271)
- exact tightest 95% CI **0.989654–1.002035**, width **0.012381**,
  coverage 0.967234 (indices 24, 41)
- exact two-sided sign p **0.168643**; 26 slower / 38 faster / 0 ties
- one-sided 95% upper bound 1.001716 (index 40, coverage 0.970029)
- CI contains 1, is inside `[0.98,1.02]`, and two-sided p >= .05 → pass

A/B result:

- median ratio **1.000438** (speedup 0.999562)
- exact tightest 95% CI **0.998352–1.005683**, width **0.007331** (0.73%),
  coverage 0.967234
- exact one-sided slowdown p **0.266154**; 35 slower / 29 faster / 0 ties
- one-sided exact 95% upper bound **1.005576** (0.56% upper bound)
- CI contains 1 and width <= 0.04, p >= .05 → verdict
  **`no_slowdown_detected_at_reported_resolution`**

This resolves the Qwen2 suspicion at the predeclared finite precision, not as
proof of zero change. The 0.73%-wide CI and 0.56% one-sided upper bound are the
achieved precision; individual process outliers remain (raw ratio ranges
A/A 0.939556–1.050431 and A/B 0.943374–1.076450), which is why the median
order-statistic sign protocol was predeclared.

Environment: load around 0.53–2.14; CPU 8/SMT sibling `8,20`; performance
profile during timing, `balanced` after; BAT1 100% not charging with supply
statuses in `environment_actual.json`.

Artifacts:

```text
phase_AA.json   538e11c9cb36a4b08ca7b5a684ca05330211822cd95df33b4baab4d551015dd6
summary_AA.json e510eb4c4acbf78442f26c9e43ac3d1bfbe0be7aec3a35c46ddff2d56ed05112
coverage_AA.json a1dd73abedfc419e2f7e890634fdbfa9197edce86d813cf99f94be8e151e7586
phase_AB.json   0bed344535ae2d3e90177e8e431e36ff51d4d1831d5b85d451402a4116d62a54
summary_AB.json 88ae8c6b7458f2773dc444032e070b0ad03104c83648f63dee9af5369840f94c
coverage_AB.json c5232af8a689056494b4153f771ccf3d1db0626a98c3e6dac00683fb1fd958a4
```

## Qwen2 batch-1 candidate diagnosis (historical)

Binary/codegen evidence:

- Both ELF binaries contain **2079 symbols; the symbol set is identical**.
- Exactly one function changed size: `tokie::norm::Norm::write`,
  `0x5c4b -> 0x5c5b` (**+16 bytes**, 5749 -> 5752 instructions). The normalized
  disassembly diff is three extra switch-range-check instructions caused by the
  new Bert kana cases.
- `Tokenizer::batch` is the same size (`0x2afd`) and its normalized disassembly
  diff is **zero instructions**. It moved `0x49bfa0 -> 0x49bfb0`:
  32-byte-aligned -> 16-byte-misaligned.
- Hot functions `emit_fast`, `pack_bounds`, `bounds_avx512<Voyage>`, `recover`,
  and `merge_small` have identical addresses and sizes.
- rustc/cargo, linker, LTO/codegen flags, and the AVX-512 symbol count are the
  same. Qwen2's tokenizer normalizer is NFC, not Bert.
- The new public API impls are dead-code eliminated: no new symbols and no
  code-size contribution. Moving them to a child module cannot by itself
  restore the `Norm::write`/`batch` layout; the layout shift comes from the
  required normalie Bert size change.

Profile/counter evidence (diagnostic, not acceptance timing):

- Qwen2 batch-1, warmup 6, 100 reps: instructions essentially identical
  (~59.2B). The hot-function profile percentages are unchanged:
  `emit_fast` ~27%, `pack_bounds` ~12%, `bounds` ~12%, `recover` ~8-9%,
  `nfc_yes_avx512` ~2.85%. `Norm::write` is not a separate hot sample.
- Cycles, L1-icache loads/misses, cache misses, and front-end stalls varied
  between runs and correlated with host load; the after/before cycle ratio
  ranged from slightly lower to ~12-16% higher.
- A same-binary null control on Qwen2 batch-1 (9 pairs) ran while the host load
  was 7.27 with multiple headless Chrome renderers: median speedup **1.169**,
  CI **1.010-1.188**, `p_slower=0.998`. A same-binary comparison producing a
  16.9% "improvement" proves the current host cannot resolve a 2% effect.
- Therefore the 2% Qwen2 result is **not proven deterministic code slowdown**;
  it is also not proven absent. The only deterministic code difference is the
  16-byte layout shift/alignment change.

Hypothesis: the required Bert fix grew the shared `Norm::write` function by 16
bytes, shifting `batch` off 32-byte alignment; that layout change is a plausible
microarchitectural contributor, but the current host noise is larger than the
effect. The API-module candidate does not address this code-size source.

Direct test for the architect/next phase, not run:

1. Re-run before/after and a same-binary null on an idle host with the SMT
   sibling idle or isolated, performance profile, and `setarch -R` as an ASLR
   control.
2. Build an experimental layout-only binary that restores `batch` alignment
   (or isolates Bert normalizer codegen) and compare it to the current after
   binary. If the 2% moves with alignment, the layout hypothesis is supported.
3. Keep the acceptance policy unchanged; do not accept the 2% without a clean
   noise floor.

Raw diagnosis artifacts: `/tmp/opencode/tokie-compat/perf-regression/` —
`SUMMARY.md`, `nm/`, `disasm/*.normalized.diff`, `perf/stat2_*.txt`,
`perf/stat_interleaved.txt`, `perf/report2_*.txt`, `null-qwen2/`.

## Caveats

- Shared host with load 2.7–4.0 for acceptance, 5.7 for the earlier
  controlled-fresh64 A/A, 7.27 for the null diagnostic, and 0.53–2.14 for the
  quiet-window experiment; no cgroup isolation. Individual process outliers
  remain, so medians/order statistics are the reported quantities.
- The strict sign test is intentionally conservative: with 7 pairs it needs
  7/7 slower for `p < 0.05`. Deepseek and Qwen3.5 cold single cleared. The old
  Qwen2 7+9 pooled result is post-selection and the fresh 9-pair run did not
  replicate it (`p=0.254`); it is not a confirmed regression.
- The earlier controlled-fresh64 A/A failed (corrected CI outside the target
  containment), so that phase had no A/B. The new quiet-window A/A passed and
  A/B returned no slowdown at the predeclared finite resolution; this is not
  proof of zero change or equivalence.
- `no_measurable_change` is not proof of zero change.
- Direct AFTER-vs-gigatoken numbers are descriptive and were run separately
  from the paired gate.
- Profiling was repeated only for diagnosis; it is not acceptance timing. The
  limiter functions are the same as the recorded baseline: pretoken boundary
  walk, cache probe/pack, merge/emit, and first-pass cache construction.
- Python performance is reported in the final handoff; the strict no-regression
  sign-off is not fully certified. Rust performance work is stopped and the
  power profile is restored to `balanced`; the quiet-window Rust result is a
  finite-resolution no-slowdown result.

## Python (final)

The final 27-config Python matrix is in the [Final handoff](#final-handoff-2026-11-11). Python performance measurement is complete; no further Python runs are planned.
