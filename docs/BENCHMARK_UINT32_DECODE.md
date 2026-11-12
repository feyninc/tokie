# Current warm benchmark: uint32 decode and shared BPE models
**Status:** final for the recorded data below. These are warm, end-to-end Python
API measurements on one host and two OpenWebText slices. They are not a universal
fastest-tokenizer claim, not a regression study, and not a release sign-off.

## Scope
The tokie series uses a **local unreleased 0.2.0 wheel**,
sha256 `8786f91b1aed3acbac9d8090234e55f0400f829da6ea8a2817bcd460332a1d85`.
The run records the source branch snapshot `bhavnick/uint32-only-decode` at
`d5674e4`; the artifact was built from the working tree before the benchmark-data
commits, so this is not a wheel-at-that-commit guarantee. Its installed metadata
says 0.2.0, but it is not the public PyPI wheel.
GigaToken 0.11.0 and Hugging Face `tokenizers` 0.23.3 are public wheels; this
report does not imply that PyPI has been updated. GPT-2, Qwen2 7B, and cl100k
share `tokenizer.json` files and ran through tokie, GigaToken, and HF. BERT-base-uncased (WordPiece) and T5-small (Unigram) ran
against HF only because this GigaToken build rejects those model types; that is
unsupported, not zero throughput. The new decode benchmark returns **Unicode
text**; the older published 0.2.0 decode benchmark returned bytes with a
different harness. Do not combine their numbers or ratios. The prior
published 0.2.0 report, JSON, and `assets/benchmark_published_*` figures remain
unchanged as the historical record for the public wheel.

## GPT-2 main comparison
This six-library GPT-2 run is the primary chart: single-text encode to token ids,
one thread, one host, five globally rotated rounds, 10 passes per round, and a
fresh child process per library per round. Import, model load, and three warmup
passes are outside the clock. The five-round rotation uses 5 of the 6 starting
positions, so it is partial rather than fully balanced; tiktoken's first-use
~4 MB HTTPS vocabulary download is also outside the timed region. The sample is
the same 179 OpenWebText documents (1,057,339 UTF-8 bytes), seed `20261111`; all
libraries ran on Python 3.12.13, with HF 1.0.0rc4 in a separate venv so
interpreters were not mixed.

| Library | Version | MiB/s median [min–max] |
| --- | --- | ---: |
| tokie | local unreleased 0.2.0, wheel `8786f91…` | 705.85 [656.89–709.57] |
| \*\* GigaToken | 0.11.0 | 744.44 [712.11–745.43] |
| HF tokenizers | 0.23.3 | 4.10 [4.04–4.18] |
| HF tokenizers | 1.0.0rc4 (prerelease; module `1.0.0-rc.4`) | 169.73 [138.37–171.96] |
| Kitoken | 0.11.1 | 33.35 [31.46–33.63] |
| tiktoken | 0.14.0 | 24.51 [24.19–24.81] |

\*\* GigaToken 0.11.0 supports BPE tokenizers only; WordPiece and Unigram are
refused (including BERT and T5) and excluded from this comparison.

Chart measures GPT-2 only, not a cross-family compatibility ranking.

All six matched HF 0.23.3 on the same GPT-2 `tokenizer.json` for all 459
validation texts (179 timing plus 24 edge and 256 fuzz), with zero id
mismatches. tiktoken also checked its canonical GPT-2 vocabulary: all 50,256
non-special ranks and the `<|endoftext|>` special matched, with zero mismatches.

Timing covers each library's full public single-text encode and id output. Tokie
and GigaToken return NumPy arrays; HF, Kitoken, and tiktoken return lists, so
binding/output costs differ and are included. This is not an isolated kernel
comparison. APIs: HF 0.23.3 uses `encode(...).ids`; HF 1.0.0rc4 compiles once
outside the clock with `Tokenizer.from_file(path).compile()`, then uses
`compiled.encode(text, add_special_tokens=False).ids` (no `encode_fast` API is
installed). Kitoken uses documented `encode_specials=True` to match HF; its
default would encode the literal as 7 regular ids. tiktoken passes
`allowed_special={'<|endoftext|>'}` explicitly; the default raises on that literal.
These arguments are recorded in the raw JSON.

GigaToken's 744.44 median is about 5.5% above tokie's 705.85 here; the JSON
records that the single-run round ranges do not overlap but the gap is close, so
no robust winner is claimed. Do not read the main chart as a tokie win. HF
1.0.0rc4 is about 41× HF 0.23.3 in this run; that is one prerelease observation,
not a general claim that 1.0 improves every model. The README headline's 190×
ceiling is supported by other current BPE rows (maximum median ratio 194.12×),
not by this GPT-2 row; tokie/HF 0.23.3 is about 172× here.

**Hero and detailed panels are separate experiments; do not combine their
medians.** The detailed GPT-2 encode median in the dashboard below is 681.2
MiB/s versus 705.85 here because host, warmup, process, and pipeline conditions
differ.

Raw data: [gpt2-hero.json](benchmarks/uint32-decode/gpt2-hero.json), sha256
`23c82bd7c1af00f2c7dfa348167dbb5c000b4eab09c879de5cc2fa679fb13c69`,
byte-unchanged after the recording commit. Recorded producer:
[scripts/bench_gpt2.py](../scripts/bench_gpt2.py), sha256
`f9de7e9804acca2d6275c28a348f21855dce699e4143f2dfd3d7cec1ec625a64`, archived
in commit `b03aa70a853108b3f5d2fc97486dd6d4572e9252`. Current
`scripts/bench_gpt2.py` sha256
`6b69850fa83c3ff0d3d843644f767519574ef755e718f1d5153b54e6417b47e4` adds
future-run HF-reference golden-checksum guards and writes `*.error.json` on
failure; it changes no recorded physical timing and does not rewrite the raw
JSON. Commands: [gpt2-hero-commands.txt](benchmarks/uint32-decode/gpt2-hero-commands.txt).
Generated table: [gpt2-hero-table.md](benchmarks/uint32-decode/gpt2-hero-table.md).

Figure/table command:
`python scripts/generate_gpt2_chart.py --results docs/benchmarks/uint32-decode/gpt2-hero.json --output-dir assets --table-out docs/benchmarks/uint32-decode/gpt2-hero-table.md`;
add `--check` to validate only and write nothing.

## Artifacts and provenance
| Item | Recorded value |
| --- | --- |
| tokie | local unreleased `tokie-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl`, 2,380,513 bytes, sha256 `8786f91b1aed3acbac9d8090234e55f0400f829da6ea8a2817bcd460332a1d85`; installed `.so` sha256 `f9e7abb0e3d5873f5f68bda5b998b6575982ab4637fc3aed9dec17f2c0e482f3`, matching the wheel member |
| tokie source | branch `bhavnick/uint32-only-decode`, recorded commit `d5674e47e92d2a3c6f00b2e2814c222c3f3423e7`; binding source sha256 `f6f0a4461eb5985cd80f569b803933b1f601f413d579746d44813ab1fca8d534`; no source changes (`source_changes: []`) |
| harness | recorded producer `scripts/bench_current.py` sha256 `ef47ef80782996ae20273ff0276556997705e8df17d107521e559f79238d8d59`, archived in commit `a04f4006cda1d435adbe6e76cac93453fb90045f`; current harness sha256 `dfbe15540efb2ef0a428180ff51038b156a72900d23c57d7d518fc654569e310` applies only a derived unit-label correction |
| GigaToken / HF | GigaToken 0.11.0 wheel sha256 `11992abc7f41e41e5be03c722df1c5205b5d0fcc4bcbd4fb440d2a0b97f7ac3f`; `tokenizers` 0.23.3 wheel sha256 `376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87` |
| Python stack | Python 3.12.13, NumPy 2.5.4, Awkward 2.14.0, Linux |
| results files after unit-label correction | `results-1t.json` `7e6b5547…`, `results-8t.json` `61142fe1…`, `results-hf-only.json` `2faa206e…` |

The recording JSON reports `source_changes: []`; tokenizer hashes and import
paths are in each JSON. Each file carries `meta.summary_correction`: producer
archive `a04f400`, corrected harness `dfbe1554…`, `measurements_changed: false`.
The correction removed token/s alias fields and renamed batch count units only;
seconds, counts, MiB/s, profiles, validation, and outputs are unchanged. The
original producer is readable with `git show a04f400:scripts/bench_current.py`.

## Method
Each library runs its public Python call over preloaded inputs. The clock includes
output construction and any adapter conversion. It does **not** isolate a native
kernel, and the libraries do not produce identical object types.

| Task | Input and timed operation | Output counted inside the clock |
| --- | --- | --- |
| encode | raw UTF-8 text -> ids | tokie `encode` uint32 ids; GigaToken uint32 ids; HF `Encoding.ids` materialization |
| Unicode decode | one shared contiguous NumPy `uint32` array per document -> `str` | tokie `decode`; GigaToken `decode(...).decode("utf-8", "replace")`; HF `decode(ids.tolist(), skip_special_tokens=False)` |
| batch | the same documents -> ragged ids | tokie flat uint32 ids plus uint64 ends; GigaToken Awkward array (not flattened inside the clock); HF list of `Encoding.ids` |

- Warm cache only; five rounds with rotated library order; one warmup pass.
  Encode/decode use 10 passes per round, batch one pass per round.
- Encode/decode sample: 179 OpenWebText documents, 1,048,248 chars, 1,057,339
  UTF-8 bytes (1.008 MiB), seed `20261111`. Batch: 3,411 documents, 16,777,020
  chars, 16,924,074 bytes (16.14 MiB), same seed; separate processes/imports.
- Decode uses decoded UTF-8 length; encode and batch use input UTF-8 bytes.
  Libraries perform the same logical task; public output construction and
  adapters differ. Ranges are min–max over five rounds, not confidence
  intervals. No cold-start measurement is included.
- Host: AMD Ryzen AI 9 HX 370, 24 logical CPUs, powersave. Mean frequency
  snapshots before validation were 1649.2 MHz (1-thread), 2304.3 MHz (8-thread),
  and 1547.2 MHz (HF-only); they are not timing-clock samples or causal evidence.

## Validation
Validation ran before timing and again through post-check checksums. All timed
models had zero encode-id and decode mismatches on 459 texts (179 timing
documents plus 24 edge and 256 fuzz). Shared BPE models also had zero batch
mismatches and matched the full 3,411-document batch flat ids and ends.

| Model | Algorithm | Texts | Encode | Decode | Batch |
| --- | --- | ---: | ---: | ---: | ---: |
| GPT-2 | BPE | 459 | 0 | 0 | 0 |
| Qwen2 7B | BPE | 459 | 0 | 0 | 0 |
| cl100k | BPE | 459 | 0 | 0 | 0 |
| BERT-base-uncased | WordPiece | 459 | 0 | 0 | not run |
| T5-small | Unigram | 459 | 0 | 0 | not run |

Shared-batch counts were 3,858,819 GPT-2 ids, 3,747,596 Qwen2 ids, and
3,655,531 cl100k ids; tokie/HF flat-id sha256 values matched per model.

## Results
Generated [`tables.md`](benchmarks/uint32-decode/tables.md) is authoritative if prose and rounding differ.

### Single-thread encode
MiB/s, median [min–max] over five rounds; 179 documents.

| Model | tokie | \*\* GigaToken | Hugging Face |
| --- | ---: | ---: | ---: |
| GPT-2 | 681.2 [659.3–707.2] | 708.3 [694.9–727.6] | 3.6 [3.4–3.6] |
| Qwen2 7B | 544.5 [533.0–565.9] | 488.5 [475.3–502.4] | 3.0 [2.9–3.1] |
| cl100k | 563.5 [427.1–582.3] | 638.9 [397.9–668.6] | 2.9 [2.9–3.2] |

GPT-2 is too close to call because the tokie and GigaToken ranges overlap. Qwen2
favors tokie within this recorded run, but this is one slice and one host. cl100k
is uncertain: both ranges are wide and overlap. HF is far slower here.

### Single-thread Unicode decode
MiB/s of decoded UTF-8 output, median [min–max]; same 179 documents and the same
contiguous uint32 ids per document.

| Model | tokie | \*\* GigaToken | Hugging Face |
| --- | ---: | ---: | ---: |
| GPT-2 | 309.3 [286.9–327.4] | 290.6 [274.6–299.3] | 30.3 [26.3–30.7] |
| Qwen2 7B | 334.4 [329.6–350.3] | 286.4 [267.7–307.1] | 30.6 [30.2–31.6] |
| cl100k | 312.7 [194.4–339.2] | 261.3 [220.7–276.8] | 28.0 [22.2–29.7] |

Models share input sample sizes; decode rates use each model’s decoded UTF-8
output size. Compare libraries within a model/task.

GPT-2 ranges overlap, so no winner is claimed. Qwen2's ranges are separated and
favor tokie in this run. cl100k has wide overlapping tokie/GigaToken ranges and
is uncertain. These include Unicode or list conversion, not pure kernel times.

### Batch
MiB/s of input text, median [min–max]; 3,411 documents. One- and eight-thread
runs used separate processes.

| Threads | Model | tokie | \*\* GigaToken | Hugging Face |
| ---: | --- | ---: | ---: | ---: |
| 1 | GPT-2 | 645.2 [570.2–658.4] | 649.8 [290.6–703.1] | 3.7 [3.7–3.9] |
| 1 | Qwen2 7B | 515.6 [424.1–566.5] | 442.6 [314.5–477.2] | 3.5 [3.3–3.5] |
| 1 | cl100k | 509.9 [382.2–573.4] | 510.4 [438.3–586.9] | 3.6 [3.3–3.8] |
| 8 | GPT-2 | 1765.7 [1532.0–2141.5] | 872.5 [673.3–992.2] | 18.1 [17.2–18.6] |
| 8 | Qwen2 7B | 1192.8 [1092.3–1778.2] | 828.7 [553.8–1135.6] | 16.4 [15.6–17.2] |
| 8 | cl100k | 1594.2 [1146.8–1704.8] | 1220.8 [605.0–2087.3] | 17.3 [15.2–17.6] |

Eight-thread GPT-2 is a strong tokie result in this run. Qwen2 has a small range
overlap and is uncertain, not a proven ordering. cl100k is noisy: GigaToken's
range is much wider than the median gap. One-thread rows are mixed and mostly
overlapping. Batch includes ragged-output construction.

### Hugging Face-only models
MiB/s, median [min–max] over five rounds; 179 documents, one thread. GigaToken's
recorded refusal is in `results-hf-only.json`.

| Model | Task | tokie | Hugging Face |
| --- | --- | ---: | ---: |
| BERT-base-uncased | encode | 265.7 [256.2–266.5] | 3.6 [3.5–3.7] |
| BERT-base-uncased | Unicode decode | 361.4 [355.0–373.8] | 7.7 [7.5–7.8] |
| T5-small | encode | 106.5 [64.6–109.9] | 2.8 [2.6–2.9] |
| T5-small | Unicode decode | 307.4 [284.9–315.0] | 25.0 [24.4–26.8] |

HF is much slower here. T5 encode has a wide tokie range, so its absolute value
is less stable.

## Figures, tables, and reproduction
`scripts/generate_benchmark_dashboard.py` renders `assets/benchmark_current_overview.png`/`.svg`
and `assets/benchmark_current_hf_only.png`/`.svg` from the three JSON files using
the faceted log-axis layout (shared-BPE 2x2; HF-only 1x2), and writes
`docs/benchmarks/uint32-decode/tables.md`.

Full argv, environment, and log paths: [`run-commands.txt`](benchmarks/uint32-decode/run-commands.txt).
Portable one-thread form:

```bash
"$VENV_PYTHON" scripts/bench_current.py --corpus "$CORPUS" \
  --model gpt2="$GPT2_JSON" --model qwen2="$QWEN2_JSON" --model cl100k="$CL100K_JSON" \
  --threads 1 --tasks encode,decode,batch --rounds 5 \
  --passes-encode 10 --passes-decode 10 --passes-batch 1 \
  --max-chars 1048576 --max-docs 256 --batch-max-chars 16777216 --batch-max-docs 4187 \
  --tokie-wheel "$TOKIE_WHEEL" --expected-tokie-wheel-sha256 "$EXPECTED" \
  --wheels-dir "$WHEELS_DIR" --output docs/benchmarks/uint32-decode/results-1t.json
```

Use `--threads 8 --tasks batch` for 8-thread, or `--hf-only-model
bert-base-uncased=... --hf-only-model t5-small=... --tasks encode,decode` for
HF-only. Set `RAYON_NUM_THREADS` and `TOKENIZERS_PARALLELISM` before native
imports. Validation-only dashboard command (writes nothing):

```bash
python scripts/generate_benchmark_dashboard.py \
  --results docs/benchmarks/uint32-decode/results-1t.json \
  --results docs/benchmarks/uint32-decode/results-8t.json \
  --results docs/benchmarks/uint32-decode/results-hf-only.json \
  --layout faceted --output-dir assets \
  --tables-out docs/benchmarks/uint32-decode/tables.md --check
```

Remove `--check` to render; test with `python scripts/test_benchmark_dashboard.py`.
The plotting environment needs `matplotlib`; the benchmark venv does not include it.

The corpus is a stride-15 prefix sample of OpenWebText shard 0; JSONL sha256
`20dc6601bee15f270b7a2c35cfde6cd2f073f8c15567c76b913453157d2c67db`, parquet
sha256 `caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e`.
Row-list and tokenizer hashes are in the JSON.

## Limits
- One host, one snapshot, one corpus family, two slices. Absolute MiB/s varies
  with CPU frequency, thermal state, load, text mix, and run.
- Five rounds; min–max is descriptive spread, not a confidence interval or
  significance test. No cold-start, load-time, memory, or regression study.
- End-to-end API costs differ by output type; "faster" means faster on this
  measured call, not proof about an isolated kernel. GigaToken covers three BPE
  models only; BERT and T5 are HF-only.
- The tokie wheel is local and unreleased; its 0.2.0 version string does not
  identify it as the public PyPI artifact.
