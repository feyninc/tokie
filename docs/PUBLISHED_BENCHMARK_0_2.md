# Published 0.2.0 benchmark: tokie vs GigaToken

Status: final for the recorded canonical, independent, and supplemental runs
below. These are measurements from public PyPI wheels, not a universal
fastest-tokenizer claim and not a release sign-off.

## Scope

Three BPE tokenizers with a shared `tokenizer.json` were timed on 179
OpenWebText documents (1,057,339 UTF-8 bytes, 1.008 MiB): GPT-2, Qwen2 7B,
and cl100k. Single-thread encode, decode, and batch were measured for tokie
0.2.0 and GigaToken 0.11.0. A separate eight-thread batch run used that small
sample and a larger 3,411-document sample (16,924,074 bytes, 16.14 MiB;
16,777,020 chars). Warm-cache throughput is the main table; first-pass numbers
are reported separately because they are not cold-start or tokenizer-load time.

Correctness covered 459 texts for the small run (179 timing documents, 24 fixed
edge cases, 256 deterministic fuzz strings) and 3,691 for the large run (3,411
timing documents plus the same edge and fuzz sets). Edge and fuzz texts are not
timed.

Hugging Face `tokenizers` 0.23.3 was the third oracle and also ran in a separate
459-text throughput check. GigaToken 0.11.0 supports these three BPE models;
WordPiece and Unigram were not run through it, so this document makes no claim
about BERT or T5 versus GigaToken.

## Provenance

- **tokie 0.2.0** PyPI wheel `tokie-0.2.0-cp312-cp312-manylinux2014_x86_64…whl`, sha256 `5c28027c24eb418fb9bcc9733c31ad4a6956d6129685b101e996e024669b6df7`.
- **GigaToken 0.11.0** PyPI wheel `gigatoken-0.11.0-cp310-abi3-manylinux_2_17_x86_64…whl`, sha256 `11992abc7f41e41e5be03c722df1c5205b5d0fcc4bcbd4fb440d2a0b97f7ac3f`.
- **Hugging Face tokenizers 0.23.3**, sha256 `376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87`; NumPy 2.5.4, Awkward 2.14.0, Python 3.12.13, Linux, clean published-package environment, separate process per thread configuration. Public wheels only: no local checkout and no path install.
- Host: AMD Ryzen AI 9 HX 370 with Radeon 890M, 24 logical CPUs, no taskset; governor and CPU frequency were not recorded. Load average before/after the 1-thread run: `2.40/2.01/1.52` to `2.23/1.99/1.52`.
- Corpus: `Skylion007/openwebtext`, `plain_text/train-00000-of-00080.parquet`, parquet sha256 `caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e`. The JSONL holds 4,187 full documents at stride 15 from that first shard (21,281,368 bytes, sha256 `20dc6601bee15f270b7a2c35cfde6cd2f073f8c15567c76b913453157d2c67db`; manifest sha256 `227927ab4de6324d427520ad496f4fd55808e27a0513d44d26fb4a17088b03a0`). Seed `20261111` selected 179 documents for the small sample (1,048,248 chars, 1,057,339 bytes; rows 135–62,670; row-list sha256 `c9f3e026003bae02bf8281be01c8ee42a00a9d87be0626c9d471b17fae0236f4`) and 3,411 documents for the large sample (16,777,020 chars, 16,924,074 bytes; rows 0–62,790; row-list sha256 `5995a89cd85810c43e86c0a5abaabf6a051c8ed4f2903e7cfc684ae5b8638fe9`). This is a stride-15 prefix sample from one shard, not the whole dataset; absolute MiB/s depends on text mix. Source JSON versions: `results-1t.json` and `results-8t.json` with script sha256 `55339f11ac66efcb947458c855136a842350e4cc4dac9d77a23d922970494af0`; `results-8t-large.json` and `independent-results-1t.json` with `0cfc27393e349087d8e8c3cd5087ddbcf47fea63fd87413619072a7f0bdca3b8`; `results-decode-inputs.json` with `3d85b3f125b72ee52b902a6b00e704ff9dc5269320e8425beb1a6b5d3f5b64dc`. Older and current scripts differ in validation metadata and HF guard requirements, and the latest added decode/list operation code; sources for older hashes are not archived. This is versioned data only, not a claim that timing code is unchanged across all runs. Older files carry zero HF mismatch counters but no `hf_ok` field.

Raw evidence is included as [results-1t.json](benchmarks/published-0.2/results-1t.json), [independent-results-1t.json](benchmarks/published-0.2/independent-results-1t.json), [results-decode-inputs.json](benchmarks/published-0.2/results-decode-inputs.json), [results-8t.json](benchmarks/published-0.2/results-8t.json), [results-8t-large.json](benchmarks/published-0.2/results-8t-large.json), and [hf-results.json](benchmarks/published-0.2/hf-results.json). The GigaToken JSON files hold per-round times, cold passes, profiles, wheel hashes, import paths, model paths, and corpus metadata; the HF JSON holds the corresponding HF run.

`hf-results.json` has its own producer: [hf-producer.py](benchmarks/published-0.2/hf-producer.py), sha256 `649f6cc1af31dc278ca831d1fcfbb0807599e62654ca5973025d588b4a0d01c6`, a byte-for-byte archive of the historical HF harness. The script takes no flags, so that hash also pins the invocation. It is not `scripts/bench_published.py`, which produced the GigaToken JSON. [hf-producer-command.md](benchmarks/published-0.2/hf-producer-command.md) records the exact invocation, the Python 3.12.13 / HF 0.23.3 / NumPy 2.5.4 environment, and why the original absolute corpus and HF cache paths are preserved.

Model paths, `tokenizer.json` sha256 values, byte sizes, and vocab sizes are recorded in each JSON.

## Correctness

All three GigaToken models had zero encode-id mismatches, zero decode-byte
mismatches, and zero batch flat/ends mismatches on both samples; the HF
tri-oracle also matched ids, decode bytes, and batch offsets. The 459-text small
run had 253,419 gpt2 tokens, 239,817 qwen2, and 236,578 cl100k; the 3,691-text
large run had 3,866,746, 3,753,894, and 3,662,379, with `hf_available`/`hf_ok`
true. Counts were asserted inside each timed pass; outputs were rechecked
outside the clock.

tokie `batch(texts)` returns `(uint32 flat_ids, uint64 ends)`. GigaToken `encode_batch(texts, parallel=...)` returns an Awkward list array whose offsets include a leading zero. Validation compares tokie `ends` with `offsets[1:]` and tokie ids with the flattened GigaToken content; no special-token config changed.

## Method

- Five alternating paired rounds: small 1-thread and 8-thread runs used 20 passes per round; the large 8-thread run used 5. Warm medians use these rounds; one unreported warmup precedes them.
- First-pass values are one observation per implementation, with tokie measured first; model load excluded.
- Ranges are min and max across the five rounds, not confidence intervals;
  medians are of the round values.
- `MiB/s` uses input UTF-8 bytes for encode and batch, and decoded UTF-8 bytes
  for decode. `Mtok/s` is token ids per second.
- `Paired elapsed-time ratio Giga/Tokie` is the median of per-round paired elapsed-time ratios (GigaToken seconds / tokie seconds), not the ratio of median throughputs; below 1 means GigaToken took less time. Throughput speedups in prose use ratios of median throughputs (for example, first-run decode 396.2 / 106.8 = 3.709, not the reciprocal of the paired 0.266).
- Both sides count their full public-API output inside the timed loop.
- `cProfile` measured `cpu_per_wall` of 0.98–1.00 for every single-thread loop, so neither side used hidden threads there; it cannot attribute native internals.

## First recorded run: one-thread warm results

The `decode` rows use NumPy uint32 input and are the older first recorded run; the focused container rerun is below.

| Model | Op | tokie MiB/s median [min–max] | GigaToken MiB/s median [min–max] | Paired elapsed-time ratio Giga/Tokie median [min,max] |
| --- | --- | ---: | ---: | ---: |
| gpt2 | encode | 791.2 [784.7–804.3] | 835.1 [803.3–848.7] | 0.959 [0.932,0.977] |
| gpt2 | decode | 106.8 [101.9–107.1] | 396.2 [391.6–402.3] | 0.266 [0.257,0.273] |
| gpt2 | batch | 848.6 [820.6–859.4] | 778.0 [636.6–866.5] | 1.099 [0.947,1.333] |
| qwen2 | encode | 642.8 [616.7–649.7] | 571.6 [556.9–576.2] | 1.128 [1.093,1.154] |
| qwen2 | decode | 112.8 [106.4–114.8] | 388.6 [375.5–403.3] | 0.285 [0.276,0.299] |
| qwen2 | batch | 667.9 [618.2–687.1] | 582.0 [576.3–596.1] | 1.129 [1.062,1.191] |
| cl100k | encode | 674.0 [662.5–678.5] | 751.9 [744.4–759.0] | 0.898 [0.873,0.905] |
| cl100k | decode | 115.6 [112.8–117.5] | 366.5 [354.9–407.1] | 0.311 [0.284,0.318] |
| cl100k | batch | 700.4 [692.5–708.0] | 738.9 [517.0–773.8] | 0.957 [0.895,1.352] |

In the first recorded run, encoding was mixed: GigaToken took 0.959× tokie's time on GPT-2, 0.898× on cl100k, and 1.128× on Qwen2. GigaToken decoded 3.2–3.7× faster on NumPy uint32 ids in that first run. Batch was mixed.

## Independent rerun

A separate 1-thread rerun used the same 179 documents, five rounds, 20 passes, then-current script `0cfc2739…`, and all 459 validation texts; encode, decode, batch, and HF mismatches were all zero and output checksums matched. Raw results: [independent-results-1t.json](benchmarks/published-0.2/independent-results-1t.json). Eight-thread batch and HF throughput were not independently retimed.

| Model | tokie encode MiB/s | GigaToken encode MiB/s |
| --- | ---: | ---: |
| gpt2 | 657.9 | 554.1 |
| qwen2 | 541.8 | 486.4 |
| cl100k | 572.5 | 648.0 |

The GPT-2 encode ordering reversed from the first recorded run (`791.2` tokie / `835.1` GigaToken), so there is no measurable reliable GPT-2 encode winner. The canonical run is internally valid, but its narrow GPT-2 encode win is not reproducible. Qwen2 and cl100k encode observations are workload- and run-dependent. On NumPy uint32 ids, decode remained robustly GigaToken-faster: the ratio of median decode throughputs was about 3.2–3.5×, versus 3.2–3.7× in the first run. Encoding comparisons are workload-dependent; this is not a formal statistical claim.

The rerun load averages were `2.38/3.69/2.74` before and `2.15/3.58/2.71` after. A verifier noted powersave CPU frequency of 2.81 GHz observed / 5.16 GHz max; neither governor nor frequency is in the JSON, and the canonical run did not record them. Uncontrolled shared-host CPU frequency and thermal state can vary; no causal claim is made.

## NumPy and list decoder paths

A focused 1-thread rerun decodes the same 179 documents with two explicit id containers: `decode` takes NumPy uint32 ids and `decode_list` takes Python lists; list conversion is outside the timed loop. Both return decoded bytes. Five rounds, 20 passes, all 459 validation texts, all id/decode/batch/HF mismatches zero, checksums match. Raw: [results-decode-inputs.json](benchmarks/published-0.2/results-decode-inputs.json).

| Model | Input | tokie MiB/s median [min–max] | GigaToken MiB/s median [min–max] |
| --- | --- | ---: | ---: |
| gpt2 | NumPy uint32 | 94.2 [87.8–94.7] | 333.8 [297.1–348.9] |
| qwen2 | NumPy uint32 | 96.4 [95.7–97.6] | 320.7 [253.1–338.7] |
| cl100k | NumPy uint32 | 97.1 [95.9–98.5] | 313.4 [266.1–342.0] |
| gpt2 | Python list | 210.0 [208.6–213.6] | 178.6 [174.2–184.2] |
| qwen2 | Python list | 202.2 [187.9–216.4] | 175.4 [139.8–182.8] |
| cl100k | Python list | 225.6 [212.6–233.8] | 184.4 [179.3–191.9] |

Decoding depends on the id container. GigaToken is 3.2–3.5× faster on NumPy uint32 ids by ratio of median throughputs; tokie is about 15–22% faster on Python lists in this run (1.15–1.22× by ratio of median throughputs). Paired time ratios can differ because they are medians of per-round ratios; for example, Qwen2 list is 1.218 paired but 1.153 by median throughputs. These include binding conversion costs; the native kernel is not isolated. These supplemental values (`94.2/333.8` GPT-2) are the canonical decode data for this report; a separate GPT-2-only rerun (`78/326`) is documentary verification only, not canonical, and is not used in the table or charts.

This supplemental run is a different host-frequency/thermal observation from the older canonical run. Before-validation snapshot: powersave governor, mean current frequency 1.08 GHz, min 0.61 GHz, max 2.79 GHz across 24 CPUs; the older run did not record frequency. This is an idle snapshot, not a timing-clock measurement. Absolute throughput varies between runs and could reflect CPU frequency, thermal state, or noise; this is not proved as the source. Do not read 94.2 vs the old 106.8 as an intrinsic kernel change.

## First inference pass (single observation)

Single observation per operation, tokie measured first, model loading excluded.
For batch, the first call is the full batch.

| Model | Op | tokie first call ms | GigaToken first call ms | tokie full pass ms | GigaToken full pass ms |
| --- | --- | ---: | ---: | ---: | ---: |
| gpt2 | encode | 0.80 | 0.31 | 4.15 | 3.69 |
| gpt2 | decode | 0.88 | 0.39 | 9.75 | 3.55 |
| gpt2 | batch | 3.78 | 13.19 | 3.78 | 13.19 |
| qwen2 | encode | 1.49 | 0.36 | 8.92 | 4.60 |
| qwen2 | decode | 4.19 | 0.37 | 12.88 | 3.80 |
| qwen2 | batch | 8.88 | 32.90 | 8.88 | 32.90 |
| cl100k | encode | 1.37 | 0.38 | 8.14 | 4.39 |
| cl100k | decode | 3.73 | 0.38 | 12.27 | 4.15 |
| cl100k | batch | 7.85 | 23.16 | 7.85 | 23.16 |

## First recorded run: eight-thread batch

Separate process with `RAYON_NUM_THREADS=8` set before import. Only batch was
measured; GigaToken used `parallel=True` and tokie used `threads(8)`. The small
sample is 1,057,339 bytes (1.008 MiB), 179 docs, 20 passes/round. The large
sample is 16,924,074 bytes (16.14 MiB), 3,411 docs, 5 passes/round. The large
result is one run of five rounds and was not independently retimed.

Warm-cache large-sample medians, with min–max across five rounds:

| Model | tokie MiB/s | GigaToken MiB/s | Giga/tokie time | cpu/wall tokie/Giga |
| --- | ---: | ---: | ---: | ---: |
| gpt2 | 1887.5 [1841.1–1966.6] | 820.4 [679.3–1120.0] | 2.391 [1.685,2.895] | 5.54/4.02 |
| qwen2 | 1516.2 [1437.0–1577.4] | 1435.3 [1227.4–1522.0] | 1.056 [0.950,1.208] | 5.54/5.16 |
| cl100k | 1572.0 [1538.3–1608.9] | 1596.1 [825.5–1847.5] | 0.968 [0.867,1.864] | 5.92/3.40 |

On the small sample the warm medians were 1465.8/841.7, 1245.2/580.3, and
1289.4/761.0 MiB/s for tokie/GigaToken, with GigaToken `cpu_per_wall` about
1.00. On the large sample GigaToken used 3.40–5.16 CPUs. GPT-2 favored tokie;
Qwen2 was near parity with overlapping ranges; cl100k is inconclusive because
GigaToken's 825.5–1847.5 range is much wider than the 0.968 median gap.

Single-observation first pass, tokie measured first, model load excluded; for batch this is one full call over all documents:

| Model | Sample | tokie ms | GigaToken ms |
| --- | --- | ---: | ---: |
| gpt2 | 1.008 MiB | 6.91 | 13.53 |
| qwen2 | 1.008 MiB | 8.87 | 32.02 |
| cl100k | 1.008 MiB | 9.43 | 23.60 |
| gpt2 | 16.14 MiB | 114.78 | 34.94 |
| qwen2 | 16.14 MiB | 295.31 | 78.19 |
| cl100k | 16.14 MiB | 207.50 | 53.56 |

These are raw single observations of one full call per implementation, not a
controlled comparison: tokie was measured first, so warm-up, allocator, and
CPU-frequency state are not balanced between the two sides. On the large
sample the GPT-2 call was 114.78 ms for tokie and 34.94 ms for GigaToken, Qwen2
295.31/78.19, and cl100k 207.50/53.56. The first pass is not tokenizer load
time, and it is separate from the warm-cache medians above.

## Hugging Face check

This run used the same 179 documents plus 24 edge and 256 fuzz texts, for 459
texts and 1,068,881 input bytes, so it is not directly comparable to the
179-document GigaToken table. It compares three different algorithms (BPE,
WordPiece, Unigram), one thread, warm cache, median of five alternating paired
rounds. The raw JSON has per-round values and the cold first pass.

| Model | Algorithm | Texts | tokie encode MiB/s | HF encode MiB/s | Encode speedup | tokie decode Mtok/s | HF decode Mtok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 | BPE | 459 | 556.1 [554.1–589.7] | 4.73 [4.67–4.80] | 117.5× | 47.3 | 10.1 |
| bert-base-uncased | WordPiece | 459 | 255.5 [250.2–264.2] | 4.25 [4.10–4.31] | 60.2× | 53.8 | 2.2 |
| t5-small | Unigram | 459 | 120.2 [118.1–123.3] | 3.26 [3.21–3.37] | 36.8× | 42.9 | 9.0 |

HF's harness passed Python lists and returned Unicode strings; the GigaToken
decode benchmarks take either NumPy arrays or Python lists and return decoded
bytes. The HF check also includes 280 edge/fuzz texts. These are separate
workloads, so the decode columns are not comparable.

## Reproduce

Install the exact public packages in a fresh Python 3.12 environment: `python3.12 -m venv .venv && .venv/bin/pip install tokie==0.2.0 gigatoken==0.11.0 tokenizers==0.23.3 numpy==2.5.4 awkward==2.14.0`.

`scripts/bench_published.py` takes a JSONL corpus with one `{"row": <int>, "text": <str>}` object per line and one or more `LABEL=PATH` tokenizer JSON files; pinned wheel hashes from the recorded run are in the results JSON.

```bash
.venv/bin/python scripts/bench_published.py \
  --corpus owt_20m.jsonl \
  --models gpt2=/path/to/gpt2/tokenizer.json --models qwen2=/path/to/qwen2/tokenizer.json --models cl100k=/path/to/cl100k/tokenizer.json \
  --threads 1 --ops encode,decode,batch --rounds 5 --passes 20 \
  --output results-1t.json --wheels-dir wheels
```

For eight threads, set `RAYON_NUM_THREADS=8` before import; use `--threads 8 --ops batch --output results-8t.json`, and add `--max-chars 16777216 --max-docs 4187 --passes 5 --output results-8t-large.json` for the large sample. The independent rerun used the same 1-thread command with `--output independent-results-1t.json`. The decode-inputs run used `--ops decode,decode_list --output results-decode-inputs.json`; current `--ops` defaults to `encode,decode,decode_list,batch`, and HF is required unless `--allow-no-hf` is passed.

Charts: `python scripts/generate_charts.py --hf-results docs/benchmarks/published-0.2/hf-results.json --giga-results docs/benchmarks/published-0.2/results-1t.json --giga-results docs/benchmarks/published-0.2/results-8t.json --giga-large-results docs/benchmarks/published-0.2/results-8t-large.json --decode-results docs/benchmarks/published-0.2/results-decode-inputs.json --output-dir assets`. Chart regeneration also needs `matplotlib`; the plotting environment is separate from the benchmark venv.

The corpus is a stride-15 prefix sample of the first OpenWebText parquet shard, not the full dataset. The JSONL cannot be rebuilt byte-for-byte from the seed alone; the manifest hash, source parquet hash, row range, and tokenizer hashes above are the auditable pins. The HF check used `tokenizers.Tokenizer.from_file(path)` and `Tokenizer.open(path)` on the same 459 texts with `add_special_tokens=False`; metadata and timings are in `hf-results.json`.

## Limits

- One 179-document small slice and one 3,411-document large slice; neither is the whole corpus. Five rounds; min–max shows run-to-run spread, not statistical significance.
- Timings use public Python APIs including output construction, not an isolated kernel. GigaToken 0.11.0 covers three BPE models only; no WordPiece or Unigram comparison was run. The eight-thread result is batch-only; the large-sample warm result is size-specific. Numbers are from the published 0.11.0 wheel, not an older GigaToken tree.
