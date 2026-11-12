# Producer for `hf-results.json`

`hf-results.json` in this directory was produced by the script archived here as
[hf-producer.py](hf-producer.py), a byte-for-byte copy of the historical
`/tmp/opencode/published-tokie-0.2-bench/benchmark.py`.

- Archived script sha256: `649f6cc1af31dc278ca831d1fcfbb0807599e62654ca5973025d588b4a0d01c6`
- Stored results sha256: `2fe2a04f612c599efe55a5b1704830138c085dae0bfb48a3514a2a3af198c47f`
- Archive check: `cmp` against the original script is clean; the archived file was
  not reformatted.

The copy is exact, so its hardcoded absolute paths are unchanged: the corpus is
read from `/tmp/opencode/tokie-compat/corpus/` and the three `tokenizer.json`
files from the local Hugging Face cache snapshots recorded in `MODELS` inside
the script. The stored JSON points back to the same script: profile frames in
`profile.gpt2.tokie_encode.top` name
`/tmp/opencode/published-tokie-0.2-bench/benchmark.py:286(<lambda>)`, and
`meta.dataset.manifest`/`jsonl` are the corpus paths above.

## What produced the JSON, and what did not

This script is **not** `scripts/bench_published.py`. That repository script is
the producer for the tokie-vs-GigaToken `results-*.json` files and has a
different schema (`status`, `giga_over_tokie_time`, Awkward batch output). It
did not produce `hf-results.json` and is not a substitute for this archive.
`hf-producer.py` has no shared code with it and does not import it.

The script compares the published tokie wheel against Hugging Face
`tokenizers` on the same `tokenizer.json`, and writes `results.json` plus
`summary.md` next to itself.

## Recorded environment

- Python 3.12.13
- Hugging Face `tokenizers` 0.23.3
- NumPy 2.5.4
- tokie 0.2.0 from the public wheel
  `tokie-0.2.0-cp312-cp312-manylinux2014_x86_64…whl`, sha256
  `5c28027c24eb418fb9bcc9733c31ad4a6956d6129685b101e996e024669b6df7`,
  imported from the run venv (see `meta.packages.tokie_import_path` in the JSON)
- one thread: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`

Those values are also inside `meta.packages` and `meta.thread_env` in
`hf-results.json`.

## Exact invocation

The original reproduction command, with the working directory and the runtime
library path used for that run:

```bash
cd /tmp/opencode/published-tokie-0.2-bench
export LD_LIBRARY_PATH=/nix/store/604gsr59rj7dzd0nrhp143rpvf7gyiaz-gcc-15.3.0-lib/lib:/nix/store/483x61iy35irm4wr2b7dwzihljhp6da2-zlib-1.3.2/lib
export TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1
.venv/bin/python benchmark.py
```

The script takes no command-line arguments and has no flags; every input,
thread setting, round count, and output path is fixed in the file. The run
wrote `results.json` and `summary.md` in that working directory and printed
its summary to stdout (`run.log` from the same run). `results.json` was copied
unchanged to `hf-results.json` here.

## Rerun notes

- The archive is historical evidence, not a maintained harness. Do not edit it
  to reformat, parameterize, or "fix" paths; that would break the byte match.
- Running the archived copy writes `results.json` and `summary.md` beside the
  archive. Copy it to a scratch directory instead of running it inside the
  repository, and expect the original corpus and HF cache at the absolute paths
  above.
- It never wrote the committed `hf-results.json` directly, and no repository
  file was mutated to create this archive. No HF benchmark was rerun for it.
