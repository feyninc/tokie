# tokie 0.1.0 — exact parity, and a much faster core

This release is the result of a deep parity audit against HuggingFace `tokenizers` followed by a performance campaign against the fastest tokenizers available. tokie now matches HF token-for-token on every supported model family we test — measured per-document on real web text, not just clean benchmark strings — and is substantially faster everywhere: loading, single-string latency, and batch throughput.

**Upgrade note: this release is required to keep using `tokiers/` pre-built tokenizers.** The `.tkz` binary format moved to v12 (it now carries the SentencePiece precompiled charsmap), and the `tokiers/` hub repos have been regenerated in the new format. tokie ≤0.0.10 cannot read v12 files.

## Accuracy: exact HF parity on web text

Our CI previously validated against enwik8 (XML-escaped Wikipedia), which systematically under-covers the unicode that real web text is full of. Testing per-document on OpenWebText surfaced a family of pretokenizer divergences, all fixed:

- Contractions no longer assume a lookahead the regex doesn't have: `O'Toole` → `O` / `'T` / `oole`, `don'ts` → `don` / `'t` / `s` (GPT-2, Qwen, cl100k families).
- `\p{N}` now includes non-ASCII numerics (`¹`, `❶`, `½`) in digit runs, space-attachment, and whitespace lookaheads (GPT-2, cl100k, SmolLM families).
- Multibyte punctuation (`”`, `…`, `«`) now consumes trailing newlines per `[^\s\p{L}\p{N}]+[\r\n]*` (Qwen/cl100k families).
- The Unicode combining-mark table is now generated, complete, and exact (Unicode 15.1). The old hand-rolled table was missing 1,015 codepoints — including the entire Gujarati block — and misclassified 126 others.
- DeepSeek-V3/R1: digit chunking is detected across multi-stage Split patterns (was silently falling back to single-digit mode when loading from `tokenizer.json`), the punct class is the positive `[\p{P}\p{S}]` (format/control chars like ZWSP and soft hyphen are handled like HF), and the nonexistent contraction rule is gone.
- SentencePiece models with a `Precompiled` normalizer (XLM-R, T5, BGE-M3, …) now apply the actual precompiled charsmap from `tokenizer.json` instead of a category-level approximation, with f64 scores and per-piece Viterbi. This closes the last known parity gap (~1% of web-text documents on xlm-roberta-base).
- SentencePiece metaspace `Prepend("▁")` is applied unconditionally like HF, fixing dropped `▁` tokens around added tokens (e.g. `</s>` in Llama-2-style chat text).

Verified: 9,909/9,909 OpenWebText documents match HF exactly across GPT-2, Qwen3, SmolLM2, DeepSeek-V3, and BERT; 20,342/20,342 on the extended 95 MB gate for GPT-2 and DeepSeek-V3; 5,000/5,000 on xlm-roberta-base. The accuracy CI now includes per-document OpenWebText comparison for one representative model per pretokenizer family, loaded from the original HF repos so the `tokenizer.json` detection path is exercised.

## Performance

Measured on an Apple M3 (8 cores), GPT-2, 48 MB of OpenWebText, best of 3, against gigatoken 0.9.0 under an identical protocol:

| | tokie 0.0.10 | tokie 0.1.0 | gigatoken 0.9.0 |
|---|---:|---:|---:|
| Warm load (`from_pretrained`) | ~170 ms | **3–17 ms** | 36–142 ms |
| Single short-string encode | 1.2 µs | **0.38 µs** | 0.54 µs |
| `encode_batch` (full Encoding objects) | 69 MB/s | **~400 MB/s** | 278 MB/s (bare int lists) |
| `encode_batch_flat` (contiguous ids) | — | **~400 MB/s** | 1.9 GB/s (awkward array) |

What changed under the hood:

- **Mask-scanner pretokenization**: a NEON/SWAR scanner classifies 64-byte blocks into class bitmasks and derives piece boundaries with bitwise algebra, falling back to the scalar state machine only in ambiguous zones. Single-thread pretokenization went from ~435 MB/s to ~1.3 GB/s, validated against the scalar implementation with 150k-string fuzzing plus full-corpus piece diffs across all seven pretokenizer families.
- **Pretoken caching**: a compact open-addressing cache (16-byte key block, up to 3 inline token ids) resolves repeated pretokens in one probe, with long-lived caches leased from a process-wide pool across batch calls.
- **Zero-allocation hot path**: encoding appends into shared buffers instead of allocating a `Vec` per pretoken (previously ~200k heap allocations per MB).
- **Work-stealing batch scheduler**: byte-balanced chunks claimed atomically by workers, so a slow core or an oversized document no longer stalls the batch.
- **Compiled load cache**: `from_pretrained` resolves the hub disk cache offline-first and stores a compiled `.tkz` next to the snapshot, so warm loads skip both the network and the JSON parse entirely.
- **Lazy Python encodings**: `Encoding.tokens` / masks materialize on access instead of eagerly under the GIL; an ids-only fast path backs `encode()` when no padding is configured.

## New APIs

- `encode_batch_flat(texts)` (Python): returns a contiguous `uint32` numpy array of ids plus per-document lengths — the zero-materialization output shape for bulk corpus processing.
- `Tokenizer::encode_ids(text, add_special_tokens)` (Rust): bare token ids without the `Encoding` wrapper.
- `TOKIE_CACHE_BITS` env var tunes the pretoken cache table size (default 16 → 2 MiB per worker).

## Breaking changes

- `.tkz` format v12: adds the precompiled-charsmap section. v12 readers load older files; pre-0.1.0 readers cannot load v12 files. The `tokiers/` repos are already on v12 — upgrade to keep using them.
- `pretokie` 0.1.0 is a required companion release (tokie now depends on the workspace-current pretokie; crates.io publishing order: `pretokie` first, then `tokie`).

## Known limitations

- Bulk flat-buffer throughput still trails gigatoken's ~1.9 GB/s on many-core batch workloads. Profiling shows the remaining bottleneck is BPE merge-table locality, not pretokenization or scheduling — that's the next campaign.
- WordPiece/Unigram and all supported families are exact vs HF on our corpora; if you find a divergence, please open an issue with the input string — the accuracy suite makes these fast to fix.
