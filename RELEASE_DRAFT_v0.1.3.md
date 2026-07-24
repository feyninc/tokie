# tokie 0.1.3

This release makes tokie faster than [gigatoken](https://github.com/marcelroed/gigatoken) on every encoding contract the two libraries share, and adds a bulk file-encoding API for tokenizing large corpora. All output stays token-exact with HuggingFace — 20342/20342 documents match on both GPT-2 and DeepSeek-V3 over real web text.

## Bulk file encoding

New `encode_files` / `count_tokens_files` read corpus files in Rust, split on a separator, and encode every document across all cores — the text never crosses the Python string boundary. They return a flat `uint32` id array plus a `uint64` per-document offset array (zero-copy numpy):

```python
ids, offsets = tokenizer.encode_files(["corpus.txt"], separator=b"<|endoftext|>")
```

Output is identical to concatenating `encode_batch_flat` over the same documents, and invalid UTF-8 is handled losslessly rather than raising.

## Faster encoding

The hot path was rebuilt end to end, each change measured in isolation on an Apple M3:

- **Cache-first encoder loop.** Cycle-accounting showed the per-piece cost was instruction overhead, not memory latency — the pretoken-cache key is now built with a single masked 16-byte load from the document slice and a fixed-width emit, halving the cache-hit path (13.6 → 5.8 ns/piece). Every single-string path now checks out the same warm cache the batch workers use.
- **Rank-table merge core.** Cache misses now resolve through a classic lowest-rank-first merge over an open-addressed pair table instead of the Aho-Corasick backtracking walk — 1.7–2.6x faster on the miss path (most visible on DeepSeek-V3), and provably identical to the old encoder on every real vocabulary.
- **Fused pretokenize + encode.** The pretokenizer and encode loop are monomorphized into a single pass, and two parallel-path cliffs were removed (an oversized per-document threshold that ran sub-chunks uncached, and batch workers that nested a thread pool per document). This roughly doubled multi-threaded batch throughput.
- **Mask-scanner bulk drain.** The SIMD pretokenizer emits piece boundaries in bulk runs instead of one `Iterator::next` call per piece, lifting single-thread pretokenization ~14%.

Against gigatoken on the same machine: warm load 3–17 ms (vs 36–142 ms), single-string latency 0.38 µs (vs 0.54 µs), and `encode_batch` throughput ~1.7x higher — with identical accuracy and lower memory.

## Fixes

- **Rank table on `.tkz` load.** The pair-table rebuild inserted degenerate self-referential entries for added/special tokens (like `<|endoftext|>`), which silently disabled the rank-merge miss path on every binary load. Fixed; existing `.tkz` files and tokiers repos pick up the faster path on next load with no regeneration needed.
- **Security.** `pyo3` and `numpy` bindings updated to 0.29, resolving [GHSA-36hh-v3qg-5jq4](https://github.com/PyO3/pyo3/security/advisories/GHSA-36hh-v3qg-5jq4) (high) and [GHSA-chgr-c6px-7xpp](https://github.com/PyO3/pyo3/security/advisories/GHSA-chgr-c6px-7xpp) (moderate).

The `pretokie` crate is bumped to 0.1.3 alongside tokie, since its pretokenizer core changed.
