<div align="center">

![tokie](https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie.png)

# pretokie

[![Crates.io](https://img.shields.io/crates/v/pretokie)](https://crates.io/crates/pretokie)
[![Crates.io Downloads](https://img.shields.io/crates/d/pretokie)](https://crates.io/crates/pretokie)
[![docs.rs](https://img.shields.io/docsrs/pretokie)](https://docs.rs/pretokie)
[![License](https://img.shields.io/crates/l/pretokie)](LICENSE-MIT)
[![GitHub Stars](https://img.shields.io/github/stars/chonkie-inc/tokie)](https://github.com/chonkie-inc/tokie)

*Fast, zero-allocation pretokenizers for every major tokenizer — 3.5x faster than regex*

[Quick Start](#quick-start) •
[Pretokenizers](#pretokenizers) •
[Benchmarks](#benchmarks) •
[Regex Fallback](#regex-fallback) •
[Why Hand-Coded?](#why-hand-coded)

</div>

**pretokie** splits text into pieces before BPE/WordPiece/Unigram encoding. Each pretokenizer is a hand-coded, single-pass iterator — no regex, no allocation, just raw byte-level dispatch at 300+ MB/s.

Part of the [tokie](https://github.com/chonkie-inc/tokie) tokenizer project.

## Quick Start

```toml
[dependencies]
pretokie = "0.2"
```

```rust
use pretokie::Gpt2;

let pieces: Vec<&str> = Gpt2::new("Hello world! It's a test.").collect();
assert_eq!(pieces, vec!["Hello", " world", "!", " It", "'s", " a", " test", "."]);
```

Every pretokenizer implements `Iterator<Item = &str>` — use `.collect()`, `.count()`, `.for_each()`, or any iterator combinator.

## Pretokenizers

| Name | Models | MB/s | Pieces* | cyc/B |
|------|--------|------|---------|-------|
| `Gpt2` | GPT-2, GPT-J, RoBERTa | **344** | 23.3M | 9.7 |
| `Cl100k` | GPT-3.5, GPT-4, Llama 3 | **345** | 21.7M | 9.7 |
| `O200k` | GPT-4o | **336** | 21.6M | 9.9 |
| `Bert` | BERT, DistilBERT, GTE, BGE, MiniLM | **351** | 26.1M | 9.5 |
| `Voyage` | Voyage 3, Voyage Code 3 | **307** | 22.8M | 10.9 |
| `SmolLM` | SmolLM2 | **343** | 24.9M | 9.7 |
| `DeepSeek` | DeepSeek-V3, DeepSeek-R1 | **311** | 21.7M | 10.7 |
| `Qwen` | Qwen3.5 | **310** | 22.8M | 10.8 |
| `Tekken` | Mistral Nemo, Nemotron | — | — | — |
| `Regex` | Any pattern (fallback) | 91 | 23.3M | 36.7 |

\* Pieces on 95 MB enwik8, Apple M3 Pro. Cycles/byte at 3.34 GHz.

## Benchmarks

All pretokenizers run at **307-351 MB/s** — 3.5x faster than the regex fallback at 91 MB/s. The fastest (BERT at 351 MB/s) processes 95 MB of English text in 272ms, yielding 26.1 million pieces.

For comparison, HuggingFace tokenizers' regex-based pretokenizer runs at ~100 MB/s. pretokie's hand-coded iterators eliminate regex overhead entirely.

## Usage

```rust
use pretokie::{Cl100k, O200k, Bert};

// CL100K: case-insensitive contractions, 3-digit number chunks
let pieces: Vec<&str> = Cl100k::new("DON'T count 12345").collect();
assert_eq!(pieces, vec!["DON", "'T", " count", " ", "123", "45"]);

// O200K: CamelCase splitting, contractions merge into words
let pieces: Vec<&str> = O200k::new("XMLHttpRequest don't").collect();
assert_eq!(pieces, vec!["XMLHttp", "Request", " don't"]);

// BERT: whitespace-delimited, individual punctuation
let pieces: Vec<&str> = Bert::new("Hello, world!").collect();
assert_eq!(pieces, vec!["Hello", ",", "world", "!"]);
```

`Bert` matches HuggingFace's `BertPreTokenizer` exactly: since 0.2.0 it does
**not** isolate CJK ideographs (HF does that in `BertNormalizer`'s
`handle_chinese_chars`). Pad CJK chars with spaces before calling it, or use
tokie's BERT normalizer. See [CHANGELOG.md](CHANGELOG.md).

## SIMD backend selection (`PRETOKIE_ISA`)

The `Mask`-based pretokenizers (all except `Bert`) classify 64-byte batches
with SIMD: NEON on aarch64, SSE2 on x86_64 (AVX2 is available but opt-in, as
it measured no faster end-to-end). The `PRETOKIE_ISA` environment variable
overrides the choice:

| Value | Backend |
|-------|---------|
| `scalar` | No SIMD; every piece takes the scalar `Core` path |
| `base` / `sse2` / `neon` | Baseline 128-bit SIMD (the default) |
| `avx2` | AVX2 (x86_64, runtime-detected) |

It is read **once** per process (on the first `Mask` construction) and
cached; unsupported or unknown values fall back to the default. It is
intended for benchmarking and differential testing only — output is
byte-identical across backends.

## Regex Fallback

For unknown tokenizer patterns, enable the `regex` feature:

```toml
[dependencies]
pretokie = { version = "0.2", features = ["regex"] }
```

```rust
use pretokie::Regex;

// Use built-in factories
let pretok = Regex::gpt2();
let pieces: Vec<&str> = pretok.split("Hello world").collect();

// Or compile a custom pattern
let pretok = Regex::new(&[
    (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|\p{L}+|\p{N}+", false),
]).unwrap();
```

Without the `regex` feature, pretokie has only one dependency (`unicode-general-category` for BERT punctuation classification).

## Why Hand-Coded?

Regex-based pretokenizers run at ~91 MB/s. The hand-coded iterators run at ~330 MB/s — **3.5x faster** — because they eliminate:

- **Regex compilation** — no NFA/DFA construction at startup
- **Branch overhead** — specialized byte-level dispatch instead of generic regex engine
- **Allocation** — zero heap allocation per piece (iterators borrow from the input)

The pretokenizer runs before every encode call, on every piece of text. At 25 million pieces per 95 MB, even small per-piece overhead compounds. These iterators process each byte with a single `if`-chain dispatch, yielding `&str` slices directly from the input.

## License

MIT OR Apache-2.0
