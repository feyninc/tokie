<div align="center">

![tokie](https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie.png)

# pretokie

[![Crates.io](https://img.shields.io/crates/v/pretokie)](https://crates.io/crates/pretokie)
[![Crates.io Downloads](https://img.shields.io/crates/d/pretokie)](https://crates.io/crates/pretokie)
[![docs.rs](https://img.shields.io/docsrs/pretokie)](https://docs.rs/pretokie)
[![License](https://img.shields.io/crates/l/pretokie)](LICENSE-MIT)
[![GitHub Stars](https://img.shields.io/github/stars/chonkie-inc/tokie)](https://github.com/chonkie-inc/tokie)

*Fast, zero-allocation pretokenizers for the major tokenizers, up to 3 times Hugging Face tokenizers*

[Quick start](#quick-start) •
[Cuts](#cuts) •
[Unknown patterns](#unknown-patterns) •
[Benchmarks](#benchmarks)

</div>

**pretokie** splits text into pieces before BPE, WordPiece, or Unigram encoding. Each cut is a zero-sized type. A call returns a piece count, or a batch of byte spans into the text you passed in. The text is at most 4 GiB.

Part of the [tokie](https://github.com/chonkie-inc/tokie) tokenizer project. The patterns and the walker are written up in [docs/PATTERNS.md](docs/PATTERNS.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Quick start

```toml
[dependencies]
pretokie = "0.2"
```

```rust
use pretokie::{Gpt2, Span};

let text = "Hello world! It's a test.";
let mut walk = Gpt2::walk(text);
let mut buf = [Span { start: 0, end: 0 }; 8];
let mut pieces = Vec::new();
loop {
    let n = walk.fill(&mut buf);
    if n == 0 {
        break;
    }
    pieces.extend(buf[..n].iter().map(|span| &text[span.start as usize..span.end as usize]));
}
assert_eq!(
    pieces,
    ["Hello", " world", "!", " It", "'s", " a", " test", "."]
);
assert_eq!(Gpt2::count(text), pieces.len());
```

`fill` writes the next pieces that fit in the buffer and returns how many it wrote. A short count, including zero, means the text is finished. `count` is the total. `count_upto` stops once the count passes a limit and returns `limit + 1` when more pieces remain.

`Gpt2::fill(text, &mut buf)` is the same walk for one batch.

Every cut has the same four calls: `count`, `count_upto`, `fill`, and `walk`.

## Cuts

Pass text that is already normalized. A space stays on the piece it prefixes, except where a cut drops whitespace.

| Type | Models | What a piece is |
| --- | --- | --- |
| `Gpt2` | GPT-2, r50k, p50k | Letters, numbers, and punctuation, with a leading space. Contractions are the lowercase forms `'s`, `'t`, `'m`, `'d`, `'ll`, `'ve`, `'re`. |
| `SmolLM` | SmolLM | Each digit is taken out, then the GPT-2 rule runs. `12345` is five pieces. |
| `Cl100k` | cl100k_base, Llama 3 | Case-insensitive contractions as their own pieces. Digits in groups of up to three. |
| `Olmo` | Olmo | The same cut as `Cl100k`. |
| `Voyage` | Voyage, Qwen2 | The cl100k cut with one digit per piece. |
| `Qwen` | Qwen3.5 | `Voyage`, and a mark continues a letter. |
| `O200k` | o200k_base, GPT-4o, Llama 4, gpt-oss | Case split on `camelCase`. Contractions stay on the word. Digits in groups of up to three. |
| `Nemotron` | Nemotron, tekken | The o200k case split. Contractions and digits are each their own pieces. |
| `Kimi` | Kimi K2 | `O200k`, with Han in its own run. |
| `DeepSeek` | DeepSeek V3 | Numbers, then the CJK range, then the main pattern. `!hello` is one piece. `!!hello` is `!!` and `hello`. |
| `Bert` | BERT, DistilBERT, ELECTRA, MiniLM, GTE, BGE | Whitespace is dropped. Each punctuation character is its own piece. |
| `T5` | T5, ALBERT, XLM-RoBERTa | Whitespace is dropped. Punctuation stays on the word. |

```rust
use pretokie::{Bert, Cl100k, O200k, T5};

let text = "DON'T";
let mut buf = [pretokie::Span { start: 0, end: 0 }; 4];
let n = Cl100k::fill(text, &mut buf);
assert_eq!(n, 2);
assert_eq!(&text[buf[0].start as usize..buf[0].end as usize], "DON");
assert_eq!(&text[buf[1].start as usize..buf[1].end as usize], "'T");

let text = "camelCase";
let n = O200k::fill(text, &mut buf);
assert_eq!(
    (0..n)
        .map(|i| &text[buf[i].start as usize..buf[i].end as usize])
        .collect::<Vec<_>>(),
    ["camel", "Case"]
);

let text = "Hello, world!";
let n = Bert::fill(text, &mut buf);
assert_eq!(
    (0..n)
        .map(|i| &text[buf[i].start as usize..buf[i].end as usize])
        .collect::<Vec<_>>(),
    ["Hello", ",", "world", "!"]
);

let n = T5::fill(text, &mut buf);
assert_eq!(
    (0..n)
        .map(|i| &text[buf[i].start as usize..buf[i].end as usize])
        .collect::<Vec<_>>(),
    ["Hello,", "world!"]
);
```

`Bert` and `T5` drop whitespace, so a span can end before the next one starts. The other cuts cover the text with no gaps.

## Unknown patterns

A known cut is its own type. Any other split pattern is refused, and pretokie does not run it:

```rust
let err = pretokie::refuse(r"\p{L}+").unwrap_err();
assert_eq!(err.to_string(), r"no pretokenizer for pattern: \p{L}+");
```

The only runtime dependency is `unicode-general-category`, used for BERT punctuation.

## Mask pretokenizers

`pretokie::mask` holds the mask-based pretokenizers tokie runs today. Each scheme is a `Mask` over a config (`Gpt2`, `Cl100k`, `O200k`, `Voyage`, `SmolLM`, `DeepSeek`, `Qwen`, `Tekken`), and `Bert` and the `regex` fallback live there too. They take piece callbacks (`for_each_piece`) and block callbacks (`for_each_block`). They stay until tokie moves onto the cuts above. The scanner is written up in [docs/MASK.md](docs/MASK.md). The Oniguruma oracle that checks it against Hugging Face is in [docs/ORACLE.md](docs/ORACLE.md).

```rust
use pretokie::mask::Gpt2;

let pieces: Vec<&str> = Gpt2::new("Hello world! It's a test.").collect();
```

## Benchmarks

One OpenWebText sample, 200,041,585 bytes. Release build, fat LTO, median of five passes after one warmup, on a machine with AVX-512 and VBMI2. pretokie fills spans in batches of 256. An empty cell means that library has no cut for the row. The full notes are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

| Cut | pretokie MB/s | gigatoken MB/s | HF tokenizers v1 MB/s |
| --- | ---: | ---: | ---: |
| GPT-2 | 2029.2 | 2040.0 | 995.8 |
| SmolLM | 2049.8 |  |  |
| cl100k | 1969.1 | 1639.3 | 908.9 |
| Voyage | 2016.7 | 1696.9 | 898.2 |
| Olmo | 1927.1 | 1627.0 | 898.9 |
| Qwen | 2012.9 | 1700.5 |  |
| o200k | 1766.3 | 1504.3 | 565.7 |
| Nemotron | 1963.4 | 1598.6 | 584.5 |
| Kimi | 1801.2 | 1521.7 | 540.3 |
| DeepSeek | 1237.7 | 664.6 | 846.4 |
| BERT | 1699.1 |  | 876.7 |
| T5 | 1861.1 |  | 1097.1 |

## License

MIT OR Apache-2.0
