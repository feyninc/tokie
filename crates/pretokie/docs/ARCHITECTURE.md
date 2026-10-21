# Architecture

Pretokie splits text that is already normalized. Each pretokenizer is a zero-sized type. A caller asks for a piece count, or for a batch of byte spans. There is no configuration object and no regex engine on this path.

The pieces below are the ones the unit tests lock in.

## Span

A `Span` is two `u32` offsets, `start` and `end`. The text they index is at most 4 GiB. `fill` writes spans into a caller-owned buffer and returns how many it wrote. A short buffer is resumed by calling `fill` again on the same walk.

```rust
let mut walk = pretokie::Gpt2::walk("Hello world");
let mut buf = [pretokie::Span { start: 0, end: 0 }; 8];
let n = walk.fill(&mut buf);
```

`Hello world` is `Hello` and ` world`. The space stays on the second piece.

## Scheme

Every cut implements `Scheme`. Three methods matter:

- `advance` is the scalar rule. Given a piece start, it returns the end of that piece. This is the ground truth the window is checked against in debug builds.
- `align` returns the next piece start at or after a position. Cuts that keep every byte leave the position alone. BERT and T5 skip whitespace here, because those bytes are not part of any piece.
- `try_window` looks at the next 64 bytes, which begin on a piece boundary, and returns the piece starts it can finish. `None` means this piece stays on `advance`.

A window is a pair of bitsets. Bit 0 of `starts` is the first byte. Each later set bit is another piece start. When `stops` is unset, each piece ends at the next start, and `finishes` means the last piece ends at byte 64. When `stops` is set, bit `k` is the exclusive end of a finished piece, so a gap can sit between an end and the next start. BERT and T5 use that gap for the whitespace they drop. The other cuts leave `stops` unset and stay contiguous.

## Walker

Count and fill each run in one kernel. The kernel inlines the scheme's window and the classifier for that instruction set. The walker does not know the split rules. On x86-64, fill uses AVX-512 and packs spans with VBMI2 when that instruction is present, AVX-512 with a scalar pack otherwise, then AVX2, then scalar. Count uses AVX-512, then AVX2, then scalar. On aarch64 both use NEON.

`count` popcounts the start bits. `fill` pairs each start with its end and writes `Span`s. A window that returns `None`, or that cannot finish the piece it is on, falls back to `advance` for that piece and tries the window again at the next boundary.

## Classification

`classify` emits seven masks for a 64-byte window:

| mask | bytes |
| --- | --- |
| letter | ASCII `A-Z` `a-z` |
| digit | ASCII `0-9` |
| space | the space character |
| newline | `\n` and `\r` |
| other whitespace | tab, vertical tab, form feed |
| apostrophe | `'` |
| non-ASCII | bytes at or above `0x80` |

A byte such as `.` is in none of them. Multibyte characters are only marked non-ASCII here. The scalar `advance` of each cut classifies those characters itself. The ASCII windows in DeepSeek, BERT, and T5 stop at the first non-ASCII byte and let `advance` finish the piece.

## How a cut is built

Each formula lives in one file under `src/impls/`. Two cuts share a file when the "where does the next piece start?" rule is the same and every difference is a const knob.

1. Write `advance` and check it against the reference, piece for piece.
2. Add a 64-byte window that reports the same starts for the ASCII it can finish.
3. Leave everything else on `advance`. Debug builds compare the window's ends with `advance`.

The const knobs:

- `cl100k.rs` has `DIGITS3` and `MARKS`. Cl100k and Olmo group digits by up to three and ignore marks. Voyage and Qwen keep one numeric character. Qwen is the one that lets a mark continue a letter.
- `o200k.rs` has `CONTRACTIONS`, `DIGITS3`, `SLASH`, and `HAN`. O200k turns all four on except Han. Nemotron drops contractions and groups digits one at a time, and still absorbs a slash into punctuation. Kimi keeps contractions and groups of three, does not absorb a slash, and gives Han its own run.

GPT-2, SmolLM, DeepSeek, BERT, and T5 each have their own file. SmolLM is the GPT-2 rule run after each numeric character has been taken out on its own. `a12b` is `a`, `1`, `2`, `b`. It is a separate formula because that first pass changes where the next piece starts.

DeepSeek walks left to right once. A numeric character and a character in the CJK range are hard boundaries, and the main pattern runs inside each region. `!hello` is one piece, because a single ASCII punctuation mark fuses with the ASCII letters after it. `!!hello` is `!!` and `hello`. `1234` is `123` and `4`.

BERT drops whitespace and isolates every punctuation character. `Hello, world!` is `Hello`, `,`, `world`, `!`. `don't` is `don`, `'`, `t`. `中文` stays one piece. The ASCII window builds a punctuation mask beside the seven class masks, and it emits `stops` so the dropped whitespace is a gap.

T5 drops whitespace and keeps everything else on the word. `Hello, world!` is `Hello,` and `world!`. `don't` and `$5` each stay one piece. The window is the BERT window without the punctuation mask.

## Oracles

Each cut has its own file under `tests/`. The oracle is the reference regex, or the digits-then-GPT-2 procedure for SmolLM. A test checks four things: the hand cases, every character in the Basic Multilingual Plane, a generated text, and an ASCII window fuzz that shifts the text across the 64-byte boundary. `fancy-regex` is a dev-dependency used by those tests. The library does not depend on it.

## Unknown patterns

A pattern that matches none of these cuts is refused. `pretokie::refuse` returns `UnsupportedPattern`, whose message is `no pretokenizer for pattern: …`. A known cut is its own type. This function does not run the pattern.
