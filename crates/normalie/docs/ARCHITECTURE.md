# Architecture

Normalie rewrites text. Pretokie then splits that string. Token offsets index the rewritten string.

Each formula is a type. A formula with no table and no blob is zero-sized. `Precompiled` and `PunctPad` hold a SentencePiece charsmap. `StripAccents` holds the small Unicode 9 mark set. There is no regex engine on this path.

The strings below are the ones the oracles lock to Hugging Face tokenizers v1.

## Write

`write` reads `text` and, when the formula changes it, pushes the result onto a caller-owned `String`. The caller passes an empty buffer. `write` returns whether it pushed. A `false` return leaves the buffer empty, and the caller keeps `text`.

```rust
let mut out = String::new();
let changed = normalie::BertUncased::write("Hello World", true, &mut out);
```

`Hello World` becomes `hello world`.

`at_start` is true for a whole input and for the segment that begins at byte 0. Added-token splits call `write` once per segment. `MetaspaceFirst` is the formula that reads the flag. The others ignore it.

Empty input stays empty. `write` returns `false` and does not invent a leading `▁`.

## Formula

Every formula implements `Formula`. Two methods matter:

- `write_scalar` is the scalar rule. It is the ground truth the window is checked against in debug builds.
- `try_window` looks at the next 64 bytes and returns the output bytes it can finish. `None` means this stretch stays on `write_scalar`.

A window reports a copy, a byte map, or an expand.

- A copy is an identity run. NFC on ASCII, and a charsmap that does not remap printable ASCII, are copies.
- A map is one output byte per input byte. ASCII lowercase ORs `0x20` into `A..=Z`. Those bytes sit inside one UTF-8 byte, so the map is safe on mixed text.
- An expand writes more bytes than it reads. A space becomes `▁` (`E2 96 81`). Byte-level turns one byte into the one or two UTF-8 bytes of its GPT-2 character.

The window also returns the input bytes it consumed. A combining mark, a multibyte character, or a charsmap grapheme that crosses the 64-byte edge is not consumed. `write_scalar` finishes that spot, and the window tries again after it.

## Kernel

`write` runs in one kernel. The kernel inlines the formula's window. It does not know the rewrite rules.

On x86-64 the order is AVX-512, then AVX2, then scalar. On aarch64 the kernel uses NEON. The scan is a 64-byte compare. A formula whose window is only a copy uses that dispatch, then `write_scalar` for the first byte the scan cannot copy.

NFC of a long string scans for bytes at or above U+0300. A starter that is already NFC is skipped. A scalar that can compose is checked with ICU, together with the previous scalar. A clean string is not rewritten. NFD, NFKC, and NFKD go to ICU or the quick-check tables after an ASCII check.

The charsmap trie is scalar. Its fast path copies a printable ASCII run as a slice and rewrites other BMP scalars through a table.

## Chains

A model that stacks formulas is a chain. `Scratch` owns two buffers and swaps them. A step that returns `false` leaves the next step reading the previous string, so an already-normal prefix is copied only when a later step changes it.

```rust
let mut scratch = normalie::Scratch::new();
let out = scratch.apply2(
    text,
    true,
    |s, at, buf| map.write(s, at, buf),
    |s, at, buf| normalie::MetaspaceWords::write(s, at, buf),
);
```

`out` borrows one of the two buffers, or `text` when both steps return `false`. T5 is `Precompiled` then `MetaspaceWords`. bge-m3 is three steps, so `Scratch::apply3` is the same swap with one more closure. `Llama` and `BertUncased` stay single formulas. Each of those sequences shares one left-to-right pass, which is what makes BERT uncased about 6× the Hugging Face chain on the same file (264 MB/s against 45 MB/s).

## Crate

The package is `crates/normalie`, edition 2024, version 0.1.0, MIT OR Apache-2.0. `lib.rs` re-exports the formulas, `Scratch`, `refuse`, and `UnsupportedNormalizer`. The trait and the kernel stay private, the same way pretokie keeps `Scheme` private and publishes `Gpt2::count`.

```text
src/lib.rs
src/formula.rs          Formula, Window
src/kernel.rs           ISA dispatch
src/scan.rs             ASCII and printable prefixes
src/form.rs             Unicode quick check and ICU
src/marks.rs            Unicode 9 mark set
src/scratch.rs          Scratch
src/charsmap.rs         precompiled_charsmap parser
src/impls/mod.rs
src/impls/unicode.rs    Nfc, Nfd, Nfkc, Nfkd
src/impls/nmt.rs
src/impls/lower.rs
src/impls/strip_accents.rs
src/impls/bert.rs       Clean, BertCased, BertUncased
src/impls/edit.rs       Prepend, ReplaceSpace, CollapseSpaces, Strip
src/impls/metaspace.rs
src/impls/llama.rs
src/impls/byte_level.rs
src/impls/precompiled.rs
src/impls/punct_pad.rs
src/impls/albert.rs
src/impls/refuse.rs
```

A zero-sized formula exposes `write` as an associated function. `Precompiled` and `PunctPad` hold a parsed charsmap, so `write` takes `&self`. Both use the parser in `charsmap.rs`. `Albert` shares the Unicode 9 mark set with `StripAccents` and holds nothing of its own, so it stays zero-sized.

`icu_normalizer` runs NFC, NFD, NFKC, and NFKD. `unicode-normalization` supplies the quick-check tables and the Unicode 9 combining-mark test. `unicode-segmentation` finds grapheme clusters on the charsmap trie path. `unicode_categories` is the BERT and Albert category test, which is what keeps U+0378. `memchr` finds spaces. A general regex crate is not a dependency.

`cargo test -p normalie` locks strings in `tests/`, one file per formula, plus `tests/t5.rs` and `tests/bge.rs` for the chains. `bench/normalize` is the live comparison against Hugging Face tokenizers v1 and gigatoken. The published crate does not link either of them.

## How a formula is built

Each formula lives in one file under `src/impls/`. Two formulas share a file when the scalar rule is the same and every difference is a const knob.

1. Write `write_scalar` and check it against Hugging Face tokenizers v1, string for string.
2. Add a 64-byte window that emits the same bytes for the ASCII it can finish.
3. Leave everything else on `write_scalar`. Debug builds compare the window's output with `write_scalar`.

`bert.rs` has `CLEAN`, `CJK`, `ACCENTS`, and `LOWER`. `BertUncased` turns all four on. `BertCased` cleans and spaces CJK. `Clean` only cleans.

`metaspace.rs` has `PREPEND` and `DROP`. `Metaspace` prepends unless the text already starts with a space or `▁`, and it turns spaces into `▁`. `MetaspaceFirst` prepends only when `at_start` is set. `MetaspaceNever` does not prepend. `MetaspaceWords` drops whitespace and prepends `▁` on every word.

`Precompiled` parses the charsmap once. Each ASCII byte and the `\r\n` grapheme get a cached replacement. The copy scan is valid when no printable ASCII byte is remapped. The T5 map qualifies. A map that rewrites a printable ASCII byte stays on the trie.

## Oracles

Each formula has its own file under `tests/`. The locked strings are the Hugging Face tokenizers v1 results. A test checks four things: the hand cases, every scalar in the Basic Multilingual Plane, a generated text, and a window fuzz that shifts the text across the 64-byte boundary. `tests/t5.rs` and `tests/bge.rs` check the chains the same way.

## Unknown types

A `tokenizer.json` normalizer that matches none of these formulas is refused. `normalie::refuse` returns `UnsupportedNormalizer`, whose message is `no normalizer for type: …`. A known formula is its own type. This function does not run the normalizer.
