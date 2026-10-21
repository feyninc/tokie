# tokie

Rust tokenizer workspace for [Chonkie](https://github.com/chonkie-inc/tokie). It loads Hugging Face `tokenizer.json` files and implements BPE, WordPiece, SentencePiece, and Unigram. The repo also contains Python bindings and a CLI.

License is MIT OR Apache-2.0.

## Crates

| Path | Package | Role |
| --- | --- | --- |
| `crates/tokie` | `tokie` 0.1.4 | Tokenizer library. Encode, decode, and load from Hugging Face. |
| `crates/pretokie` | `pretokie` 0.2.0 | Pretokenizers used by tokie. Splits text into pieces before the model encodes them. |
| `crates/tokie-python` | `tokie-python` | PyO3 bindings. The Python module name is `tokie`. |
| `crates/tokie-cli` | `tokie-cli` | `tokie` binary for building, verifying, and uploading tokenizers. |

The workspace members are `crates/*`. Project docs are `README.md`, `docs/MIGRATION.md`, and `docs/TOKENIZERS.md`.

## Pretokie on this branch

This branch deleted the old `crates/pretokie/src` and `examples/`. Do not restore `PretokConfig`, `Mask<C>`, or `Gpt2::new`.

Each pretokenizer lives in its own file under `src/impls/`. Each pretokenizer's oracle tests live in their own file under `crates/pretokie/tests/`. Do not share a test file across pretokenizers. `span.rs` holds `Span`. `src/walk/` is the shared walker: `mod.rs` is the kernel, and `classify.rs` emits seven masks for a 64-byte window (ASCII letter, digit, space, newline, other whitespace, apostrophe, and non-ASCII). The kernel inlines the classifier and the scheme's window. It does not know the split rules. `Gpt2::count`, `Gpt2::count_upto`, and `Gpt2::walk` are the calls. A pretokenizer is a zero-sized type. Callers pass text that is already normalized. Design notes, when present, are in `notes/` and gitignored.

`cargo check -p pretokie` succeeds. `crates/tokie` still names the deleted types, so tokie does not build until those calls change.

## Commands

```bash
cargo check -p pretokie
cargo test -p pretokie
cargo test -p tokie
```

Check or test one package. The workspace build fails while tokie still calls the deleted pretokie types.

## Commit messages

Every commit subject uses one of these two prefixes:

```
feat(scope): short imperative summary
bug(scope): short imperative summary
```

`feat` is new behavior, a rewrite, docs, or a refactor. `bug` fixes incorrect behavior. Use no other prefix. That includes `docs`, `chore`, `fix`, and `refactor`.

`scope` is `pretokie`, `tokie`, `python`, `cli`, or `docs`. The summary is lowercase and imperative, with no trailing period. Add a body when the subject does not say why.

```
feat(pretokie): replace the config matrix with one file per formula

bug(tokie): keep a tab from prefixing the next gpt-2 piece
```

Commit only when the user asks.
