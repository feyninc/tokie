# tokie

Rust tokenizer workspace for [Chonkie](https://github.com/chonkie-inc/tokie). It loads Hugging Face `tokenizer.json` files and implements BPE, WordPiece, SentencePiece, and Unigram. The repo also contains Python bindings and a CLI.

License is MIT OR Apache-2.0.

## Crates

| Path | Package | Role |
| --- | --- | --- |
| `crates/tokie` | `tokie` 0.2.0 | Tokenizer. Normalie rewrites the string, pretokie splits it, tokie emits ids. |
| `crates/normalie` | `normalie` 0.1.0 | Normalizers used by tokie. |
| `crates/pretokie` | `pretokie` 0.2.0 | Pretokenizers used by tokie. Splits text into pieces before the model encodes them. |
| `crates/tokie-python` | `tokie-python` | PyO3 bindings. The Python module name is `tokie`. |
| `crates/tokie-cli` | `tokie-cli` | `tokie` binary for building, verifying, and uploading tokenizers. |

The workspace members are `crates/normalie`, `crates/pretokie`, and `crates/tokie`. `tokie-python` and `tokie-cli` stay in the tree and are not members until they are rewritten onto this tokie.

## Pretokie on this branch

This branch deleted the old `crates/pretokie/src` and `examples/`. Do not restore `PretokConfig`, `Mask<C>`, or `Gpt2::new`.

Each pretokenizer lives in its own file under `src/impls/`. Each pretokenizer's oracle tests live in their own file under `crates/pretokie/tests/`. Do not share a test file across pretokenizers. `span.rs` holds `Span`. `src/walk/` is the shared walker. `mod.rs` is the kernel. `classify.rs` emits seven masks for a 64-byte window: ASCII letter, digit, space, newline, other whitespace, apostrophe, and non-ASCII. The kernel inlines the classifier and the scheme's window. It does not know the split rules. `Gpt2::count`, `Gpt2::count_upto`, and `Gpt2::walk` are the calls. A pretokenizer is a zero-sized type. Callers pass text that is already normalized. Design notes, when present, are in `notes/` and are gitignored.

`cargo check -p pretokie` succeeds. `tokie` 0.2 depends on this pretokie and on normalie.

## Commands

```bash
cargo check -p pretokie
cargo test -p pretokie
cargo test -p tokie
```

Check or test one package. `tokie` 0.2 builds against this pretokie and normalie. Pretokie is 0.2.0. Normalie is 0.1.0.

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
