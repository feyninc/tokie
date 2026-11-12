# tokie 0.2.0

tokie 0.2.0 is a breaking release for Rust and Python. The core is rebuilt
around caller-owned buffers: encode appends into a caller-owned vector or
string and returns `Result`. It also replaces the legacy `.tkz` cache with the
new `.tokie` file and restores the useful 0.1.4 mainline names as the 0.2 calls.

The full call list is in the [API guide](https://github.com/feyninc/tokie/blob/main/docs/API.md);
the [API changes](https://github.com/feyninc/tokie/blob/main/docs/API_CHANGES_0_2.md) and
[migration guide](https://github.com/feyninc/tokie/blob/main/docs/MIGRATION.md)
list every renamed, changed, and removed surface. This is not a compatibility
facade; there is no full 0.1.4 guarantee.

## Rust

- Encode methods take `&mut self`, append into the caller's `Vec<TokenId>`,
  and return `Result<(), tokie::Error>`. They never clear the vector.
- `encode` leaves the parsed template off; `encode_plus` applies it.
  `Plus::new(text).pair(other).truncate(n).pad(n)` adds a pair, a cut, or
  padding. One `max_len` counts template ids; truncation drops from the
  right of the longer sequence first, and padding is fixed on the right.
- `batch` appends into one id buffer and a `Vec<usize>` of cumulative ends.
  Record `ids.len()` and `ends.len()` before the call: the first new text starts
  at the old `ids.len()`, and the new ends are absolute `ids.len()` boundaries
  for each text, not per-text lengths. Existing ids and ends stay in place.
- `decode` appends into the caller's `String` and takes `Specials::Keep` or
  `Specials::Skip`. `decode_bytes` returns raw bytes. `Specials::Skip` omits
  specials in both.
- `count_tokens` and `count_tokens_batch` return `Result` lengths. They encode
  the text; they are not lazy and not allocation-free. The template stays off.
- `TokenId` is `u32` at the crate root. `token` and `id` use the raw vocab
  spelling, as does `get_vocab`. `vocab_size` is the highest id plus one, so it
  can exceed the `get_vocab` entry count. `pad_token_id` checks `[PAD]`, then
  `<pad>`.
- Loading is `open(path)` by magic, `json(text)`, and `pretrained(repo)` for
  the Hub. `from_json(path)` and `from_file(path)` are path aliases for
  `open`; `from_pretrained` aliases `pretrained`; `to_file` aliases `save`.
  The `hub` Cargo feature is on by default, and `hf` is its alias.

## Python

- `encode` and `encode_plus` return NumPy `uint32` arrays, not `Encoding`
  objects. There is no `.ids`, `.tokens`, offsets, attention mask, or type ids;
  use `.tolist()`.
- The template default is gone: `encode` is always template off. Use
  `encode_plus` for 0.1.4's old default.
- `batch` returns `(uint32 ids, uint64 ends)` with cumulative ends, not one
  object or one length per document.
- `pair`, `truncate`, and `pad` are per-call arguments. Truncation is from the
  right with the longer sequence first; padding is fixed on the right; one
  `max_len` covers both and counts the template ids.
- `decode` returns `str` (`skip=False` keeps specials) and is lossy for
  invalid UTF-8; `decode_bytes(..., skip=True)` returns raw bytes. Unknown or
  gap ids are errors.

```python
# 0.1.4
enc = tokenizer.encode("Hello")       # template on by default
ids = enc.ids

# 0.2
ids = tokenizer.encode_plus("Hello")  # uint32 array, template on
tokenizer.encode("Hello")             # template off
tokenizer.decode(ids.tolist(), skip=True)
```

## Removed surface

The `Encoding` object and its masks, offsets, and type ids are gone. Also
removed: Python `__call__`, `encode_pair`, `encode_with_offsets`,
`encode_bytes`, and `decode_batch`; the `id_to_token`/`token_to_id` aliases;
`encode_batch`/`encode_batch_flat`; file-corpus and lazy-counting calls;
configurable padding and truncation; old component and options construction;
and typed legacy errors. `tokie::Error` covers load, encode, and decode.
`token` and `id` use the raw vocab spelling, and the old lookup names are not
aliases.

## Loading and `.tokie`

`save` writes a `.tokie` file of normalized tables and loaded pieces, so a
warm load is deserialization. `open` reads a `.tokie` file or a
`tokenizer.json` by magic. The legacy 0.1.4 `.tkz` format is unsupported: a
genuine legacy `TOKI` image is refused by its magic, not by filename, and a
JSON file merely named `.tkz` is parsed as JSON. Legacy 0.1.4 binaries cannot
read `.tokie`; rebuild from `tokenizer.json` with 0.2 code, because renaming a
`.tkz` file is not enough.

`pretrained`/`from_pretrained` open exactly one `.tokie` file from a Hub repo,
whatever its name. With no `.tokie` file, `tokenizer.json` is parsed; several
`.tokie` files is an error. No runtime `.tkz` fallback.

## CLI

The 0.2 CLI source provides `encode`, `ids`, `decode`, `save`, and `bench`.
The `tokie-cli` package is not published in this release.

## Crates

`tokie` 0.2.0 depends on `normalie` 0.1.0 and `pretokie` 0.2.0, both standalone workspace crates.

## Correctness and performance

Correctness gates pass: workspace Rust and Python tests, tokenizer id oracles, and the supported/refused model matrix.

The only speed figures remain the historical 0.1.3/0.1.4 benchmarks. 0.2 has
not been re-timed against Hugging Face tokenizers, so these notes make no new
speed claim. See [BENCHMARK_AFTER.md](https://github.com/feyninc/tokie/blob/main/docs/BENCHMARK_AFTER.md)
for the final measurement report.

### Known measurement limitations

The strict performance criterion is uncertified/partial. The Rust quiet-window
paired run found no slowdown at its reported finite resolution (median
after/before 1.000438, 95% CI 0.998352–1.005683), which is not proof of zero
change. In the final Python matrix there is no statistically reliable
familywise regression, but one raw `o200k:batch:1` row is about 0.7% slower
with Holm-adjusted p=1.0, so it is unconfirmed; two batch-8 rows are
inconclusive. 0.2.0 ships with that caveat recorded; this is not a 0% regression proof.

## License

MIT OR Apache-2.0.
