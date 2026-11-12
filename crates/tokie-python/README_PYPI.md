<p align="center">
  <img src="https://raw.githubusercontent.com/feyninc/tokie/main/assets/tokie-banner.png" alt="tokie">
</p>

<p align="center">
  <a href="https://github.com/feyninc/tokie">GitHub</a> · <a href="https://crates.io/crates/tokie">crates.io</a>
</p>

Python bindings for the tokie 0.2 tokenizer. `encode`, `encode_plus`, and `batch` release the GIL and return NumPy arrays. There is no `Encoding` object and no `.ids` attribute; use `.tolist()` only to print or assert.

`decode` and `decode_bytes` take the array returned by `encode`, `encode_plus`, or a slice of `batch`'s flat `uint32` ids. Pass it directly; do not call `.tolist()` first. The array must be one-dimensional, native-endian, C-contiguous, and aligned `uint32`. Lists and tuples are rejected in the current source tree; the published 0.2.0 wheel also accepts them, so these array examples work with both.

## Installation

```bash
pip install tokie
```

## Quick start

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("openai-community/gpt2")
# from_json(path) and from_file(path) are aliases for open(path).

ids = tokenizer.encode("Hello world")            # uint32 array, template off
print(ids.tolist())
print(tokenizer.decode(ids))
print(tokenizer.decode_bytes(ids))               # raw bytes

# The single-sequence template. GPT-2 has none, so this matches encode.
print(tokenizer.encode_plus("Hello world").tolist())

flat, ends = tokenizer.batch(["Hello world", "a12b"])  # ends are cumulative
print(tokenizer.count_tokens("Hello world"))           # template off
print(tokenizer.count_tokens_batch(["Hello world", "a12b"]))
print(tokenizer.vocab_size)                            # highest id + 1
print(tokenizer.pad_token_id)                          # [PAD] then <pad>
print(tokenizer.num_special_tokens_to_add(is_pair=False))
tokenizer.save("gpt2.tokie")
```

`encode` leaves the template off. `encode_plus` applies it and accepts `pair`, `truncate`, and `pad`; truncation is from the right, longest sequence first, and padding is fixed on the right. One `max_len` covers both and counts the template ids.

`from_pretrained` opens exactly one `.tokie` file from the repo, whatever its name. A repo with no `.tokie` file is parsed from `tokenizer.json`; several `.tokie` files is an error. There is no runtime `.tkz` fallback: a genuine legacy `TOKI` image is refused by magic, not by extension, while a JSON file merely named `.tkz` is parsed as JSON. `open` reads either file from disk by its magic, `from_json(path)` and `from_file(path)` are path aliases for `open`, and `json(text)` parses JSON text.

`decode` returns `str` (lossy only where needed). `decode_bytes` returns raw bytes. Both take the `uint32` array as their first argument and keep specials by default; `skip=True` omits them. For ids built by hand, create a NumPy array explicitly: `numpy.array(values, dtype=numpy.uint32)`. `get_vocab()` returns raw token strings and may have fewer entries than `vocab_size`.

## Benchmark

`bench.py` times `Tokenizer.encode` on one file against the `tokie bench` command, which is the Rust path.

## License

MIT OR Apache-2.0
