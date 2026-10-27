<p align="center">
  <img src="https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie-banner.png" alt="tokie">
</p>

<p align="center">
  <a href="https://github.com/chonkie-inc/tokie">GitHub</a> · <a href="https://crates.io/crates/tokie">crates.io</a>
</p>

Python bindings for the tokie 0.2 tokenizer. `encode` and `batch` release the GIL and return NumPy arrays.

## Installation

```bash
pip install tokie
```

## Quick start

```python
import tokie

tokenizer = tokie.Tokenizer.pretrained("openai-community/gpt2")
ids = tokenizer.encode("Hello world")
print(ids.tolist())
print(tokenizer.decode(ids.tolist()))

# The single-sequence template. GPT-2 has none, so this matches encode.
print(tokenizer.wrap("Hello world").tolist())

flat, ends = tokenizer.batch(["Hello world", "a12b"])
tokenizer.save("gpt2.tokie")
```

`open` reads a `.tokie` file or a `tokenizer.json`. `json` parses the file contents. `decode(..., skip=True)` omits specials.

## Benchmark

`bench.py` times `Tokenizer.encode` on one file against the `tokie bench` command, which is the Rust path.

## License

MIT OR Apache-2.0
