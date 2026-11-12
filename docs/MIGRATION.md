# Migrating from Hugging Face `tokenizers`

> [!NOTE]
> This guide targets Python 0.3.0 and Rust 0.2.0. Python 0.3.0 requires a one-dimensional native `uint32` NumPy array for `decode` and `decode_bytes`; Python 0.2.0 also accepted lists. See [the decode input note](RELEASE_NOTE_DECODE_UINT32.md) and the [Python 0.3.0 release notes](PYTHON_RELEASE_0_3.md).

`encode` appends ids and leaves the template off. `encode_plus` applies the template. Hugging Face `encode` adds the template by default, so use `encode_plus` where you relied on that. For GPT-2 the template is empty, so `encode` and `encode_plus` match and both take the same text strings. Coming from tokie 0.2.0 Python, `encode`, `encode_plus`, `batch`, and the vocabulary calls are unchanged; only the `decode` input changed. Coming from tokie 0.1.4 rather than Hugging Face? See [API.md](API.md) and [API_CHANGES_0_2.md](API_CHANGES_0_2.md).

## Python

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
# from_json(path) and from_file(path) load a path, like open(path).

ids = tokenizer.encode("Hello, world!")        # template off
plus = tokenizer.encode_plus("Hello, world!")  # [CLS], ids, [SEP] for BERT uncased
print(tokenizer.decode(plus, skip=True))
print(tokenizer.decode_bytes(plus, skip=True))

flat, ends = tokenizer.batch(["Hello!", "World"])
print(tokenizer.token(101))
print(tokenizer.id("[SEP]"))
print(tokenizer.vocab_size)
print(tokenizer.num_special_tokens_to_add(is_pair=False))

tokenizer.save("model.tokie")
again = tokie.Tokenizer.open("model.tokie")
```

`encode` and `encode_plus` return a `uint32` array. `batch` returns `(uint32 ids, uint64 ends)`. There is no `Encoding` and no `.ids`; use `.tolist()` only to print or assert. `decode` and `decode_bytes` take a one-dimensional native-endian NumPy `uint32` array that is C-contiguous and aligned; pass `plus` or `flat[start:end]` directly. `decode` returns a `str` and is lossy when the bytes are not UTF-8. `decode_bytes` returns raw bytes. Both keep specials by default; `skip=True` omits them. `ends[i]` is the cumulative id count after text `i`.

Hugging Face `tokenizers` `encode` returns an `Encoding` object, not a list; its `.ids` is a Python `list[int]`. To reuse those ids with tokie, convert the ids list once: `tokie_ids = numpy.asarray(hf_tokenizer.encode(text).ids, dtype=numpy.uint32)`. That allocates a fresh aligned, C-contiguous array, so it is already valid decode input. For an arbitrary existing array, copy explicitly with `numpy.array(values, dtype=numpy.uint32, copy=True, order="C")`; `numpy.ascontiguousarray` alone leaves an already C-contiguous but misaligned buffer unaligned. Lists, tuples, bytes, memoryview, other dtypes, other ranks, and foreign byte order raise `TypeError`; non-contiguous or unaligned non-empty arrays raise `ValueError`. Empty and read-only arrays are accepted. Gap and unknown ids, including `2**32 - 1`, still raise `tokie.TokieError`.

`count_tokens` and `count_tokens_batch` measure `encode`, not `encode_plus`. The template is excluded in both 0.1.4 and 0.2.

## Rust

```rust
use tokie::{Specials, TokenId, Tokenizer};

let mut tokenizer = Tokenizer::pretrained("google-bert/bert-base-uncased")?;
let mut ids: Vec<TokenId> = Vec::new();
tokenizer.encode("Hello, world!", &mut ids)?;

ids.clear();
tokenizer.encode_plus("Hello, world!", &mut ids)?;

let mut text = String::new();
tokenizer.decode(&ids, &mut text, Specials::Skip)?;
let raw: Vec<u8> = tokenizer.decode_bytes(&ids, Specials::Skip)?;

ids.clear();
let lines = ["Hello!", "World"];
let mut ends = Vec::new();
tokenizer.batch(&lines, &mut ids, &mut ends)?;

tokenizer.save("model.tokie")?;
let again = Tokenizer::open("model.tokie")?;
```

`encode`, `encode_plus`, and `batch` append into the caller's `Vec`. `decode` appends into the caller's `String`; `decode_bytes` returns a byte vector. `from_json` and `from_file` are aliases for `open`; `from_pretrained` is an alias for `pretrained`; `to_file` is an alias for `save`.

## Names

| Hugging Face | tokie |
| --- | --- |
| `Tokenizer.from_pretrained(repo)` | Python `from_pretrained(repo)`. Rust `pretrained(repo)` or `from_pretrained(repo)`. |
| `encode(text)` with the template on | `encode_plus`. A parsed template. `BertProcessing` and `RobertaProcessing` add no ids. |
| `encode(text, text_pair)` | `encode_plus` with `pair`. The pair template, or an error when the file has none. |
| truncation or padding | `truncate` and `pad` on the same call. Right truncation, longest sequence first. Right fixed padding. One `max_len` covers both and counts the template ids. |
| `encode(text, add_special_tokens=False)` | `encode` |
| `encode_batch(texts)` | `batch`. The template stays off. Call `encode_plus` per text when the template is required. |
| `decode(ids)` | Python `decode(ids_array)` with a native `uint32` NumPy array; Rust `decode` with `Specials::Keep`. Use `skip=True` / `Specials::Skip` to omit specials. |
| `decode` to raw bytes | `decode_bytes`. Raw bytes, not lossy UTF-8. |
| `id_to_token(id)` | `token`, returning the raw vocab spelling. 0.1.4 used the old decoder spelling; the keys are not interchangeable, so this is not an alias. |
| `token_to_id(token)` | `id`, using the raw vocab spelling. |
| `get_vocab()` | `get_vocab()`. Raw token strings. The entry count may be smaller than `vocab_size`. |
| `vocab_size` | `vocab_size`. Highest id + 1, including gaps and added ids. |
| pad token | `pad_token_id`. `[PAD]` then `<pad>`, and nothing else. |
| special token count | `num_special_tokens_to_add(is_pair=False)`. `0` when the file has no pair template. |
| `save(path)` | `save` / `to_file`. The file is `.tokie`. |
| load a path | `open`. `from_json` and `from_file` are path aliases. |

## Loading and file format

`from_pretrained` (Python) and `pretrained` (Rust) open exactly one `.tokie` file from the repo, whatever its name. A repo with none is parsed from `tokenizer.json`. Several `.tokie` files is an error. There is no `.tkz` fallback.

A genuine legacy `TOKI` `.tkz` image is refused by its magic, whatever its filename; a JSON file only named `.tkz` is parsed as JSON. Legacy 0.1.4 binaries cannot read a `.tokie` file. Rebuild from `tokenizer.json` with the 0.2 code; renaming the file is not enough.
