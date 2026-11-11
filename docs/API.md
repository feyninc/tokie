# API

tokie 0.2 has two layers. `encode`, `batch`, `decode`, `count_tokens`, and the vocabulary calls are the fast core. They append into buffers the caller owns, or return a small result. `encode_plus` is the rich call: it applies the parsed template, optionally encodes a pair, truncates from the right, and pads on the right. The 0.1.4 facade is not back. [API_CHANGES_0_2.md](API_CHANGES_0_2.md) lists every restored name, intentional difference, and removed surface. There is no full-compatibility guarantee.

## Template

`encode` leaves the template off. `encode_plus` applies it. 0.1.4's Rust `encode(text, add_special_tokens: bool)` made the choice explicit, while 0.1.4's Python `encode(text, add_special_tokens=True)` defaulted the template on. 0.2 keeps no bool in either language: `encode` is always template-off, and `encode_plus` is always template-on. Python callers that relied on the old default must switch to `encode_plus`. `encode_plus` also takes an optional pair, right truncation, and right fixed padding. One `max_len` covers the cut and the pad and counts the template ids.

## Names

| 0.1.4 (published) | 0.2 target | Notes |
| --- | --- | --- |
| `encode(text, add_special_tokens=False)` | `encode` | Appends ids; template off. |
| `encode(text, add_special_tokens=True)` | `encode_plus` | `encode`, then the parsed template. `BertProcessing` and `RobertaProcessing` add no ids. |
| `encode_pair`, or Python `tokenizer(text, text_pair=...)` | `encode_plus` with `pair` | The pair template. |
| `encode` with padding or truncation | `encode_plus` with `truncate` / `pad` | Right truncation, longest sequence first. Right fixed padding. |
| `encode_ids` | The same append call | The caller passes the `Vec` (`&mut Vec<TokenId>` in Rust). |
| `encode_batch`, `encode_batch_flat` | `batch` | One id buffer and one cumulative `ends` entry per text. |
| `count_tokens` | `count_tokens` | Encode into tokenizer-owned scratch and return the length. Builds the ids; not allocation-free or early-exit. Template off in both releases. |
| `count_tokens_batch` | `count_tokens_batch` | `batch` into that buffer and return one length per text. Template off. |
| `decode` | `decode` | Python returns `str` and stays lossy. Rust appends into a `String`. |
| `decode_bytes` | `decode_bytes` | Raw bytes; no lossy UTF-8. `skip` omits specials. |
| `id_to_token` / `token_to_id` | `token` / `id` | Renamed, not aliased. 0.1.4 lookups used the decoder spelling; new `token`/`id` use the raw vocab spelling, so the keys are not interchangeable. |
| `get_vocab` | `get_vocab` | Raw vocab spelling, consistent with `token`/`id`. The 0.1.4 decoder keys no longer apply. The entry count may be smaller than `vocab_size`. |
| `vocab_size` | `vocab_size` | Highest id + 1, including gaps and added ids. |
| `pad_token_id` | `pad_token_id` | `[PAD]`, then `<pad>`, and nothing else. |
| `num_special_tokens_to_add` | `num_special_tokens_to_add` | Parsed template added count. `is_pair=False` in Python, a `bool` in Rust. `0` when the file has no pair template. |
| `from_json(path)` | `open(path)` | Path loading. `from_json(path)` and `from_file(path)` stay as path aliases. `json(text)` parses text. |
| `from_file(path)` | `open(path)` | No `.tkz` format. A genuine legacy `TOKI` image is refused by magic, not by extension. See [FORMAT.md](FORMAT.md). |
| `from_pretrained` | Python `from_pretrained`, Rust `pretrained` | Rust also aliases `from_pretrained`; Python keeps the old name. |
| Python `save(path)`; Rust `to_file(path)` / `save(writer)` | Python and Rust `save(path)`; Rust `to_file(path)` alias | `.tokie`; writer form not revived. |

### Not provided

The following are deliberately absent, not hidden behind a compatibility switch: `Encoding`, a `compat` namespace, a second `Tokenizer`, `__call__`, `encode_pair`, `encode_with_offsets`, `encode_bytes`, `decode_batch`, `token_count`, `encode_files`, `count_tokens_files`, `id_to_token`/`token_to_id` aliases, configurable padding and truncation (`enable_padding`, `enable_truncation`, `no_padding`, `no_truncation`, `padding`, `truncation`, `set_pad_token_id`), attention masks, type ids, offsets, typed legacy error classes, and genuine legacy `TOKI` images. The CLI has no `convert`, `verify`, or `cache` command, and there is no `build` feature.

## Rust

`TokenId` is `u32` and is exported from the crate root. The caller owns the buffers. `encode`, `encode_plus`, and `batch` append and never clear. `decode` appends into a `String`; `decode_bytes` returns bytes. These are methods on `Tokenizer`:

```rust
use tokie::{Specials, TokenId, Tokenizer};

pub fn encode(&mut self, text: &str, ids: &mut Vec<TokenId>) -> Result<(), Error>;
pub fn encode_plus<'a>(&mut self, text: impl EncodePlus<'a>, ids: &mut Vec<TokenId>) -> Result<(), Error>;
pub fn batch(&mut self, texts: &[&str], ids: &mut Vec<TokenId>, ends: &mut Vec<usize>) -> Result<(), Error>;
pub fn decode(&self, ids: &[TokenId], text: &mut String, specials: Specials) -> Result<(), Error>;
pub fn decode_bytes(&self, ids: &[TokenId], specials: Specials) -> Result<Vec<u8>, Error>;
pub fn count_tokens(&mut self, text: &str) -> Result<usize, Error>;
pub fn count_tokens_batch(&mut self, texts: &[&str]) -> Result<Vec<usize>, Error>;

pub fn open(path: impl AsRef<Path>) -> Result<Self, Error>;        // from_json / from_file aliases
pub fn json(text: &str) -> Result<Self, Error>;                    // parse text
pub fn pretrained(repo: &str) -> Result<Self, Error>;              // from_pretrained alias
pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Error>;   // to_file alias
```

`encode_plus(text, ids)` is the single-sequence template. `encode_plus(Plus::new(text).pair(other).truncate(n).pad(n), ids)` adds the rest. `batch` writes cumulative `ends`: text `i` is `ids[prev..ends[i]]`.

The metadata calls are `vocab_size()`, `get_vocab()`, `pad_token_id()`, `num_special_tokens_to_add(is_pair)`, `token(id)`, and `id(token)`. `get_vocab` returns the raw token strings. `vocab_size` is the highest id plus one, so it can exceed the number of entries in `get_vocab`.

## Python

`encode` and `encode_plus` return a NumPy `uint32` array. `batch` returns a `uint32` ids array and a `uint64` cumulative `ends` array. There is no `Encoding` object and no `.ids`; use `.tolist()`. `decode` returns `str`; `decode_bytes` returns `bytes`.

```python
tokenizer = tokie.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
# from_json(path) and from_file(path) are aliases for open(path).

tokenizer.encode("Hello")                  # uint32 array, template off
tokenizer.encode_plus("Hello")             # template on
tokenizer.encode_plus("Hello", pair="world", truncate=True, pad=True, max_len=8)
flat, ends = tokenizer.batch(["Hello", "world"])  # template off; ends are cumulative
tokenizer.decode(flat[: ends[0]].tolist())        # str, lossy if needed
tokenizer.decode_bytes(flat[: ends[0]].tolist(), skip=True)  # raw bytes

tokenizer.vocab_size
tokenizer.get_vocab()                      # raw token -> id; may be smaller than vocab_size
tokenizer.pad_token_id                     # [PAD] then <pad>, else None
tokenizer.num_special_tokens_to_add(is_pair=False)
```

`encode` and `encode_plus` release the GIL. `batch` does too. `threads(n)` sets the worker count. `count_tokens` and `count_tokens_batch` write into reusable scratch on the tokenizer; they build the ids and stay template-off in both releases. Neither is allocation-free or an early-exit counter.

## Loading

`open(path)` reads a `.tokie` image or a `tokenizer.json` by its magic. `json(text)` parses JSON text. `from_json(path)` and `from_file(path)` are path aliases for `open`, not alternate file formats. `save` / `to_file` writes `.tokie`.

`from_pretrained` in Python and `pretrained` in Rust open exactly one `.tokie` file from the Hub repo, whatever its name. A repo with no `.tokie` file parses `tokenizer.json`. Several `.tokie` files is an error. There is no `.tkz` fallback. A genuine legacy `TOKI` image is refused by magic, whatever its filename; a JSON file merely named `.tkz` is parsed as JSON. Legacy 0.1.4 binaries cannot read a `.tokie` file. Rebuild from `tokenizer.json` with the new API; renaming a `.tkz` file is not enough.

The Cargo feature is `hub`; `hf` is an alias.

## Speed

These are design expectations for the 0.2 shape, not measurements. [BENCHMARK_BASELINE.md](BENCHMARK_BASELINE.md) records the before-state. The after-implementation gate is pending.

| Change | Expected effect |
| --- | --- |
| Leave the body of `encode` as it is | No regression. The bench calls it. |
| `encode_plus` calls `encode`, then the template | GPT-2's template is empty, so the id slide is skipped. |
| Put the template branch inside `encode` | A branch on every call, including template off. |
| `batch` keeps its body | Borrows the texts and keeps the worker caches. |
| `encode_batch` returns one `Encoding` per text | Copies every document and stitches the id buffers. That is the regression `batch` removed. |
| Python takes `Vec<String>` | Copies every document before a worker starts. |
| One NumPy array per text | An object per document and a second pass over the ids. |
| Lengths from `ends` after `batch` returns | One subtraction per document. Not inside the worker. |
| A second id buffer inside the worker | Another copy of every id. The worker already copies once into its region. |
| `count_tokens` | Same work as `encode`, then the length. Not faster. |
| A new `Vec` on every count | Slower than `encode`. Reuse one buffer on the tokenizer. |
| `count_tokens_batch` calls `batch` | Worker caches stay warm. The lengths are one small allocation. |
| Attention mask and type ids on `encode_plus` | Two more buffers the length of the ids. Not provided. |
| Truncation on `encode_plus` | Cuts the longer sequence after `encode` returns. The bench still calls `encode`. |
| Padding on `encode_plus` | Appends the pad id onto the ids this call wrote. |
| Pair encoding on `encode_plus` | `encode` runs twice, then the pair template. |

## Remaining

Offsets, attention masks, and type ids are still out of scope. They stay on the [GOAL.md](GOAL.md) Remaining list.
