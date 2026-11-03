# API

0.2 appends ids into a buffer the caller owns. tokie 0.1.4 returned an owned `Encoding` from `encode` and `encode_batch`, and counted with `count_tokens`. This conversion puts those names on the 0.2 engine. It does not bring back the `Encoding` struct, padding, truncation, offsets, or pair encoding. Those stay in [GOAL.md](GOAL.md).

`encode_plus` is `encode`, then the template. A call that passes only the text stays `encode`, so the template stays off. 0.1.4 turned the template on by default inside `encode`. `Plus` on `encode_plus` adds the second sequence, cuts to a length, or pads to that length. Offsets, attention masks, and type ids stay in [GOAL.md](GOAL.md).

## Names

| 0.1.4 | 0.2 |
| --- | --- |
| `encode(text, add_special_tokens=False)` | `encode`. The template stays off. |
| `encode(text, add_special_tokens=True)` | `encode_plus`. `encode`, then a `TemplateProcessing` single sequence. `BertProcessing` and `RobertaProcessing` add no ids. |
| `encode(text, text_pair)` | `encode_plus(Plus::new(text).pair(text_pair))`. The pair template. |
| `encode` with truncation or padding | `encode_plus(Plus::new(text).truncate(n).pad(n))`. One length for the cut and the pad. |
| `encode_ids` | The same append call. The caller passes the `Vec`. |
| `encode_batch` | `batch`. One name. One id buffer and one `ends` entry per text. |
| `encode_batch_flat` | `batch`. 0.1.4 returned per-document lengths. 0.2 `ends[i]` is the id count after text `i`. |
| `count_tokens` | `count_tokens`. Encode into a tokenizer-owned buffer and return the length. No template. |
| `count_tokens_batch` | `count_tokens_batch`. `batch` into that buffer and return one length per text. |
| `decode` | Already appends into a `String`. Python keeps returning a `str`. |
| `id_to_token` / `token_to_id` | `token` / `id`. |
| `from_file` | `open`. |
| `from_pretrained` | `pretrained`. |
| `from_json(path)` | `open` for a path. `json` parses the text of a file. 0.1.4's `from_json` took a path. |
| `save` | Already writes a `.tokie` file. |

`encode_files`, `count_tokens_files`, `encode_bytes`, `decode_bytes`, `decode_batch`, and `token_count` are not in this pass. `token_count` stopped early. `count_tokens` builds the ids and drops them, which is what 0.1.4 did.

## Speed

| Change | Effect |
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
| Count into `scratch.piece` or the caller's ids | Corrupts the merge scratch, or clears ids the caller was appending. |
| `count_tokens_batch` calls `batch` | Worker caches stay warm. The lengths are one small allocation. |
| `threads` or a fresh `fork` inside count | Drops the warm workers. The next `batch` starts cold. |
| `token`, `id`, `open` | No id movement. |
| `open`, `json`, `pretrained`, `save` | Load and store. Outside the encode timing. |
| Attention mask and type ids on `encode_plus` | Two more buffers the length of the ids. Left for later. |
| Truncation on `encode_plus` | Cuts the longer sequence after `encode` returns. The bench still calls `encode`. |
| Padding on `encode_plus` | Appends the pad id onto the ids this call wrote. |
| Pair encoding on `encode_plus` | `encode` runs twice, then the pair template. |

## Rust

`encode`, `batch`, and `decode` keep their signatures. Benches and the piece cache call `encode`. `encode_plus` appends into the same buffer. The batch call is `batch`. The vocab calls are `token` and `id`. The path call is `open`.

```rust
pub fn encode_plus<'a>(&mut self, text: impl EncodePlus<'a>, ids: &mut Vec<u32>) -> Result<(), Error>;
// encode_plus(text, ids) is the single-sequence template.
// encode_plus(Plus::new(text).pair(other).truncate(n).pad(n), ids) adds the rest.
pub fn count_tokens(&mut self, text: &str) -> Result<usize, Error>;
pub fn count_tokens_batch(&mut self, texts: &[&str]) -> Result<Vec<usize>, Error>;
```

`count_tokens` writes into the tokenizer's scratch and returns that length. `count_tokens_batch` does the same with `batch`, then turns cumulative `ends` into per-text lengths. Neither applies the template. A second call may reuse the scratch.

## Python

`encode` stays the ids of the text. `encode_plus` adds the template and still returns a `uint32` array.

```python
tokenizer.encode("Hello")                  # template off
tokenizer.encode_plus("Hello")             # [CLS], ids, [SEP] when the template says so
tokenizer.encode_plus("Hello", pair="world", truncate=True, pad=True, max_len=8)
tokenizer.batch(["Hello", "world"])         # (ids, ends), template off
tokenizer.count_tokens("Hello")            # length of encode, template off
tokenizer.count_tokens_batch(["Hello", "world"])
```

`batch` returns one id array and one ends array. `token` and `id` are the vocab calls. `open` reads a path.

## Order

1. `encode_plus` is in. `batch`, `token`, `id`, and `open` stay the only names for those jobs.
2. The count methods are on the tokenizer. A test checks `count_tokens` against `encode` with an empty buffer, and `count_tokens_batch` against `batch` ends. The count buffer is a field on the tokenizer.
3. Truncation, padding, and pair encoding are on `encode_plus`. Offsets, attention masks, and type ids stay on the Remaining list.
