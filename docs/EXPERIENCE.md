# Experience

A caller holds a `Tokenizer`, points it at a file or a Hub repo, and appends ids. The crate root is that type, the error, and the small enum `decode` takes. The encode loop, the piece cache, and the merge stay as they are.

## Names

A public name is one or two words a caller already uses for the job. `encode`, `decode`, `batch`, `open`, `pretrained`.

A private name can be as long as the code needs. `single_sequence_template` and `split_added_tokens` are the right shape inside the crate. The public name stays the short one.

The root exports the tokenizer. `Cut`, `Norm`, `Cache`, `merge`, `wordpiece`, and `unigram` stay in their modules for a caller who is building a stage. A caller who has a `tokenizer.json` imports `Tokenizer`.

## Call

```rust
let mut tokenizer = Tokenizer::open("tokenizer.json")?;
let mut ids = Vec::new();
tokenizer.encode("Hello world", &mut ids)?;

let mut text = String::new();
tokenizer.decode(&ids, &mut text, Specials::Keep)?;
```

`encode` appends. The caller clears `ids` to start a new sequence. `decode` and `batch` follow the same rule. The bench call writes into a buffer the caller allocated and reuses.

## Load

| Call | Input |
| --- | --- |
| `open` | A path. The file is `tokenizer.json`. |
| `json` | The text of a `tokenizer.json`. |
| `pretrained` | A Hub repo id, such as `"gpt2"`. The file is cached on disk, then parsed by `json`. |

`open` and `json` return `Error` when the pattern or the normalizer is one this crate refuses. `pretrained` also returns `Error` when the repo cannot be fetched. `Error` implements `Display` and `std::error::Error`. The variants stay private.

`pretrained` is the call a new user types. `open` is the call a bench types, because the file is already on disk.

## Encode

```rust
tokenizer.encode(text, &mut ids)?;
tokenizer.wrap(text, &mut ids)?;
```

`encode` emits a special token when its text occurs in the input. The template stays off. `wrap` runs `encode`, then writes the single-sequence template around those ids. BERT uncased is `[CLS]`, the ids, `[SEP]`. A ByteLevel template adds nothing, so `wrap` and `encode` write the same ids for GPT-2.

`encode` is the short call because the speed bench and the id checks use it. `wrap` splices a short prefix and a short suffix after that call returns.

Both take `&mut self`. The piece cache and the scratch live on the tokenizer.

## Decode

```rust
tokenizer.decode(&ids, &mut text, Specials::Keep)?;
tokenizer.decode(&ids, &mut text, Specials::Skip)?;
```

`decode` takes `&self`. It reads an id-to-bytes table built at load and writes into the caller's `String`. `Keep` writes every id. That is the round trip of `encode`. `Skip` omits specials.

An unknown id is `Error`. The table is a byte arena plus an offset and a length per id. The byte-level mapping, the WordPiece `##`, and the Unigram space mark are already applied in that arena. `decode` copies slices. `encode` leaves the table untouched.

```rust
tokenizer.token(464)  // Option<&str>, borrowed from the tokenizer
tokenizer.id("Hello") // Option<u32>
```

`token` and `id` inspect one entry. `decode` is the fast call.

## Batch

```rust
tokenizer.threads(8);
tokenizer.batch(&lines, &mut ids, &mut ends)?;
let worker = tokenizer.fork();
```

`batch` encodes each text with `encode`, in order, on a pool of forked tokenizers. `ends[i]` is `ids.len()` after text `i`. An empty text records the same end as the text before it.

The pool stays on the tokenizer. The next `batch` reuses those caches.

`fork` returns a tokenizer with the same tables and a copy of the loaded piece cache. A JSON open has an empty list, so that copy starts empty. Encode after the fork stays on that tokenizer. The tables sit behind an `Arc`. Each tokenizer owns the cache the hit loop uses. `encode` loads the cut and the cache once per call. Two threads encode by owning one tokenizer each. A `Tokenizer` is `Send`.

`threads` sets the pool size. `1` runs `batch` on the calling thread, for a worker that already has a thread. The default is the machine's available parallelism.

On `Error`, `ids` and `ends` hold the texts that finished. The caller truncates back to the last end they still want.

## Root

```rust
pub use tokenizer::{Tokenizer, Error, Specials};
```

`Specials` is `Keep` and `Skip`.

These leave the root and keep the names they have now, one module in:

| Module | What a stage caller uses |
| --- | --- |
| `bpe` | `merge` |
| `wordpiece` | `wordpiece` |
| `unigram` | `Piece`, `unigram` |
| `cache` | `Cache` |
| `cut` | `Cut` |
| `norm` | `Norm`, `apply`, `split` |

## Speed

The single-string encode path stays in place.

- `wrap` calls `encode`, then splices the template.
- `decode`, `token`, and `id` read the load-time table.
- `pretrained` downloads, then calls `json`.
- `batch` calls `encode` on forked tokenizers.
- `fork` shares the tables and copies the loaded piece cache. The hit loop runs on the cache that tokenizer owns.

The piece cache, the pretokie walk, and the merge loop stay as they are. Pretokie and normalie stay their own crates. Importing them is for work on those crates.

## Later

These names stay free for the calls that come after this API. The open work is listed in [GOAL.md](GOAL.md).

| Name | Job |
| --- | --- |
| `pair` | Two sequences and the pair template |
| `pad` | Pad a batch to a width |
| `truncate` | Cut ids to a length |
| `offsets` | Byte offsets for each id |

`open` reads a `.tokie` file or `tokenizer.json`. `save` writes the `.tokie` file. The layout is in [FORMAT.md](FORMAT.md).
