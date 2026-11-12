# tokie

`tokie` loads a Hugging Face `tokenizer.json` or a `.tokie` image and emits token ids. It implements BPE, WordPiece, SentencePiece BPE, and Unigram. The normalizer is [`normalie`](https://crates.io/crates/normalie); the pretokenizer is [`pretokie`](https://crates.io/crates/pretokie). This crate is 0.2.0.

```toml
[dependencies]
tokie = "0.2"
```

The `hub` feature is on by default. `hf` is an alias for it; turn the feature off to build without Hub downloads.

## Load

```rust
use tokie::Tokenizer;

fn load() -> Result<(), tokie::Error> {
    // A path to `tokenizer.json` or to a saved `.tokie` file.
    let _from_path = Tokenizer::open("tokenizer.json")?;

    // JSON text, a Hub repo, and the restored path aliases.
    let text = std::fs::read_to_string("tokenizer.json")
        .map_err(|e| tokie::Error::from(e.to_string()))?;
    let _from_text = Tokenizer::json(&text)?;
    let _from_hub = Tokenizer::pretrained("google-bert/bert-base-uncased")?;
    let _alias = Tokenizer::from_json("tokenizer.json")?;
    Ok(())
}
```

`Tokenizer::open` reads either format by magic. `Tokenizer::json` parses text. `Tokenizer::pretrained` reads one `.tokie` file from the Hub repo, or parses `tokenizer.json` when the repo has none; `from_pretrained` is an alias. `from_json` and `from_file` are path aliases for `open`. `save` and `to_file` write a `.tokie` file.

## Encode

```rust
use tokie::{Specials, TokenId, Tokenizer};

fn example() -> Result<(), tokie::Error> {
    // Replace with your own path.
    let mut tokenizer = Tokenizer::open("tokenizer.json")?;
    let mut ids: Vec<TokenId> = Vec::new();

    tokenizer.encode("Hello, world!", &mut ids)?; // template off
    ids.clear();
    tokenizer.encode_plus("Hello, world!", &mut ids)?; // template on

    let mut text = String::new();
    tokenizer.decode(&ids, &mut text, Specials::Skip)?;
    Ok(())
}
```

Every encode call takes `&mut self`, appends into the `Vec` the caller owns, and returns `Result<(), tokie::Error>`. It never clears the vector; clear it to start a new sequence. `encode` leaves the parsed template off. `encode_plus` applies it. For a pair, a cut, or padding, pass `Plus`:

```rust
tokenizer.encode_plus(
    tokie::Plus::new("Hello").pair("world").truncate(8).pad(8),
    &mut ids,
)?;
```

`truncate` and `pad` share one length, which counts the template ids. Truncation drops ids from the right of the longer sequence first; padding is fixed on the right.

`decode` appends into the caller's `String` and returns `Result`. `Specials::Keep` writes every id; `Specials::Skip` omits specials. Invalid UTF-8 becomes replacement text; an unknown or gap id is an error. `decode_bytes` returns the raw `Vec<u8>` with no lossy conversion.

## Batch and counts

```rust
let texts = ["Hello!", "World"];
let mut ids = Vec::new();
let mut ends: Vec<usize> = Vec::new();
tokenizer.batch(&texts, &mut ids, &mut ends)?;
```

`batch` leaves the template off and writes one contiguous id buffer plus a cumulative `ends` entry per text: text `i` is `ids[prev..ends[i]]`, where `prev` is the previous end. `ends` is not a list of lengths.

`count_tokens(text)` returns the length `encode` would append. `count_tokens_batch(texts)` returns one length per text and calls `batch`. Both return `Result`, leave the template off, and build ids; they are not lazy and not allocation-free.

## Metadata

`TokenId` is `u32` and is exported at the crate root. `vocab_size()` is the highest id plus one, so it includes gaps and added ids. `get_vocab()` maps the raw vocab spelling to id; it can have fewer entries than `vocab_size()`. `token(id)` and `id(token)` also use the raw vocab spelling, so the 0.1.4 names are not aliases for them. `pad_token_id()` looks up `[PAD]`, then `<pad>`, and nothing else. `num_special_tokens_to_add(is_pair)` returns the parsed template count.

## `.tokie` files

`save` stores the normalized tables and the loaded piece list, so opening a `.tokie` file is deserialization. It is not the legacy 0.1.4 `.tkz` format: a genuine legacy `TOKI` image is refused by its magic, whatever its filename, and a JSON file merely named `.tkz` is parsed as JSON. Legacy 0.1.4 binaries cannot read `.tokie`. Rebuild from `tokenizer.json` with 0.2; renaming the file is not enough. A Hub repo with more than one `.tokie` file is an error, and there is no `.tkz` fallback.

## Python

The Python package `tokie` exposes the same calls. `encode` and `encode_plus` return NumPy `uint32` arrays, not `Encoding` objects, so there is no `.ids`; use `.tolist()`. `batch` returns `uint32` ids and `uint64` cumulative ends. See the [API guide](https://github.com/feyninc/tokie/blob/main/docs/API.md) and [migration details](https://github.com/feyninc/tokie/blob/main/docs/MIGRATION.md).

## License

MIT OR Apache-2.0. See [LICENSE-MIT](https://github.com/feyninc/tokie/blob/main/crates/tokie/LICENSE-MIT) and [LICENSE-APACHE](https://github.com/feyninc/tokie/blob/main/crates/tokie/LICENSE-APACHE).
