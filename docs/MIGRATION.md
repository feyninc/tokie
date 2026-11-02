# Migrating from Hugging Face tokenizers

`encode` appends ids and leaves the template off. `encode_plus` writes the single-sequence template around those ids. BERT uncased is `[CLS]`, the ids, `[SEP]`.

## Python

```python
import tokie

tokenizer = tokie.Tokenizer.from_pretrained("google-bert/bert-base-uncased")

ids = tokenizer.encode("Hello, world!")
plus = tokenizer.encode_plus("Hello, world!")
print(tokenizer.decode(plus.tolist()))

flat, ends = tokenizer.batch(["Hello!", "World"])
print(tokenizer.token(101))
print(tokenizer.id("[SEP]"))

tokenizer.save("model.tokie")
again = tokie.Tokenizer.open("model.tokie")
```

`encode` and `encode_plus` return a `uint32` array. `decode` returns a `str`. `decode(ids, skip=True)` omits specials. `ends[i]` is the id count after text `i`.

## Rust

```rust
use tokie::{Specials, Tokenizer};

let mut tokenizer = Tokenizer::pretrained("google-bert/bert-base-uncased")?;
let mut ids = Vec::new();
tokenizer.encode("Hello, world!", &mut ids)?;

ids.clear();
tokenizer.encode_plus("Hello, world!", &mut ids)?;

let mut text = String::new();
tokenizer.decode(&ids, &mut text, Specials::Keep)?;

let lines = ["Hello!", "World"];
let mut ends = Vec::new();
tokenizer.batch(&lines, &mut ids, &mut ends)?;

tokenizer.save("model.tokie")?;
let again = Tokenizer::open("model.tokie")?;
```

`encode`, `encode_plus`, and `batch` append into the caller's `Vec`. `decode` appends into the caller's `String`. `Specials::Skip` omits specials.

## Names

| Hugging Face | tokie |
| --- | --- |
| `Tokenizer.from_pretrained(repo)` | Python `Tokenizer.from_pretrained(repo)`. Rust `Tokenizer::pretrained(repo)`. |
| `encode(text)` with the template on | `encode_plus` |
| `encode(text, add_special_tokens=False)` | `encode` |
| `encode_batch(texts)` | `batch` |
| `decode(ids)` | `decode` |
| `id_to_token(id)` | `token` |
| `token_to_id(token)` | `id` |
| `save(path)` | `save`. The file is `.tokie`. |
| load a path | `open` |
