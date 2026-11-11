# API changes from tokie 0.1.4 to 0.2

**Published 0.1.4:** commit `1a378c8` (crates.io and PyPI).
**0.2 target:** commit `c59c8c8` plus the `bhavnick/api-014-compat` branch. It is unreleased at the time of writing.

0.2 keeps the useful mainline names, but it is not a compatibility facade for 0.1.4: dead, conflicting, and facade-only surfaces are deliberately removed. There is **no full compatibility guarantee**. Code written against 0.1.4 must be checked against these tables.

## Kept or restored mainline names

| Job | 0.1.4 published | 0.2 target |
| --- | --- | --- |
| Load a JSON path | `from_json(path)` | `open(path)`; `from_json(path)` and `from_file(path)` are path aliases |
| Parse JSON text | Rust `hf::from_json_str(text)`; Python had no text constructor | `json(text)` in Rust and Python |
| Load from the Hub | `from_pretrained(repo)` | Python `from_pretrained(repo)`; Rust `pretrained(repo)`, with `from_pretrained` alias |
| Save | Python `save(path)`; Rust `to_file(path)` and `save(writer)` | Python `save(path)`; Rust `save(path)`, with `to_file(path)` alias. The Rust writer form is not revived. |
| Encode without the template | `encode(text, false)` / `encode(text, add_special_tokens=False)` | `encode` |
| Encode with the template | `encode(text, true)` / Python default | `encode_plus` |
| Encode a pair | Rust `encode_pair(a, b, add_special_tokens)`; Python `encode_pair(...)` or callable `tokenizer(text, text_pair=...)` | `encode_plus` with `pair` |
| Truncate or pad | `enable_truncation` / `enable_padding` config | `truncate` / `pad` on the `encode_plus` call |
| Batch encode | `encode_batch`, `encode_batch_flat` | `batch` |
| Count tokens (template off) | `count_tokens`, `count_tokens_batch` | Same names. Both build ids; 0.2 reuses tokenizer scratch. Neither is allocation-free or an early-exit counter. |
| Token/id lookup | `id_to_token`, `token_to_id` | `token`, `id` (renamed, not aliased) |
| Vocabulary | `get_vocab`, `vocab_size` | Same names; raw spelling in `get_vocab` |
| Decode to text | `decode` | `decode`; Python returns `str`, Rust appends to `String` |
| Decode to bytes | `decode_bytes` | `decode_bytes`; raw bytes |
| Pad token | `pad_token_id` | `pad_token_id`: `[PAD]`, then `<pad>` |
| Template special count | `num_special_tokens_to_add(is_pair)` | Same call; parsed template count, `0` when no pair |
| Hub Cargo feature | `hf` | `hf` alias for `hub` |

## Intentionally changed shapes and semantics

| Area | 0.1.4 | 0.2 target | Why it differs |
| --- | --- | --- | --- |
| Rust encode result | `encode(text, add_special_tokens)` returns an owned `Encoding` | `encode(text, &mut ids)` appends and returns `Result<(), Error>` | Keeps the fast core allocation-free and caller-owned. |
| Python encode result | `Encoding` object with `.ids`, `.tokens`, masks, offsets | NumPy `uint32` array; no `Encoding`, no `.ids` | The array is the result; use `.tolist()`. |
| Python template default | `encode(text)` defaulted `add_special_tokens=True` | `encode(text)` always leaves the template off; use `encode_plus` | Rust 0.1.4 had an explicit bool, Python did not. 0.2 makes the split explicit in both. |
| Batch result | List of `Encoding` objects (`encode_batch`) or flat ids plus per-document lengths | One contiguous id array plus cumulative `ends` | `ends[i]` is the id count after text `i`; text `i` is `ids[prev..ends[i]]`. |
| Python batch result | Python list of lists or an object per document | `(uint32 ids, uint64 ends)` NumPy arrays | No per-document Python objects. |
| `count_tokens` | Template off | Template off | Unchanged: the template count was not included in 0.1.4 and is not included in 0.2. Use `encode_plus` and take its length for the template count. |
| `get_vocab` keys | 0.1.4 decoder spelling | Raw vocab spelling, consistent with `token`/`id` | Old keys matched the old decoder spelling and did round-trip through the old lookups. They no longer correspond to the new raw `token`/`id`, so no alias is provided. |
| `vocab_size` | Vocabulary size | Highest id + 1, including gaps and added ids | `get_vocab` may have fewer entries than `vocab_size`. |
| `pad_token_id` | Configurable pad parameters | `[PAD]`, then `<pad>`, and nothing else | `encode_plus` pad uses this id; there is no padding configuration. |
| Truncation | Configurable direction and strategy | Right truncation, longest sequence first | Per-call `truncate(max_len)`; the template ids count toward the length. |
| Padding | Configurable strategy, direction, pad id, multiple-of | Right fixed padding to `max_len` | Per-call `pad(max_len)`; the template ids count toward the length. |
| `decode_bytes` | Raw-byte decode existed, with no skip option | Raw bytes; explicit `Specials::Keep`/`Skip` (Python `skip=False`); same lead slice and lead-space strip as `decode`; unknown or gap ids error | The byte path is not lossy. Changed specials and invalid-id behavior is separate; `decode` remains the lossy text call. |
| `decode` result | Rust `Option<String>`, Python `Optional[str]`; invalid UTF-8 returned `None` | Rust appends into a caller `String` and returns `Result`; Python returns `str` with replacement | Use `decode_bytes` when raw bytes matter. |
| `num_special_tokens_to_add` | Reported the configured processor | Parsed template added count; `0` when the file has no pair template | Derived from the loaded file, not from mutable configuration. |
| `TokenId` | `u32` | `u32`, exported as `tokie::TokenId` at the crate root | One id type. |

## Deliberately removed

These surfaces are gone. They are not behind a feature flag, a `compat` namespace, or a second `Tokenizer`:

- `Encoding` and its `.ids`, `.tokens`, `special_tokens_mask`, `attention_mask`, `type_ids`, and `offsets`.
- Python `__call__`, `encode_pair`, `encode_with_offsets`, `encode_bytes`, and `decode_batch`; the Rust `Option<String>` and Python `Optional[str]` decode returns.
- `id_to_token` / `token_to_id` aliases. Use `token` / `id`, which use the raw vocab spelling.
- Old batch names: `encode_batch` and `encode_batch_flat`. Use `batch`.
- File-corpus APIs: `encode_files`, `count_tokens_files`, and their separator conventions.
- Configurable padding and truncation: `enable_padding`, `enable_truncation`, `no_padding`, `no_truncation`, `padding`, `truncation`, `set_pad_token_id`.
- `token_count` lazy counting. Use `count_tokens` or `encode`.
- Attention masks, type ids, and offsets on `encode_plus`. They remain out of scope in [GOAL.md](GOAL.md).
- Typed legacy error classes. Load, encode, and decode failures are one `tokie::Error` in Rust and `tokie.TokieError` in Python.
- Genuine legacy `TOKI` `.tkz` images. They are refused by magic; a JSON file merely named `.tkz` still parses as JSON. Load `.tokie` or `tokenizer.json`.
- CLI `convert`, `verify`, `cache`, and `build` commands/features. The 0.2 CLI provides `encode`, `ids`, `decode`, `save`, and `bench`.

## Companion and low-level gap

- `normalie` is a standalone crate (0.1.0). It is not re-exported through an old tokie normalizer API.
- `pretokie` replaces the old iterator/config exports with zero-sized `Gpt2`-style walk/fill calls over `Span` and `u32` normalized-byte offsets. It uses the compiled scheme windows and no runtime regex.
- The old tokie encoder, decoder, post-processor, component construction, options structs, read/write/stream forms, and iterators are not revived.

## Rust example

```rust
use tokie::{Specials, TokenId, Tokenizer};

fn main() -> Result<(), tokie::Error> {
    let mut tok = Tokenizer::open("tokenizer.json")?; // from_json / from_file aliases
    let mut ids: Vec<TokenId> = Vec::new();

    tok.encode("Hello", &mut ids)?;           // template off
    ids.clear();
    tok.encode_plus("Hello", &mut ids)?;      // template on

    let raw: Vec<u8> = tok.decode_bytes(&ids, Specials::Skip)?;
    let _ = raw;

    ids.clear();
    let mut ends = Vec::new();
    tok.batch(&["Hello", "world"], &mut ids, &mut ends)?;

    let n = tok.num_special_tokens_to_add(false);
    let size = tok.vocab_size();
    let _ = (n, size);

    tok.save("model.tokie")?;                 // to_file alias
    Ok(())
}
```

## Python example

```python
import tokie

t = tokie.Tokenizer.from_pretrained("google-bert/bert-base-uncased")
# from_json(path) and from_file(path) are aliases for open(path).

ids = t.encode("Hello")                       # uint32 array, template off
plus = t.encode_plus("Hello")                 # template on
t.encode_plus("Hello", pair="world", truncate=True, pad=True, max_len=8)

flat, ends = t.batch(["Hello", "world"])      # template off; ends cumulative
raw = t.decode_bytes(plus.tolist(), skip=True)  # bytes, not lossy text
print(t.decode(plus.tolist(), skip=True))     # str
print(t.vocab_size, t.pad_token_id)
print(t.get_vocab())                          # raw token -> id
print(t.num_special_tokens_to_add(is_pair=False))
t.save("model.tokie")
```

## Alias caveats

- `from_json(path)` and `from_file(path)` are **path** aliases for `open`. They do not accept JSON text and they do not select a different format. Use `json(text)` for text.
- Python `from_pretrained` is retained. Rust's canonical name is `pretrained`; `from_pretrained` is only an alias.
- `to_file` writes a `.tokie` file, not a legacy `.tkz` image.
- `from_file` no longer means `.tkz`. A genuine legacy `TOKI` image is refused by its magic, whatever its filename; a JSON file only named `.tkz` is parsed as JSON. Legacy 0.1.4 binaries cannot read `.tokie`; rebuild from `tokenizer.json` with 0.2 code. Renaming the file is not enough.
- `token` / `id` and `get_vocab` use raw vocab spelling. The 0.1.4 lookups used the old decoder spelling (internally consistent in that spelling), so those keys no longer correspond to the new raw `token`/`id`; that is why no old-name alias is provided.
- The Cargo feature is `hub`; `hf` is an alias for it.
- Python returns NumPy `uint32` ids and `uint64` ends. Do not expect lists or `.ids`.

## Verification and performance status

Implementation and correctness are complete. The strict performance criterion is **UNCERTIFIED/partial**; there is no release sign-off.

- API scope: mainline only. Rust `TokenId`, path aliases, `from_pretrained`/`to_file`; Python `from_json`/`from_file`; both `vocab_size`, raw `get_vocab`, pad getter, template count, and `decode_bytes`; Cargo `hf` alias. NumPy stubs and metadata tests are truthful. No legacy facade or dead-shape support; old conflicting `encode`/`batch`/`decode` names are the new APIs. Fast bodies are unchanged; the new `get_vocab` copy releases the GIL. Restored source hashes `tokenizer.rs` `26b9336d…` and `lib.rs` `e754db1c…` match the final tested artifacts. The API relocation experiment is retired with no performance claim.
- Correctness: **50 Tokie** (26 lib + 12 API + 8 pipeline + 4 file), **97 Pretokie**, **28 Normalie**, **16 normalize oracles**, **1 CLI**, **4 tokenizer oracles**, **1 ids** test (13 checked + 2 existing exclusions: SmolLM ref unsupported, Bert-cased missing), and **37 Python** tests with zero skips. No source edits after tests; no commits. Logs: `/tmp/opencode/tokie-compat/verification-rust/`, `/tmp/opencode/tokie-compat/verification-python/parent-final-pytest.log`.
- Normalie `BertUncased` keeps the three unassigned Cn points U+3040/U+3097/U+3098 and still drops actual Mn U+3099/U+309A. The historical normalization harness's tokie 0.1.4 registry pin is oracle-only, not a production compatibility facade.
- Rust quiet window: A/A 64 pairs passed, exact CI ratio **[0.989654, 1.002035]**; A/B 64 pairs median after/before **1.000438**, CI **[0.998352, 1.005683]**, one-sided p **0.266154**, upper **1.005576**. No detected slowdown at finite precision; this is **not equivalence or proof of zero change**.
- Python final matrix: **27 configs × 16 pairs × 2 sides = 864 samples**, all completed with identical IDs/checksums and `source_guard_unchanged=true`. Overall **`potential_slowdown`**; raw flag **`o200k:batch:1`** median **0.9926726** (~0.733% slower), exact CI **[0.98275, 1.00207]**, raw p **0.038406**, Holm **1.0** (unconfirmed, no familywise evidence); **`gpt2:batch:8`** and **`bert:batch:8`** inconclusive. Other medians are roughly ±1.5% or less except broader batch-8 spreads. No statistically reliable familywise regression, but the strict no-regression sign-off is **not fully certified**; no waiver and no 0% proof.
- Direct Rust AFTER vs gigatoken remains separate/descriptive: warm tokie wins **12/18**, median ratio **1.122**; cold gigatoken wins **16/18**, ratio **0.440**; every count/checksum matches. BERT is a before/after comparison only because gigatoken has no WordPiece.
- Evidence: [BENCHMARK_AFTER.md](BENCHMARK_AFTER.md); [api-decisions.tsv](api-decisions.tsv). No more benchmark passes.
