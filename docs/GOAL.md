# Goal

Tokie encodes text to token ids and decodes those ids. The ids match Hugging Face tokenizers v1. On OpenWebText, tokie is at least as fast as gigatoken on every model gigatoken implements, and at least as fast as tokenizers v1 on the rest.

Normalie rewrites the string. Pretokie splits that string. Tokie owns the vocabulary, the piece cache, and the ids.

A change in pretokie is allowed when the cache needs it. The spans stay the ones the pretokie oracles already lock, and those oracles and benches are re-run.

## Models

The spec for a row is that model's `tokenizer.json`, encoded by tokenizers v1 with special tokens off and on. The table is the set pretokie and normalie can name. When the table and the file disagree, encode the file. An unknown pattern is `pretokie::refuse`. An unknown normalizer is `normalie::refuse`.

| Row | Normalizer | Cut | Model |
| --- | --- | --- | --- |
| GPT-2, r50k, p50k | none | `Gpt2` | byte-level BPE |
| RoBERTa | none | `Gpt2` | byte-level BPE, rank is not the id |
| cl100k, Llama 3 | none | `Cl100k` | byte-level BPE |
| Olmo | none | `Olmo` | byte-level BPE |
| Voyage, Qwen2 | `Nfc` when the file asks | `Voyage` | byte-level BPE |
| Qwen | `Nfc` when the file asks | `Qwen` | byte-level BPE |
| o200k, Llama 4 | none | `O200k` | byte-level BPE |
| Nemotron | none | `Nemotron` | byte-level BPE |
| Kimi | none | `Kimi` | byte-level BPE |
| SmolLM | none | `SmolLM` | byte-level BPE |
| DeepSeek V3 | none | `DeepSeek` | byte-level BPE |
| BERT cased | `BertCased` | `Bert` | WordPiece |
| BERT uncased, BGE, GTE, E5, MiniLM | `BertUncased` | `Bert` | WordPiece |
| T5 | `Precompiled`, then `MetaspaceWords` | `T5` | Unigram |
| ALBERT | `Albert` | `T5` | Unigram |
| Llama 2, Mistral, Gemma | `Llama` | none | SentencePiece BPE |

## Speed

The bench is `bench/tokie`. One warmup, then the median of five. It times the whole file and every line of the OpenWebText sample. Rates are input megabytes per second. A row gigatoken does not implement is compared with tokenizers v1 only.

## Cache

Most pieces are 15 bytes or fewer and repeat. Those pack into one `u128`. A hit copies ids. A miss runs the model into a reused scratch buffer and stores the ids. Encodings of four ids or fewer sit in the entry. The rest sit in an arena.

## Remaining

- GPT-2 on the whole OpenWebText file is still sometimes under gigatoken. GPT-2 and cl100k line rates sit a few percent under. WordPiece, Unigram, and unaffixed SentencePiece still need a timing against tokenizers v1 on this binary.
- Rebuild `tokie-python` and `tokie-cli` on this `Tokenizer`. Both still call the old `Tokenizer`, so they do not open or save a `.tokie` file.
- Pair encoding, padding, truncation, and offsets. `wrap` only wraps the single-sequence template.
- Added-token `lstrip`, `rstrip`, and `single_word`. A `normalized` token is matched as written, and the normalizer does not rewrite that needle.
- Affixed SentencePiece BPE. A continuing-subword prefix or an end-of-word suffix is refused. Llama 2, Mistral, and Gemma use that form.
- An unknown ByteLevel regex is encoded as GPT-2. `BertNormalizer` ignores `clean_text` and `handle_chinese_chars` in the file.
- BERT cased has no `tokenizer.json` in the bench. SmolLM's byte vocab is missing bytes, so tokenizers v1 and gigatoken both refuse the file and the ids stay unchecked.
- Qwen's encode matches gigatoken. Tokenizers v1 refuses that Split pattern, so `wrap` stays unchecked.
