<p align="center">
  <img src="https://raw.githubusercontent.com/chonkie-inc/tokie/main/assets/tokie-banner.png" alt="tokie">
</p>

<p align="center">
  <b>10-136x faster tokenization, 10x smaller model files, 100% accurate</b>
</p>

<p align="center">
  <a href="https://github.com/chonkie-inc/tokie">GitHub</a> · <a href="https://crates.io/crates/tokie">crates.io</a> · <a href="https://huggingface.co/tokiers">HuggingFace</a>
</p>

---

**tokie** is a fast, correct tokenizer library built in Rust with Python bindings. Drop-in replacement for HuggingFace tokenizers — supports BPE (GPT-2, tiktoken, SentencePiece), WordPiece (BERT), and Unigram encoders.

## Installation

```bash
pip install tokie
```

## Quick Start

```python
import tokie

# Load from HuggingFace Hub (tries .tkz first, falls back to tokenizer.json)
tokenizer = tokie.Tokenizer.from_pretrained("bert-base-uncased")

# Encode — callable syntax or .encode()
encoding = tokenizer("Hello, world!")
print(encoding.ids)               # [101, 7592, 1010, 2088, 999, 102]
print(encoding.tokens)            # ['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
print(encoding.attention_mask)    # [1, 1, 1, 1, 1, 1]
print(encoding.special_tokens_mask)  # [1, 0, 0, 0, 0, 1]

# Decode
text = tokenizer.decode(encoding.ids)  # "hello , world !"

# Token count (fast, no Encoding overhead)
count = tokenizer.count_tokens("Hello, world!")

# Batch encode (parallel across all cores)
encodings = tokenizer.encode_batch(["Hello!", "World"], add_special_tokens=True)
```

## Padding & Truncation

```python
# Truncate to max length (special tokens preserved)
tokenizer.enable_truncation(max_length=32)

# Pad all sequences in a batch to the same length
tokenizer.enable_padding(length=32, pad_id=tokenizer.pad_token_id or 0)

# Batch encode — all sequences same length, ready for model input
texts = ["Hello world", "Short", "A much longer sentence for testing"]
batch = tokenizer.encode_batch(texts, add_special_tokens=True)
for enc in batch:
    print(len(enc), enc.ids[:5])  # All length 32
```

## Pair Encoding (Cross-Encoders)

```python
pair = tokenizer("How are you?", "I am fine.")  # or tokenizer.encode_pair(...)
print(pair.ids)                # [101, 2129, 2024, 2017, 1029, 102, 1045, 2572, 2986, 1012, 102]
print(pair.type_ids)           # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
print(pair.special_tokens_mask)  # [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
```

## Byte Offsets

```python
enc = tokenizer.encode_with_offsets("Hello world")
for token_id, (start, end) in zip(enc.ids, enc.offsets):
    print(f"  token {token_id}: bytes [{start}:{end}]")
```

## Save & Load (.tkz format)

tokie's binary `.tkz` format is ~10x smaller than `tokenizer.json` and loads in ~5ms:

```python
tokenizer.save("model.tkz")
tokenizer = tokie.Tokenizer.from_file("model.tkz")
```

## Supported Models

Works with any HuggingFace tokenizer — GPT-2, BERT, Llama 3/4, Mistral, Phi, Qwen, T5, XLM-RoBERTa, and more.

## Benchmarks

| Model | Text Size | tokie | HF tokenizers | Speedup |
|-------|-----------|-------|---------------|---------|
| BERT | 900 KB | 1.69 ms | 229 ms | **136x** |
| GPT-2 | 900 KB | 1.70 ms | 181 ms | **107x** |
| Llama 3 | 900 KB | 2.04 ms | 190 ms | **93x** |
| Qwen 3 | 45 KB | 0.15 ms | 8.18 ms | **54x** |
| Gemma 3 | 45 KB | 1.01 ms | 9.62 ms | **10x** |

100% token-accurate across all models.

## License

MIT OR Apache-2.0
