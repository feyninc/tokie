# Supported Tokenizers

This document lists tokenizers and their compatibility with tokie.

## Status Key

| Status | Meaning |
|--------|---------|
| ✅ | Fully supported and tested |
| ⚠️ | Loads and encodes, but may have edge cases |
| ❌ | Not supported (known issues) |
| ❓ | Unknown/untested |
| - | Not yet implemented |

## All Models

*Performance: Load time (ms), Throughput (MB/s), Speedup vs HuggingFace tokenizers. Tested on War and Peace (3.36 MB), Apple M3 Max.*

| Provider | Model | Vocab Size | Tokie Encoder | Status | Load | Throughput | Speedup |
|----------|-------|------------|---------------|--------|------|------------|---------|
| AI21 | Jamba | ~256,000 | SentencePiece | - | - | - | - |
| AI2 | OLMo | 50,280 | Backtracking | - | - | - | - |
| AI2 | OLMo 2 | 100,278 | Backtracking | - | - | - | - |
| Alibaba | Qwen | 151,936 | Backtracking | ✅ | 17ms | 272 MB/s | ~50x |
| Alibaba | Qwen 2 | 151,936 | Backtracking | ✅ | 17ms | 272 MB/s | ~50x |
| Alibaba | Qwen 3 | ~152,000 | Backtracking | - | - | - | - |
| Anthropic | Claude | ~100,000 | Backtracking | ⚠️ | - | - | - |
| Baichuan | Baichuan | 64,000 | SentencePiece | - | - | - | - |
| Baichuan | Baichuan 2 | 125,696 | SentencePiece | - | - | - | - |
| Cohere | Command R | 256,000 | Backtracking | - | - | - | - |
| Cohere | Command R+ | 256,000 | Backtracking | - | - | - | - |
| Cohere | Cohere Embed | ~256,000 | - | ❓ | - | - | - |
| Databricks | DBRX | 100,277 | Backtracking | - | - | - | - |
| DeepSeek | DeepSeek | 102,400 | Simple | ⚠️ | - | - | - |
| DeepSeek | DeepSeek V2/V3 | 129,280 | Simple | ⚠️ | - | - | - |
| DeepSeek | DeepSeek R1 | 129,280 | Simple | - | - | - | - |
| Google | ALBERT | 30,000 | Unigram | ✅ | - | - | - |
| Google | BERT | 30,522 | WordPiece | ✅ | 3ms | 315 MB/s | ~50x |
| Google | DistilBERT | 30,522 | WordPiece | ✅ | 3ms | 315 MB/s | ~50x |
| Google | ELECTRA | 30,522 | WordPiece | ✅ | 3ms | 315 MB/s | ~50x |
| Google | Gemini | ~256,000 | SentencePiece | - | - | - | - |
| Google | Gemini 2 | ~256,000 | SentencePiece | - | - | - | - |
| Google | Gemma | 256,000 | SentencePiece | ✅ | - | - | - |
| Google | Gemma 2 | 256,000 | SentencePiece | ✅ | - | - | - |
| Google | ModernBERT | 50,280 | Backtracking | ✅ | 5ms | 127 MB/s | ~20x |
| Google | mT5 | 250,000 | Unigram | ✅ | - | - | - |
| Google | PaLM | 256,000 | SentencePiece | ⚠️ | - | - | - |
| Google | T5 | 32,100 | Unigram | ✅ | 3ms | 57 MB/s | ~15x |
| Google | T5 Base | 32,100 | Unigram | ✅ | 3ms | 57 MB/s | ~15x |
| Intfloat | E5 | 30,522 | WordPiece | ✅ | 3ms | 320 MB/s | ~50x |
| Jina | Jina Embeddings v2 | 30,528 | WordPiece | ✅ | 3ms | 321 MB/s | ~50x |
| Jina | Jina Embeddings v3 | ~30,000 | Backtracking | - | - | - | - |
| Meta | Code Llama | 32,016 | SentencePiece | ✅ | 2ms | 86 MB/s | ~15x |
| Meta | DeBERTa | ~128,000 | Backtracking | ✅ | - | - | - |
| Meta | DeBERTa v3 | 128,000 | SentencePiece | ✅ | - | - | - |
| Meta | Llama 1 | 32,000 | SentencePiece | ✅ | - | - | - |
| Meta | Llama 2 | 32,000 | SentencePiece | ✅ | - | - | - |
| Meta | Llama 3 | 128,000 | Backtracking | ✅ | 15ms | 610 MB/s | ~100x |
| Meta | Llama 3.1 | 128,000 | Backtracking | ✅ | 15ms | 610 MB/s | ~100x |
| Meta | Llama 3.2 | 128,000 | Backtracking | ✅ | 15ms | 610 MB/s | ~100x |
| Meta | Llama 4 | 200,000 | Simple | ✅ | - | - | - |
| Meta | RoBERTa | 50,265 | Simple | ✅ | 9ms | 319 MB/s | ~50x |
| Meta | XLM-RoBERTa | 250,002 | Unigram | ✅ | 23ms | 54 MB/s | ~15x |
| Microsoft | Phi-1/2 | 51,200 | Backtracking | ✅ | 5ms | 282 MB/s | ~50x |
| Microsoft | Phi-3 | 32,064 | SentencePiece | ✅ | 2ms | 85 MB/s | ~15x |
| Mistral | Mistral 7B | 32,000 | SentencePiece | ✅ | 2ms | 85 MB/s | ~15x |
| Mistral | Mistral Large | 32,768 | Simple | ✅ | - | - | - |
| Mistral | Mistral Nemo | 131,072 | Simple | ✅ | 39ms | 324 MB/s | ~50x |
| Mistral | Mixtral 8x7B | 32,000 | SentencePiece | ✅ | 2ms | 86 MB/s | ~15x |
| MosaicML | MPT | 50,432 | Backtracking | ✅ | - | - | - |
| Moonshot AI | Kimi K2 | 163,840 | Backtracking | - | - | - | - |
| Nomic | Nomic Embed | 30,522 | WordPiece | ✅ | 3ms | 320 MB/s | ~50x |
| OpenAI | GPT-2 | 50,257 | Backtracking | ✅ | 5ms | 312 MB/s | ~50x |
| OpenAI | GPT-3 | 50,257 | Backtracking | ✅ | 5ms | 312 MB/s | ~50x |
| OpenAI | GPT-3.5 | 50,281 | Backtracking | ✅ | 5ms | 312 MB/s | ~50x |
| OpenAI | GPT-4 | 100,256 | Backtracking | ✅ | 11ms | 315 MB/s | ~50x |
| OpenAI | GPT-4 Turbo | 100,256 | Backtracking | ✅ | 11ms | 315 MB/s | ~50x |
| OpenAI | GPT-4o | 200,000 | Backtracking | ✅ | 21ms | 287 MB/s | ~50x |
| OpenAI | GPT-4o mini | 200,000 | Backtracking | ✅ | 21ms | 287 MB/s | ~50x |
| OpenAI | GPT-5 | ~200,000 | Backtracking | - | - | - | - |
| OpenAI | text-davinci-003 | 50,281 | Backtracking | ✅ | 5ms | 312 MB/s | ~50x |
| OpenAI | text-embedding | 100,256 | Backtracking | ✅ | 11ms | 315 MB/s | ~50x |
| OpenAI | text-embedding-3 | ~100,000 | Backtracking | - | - | - | - |
| SentenceTransformers | MiniLM | 30,522 | WordPiece | ✅ | 3ms | 306 MB/s | ~50x |
| Shanghai AI Lab | InternLM | 103,168 | SentencePiece | - | - | - | - |
| Shanghai AI Lab | InternLM 2 | 92,544 | SentencePiece | - | - | - | - |
| Stability AI | StableLM | 50,254 | Backtracking | - | - | - | - |
| Stability AI | StableLM 2 | 100,289 | Backtracking | - | - | - | - |
| BigCode | StarCoder | 49,152 | Backtracking | ✅ | - | - | - |
| Thenlper | GTE | 30,522 | WordPiece | ✅ | 3ms | 317 MB/s | ~50x |
| TII | Falcon | 65,024 | Backtracking | ⚠️ | - | - | - |
| Voyage | Voyage 3 | 151,665 | Backtracking | ✅ | 17ms | 310 MB/s | ~50x |
| Voyage | Voyage 3 Large | 151,665 | Backtracking | ✅ | 17ms | 310 MB/s | ~50x |
| Voyage | Voyage Code 3 | 151,665 | Backtracking | ✅ | 17ms | 310 MB/s | ~50x |
| xAI | Grok | ~131,000 | Backtracking | - | - | - | - |
| xAI | Grok 2 | ~131,000 | Backtracking | - | - | - | - |
| 01.AI | Yi | 64,000 | SentencePiece | ❌ | - | - | - |
| 01.AI | Yi 1.5 | 64,000 | SentencePiece | ❌ | - | - | - |
| Zhipu AI | ChatGLM | 65,024 | SentencePiece | - | - | - | - |
| Zhipu AI | ChatGLM 2/3 | 64,794 | SentencePiece | - | - | - | - |
| BAAI | BGE | 30,522 | WordPiece | ✅ | 3ms | 310 MB/s | ~50x |

## Known Issues

### Yi (❌ Not Supported)

Yi tokenizers use a **sparse vocabulary** where token IDs have gaps (missing IDs in the sequence). Tokie's current implementation assumes contiguous token IDs, causing index out of bounds errors.

**Required fix:** Handle sparse vocabularies by using a HashMap for token lookups instead of assuming contiguous array indices.

### Falcon (⚠️ Pretokenizer Mismatch)

Falcon uses a `ByteLevel` pretokenizer but with different configuration than GPT-2. The pretokenizer produces ~2x more tokens than expected, causing tokenization mismatches.

**Required fix:** Investigate Falcon's specific ByteLevel configuration and add support for it.

### DeepSeek V2/V3 (⚠️ Pretokenizer Mismatch)

DeepSeek V2/V3 uses a complex `Sequence` pretokenizer that doesn't match tokie's standard patterns. The tokenization loads but produces different results than HuggingFace.

**Required fix:** Analyze DeepSeek's Sequence pretokenizer components and add support.

### Claude (⚠️ Untested)

Claude tokenizers are not publicly available for verification. Status based on expected behavior.

### PaLM (⚠️ Untested)

PaLM tokenizers require special access and haven't been fully tested against tokie.

### Cohere Embed (❓ Unknown)

Cohere embedding models use a proprietary tokenizer format that hasn't been analyzed.

## Models Needing Work

| Priority | Model | Issue | Effort |
|----------|-------|-------|--------|
| High | Yi | Sparse vocab support | Medium |
| High | Gemini | Not implemented | Unknown |
| High | Grok | Not implemented | Unknown |
| Medium | Kimi K2 | tiktoken.model format + Chinese regex | Medium |
| Medium | DeepSeek V2/V3 | Complex pretokenizer | Medium |
| Medium | Falcon | ByteLevel config | Low |
| Medium | Command R | Not implemented | Low |
| Low | InternLM | Not implemented | Low |
| Low | ChatGLM | Not implemented | Low |
| Low | Cohere | Unknown format | Unknown |

## Notes

### Encoder Types in Tokie

- **Backtracking**: Uses Aho-Corasick automata for O(n) greedy matching. Produces correct results for tiktoken-style tokenizers (OpenAI models) where greedy matching is the intended behavior.

- **Simple**: Standard O(n²) BPE that always applies the lowest-rank merge first. Produces correct results for all BPE tokenizers, including those where merge priority matters (Llama 3, Qwen, etc.).

- **SentencePiece**: Specialized encoder for SentencePiece BPE models (Llama 1/2, Mistral, Gemma). Handles metaspace (▁) normalization and byte fallback tokens.

- **Unigram**: Viterbi dynamic programming algorithm for probabilistic tokenization. Used by T5, ALBERT, XLM-RoBERTa, and other SentencePiece Unigram models. Finds the segmentation that maximizes total log probability.

- **WordPiece**: BERT-style subword tokenization with `##` continuation prefix.

### Pre-tokenizer Types in Tokie

| Type | Description | Used By |
|------|-------------|---------|
| `Gpt2` | GPT-2 regex pattern for words, numbers, punctuation | GPT-2, GPT-3, p50k, r50k, RoBERTa, ModernBERT |
| `Cl100k` | Extended pattern with better Unicode handling | cl100k, Llama 3, Qwen |
| `O200k` | Further extended for GPT-4o | o200k |
| `Bert` | Whitespace + punctuation splitting | BERT, DistilBERT, GTE, BGE, E5, MiniLM |
| `None` | No pre-tokenization | SentencePiece models |

### Normalizers in Tokie

| Normalizer | Description | Used By |
|------------|-------------|---------|
| `None` | No normalization | GPT-2, RoBERTa, Llama 3 |
| `BertUncased` | Lowercase + strip accents + clean text | BERT uncased, GTE, BGE, E5, MiniLM |
| `BertCased` | Clean text only (no lowercasing) | BERT cased |
| `Nfc` | Unicode NFC normalization | ModernBERT, Qwen |
| `Metaspace` | Replace spaces with ▁, prepend ▁ | Llama 1/2, Mistral, Gemma |
| `SentencePiece` | NFKC + whitespace collapse + metaspace | T5, XLM-RoBERTa, mT5 |
| `SentencePieceLowercase` | NFKD + strip accents + lowercase + metaspace | ALBERT |

### Loading Tokenizers

```rust
use tokie::{Tokenizer, EncoderType, PretokType};
use tokie::hf::from_json_with_options;

// Auto-detect encoder type (recommended)
let tokenizer = Tokenizer::from_json("tokenizer.json")?;

// OpenAI-style (use Backtracking)
let gpt4 = Tokenizer::from_json("cl100k_tokenizer.json")?;

// Llama 3 style (use Simple for correctness)
let llama3 = from_json_with_options(
    "llama3_tokenizer.json",
    EncoderType::Simple,
    PretokType::Cl100k,
)?;

// Unigram models (auto-detected)
let t5 = Tokenizer::from_json("t5_tokenizer.json")?;
let albert = Tokenizer::from_json("albert_tokenizer.json")?;
```

### Binary Format

Tokie can save/load tokenizers in a fast binary format (`.tkz`):

```rust
// Save
tokenizer.to_file("model.tkz")?;

// Load (instant, no parsing)
let tokenizer = Tokenizer::from_file("model.tkz")?;
```

Pre-built binary files are available in the `models/` directory:

**OpenAI / tiktoken-style (Backtracking):**
- `gpt2.tkz` - GPT-2 (50K vocab)
- `cl100k.tkz` - GPT-4 (100K vocab)
- `o200k.tkz` - GPT-4o (200K vocab)

**Large vocab BPE (Backtracking):**
- `llama3.tkz` - Llama 3 (128K vocab)
- `qwen2.tkz` - Qwen 2 (152K vocab)
- `voyage3_large.tkz` - Voyage 3 (152K vocab)
- `mistral_nemo.tkz` - Mistral Nemo (131K vocab, Simple)
- `modernbert.tkz` - ModernBERT (50K vocab)
- `phi2.tkz` - Phi-2 (50K vocab)
- `roberta.tkz` - RoBERTa (50K vocab, Simple)

**Unigram:**
- `t5.tkz` - T5 (32K vocab)
- `xlm_roberta.tkz` - XLM-RoBERTa (250K vocab)

**WordPiece:**
- `bert.tkz` - BERT (30K vocab)

**Embedding models (WordPiece, 30K vocab):**
- `baai_bge_*.tkz` - BGE embeddings
- `intfloat_e5_*.tkz` - E5 embeddings
- `thenlper_gte_*.tkz` - GTE embeddings
- `sentence_transformers_*.tkz` - MiniLM, MPNet
- `nomic_ai_*.tkz` - Nomic Embed
- `jinaai_*.tkz` - Jina Embeddings v2
