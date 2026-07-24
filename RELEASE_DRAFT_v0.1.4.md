# tokie 0.1.4

This release makes Unigram tokenizers 3–4× faster while keeping output exactly HuggingFace-identical. BPE and WordPiece models are unchanged.

## Faster Unigram encoding

SentencePiece-Unigram models (T5, XLM-RoBERTa, ALBERT, and similar multilingual vocabularies) previously ran a single Viterbi pass over the whole input — the slowest encoder in the library. tokie now splits the normalized text at metaspace (`▁`) boundaries, runs Viterbi per word unit, and memoizes the Zipf-frequent units (`▁the`, `▁of`, …) in a per-worker cache. On an Apple M3 over OpenWebText:

| Model | single-thread | batch |
|---|---|---|
| T5-base | 22.8 → 90.2 MB/s (**3.9×**) | 105 → 336 MB/s (**3.2×**) |
| XLM-RoBERTa-base | 21.4 → 89.1 MB/s (**4.2×**) | 94 → 325 MB/s (**3.4×**) |

Output is byte-for-byte identical to the previous encoder and to HuggingFace: 20342/20342 documents match on both T5-base and XLM-RoBERTa-base over real web text, now covered in the accuracy CI.

### Correctness guard

Splitting at every `▁` is only lossless when no vocabulary token spans a word boundary. At construction, tokie scans the vocabulary for tokens containing an interior `▁` (a multi-word token); if any exist, that model automatically falls back to exact whole-string Viterbi. Every production Unigram vocabulary tested has no such tokens and takes the fast path, but the guard means a vocabulary that does can never be silently mis-tokenized.

## Unchanged

BPE (GPT-2, Llama, Qwen, DeepSeek) and WordPiece (BERT) encoders are untouched — the shared worker-cache plumbing was extended, not altered, and the BPE hot path is byte-identical. Verified: 20342/20342 on GPT-2 and DeepSeek-V3, full test suite green. `pretokie` is unchanged and stays at 0.1.3.
