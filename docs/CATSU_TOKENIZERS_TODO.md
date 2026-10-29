# Catsu Tokenizers - TODO

Tokenizers used in catsu's models.json that are not yet supported in tokie.

## Not Available on HuggingFace (no tokenizer.json)

These models don't have a `tokenizer.json` on HuggingFace (404), so they can't be loaded via `from_pretrained()`:

- [ ] `Cohere/Cohere-embed-v4` — gated/no tokenizer.json
- [ ] `jinaai/jina-code-embeddings-0.5b` — no tokenizer.json
- [ ] `jinaai/jina-code-embeddings-1.5b` — no tokenizer.json
- [ ] `pfnet/plamo-1.0-embedding` — no tokenizer.json
- [ ] `google/embeddinggemma-300m` — not tested yet

---

## Already Supported (46 tokenizers)

### Direct Support (40)
- [x] `BAAI/bge-base-en-v1.5` -> baai_bge_base_en_v1.5.tkz
- [x] `BAAI/bge-large-en-v1.5` -> baai_bge_large_en_v1.5.tkz
- [x] `BAAI/bge-small-en-v1.5` -> baai_bge_small_en_v1.5.tkz
- [x] `BAAI/bge-en-icl` -> baai_bge_en_icl.tkz
- [x] `Alibaba-NLP/gte-Qwen2-7B-instruct` -> alibaba_nlp_gte_qwen2_7b_instruct.tkz
- [x] `Cohere/Cohere-embed-english-v3.0` -> cohere_embed_english_v3.0.tkz
- [x] `Cohere/Cohere-embed-english-light-v3.0` -> cohere_embed_english_light_v3.0.tkz
- [x] `Cohere/Cohere-embed-multilingual-v3.0` -> cohere_embed_multilingual_v3.0.tkz
- [x] `Cohere/Cohere-embed-multilingual-light-v3.0` -> cohere_embed_multilingual_light_v3.0.tkz
- [x] `intfloat/e5-base-v2` -> intfloat_e5_base_v2.tkz
- [x] `intfloat/e5-large-v2` -> intfloat_e5_large_v2.tkz
- [x] `jinaai/jina-embeddings-v2-base-code` -> jinaai_jina_embeddings_v2_base_code.tkz
- [x] `jinaai/jina-embeddings-v2-base-en` -> jinaai_jina_embeddings_v2_base_en.tkz
- [x] `jinaai/jina-embeddings-v3` -> jinaai_jina_embeddings_v3.tkz
- [x] `jinaai/jina-embeddings-v4` -> jinaai_jina_embeddings_v4.tkz
- [x] `mixedbread-ai/mxbai-embed-large-v1` -> mixedbread_ai_mxbai_embed_large_v1.tkz
- [x] `mixedbread-ai/mxbai-embed-2d-large-v1` -> mixedbread_ai_mxbai_embed_2d_large_v1.tkz
- [x] `mixedbread-ai/mxbai-embed-xsmall-v1` -> mixedbread_ai_mxbai_embed_xsmall_v1.tkz
- [x] `mixedbread-ai/deepset-mxbai-embed-de-large-v1` -> mixedbread_ai_deepset_mxbai_embed_de_large_v1.tkz
- [x] `nomic-ai/nomic-embed-text-v1` -> nomic_ai_nomic_embed_text_v1.tkz
- [x] `Qwen/Qwen3-Embedding-0.6B` -> qwen3_embedding_0.6b.tkz
- [x] `Qwen/Qwen3-Embedding-4B` -> qwen3_embedding_4b.tkz
- [x] `Qwen/Qwen3-Embedding-8B` -> qwen3_embedding_8b.tkz
- [x] `sentence-transformers/all-MiniLM-L12-v2` -> sentence_transformers_all_minilm_l12_v2.tkz
- [x] `sentence-transformers/all-MiniLM-L6-v2` -> sentence_transformers_all_minilm_l6_v2.tkz
- [x] `sentence-transformers/all-mpnet-base-v2` -> sentence_transformers_all_mpnet_base_v2.tkz
- [x] `thenlper/gte-base` -> thenlper_gte_base.tkz
- [x] `thenlper/gte-large` -> thenlper_gte_large.tkz
- [x] `voyageai/voyage-3` -> voyageai_voyage_3.tkz
- [x] `voyageai/voyage-3-lite` -> voyageai_voyage_3_lite.tkz
- [x] `voyageai/voyage-3-large` -> voyage3_large.tkz
- [x] `voyageai/voyage-3.5` -> voyageai_voyage_3.5.tkz
- [x] `voyageai/voyage-3.5-lite` -> voyageai_voyage_3.5_lite.tkz
- [x] `voyageai/voyage-code-2` -> voyageai_voyage_code_2.tkz
- [x] `voyageai/voyage-code-3` -> voyageai_voyage_code_3.tkz
- [x] `voyageai/voyage-finance-2` -> voyageai_voyage_finance_2.tkz
- [x] `voyageai/voyage-law-2` -> voyageai_voyage_law_2.tkz
- [x] `voyageai/voyage-multilingual-2` -> voyageai_voyage_multilingual_2.tkz
- [x] `voyageai/voyage-multimodal-3` -> voyageai_voyage_multimodal_3.tkz
- [x] `cl100k_base` (tiktoken) -> cl100k.tkz

### Compatible (6)
- [x] `nomic-ai/nomic-embed-text-v1.5` ~ nomic_ai_nomic_embed_text_v1.tkz
- [x] `Alibaba-NLP/gte-modernbert-base` ~ modernbert.tkz
- [x] `BAAI/bge-m3` ~ xlm_roberta.tkz
- [x] `intfloat/multilingual-e5-large` ~ xlm_roberta.tkz
- [x] `WhereIsAI/UAE-Large-V1` ~ bert.tkz
- [x] `togethercomputer/m2-bert-80M-8k-retrieval` ~ bert.tkz

---

## Progress

- **Supported:** 46/51 (90.2%)
- **Unavailable on HF:** 5/51 (9.8%)
