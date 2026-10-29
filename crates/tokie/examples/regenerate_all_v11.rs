//! Regenerate all .tkz model files to v11 format (with pad_token_id).
//!
//! Run with: cargo run --release --example regenerate_all_v11 --features hf
//!
//! This loads each model via from_pretrained() (which extracts pad_token_id
//! from tokenizer.json), then re-saves as .tkz v11.

use tokie::Tokenizer;

fn main() {
    // (HF repo, .tkz filename) — covers all 60 models in models/
    let models: &[(&str, &str)] = &[
        // BERT-family (WordPiece)
        ("bert-base-uncased", "bert"),
        ("FacebookAI/roberta-base", "roberta"),
        ("answerdotai/ModernBERT-base", "modernbert"),
        ("cross-encoder/ms-marco-MiniLM-L-4-v2", "cross_encoder_ms_marco_minilm_l4_v2"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "cross_encoder_ms_marco_minilm_l6_v2"),
        ("sentence-transformers/all-MiniLM-L6-v2", "sentence_transformers_all_minilm_l6_v2"),
        ("sentence-transformers/all-MiniLM-L12-v2", "sentence_transformers_all_minilm_l12_v2"),
        ("sentence-transformers/all-mpnet-base-v2", "sentence_transformers_all_mpnet_base_v2"),
        ("intfloat/e5-small-v2", "intfloat_e5_small_v2"),
        ("intfloat/e5-base-v2", "intfloat_e5_base_v2"),
        ("intfloat/e5-large-v2", "intfloat_e5_large_v2"),
        ("thenlper/gte-small", "thenlper_gte_small"),
        ("thenlper/gte-base", "thenlper_gte_base"),
        ("thenlper/gte-large", "thenlper_gte_large"),
        ("BAAI/bge-small-en-v1.5", "baai_bge_small_en_v1.5"),
        ("BAAI/bge-base-en-v1.5", "baai_bge_base_en_v1.5"),
        ("BAAI/bge-large-en-v1.5", "baai_bge_large_en_v1.5"),
        ("BAAI/bge-en-icl", "baai_bge_en_icl"),
        ("nomic-ai/nomic-embed-text-v1", "nomic_ai_nomic_embed_text_v1"),
        // GPT-2 style (BPE, ByteLevel)
        ("openai-community/gpt2", "gpt2"),
        ("Xenova/gpt-4", "cl100k"),
        ("Xenova/gpt-4o", "o200k"),
        // LLaMA family
        ("meta-llama/Llama-3.2-1B", "llama3"),
        ("meta-llama/Llama-4-Scout-17B-16E", "llama4"),
        ("codellama/CodeLlama-7b-hf", "codellama"),
        // Mistral
        ("mistralai/Mistral-7B-v0.1", "mistral_7b"),
        ("mistralai/Mistral-Nemo-Base-2407", "mistral_nemo"),
        ("mistralai/Mixtral-8x7B-v0.1", "mixtral_8x7b"),
        // Phi
        ("microsoft/phi-2", "phi2"),
        ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        // Qwen
        ("Qwen/Qwen2-7B", "qwen2"),
        ("Qwen/Qwen3-Embedding-0.6B", "qwen3_embedding_0.6b"),
        ("Qwen/Qwen3-Embedding-4B", "qwen3_embedding_4b"),
        ("Qwen/Qwen3-Embedding-8B", "qwen3_embedding_8b"),
        ("Alibaba-NLP/gte-Qwen2-7B-instruct", "alibaba_nlp_gte_qwen2_7b_instruct"),
        // SentencePiece (Unigram)
        ("google-t5/t5-base", "t5"),
        ("FacebookAI/xlm-roberta-base", "xlm_roberta"),
        // Cohere
        ("Cohere/Cohere-embed-english-v3.0", "cohere_embed_english_v3.0"),
        ("Cohere/Cohere-embed-english-light-v3.0", "cohere_embed_english_light_v3.0"),
        ("Cohere/Cohere-embed-multilingual-v3.0", "cohere_embed_multilingual_v3.0"),
        ("Cohere/Cohere-embed-multilingual-light-v3.0", "cohere_embed_multilingual_light_v3.0"),
        // Jina AI
        ("jinaai/jina-embeddings-v2-base-en", "jinaai_jina_embeddings_v2_base_en"),
        ("jinaai/jina-embeddings-v2-base-code", "jinaai_jina_embeddings_v2_base_code"),
        ("jinaai/jina-embeddings-v3", "jinaai_jina_embeddings_v3"),
        ("jinaai/jina-embeddings-v4", "jinaai_jina_embeddings_v4"),
        // Mixedbread
        ("mixedbread-ai/mxbai-embed-large-v1", "mixedbread_ai_mxbai_embed_large_v1"),
        ("mixedbread-ai/mxbai-embed-2d-large-v1", "mixedbread_ai_mxbai_embed_2d_large_v1"),
        ("mixedbread-ai/mxbai-embed-xsmall-v1", "mixedbread_ai_mxbai_embed_xsmall_v1"),
        ("mixedbread-ai/deepset-mxbai-embed-de-large-v1", "mixedbread_ai_deepset_mxbai_embed_de_large_v1"),
        // Voyage AI
        ("voyageai/voyage-3", "voyageai_voyage_3"),
        ("voyageai/voyage-3-lite", "voyageai_voyage_3_lite"),
        ("voyageai/voyage-3.5", "voyageai_voyage_3.5"),
        ("voyageai/voyage-3.5-lite", "voyageai_voyage_3.5_lite"),
        ("voyageai/voyage-code-2", "voyageai_voyage_code_2"),
        ("voyageai/voyage-code-3", "voyageai_voyage_code_3"),
        ("voyageai/voyage-finance-2", "voyageai_voyage_finance_2"),
        ("voyageai/voyage-law-2", "voyageai_voyage_law_2"),
        ("voyageai/voyage-multilingual-2", "voyageai_voyage_multilingual_2"),
        ("voyageai/voyage-multimodal-3", "voyageai_voyage_multimodal_3"),
        ("voyageai/voyage-3-large", "voyage3_large"),
    ];

    std::fs::create_dir_all("models").unwrap();

    let mut success = 0;
    let mut failed = Vec::new();

    for (repo, tkz_name) in models {
        let tkz_path = format!("models/{}.tkz", tkz_name);
        print!("{:<55} ", repo);

        match Tokenizer::from_pretrained(repo) {
            Ok(tok) => {
                let pad_info = match tok.pad_token_id() {
                    Some(id) => format!("pad_id={}", id),
                    None => "no pad_id".to_string(),
                };

                if let Err(e) = tok.to_file(&tkz_path) {
                    println!("SAVE FAIL: {}", e);
                    failed.push((*repo, format!("save: {}", e)));
                    continue;
                }

                println!("OK  vocab={}  {}", tok.vocab_size(), pad_info);
                success += 1;
            }
            Err(e) => {
                println!("LOAD FAIL: {}", e);
                failed.push((*repo, format!("{}", e)));
            }
        }
    }

    println!("\n=== Summary ===");
    println!("  Success: {}/{}", success, models.len());
    println!("  Failed:  {}/{}", failed.len(), models.len());

    if !failed.is_empty() {
        println!("\n  Failures:");
        for (name, err) in &failed {
            println!("    {} -- {}", name, err);
        }
    }
}
