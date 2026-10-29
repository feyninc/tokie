//! Convert and verify new tokenizers from HuggingFace
//!
//! Run with: cargo run --release --example convert_and_verify
//!
//! This attempts to load each model via from_pretrained(), save as .tkz,
//! and verify correctness against HuggingFace tokenizers.

use tokenizers::Tokenizer as HfTokenizer;


fn main() {
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models encode text into dense vector representations.",
        "Tokenization is the process of splitting text into smaller units called tokens.",
        "BGE, GTE, and E5 are popular embedding models for semantic search.",
        "The 18th century was a time of great change.",
        "user@example.com visited https://example.org/path?query=1",
        "I can't believe it's not butter! Don't you think so?",
    ];

    // All models to convert
    let models: Vec<(&str, &str)> = vec![
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
        // Cohere
        ("Cohere/Cohere-embed-v4", "cohere_embed_v4"),
        ("Cohere/Cohere-embed-english-v3.0", "cohere_embed_english_v3.0"),
        ("Cohere/Cohere-embed-english-light-v3.0", "cohere_embed_english_light_v3.0"),
        ("Cohere/Cohere-embed-multilingual-v3.0", "cohere_embed_multilingual_v3.0"),
        ("Cohere/Cohere-embed-multilingual-light-v3.0", "cohere_embed_multilingual_light_v3.0"),
        // Jina AI
        ("jinaai/jina-embeddings-v3", "jinaai_jina_embeddings_v3"),
        ("jinaai/jina-embeddings-v4", "jinaai_jina_embeddings_v4"),
        ("jinaai/jina-code-embeddings-0.5b", "jinaai_jina_code_embeddings_0.5b"),
        ("jinaai/jina-code-embeddings-1.5b", "jinaai_jina_code_embeddings_1.5b"),
        // Mixedbread
        ("mixedbread-ai/mxbai-embed-large-v1", "mixedbread_ai_mxbai_embed_large_v1"),
        ("mixedbread-ai/mxbai-embed-2d-large-v1", "mixedbread_ai_mxbai_embed_2d_large_v1"),
        ("mixedbread-ai/mxbai-embed-xsmall-v1", "mixedbread_ai_mxbai_embed_xsmall_v1"),
        ("mixedbread-ai/deepset-mxbai-embed-de-large-v1", "mixedbread_ai_deepset_mxbai_embed_de_large_v1"),
        // Medium priority
        ("Qwen/Qwen3-Embedding-0.6B", "qwen3_embedding_0.6b"),
        ("Qwen/Qwen3-Embedding-4B", "qwen3_embedding_4b"),
        ("Qwen/Qwen3-Embedding-8B", "qwen3_embedding_8b"),
        ("Alibaba-NLP/gte-Qwen2-7B-instruct", "alibaba_nlp_gte_qwen2_7b_instruct"),
        ("BAAI/bge-en-icl", "baai_bge_en_icl"),
        // Lower priority
        ("pfnet/plamo-1.0-embedding", "pfnet_plamo_1.0_embedding"),
    ];

    let mut loaded = 0;
    let mut verified = 0;
    let mut failed_load = Vec::new();
    let mut failed_verify = Vec::new();

    for (hf_name, tkz_name) in &models {
        print!("{:<55}", hf_name);

        // Try loading with tokie
        let tokie = match tokie::Tokenizer::from_pretrained(hf_name) {
            Ok(t) => t,
            Err(e) => {
                println!("LOAD FAIL: {}", e);
                failed_load.push((*hf_name, format!("{}", e)));
                continue;
            }
        };

        // Save as .tkz
        let tkz_path = format!("../../models/{}.tkz", tkz_name);
        if let Err(e) = tokie.to_file(&tkz_path) {
            println!("SAVE FAIL: {}", e);
            failed_load.push((*hf_name, format!("save: {}", e)));
            continue;
        }
        loaded += 1;

        // Load HF tokenizer for verification
        let hf = match HfTokenizer::from_pretrained(hf_name, None) {
            Ok(t) => t,
            Err(e) => {
                println!("OK (saved .tkz, HF load failed for verify: {})", e);
                continue;
            }
        };

        // Verify
        let mut mismatches = 0;
        for text in &test_texts {
            let tokie_ids = tokie.encode(text, false).ids;
            let hf_encoding = hf.encode(text.to_string(), false).unwrap();
            let hf_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            if tokie_ids != hf_ids {
                mismatches += 1;
                if mismatches == 1 {
                    println!();
                    println!("  MISMATCH on: \"{}\"", &text[..text.len().min(60)]);
                    println!("    tokie ({} tokens): {:?}", tokie_ids.len(), &tokie_ids[..tokie_ids.len().min(15)]);
                    println!("    hf    ({} tokens): {:?}", hf_ids.len(), &hf_ids[..hf_ids.len().min(15)]);
                }
            }
        }

        if mismatches == 0 {
            println!("OK  vocab={}", tokie.vocab_size());
            verified += 1;
        } else {
            println!("  {} of {} texts mismatch", mismatches, test_texts.len());
            failed_verify.push((*hf_name, mismatches));
        }
    }

    println!("\n=== Summary ===");
    println!("  Loaded & saved:  {}/{}", loaded, models.len());
    println!("  Verified (100%): {}/{}", verified, models.len());

    if !failed_load.is_empty() {
        println!("\n  Failed to load:");
        for (name, err) in &failed_load {
            println!("    {} — {}", name, err);
        }
    }
    if !failed_verify.is_empty() {
        println!("\n  Verification failures:");
        for (name, mismatches) in &failed_verify {
            println!("    {} — {} mismatches", name, mismatches);
        }
    }
}
