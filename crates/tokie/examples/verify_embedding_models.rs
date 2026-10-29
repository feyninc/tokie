//! Verify embedding model tokenizers produce correct outputs
//!
//! Run with: cargo run --release --example verify_embedding_models

use tokenizers::Tokenizer as HfTokenizer;

/// Strip trailing padding from token IDs (handles pad_id 0 or 1)
fn strip_padding(ids: &[u32]) -> Vec<u32> {
    if ids.is_empty() {
        return vec![];
    }
    // Detect pad token (last token if repeated many times)
    let last = ids[ids.len() - 1];
    if last > 1 {
        return ids.to_vec(); // No padding detected
    }
    let mut end = ids.len();
    while end > 0 && ids[end - 1] == last {
        end -= 1;
    }
    ids[..end].to_vec()
}

fn main() {
    println!("=== Embedding Model Tokenizer Verification ===\n");

    // Focus on ASCII/English text to avoid CJK character handling differences
    let test_texts = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models encode text into dense vector representations.",
        "Tokenization is the process of splitting text into smaller units called tokens.",
        "BGE, GTE, and E5 are popular embedding models for semantic search.",
        "Embeddings power semantic search and RAG systems!",
        "The 18th century was a time of great change.",
        "user@example.com visited https://example.org/path?query=1",
    ];

    // Test BERT-based models
    let bert_models = vec![
        ("sentence_transformers_all_minilm_l6_v2", "sentence-transformers/all-MiniLM-L6-v2"),
        ("sentence_transformers_all_mpnet_base_v2", "sentence-transformers/all-mpnet-base-v2"),
        ("baai_bge_small_en_v1.5", "BAAI/bge-small-en-v1.5"),
        ("baai_bge_base_en_v1.5", "BAAI/bge-base-en-v1.5"),
        ("thenlper_gte_base", "thenlper/gte-base"),
        ("intfloat_e5_base_v2", "intfloat/e5-base-v2"),
        ("jinaai_jina_embeddings_v2_base_en", "jinaai/jina-embeddings-v2-base-en"),
        ("nomic_ai_nomic_embed_text_v1", "nomic-ai/nomic-embed-text-v1"),
    ];

    let mut total_pass = 0;
    let mut total_fail = 0;

    for (tkz_name, hf_name) in &bert_models {
        println!("=== {} ===", hf_name);

        // Load tokie tokenizer
        let tkz_path = format!("models/{}.tkz", tkz_name);
        let tokie = match tokie::Tokenizer::from_file(&tkz_path) {
            Ok(t) => t,
            Err(e) => {
                println!("  Failed to load tokie {}: {:?}\n", tkz_path, e);
                total_fail += 1;
                continue;
            }
        };

        // Load HuggingFace tokenizer
        let hf = match HfTokenizer::from_pretrained(hf_name, None) {
            Ok(t) => t,
            Err(e) => {
                println!("  Failed to load HF {}: {:?}\n", hf_name, e);
                total_fail += 1;
                continue;
            }
        };

        let mut pass = 0;
        let mut fail = 0;
        for text in &test_texts {
            let tokie_ids = tokie.encode(text, false).ids;
            let hf_encoding = hf.encode(text.to_string(), false).unwrap();
            let hf_ids = strip_padding(hf_encoding.get_ids());

            if tokie_ids == hf_ids {
                pass += 1;
            } else {
                fail += 1;
                println!("  MISMATCH: \"{}\"", text);
                println!("    tokie: {:?}", tokie_ids);
                println!("    hf:    {:?}", hf_ids);
            }
        }

        if fail == 0 {
            println!("  ✓ All {} tests pass!\n", pass);
            total_pass += 1;
        } else {
            println!("  {} pass, {} fail\n", pass, fail);
            total_fail += 1;
        }
    }

    // Test ModernBERT separately (known to have issues)
    println!("=== Alibaba-NLP/gte-modernbert-base (checking) ===");
    let tokie = tokie::Tokenizer::from_file("models/alibaba_nlp_gte_modernbert_base.tkz");
    let hf = HfTokenizer::from_pretrained("Alibaba-NLP/gte-modernbert-base", None);

    match (tokie, hf) {
        (Ok(t), Ok(h)) => {
            let text = "Hello world";
            let tokie_ids = t.encode(text, false).ids;
            let hf_encoding = h.encode(text.to_string(), false).unwrap();
            let hf_ids = strip_padding(hf_encoding.get_ids());
            println!("  tokie: {:?}", tokie_ids);
            println!("  hf:    {:?}", hf_ids);
            if tokie_ids == hf_ids {
                println!("  ✓ Match!\n");
                total_pass += 1;
            } else {
                println!("  ✗ Mismatch (needs investigation)\n");
                total_fail += 1;
            }
        }
        _ => {
            println!("  Failed to load\n");
            total_fail += 1;
        }
    }

    println!("=== Summary ===");
    println!("  {} models pass, {} need work", total_pass, total_fail);
}
