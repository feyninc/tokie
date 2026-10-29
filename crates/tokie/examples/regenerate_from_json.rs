//! Regenerate .tkz files from tokenizer.json (bypassing cached .tkz).
//!
//! Run with: cargo run --release --example regenerate_from_json --features hf
//!
//! This downloads tokenizer.json from HuggingFace, loads it directly,
//! saves as .tkz, and optionally uploads to the tokiers/ org.

use tokie::Tokenizer;

fn main() {
    let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();

    // (HF source repo, tokiers repo name)
    let models: &[(&str, &str)] = &[
        // BPE byte-level (affected by \u{3000} fix)
        ("openai-community/gpt2", "gpt2"),
        ("Xenova/gpt-4", "cl100k"),
        ("Xenova/gpt-4o", "o200k"),
        ("FacebookAI/roberta-base", "roberta-base"),
        ("microsoft/phi-2", "phi-2"),
        ("microsoft/Phi-3-mini-4k-instruct", "Phi-3-mini-4k-instruct"),
        ("answerdotai/ModernBERT-base", "ModernBERT-base"),
        ("codellama/CodeLlama-7b-hf", "CodeLlama-7b-hf"),
        ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
        ("meta-llama/Llama-4-Scout-17B-16E", "Llama-4-Scout-17B-16E"),
        ("mistralai/Mistral-7B-v0.1", "Mistral-7B-v0.1"),
        ("mistralai/Mistral-Nemo-Base-2407", "Mistral-Nemo-Base-2407"),
        ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B-v0.1"),
        ("Qwen/Qwen2-7B", "Qwen2-7B"),
        ("Qwen/Qwen3-Embedding-0.6B", "Qwen3-Embedding-0.6B"),
        ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding-4B"),
        ("Qwen/Qwen3-Embedding-8B", "Qwen3-Embedding-8B"),
        ("Alibaba-NLP/gte-Qwen2-7B-instruct", "gte-Qwen2-7B-instruct"),
        ("BAAI/bge-en-icl", "bge-en-icl"),
        // Cohere
        ("Cohere/Cohere-embed-english-v3.0", "Cohere-embed-english-v3.0"),
        ("Cohere/Cohere-embed-english-light-v3.0", "Cohere-embed-english-light-v3.0"),
        ("Cohere/Cohere-embed-multilingual-v3.0", "Cohere-embed-multilingual-v3.0"),
        ("Cohere/Cohere-embed-multilingual-light-v3.0", "Cohere-embed-multilingual-light-v3.0"),
        // Jina
        ("jinaai/jina-embeddings-v2-base-en", "jina-embeddings-v2-base-en"),
        ("jinaai/jina-embeddings-v2-base-code", "jina-embeddings-v2-base-code"),
        ("jinaai/jina-embeddings-v3", "jina-embeddings-v3"),
        ("jinaai/jina-embeddings-v4", "jina-embeddings-v4"),
        // Voyage
        ("voyageai/voyage-3", "voyage-3"),
        ("voyageai/voyage-3-large", "voyage-3-large"),
        ("voyageai/voyage-3-lite", "voyage-3-lite"),
        ("voyageai/voyage-3.5", "voyage-3.5"),
        ("voyageai/voyage-3.5-lite", "voyage-3.5-lite"),
        ("voyageai/voyage-code-2", "voyage-code-2"),
        ("voyageai/voyage-code-3", "voyage-code-3"),
        ("voyageai/voyage-finance-2", "voyage-finance-2"),
        ("voyageai/voyage-law-2", "voyage-law-2"),
        ("voyageai/voyage-multilingual-2", "voyage-multilingual-2"),
        ("voyageai/voyage-multimodal-3", "voyage-multimodal-3"),
        // SentencePiece / Unigram
        ("google-t5/t5-base", "t5-base"),
        ("FacebookAI/xlm-roberta-base", "xlm-roberta-base"),
        // WordPiece (shouldn't be affected but regenerate for consistency)
        ("bert-base-uncased", "bert-base-uncased"),
        ("sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"),
        ("sentence-transformers/all-MiniLM-L12-v2", "all-MiniLM-L12-v2"),
        ("sentence-transformers/all-mpnet-base-v2", "all-mpnet-base-v2"),
        ("BAAI/bge-base-en-v1.5", "bge-base-en-v1.5"),
        ("BAAI/bge-large-en-v1.5", "bge-large-en-v1.5"),
        ("BAAI/bge-small-en-v1.5", "bge-small-en-v1.5"),
        ("intfloat/e5-base-v2", "e5-base-v2"),
        ("intfloat/e5-large-v2", "e5-large-v2"),
        ("intfloat/e5-small-v2", "e5-small-v2"),
        ("thenlper/gte-base", "gte-base"),
        ("thenlper/gte-large", "gte-large"),
        ("thenlper/gte-small", "gte-small"),
        ("cross-encoder/ms-marco-MiniLM-L-4-v2", "ms-marco-MiniLM-L-4-v2"),
        ("cross-encoder/ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-6-v2"),
        ("mixedbread-ai/mxbai-embed-large-v1", "mxbai-embed-large-v1"),
        ("mixedbread-ai/mxbai-embed-2d-large-v1", "mxbai-embed-2d-large-v1"),
        ("mixedbread-ai/mxbai-embed-xsmall-v1", "mxbai-embed-xsmall-v1"),
        ("mixedbread-ai/deepset-mxbai-embed-de-large-v1", "deepset-mxbai-embed-de-large-v1"),
        ("nomic-ai/nomic-embed-text-v1", "nomic-embed-text-v1"),
        // New models (DeepSeek, Gemma, etc.)
        ("deepseek-ai/DeepSeek-V3", "DeepSeek-V3"),
        ("deepseek-ai/DeepSeek-R1", "DeepSeek-R1"),
        ("google/gemma-2-2b", "gemma-2-2b"),
        ("google/gemma-3-4b-it", "gemma-3-4b-it"),
        ("BAAI/bge-m3", "bge-m3"),
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "snowflake-arctic-embed-l-v2.0"),
        ("nvidia/NV-Embed-v2", "NV-Embed-v2"),
        ("HuggingFaceTB/SmolLM2-135M", "SmolLM2-135M"),
        ("stabilityai/stablelm-2-1_6b", "stablelm-2-1_6b"),
        // Qwen3 / Qwen3.5
        ("Qwen/Qwen3-0.6B", "Qwen3-0.6B"),
        ("Qwen/Qwen3-1.7B", "Qwen3-1.7B"),
        ("Qwen/Qwen3-4B", "Qwen3-4B"),
        ("Qwen/Qwen3-8B", "Qwen3-8B"),
        ("Qwen/Qwen3-14B", "Qwen3-14B"),
        ("Qwen/Qwen3-32B", "Qwen3-32B"),
        ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "Qwen3-Coder-30B-A3B-Instruct"),
        ("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B"),
        ("Qwen/Qwen3.5-2B", "Qwen3.5-2B"),
        ("Qwen/Qwen3.5-4B", "Qwen3.5-4B"),
        ("Qwen/Qwen3.5-9B", "Qwen3.5-9B"),
        ("Qwen/Qwen3.5-27B", "Qwen3.5-27B"),
        ("Qwen/Qwen3.5-35B-A3B", "Qwen3.5-35B-A3B"),
    ];

    std::fs::create_dir_all("models").unwrap();

    let mut success = 0;
    let mut failed = Vec::new();

    for (hf_repo, tokiers_name) in models {
        print!("{:<55} ", hf_repo);

        // Download tokenizer.json directly (bypass .tkz cache)
        let repo = hf_hub::Repo::model(hf_repo.to_string());
        let repo_api = api.repo(repo);
        let json_path = match repo_api.get("tokenizer.json") {
            Ok(p) => p,
            Err(e) => {
                println!("SKIP (no tokenizer.json: {})", e);
                failed.push((*hf_repo, format!("{}", e)));
                continue;
            }
        };

        // Load from JSON (NOT from .tkz)
        let tok = match Tokenizer::from_json(&json_path) {
            Ok(t) => t,
            Err(e) => {
                println!("LOAD FAIL: {}", e);
                failed.push((*hf_repo, format!("{}", e)));
                continue;
            }
        };

        let tkz_path = format!("models/{}.tkz", tokiers_name);
        match tok.to_file(&tkz_path) {
            Ok(_) => {
                println!("OK  vocab={}", tok.vocab_size());
                success += 1;
            }
            Err(e) => {
                println!("SAVE FAIL: {}", e);
                failed.push((*hf_repo, format!("save: {}", e)));
            }
        }
    }

    println!("\n=== Summary ===");
    println!("  Regenerated: {}/{}", success, models.len());

    if !failed.is_empty() {
        println!("\n  Failed:");
        for (name, err) in &failed {
            println!("    {} — {}", name, err);
        }
    }

    println!("\nNext steps:");
    println!("  1. Upload .tkz files to tokiers/ repos:");
    println!("     for f in models/*.tkz; do");
    println!("       name=$(basename $f .tkz)");
    println!("       hf upload tokiers/$name $f tokenizer.tkz");
    println!("     done");
}
