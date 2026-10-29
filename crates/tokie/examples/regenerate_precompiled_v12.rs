//! Regenerate v12 .tkz files for Unigram / precompiled-charsmap models.
//!
//! Only these models' .tkz semantics changed in v12 (exact charsmap blob +
//! f64 unigram scores); all other repos stay on their current files.
//!
//! Run with: cargo run --release --example regenerate_precompiled_v12 --features hf

use tokie::{EncoderType, Normalizer, Tokenizer};

fn main() {
    let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();

    // (HF source repo, tokiers repo name) — the full tokiers roster; only
    // Unigram-encoder models are regenerated.
    let models: &[(&str, &str)] = &[
        ("google-t5/t5-base", "t5-base"),
        ("FacebookAI/xlm-roberta-base", "xlm-roberta-base"),
        ("BAAI/bge-m3", "bge-m3"),
        ("Snowflake/snowflake-arctic-embed-l-v2.0", "snowflake-arctic-embed-l-v2.0"),
        // XLM-R–derived and multilingual candidates — regenerated only if
        // they turn out to be Unigram:
        ("jinaai/jina-embeddings-v3", "jina-embeddings-v3"),
        ("Cohere/Cohere-embed-multilingual-v3.0", "Cohere-embed-multilingual-v3.0"),
        ("Cohere/Cohere-embed-multilingual-light-v3.0", "Cohere-embed-multilingual-light-v3.0"),
        ("nvidia/NV-Embed-v2", "NV-Embed-v2"),
        ("mixedbread-ai/deepset-mxbai-embed-de-large-v1", "deepset-mxbai-embed-de-large-v1"),
    ];

    std::fs::create_dir_all("models").unwrap();
    let mut regenerated = Vec::new();

    for (hf_repo, tokiers_name) in models {
        print!("{hf_repo:<50} ");
        let json_path = match api.repo(hf_hub::Repo::model(hf_repo.to_string())).get("tokenizer.json") {
            Ok(p) => p,
            Err(e) => {
                println!("SKIP (no tokenizer.json: {e})");
                continue;
            }
        };
        let tok = match Tokenizer::from_json(&json_path) {
            Ok(t) => t,
            Err(e) => {
                println!("LOAD FAIL: {e}");
                continue;
            }
        };
        let is_unigram = tok.encoder_type() == EncoderType::Unigram;
        let is_precompiled = matches!(tok.normalizer(), Normalizer::SentencePiecePrecompiled { .. });
        if !is_unigram && !is_precompiled {
            println!("skip ({:?}, unaffected)", tok.encoder_type());
            continue;
        }
        let tkz_path = format!("models/{tokiers_name}.tkz");
        match tok.to_file(&tkz_path) {
            Ok(_) => {
                println!(
                    "REGENERATED  vocab={} unigram={is_unigram} precompiled={is_precompiled}",
                    tok.vocab_size()
                );
                regenerated.push(*tokiers_name);
            }
            Err(e) => println!("SAVE FAIL: {e}"),
        }
    }

    println!("\nRegenerated {} models: {:?}", regenerated.len(), regenerated);
    println!("Upload with:");
    for name in &regenerated {
        println!("  hf upload tokiers/{name} models/{name}.tkz tokenizer.tkz");
    }
}
