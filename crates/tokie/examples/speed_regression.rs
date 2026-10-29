//! Speed regression benchmark: measure encoding throughput on enwik8 for all supported tokenizers.
//! Run: cargo run --example speed_regression --release --features hf
//!
//! Outputs JSON to stdout with per-model MB/s throughput.

use std::path::Path;
use std::time::Instant;
use tokie::Tokenizer;

fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("Failed to read {}: {e}", path.display()));
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

fn bench_model(name: &str, tokiers_repo: &str, text: &str, warmup: usize, iters: usize) -> f64 {
    let tok = match Tokenizer::from_pretrained(tokiers_repo) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[{name}] SKIP: {e}");
            return 0.0;
        }
    };

    // Warmup
    for _ in 0..warmup {
        let _ = tok.encode(text, false);
    }

    // Timed runs
    let start = Instant::now();
    for _ in 0..iters {
        let _ = tok.encode(text, false);
    }
    let elapsed = start.elapsed();

    let total_bytes = text.len() as f64 * iters as f64;
    let mb_per_sec = total_bytes / elapsed.as_secs_f64() / 1_000_000.0;
    mb_per_sec
}

fn main() {
    let text = load_enwik8(1_000_000);
    let warmup = 2;
    let iters = 5;

    let models: Vec<(&str, &str, &str)> = vec![
        // WordPiece
        ("bert-base-uncased", "tokiers/bert-base-uncased", "wordpiece"),
        ("all-MiniLM-L6-v2", "tokiers/all-MiniLM-L6-v2", "wordpiece"),
        // BPE (byte-level)
        ("gpt2", "tokiers/gpt2", "bpe"),
        ("Phi-3-mini", "tokiers/Phi-3-mini-4k-instruct", "bpe"),
        ("CodeLlama-7b", "tokiers/CodeLlama-7b-hf", "bpe"),
        ("Llama-3.2-1B", "tokiers/Llama-3.2-1B", "bpe"),
        ("Mistral-7B", "tokiers/Mistral-7B-v0.1", "bpe"),
        ("Mixtral-8x7B", "tokiers/Mixtral-8x7B-v0.1", "bpe"),
        // Jina
        ("jina-v2-base-en", "tokiers/jina-embeddings-v2-base-en", "bpe"),
        // Cohere
        ("Cohere-english-v3", "tokiers/Cohere-embed-english-v3.0", "bpe"),
        // tiktoken-style
        ("cl100k", "tokiers/cl100k", "bpe"),
        ("o200k", "tokiers/o200k", "bpe"),
        // SentencePiece / Unigram
        ("t5-base", "tokiers/t5-base", "unigram"),
        ("xlm-roberta-base", "tokiers/xlm-roberta-base", "sentencepiece"),
    ];

    eprintln!("Benchmarking {} models on {} bytes of enwik8 ({} warmup, {} iters)", models.len(), text.len(), warmup, iters);

    let mut results = Vec::new();
    for (name, repo, typ) in &models {
        let mb_s = bench_model(name, repo, &text, warmup, iters);
        eprintln!("  {name:<30} {mb_s:>8.1} MB/s  ({typ})");
        results.push(serde_json::json!({
            "model": name,
            "repo": repo,
            "type": typ,
            "mb_per_sec": (mb_s * 10.0).round() / 10.0,
        }));
    }

    let output = serde_json::json!({
        "text_bytes": text.len(),
        "warmup": warmup,
        "iters": iters,
        "results": results,
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
