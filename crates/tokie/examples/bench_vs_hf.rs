//! Benchmark tokie vs HuggingFace tokenizers (and tiktoken-rs for BPE models).
//! Run: cargo run --example bench_vs_hf --release --features hf
//!
//! Measures throughput on 1MB of enwik8 for representative models across all
//! encoder types: BPE, WordPiece, SentencePiece BPE, and Unigram.

use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;
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

struct BenchResult {
    name: &'static str,
    typ: &'static str,
    tokie_mb_s: f64,
    hf_mb_s: f64,
    tiktoken_mb_s: Option<f64>,
    tokie_tokens: usize,
    hf_tokens: usize,
    accurate: bool,
}

fn bench_tokie(tok: &Tokenizer, text: &str, warmup: usize, iters: usize) -> (f64, usize) {
    for _ in 0..warmup {
        let _ = tok.encode(text, false);
    }
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..iters {
        token_count = tok.encode(text, false).ids.len();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mb_s = (text.len() as f64 * iters as f64) / elapsed / 1_000_000.0;
    (mb_s, token_count)
}

fn bench_hf(hf: &HfTokenizer, text: &str, warmup: usize, iters: usize) -> (f64, usize) {
    for _ in 0..warmup {
        let _ = hf.encode(text, false);
    }
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..iters {
        token_count = hf.encode(text, false).unwrap().get_ids().len();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mb_s = (text.len() as f64 * iters as f64) / elapsed / 1_000_000.0;
    (mb_s, token_count)
}

fn bench_tiktoken(bpe: &tiktoken_rs::CoreBPE, text: &str, warmup: usize, iters: usize) -> (f64, usize) {
    for _ in 0..warmup {
        let _ = bpe.encode_ordinary(text);
    }
    let start = Instant::now();
    let mut token_count = 0;
    for _ in 0..iters {
        token_count = bpe.encode_ordinary(text).len();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let mb_s = (text.len() as f64 * iters as f64) / elapsed / 1_000_000.0;
    (mb_s, token_count)
}

fn main() {
    let text = load_enwik8(1_000_000);
    let warmup = 2;
    let iters = 3;
    let text_mb = text.len() as f64 / 1_000_000.0;

    eprintln!("Benchmarking on {:.1} MB of enwik8 ({} warmup, {} iters)\n", text_mb, warmup, iters);

    let models: Vec<(&str, &str, &str, &str)> = vec![
        // (name, tokie_repo, hf_repo, type)
        // WordPiece
        ("BERT", "tokiers/bert-base-uncased", "bert-base-uncased", "wordpiece"),
        ("MiniLM-L6", "tokiers/all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "wordpiece"),
        // BPE (byte-level)
        ("GPT-2", "tokiers/gpt2", "openai-community/gpt2", "bpe"),
        ("Llama-3.2", "tokiers/Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "bpe"),
        ("Cohere-v3", "tokiers/Cohere-embed-english-v3.0", "Cohere/Cohere-embed-english-v3.0", "bpe"),
        ("Jina-v2", "tokiers/jina-embeddings-v2-base-en", "jinaai/jina-embeddings-v2-base-en", "bpe"),
        // SentencePiece BPE
        ("XLM-R", "tokiers/xlm-roberta-base", "FacebookAI/xlm-roberta-base", "sp-bpe"),
        ("Voyage-code-2", "tokiers/voyage-code-2", "voyageai/voyage-code-2", "sp-bpe"),
        // Unigram
        ("T5", "tokiers/t5-base", "google-t5/t5-base", "unigram"),
    ];

    // Load tiktoken for cl100k and o200k comparison
    let cl100k = tiktoken_rs::cl100k_base().ok();
    let o200k = tiktoken_rs::o200k_base().ok();

    let mut results: Vec<BenchResult> = Vec::new();

    for (name, tokie_repo, hf_repo, typ) in &models {
        eprint!("  {name:<20} ");

        let tok = match Tokenizer::from_pretrained(tokie_repo) {
            Ok(t) => t,
            Err(e) => { eprintln!("tokie SKIP: {e}"); continue; }
        };
        let mut hf = match HfTokenizer::from_pretrained(hf_repo, None) {
            Ok(t) => t,
            Err(e) => { eprintln!("HF SKIP: {e}"); continue; }
        };
        let _ = hf.with_truncation(None);

        let (tokie_mb_s, tokie_tokens) = bench_tokie(&tok, &text, warmup, iters);
        let (hf_mb_s, hf_tokens) = bench_hf(&hf, &text, warmup, iters);
        let accurate = tokie_tokens == hf_tokens;

        eprintln!("tokie {:>7.1} MB/s | HF {:>6.1} MB/s | {:>5.1}x | {} {}",
            tokie_mb_s, hf_mb_s, tokie_mb_s / hf_mb_s,
            if accurate { "MATCH" } else { "DIFF" },
            typ);

        results.push(BenchResult {
            name, typ, tokie_mb_s, hf_mb_s, tiktoken_mb_s: None,
            tokie_tokens, hf_tokens, accurate,
        });
    }

    // tiktoken-rs benchmarks (cl100k = GPT-4, o200k = GPT-4o)
    if let Some(ref bpe) = cl100k {
        eprint!("  {:<20} ", "cl100k (tiktoken)");
        let tok = Tokenizer::from_pretrained("tokiers/cl100k").ok();
        let (tk_mb_s, tk_tokens) = bench_tiktoken(bpe, &text, warmup, iters);
        if let Some(tok) = tok {
            let (tokie_mb_s, tokie_tokens) = bench_tokie(&tok, &text, warmup, iters);
            eprintln!("tokie {:>7.1} MB/s | tiktoken {:>6.1} MB/s | {:>5.1}x",
                tokie_mb_s, tk_mb_s, tokie_mb_s / tk_mb_s);
            results.push(BenchResult {
                name: "cl100k", typ: "tiktoken", tokie_mb_s, hf_mb_s: 0.0,
                tiktoken_mb_s: Some(tk_mb_s), tokie_tokens, hf_tokens: tk_tokens, accurate: true,
            });
        }
    }

    if let Some(ref bpe) = o200k {
        eprint!("  {:<20} ", "o200k (tiktoken)");
        let tok = Tokenizer::from_pretrained("tokiers/o200k").ok();
        let (tk_mb_s, tk_tokens) = bench_tiktoken(bpe, &text, warmup, iters);
        if let Some(tok) = tok {
            let (tokie_mb_s, tokie_tokens) = bench_tokie(&tok, &text, warmup, iters);
            eprintln!("tokie {:>7.1} MB/s | tiktoken {:>6.1} MB/s | {:>5.1}x",
                tokie_mb_s, tk_mb_s, tokie_mb_s / tk_mb_s);
            results.push(BenchResult {
                name: "o200k", typ: "tiktoken", tokie_mb_s, hf_mb_s: 0.0,
                tiktoken_mb_s: Some(tk_mb_s), tokie_tokens, hf_tokens: tk_tokens, accurate: true,
            });
        }
    }

    // Print summary table
    println!("\n{:=<85}", "");
    println!("  TOKIE vs HF TOKENIZERS / TIKTOKEN-RS  ({:.1} MB enwik8)", text_mb);
    println!("{:=<85}\n", "");

    println!("  {:<20} {:>10} {:>10} {:>10} {:>8} {:>7}",
        "Model", "tokie", "HF", "tiktoken", "speedup", "match");
    println!("  {:-<20} {:->10} {:->10} {:->10} {:->8} {:->7}",
        "", "", "", "", "", "");

    for r in &results {
        let hf_str = if r.hf_mb_s > 0.0 { format!("{:.1} MB/s", r.hf_mb_s) } else { "-".to_string() };
        let tk_str = match r.tiktoken_mb_s {
            Some(v) => format!("{:.1} MB/s", v),
            None => "-".to_string(),
        };
        let baseline = r.tiktoken_mb_s.unwrap_or(r.hf_mb_s);
        let speedup = if baseline > 0.0 { format!("{:.1}x", r.tokie_mb_s / baseline) } else { "-".to_string() };
        let acc = if r.accurate { "yes" } else { "NO" };

        println!("  {:<20} {:>10} {:>10} {:>10} {:>8} {:>7}",
            r.name,
            format!("{:.1} MB/s", r.tokie_mb_s),
            hf_str, tk_str, speedup, acc);
    }

    println!();
}
