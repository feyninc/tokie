//! Unigram throughput on current tokie (T5, XLM-R).
//!
//! Run: cargo run --release -p tokie --features hf --example bench_unigram_throughput

use std::path::Path;
use std::time::Instant;
use tokie::{EncoderType, Tokenizer};

fn load_text(max_bytes: usize) -> (String, &'static str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data");
    // Prefer OWT sample (web text); fall back to enwik8.
    for (file, label) in [("owt_sample.txt", "owt"), ("enwik8", "enwik8")] {
        let path = root.join(file);
        if let Ok(data) = std::fs::read(&path) {
            let n = data.len().min(max_bytes);
            let text = String::from_utf8_lossy(&data[..n]).into_owned();
            return (text, label);
        }
    }
    panic!("no benches/data/{{owt_sample.txt,enwik8}}");
}

fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn bench_once(label: &str, bytes: usize, f: impl Fn()) -> f64 {
    let t0 = Instant::now();
    f();
    let el = t0.elapsed().as_secs_f64().max(1e-12);
    let mbs = bytes as f64 / 1e6 / el;
    println!("  {label:<22} {mbs:7.1} MB/s  ({el:.3}s)");
    mbs
}

fn bench_model(name: &str, repo: &str, text: &str, docs: &[&str]) {
    println!("\n=== {name} ({repo}) ===");
    let tok = Tokenizer::from_pretrained(repo).unwrap_or_else(|e| panic!("load {repo}: {e}"));
    println!(
        "  encoder={:?}  pretok={}  text={:.2} MB  docs={}",
        tok.encoder_type(),
        if tok.pretokenizer().is_some() {
            "yes"
        } else {
            "none (metaspace/normalize path)"
        },
        text.len() as f64 / 1e6,
        docs.len()
    );
    assert_eq!(
        tok.encoder_type(),
        EncoderType::Unigram,
        "{name} should load as Unigram"
    );

    // Warmup
    for _ in 0..2 {
        let _ = tok.encode(text, false);
        let _ = tok.count_tokens(text);
        let _ = tok.count_tokens_batch(docs);
    }

    let nbytes = text.len();
    let doc_bytes: usize = docs.iter().map(|d| d.len()).sum();

    let mut count_1t = Vec::new();
    let mut encode_1t = Vec::new();
    let mut count_mt = Vec::new();
    let mut encode_batch = Vec::new();
    let mut flat_batch = Vec::new();

    for round in 1..=5 {
        print!("  round {round}:\n");
        count_1t.push(bench_once("count_tokens 1T", nbytes, || {
            let _ = tok.count_tokens(text);
        }));
        encode_1t.push(bench_once("encode 1T", nbytes, || {
            let _ = tok.encode(text, false);
        }));
        count_mt.push(bench_once("count_batch MT", doc_bytes, || {
            let _ = tok.count_tokens_batch(docs);
        }));
        encode_batch.push(bench_once("encode_batch MT", doc_bytes, || {
            let _ = tok.encode_batch(docs, false);
        }));
        flat_batch.push(bench_once("encode_batch_flat", doc_bytes, || {
            let _ = tok.encode_batch_flat(docs, false);
        }));
    }

    println!("  --- medians of 5 ---");
    println!("  count_tokens 1T      {:7.1} MB/s", median(&mut count_1t));
    println!("  encode 1T            {:7.1} MB/s", median(&mut encode_1t));
    println!("  count_batch MT       {:7.1} MB/s", median(&mut count_mt));
    println!("  encode_batch MT      {:7.1} MB/s", median(&mut encode_batch));
    println!("  encode_batch_flat    {:7.1} MB/s", median(&mut flat_batch));

    // Token count sanity
    let n = tok.count_tokens(text);
    println!("  tokens on full text: {n}");
}

fn main() {
    let (text, corpus) = load_text(10_000_000);
    let docs: Vec<&str> = if corpus == "owt" {
        text.split("<|endoftext|>")
            .filter(|d| !d.is_empty())
            .collect()
    } else {
        // Fake docs for enwik8: ~4KB chunks
        text.as_bytes()
            .chunks(4096)
            .map(|c| std::str::from_utf8(c).unwrap_or(""))
            .filter(|s| !s.is_empty())
            .collect()
    };
    // Re-borrow docs from text for owt; for enwik8 chunks are sub-slices of text so OK.
    let docs = docs;

    println!(
        "corpus={corpus}  bytes={}  docs={}  cpus={:?}",
        text.len(),
        docs.len(),
        std::thread::available_parallelism()
    );

    for (name, repo) in [
        ("T5-base", "tokiers/t5-base"),
        ("XLM-RoBERTa", "tokiers/xlm-roberta-base"),
    ] {
        bench_model(name, repo, &text, &docs);
    }
}
