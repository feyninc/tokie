//! Isolated miss-path benchmark: ns/piece for DAAC walk vs rank-merge core.
//!
//! Population: unique pretokenized pieces from 10MB of OWT that would MISS
//! the token_cache early exit (multi-token encoding or longer than 16 bytes),
//! capped at 32 bytes — i.e., exactly the pieces the new path serves.
//!
//! Run: TOKIE_JSON=<gpt2 tokenizer.json> \
//!      cargo run --release -p tokie --features hf --example rank_merge_bench

use std::collections::HashSet;
use std::time::Instant;

fn bench_tokenizer(name: &str, tok: &tokie::Tokenizer, docs: &[&str]) {
    let enc = tok.encoder().as_backtracking().expect("backtracking encoder");
    let pretok = tok.pretokenizer().expect("pretokenizer");

    // Collect the miss-path piece population.
    let mut seen: HashSet<Vec<u8>> = HashSet::new();
    let mut pieces: Vec<Vec<u8>> = Vec::new();
    let mut tmp = Vec::new();
    for d in docs {
        for piece in pretok.split(d) {
            let b = piece.as_bytes();
            if b.is_empty() || b.len() > 32 {
                continue;
            }
            if !seen.insert(b.to_vec()) {
                continue;
            }
            tmp.clear();
            enc.encode_sequential_into(b, &mut tmp);
            // token_cache hit (single token, <=16 bytes) never reaches the miss path
            if tmp.len() == 1 && b.len() <= 16 {
                continue;
            }
            pieces.push(b.to_vec());
        }
    }
    let total_bytes: usize = pieces.iter().map(|p| p.len()).sum();
    println!(
        "{name}: {} miss-path pieces, mean {:.1} bytes",
        pieces.len(),
        total_bytes as f64 / pieces.len() as f64
    );

    const REPS: usize = 20;
    let mut out: Vec<u32> = Vec::with_capacity(1 << 20);

    let mut run = |label: &str, f: &dyn Fn(&[u8], &mut Vec<u32>)| {
        // warmup
        out.clear();
        for p in &pieces {
            f(p, &mut out);
        }
        let t0 = Instant::now();
        let mut n_tok = 0usize;
        for _ in 0..REPS {
            out.clear();
            for p in &pieces {
                f(p, &mut out);
            }
            n_tok += out.len();
        }
        let el = t0.elapsed();
        let ns_per_piece = el.as_nanos() as f64 / (REPS * pieces.len()) as f64;
        println!(
            "  {label:20} {ns_per_piece:7.1} ns/piece  ({:.0} MB/s piece bytes, {} tokens)",
            (REPS * total_bytes) as f64 / el.as_secs_f64() / 1e6,
            n_tok / REPS
        );
    };

    run("daac backtracking", &|p, o| enc.encode_sequential_into(p, o));
    if enc.has_rank_merge() {
        run("rank-merge (dense)", &|p, o| enc.encode_rank_merge(p, o));
        run("rank-merge (flat)", &|p, o| enc.encode_rank_merge_flat(p, o));
    }
}

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let text = String::from_utf8_lossy(&data[..data.len().min(10_000_000)]).into_owned();
    let docs: Vec<&str> = text
        .split("<|endoftext|>")
        .filter(|d| !d.is_empty())
        .collect();

    let gpt2 = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };
    bench_tokenizer("gpt2", &gpt2, &docs);

    let dsv3 = tokie::Tokenizer::from_pretrained("tokiers/DeepSeek-V3").unwrap();
    bench_tokenizer("DeepSeek-V3", &dsv3, &docs);
}
