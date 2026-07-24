//! Single-thread hot-path profile: where does encode time go?
//!
//! Run: cargo run --release -p tokie --features build --example profile_hotpath

use std::time::Instant;

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let text = String::from_utf8_lossy(&data[..data.len().min(10_000_000)]).into_owned();
    let docs: Vec<&str> = text.split("<|endoftext|>").filter(|d| !d.is_empty()).collect();
    let nbytes: usize = docs.iter().map(|d| d.len()).sum();
    println!("{} docs, {:.1} MB", docs.len(), nbytes as f64 / 1e6);

    // Load from a local tokenizer.json (path via TOKIE_JSON env), falling back
    // to the hub. Keeps the profile independent of .tkz format churn.
    let tok = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };

    // Pretokenize only
    let pretok = pretokie::Gpt2::new("");
    drop(pretok);
    let t0 = Instant::now();
    let mut pieces = 0usize;
    for d in &docs {
        pieces += pretokie::Gpt2::new(d).count();
    }
    let el = t0.elapsed().as_secs_f64();
    println!("pretokenize only : {:7.1} MB/s  ({} pieces)", nbytes as f64 / 1e6 / el, pieces);

    // Full single-threaded encode (count path, no Encoding build)
    let t0 = Instant::now();
    let mut toks = 0usize;
    for d in &docs {
        toks += tok.count_tokens(d);
    }
    let el = t0.elapsed().as_secs_f64();
    println!("count_tokens 1T  : {:7.1} MB/s  ({} tokens)", nbytes as f64 / 1e6 / el, toks);

    // Single-threaded encode via the batch hot path (WorkerCaches)
    let t0 = Instant::now();
    let mut cache = tokie::encoder::WorkerCaches::new();
    let mut toks_c = 0usize;
    let pretok = tok.pretokenizer().expect("pretokenizer");
    let mut out: Vec<u32> = Vec::new();
    for d in &docs {
        out.clear();
        let db = d.as_bytes();
        for piece in pretok.split(d) {
            tok.encoder().encode_piece_into(db, piece.as_bytes(), Some(&mut cache), &mut out);
        }
        toks_c += out.len();
    }
    let el = t0.elapsed().as_secs_f64();
    println!("count cached 1T  : {:7.1} MB/s  ({} tokens)", nbytes as f64 / 1e6 / el, toks_c);

    // Single-threaded encode via the bulk for_each_piece drain (the path
    // encode_sequential_into now takes for backtracking encoders).
    let t0 = Instant::now();
    let mut cache_b = tokie::encoder::WorkerCaches::new();
    let mut toks_b = 0usize;
    let mut out_b: Vec<u32> = Vec::new();
    for d in &docs {
        out_b.clear();
        let db = d.as_bytes();
        pretok.for_each_piece(d, |piece| {
            tok.encoder().encode_piece_into(db, piece.as_bytes(), Some(&mut cache_b), &mut out_b);
        });
        toks_b += out_b.len();
    }
    let el = t0.elapsed().as_secs_f64();
    println!("count bulk   1T  : {:7.1} MB/s  ({} tokens)", nbytes as f64 / 1e6 / el, toks_b);

    // Full single-threaded encode (with Encoding build)
    let t0 = Instant::now();
    let mut toks2 = 0usize;
    for d in &docs {
        toks2 += tok.encode(d, false).ids.len();
    }
    let el = t0.elapsed().as_secs_f64();
    println!("encode 1T        : {:7.1} MB/s  ({} tokens)", nbytes as f64 / 1e6 / el, toks2);

    // Batch (all cores)
    let t0 = Instant::now();
    let n: usize = tok.count_tokens_batch(&docs).iter().sum();
    let el = t0.elapsed().as_secs_f64();
    println!("count batch MT   : {:7.1} MB/s  ({} tokens)", nbytes as f64 / 1e6 / el, n);
}
