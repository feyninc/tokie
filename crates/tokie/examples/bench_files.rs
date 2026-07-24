//! Single timed encode_files_flat call in a fresh process (harness contract).
//! Run: cargo run --release -p tokie --features build --example bench_files -- <file>

use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench_files <file>");
    let tok = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };
    let nbytes = std::fs::metadata(&path).unwrap().len() as f64;
    let t0 = Instant::now();
    let (ids, offs) = tok.encode_files_flat(&[&path], b"<|endoftext|>", false).unwrap();
    let el = t0.elapsed().as_secs_f64();
    println!(
        "encode_files: {:6.1} ms ({:.0} MB/s, {} tokens, {} docs)",
        el * 1e3,
        nbytes / 1e6 / el,
        ids.len(),
        offs.len() - 1
    );
}
