//! Load-time breakdown: where do the milliseconds go on from_json?
//!
//! Run: TOKIE_JSON=path cargo run --release -p tokie --features hf --example profile_load

use std::time::Instant;

fn main() {
    let path = std::env::var("TOKIE_JSON").expect("set TOKIE_JSON");

    // Raw file read + JSON parse
    let t0 = Instant::now();
    let raw = std::fs::read_to_string(&path).unwrap();
    let read_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t0 = Instant::now();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let parse_ms = t0.elapsed().as_secs_f64() * 1e3;
    drop(v);

    // Full from_json, repeated for a warm number
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let tok = tokie::Tokenizer::from_json(&path).unwrap();
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&tok);
    }

    // Save to .tkz and reload
    let tkz = std::env::temp_dir().join("profile_load.tkz");
    let tok = tokie::Tokenizer::from_json(&path).unwrap();
    tok.to_file(&tkz).unwrap();
    let tkz_size = std::fs::metadata(&tkz).unwrap().len();
    let mut best_tkz = f64::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let tok = tokie::Tokenizer::from_file(&tkz).unwrap();
        best_tkz = best_tkz.min(t0.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(&tok);
    }

    println!("file read       : {read_ms:7.1} ms");
    println!("serde_json parse: {parse_ms:7.1} ms");
    println!("from_json total : {best:7.1} ms");
    println!("from_file (.tkz): {best_tkz:7.1} ms  ({:.1} MB)", tkz_size as f64 / 1e6);
}
