//! Benchmark pretokenizer throughput on enwik8.
//!
//! Usage: cargo run -p pretokie --example bench_specialized --release

use pretokie::mask::{Gpt2, Cl100k, O200k, Voyage, SmolLM, DeepSeek, Qwen, Bert};
use std::time::Instant;

macro_rules! bench {
    ($name:expr, $ty:ident, $text:expr, $iters:expr, $mb:expr) => {{
        let _ = $ty::new($text).count();
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..$iters {
            count = $ty::new($text).count();
        }
        let elapsed = start.elapsed();
        let throughput = $mb * $iters as f64 / elapsed.as_secs_f64();
        println!("{:10} {:>8.1} MB/s  ({count} pieces)", $name, throughput);
    }};
}

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    let iters = 20;

    bench!("GPT-2:", Gpt2, &text, iters, mb);
    bench!("CL100K:", Cl100k, &text, iters, mb);
    bench!("O200K:", O200k, &text, iters, mb);
    bench!("Voyage:", Voyage, &text, iters, mb);
    bench!("SmolLM:", SmolLM, &text, iters, mb);
    bench!("DeepSeek:", DeepSeek, &text, iters, mb);
    bench!("Qwen3.5:", Qwen, &text, iters, mb);
    bench!("BERT:", Bert, &text, iters, mb);
}
