//! Pretokenize-only throughput per classification backend.
//!
//! Run: cargo run --release -p pretokie --example bench_isa -- [FILE...]
//! (defaults to benches/data/enwik8). Best of 3 runs, MB/s.

use std::time::Instant;

use pretokie::*;

fn best_of_3(mut f: impl FnMut() -> usize) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut n = 0;
    for _ in 0..3 {
        let t0 = Instant::now();
        n = std::hint::black_box(f());
        best = best.min(t0.elapsed().as_secs_f64());
    }
    (best, n)
}

macro_rules! run {
    ($name:expr, $cfg:ty, $text:expr) => {{
        let (name, text): (&str, &str) = ($name, $text);
        let mb = text.len() as f64 / 1e6;
        let (t, core_n) = best_of_3(|| Core::<$cfg>::new(text).count());
        let mut row = format!("{name:<9} core {:7.1}", mb / t);
        for isa in [MaskIsa::Scalar, MaskIsa::Base, MaskIsa::Avx2] {
            if !isa.supported() {
                continue;
            }
            let (ti, ni) = best_of_3(|| Mask::<$cfg>::with_isa(text, isa).count());
            let (tb, nb) = best_of_3(|| {
                let mut n = 0;
                Mask::<$cfg>::with_isa(text, isa).for_each_piece(|_| n += 1);
                n
            });
            assert_eq!((ni, nb), (core_n, core_n), "{name} {isa:?} piece count");
            row += &format!(" | {isa:?} iter {:7.1} bulk {:7.1}", mb / ti, mb / tb);
        }
        println!("{row}");
    }};
}

fn main() {
    let mut files: Vec<String> = std::env::args().skip(1).collect();
    if files.is_empty() {
        files.push(concat!(env!("CARGO_MANIFEST_DIR"), "/../../benches/data/enwik8").into());
    }
    for f in files {
        let data = std::fs::read(&f).unwrap_or_else(|e| panic!("{f}: {e}"));
        let text = String::from_utf8_lossy(&data).into_owned();
        println!("== {f} ({:.1} MB)", text.len() as f64 / 1e6);
        run!("gpt2", Gpt2Config, &text);
        run!("cl100k", Cl100kConfig, &text);
        run!("o200k", O200kConfig, &text);
        run!("voyage", VoyageConfig, &text);
        run!("smollm", SmolLMConfig, &text);
        run!("deepseek", DeepSeekConfig, &text);
        run!("qwen", QwenConfig, &text);
    }
}
