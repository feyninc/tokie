//! Single-thread `encode_ids` throughput on 10 KiB chunks (tokbench's
//! layout): one warm-up pass, then best of `REPS` passes. Prints MB/s and
//! an FNV-1a hash of every id, so two builds can be compared for speed and
//! exactness on the same input.
//!
//! - `PE_TB=1`: first replay tokbench's `measure encode` (warm on slice 0
//!   of 6, time the 5 disjoint slices once each, median), printed as
//!   `<corpus>-tb`. Those slices are unseen text, so cache misses count;
//!   the warm number above barely has any.
//! - `PE_SPLIT=1`: also time the pretokenizer alone (block drain), in
//!   ns/piece next to the full encode.
//!
//! Run: cargo run --release -p tokie --features hf --example perf_encode -- <tokenizer.json> <corpus.txt>...

use std::hint::black_box;
use std::time::Instant;

const CHUNK: usize = 10 * 1024;
const REPS: usize = 7;

fn chunks(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut s = 0;
    while s < text.len() {
        let mut e = (s + CHUNK).min(text.len());
        while !text.is_char_boundary(e) {
            e -= 1;
        }
        out.push(&text[s..e]);
        s = e;
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let tok = tokie::Tokenizer::from_json(&args[0]).expect("tokenizer.json");
    for path in &args[1..] {
        let raw = std::fs::read(path).expect("corpus");
        let text = String::from_utf8_lossy(&raw[..raw.len().min(1 << 20)]).into_owned();
        let docs = chunks(&text);
        let name = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
        if std::env::var_os("PE_TB").is_some() {
            // tokbench's `measure`: warm on slice 0 of 6, then time each of
            // the 5 disjoint slices once (unseen text, so cache misses
            // count) and take the median s/byte. Fresh tokenizer per corpus.
            let tok = tokie::Tokenizer::from_json(&args[0]).expect("tokenizer.json");
            let per = docs.len() / 6;
            let slice = |i: usize| &docs[i * per..(i + 1) * per];
            for d in slice(0) {
                black_box(tok.encode_ids(d, false));
            }
            let mut spb = Vec::new();
            for r in 1..6 {
                let s = slice(r);
                let bytes: usize = s.iter().map(|d| d.len()).sum();
                let t = Instant::now();
                for d in s {
                    black_box(tok.encode_ids(d, false));
                }
                spb.push(t.elapsed().as_secs_f64() / bytes as f64);
            }
            spb.sort_by(f64::total_cmp);
            println!("{:>12}  {:7.1} MB/s  hash tb", format!("{name}-tb"), 1e-6 / spb[2]);
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for d in &docs {
            for id in tok.encode_ids(d, false) {
                for b in id.to_le_bytes() {
                    hash = (hash ^ b as u64).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        let mut best = f64::INFINITY;
        for _ in 0..REPS {
            let t = Instant::now();
            let mut n = 0usize;
            for d in &docs {
                n += black_box(tok.encode_ids(d, false)).len();
            }
            black_box(n);
            best = best.min(t.elapsed().as_secs_f64());
        }
        if std::env::var_os("PE_SPLIT").is_some() {
            let pt = tok.pretokenizer().unwrap();
            let mut pbest = f64::INFINITY;
            let mut pieces = 0usize;
            for _ in 0..REPS {
                let t = Instant::now();
                pieces = 0;
                for d in &docs {
                    pt.for_each_block(d, &mut |_, e| pieces += black_box(e).len());
                }
                pbest = pbest.min(t.elapsed().as_secs_f64());
            }
            eprintln!("  pretok {:7.1} MB/s  {:.2} ns/piece  total {:.2} ns/piece  pieces {pieces}",
                text.len() as f64 / 1e6 / pbest, pbest * 1e9 / pieces as f64, best * 1e9 / pieces as f64);
        }
        println!("{name:>12}  {:7.1} MB/s  hash {hash:016x}", text.len() as f64 / 1e6 / best);
    }
}
