//! Micro-measurements behind the PARALLEL_CHUNK_THRESHOLD choice:
//! thread::scope spawn/join cost, and per-bucket count_tokens vs cached
//! sequential throughput by doc size.
//!
//! Phase-1 numbers (M3, gpt2/OWT): scope spawn+join of 8 workers ~84-143us;
//! with the old 10KB threshold the 10-50KB bucket ran 140.7 MB/s through
//! count_tokens vs 267.3 MB/s for the cached sequential loop.
//!
//! Run: cargo run --release -p tokie --features hf --example profile_spawn_cost

use std::hint::black_box;
use std::time::Instant;

use tokie::encoder::PretokenCache;

fn main() {
    // --- thread::scope spawn cost, 8 workers, trivial work ---
    let cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
    for reps in [100u32, 300] {
        let t0 = Instant::now();
        for _ in 0..reps {
            std::thread::scope(|s| {
                let hs: Vec<_> = (0..cpus).map(|i| s.spawn(move || black_box(i))).collect();
                for h in hs {
                    black_box(h.join().unwrap());
                }
            });
        }
        let el = t0.elapsed().as_secs_f64();
        println!("scope spawn+join x{} threads: {:8.1} us/scope  ({} reps)", cpus, el / reps as f64 * 1e6, reps);
    }

    // --- per-bucket throughput ---
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let text = String::from_utf8_lossy(&data[..data.len().min(10_000_000)]).into_owned();
    let docs: Vec<&str> = text.split("<|endoftext|>").filter(|d| !d.is_empty()).collect();

    let tok = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };
    let enc = tok.encoder().as_backtracking().expect("backtracking");
    let pretok = tok.pretokenizer().expect("pretokenizer");

    let mut cache = PretokenCache::new();
    let mut out: Vec<u32> = Vec::new();
    // warm
    for d in &docs {
        let db = d.as_bytes();
        for p in pretok.split(d) {
            enc.encode_piece_into(db, p.as_bytes(), Some(&mut cache), &mut out);
        }
    }

    let buckets: &[(usize, usize)] = &[(0, 2_000), (2_000, 10_000), (10_000, 50_000), (50_000, 1 << 30)];
    for &(lo, hi) in buckets {
        let sel: Vec<&str> = docs.iter().copied().filter(|d| d.len() >= lo && d.len() < hi).collect();
        if sel.is_empty() {
            continue;
        }
        let nb: usize = sel.iter().map(|d| d.len()).sum();

        // current count_tokens path
        let mut best_ct = f64::INFINITY;
        let mut best_seq = f64::INFINITY;
        for _ in 0..5 {
            let t0 = Instant::now();
            let mut n = 0usize;
            for d in &sel {
                n += tok.count_tokens(d);
            }
            black_box(n);
            best_ct = best_ct.min(t0.elapsed().as_secs_f64());

            let t0 = Instant::now();
            out.clear();
            for d in &sel {
                let db = d.as_bytes();
                for p in pretok.split(d) {
                    enc.encode_piece_into(db, p.as_bytes(), Some(&mut cache), &mut out);
                }
            }
            black_box(out.len());
            best_seq = best_seq.min(t0.elapsed().as_secs_f64());
        }
        println!(
            "bucket {:>6}-{:<9} {:4} docs {:9} B : count_tokens {:7.1} MB/s   seq-cached {:7.1} MB/s",
            lo, hi, sel.len(), nb,
            nb as f64 / 1e6 / best_ct,
            nb as f64 / 1e6 / best_seq,
        );
    }
}
