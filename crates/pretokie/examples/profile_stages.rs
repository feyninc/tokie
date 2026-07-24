//! Phase-1 stage profiler for the mask-scanner pipeline.
//!
//! Splits the per-64-byte-batch cost into stages:
//!   (a) NEON class-mask computation (ascii_masks + movemasks)
//!   (b) full boundary algebra (batch_masks = a + boundary bits)
//!   (c) bit-pop walker + per-piece bookkeeping (Mask iterator)
//!   (d) scalar bad-zone fallback frequency and cost
//! plus the for_each_piece / iterator callback boundary.
//!
//! Run: cargo run -p pretokie --release --example profile_stages

use pretokie::bench_internal as bi;
use pretokie::{Core, Gpt2};
use pretokie::Gpt2Config;
use std::time::Instant;

type C = Gpt2Config;

fn read_owt() -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing (symlink it into benches/data/)");
    let cap = std::env::var("PROFILE_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(20) * 1_000_000;
    String::from_utf8_lossy(&data[..data.len().min(cap)]).into_owned()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let text = read_owt();
    let bytes = text.as_bytes();
    let n = bytes.len();
    let nbatches = if n >= 64 { n - 64 + 1 } else { 0 }; // every aligned base we can classify
    // Use grid-aligned batch bases (scan = 0, 64, 128, ...), matching the walker.
    let grid_bases: Vec<usize> = (0..).map(|i| i * 64).take_while(|&b| b + 65 <= n).collect();
    let nb = grid_bases.len();
    println!("corpus: {:.1} MB, {} grid batches (64B, +1 lookahead)", n as f64 / 1e6, nb);
    println!();

    let reps = 7;

    // ---- Stage (a): NEON classification + movemasks only ----
    let mut a_ns = Vec::new();
    let mut sink = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        for &b in &grid_bases {
            sink ^= bi::classify_fold::<C>(bytes, b);
        }
        a_ns.push(t.elapsed().as_secs_f64() * 1e9 / nb as f64);
    }
    let a = median(a_ns);
    std::hint::black_box(sink);

    // ---- Stage (a+b): full boundary algebra ----
    let mut ab_ns = Vec::new();
    let mut sink2 = 0u64;
    for _ in 0..reps {
        let t = Instant::now();
        for &b in &grid_bases {
            let (u, bad) = bi::batch_masks::<C>(bytes, b);
            sink2 ^= u ^ bad;
        }
        ab_ns.push(t.elapsed().as_secs_f64() * 1e9 / nb as f64);
    }
    let ab = median(ab_ns);
    std::hint::black_box(sink2);

    // ---- Bad-zone frequency and per-piece counts ----
    let mut clean_batches = 0usize;
    let mut dirty_batches = 0usize;
    let mut bad_bits_total = 0u64;
    for &b in &grid_bases {
        let (_u, bad) = bi::batch_masks::<C>(bytes, b);
        if bad == 0 { clean_batches += 1; } else { dirty_batches += 1; bad_bits_total += bad.count_ones() as u64; }
    }

    // ---- Stage (c+d): full Mask iterator (walker + bad zones + tail) ----
    let mut pieces = 0usize;
    let mut mask_ns = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let mut p = 0usize;
        for _ in Gpt2::new(&text) { p += 1; }
        mask_ns.push(t.elapsed().as_secs_f64() * 1e9);
        pieces = p;
    }
    let mask_total = median(mask_ns);
    let ns_per_piece_mask = mask_total / pieces as f64;
    let ns_per_batch_mask = mask_total / nb as f64;

    // ---- Pure scalar Core iterator (ground truth, bad-zone executor cost) ----
    let mut core_pieces = 0usize;
    let mut core_ns = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let mut p = 0usize;
        for _ in Core::<C>::new(&text) { p += 1; }
        core_ns.push(t.elapsed().as_secs_f64() * 1e9);
        core_pieces = p;
    }
    let core_total = median(core_ns);
    assert_eq!(core_pieces, pieces, "core vs mask piece count mismatch");

    // ---- Measure scalar_advance cost per call (amortized over pieces) ----
    // Drive scalar_advance across the whole corpus, one piece at a time.
    let mut adv_ns = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        let mut pos = 0usize;
        let mut cnt = 0usize;
        while pos < n {
            pos = bi::scalar_advance::<C>(&text, pos);
            cnt += 1;
        }
        std::hint::black_box(cnt);
        adv_ns.push(t.elapsed().as_secs_f64() * 1e9 / cnt as f64);
    }
    let adv_per_piece = median(adv_ns);

    // ---- for_each-style callback vs Iterator::next boundary ----
    // Iterator collect-count is `mask_total`. Compare to a raw for-loop that
    // sums lengths (callback consumer inlined).
    let mut cb_ns = Vec::new();
    let mut cb_sum = 0usize;
    for _ in 0..reps {
        let t = Instant::now();
        let mut s = 0usize;
        for piece in Gpt2::new(&text) { s += piece.len(); }
        cb_ns.push(t.elapsed().as_secs_f64() * 1e9);
        cb_sum = s;
    }
    let cb_total = median(cb_ns);
    std::hint::black_box(cb_sum);

    // ---- bulk-drain for_each_piece (sum len) ----
    let mut fe_ns = Vec::new();
    let mut fe_sum = 0usize;
    let mut fe_pieces = 0usize;
    for _ in 0..reps {
        let t = Instant::now();
        let mut s = 0usize;
        let mut p = 0usize;
        Gpt2::new(&text).for_each_piece(|piece| { s += piece.len(); p += 1; });
        fe_ns.push(t.elapsed().as_secs_f64() * 1e9);
        fe_sum = s;
        fe_pieces = p;
    }
    let fe_total = median(fe_ns);
    std::hint::black_box(fe_sum);
    assert_eq!(fe_pieces, pieces, "for_each piece count mismatch");

    let mb = n as f64 / 1e6;
    println!("=== per-batch stage costs (ns / 64B grid batch) ===");
    println!("(a) NEON classify+movemask : {:6.2} ns/batch   ({:6.1} MB/s isolated)", a, mb / (a * nb as f64 / 1e9) / 1.0);
    println!("(b) boundary algebra (a+b) : {:6.2} ns/batch   (b alone ~{:.2})", ab, ab - a);
    println!("(c+d) full Mask walker     : {:6.2} ns/batch", ns_per_batch_mask);
    println!();
    println!("=== per-piece costs (ns / piece) ===");
    println!("total pieces               : {}", pieces);
    println!("bytes/piece                : {:.2}", n as f64 / pieces as f64);
    println!("full Mask iterator         : {:6.3} ns/piece", ns_per_piece_mask);
    println!("full Core (scalar) iterator: {:6.3} ns/piece", core_total / pieces as f64);
    println!("scalar_advance (isolated)  : {:6.3} ns/piece", adv_per_piece);
    println!();
    println!("=== bad-zone frequency ===");
    println!("clean batches              : {} ({:.2}%)", clean_batches, 100.0 * clean_batches as f64 / nb as f64);
    println!("dirty batches (any bad)    : {} ({:.2}%)", dirty_batches, 100.0 * dirty_batches as f64 / nb as f64);
    println!("avg bad bits / dirty batch : {:.1}", bad_bits_total as f64 / dirty_batches.max(1) as f64);
    println!();
    println!("=== throughput ===");
    println!("pretokenize-only (Mask it) : {:7.1} MB/s  ({:.3} ns/piece)", mb / (mask_total / 1e9), mask_total / pieces as f64);
    println!("for_each_piece (bulk drain): {:7.1} MB/s  ({:.3} ns/piece)", mb / (fe_total / 1e9), fe_total / pieces as f64);
    println!("iter callback (sum len)    : {:7.1} MB/s", mb / (cb_total / 1e9));
    println!("boundary-algebra only      : {:7.1} MB/s (a+b over all batches)", mb / (ab * nb as f64 / 1e9));
    let _ = nbatches;
}
