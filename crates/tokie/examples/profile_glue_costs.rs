//! Phase-1 glue-cost decomposition for the single-thread bulk pipeline.
//!
//! Prior cycle accounting (profile_piece_costs) put the components at
//! ~4 ns/piece pretokenize + ~5.8 ns/piece warm encode, yet the combined
//! loop (profile_hotpath "count cached 1T") runs ~18-22 ns/piece. This
//! harness measures where the missing ~8-12 ns/piece goes:
//!   (A) pretokenize-only, monomorphized mask-scanner walk
//!   (B) pretokenize-only through the Pretokenizer enum iterator
//!   (C) walker -> 64-span block fill, no encode
//!   (D) encode-only replay of a recorded piece stream (max ILP)
//!   (E) fused naive loop, enum pretok + Encoder enum (current pipeline)
//!   (F) fused naive loop, monomorphized pretok + direct encoder
//!   (G) fused block loop, monomorphized: 64-span blocks -> probe loop
//!   (H) count_tokens full path (per-doc lease + alloc + normalizer)
//!
//! Run: cargo run --release -p tokie --features hf --example profile_glue_costs
//!      (TOKIE_JSON=path/to/tokenizer.json for non-gpt2 models)

use std::hint::black_box;
use std::time::Instant;

use tokie::encoder::PretokenCache;
use tokie::pretok::PretokType;

const REPS: usize = 5;
const BLOCK: usize = 64;

fn best_ns<F: FnMut() -> u64>(mut f: F) -> (f64, u64) {
    let mut best = f64::INFINITY;
    let mut check = 0u64;
    for _ in 0..REPS {
        let t0 = Instant::now();
        check = f();
        let el = t0.elapsed().as_secs_f64() * 1e9;
        if el < best {
            best = el;
        }
    }
    (best, check)
}

/// Dispatch a generic body over the monomorphized mask-scanner type for
/// the tokenizer's pretok config. `$f` is a generic fn taking a
/// make-iterator closure.
macro_rules! dispatch_mask {
    ($ty:expr, $f:expr) => {
        match $ty {
            PretokType::Gpt2 => $f(|d| pretokie::Gpt2::new(d)),
            PretokType::Cl100k => $f(|d| pretokie::Cl100k::new(d)),
            PretokType::O200k => $f(|d| pretokie::O200k::new(d)),
            PretokType::Voyage => $f(|d| pretokie::Voyage::new(d)),
            PretokType::SmolLM => $f(|d| pretokie::SmolLM::new(d)),
            PretokType::DeepSeek => $f(|d| pretokie::DeepSeek::new(d)),
            PretokType::Qwen35 => $f(|d| pretokie::Qwen::new(d)),
            other => panic!("no mask scanner for {:?}", other),
        }
    };
}

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap().parent().unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let text = String::from_utf8_lossy(&data[..data.len().min(10_000_000)]).into_owned();
    let docs: Vec<&str> = text.split("<|endoftext|>").filter(|d| !d.is_empty()).collect();
    let nbytes: usize = docs.iter().map(|d| d.len()).sum();

    let tok = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };
    let enc = tok.encoder().as_backtracking().expect("backtracking encoder");
    let encoder_enum = tok.encoder();
    let pretok = tok.pretokenizer().expect("pretokenizer");
    let ptype = tok.pretokenizer_type();

    // Recorded piece stream (excludes pretokenizer cost on replay).
    let mut pieces: Vec<&[u8]> = Vec::new();
    for d in &docs {
        for p in pretok.split(d) {
            pieces.push(p.as_bytes());
        }
    }
    let np = pieces.len();
    println!("{} docs, {:.1} MB, {} pieces ({:.2} B/piece), pretok {:?}",
        docs.len(), nbytes as f64 / 1e6, np, nbytes as f64 / np as f64, ptype);

    let mbps = |ns: f64| nbytes as f64 / 1e6 / (ns / 1e9);
    let perp = |ns: f64| ns / np as f64;

    // Warm cache once.
    let mut cache = PretokenCache::new();
    let mut out: Vec<u32> = Vec::with_capacity(4 * np);
    for d in &docs {
        let db = d.as_bytes();
        for p in pretok.split(d) {
            enc.encode_piece_into(db, p.as_bytes(), Some(&mut cache), &mut out);
        }
    }
    let ntokens = out.len();
    println!("{} tokens ({:.3} tok/piece)\n", ntokens, ntokens as f64 / np as f64);

    // --- A: pretokenize only, monomorphized mask scanner ---
    let (ns, chk) = dispatch_mask!(ptype, |make: _| {
        fn run<'a, I: Iterator<Item = &'a str>, F: Fn(&'a str) -> I>(
            docs: &[&'a str], make: F,
        ) -> (f64, u64) {
            best_ns(|| {
                let mut n = 0u64;
                for d in docs {
                    for p in make(d) {
                        n += p.len() as u64;
                    }
                }
                black_box(n)
            })
        }
        run(&docs, make)
    });
    println!("A  pretok only, monomorphized     : {:5.2} ns/piece  {:7.1} MB/s  [chk {}]", perp(ns), mbps(ns), chk);

    // --- B: pretokenize only, enum iterator (Pretokenizer::split) ---
    let (ns, chk) = best_ns(|| {
        let mut n = 0u64;
        for d in &docs {
            for p in pretok.split(d) {
                n += p.len() as u64;
            }
        }
        black_box(n)
    });
    println!("B  pretok only, enum iter         : {:5.2} ns/piece  {:7.1} MB/s  [chk {}]", perp(ns), mbps(ns), chk);

    // --- C: walker -> 64-span block fill, no encode ---
    let (ns, chk) = dispatch_mask!(ptype, |make: _| {
        fn run<'a, I: Iterator<Item = &'a str>, F: Fn(&'a str) -> I>(
            docs: &[&'a str], make: F,
        ) -> (f64, u64) {
            best_ns(|| {
                let mut acc = 0u64;
                let mut spans = [(0u32, 0u32); BLOCK];
                for d in docs {
                    let base = d.as_ptr() as usize;
                    let mut it = make(d);
                    loop {
                        let mut n = 0;
                        while n < BLOCK {
                            match it.next() {
                                Some(p) => {
                                    spans[n] = ((p.as_ptr() as usize - base) as u32, p.len() as u32);
                                    n += 1;
                                }
                                None => break,
                            }
                        }
                        for &(o, l) in &spans[..n] {
                            acc = acc.wrapping_add(o as u64) ^ l as u64;
                        }
                        if n < BLOCK {
                            break;
                        }
                    }
                }
                black_box(acc)
            })
        }
        run(&docs, make)
    });
    println!("C  block fill only (64 spans)     : {:5.2} ns/piece  {:7.1} MB/s  [chk {}]", perp(ns), mbps(ns), chk & 0xffff);

    // --- D: encode-only replay of recorded pieces, warm cache ---
    let text_bytes = text.as_bytes();
    let (ns, chk) = best_ns(|| {
        out.clear();
        for &p in &pieces {
            enc.encode_piece_into(text_bytes, p, Some(&mut cache), &mut out);
        }
        black_box(out.len() as u64)
    });
    println!("D  encode-only replay, warm       : {:5.2} ns/piece  {:7.1} MB/s  [toks {}]", perp(ns), mbps(ns), chk);

    // --- E: fused naive, enum pretok + Encoder enum (current pipeline) ---
    let mut wcache = tokie::encoder::WorkerCaches::new();
    let (ns, chk) = best_ns(|| {
        out.clear();
        for d in &docs {
            let db = d.as_bytes();
            for p in pretok.split(d) {
                encoder_enum.encode_piece_into(db, p.as_bytes(), Some(&mut wcache), &mut out);
            }
        }
        black_box(out.len() as u64)
    });
    println!("E  fused naive, enum+enum         : {:5.2} ns/piece  {:7.1} MB/s  [toks {}]", perp(ns), mbps(ns), chk);

    // --- F: fused naive, monomorphized pretok + direct encoder ---
    let (ns, chk) = dispatch_mask!(ptype, |make: _| {
        fn run<'a, I: Iterator<Item = &'a str>, F: Fn(&'a str) -> I>(
            docs: &[&'a str], make: F,
            enc: &tokie::encoder::BacktrackingBytePairEncoder,
            cache: &mut PretokenCache, out: &mut Vec<u32>,
        ) -> (f64, u64) {
            best_ns(|| {
                out.clear();
                for d in docs {
                    let db = d.as_bytes();
                    for p in make(d) {
                        enc.encode_piece_into(db, p.as_bytes(), Some(cache), out);
                    }
                }
                black_box(out.len() as u64)
            })
        }
        run(&docs, make, enc, &mut cache, &mut out)
    });
    println!("F  fused naive, monomorphized     : {:5.2} ns/piece  {:7.1} MB/s  [toks {}]", perp(ns), mbps(ns), chk);

    // --- G: fused block, monomorphized ---
    let (ns, chk) = dispatch_mask!(ptype, |make: _| {
        fn run<'a, I: Iterator<Item = &'a str>, F: Fn(&'a str) -> I>(
            docs: &[&'a str], make: F,
            enc: &tokie::encoder::BacktrackingBytePairEncoder,
            cache: &mut PretokenCache, out: &mut Vec<u32>,
        ) -> (f64, u64) {
            best_ns(|| {
                out.clear();
                let mut spans = [(0u32, 0u32); BLOCK];
                for d in docs {
                    let db = d.as_bytes();
                    let base = db.as_ptr() as usize;
                    let mut it = make(d);
                    loop {
                        let mut n = 0;
                        while n < BLOCK {
                            match it.next() {
                                Some(p) => {
                                    spans[n] = ((p.as_ptr() as usize - base) as u32, p.len() as u32);
                                    n += 1;
                                }
                                None => break,
                            }
                        }
                        for &(o, l) in &spans[..n] {
                            let piece = &db[o as usize..(o + l) as usize];
                            enc.encode_piece_into(db, piece, Some(cache), out);
                        }
                        if n < BLOCK {
                            break;
                        }
                    }
                }
                black_box(out.len() as u64)
            })
        }
        run(&docs, make, enc, &mut cache, &mut out)
    });
    println!("G  fused block (64), monomorph    : {:5.2} ns/piece  {:7.1} MB/s  [toks {}]", perp(ns), mbps(ns), chk);

    // --- H: count_tokens full path ---
    let (ns, chk) = best_ns(|| {
        let mut n = 0u64;
        for d in &docs {
            n += tok.count_tokens(d) as u64;
        }
        black_box(n)
    });
    println!("H  count_tokens full path         : {:5.2} ns/piece  {:7.1} MB/s  [toks {}]", perp(ns), mbps(ns), chk);

    // --- H decomposition: per-doc overheads ---
    // H2: added-token scan (split_on_added over the whole doc)
    let (ns, chk) = best_ns(|| {
        let mut n = 0u64;
        for d in &docs {
            n += tok.debug_split_added(d).len() as u64;
        }
        black_box(n)
    });
    println!("H2 added-token scan, per doc      : {:5.2} ns/piece  {:7.1} MB/s  [splits {}]", perp(ns), mbps(ns), chk);

    // H3: normalizer pass
    let (ns, chk) = best_ns(|| {
        let mut n = 0u64;
        for d in &docs {
            n += tok.normalizer().normalize(d).len() as u64;
        }
        black_box(n)
    });
    println!("H3 normalize, per doc             : {:5.2} ns/piece  {:7.1} MB/s  [chk {}]", perp(ns), mbps(ns), chk & 0xffffff);

    // H4: per-doc Vec::with_capacity(len/3) alloc+drop
    let (ns, chk) = best_ns(|| {
        let mut n = 0u64;
        for d in &docs {
            let v: Vec<u32> = Vec::with_capacity(d.len() / 3);
            n += v.capacity() as u64;
        }
        black_box(n)
    });
    println!("H4 per-doc Vec alloc              : {:5.2} ns/piece  {:7.1} MB/s  [chk {}]", perp(ns), mbps(ns), chk & 0xffffff);

    println!("\nderived:");
    println!("  enum-iter tax        = B - A");
    println!("  block-fill tax       = C - A");
    println!("  interleave glue      = E - (B + D)  [what fusion must recover]");
    println!("  monomorphization win = E - F");
    println!("  block-processing win = F - G");
    println!("  per-doc overhead     = H - E");
}
