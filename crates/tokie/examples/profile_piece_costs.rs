//! Cycle accounting for the per-piece encode step (phase-1 harness).
//!
//! Replays the recorded gpt2 piece stream from benches/data/owt_sample.txt
//! and measures, in isolation:
//!   (a) PretokenCache hit rate and per-hit cost (pre-warmed cache)
//!   (b) per-miss cost, split into token_cache path and full backtrack path
//!   (c) fixed loop overhead (piece iteration + out push) with a no-op encoder
//!
//! Run: TOKIE_JSON=... cargo run --release -p tokie --features hf --example profile_piece_costs

use std::hint::black_box;
use std::time::Instant;

use tokie::encoder::PretokenCache;

const REPS: usize = 5;

/// Time `f` REPS times, return best-of ns total.
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
    let pretok = tok.pretokenizer().expect("pretokenizer");

    // Record the piece stream once so replay excludes pretokenizer cost.
    let mut pieces: Vec<&[u8]> = Vec::new();
    for d in &docs {
        for p in pretok.split(d) {
            if !p.is_empty() {
                pieces.push(p.as_bytes());
            }
        }
    }
    let np = pieces.len();
    println!("{} docs, {:.1} MB, {} pieces ({:.2} B/piece avg)",
        docs.len(), nbytes as f64 / 1e6, np, nbytes as f64 / np as f64);

    // --- Warm the cache with a full pass ---
    let mut cache = PretokenCache::new();
    let mut out: Vec<u32> = Vec::with_capacity(4 * np);
    for &p in &pieces {
        enc.encode_into(p, Some(&mut cache), &mut out);
    }
    let ntokens = out.len();
    println!("{} tokens ({:.3} tok/piece avg)", ntokens, ntokens as f64 / np as f64);

    // --- Classification pass (warm cache, untimed) ---
    let mut hits = 0usize;
    let mut miss_too_long = 0usize; // len > KEY_MAX, never cacheable
    let mut miss_tokcache = 0usize; // missed pretoken cache, single-token via token_cache
    let mut miss_backtrack = 0usize; // full backtracking path
    let mut miss_too_many_toks = 0usize; // of backtrack misses: piece encodes to >MAX_TOKENS
    let mut hit_pieces: Vec<&[u8]> = Vec::new();
    let mut tokcache_pieces: Vec<&[u8]> = Vec::new();
    let mut backtrack_pieces: Vec<&[u8]> = Vec::new();
    let mut long_pieces: Vec<&[u8]> = Vec::new();
    let mut scratch: Vec<u32> = Vec::with_capacity(64);
    for &p in &pieces {
        if p.len() <= PretokenCache::KEY_MAX {
            scratch.clear();
            if cache.get(p, &mut scratch) {
                hits += 1;
                hit_pieces.push(p);
                continue;
            }
        } else {
            miss_too_long += 1;
            long_pieces.push(p);
            continue;
        }
        if p.len() <= 16 && enc.token_cache_get(p).is_some() {
            miss_tokcache += 1;
            tokcache_pieces.push(p);
        } else {
            miss_backtrack += 1;
            backtrack_pieces.push(p);
            scratch.clear();
            enc.encode_backtrack_into(p, &mut scratch);
            if scratch.len() > PretokenCache::MAX_TOKENS {
                miss_too_many_toks += 1;
            }
        }
    }
    println!("\n=== warm-cache classification over {} pieces ===", np);
    println!("cache hits        : {:9}  ({:.2}%)", hits, 100.0 * hits as f64 / np as f64);
    println!("miss: piece >15B  : {:9}  ({:.2}%)", miss_too_long, 100.0 * miss_too_long as f64 / np as f64);
    println!("miss: token_cache : {:9}  ({:.2}%)", miss_tokcache, 100.0 * miss_tokcache as f64 / np as f64);
    println!("miss: backtrack   : {:9}  ({:.2}%)  (of which >3 toks: {})",
        miss_backtrack, 100.0 * miss_backtrack as f64 / np as f64, miss_too_many_toks);

    // --- (c) fixed loop cost: no-op encoder ---
    let (ns, chk) = best_ns(|| {
        out.clear();
        for &p in &pieces {
            out.push(p.len() as u32);
        }
        black_box(out.len() as u64)
    });
    println!("\n=== timed buckets (best of {REPS}) ===");
    println!("noop loop (slice+push)         : {:6.2} ns/piece   [chk {}]", ns / np as f64, chk);

    // --- (a) per-hit cost: cache.get over full stream (real mix) ---
    let (ns, chk) = best_ns(|| {
        out.clear();
        let mut h = 0u64;
        for &p in &pieces {
            if p.len() <= PretokenCache::KEY_MAX && cache.get(p, &mut out) {
                h += 1;
            }
        }
        black_box(h)
    });
    println!("cache.get, full piece stream   : {:6.2} ns/piece   [hits {}]", ns / np as f64, chk);

    // --- (a) per-hit cost: hit subset only ---
    let (ns, chk) = best_ns(|| {
        out.clear();
        let mut h = 0u64;
        for &p in &hit_pieces {
            if cache.get(p, &mut out) {
                h += 1;
            }
        }
        black_box(h)
    });
    println!("cache.get, hit subset          : {:6.2} ns/hit     [hits {}]", ns / hit_pieces.len() as f64, chk);

    // --- hit-path decomposition ---
    // 1) key_block construction only
    let (ns, chk) = best_ns(|| {
        let mut acc = 0u64;
        for &p in &hit_pieces {
            let (lo, hi) = PretokenCache::key_of(p);
            acc = acc.wrapping_add(lo) ^ hi;
        }
        black_box(acc)
    });
    println!("  hit decomp: key_block only   : {:6.2} ns/hit     [chk {}]", ns / hit_pieces.len() as f64, chk & 0xffff);

    // 1b) contextual key construction: unconditional 16B load + u128 mask
    // (valid because pieces are subslices of the doc; guard the doc tail)
    let text_bytes = text.as_bytes();
    let base = text_bytes.as_ptr() as usize;
    #[inline(always)]
    fn key_words_ctx(doc: &[u8], start: usize, len: usize) -> (u64, u64) {
        if start + 16 <= doc.len() {
            let raw = u128::from_le_bytes(doc[start..start + 16].try_into().unwrap());
            let masked = raw & (u128::MAX >> (128 - 8 * len));
            (masked as u64, (masked >> 64) as u64 | ((len as u64) << 56))
        } else {
            PretokenCache::key_of(&doc[start..start + len])
        }
    }
    let hit_offsets: Vec<(u32, u32)> = hit_pieces.iter()
        .map(|p| ((p.as_ptr() as usize - base) as u32, p.len() as u32))
        .collect();
    // sanity: ctx keys must equal canonical keys
    for (&p, &(off, len)) in hit_pieces.iter().zip(&hit_offsets).take(100000) {
        assert_eq!(key_words_ctx(text_bytes, off as usize, len as usize), PretokenCache::key_of(p));
    }
    let (ns, chk) = best_ns(|| {
        let mut acc = 0u64;
        for &(off, len) in &hit_offsets {
            let (lo, hi) = key_words_ctx(text_bytes, off as usize, len as usize);
            acc = acc.wrapping_add(lo) ^ hi;
        }
        black_box(acc)
    });
    println!("  hit decomp: key ctx-u128     : {:6.2} ns/hit     [chk {}]", ns / hit_offsets.len() as f64, chk & 0xffff);

    // 2) probe with precomputed keys (hash + load + compare + extend)
    let hit_keys: Vec<(u64, u64)> = hit_pieces.iter().map(|p| PretokenCache::key_of(p)).collect();
    let (ns, chk) = best_ns(|| {
        out.clear();
        let mut h = 0u64;
        for &(lo, hi) in &hit_keys {
            if cache.get_with_key(lo, hi, &mut out) {
                h += 1;
            }
        }
        black_box(h)
    });
    println!("  hit decomp: probe w/ precomp : {:6.2} ns/hit     [hits {}]", ns / hit_keys.len() as f64, chk);

    // --- (b) miss cost: token_cache subset ---
    if !tokcache_pieces.is_empty() {
        let (ns, chk) = best_ns(|| {
            out.clear();
            for &p in &tokcache_pieces {
                if let Some(t) = enc.token_cache_get(p) {
                    out.push(t);
                }
            }
            black_box(out.len() as u64)
        });
        println!("token_cache path, miss subset  : {:6.2} ns/miss    [chk {}]", ns / tokcache_pieces.len() as f64, chk);
    }

    // --- (b) miss cost: full backtrack subset ---
    for (name, subset) in [("backtrack path, miss subset", &backtrack_pieces), (">15B pieces, backtrack", &long_pieces)] {
        if subset.is_empty() {
            continue;
        }
        let (ns, chk) = best_ns(|| {
            out.clear();
            for &p in subset.iter() {
                enc.encode_backtrack_into(p, &mut out);
            }
            black_box(out.len() as u64)
        });
        println!("{:31}: {:6.2} ns/miss    [chk {}]", name, ns / subset.len() as f64, chk);
    }

    // --- full encode_piece_into over recorded stream, warm cache ---
    let (ns, chk) = best_ns(|| {
        out.clear();
        for &p in &pieces {
            enc.encode_piece_into(text_bytes, p, Some(&mut cache), &mut out);
        }
        black_box(out.len() as u64)
    });
    println!("encode_into, warm, full stream : {:6.2} ns/piece   [toks {}]", ns / np as f64, chk);

    // --- full pipeline for reference (pretokenize + encode_piece_into) ---
    let (ns, chk) = best_ns(|| {
        out.clear();
        for d in &docs {
            let db = d.as_bytes();
            for p in pretok.split(d) {
                enc.encode_piece_into(db, p.as_bytes(), Some(&mut cache), &mut out);
            }
        }
        black_box(out.len() as u64)
    });
    println!("pretok + encode_into pipeline  : {:6.2} ns/piece   ({:.1} MB/s) [toks {}]",
        ns / np as f64, nbytes as f64 / 1e6 / (ns / 1e9), chk);
}
