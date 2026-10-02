//! W3 research harness for the Simple BPE encoder.
//!
//! cargo run --release --features hf --example w3_simple -- types
//! cargo run --release --features hf --example w3_simple -- equiv
//! cargo run --release --features hf --example w3_simple -- bench [filter]

use std::path::{Path, PathBuf};
use std::time::Instant;
use tokie::Tokenizer;

const TOKBENCH: &str = "C:/Users/lain/Desktop/github/tokbench/data";

/// (name, from_pretrained repo, raw tokenizer.json source: hub repo or local path)
const MODELS: &[(&str, &str, &str)] = &[
    ("gpt2", "tokiers/gpt2", "tokbench:gpt2"),
    ("llama-3", "tokiers/Llama-3.2-1B", "tokbench:llama-3"),
    ("deepseek-v4", "", "tokbench:deepseek-v4"),
    ("qwen3", "tokiers/Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    ("roberta", "tokiers/roberta-base", "FacebookAI/roberta-base"),
    ("mistral-nemo", "tokiers/Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Nemo-Instruct-2407"),
];

fn json_path(src: &str) -> Option<PathBuf> {
    if let Some(name) = src.strip_prefix("tokbench:") {
        return Some(Path::new(TOKBENCH).join("models").join(name).join("tokenizer.json"));
    }
    let api = hf_hub::api::sync::Api::new().ok()?;
    match api.model(src.to_string()).get("tokenizer.json") {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("  (download {src} failed: {e})");
            None
        }
    }
}

/// Every loadable (label, tokenizer) combination.
fn load_all(filter: Option<&str>) -> Vec<(String, Tokenizer)> {
    let mut v = Vec::new();
    for &(name, repo, src) in MODELS {
        if filter.is_some_and(|f| !name.contains(f)) {
            continue;
        }
        if !repo.is_empty() {
            match Tokenizer::from_pretrained(repo) {
                Ok(t) => v.push((format!("{name}/pretrained"), t)),
                Err(e) => eprintln!("  {name}/pretrained ({repo}): {e}"),
            }
        }
        if let Some(p) = json_path(src) {
            match Tokenizer::from_json(&p) {
                Ok(t) => v.push((format!("{name}/json"), t)),
                Err(e) => eprintln!("  {name}/json ({}): {e}", p.display()),
            }
        }
    }
    v
}

fn enwik8(max: usize) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benches/data/enwik8");
    let d = std::fs::read(p).expect("benches/data/enwik8");
    String::from_utf8_lossy(&d[..d.len().min(max)]).into_owned()
}

fn fixtures() -> Vec<(&'static str, String)> {
    ["english", "chinese", "japanese", "agentic-swe"]
        .iter()
        .filter_map(|n| {
            let p = Path::new(TOKBENCH).join("fixtures").join(format!("{n}.txt"));
            std::fs::read_to_string(p).ok().map(|s| (*n, s))
        })
        .collect()
}

/// Split into ~10 KiB chunks at char boundaries.
fn chunks(s: &str, size: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let mut j = (i + size).min(s.len());
        while !s.is_char_boundary(j) {
            j += 1;
        }
        out.push(&s[i..j]);
        i = j;
    }
    out
}

fn types() {
    for (label, t) in load_all(None) {
        println!("{label:<28} {:?}  pretok={:?}", t.encoder_type(), t.pretokenizer_type());
    }
}

/// New Simple (encode) vs old Simple (encode_reference) on every pretoken.
fn equiv() {
    let text = enwik8(1_000_000);
    let fx = fixtures();
    for (label, t) in load_all(None) {
        let Some(enc) = t.encoder().as_simple() else { continue };
        let Some(pretok) = t.pretokenizer() else {
            println!("{label}: no pretokenizer, skipped");
            continue;
        };
        let mut n = 0usize;
        let mut bad = 0usize;
        let mut cache = tokie::encoder::PretokenCache::new();
        let mut check = |piece: &[u8]| {
            n += 1;
            let want = enc.encode_reference(piece);
            let got = enc.encode(piece);
            let mut via_cache = Vec::new();
            enc.encode_into(piece, Some(&mut cache), &mut via_cache);
            let mut via_cache2 = Vec::new();
            enc.encode_into(piece, Some(&mut cache), &mut via_cache2);
            if want != got || want != via_cache || want != via_cache2 {
                bad += 1;
                if bad <= 5 {
                    eprintln!("  MISMATCH {:?}: ref={want:?} new={got:?} cache={via_cache:?}/{via_cache2:?}", String::from_utf8_lossy(piece));
                }
            }
        };
        for p in pretok.split(&text) {
            check(p.as_bytes());
        }
        for (_, s) in &fx {
            for p in pretok.split(s) {
                check(p.as_bytes());
            }
            // Whole unsplit lines exercise the long-piece queue path.
            for line in s.lines().take(2000) {
                check(line.as_bytes());
            }
        }
        for line in text.lines().take(5000) {
            check(line.as_bytes());
        }
        // Random byte fuzz (xorshift), biased toward repeated symbols.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let alphabet = b"aaaaeeettt  \n\t0123456789abcdefghijklmnopqrstuvwxyz.,-_()'\"\xc3\xa9\xe4\xb8\xad";
        let mut buf = Vec::new();
        for i in 0..200_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let len = (x % if i % 50 == 0 { 600 } else { 40 }) as usize;
            buf.clear();
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                buf.push(if x & 0x100 == 0 { alphabet[(x >> 16) as usize % alphabet.len()] } else { (x >> 24) as u8 });
            }
            check(&buf);
        }
        println!("{label:<28} pieces={n} mismatches={bad}");
    }
}

fn mbps(bytes: usize, secs: f64) -> f64 {
    bytes as f64 / secs / 1e6
}

fn bench(filter: Option<&str>) {
    let text = enwik8(1_000_000);
    let fx = fixtures();
    for (label, t) in load_all(filter) {
        if t.encoder().as_simple().is_none() && std::env::var("ALL").is_err() {
            continue;
        }
        // single string, enwik8 1 MB: warmup + best of 5
        let _ = t.encode_ids(&text, false);
        let mut best = f64::MAX;
        for _ in 0..5 {
            let s = Instant::now();
            std::hint::black_box(t.encode_ids(&text, false));
            best = best.min(s.elapsed().as_secs_f64());
        }
        print!("{label:<22} enwik8-1MB {:>7.1} MB/s", mbps(text.len(), best));
        for (name, s) in &fx {
            let cs = chunks(s, 10 * 1024);
            let total: usize = cs.iter().map(|c| c.len()).sum();
            for c in &cs {
                std::hint::black_box(t.encode_ids(c, false));
            }
            let mut best = f64::MAX;
            for _ in 0..5 {
                let st = Instant::now();
                for c in &cs {
                    std::hint::black_box(t.encode_ids(c, false));
                }
                best = best.min(st.elapsed().as_secs_f64());
            }
            print!("  {name} {:>6.1}", mbps(total, best));
        }
        println!();
    }
}

/// Routing experiment: load each Backtracking json model as Simple too,
/// compare full-pipeline ids and chunk throughput.
fn route() {
    let text = enwik8(1_000_000);
    let fx = fixtures();
    for &(name, _, src) in MODELS {
        let Some(p) = json_path(src) else { continue };
        let Ok(bt) = Tokenizer::from_json(&p) else { continue };
        if bt.encoder().as_backtracking().is_none() {
            continue;
        }
        let si = Tokenizer::from_json_with_encoder(&p, tokie::EncoderType::Simple).unwrap();
        let mut docs: Vec<&str> = vec![text.as_str()];
        for (_, s) in &fx {
            docs.extend(chunks(s, 10 * 1024));
        }
        let mut diff = 0;
        for d in &docs {
            if bt.encode_ids(d, false) != si.encode_ids(d, false) {
                diff += 1;
            }
        }
        println!("{name}: simple-vs-backtracking differing docs {diff}/{}", docs.len());
        for (label, t) in [("backtracking", &bt), ("simple", &si)] {
            print!("  {label:<13}");
            for (fname, s) in &fx {
                let cs = chunks(s, 10 * 1024);
                let total: usize = cs.iter().map(|c| c.len()).sum();
                let mut best = f64::MAX;
                for _ in 0..5 {
                    let st = Instant::now();
                    for c in &cs {
                        std::hint::black_box(t.encode_ids(c, false));
                    }
                    best = best.min(st.elapsed().as_secs_f64());
                }
                print!("  {fname} {:>6.1}", mbps(total, best));
            }
            println!();
        }
    }
}

/// HF tokenizers (0.22) on the same 10 KiB chunks, for scale.
#[cfg(feature = "build")]
fn hfbench() {
    let fx = fixtures();
    for &(name, _, src) in MODELS {
        let Some(p) = json_path(src) else { continue };
        let Ok(t) = Tokenizer::from_json(&p) else { continue };
        if t.encoder().as_simple().is_none() && std::env::var("ALL").is_err() {
            continue;
        }
        let mut hf = tokenizers::Tokenizer::from_file(&p).unwrap();
        let _ = hf.with_truncation(None);
        let _ = hf.with_padding(None);
        print!("hf/{name:<19}");
        for (fname, s) in &fx {
            let cs = chunks(s, 10 * 1024);
            let total: usize = cs.iter().map(|c| c.len()).sum();
            let mut best = f64::MAX;
            for _ in 0..3 {
                let st = Instant::now();
                for c in &cs {
                    std::hint::black_box(hf.encode_fast(*c, false).unwrap());
                }
                best = best.min(st.elapsed().as_secs_f64());
            }
            print!("  {fname} {:>6.1}", mbps(total, best));
        }
        println!();
    }
}
#[cfg(not(feature = "build"))]
fn hfbench() {}

/// Write full-pipeline ids (enwik8 1 MB + 10 KiB fixture chunks) per model,
/// for diffing between builds.
fn dump(dir: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let text = enwik8(1_000_000);
    let fx = fixtures();
    for (label, t) in load_all(None) {
        if t.encoder().as_simple().is_none() && std::env::var("ALL").is_err() {
            continue;
        }
        let mut ids = t.encode_ids(&text, false);
        for (_, s) in &fx {
            for c in chunks(s, 10 * 1024) {
                ids.extend(t.encode_ids(c, false));
            }
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(Path::new(dir).join(label.replace('/', "_")), bytes).unwrap();
        println!("{label}: {} ids", ids.len());
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("types") => types(),
        Some("equiv") => equiv(),
        Some("bench") => bench(args.get(2).map(String::as_str)),
        Some("dump") => dump(&args[2]),
        Some("hist") => hist(&args[2]),
        Some("micro") => micro(&args[2]),
        Some("route") => route(),
        Some("hfbench") => hfbench(),
        _ => eprintln!("usage: w3_simple types|equiv|bench [filter]"),
    }
}

/// Merge-core microbench: uncached `encode` over pieces bucketed by length.
pub fn micro(filter: &str) {
    let fx = fixtures();
    let text = enwik8(1_000_000);
    for (label, t) in load_all(Some(filter)).into_iter().take(1) {
        let enc = t.encoder().as_simple().unwrap();
        let pretok = t.pretokenizer().unwrap();
        let mut all: Vec<(&str, &str)> = fx.iter().map(|(n, s)| (*n, s.as_str())).collect();
        all.push(("enwik8", text.as_str()));
        for (name, s) in all {
            let pieces: Vec<&[u8]> = pretok.split(s).map(|p| p.as_bytes()).collect();
            for (lo, hi) in [(2usize, 15usize), (16, 64), (65, 256), (257, usize::MAX)] {
                let ps: Vec<&[u8]> = pieces.iter().copied().filter(|p| p.len() >= lo && p.len() <= hi).collect();
                let bytes: usize = ps.iter().map(|p| p.len()).sum();
                if bytes < 10_000 {
                    continue;
                }
                let mut best = f64::MAX;
                let mut best_ref = f64::MAX;
                for _ in 0..std::env::var("REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(3) {
                    let st = Instant::now();
                    for p in &ps {
                        std::hint::black_box(enc.encode(p));
                    }
                    best = best.min(st.elapsed().as_secs_f64());
                    let st = Instant::now();
                    for p in &ps {
                        std::hint::black_box(enc.encode_reference(p));
                    }
                    best_ref = best_ref.min(st.elapsed().as_secs_f64());
                }
                println!("{label} {name:<12} {lo:>3}-{hi:<5} bytes={bytes:>8}  new {:>6.1} MB/s  ref {:>6.1} MB/s", mbps(bytes, best), mbps(bytes, best_ref));
            }
        }
    }
}

/// Piece length histogram (share of bytes) per fixture for one model.
#[allow(dead_code)]
pub fn hist(filter: &str) {
    let fx = fixtures();
    for (label, t) in load_all(Some(filter)).into_iter().take(1) {
        let pretok = t.pretokenizer().unwrap();
        for (name, s) in &fx {
            let mut b = [0usize; 5];
            let mut np = 0;
            for p in pretok.split(s) {
                np += 1;
                let l = p.len();
                let k = if l <= 1 { 0 } else if l <= 15 { 1 } else if l <= 64 { 2 } else if l <= 256 { 3 } else { 4 };
                b[k] += l;
            }
            let tot: usize = b.iter().sum();
            // Bytes in 16+ pieces that repeat an earlier piece (a long-piece cache hit ceiling).
            let mut seen = std::collections::HashSet::new();
            let (mut long_b, mut rep_b) = (0usize, 0usize);
            for p in pretok.split(s) {
                if p.len() > 15 {
                    long_b += p.len();
                    if !seen.insert(p) { rep_b += p.len(); }
                }
            }
            println!("  long-piece repeat share: {:.0}% of {} long bytes", 100.0 * rep_b as f64 / long_b.max(1) as f64, long_b);
            println!("{label} {name}: pieces={np} avg={:.1}B  bytes%: 1B {:.0} | 2-15 {:.0} | 16-64 {:.0} | 65-256 {:.0} | >256 {:.0}",
                tot as f64 / np as f64,
                100.0 * b[0] as f64 / tot as f64, 100.0 * b[1] as f64 / tot as f64, 100.0 * b[2] as f64 / tot as f64,
                100.0 * b[3] as f64 / tot as f64, 100.0 * b[4] as f64 / tot as f64);
        }
    }
}
