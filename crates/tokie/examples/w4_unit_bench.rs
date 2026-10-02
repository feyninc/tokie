//! W4 harness: throughput + output fingerprints for SentencePiece BPE,
//! Unigram and WordPiece models.
//!
//! Run: cargo run --release -p tokie --features hf --example w4_unit_bench [-- model...]
//!
//! For each model prints encoder type, MB/s on 1 MB enwik8 (one string),
//! MB/s on the tokbench fixtures in 10 KiB chunks (sequential 1T and
//! `encode_batch` MT), and an FNV-1a hash of every id sequence so runs
//! before/after a change can be diffed for exact equality.

use std::path::{Path, PathBuf};
use std::time::Instant;
use tokie::Tokenizer;

const MODELS: &[&str] = &[
    "tokiers/Llama-2-7b-hf",
    "tokiers/CodeLlama-7b-hf",
    "tokiers/Mistral-7B-v0.1",
    "tokiers/gemma-2-2b",
    "tokiers/gemma-3-4b-it",
    "tokiers/t5-base",
    "tokiers/xlm-roberta-base",
    "tokiers/bge-m3",
    "tokiers/bert-base-uncased",
];

fn fnv(h: &mut u64, ids: &[u32]) {
    for &id in ids {
        for b in id.to_le_bytes() {
            *h ^= b as u64;
            *h = h.wrapping_mul(0x100_0000_01b3);
        }
    }
    // Sequence separator so chunk boundaries are part of the fingerprint.
    *h ^= 0xff;
    *h = h.wrapping_mul(0x100_0000_01b3);
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn chunks_10k(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < s.len() {
        let mut end = (start + 10 * 1024).min(s.len());
        while !s.is_char_boundary(end) {
            end += 1;
        }
        out.push(&s[start..end]);
        start = end;
    }
    out
}

fn best_of<F: FnMut()>(iters: usize, mut f: F) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let models: Vec<&str> = if args.is_empty() { MODELS.to_vec() } else { args.iter().map(|s| s.as_str()).collect() };
    let iters: usize = std::env::var("W4_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

    let enwik = std::fs::read(root().join("benches/data/enwik8")).expect("enwik8");
    let enwik = String::from_utf8_lossy(&enwik[..1_000_000]).into_owned();

    let fixture_dir = PathBuf::from(
        std::env::var("TOKBENCH_FIXTURES")
            .unwrap_or_else(|_| "C:/Users/lain/Desktop/github/tokbench/data/fixtures".into()),
    );
    let fixtures: Vec<(String, String)> = ["english", "chinese", "japanese", "agentic-swe"]
        .iter()
        .filter_map(|n| {
            std::fs::read_to_string(fixture_dir.join(format!("{n}.txt")))
                .ok()
                .map(|t| (n.to_string(), t))
        })
        .collect();

    println!(
        "{:<28} {:<13} {:>9} | {:>30} | hashes",
        "model", "encoder", "enwik MB/s", "fixtures 10KiB 1T / MT MB/s"
    );
    for repo in models {
        let tok = match Tokenizer::from_pretrained(repo) {
            Ok(t) => t,
            Err(e) => {
                println!("{repo:<28} SKIP: {e}");
                continue;
            }
        };
        let split = match tok.encoder() {
            tokie::encoder::Encoder::SentencePiece(e) => Some(e.unit_split()),
            tokie::encoder::Encoder::Unigram(e) => Some(e.unit_split()),
            _ => None,
        };
        let enc = match split {
            Some(s) => format!("{:?}/{:?}", tok.encoder_type(), s),
            None => format!("{:?}", tok.encoder_type()),
        };

        // Split path vs whole-text reference on the normalized inputs.
        if split.is_some() && std::env::var("W4_NO_VERIFY").is_err() {
            let whole = |t: &[u8]| match tok.encoder() {
                tokie::encoder::Encoder::SentencePiece(e) => e.encode_whole(t),
                tokie::encoder::Encoder::Unigram(e) => e.encode_whole(t),
                _ => unreachable!(),
            };
            let mut inputs: Vec<(String, &str)> = vec![("enwik8".into(), enwik.as_str())];
            for (name, text) in &fixtures {
                for (i, c) in chunks_10k(text).into_iter().enumerate() {
                    inputs.push((format!("{name}#{i}"), c));
                }
            }
            let (mut ok, mut bad) = (0usize, 0usize);
            for (name, text) in &inputs {
                let norm = tok.normalizer().normalize(text);
                let a = tok.encoder().encode(norm.as_bytes());
                let b = whole(norm.as_bytes());
                if a == b {
                    ok += 1;
                } else {
                    if bad < 3 {
                        let i = a.iter().zip(&b).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                        eprintln!("  MISMATCH {repo} {name} at token {i}: split={:?} whole={:?}", &a[i..(i + 5).min(a.len())], &b[i..(i + 5).min(b.len())]);
                    }
                    bad += 1;
                }
            }
            println!("{repo:<28} verify split==whole: {ok}/{} inputs ({bad} mismatches)", ok + bad);
        }

        // Fingerprints first (also serves as warmup).
        let mut h_enwik = 0xcbf2_9ce4_8422_2325u64;
        fnv(&mut h_enwik, &tok.encode(&enwik, false).ids);
        let mut fx_hashes = Vec::new();
        for (name, text) in &fixtures {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            let chunks = chunks_10k(text);
            for c in &chunks {
                fnv(&mut h, &tok.encode(c, false).ids);
            }
            // Batch path must agree with the sequential one.
            let mut hb = 0xcbf2_9ce4_8422_2325u64;
            for e in tok.encode_batch(&chunks, false) {
                fnv(&mut hb, &e.ids);
            }
            assert_eq!(h, hb, "{repo}/{name}: batch != sequential");
            // Whole-fixture single string too.
            let mut hw = 0xcbf2_9ce4_8422_2325u64;
            fnv(&mut hw, &tok.encode(text, false).ids);
            fx_hashes.push(format!("{name}={h:016x}/{hw:016x}"));
        }

        let t = best_of(iters, || {
            std::hint::black_box(tok.encode(&enwik, false));
        });
        let enwik_mbs = enwik.len() as f64 / t / 1e6;

        let mut total = 0usize;
        let mut t1 = 0.0;
        let mut tm = 0.0;
        for (_, text) in &fixtures {
            let chunks = chunks_10k(text);
            total += text.len();
            t1 += best_of(iters.min(3), || {
                for c in &chunks {
                    std::hint::black_box(tok.encode(c, false));
                }
            });
            tm += best_of(iters, || {
                std::hint::black_box(tok.encode_batch(&chunks, false));
            });
        }
        println!(
            "{:<28} {:<13} {:>9.1} | {:>14.1} / {:>13.1} | enwik={:016x} {}",
            repo,
            enc,
            enwik_mbs,
            total as f64 / t1 / 1e6,
            total as f64 / tm / 1e6,
            h_enwik,
            fx_hashes.join(" ")
        );
    }
}
