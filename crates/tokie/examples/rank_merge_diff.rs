//! Differential identity test: rank-merge core vs DAAC backtracking walk.
//!
//! For every piece produced by the pretokenizer over 10MB of OWT, and for
//! fuzzed random byte strings, assert encode_rank_merge == encode_sequential_into.
//!
//! Run: TOKIE_JSON=<gpt2 tokenizer.json> \
//!      cargo run --release -p tokie --features hf --example rank_merge_diff

use std::collections::HashSet;

fn diff_tokenizer(name: &str, tok: &tokie::Tokenizer, docs: &[&str]) -> usize {
    let enc = tok.encoder().as_backtracking().expect("backtracking encoder");
    assert!(enc.has_rank_merge(), "{name}: rank table unavailable");
    let pretok = tok.pretokenizer().expect("pretokenizer");

    let mut pieces_checked = 0usize;
    let mut unique: HashSet<Vec<u8>> = HashSet::new();
    let mut divergences = 0usize;

    for d in docs {
        for piece in pretok.split(d) {
            let bytes = piece.as_bytes();
            if bytes.is_empty() || bytes.len() > 64 {
                continue;
            }
            if !unique.insert(bytes.to_vec()) {
                continue; // already checked this piece
            }
            let mut daac = Vec::new();
            enc.encode_sequential_into(bytes, &mut daac);
            let mut rank = Vec::new();
            enc.encode_rank_merge(bytes, &mut rank);
            pieces_checked += 1;
            if daac != rank {
                divergences += 1;
                if divergences <= 10 {
                    println!(
                        "  DIVERGENCE piece {:?} ({} bytes): daac={:?} rank={:?}",
                        String::from_utf8_lossy(bytes),
                        bytes.len(),
                        daac,
                        rank
                    );
                }
            }
        }
    }
    println!(
        "{name}: {pieces_checked} unique pieces checked, {divergences} divergences"
    );
    divergences
}

fn fuzz_tokenizer(name: &str, tok: &tokie::Tokenizer, iters: usize) -> usize {
    let enc = tok.encoder().as_backtracking().expect("backtracking encoder");
    // xorshift64 for deterministic fuzz
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut divergences = 0usize;
    for _ in 0..iters {
        let len = 1 + (next() as usize) % 40;
        let mut bytes = vec![0u8; len];
        for b in bytes.iter_mut() {
            *b = next() as u8;
        }
        let mut daac = Vec::new();
        enc.encode_sequential_into(&bytes, &mut daac);
        let mut rank = Vec::new();
        enc.encode_rank_merge(&bytes, &mut rank);
        if daac != rank {
            divergences += 1;
            if divergences <= 10 {
                println!(
                    "  FUZZ DIVERGENCE bytes {:?}: daac={:?} rank={:?}",
                    bytes, daac, rank
                );
            }
        }
    }
    println!("{name}: {iters} fuzz strings (1-40 random bytes), {divergences} divergences");
    divergences
}

fn main() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("owt_sample.txt missing");
    let text = String::from_utf8_lossy(&data[..data.len().min(10_000_000)]).into_owned();
    let docs: Vec<&str> = text
        .split("<|endoftext|>")
        .filter(|d| !d.is_empty())
        .collect();
    println!(
        "{} docs, {:.1} MB",
        docs.len(),
        docs.iter().map(|d| d.len()).sum::<usize>() as f64 / 1e6
    );

    let mut total = 0usize;

    let gpt2 = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };
    total += diff_tokenizer("gpt2", &gpt2, &docs);
    total += fuzz_tokenizer("gpt2 fuzz", &gpt2, 200_000);

    let dsv3 = tokie::Tokenizer::from_pretrained("tokiers/DeepSeek-V3").unwrap();
    total += diff_tokenizer("DeepSeek-V3", &dsv3, &docs);
    total += fuzz_tokenizer("DeepSeek-V3 fuzz", &dsv3, 200_000);

    if total == 0 {
        println!("IDENTITY HOLDS: rank-merge == DAAC backtracking on all inputs");
    } else {
        println!("IDENTITY FAILED: {total} divergences");
        std::process::exit(1);
    }
}
