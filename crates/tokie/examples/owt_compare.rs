//! Per-document OWT comparison between tokie and HuggingFace.
//! Run: cargo run --release --example owt_compare --features build -- <tokiers-repo> <hf-model> [max-docs]

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;
use tokie::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tokiers_repo = args.get(1).map(String::as_str).unwrap_or("tokiers/DeepSeek-V3");
    let hf_model = args.get(2).map(String::as_str).unwrap_or("deepseek-ai/DeepSeek-V3");
    let max_docs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/owt_slice.txt");
    let text = std::fs::read_to_string(&path).expect("read owt_slice.txt");
    let docs: Vec<&str> = text
        .split("<|endoftext|>")
        .map(|d| d.trim_start_matches('\n'))
        .filter(|d| !d.is_empty())
        .take(max_docs)
        .collect();

    let tok = Tokenizer::from_pretrained(tokiers_repo).expect("tokie load");
    let mut hf = HfTokenizer::from_pretrained(hf_model, None).expect("hf load");
    let _ = hf.with_truncation(None);

    let mut pass = 0usize;
    let mut failures: Vec<usize> = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        let tokie_ids = tok.encode(doc, false).ids;
        let hf_ids = hf.encode(*doc, false).unwrap();
        if tokie_ids.as_slice() == hf_ids.get_ids() {
            pass += 1;
        } else {
            failures.push(i);
        }
    }
    println!("{tokiers_repo} vs {hf_model}: {pass}/{} docs match", docs.len());
    if !failures.is_empty() {
        println!("first failing docs: {:?}", &failures[..failures.len().min(10)]);
        // Show context around first token diff for the first few failures
        for &i in failures.iter().take(3) {
            let doc = docs[i];
            let tokie_ids = tok.encode(doc, false).ids;
            let hf_enc = hf.encode(doc, false).unwrap();
            let hf_ids = hf_enc.get_ids();
            let d = tokie_ids.iter().zip(hf_ids.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(tokie_ids.len().min(hf_ids.len()));
            let s = d.saturating_sub(3);
            println!("doc {i}: diff at token {d} (tokie {} vs hf {} tokens)", tokie_ids.len(), hf_ids.len());
            let te = (d + 5).min(tokie_ids.len());
            let he = (d + 5).min(hf_ids.len());
            let tokie_str: Vec<String> = tokie_ids[s..te].iter().map(|&id| tok.decode(&[id]).unwrap_or_default()).collect();
            let hf_str: Vec<String> = hf_ids[s..he].iter().map(|&id| hf.decode(&[id], false).unwrap_or_default()).collect();
            println!("  tokie: {:?}", tokie_str);
            println!("  hf:    {:?}", hf_str);
        }
        std::process::exit(1);
    }
}
