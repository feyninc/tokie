//! Full enwik8 test.
//! cargo run --example check_models --release --features hf

use std::path::Path;
use tokie::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).unwrap();
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

fn main() {
    let text = load_enwik8(1_000_000);
    eprintln!("Testing on {:.1} MB of enwik8\n", text.len() as f64 / 1_000_000.0);

    let models: Vec<(&str, &str)> = vec![
        // Qwen3 / Qwen3.5
        ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
        ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ("Qwen3-Coder-30B", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),
        ("Qwen3.5-0.8B", "Qwen/Qwen3.5-0.8B"),
        ("Qwen3.5-4B", "Qwen/Qwen3.5-4B"),
        ("Qwen3.5-27B", "Qwen/Qwen3.5-27B"),
        // Recent models
        ("DeepSeek-V3", "tokiers/DeepSeek-V3"),
        ("DeepSeek-R1", "tokiers/DeepSeek-R1"),
        ("Gemma-3-4B", "google/gemma-3-4b-it"),
        ("SmolLM2-135M", "tokiers/SmolLM2-135M"),
        // Regression check
        ("BERT", "tokiers/bert-base-uncased"),
        ("GPT-2", "tokiers/gpt2"),
        ("Llama-3.2", "tokiers/Llama-3.2-1B"),
        ("Qwen2-7B", "tokiers/Qwen2-7B"),
        ("XLM-RoBERTa", "tokiers/xlm-roberta-base"),
        ("Voyage-code-2", "tokiers/voyage-code-2"),
    ];

    let mut pass = 0;
    let mut fail = 0;

    for (name, repo) in &models {
        eprint!("{:<25} ", name);
        let tok = match Tokenizer::from_pretrained(repo) {
            Ok(t) => t,
            Err(e) => { eprintln!("⚠️  {}", e); fail += 1; continue; }
        };
        let mut hf = match HfTokenizer::from_pretrained(repo, None) {
            Ok(t) => t,
            Err(e) => { eprintln!("⚠️  {}", e); fail += 1; continue; }
        };
        let _ = hf.with_truncation(None);
        let _ = hf.with_padding(None);
        let tokie_ids = tok.encode(&text, false).ids;
        let hf_ids = hf.encode(text.as_str(), false).unwrap().get_ids().to_vec();
        if tokie_ids == hf_ids {
            eprintln!("✅ PASS ({} tokens)", tokie_ids.len());
            pass += 1;
        } else {
            let d = tokie_ids.iter().zip(hf_ids.iter()).position(|(t,h)| t!=h).unwrap_or(0);
            eprintln!("❌ FAIL (tokie={}, hf={}, diff@{})", tokie_ids.len(), hf_ids.len(), d);
            let start = d.saturating_sub(3);
            let end = (d + 5).min(tokie_ids.len()).min(hf_ids.len());
            eprintln!("  tokie[{}..{}]: {:?}", start, end, &tokie_ids[start..end]);
            eprintln!("  hf   [{}..{}]: {:?}", start, end, &hf_ids[start..end]);
            fail += 1;
        }
    }
    eprintln!("\n{pass} pass, {fail} fail out of {}", pass + fail);
}
