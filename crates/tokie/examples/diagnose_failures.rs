//! Diagnose token mismatches between tokie and HuggingFace.
//! Run: cargo run --example diagnose_failures --features hf

use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;
use tokie::Tokenizer;

fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).expect("read enwik8");
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

fn diagnose(name: &str, tokiers_repo: &str, hf_model: &str, text: &str) {
    let tok = match Tokenizer::from_pretrained(tokiers_repo) {
        Ok(t) => t,
        Err(e) => { println!("[{name}] tokie load FAILED: {e}"); return; }
    };
    let mut hf = match HfTokenizer::from_pretrained(hf_model, None) {
        Ok(t) => t,
        Err(e) => { println!("[{name}] HF load FAILED: {e}"); return; }
    };
    // Disable HF truncation so we compare full outputs (some models like Cohere
    // have truncation.max_length in tokenizer.json which truncates during encode)
    let _ = hf.with_truncation(None);

    let tokie_enc = tok.encode(text, false);
    let tokie_ids = &tokie_enc.ids;
    let hf_enc = hf.encode(text, false).unwrap();
    let hf_ids = hf_enc.get_ids();

    if tokie_ids.as_slice() == hf_ids {
        println!("[{name}] PASS ({} tokens)", tokie_ids.len());
        return;
    }

    let diff_idx = tokie_ids.iter().zip(hf_ids.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(tokie_ids.len().min(hf_ids.len()));

    println!("[{name}] FAIL at index {diff_idx} (tokie={} tokens, hf={} tokens)", tokie_ids.len(), hf_ids.len());

    let start = diff_idx.saturating_sub(3);
    let end = (diff_idx + 5).min(tokie_ids.len()).min(hf_ids.len());

    println!("  tokie ids: {:?}", &tokie_ids[start..end.min(tokie_ids.len())]);
    println!("  hf    ids: {:?}", &hf_ids[start..end.min(hf_ids.len())]);

    // Decode individual tokens around diff
    println!("  tokie tokens around diff:");
    for i in start..end.min(tokie_ids.len()) {
        let marker = if i == diff_idx { " <-- DIFF" } else { "" };
        let decoded = tok.decode(&[tokie_ids[i]]).unwrap_or_default();
        println!("    [{i}] id={} {:?}{}", tokie_ids[i], decoded, marker);
    }
    println!("  hf tokens around diff:");
    for i in start..end.min(hf_ids.len()) {
        let marker = if i == diff_idx { " <-- DIFF" } else { "" };
        let decoded = hf.decode(&[hf_ids[i]], false).unwrap_or_default();
        println!("    [{i}] id={} {:?}{}", hf_ids[i], decoded, marker);
    }

    // Show the source text around the mismatch
    // Find byte offset by decoding tokens up to diff
    let prefix_ids = &hf_ids[..diff_idx];
    if let Ok(prefix_text) = hf.decode(prefix_ids, false) {
        let mut byte_start = prefix_text.len().saturating_sub(20);
        while byte_start > 0 && !text.is_char_boundary(byte_start) {
            byte_start -= 1;
        }
        let mut byte_end = (prefix_text.len() + 40).min(text.len());
        while byte_end < text.len() && !text.is_char_boundary(byte_end) {
            byte_end += 1;
        }
        byte_end = byte_end.min(text.len());
        println!("  source text around diff: {:?}", &text[byte_start..byte_end]);
    }
    println!();
}

fn main() {
    let text = load_enwik8(1_000_000);
    println!("Loaded {} bytes of enwik8\n", text.len());

    let models: Vec<(&str, &str, &str)> = vec![
        ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
        ("GPT-2", "tokiers/gpt2", "openai-community/gpt2"),
        ("RoBERTa", "tokiers/roberta-base", "FacebookAI/roberta-base"),
        ("Phi-2", "tokiers/phi-2", "microsoft/phi-2"),
        ("ModernBERT", "tokiers/ModernBERT-base", "answerdotai/ModernBERT-base"),
        ("Llama-3.2", "tokiers/Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
        ("Llama-4-Scout", "tokiers/Llama-4-Scout-17B-16E", "meta-llama/Llama-4-Scout-17B-16E"),
        ("Mistral-Nemo", "tokiers/Mistral-Nemo-Base-2407", "mistralai/Mistral-Nemo-Base-2407"),
        ("Qwen2-7B", "tokiers/Qwen2-7B", "Qwen/Qwen2-7B"),
        ("Qwen3-Embed-0.6B", "tokiers/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-0.6B"),
        ("Cohere-english-v3", "tokiers/Cohere-embed-english-v3.0", "Cohere/Cohere-embed-english-v3.0"),
        ("Cohere-multi-v3", "tokiers/Cohere-embed-multilingual-v3.0", "Cohere/Cohere-embed-multilingual-v3.0"),
        ("Jina-v2-code", "tokiers/jina-embeddings-v2-base-code", "jinaai/jina-embeddings-v2-base-code"),
        ("Jina-v3", "tokiers/jina-embeddings-v3", "jinaai/jina-embeddings-v3"),
        ("T5", "tokiers/t5-base", "google-t5/t5-base"),
        ("XLM-R", "tokiers/xlm-roberta-base", "FacebookAI/xlm-roberta-base"),
        ("Voyage-3", "tokiers/voyage-3", "voyageai/voyage-3"),
        ("Voyage-3-large", "tokiers/voyage-3-large", "voyageai/voyage-3-large"),
        ("Voyage-3-lite", "tokiers/voyage-3-lite", "voyageai/voyage-3-lite"),
        ("Voyage-code-2", "tokiers/voyage-code-2", "voyageai/voyage-code-2"),
        ("Voyage-code-3", "tokiers/voyage-code-3", "voyageai/voyage-code-3"),
        ("Voyage-law-2", "tokiers/voyage-law-2", "voyageai/voyage-law-2"),
        ("Voyage-multimodal-3", "tokiers/voyage-multimodal-3", "voyageai/voyage-multimodal-3"),
    ];

    for (name, tokiers, hf) in &models {
        diagnose(name, tokiers, hf, &text);
    }
}
