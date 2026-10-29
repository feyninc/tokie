//! Test tiktoken compatibility
//!
//! Run with: cargo run --example tiktoken_compat --features hf

use tokie::Tokenizer;

fn main() {
    println!("Tiktoken Compatibility Test");
    println!("===========================\n");

    let test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def hello():\n    print('Hello, world!')",
        "12345 + 67890 = 80235",
        "don't won't can't",
    ];

    // Test GPT-2 / r50k_base
    println!("=== GPT-2 (r50k_base) ===");
    test_model("openai-community/gpt2", &test_texts);

    // Test cl100k_base models
    println!("\n=== CL100K (cl100k_base) ===");
    test_model("Xenova/gpt-4", &test_texts);

    // Test o200k_base models
    println!("\n=== O200K (o200k_base) ===");
    test_model("Xenova/gpt-4o", &test_texts);

    // Test p50k models (Codex)
    println!("\n=== P50K (p50k_base - Codex) ===");
    test_model("Xenova/text-davinci-003", &test_texts);

    // Additional models
    println!("\n=== Additional Models ===");

    println!("\n--- LLaMA 3 ---");
    test_model("meta-llama/Llama-3.2-1B", &test_texts);

    println!("\n--- Mistral ---");
    test_model("mistralai/Mistral-7B-v0.1", &test_texts);
}

fn test_model(repo_id: &str, texts: &[&str]) {
    println!("Loading: {}", repo_id);

    match Tokenizer::from_pretrained(repo_id) {
        Ok(tokenizer) => {
            println!("  Vocab size: {}", tokenizer.vocab_size());
            println!("  Pretokenizer: {:?}", tokenizer.pretokenizer_type());

            for text in texts.iter().take(2) {
                let tokie_encoding = tokenizer.encode(text, false);
                let display = if text.len() > 27 {
                    format!("{}...", &text[..27])
                } else {
                    text.to_string()
                };
                println!("  {:30} -> {} tokens", display, tokie_encoding.ids.len());
            }

            // Roundtrip test
            let test = "Hello, world!";
            let encoding = tokenizer.encode(test, false);
            match tokenizer.decode(&encoding.ids) {
                Some(decoded) if decoded == test => println!("  Roundtrip: OK"),
                Some(decoded) => println!("  Roundtrip: MISMATCH ({:?})", decoded),
                None => println!("  Roundtrip: DECODE FAILED"),
            }
        }
        Err(e) => {
            println!("  FAILED: {}", e);
        }
    }
}
