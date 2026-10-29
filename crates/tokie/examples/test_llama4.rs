//! Test Llama 4 tokenizer against HuggingFace
//!
//! Run with: cargo run --release --example test_llama4 --features hf

use tokie::Tokenizer;
use tokenizers::Tokenizer as HfTokenizer;

fn main() {
    println!("=== Llama 4 Tokenizer Comparison ===\n");

    // Load tokie tokenizer
    println!("Loading tokie tokenizer...");
    let tokie = Tokenizer::from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct")
        .expect("Failed to load tokie tokenizer");
    println!("  Vocab size: {}", tokie.vocab_size());
    println!("  Encoder type: {:?}", tokie.encoder_type());

    // Load HuggingFace tokenizer
    println!("\nLoading HuggingFace tokenizer...");
    let hf = HfTokenizer::from_pretrained("meta-llama/Llama-4-Scout-17B-16E-Instruct", None)
        .expect("Failed to load HF tokenizer");
    println!("  Vocab size: {}", hf.get_vocab_size(true));

    let test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "🎉 Unicode test: café, naïve, 中文, العربية",
        "   Multiple   spaces   and\ttabs\nand\nnewlines",
        "This is a longer text to test tokenization performance and accuracy across different scenarios.",
    ];

    println!("\n=== Comparison ===\n");

    let mut matches = 0;
    let mut total = 0;

    for text in &test_cases {
        let tokie_encoding = tokie.encode(text, false);
        let tokie_tokens = &tokie_encoding.ids;
        let hf_encoding = hf.encode(*text, false).expect("HF encode failed");
        let hf_tokens: Vec<u32> = hf_encoding.get_ids().to_vec();

        let is_match = *tokie_tokens == hf_tokens;
        if is_match {
            matches += 1;
        }
        total += 1;

        let preview = if text.len() > 40 { &text[..40] } else { text };
        println!("Input: {:?}...", preview);
        println!("  Tokie: {} tokens {:?}", tokie_tokens.len(), &tokie_tokens[..tokie_tokens.len().min(8)]);
        println!("  HF:    {} tokens {:?}", hf_tokens.len(), &hf_tokens[..hf_tokens.len().min(8)]);
        println!("  Match: {}\n", if is_match { "✓" } else { "✗" });

        if !is_match {
            // Find first difference
            for (i, (t, h)) in tokie_tokens.iter().zip(hf_tokens.iter()).enumerate() {
                if t != h {
                    println!("  First diff at {}: tokie={}, hf={}", i, t, h);
                    break;
                }
            }
            if tokie_tokens.len() != hf_tokens.len() {
                println!("  Length diff: tokie={}, hf={}", tokie_tokens.len(), hf_tokens.len());
            }
        }
    }

    println!("=== Results ===");
    println!("Matches: {}/{}", matches, total);

    if matches == total {
        println!("\n✓ All tests passed! Llama 4 tokenizer is compatible.");
    } else {
        println!("\n✗ Some tests failed. Need investigation.");
    }
}
