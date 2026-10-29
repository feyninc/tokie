//! Example: Load tokenizer from HuggingFace Hub
//!
//! Run with: cargo run --example from_pretrained --features hf

use tokie::Tokenizer;

fn main() {
    println!("Loading GPT-2 tokenizer from HuggingFace Hub...\n");

    let tokenizer = Tokenizer::from_pretrained("openai-community/gpt2")
        .expect("Failed to load tokenizer");

    println!("Loaded! Vocab size: {}\n", tokenizer.vocab_size());

    let text = "Hello, world! This is a test of the tokie tokenizer.";
    println!("Text: {:?}\n", text);

    let encoding = tokenizer.encode(text, false);
    println!("Tokens: {:?}", encoding.ids);
    println!("Token count: {}\n", encoding.ids.len());

    let decoded = tokenizer.decode(&encoding.ids).unwrap();
    println!("Decoded: {:?}", decoded);

    assert_eq!(text, decoded, "Roundtrip failed!");
    println!("\nRoundtrip successful!");
}
