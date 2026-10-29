//! Basic tokie usage: load a tokenizer, encode/decode text, count tokens.
//!
//! Run with: cargo run --release --example basic_usage --features hf

use tokie::Tokenizer;

fn main() {
    // Load a tokenizer from HuggingFace Hub
    // First call downloads tokenizer.json; subsequent calls use cached .tkz
    let tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();
    println!("Loaded BERT tokenizer (vocab_size={})", tokenizer.vocab_size());

    // Encode text — returns Encoding with ids, attention_mask, type_ids
    let encoding = tokenizer.encode("Hello, world!", true);
    println!("\nEncoding for \"Hello, world!\" (with special tokens):");
    println!("  ids:            {:?}", encoding.ids);
    println!("  attention_mask: {:?}", encoding.attention_mask);
    println!("  type_ids:       {:?}", encoding.type_ids);

    // Encode without special tokens (no [CLS]/[SEP])
    let encoding_raw = tokenizer.encode("Hello, world!", false);
    println!("\nWithout special tokens:");
    println!("  ids: {:?}", encoding_raw.ids);

    // Decode back to text
    let decoded = tokenizer.decode(&encoding.ids).unwrap();
    println!("\nDecoded: \"{}\"", decoded);

    // Count tokens (faster than encode — skips Encoding allocation)
    let count = tokenizer.count_tokens("The quick brown fox jumps over the lazy dog.");
    println!("\nToken count: {}", count);

    // Save as .tkz for instant loading (~5ms vs ~200ms from JSON)
    tokenizer.to_file("/tmp/bert.tkz").unwrap();
    let fast = Tokenizer::from_file("/tmp/bert.tkz").unwrap();
    println!("\nLoaded from .tkz (vocab_size={})", fast.vocab_size());
}
