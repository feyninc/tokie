//! Padding and truncation for batch ML inference.
//!
//! Run with: cargo run --release --example padding_truncation --features hf

use tokie::Tokenizer;

fn main() {
    let mut tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();

    // --- Truncation ---
    // Limit sequences to max_length (content tokens truncated, special tokens preserved)
    tokenizer.enable_truncation(tokie::TruncationParams {
        max_length: 10,
        ..Default::default()
    });

    let enc = tokenizer.encode("This is a longer sentence that will be truncated to fit", true);
    println!("Truncated to 10 tokens: {:?}", enc.ids);
    println!("Length: {}", enc.ids.len());

    // --- Padding ---
    // Pad all sequences in a batch to the same length
    tokenizer.enable_padding(tokie::PaddingParams {
        strategy: tokie::PaddingStrategy::Fixed(16),
        pad_id: tokenizer.pad_token_id().unwrap_or(0),
        ..Default::default()
    });

    let texts = &["Hello world", "Short", "A much longer sentence for testing purposes"];
    let batch = tokenizer.encode_batch(texts, true);

    println!("\nBatch encoding (truncated to 10, padded to 16):");
    for (i, enc) in batch.iter().enumerate() {
        println!("  [{}] ids={:?}", i, enc.ids);
        println!("      mask={:?}", enc.attention_mask);
    }

    // Verify all same length
    assert!(batch.iter().all(|e| e.len() == 16));
    println!("\nAll sequences padded to length 16");

    // --- Pair encoding with truncation ---
    tokenizer.no_padding(); // disable padding for this demo
    let pair = tokenizer.encode_pair("What is machine learning?", "Machine learning is a subset of AI.", true);
    println!("\nPair encoding:");
    println!("  ids:      {:?}", pair.ids);
    println!("  type_ids: {:?}", pair.type_ids);
    println!("  Length: {} (truncated to max 10)", pair.ids.len());

    // --- Reset ---
    tokenizer.no_truncation();
    tokenizer.no_padding();
    let enc = tokenizer.encode("Back to normal — no truncation or padding", true);
    println!("\nAfter reset: {} tokens", enc.ids.len());
}
