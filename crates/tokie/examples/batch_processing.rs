//! Batch processing: encode many texts in parallel with padding.
//!
//! Run with: cargo run --release --example batch_processing --features hf

use std::time::Instant;
use tokie::Tokenizer;

fn main() {
    let mut tokenizer = Tokenizer::from_pretrained("bert-base-uncased").unwrap();

    // Generate sample data
    let texts: Vec<String> = (0..10_000)
        .map(|i| format!("This is sentence number {} with some additional content for tokenization benchmarking.", i))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    // --- Batch encode (parallel, no padding) ---
    let start = Instant::now();
    let batch = tokenizer.encode_batch(&text_refs, false);
    let elapsed = start.elapsed();

    let total_tokens: usize = batch.iter().map(|e| e.ids.len()).count();
    println!("Encoded {} texts in {:.2}ms", texts.len(), elapsed.as_secs_f64() * 1000.0);
    println!("  {} encodings, first has {} tokens", total_tokens, batch[0].ids.len());

    // --- Batch encode with padding (for model input) ---
    tokenizer.enable_truncation(tokie::TruncationParams {
        max_length: 32,
        ..Default::default()
    });
    tokenizer.enable_padding(tokie::PaddingParams {
        strategy: tokie::PaddingStrategy::BatchLongest,
        pad_id: tokenizer.pad_token_id().unwrap_or(0),
        ..Default::default()
    });

    let start = Instant::now();
    let padded = tokenizer.encode_batch(&text_refs, true);
    let elapsed = start.elapsed();

    let seq_len = padded[0].ids.len();
    println!("\nWith truncation(32) + padding(BatchLongest):");
    println!("  Encoded {} texts in {:.2}ms", texts.len(), elapsed.as_secs_f64() * 1000.0);
    println!("  All sequences: length={}", seq_len);
    assert!(padded.iter().all(|e| e.len() == seq_len));

    // --- Count tokens (even faster, no Encoding overhead) ---
    let start = Instant::now();
    let counts = tokenizer.count_tokens_batch(&text_refs);
    let elapsed = start.elapsed();
    let total: usize = counts.iter().sum();
    println!("\nCounted tokens for {} texts in {:.2}ms ({} total tokens)",
        texts.len(), elapsed.as_secs_f64() * 1000.0, total);
}
