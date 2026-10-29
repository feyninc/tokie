//! Test BacktrackingBytePairEncoder with GPT-2/tiktoken vocabulary

fn main() {
    // Load cl100k tokenizer (tiktoken-style)
    let tokenizer = tokie::Tokenizer::from_pretrained("Xenova/gpt-4").unwrap();

    let text_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/data/war_and_peace.txt");
    let text = std::fs::read_to_string(&text_path).unwrap();

    println!("Text: {:.2} MB", text.len() as f64 / 1_000_000.0);
    println!("Vocab size: {}", tokenizer.vocab_size());

    // Test on progressively longer text
    for size in [100, 500, 1000, 5000, 10000, 50000] {
        let slice = &text[..size.min(text.len())];
        let encoding = tokenizer.encode(slice, false);
        println!("{:6} bytes -> {:5} tokens", size, encoding.ids.len());

        // Verify decode roundtrip
        let decoded = tokenizer.decode(&encoding.ids).unwrap();
        if decoded != slice {
            println!("  WARNING: Decode mismatch!");
        }
    }

    // Full benchmark
    println!("\nFull encoding:");
    let start = std::time::Instant::now();
    let encoding = tokenizer.encode(&text, false);
    let elapsed = start.elapsed();
    println!("  {} tokens in {:.1} ms ({:.1} MB/s)",
             encoding.ids.len(),
             elapsed.as_secs_f64() * 1000.0,
             text.len() as f64 / elapsed.as_secs_f64() / 1_000_000.0);
}
