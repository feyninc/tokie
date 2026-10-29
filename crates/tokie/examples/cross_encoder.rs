//! Cross-encoder sentence pair encoding for reranking models.
//!
//! Run with: cargo run --release --example cross_encoder --features hf

use tokie::Tokenizer;

fn main() {
    // Load a cross-encoder tokenizer (BERT-based)
    let tokenizer = Tokenizer::from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").unwrap();
    println!("Loaded cross-encoder tokenizer (vocab={})", tokenizer.vocab_size());

    // Reranking: score query-document pairs
    let query = "What is machine learning?";
    let documents = [
        "Machine learning is a branch of artificial intelligence.",
        "The weather today is sunny with a high of 75F.",
        "Deep learning uses neural networks with many layers.",
        "I like to eat pizza on Fridays.",
    ];

    println!("\nQuery: \"{}\"", query);
    println!("\nEncoding query-document pairs:");
    for (i, doc) in documents.iter().enumerate() {
        let encoding = tokenizer.encode_pair(query, doc, true);

        // type_ids: 0 = query tokens, 1 = document tokens
        let query_tokens = encoding.type_ids.iter().filter(|&&t| t == 0).count();
        let doc_tokens = encoding.type_ids.iter().filter(|&&t| t == 1).count();

        println!("  [{}] {} total tokens (query={}, doc={})",
            i, encoding.ids.len(), query_tokens, doc_tokens);
        println!("      doc: \"{}\"", &doc[..doc.len().min(60)]);
    }

    // These encodings (ids + attention_mask + type_ids) are ready for
    // model inference — feed directly to a cross-encoder for scoring.
}
