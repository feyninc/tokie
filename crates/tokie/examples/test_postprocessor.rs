//! Test post-processor functionality with real tokenizers.
//!
//! Tests BERT and LLaMA 3 tokenizers to verify special tokens are added correctly.

use tokie::Tokenizer;

fn main() {
    println!("Testing PostProcessor functionality\n");

    // Test BERT tokenizer
    if let Ok(bert) = Tokenizer::from_json("/tmp/bert_tokenizer.json") {
        println!("=== BERT Tokenizer ===");
        println!("Post-processor: {:?}", bert.post_processor());

        let text = "Hello";

        // Without special tokens
        let tokens_no_special = bert.encode(text, false).ids;
        println!("encode({:?}, false) = {:?}", text, tokens_no_special);

        // With special tokens
        let tokens_with_special = bert.encode(text, true).ids;
        println!("encode({:?}, true)  = {:?}", text, tokens_with_special);

        // BERT should add [CLS] at start and [SEP] at end
        // [CLS] = 101, [SEP] = 102 typically
        if tokens_with_special.len() == tokens_no_special.len() + 2 {
            println!("✓ BERT adds 2 special tokens (CLS + SEP)");
        } else {
            println!("✗ Expected {} + 2 tokens, got {}",
                     tokens_no_special.len(), tokens_with_special.len());
        }
        println!();
    } else {
        println!("BERT tokenizer not found at /tmp/bert_tokenizer.json");
        println!("Download with: python -c \"from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('bert-base-uncased'); t.save_pretrained('/tmp/bert_tokenizer')\"");
        println!();
    }

    // Test GPT-2 tokenizer
    if let Ok(gpt2) = Tokenizer::from_json("/tmp/gpt2_tokenizer.json") {
        println!("=== GPT-2 Tokenizer ===");
        println!("Post-processor: {:?}", gpt2.post_processor());

        let text = "Hello";

        let tokens_no_special = gpt2.encode(text, false).ids;
        println!("encode({:?}, false) = {:?}", text, tokens_no_special);

        let tokens_with_special = gpt2.encode(text, true).ids;
        println!("encode({:?}, true)  = {:?}", text, tokens_with_special);

        // GPT-2 has no post-processor, so both should be the same
        if tokens_with_special == tokens_no_special {
            println!("✓ GPT-2 has no special tokens (None post-processor)");
        } else {
            println!("✗ Expected same tokens for GPT-2");
        }
        println!();
    } else {
        println!("GPT-2 tokenizer not found at /tmp/gpt2_tokenizer.json");
        println!();
    }

    // Test LLaMA 3 tokenizer
    if let Ok(llama) = Tokenizer::from_json("/tmp/llama3_tokenizer.json") {
        println!("=== LLaMA 3 Tokenizer ===");
        println!("Post-processor: {:?}", llama.post_processor());

        let text = "Hello";

        let tokens_no_special = llama.encode(text, false).ids;
        println!("encode({:?}, false) = {:?}", text, tokens_no_special);

        let tokens_with_special = llama.encode(text, true).ids;
        println!("encode({:?}, true)  = {:?}", text, tokens_with_special);

        // LLaMA 3 should add <|begin_of_text|> (128000) at start
        if tokens_with_special.len() == tokens_no_special.len() + 1 {
            println!("✓ LLaMA 3 adds 1 special token (BOS)");
            if tokens_with_special[0] == 128000 {
                println!("✓ First token is <|begin_of_text|> (128000)");
            }
        } else {
            println!("✗ Expected {} + 1 tokens, got {}",
                     tokens_no_special.len(), tokens_with_special.len());
        }
        println!();
    } else {
        println!("LLaMA 3 tokenizer not found at /tmp/llama3_tokenizer.json");
        println!();
    }

    println!("Done!");
}
