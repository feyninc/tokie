//! Comprehensive tokenizer test suite using enwik8.
//!
//! Tests all supported tokenizers against HuggingFace reference implementations.
//!
//! Run with: cargo run --release --example test_enwik8
//!
//! Prerequisites:
//!   - Download enwik8: wget http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip
//!   - Place in benches/data/enwik8

use std::fs;
use std::path::Path;
use std::time::Instant;
use tokenizers::Tokenizer as HfTokenizer;
use tokie::diff;

/// Test configuration for a tokenizer
struct TokenizerTest {
    name: &'static str,
    tokie_model: &'static str,
    hf_model: &'static str,
}

const TOKENIZER_TESTS: &[TokenizerTest] = &[
    TokenizerTest {
        name: "BERT (uncased)",
        tokie_model: "models/bert.tkz",
        hf_model: "bert-base-uncased",
    },
    TokenizerTest {
        name: "BGE Small",
        tokie_model: "models/baai_bge_small_en_v1.5.tkz",
        hf_model: "BAAI/bge-small-en-v1.5",
    },
    TokenizerTest {
        name: "GTE Base",
        tokie_model: "models/thenlper_gte_base.tkz",
        hf_model: "thenlper/gte-base",
    },
    TokenizerTest {
        name: "E5 Base",
        tokie_model: "models/intfloat_e5_base_v2.tkz",
        hf_model: "intfloat/e5-base-v2",
    },
    TokenizerTest {
        name: "MiniLM",
        tokie_model: "models/sentence_transformers_all_minilm_l6_v2.tkz",
        hf_model: "sentence-transformers/all-MiniLM-L6-v2",
    },
    TokenizerTest {
        name: "Jina v2",
        tokie_model: "models/jinaai_jina_embeddings_v2_base_en.tkz",
        hf_model: "jinaai/jina-embeddings-v2-base-en",
    },
    TokenizerTest {
        name: "Nomic",
        tokie_model: "models/nomic_ai_nomic_embed_text_v1.tkz",
        hf_model: "nomic-ai/nomic-embed-text-v1",
    },
];

/// Strip padding tokens from HF output
fn strip_padding(ids: &[u32]) -> Vec<u32> {
    if ids.is_empty() {
        return vec![];
    }
    let last = ids[ids.len() - 1];
    if last > 1 {
        return ids.to_vec();
    }
    let mut end = ids.len();
    while end > 0 && ids[end - 1] == last {
        end -= 1;
    }
    ids[..end].to_vec()
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         Tokie Tokenizer Test Suite (enwik8)                  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Check for enwik8
    let enwik8_path = "benches/data/enwik8";
    let text = if Path::new(enwik8_path).exists() {
        println!("Loading enwik8...");
        let bytes = fs::read(enwik8_path).expect("Failed to read enwik8");
        // Convert to valid UTF-8 (enwik8 is mostly ASCII/UTF-8)
        String::from_utf8_lossy(&bytes).to_string()
    } else {
        println!("enwik8 not found at {}.", enwik8_path);
        println!("Using War and Peace as fallback...\n");

        let fallback = "benches/data/war_and_peace.txt";
        if Path::new(fallback).exists() {
            fs::read_to_string(fallback).expect("Failed to read fallback")
        } else {
            eprintln!("No test data found!");
            eprintln!("Please download enwik8: wget http://mattmahoney.net/dc/enwik8.zip");
            std::process::exit(1);
        }
    };

    let text_mb = text.len() as f64 / 1_000_000.0;
    println!("Test data: {:.2} MB ({} bytes)\n", text_mb, text.len());

    // Test each tokenizer
    let mut results = Vec::new();

    for test in TOKENIZER_TESTS {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("Testing: {}", test.name);
        println!("  tokie: {}", test.tokie_model);
        println!("  hf:    {}", test.hf_model);
        println!();

        // Load tokenizers
        let tokie = match tokie::Tokenizer::from_file(test.tokie_model) {
            Ok(t) => t,
            Err(e) => {
                println!("  ✗ Failed to load tokie: {:?}\n", e);
                results.push((test.name, false, 0, 0, 0.0, 0.0));
                continue;
            }
        };

        let mut hf = match HfTokenizer::from_pretrained(test.hf_model, None) {
            Ok(t) => t,
            Err(e) => {
                println!("  ✗ Failed to load HF: {:?}\n", e);
                results.push((test.name, false, 0, 0, 0.0, 0.0));
                continue;
            }
        };

        // Disable truncation so we get all tokens for comparison
        let _ = hf.with_truncation(None);

        // Encode with tokie
        let start = Instant::now();
        let tokie_encoding = tokie.encode(&text, false);
        let tokie_tokens = tokie_encoding.ids;
        let tokie_time = start.elapsed().as_secs_f64();
        let tokie_throughput = text_mb / tokie_time;

        // Encode with HF
        let start = Instant::now();
        let hf_encoding = hf.encode(text.clone(), false).expect("HF encode failed");
        let hf_time = start.elapsed().as_secs_f64();
        let hf_throughput = text_mb / hf_time;

        let hf_tokens = strip_padding(hf_encoding.get_ids());

        // Compare
        let identical = diff::is_identical(&tokie_tokens, &hf_tokens);

        println!("  Tokens:     {} (tokie) vs {} (hf)", tokie_tokens.len(), hf_tokens.len());
        println!("  Time:       {:.2}s (tokie) vs {:.2}s (hf)", tokie_time, hf_time);
        println!("  Throughput: {:.0} MB/s (tokie) vs {:.0} MB/s (hf)", tokie_throughput, hf_throughput);
        println!("  Speedup:    {:.1}x", hf_time / tokie_time);

        if identical {
            println!("  Result:     ✓ IDENTICAL\n");
        } else {
            // Find first difference
            let first_diff = diff::first_diff_index(&tokie_tokens, &hf_tokens);
            println!("  Result:     ✗ DIFFERENT (first diff at token {})", first_diff.unwrap_or(0));

            // Show sample diffs
            if let Some(idx) = first_diff {
                // Find byte position of this token
                let mut byte_pos = 0;
                for &tid in &tokie_tokens[..idx.min(tokie_tokens.len())] {
                    if let Some(s) = tokie.decode(&[tid]) {
                        byte_pos += s.len();
                    }
                }

                // Show context
                let start = byte_pos.saturating_sub(20);
                let end = (byte_pos + 50).min(text.len());
                let context = &text[start..end];
                println!("  Context:    ...{:?}...", context);

                // Show tokens around the diff
                let range_start = idx.saturating_sub(2);
                let range_end = (idx + 3).min(tokie_tokens.len()).min(hf_tokens.len());

                println!("  tokie[{}-{}]: {:?}",
                    range_start, range_end,
                    &tokie_tokens[range_start..range_end.min(tokie_tokens.len())]);
                println!("  hf[{}-{}]:    {:?}",
                    range_start, range_end,
                    &hf_tokens[range_start..range_end.min(hf_tokens.len())]);
            }
            println!();
        }

        results.push((test.name, identical, tokie_tokens.len(), hf_tokens.len(), tokie_throughput, hf_time / tokie_time));
    }

    // Summary
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        SUMMARY                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("┌────────────────────┬──────────┬────────────┬────────────┬─────────┐");
    println!("│ Tokenizer          │ Status   │ Throughput │ vs HF      │ Tokens  │");
    println!("├────────────────────┼──────────┼────────────┼────────────┼─────────┤");

    let mut pass_count = 0;
    let mut fail_count = 0;

    for (name, identical, tokie_count, _hf_count, throughput, speedup) in &results {
        let status = if *identical { "✓ PASS" } else { "✗ FAIL" };
        if *identical {
            pass_count += 1;
        } else {
            fail_count += 1;
        }

        println!("│ {:18} │ {:8} │ {:7.0} MB/s │ {:6.1}x     │ {:>7} │",
            name, status, throughput, speedup, tokie_count);
    }

    println!("└────────────────────┴──────────┴────────────┴────────────┴─────────┘");
    println!("\nTotal: {} pass, {} fail", pass_count, fail_count);

    if fail_count > 0 {
        std::process::exit(1);
    }
}
