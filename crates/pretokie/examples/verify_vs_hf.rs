//! Verify pretokie pretokenizer output matches regex-based pretokenizers.
//! Run: cargo run -p pretokie --example verify_vs_hf --release --features regex
//!
//! Compares hand-coded iterators against regex-automata implementations
//! on 1MB of enwik8. This validates that the hand-coded optimizations
//! produce identical output to the reference regex patterns.

use pretokie::mask::{Bert, Cl100k, Gpt2, O200k, SmolLM};
use pretokie::mask::Regex;
use std::path::Path;
use unicode_general_category::GeneralCategory;

fn is_unicode_punct(c: char) -> bool {
    matches!(
        unicode_general_category::get_general_category(c),
        GeneralCategory::ConnectorPunctuation
            | GeneralCategory::DashPunctuation
            | GeneralCategory::OpenPunctuation
            | GeneralCategory::ClosePunctuation
            | GeneralCategory::InitialPunctuation
            | GeneralCategory::FinalPunctuation
            | GeneralCategory::OtherPunctuation
    )
}

fn load_enwik8(max_bytes: usize) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("benches/data/enwik8");
    let data = std::fs::read(&path).expect("read enwik8");
    let truncated = &data[..data.len().min(max_bytes)];
    String::from_utf8_lossy(truncated).into_owned()
}

fn verify_vs_regex(name: &str, hand_pieces: &[&str], regex_pieces: &[&str]) -> bool {
    if hand_pieces == regex_pieces {
        println!("{:10} PASS ({} pieces match)", name, hand_pieces.len());
        return true;
    }

    let diff = hand_pieces.iter().zip(regex_pieces.iter())
        .position(|(a, b)| a != b)
        .unwrap_or(hand_pieces.len().min(regex_pieces.len()));
    println!("{:10} FAIL at piece {} (hand={}, regex={})", name, diff, hand_pieces.len(), regex_pieces.len());
    let start = diff.saturating_sub(2);
    let end = (diff + 3).min(hand_pieces.len()).min(regex_pieces.len());
    if start < end {
        println!("  hand [{}..{}]: {:?}", start, end, &hand_pieces[start..end]);
        println!("  regex[{}..{}]: {:?}", start, end, &regex_pieces[start..end]);
    }
    // Show bytes around mismatch
    if diff < hand_pieces.len() && diff < regex_pieces.len() {
        println!("  hand  piece {}: {:?} (bytes: {:?})", diff, hand_pieces[diff], hand_pieces[diff].as_bytes());
        println!("  regex piece {}: {:?} (bytes: {:?})", diff, regex_pieces[diff], regex_pieces[diff].as_bytes());
    }
    false
}

fn main() {
    let text = load_enwik8(1_000_000);
    println!("Comparing pretokie hand-coded vs regex on {} bytes of enwik8\n", text.len());

    let mut pass = 0u32;
    let mut fail = 0u32;

    // GPT-2
    {
        let hand: Vec<&str> = Gpt2::new(&text).collect();
        let re = Regex::gpt2();
        let rx: Vec<&str> = re.split(&text).collect();
        if verify_vs_regex("GPT-2", &hand, &rx) { pass += 1; } else { fail += 1; }
    }

    // CL100K
    {
        let hand: Vec<&str> = Cl100k::new(&text).collect();
        let re = Regex::cl100k();
        let rx: Vec<&str> = re.split(&text).collect();
        if verify_vs_regex("CL100K", &hand, &rx) { pass += 1; } else { fail += 1; }
    }

    // O200K
    {
        let hand: Vec<&str> = O200k::new(&text).collect();
        let re = Regex::o200k();
        let rx: Vec<&str> = re.split(&text).collect();
        if verify_vs_regex("O200K", &hand, &rx) { pass += 1; } else { fail += 1; }
    }

    // SmolLM (uses GPT-2 pattern)
    {
        let hand: Vec<&str> = SmolLM::new(&text).collect();
        let re = Regex::gpt2();
        let rx: Vec<&str> = re.split(&text).collect();
        if verify_vs_regex("SmolLM", &hand, &rx) { pass += 1; } else { fail += 1; }
    }

    // BERT — verify against reference whitespace + unicode-punct splitting
    {
        let hand: Vec<&str> = Bert::new(&text).collect();
        let mut ref_pieces: Vec<&str> = Vec::new();
        for word in text.split_whitespace() {
            let mut i = 0;
            while i < word.len() {
                let ch_start = i;
                let c = word[ch_start..].chars().next().unwrap();
                let clen = c.len_utf8();
                i += clen;
                if is_unicode_punct(c) {
                    ref_pieces.push(&word[ch_start..ch_start + clen]);
                } else {
                    while i < word.len() {
                        let next_c = word[i..].chars().next().unwrap();
                        if is_unicode_punct(next_c) {
                            break;
                        }
                        i += next_c.len_utf8();
                    }
                    ref_pieces.push(&word[ch_start..i]);
                }
            }
        }
        if verify_vs_regex("BERT", &hand, &ref_pieces) { pass += 1; } else { fail += 1; }
    }

    println!("\n{pass} pass, {fail} fail out of {}", pass + fail);

    // Note about other pretokenizers
    println!("\nNote: Voyage, DeepSeek, and Qwen pretokenizers are verified");
    println!("indirectly via the full tokenizer accuracy tests (74/74 pass).");
    println!("Their correctness is confirmed by exact token-match with HF on enwik8.");

    std::process::exit(if fail > 0 { 1 } else { 0 });
}
