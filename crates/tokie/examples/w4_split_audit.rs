//! W4 audit: which vocab tokens block the `▁` unit split rules.
//!
//! Run: cargo run --release -p tokie --features hf --example w4_split_audit -- tokiers/gemma-3-4b-it
//!
//! Prints, per model, the tokens with `▁` at an offset > 0 (block the
//! every-`▁` rule), those with a non-`▁` char directly before a `▁` (block
//! the word-start rule), and multi-byte tokens that are not valid UTF-8.

use tokie::encoder::UnitSplit;
use tokie::Tokenizer;

const MS: &[u8] = "▁".as_bytes();

fn main() {
    for repo in std::env::args().skip(1) {
        let tok = match Tokenizer::from_pretrained(&repo) {
            Ok(t) => t,
            Err(e) => {
                println!("{repo}: SKIP {e}");
                continue;
            }
        };
        let all = tok.decoder().token_bytes();
        let (mut interior, mut word, mut invalid) = (Vec::new(), Vec::new(), Vec::new());
        for (id, b) in all.iter().enumerate() {
            if b.len() > 1 && std::str::from_utf8(b).is_err() {
                invalid.push(id);
                continue;
            }
            let mut has_interior = false;
            let mut has_word = false;
            for pos in memchr::memmem::find_iter(b, MS) {
                if pos > 0 {
                    has_interior = true;
                    if pos < 3 || &b[pos - 3..pos] != MS {
                        has_word = true;
                    }
                }
            }
            if has_interior {
                interior.push(id);
            }
            if has_word {
                word.push(id);
            }
        }
        let show = |ids: &[usize]| -> String {
            ids.iter()
                .take(12)
                .map(|&i| format!("{i}:{:?}", String::from_utf8_lossy(&all[i])))
                .collect::<Vec<_>>()
                .join(" ")
        };
        let vocab_rule = UnitSplit::classify(all.iter().map(|b| b.as_slice()));
        println!(
            "{repo}: vocab={} encoder={:?} whole-vocab-rule={vocab_rule:?} interior-▁={} non-▁-then-▁={} invalid-utf8={}",
            all.len(),
            match tok.encoder() {
                tokie::encoder::Encoder::SentencePiece(e) => Some(e.unit_split()),
                tokie::encoder::Encoder::Unigram(e) => Some(e.unit_split()),
                _ => None,
            },
            interior.len(),
            word.len(),
            invalid.len()
        );
        println!("  word-rule blockers: {}", show(&word));
        println!("  interior examples : {}", show(&interior));
        if !invalid.is_empty() {
            println!("  invalid examples  : {}", show(&invalid));
        }
    }
}
