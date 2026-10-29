//! Verify precompiled-charsmap parity: tokie (from tokenizer.json, bypassing
//! any cached .tkz) vs HuggingFace tokenizers, per-document on OpenWebText.
//!
//! Run with:
//!   cargo run --release --example verify_precompiled_charsmap --features build [model ...]
//!
//! Defaults to the Precompiled-normalizer models. Requires benches/data/owt_sample.txt.

use std::path::Path;

use tokenizers::Tokenizer as HfTokenizer;
use tokie::Tokenizer;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // --pretrained: load tokie via from_pretrained (tokiers/*.tkz baseline)
    // instead of from_json, to measure the old normalizer's gap.
    let use_pretrained = if let Some(pos) = args.iter().position(|a| a == "--pretrained") {
        args.remove(pos);
        true
    } else {
        false
    };
    let models: Vec<String> = if args.is_empty() {
        vec![
            "FacebookAI/xlm-roberta-base".into(),
            "google-t5/t5-base".into(),
            "BAAI/bge-m3".into(),
            "Snowflake/snowflake-arctic-embed-l-v2.0".into(),
        ]
    } else {
        args
    };

    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("benches/data/owt_sample.txt");
    let data = std::fs::read(&path).expect("missing owt_sample.txt");
    let text = String::from_utf8_lossy(&data[..data.len().min(25_000_000)]);
    let docs: Vec<&str> = text
        .split("<|endoftext|>")
        .filter(|d| !d.is_empty())
        .collect();
    println!("{} docs loaded", docs.len());

    let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();
    let mut any_fail = false;

    for model in &models {
        let tok = if use_pretrained {
            Tokenizer::from_pretrained(model)
                .unwrap_or_else(|e| panic!("tokie from_pretrained {model}: {e}"))
        } else {
            let json_path = api
                .repo(hf_hub::Repo::model(model.clone()))
                .get("tokenizer.json")
                .unwrap_or_else(|e| panic!("download tokenizer.json for {model}: {e}"));
            Tokenizer::from_json(&json_path)
                .unwrap_or_else(|e| panic!("tokie from_json {model}: {e}"))
        };
        println!("{model}: normalizer = {:?}", tok.normalizer());
        let mut hf = HfTokenizer::from_pretrained(model, None)
            .unwrap_or_else(|e| panic!("HF load {model}: {e}"));
        let _ = hf.with_truncation(None);

        let mut mismatched = 0usize;
        let mut first: Option<String> = None;
        for (i, doc) in docs.iter().enumerate() {
            let tokie_ids = tok.encode(doc, false).ids;
            let hf_enc = hf.encode(*doc, false).unwrap();
            let hf_ids = hf_enc.get_ids();
            if tokie_ids.as_slice() != hf_ids {
                mismatched += 1;
                if first.is_none() {
                    let diff = tokie_ids
                        .iter()
                        .zip(hf_ids.iter())
                        .position(|(a, b)| a != b)
                        .unwrap_or(tokie_ids.len().min(hf_ids.len()));
                    let byte = hf_enc.get_offsets().get(diff).map_or(0, |o| o.0);
                    let lo = (0..=byte.min(doc.len()))
                        .rev()
                        .take(40)
                        .find(|&p| doc.is_char_boundary(p))
                        .unwrap_or(0);
                    let hi = (byte..=doc.len())
                        .take(80)
                        .filter(|&p| doc.is_char_boundary(p))
                        .last()
                        .unwrap_or(doc.len());
                    let window = diff.saturating_sub(3)..(diff + 8);
                    let tokie_toks: Vec<String> = window
                        .clone()
                        .filter_map(|k| tokie_ids.get(k))
                        .map(|&id| tok.decode(&[id]).unwrap_or_default())
                        .collect();
                    let hf_toks: Vec<&str> = window
                        .filter_map(|k| hf_enc.get_tokens().get(k))
                        .map(|s| s.as_str())
                        .collect();
                    first = Some(format!(
                        "doc {i} token {diff} byte {byte}: {:?}\n  tokie: {tokie_toks:?}\n  hf:    {hf_toks:?}",
                        &doc[lo..hi]
                    ));
                }
            }
        }
        if mismatched == 0 {
            println!("  PASS {}/{} docs match", docs.len(), docs.len());
        } else {
            any_fail = true;
            println!("  FAIL {mismatched}/{} docs mismatch", docs.len());
            println!("  first: {}", first.unwrap());
        }

        // .tkz round-trip: save, reload, and re-encode a sample of docs
        let tkz_path = std::env::temp_dir().join("verify_charsmap_roundtrip.tkz");
        tok.to_file(&tkz_path).expect("save .tkz");
        let tok2 = Tokenizer::from_file(&tkz_path).expect("load .tkz");
        let rt_mismatch = docs
            .iter()
            .step_by(50)
            .filter(|doc| tok.encode(doc, false).ids != tok2.encode(doc, false).ids)
            .count();
        if rt_mismatch == 0 {
            println!("  PASS .tkz round-trip identical on sampled docs");
        } else {
            any_fail = true;
            println!("  FAIL .tkz round-trip: {rt_mismatch} sampled docs differ");
        }
    }

    if any_fail {
        std::process::exit(1);
    }
}
