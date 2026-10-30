//! A `.tokie` file encodes and decodes like the `tokenizer.json` it was saved from.

use std::path::{Path, PathBuf};

use tokie::Tokenizer;

fn roundtrip(json: &Path) {
    let mut original = Tokenizer::open(json).unwrap_or_else(|err| panic!("{}: {err}", json.display()));
    let path = temp_path();
    original.save(&path).unwrap_or_else(|err| panic!("save: {err}"));
    let mut loaded = Tokenizer::open(&path).unwrap_or_else(|err| panic!("load: {err}"));
    let samples = ["", "Hello world", "  leading", "café", "a\nb", "<|endoftext|>"];
    for sample in samples {
        let mut left = Vec::new();
        let mut right = Vec::new();
        original.encode(sample, &mut left).unwrap();
        loaded.encode(sample, &mut right).unwrap();
        assert_eq!(left, right, "{sample:?}");
        let mut wrapped_left = Vec::new();
        let mut wrapped_right = Vec::new();
        original.wrap(sample, &mut wrapped_left).unwrap();
        loaded.wrap(sample, &mut wrapped_right).unwrap();
        assert_eq!(wrapped_left, wrapped_right, "wrap {sample:?}");
        let mut text_left = String::new();
        let mut text_right = String::new();
        original
            .decode(&left, &mut text_left, tokie::Specials::Keep)
            .unwrap();
        loaded
            .decode(&right, &mut text_right, tokie::Specials::Keep)
            .unwrap();
        assert_eq!(text_left, text_right);
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(format!("{}.tmp", path.display()));
}

fn temp_path() -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("tokie-roundtrip-{}-{id}.tokie", std::process::id()));
    path
}

#[test]
fn old_tkz_magic_is_refused() {
    let path = temp_path();
    std::fs::write(&path, b"TOKI\0\0\0\0not a tokenizer").unwrap();
    let err = match Tokenizer::open(&path) {
        Ok(_) => panic!("old .tkz magic was accepted"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("old .tkz"), "{err}");
    let _ = std::fs::remove_file(path);
}

#[test]
fn gpt2_roundtrip() {
    let path = Path::new(concat!(
        "/home/bhavnick/.cache/huggingface/hub/",
        "models--openai-community--gpt2/snapshots/",
        "607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json"
    ));
    if path.exists() {
        roundtrip(path);
    }
}

#[test]
fn bert_roundtrip() {
    let path = Path::new(concat!(
        "/home/bhavnick/.cache/huggingface/hub/",
        "models--google-bert--bert-base-uncased/snapshots/",
        "86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json"
    ));
    if path.exists() {
        roundtrip(path);
    }
}

#[test]
fn t5_roundtrip() {
    let path = Path::new(
        "/home/bhavnick/.cache/huggingface/hub/models--t5-small/snapshots/main/tokenizer.json",
    );
    if path.exists() {
        roundtrip(path);
    }
}
