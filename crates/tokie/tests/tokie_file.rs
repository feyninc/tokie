//! A `.tokie` file encodes and decodes like the `tokenizer.json` it was saved from.

use std::path::{Path, PathBuf};

use tokie::{Plus, Tokenizer};

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
        original.encode_plus(sample, &mut wrapped_left).unwrap();
        loaded.encode_plus(sample, &mut wrapped_right).unwrap();
        assert_eq!(wrapped_left, wrapped_right, "encode_plus {sample:?}");
        let mut pair_left = Vec::new();
        let mut pair_right = Vec::new();
        let pair_ok_left = original.encode_plus(Plus::new(sample).pair("other"), &mut pair_left);
        let pair_ok_right = loaded.encode_plus(Plus::new(sample).pair("other"), &mut pair_right);
        match (pair_ok_left, pair_ok_right) {
            (Ok(()), Ok(())) => assert_eq!(pair_left, pair_right, "pair {sample:?}"),
            (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "pair {sample:?}"),
            (Ok(()), Err(err)) => panic!("loaded pair failed: {err}"),
            (Err(err), Ok(())) => panic!("saved pair failed: {err}"),
        }
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

/// Byte-level BPE over GPT-2's 256 byte chars, with one merge, `Ġ` `a`.
fn tiny_json() -> String {
    let mut shown: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut chars = shown.clone();
    let mut extra = 0;
    for byte in 0..256 {
        if !shown.contains(&byte) {
            shown.push(byte);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let quoted = |ch: u32| serde_json::to_string(&char::from_u32(ch).unwrap().to_string()).unwrap();
    let mut vocab: Vec<String> = chars.iter().enumerate().map(|(id, &ch)| format!("{}: {id}", quoted(ch))).collect();
    vocab.push("\"Ġa\": 256".to_string());
    format!(
        r#"{{
            "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}},
            "decoder": {{"type": "ByteLevel"}},
            "model": {{"type": "BPE", "vocab": {{{}}}, "merges": ["Ġ a"]}}
        }}"#,
        vocab.join(", ")
    )
}

fn ids(tok: &mut Tokenizer, text: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    tok.encode(text, &mut ids).unwrap();
    ids
}

#[test]
fn from_bytes_loads_a_tokie_file_and_json() {
    let json = tiny_json();
    let mut original = Tokenizer::json(&json).unwrap();
    let path = temp_path();
    original.save(&path).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let mut from_tokie = Tokenizer::from_bytes(&bytes).unwrap();
    let mut from_json = Tokenizer::from_bytes(json.as_bytes()).unwrap();
    for text in ["", "Hello world", "café 😀"] {
        let want = ids(&mut original, text);
        assert_eq!(ids(&mut from_tokie, text), want, "{text:?}");
        assert_eq!(ids(&mut from_json, text), want, "{text:?}");
    }
    assert!(Tokenizer::from_bytes(b"TOKI\0\0\0\0").is_err());
    assert!(Tokenizer::from_bytes(&[0xff, 0xfe]).is_err());
}

/// A `.tokie` in the Hugging Face cache loads without asking the Hub, also
/// from a folder of the snapshot.
#[cfg(feature = "hub")]
#[test]
fn pretrained_reads_a_cached_tokie_first() {
    let home = temp_path().with_extension("hf");
    let repo = home.join("hub").join("models--tokie-test--cache-first");
    let snapshot = repo.join("snapshots").join("0123abcd").join("assets");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs").join("main"), "0123abcd\n").unwrap();
    let mut original = Tokenizer::json(&tiny_json()).unwrap();
    original.save(snapshot.join("cache-first.tokie")).unwrap();
    // SAFETY: no other test in this binary reads HF_HOME.
    unsafe { std::env::set_var("HF_HOME", &home) };
    let mut loaded = Tokenizer::pretrained("tokie-test/cache-first").unwrap();
    assert_eq!(ids(&mut loaded, "Hello world"), ids(&mut original, "Hello world"));
    let _ = std::fs::remove_dir_all(&home);
}
