//! The mainline API: load aliases, raw vocab views, pad, template counts,
//! and byte decode. Every fixture is a small synthetic `tokenizer.json`, so a
//! checkout with no downloaded model still runs these tests.

use std::path::PathBuf;

use tokie::{Plus, Specials, TokenId, Tokenizer};

/// Small BPE. `merges` must name three vocab strings, so the pair is `.` `:`.
const BASE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2, ".:.:": 3},
        "merges": [". :"]
    }
}"#;

/// Sparse ids and an added token above them.
const SPARSE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {"a": 0, "b": 2, "c": 5, "ab": 6, "Ġa": 7},
        "merges": ["a b"]
    },
    "added_tokens": [
        {
            "id": 9,
            "content": "<|z|>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }
    ]
}"#;

/// Both pad spellings. `[PAD]` wins.
const PAD_BOTH: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2, "[PAD]": 6, "<pad>": 7},
        "merges": [". :"]
    }
}"#;

/// Only the lowercase spelling.
const PAD_ANGLE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2, "<pad>": 7},
        "merges": [". :"]
    }
}"#;

/// Single template whose specials are multi-id runs: 2 before A, 1 after.
const SINGLE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2},
        "merges": [". :"]
    },
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [4, 5], "tokens": ["[CLS]", "[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [6], "tokens": ["[SEP]"]}
        }
    }
}"#;

/// Pair template: 2 before A, 1 between, 3 after B.
const PAIR: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2},
        "merges": [". :"]
    },
    "post_processor": {
        "type": "TemplateProcessing",
        "single": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
        ],
        "pair": [
            {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}},
            {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
            {"Sequence": {"id": "B", "type_id": 1}},
            {"SpecialToken": {"id": "[END]", "type_id": 1}}
        ],
        "special_tokens": {
            "[CLS]": {"id": "[CLS]", "ids": [4, 5], "tokens": ["[CLS]", "[CLS]"]},
            "[SEP]": {"id": "[SEP]", "ids": [6], "tokens": ["[SEP]"]},
            "[END]": {"id": "[END]", "ids": [7, 8, 9], "tokens": ["[END]", "[END]", "[END]"]}
        }
    }
}"#;

/// A post-processor that is not a template adds nothing.
const BYTELEVEL_POST: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {".": 0, ":": 1, ".:": 2},
        "merges": [". :"]
    },
    "post_processor": {"type": "ByteLevel", "add_prefix_space": true}
}"#;

/// SentencePiece-style bytes on a BPE model: the decoder asks for raw byte
/// tokens, and `Strip` asks for the lead-space strip. Ids 0..=4 are single
/// bytes, 5 is `b`, 6 is a gap, 7 is a merge result, 8 is an added special.
const BYTES: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {
            "<0x20>": 0,
            "<0x61>": 1,
            "<0xFF>": 2,
            "<0xC3>": 3,
            "<0xA9>": 4,
            "<0x62>": 5,
            "<0x61><0x62>": 7
        },
        "merges": ["<0x61> <0x62>"]
    },
    "decoder": {
        "type": "Sequence",
        "decoders": [{"type": "ByteFallback"}, {"type": "Strip"}]
    },
    "added_tokens": [
        {
            "id": 8,
            "content": "<|eot|>",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }
    ]
}"#;

/// WordPiece: the decoder gives `hello` a leading ` ` in its next form and
/// `##ing` a bare `ing` next form. Id 4 is the gap between the vocab and the
/// added special.
const WORDPIECE: &str = r###"{
    "model": {
        "type": "WordPiece",
        "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "##ing": 3},
        "unk_token": "[UNK]"
    },
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": false},
    "added_tokens": [
        {
            "id": 5,
            "content": "[SPECIAL]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }
    ]
}"###;

/// BPE with an explicit Metaspace decoder, so the form is the same one a
/// Unigram file gets: the lead form drops `▁`, the next form replaces it with
/// a space. Id 5 is the gap before the added special.
const METASPACE: &str = r#"{
    "model": {
        "type": "BPE",
        "vocab": {"▁Hello": 0, "▁world": 1, "H": 2, "w": 3, "Hw": 4},
        "merges": ["H w"]
    },
    "decoder": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never"},
    "added_tokens": [
        {
            "id": 6,
            "content": "[SPECIAL]",
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        }
    ]
}"#;

fn temp_path(name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let id = NEXT.fetch_add(1, Ordering::Relaxed);
    let mut path = std::env::temp_dir();
    path.push(format!("tokie-api-{}-{id}-{name}", std::process::id()));
    path
}

fn cleanup(path: &PathBuf) {
    let _ = std::fs::remove_file(path);
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let _ = std::fs::remove_file(PathBuf::from(tmp));
}

/// `Tokenizer` has no `Debug`, so `unwrap_err` is not available.
fn load_err(result: Result<Tokenizer, tokie::Error>) -> tokie::Error {
    match result {
        Ok(_) => panic!("tokenizer loaded"),
        Err(err) => err,
    }
}

#[test]
fn from_json_and_from_file_take_a_path() {
    let path = temp_path("base.json");
    std::fs::write(&path, BASE).unwrap();

    let mut from_json = Tokenizer::from_json(&path).unwrap();
    let mut from_file = Tokenizer::from_file(&path).unwrap();
    let mut from_text = Tokenizer::json(BASE).unwrap();

    for tokenizer in [&mut from_json, &mut from_file, &mut from_text] {
        let mut ids = Vec::new();
        tokenizer.encode(".:.", &mut ids).unwrap();
        assert_eq!(ids, [2, 0]);
    }

    // `json` takes the contents. The aliases take a path, so the contents are
    // not a file name and the load fails.
    assert!(Tokenizer::json(BASE).is_ok());
    assert!(Tokenizer::from_json(BASE).is_err());
    assert!(Tokenizer::from_file(BASE).is_err());

    cleanup(&path);
}

#[test]
fn to_file_saves_a_new_tokie_file() {
    let path = temp_path("saved.tokie");
    let mut original = Tokenizer::json(SPARSE).unwrap();
    original.to_file(&path).unwrap();

    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"TOKIE\x89\r\n");

    let mut loaded = Tokenizer::from_file(&path).unwrap();
    let mut via_json = Tokenizer::from_json(&path).unwrap();
    assert_eq!(loaded.vocab_size(), original.vocab_size());
    assert_eq!(via_json.vocab_size(), original.vocab_size());
    assert_eq!(loaded.get_vocab(), original.get_vocab());
    assert_eq!(via_json.get_vocab(), original.get_vocab());

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    original.encode("ab", &mut a).unwrap();
    loaded.encode("ab", &mut b).unwrap();
    via_json.encode("ab", &mut c).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, c);

    cleanup(&path);
}

#[test]
fn saving_over_a_saved_file_reopens_with_the_new_vocab() {
    let path = temp_path("resaved.tokie");
    let base = Tokenizer::json(BASE).unwrap();
    base.save(&path).unwrap();

    // The second save replaces the first. Its temporary is reopened for
    // writing before the flush, so Windows must accept it the same way.
    let sparse = Tokenizer::json(SPARSE).unwrap();
    sparse.save(&path).unwrap();

    let mut reopened = Tokenizer::from_file(&path).unwrap();
    assert_eq!(reopened.get_vocab(), sparse.get_vocab());
    assert_ne!(reopened.get_vocab(), base.get_vocab());

    let mut ids = Vec::new();
    reopened.encode("ab", &mut ids).unwrap();
    assert_eq!(ids, [6]);

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    assert!(
        !PathBuf::from(tmp).exists(),
        "save left the temporary file behind"
    );

    cleanup(&path);
}

#[test]
fn old_toki_magic_is_refused_by_the_aliases() {
    let path = temp_path("old.tkz");
    std::fs::write(&path, b"TOKI\0\0\0\0not a tokenizer").unwrap();

    for result in [Tokenizer::from_json(&path), Tokenizer::from_file(&path)] {
        let err = load_err(result);
        assert!(err.to_string().contains("old .tkz"), "{err}");
    }

    cleanup(&path);
}

#[test]
fn missing_and_invalid_files_are_errors() {
    let missing = temp_path("missing.json");
    assert!(Tokenizer::from_json(&missing).is_err());
    assert!(Tokenizer::from_file(&missing).is_err());

    let bad = temp_path("bad.json");
    std::fs::write(&bad, b"not json").unwrap();
    let err = load_err(Tokenizer::from_file(&bad));
    assert!(err.to_string().contains("json"), "{err}");

    let non_utf8 = temp_path("non-utf8.json");
    std::fs::write(&non_utf8, [0xff_u8, 0xfe]).unwrap();
    let err = load_err(Tokenizer::from_json(&non_utf8));
    assert!(err.to_string().contains("utf-8"), "{err}");

    cleanup(&bad);
    cleanup(&non_utf8);
}

#[test]
fn get_vocab_keeps_raw_spellings_and_size_counts_gaps() {
    let tok = Tokenizer::json(SPARSE).unwrap();
    assert_eq!(tok.vocab_size(), 10);

    let vocab = tok.get_vocab();
    assert_eq!(vocab.len(), 6);

    let a: TokenId = *vocab.get("a").unwrap();
    assert_eq!(a, 0);
    assert_eq!(vocab.get("b"), Some(&2));
    assert_eq!(vocab.get("c"), Some(&5));
    assert_eq!(vocab.get("ab"), Some(&6));
    assert_eq!(vocab.get("<|z|>"), Some(&9));

    // The raw spelling survives. A decoded spelling would be ` a` here.
    assert_eq!(vocab.get("Ġa"), Some(&7));
    assert!(!vocab.contains_key(" a"));

    for (token, id) in &vocab {
        assert_eq!(tok.id(token), Some(*id), "{token}");
        assert_eq!(tok.token(*id), Some(token.as_str()), "{token}");
    }

    assert_eq!(tok.token(1), None);
    assert_eq!(tok.token(3), None);
    assert_eq!(tok.token(10), None);
}

#[test]
fn pad_token_id_takes_only_the_two_names() {
    assert_eq!(Tokenizer::json(PAD_BOTH).unwrap().pad_token_id(), Some(6));
    assert_eq!(Tokenizer::json(PAD_ANGLE).unwrap().pad_token_id(), Some(7));
    assert_eq!(Tokenizer::json(BASE).unwrap().pad_token_id(), None);

    // The id `encode_plus` pads with is the one this call reports.
    let mut tok = Tokenizer::json(PAD_BOTH).unwrap();
    let pad = tok.pad_token_id().unwrap();
    let mut ids = Vec::new();
    tok.encode_plus(Plus::new(".:.").pad(6), &mut ids).unwrap();
    assert_eq!(ids, [2, 0, pad, pad, pad, pad]);
}

#[test]
fn num_special_tokens_to_add_counts_template_runs() {
    let plain = Tokenizer::json(BASE).unwrap();
    assert_eq!(plain.num_special_tokens_to_add(false), 0);
    assert_eq!(plain.num_special_tokens_to_add(true), 0);

    let bytelevel = Tokenizer::json(BYTELEVEL_POST).unwrap();
    assert_eq!(bytelevel.num_special_tokens_to_add(false), 0);
    assert_eq!(bytelevel.num_special_tokens_to_add(true), 0);

    let mut single = Tokenizer::json(SINGLE).unwrap();
    assert_eq!(single.num_special_tokens_to_add(false), 3);
    // No pair template, so a pair adds nothing.
    assert_eq!(single.num_special_tokens_to_add(true), 0);

    let mut bare = Vec::new();
    single.encode(".:.", &mut bare).unwrap();
    let mut wrapped = Vec::new();
    single.encode_plus(".:.", &mut wrapped).unwrap();
    assert_eq!(wrapped.len() - bare.len(), single.num_special_tokens_to_add(false));

    let mut pair = Tokenizer::json(PAIR).unwrap();
    assert_eq!(pair.num_special_tokens_to_add(false), 3);
    assert_eq!(pair.num_special_tokens_to_add(true), 6);

    let mut first = Vec::new();
    let mut second = Vec::new();
    pair.encode(".:.", &mut first).unwrap();
    pair.encode(":", &mut second).unwrap();
    let mut paired = Vec::new();
    pair.encode_plus(Plus::new(".:.").pair(":"), &mut paired).unwrap();
    assert_eq!(
        paired.len() - first.len() - second.len(),
        pair.num_special_tokens_to_add(true)
    );
}

#[test]
fn decode_bytes_matches_decode_and_keeps_raw_bytes() {
    let tok = Tokenizer::json(BYTES).unwrap();
    assert_eq!(tok.vocab_size(), 9);

    let cases: &[&[u32]] = &[
        &[],
        &[1],
        &[0, 1],
        &[1, 0],
        &[2],
        &[3],
        &[4],
        &[3, 4],
        &[8],
        &[8, 1],
        &[0, 8, 1],
    ];
    for ids in cases {
        for specials in [Specials::Keep, Specials::Skip] {
            let bytes = tok.decode_bytes(ids, specials).unwrap();
            let mut text = String::new();
            tok.decode(ids, &mut text, specials).unwrap();
            let decoded = String::from_utf8_lossy(&bytes);
            assert_eq!(&*decoded, text.as_str(), "{ids:?} {specials:?}");
        }
    }

    // A lone `0xFF` is not UTF-8. The bytes stay as they are, while `decode`
    // writes the replacement character.
    let raw = tok.decode_bytes(&[2], Specials::Keep).unwrap();
    assert_eq!(raw, [0xFF]);
    let mut lossy = String::new();
    tok.decode(&[2], &mut lossy, Specials::Keep).unwrap();
    assert_eq!(lossy, "\u{FFFD}");
    let raw_lossy = String::from_utf8_lossy(&raw);
    assert_eq!(&*raw_lossy, lossy.as_str());

    // Two bytes that are one character together.
    assert_eq!(tok.decode_bytes(&[3, 4], Specials::Keep).unwrap().as_slice(), "é".as_bytes());

    // An added special decodes to the bytes of its literal spelling, and
    // `Skip` drops it.
    assert_eq!(tok.decode_bytes(&[8], Specials::Keep).unwrap().as_slice(), b"<|eot|>");
    assert!(tok.decode_bytes(&[8], Specials::Skip).unwrap().is_empty());

    // The lead space is stripped only in front of the first kept id.
    assert_eq!(tok.decode_bytes(&[0, 1], Specials::Keep).unwrap().as_slice(), b"a");
    assert_eq!(tok.decode_bytes(&[1, 0], Specials::Keep).unwrap().as_slice(), b"a ");
    assert_eq!(tok.decode_bytes(&[0, 8, 1], Specials::Keep).unwrap().as_slice(), b"<|eot|>a");
    assert_eq!(tok.decode_bytes(&[0, 8, 1], Specials::Skip).unwrap().as_slice(), b"a");

    // Ids past the slots and gap ids are errors.
    for id in [6, 9, 100] {
        let err = match tok.decode_bytes(&[id], Specials::Keep) {
            Ok(_) => panic!("id {id} decoded"),
            Err(err) => err,
        };
        assert!(err.to_string().contains(&format!("unknown id {id}")), "{err}");
    }
}

#[test]
fn decode_bytes_uses_wordpiece_lead_and_next_forms() {
    let tok = Tokenizer::json(WORDPIECE).unwrap();
    assert_eq!(tok.vocab_size(), 6);

    // `hello` takes its lead form. `##ing` is not first, so it takes the next
    // form, which drops the `##`.
    assert_eq!(tok.decode_bytes(&[1, 3], Specials::Keep).unwrap(), b"helloing".to_vec());

    // A leading `##ing` keeps its prefix.
    assert_eq!(tok.decode_bytes(&[3], Specials::Keep).unwrap(), b"##ing".to_vec());

    // A word after another takes the next form, which starts with a space.
    assert_eq!(tok.decode_bytes(&[1, 2], Specials::Keep).unwrap(), b"hello world".to_vec());

    // A skipped leading special does not mark an id emitted, so `##ing` is
    // still a lead form.
    assert_eq!(tok.decode_bytes(&[5, 3], Specials::Skip).unwrap(), b"##ing".to_vec());

    // Kept, the special is its literal spelling and `##ing` takes its next
    // form.
    assert_eq!(tok.decode_bytes(&[5, 3], Specials::Keep).unwrap(), b"[SPECIAL]ing".to_vec());

    let keep: &[(&[u32], &str)] = &[
        (&[1, 3], "helloing"),
        (&[3], "##ing"),
        (&[1, 2], "hello world"),
        (&[1, 3, 2], "helloing world"),
        (&[0], "[UNK]"),
        (&[5, 3], "[SPECIAL]ing"),
    ];
    for (ids, expect) in keep {
        let bytes = tok.decode_bytes(ids, Specials::Keep).unwrap();
        assert_eq!(bytes.as_slice(), expect.as_bytes(), "{ids:?}");
        let mut text = String::new();
        tok.decode(ids, &mut text, Specials::Keep).unwrap();
        assert_eq!(text, *expect, "{ids:?}");
    }

    let skipped = tok.decode_bytes(&[5, 3], Specials::Skip).unwrap();
    assert_eq!(skipped.as_slice(), b"##ing");
    let mut text = String::new();
    tok.decode(&[5, 3], &mut text, Specials::Skip).unwrap();
    assert_eq!(text, "##ing");
}

#[test]
fn decode_bytes_uses_metaspace_lead_and_next_forms() {
    let tok = Tokenizer::json(METASPACE).unwrap();
    assert_eq!(tok.vocab_size(), 7);

    // The first id drops the `▁` marker and the next id turns it into a space.
    assert_eq!(tok.decode_bytes(&[0, 1], Specials::Keep).unwrap(), b"Hello world".to_vec());
    assert_eq!(tok.decode_bytes(&[1], Specials::Keep).unwrap(), b"world".to_vec());
    assert_eq!(tok.decode_bytes(&[0], Specials::Keep).unwrap(), b"Hello".to_vec());

    let keep: &[(&[u32], &str)] = &[
        (&[0], "Hello"),
        (&[1], "world"),
        (&[0, 1], "Hello world"),
        (&[6], "[SPECIAL]"),
        (&[6, 1], "[SPECIAL] world"),
    ];
    for (ids, expect) in keep {
        let bytes = tok.decode_bytes(ids, Specials::Keep).unwrap();
        assert_eq!(bytes.as_slice(), expect.as_bytes(), "{ids:?}");
        let mut text = String::new();
        tok.decode(ids, &mut text, Specials::Keep).unwrap();
        assert_eq!(text, *expect, "{ids:?}");
    }

    // A skipped leading special does not mark an id emitted, so `world` drops
    // its own marker instead of taking the space form.
    let skipped = tok.decode_bytes(&[6, 1], Specials::Skip).unwrap();
    assert_eq!(skipped.as_slice(), b"world");
    let mut text = String::new();
    tok.decode(&[6, 1], &mut text, Specials::Skip).unwrap();
    assert_eq!(text, "world");
}

#[test]
fn encode_still_appends_and_keeps_the_json_ids() {
    let mut tok = Tokenizer::json(BASE).unwrap();
    let mut ids = vec![9];
    tok.encode(".", &mut ids).unwrap();
    assert_eq!(ids, [9, 0]);
    tok.encode(".:.", &mut ids).unwrap();
    assert_eq!(ids, [9, 0, 2, 0]);
}

#[test]
fn from_pretrained_alias_has_the_mainline_signature() {
    // Compile check only. Calling either closure would reach the Hub.
    let load: fn(&str) -> Result<Tokenizer, tokie::Error> = |repo| Tokenizer::from_pretrained(repo);
    let _ = load;
    let _ = |repo: String| Tokenizer::from_pretrained(repo);
}
