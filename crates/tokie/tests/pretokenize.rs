//! `pretokenize` and `count_pretokens`: pieces are spans of the caller's
//! text, and there are never more pieces than tokens.

use tokie::Tokenizer;

/// GPT-2's byte-to-char table, so every byte is one vocab entry.
fn byte_chars() -> Vec<char> {
    let mut shown: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut chars: Vec<u32> = shown.clone();
    let mut extra = 0;
    for byte in 0..256 {
        if !shown.contains(&byte) {
            shown.push(byte);
            chars.push(256 + extra);
            extra += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (byte, ch) in shown.iter().zip(&chars) {
        table[*byte as usize] = char::from_u32(*ch).unwrap();
    }
    table
}

fn quoted(ch: char) -> String {
    serde_json::to_string(&ch.to_string()).unwrap()
}

/// Byte-level BPE with one merge, `Ġ` `a`, plus one added token.
fn byte_level(prefix_space: bool) -> Tokenizer {
    let mut vocab: Vec<String> =
        byte_chars().iter().enumerate().map(|(id, &ch)| format!("{}: {id}", quoted(ch))).collect();
    vocab.push("\"Ġa\": 256".to_string());
    let json = format!(
        r#"{{
            "added_tokens": [{{"id": 257, "content": "<|x|>", "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true}}],
            "pre_tokenizer": {{"type": "ByteLevel", "add_prefix_space": {prefix_space}, "use_regex": true}},
            "decoder": {{"type": "ByteLevel"}},
            "model": {{"type": "BPE", "vocab": {{{}}}, "merges": ["Ġ a"]}}
        }}"#,
        vocab.join(", ")
    );
    Tokenizer::json(&json).unwrap()
}

/// Lowercasing WordPiece: the normalizer rewrites the text.
fn bert() -> Tokenizer {
    let words = ["[UNK]", "hello", "world", ",", "!"];
    let vocab: Vec<String> = words.iter().enumerate().map(|(id, w)| format!("\"{w}\": {id}")).collect();
    let json = format!(
        r###"{{
            "normalizer": {{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true,
                "strip_accents": null, "lowercase": true}},
            "pre_tokenizer": {{"type": "BertPreTokenizer"}},
            "model": {{"type": "WordPiece", "unk_token": "[UNK]", "continuing_subword_prefix": "##",
                "max_input_chars_per_word": 100, "vocab": {{{}}}}}
        }}"###,
        vocab.join(", ")
    );
    Tokenizer::json(&json).unwrap()
}

/// T5-style: whitespace split plus metaspace, no normalizer.
fn metaspace() -> Tokenizer {
    let mut pieces = vec![r#"["<unk>", 0.0]"#.to_string(), r#"["▁", -1.0]"#.to_string()];
    for ch in "abcdefghijklmnopqrstuvwxyz,.!'".chars() {
        pieces.push(format!("[{}, -2.0]", quoted(ch)));
    }
    let json = format!(
        r#"{{
            "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [
                {{"type": "WhitespaceSplit"}},
                {{"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}}
            ]}},
            "model": {{"type": "Unigram", "unk_id": 0, "vocab": [{}]}}
        }}"#,
        pieces.join(", ")
    );
    Tokenizer::json(&json).unwrap()
}

fn pieces<'a>(tok: &mut Tokenizer, text: &'a str) -> Vec<&'a str> {
    let mut spans = Vec::new();
    tok.pretokenize(text, &mut spans);
    spans.into_iter().map(|s| &text[s.start as usize..s.end as usize]).collect()
}

const CORPUS: &[&str] = &[
    "",
    " ",
    "\n\n\t ",
    "Hello world",
    " Hello  world  ",
    "Hello, World!",
    "a\r\nb",
    "don't you've",
    "naïve café e\u{301}",
    "日本語 😀👍🏽",
    "x\u{a0}y\u{2003}z",
    "Ab\u{4e2d}cd, X\u{0}y\nCa\u{301}fe\u{301}",
    "<|x|>",
    "a<|x|><|x|>b <|x|>",
    "fn main() {\n    println!(\"hi\");\n}\n",
];

/// Spans are in order, non-empty, on char boundaries, and no more than the
/// tokens. `count_pretokens` agrees with `pretokenize`.
fn check(tok: &mut Tokenizer) {
    let mut spans = Vec::new();
    for text in CORPUS {
        tok.pretokenize(text, &mut spans);
        let mut at = 0;
        for span in &spans {
            let (start, end) = (span.start as usize, span.end as usize);
            assert!(at <= start && start < end, "{text:?}: {spans:?}");
            assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            at = end;
        }
        let tokens = tok.count_tokens(text).unwrap();
        assert!(spans.len() <= tokens, "{text:?}: {} pieces, {tokens} tokens", spans.len());
        assert_eq!(tok.count_pretokens(text), spans.len(), "{text:?}");
    }
}

#[test]
fn byte_level_pieces_are_exact() {
    let mut tok = byte_level(false);
    check(&mut tok);
    assert_eq!(pieces(&mut tok, "Hello world!"), ["Hello", " world", "!"]);
    assert_eq!(pieces(&mut tok, "a<|x|>b c"), ["a", "<|x|>", "b", " c"]);
    assert_eq!(pieces(&mut tok, ""), Vec::<&str>::new());
}

#[test]
fn byte_level_pieces_join_back_to_the_text() {
    let mut tok = byte_level(false);
    for text in CORPUS {
        assert_eq!(pieces(&mut tok, text).concat(), *text);
    }
}

#[test]
fn prefix_space_stays_out_of_the_first_piece() {
    let mut tok = byte_level(true);
    check(&mut tok);
    assert_eq!(pieces(&mut tok, "Hello world"), ["Hello", " world"]);
    assert_eq!(pieces(&mut tok, " Hello"), [" Hello"]);
}

#[test]
fn rewritten_text_keeps_its_cut_when_the_counts_match() {
    let mut tok = bert();
    check(&mut tok);
    assert_eq!(pieces(&mut tok, "Hello, World!"), ["Hello", ",", "World", "!"]);
    assert_eq!(pieces(&mut tok, "\n\t "), Vec::<&str>::new());
}

#[test]
fn rewritten_word_with_a_new_cut_is_one_piece() {
    let mut tok = bert();
    assert_eq!(pieces(&mut tok, "Ab\u{4e2d}cd, ef\nHi!"), ["Ab\u{4e2d}cd,", "ef", "Hi", "!"]);
    assert_eq!(pieces(&mut tok, "X\u{0}y z"), ["X\u{0}y", "z"]);
}

#[test]
fn metaspace_pieces_are_words() {
    let mut tok = metaspace();
    check(&mut tok);
    assert_eq!(pieces(&mut tok, " hello,  world!\n"), ["hello,", "world!"]);
    assert_eq!(pieces(&mut tok, "\u{2003}"), Vec::<&str>::new());
}

