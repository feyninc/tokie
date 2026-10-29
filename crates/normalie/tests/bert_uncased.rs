fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::BertUncased::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn bert_uncased_is_one_pass() {
    assert_eq!(apply("Hello World"), "hello world");
    assert_eq!(apply("CAFÉ"), "cafe");
    assert_eq!(apply("a中b"), "a 中 b");
    assert_eq!(apply("Hello\nWorld"), "hello world");
    assert_eq!(apply("\n"), " ");
    assert_eq!(apply("\u{7f}"), "");
    assert_eq!(apply("\u{378}"), "\u{378}");
}
