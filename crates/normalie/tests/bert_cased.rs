fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::BertCased::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn bert_cased_spaces_cjk_and_keeps_case() {
    assert_eq!(apply("Hello\nWorld"), "Hello World");
    assert_eq!(apply("CAFÉ"), "CAFÉ");
    assert_eq!(apply("a中b"), "a 中 b");
}
