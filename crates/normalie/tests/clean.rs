fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Clean::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn clean_folds_whitespace_and_drops_controls() {
    assert_eq!(apply("Hello\nWorld"), "Hello World");
    assert_eq!(apply("a\u{00A0}b"), "a b");
    assert_eq!(apply("null\0here"), "nullhere");
    assert_eq!(apply("Hello"), "Hello");
}
