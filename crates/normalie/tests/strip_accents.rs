fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::StripAccents::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn strip_accents_keeps_precomposed_letters() {
    assert_eq!(apply("cafe\u{301}"), "cafe");
    assert_eq!(apply("café"), "café");
    assert_eq!(apply("a\u{07FD}b"), "a\u{07FD}b");
    assert_eq!(apply("a\u{1CF2}b"), "ab");
}
