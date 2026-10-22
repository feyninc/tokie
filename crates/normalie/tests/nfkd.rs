fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Nfkd::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn nfkd_decomposes_compatibility() {
    assert_eq!(apply("ﬁ"), "fi");
    assert_eq!(apply("é"), "e\u{301}");
    assert_eq!(apply("Hello"), "Hello");
}
