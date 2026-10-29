fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Nfd::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn nfd_decomposes() {
    assert_eq!(apply("é"), "e\u{301}");
    assert_eq!(apply("e\u{301}"), "e\u{301}");
    assert_eq!(apply("Hello"), "Hello");
}
