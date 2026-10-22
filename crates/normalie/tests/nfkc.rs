fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Nfkc::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn nfkc_folds_compatibility() {
    assert_eq!(apply("ﬁ"), "fi");
    assert_eq!(apply("①"), "1");
    assert_eq!(apply("Hello"), "Hello");
}
