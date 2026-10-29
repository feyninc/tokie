fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Prepend::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn prepend_writes_the_mark() {
    assert_eq!(apply(""), "");
    assert_eq!(apply("Hello"), "▁Hello");
    assert_eq!(apply(" world"), "▁ world");
}
