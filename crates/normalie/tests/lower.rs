fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Lower::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn lower_casefolds_one_character() {
    assert_eq!(apply("HELLO"), "hello");
    assert_eq!(apply("Straße"), "straße");
    assert_eq!(apply("İ"), "i\u{0307}");
    assert_eq!(apply("中文"), "中文");
    assert_eq!(apply(""), "");
}
