fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::ByteLevel::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn printable_ascii_stays_and_space_becomes_u0120() {
    assert_eq!(apply("Hello"), "Hello");
    assert_eq!(apply("Hi!"), "Hi!");
    assert_eq!(apply(" "), "\u{0120}");
    assert_eq!(apply(""), "");
}
