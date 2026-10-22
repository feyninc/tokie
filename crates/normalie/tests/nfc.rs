fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Nfc::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn nfc_composes() {
    assert_eq!(apply(""), "");
    assert_eq!(apply("Hello"), "Hello");
    assert_eq!(apply("e\u{301}"), "é");
    assert_eq!(apply("café"), "café");
}

#[test]
fn nfc_long_matches_short() {
    let pieces = ["café ", "e\u{301}", "\u{1100}\u{1161}", "\u{212B}", "’", "가\u{11A8}"];
    for piece in pieces {
        let short = apply(piece);
        let long = apply(&piece.repeat(400));
        assert_eq!(long, short.repeat(400), "{piece:?}");
    }
}
