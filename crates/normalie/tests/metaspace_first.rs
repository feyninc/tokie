fn apply(text: &str, at_start: bool) -> String {
    let mut out = String::new();
    if normalie::MetaspaceFirst::write(text, at_start, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn the_leading_mark_is_only_at_the_start() {
    assert_eq!(apply("aa bb cc", true), "▁aa▁bb▁cc");
    assert_eq!(apply("aa bb cc", false), "aa▁bb▁cc");
    assert_eq!(apply(" aa bb", false), "▁aa▁bb");
}
