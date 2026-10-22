fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::MetaspaceWords::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn whitespace_is_dropped_and_each_word_is_marked() {
    assert_eq!(apply("aa bb cc"), "▁aa▁bb▁cc");
    assert_eq!(apply("aa\tbb  cc"), "▁aa▁bb▁cc");
    assert_eq!(apply(" aa bb"), "▁aa▁bb");
    assert_eq!(apply(" \t\n"), "");
    assert_eq!(apply(""), "");
}
