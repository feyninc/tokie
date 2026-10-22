fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::MetaspaceNever::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn spaces_become_marks_and_nothing_is_prepended() {
    assert_eq!(apply("aa bb cc"), "aa▁bb▁cc");
    assert_eq!(apply(" aa bb"), "▁aa▁bb");
}
