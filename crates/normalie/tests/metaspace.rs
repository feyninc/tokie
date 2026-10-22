fn apply(text: &str, at_start: bool) -> String {
    let mut out = String::new();
    if normalie::Metaspace::write(text, at_start, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn metaspace_always_prepends_unless_a_space_or_mark_leads() {
    assert_eq!(apply("aa bb cc", true), "▁aa▁bb▁cc");
    assert_eq!(apply(" aa bb", true), "▁aa▁bb");
    assert_eq!(apply("aa\tbb  cc", true), "▁aa\tbb▁▁cc");
    assert_eq!(apply("▁hello", true), "▁hello");
    assert_eq!(apply("", true), "");
}
