fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::CollapseSpaces::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn runs_of_spaces_collapse() {
    assert_eq!(apply("a  b"), "a b");
    assert_eq!(apply("a   b"), "a b");
    assert_eq!(apply("a b"), "a b");
    assert_eq!(apply("a\t\tb"), "a\t\tb");
    assert_eq!(apply("  "), " ");
}
