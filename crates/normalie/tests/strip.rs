fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Strip::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn strip_trims_both_ends() {
    assert_eq!(apply("  a  "), "a");
    assert_eq!(apply("\n\thi\t\n"), "hi");
    assert_eq!(apply("a"), "a");
    assert_eq!(apply("   "), "");
}
