fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Albert::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn albert_folds_to_metaspace() {
    assert_eq!(apply("  Héllo   Wörld\n"), "▁hello▁world");
    assert_eq!(apply(""), "");
    assert_eq!(apply("   "), "");
}
