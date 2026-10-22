fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Nmt::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn nmt_folds_the_documented_scalars() {
    assert_eq!(apply("\u{0001}a\tb"), "a b");
    assert_eq!(apply("▁x"), " x");
    assert_eq!(apply("\u{FEFF}hi"), " hi");
    assert_eq!(apply("a\u{007F}b"), "ab");
    assert_eq!(apply("a\u{000B}b"), "ab");
    assert_eq!(apply("keep\n"), "keep ");
}
