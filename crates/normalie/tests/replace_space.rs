fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::ReplaceSpace::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn spaces_become_the_mark() {
    assert_eq!(apply("a b"), "a▁b");
    assert_eq!(apply(" a "), "▁a▁");
    assert_eq!(apply("ab"), "ab");
}
