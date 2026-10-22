const T5: &[u8] = include_bytes!("../../../bench/normalize/data/t5.charsmap");

fn map() -> normalie::Precompiled {
    normalie::Precompiled::parse(T5).unwrap()
}

fn apply(text: &str) -> String {
    let mut out = String::new();
    if map().write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn printable_ascii_is_kept() {
    assert_eq!(apply("Hello"), "Hello");
    assert_eq!(apply(""), "");
}
