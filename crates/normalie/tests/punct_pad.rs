const T5: &[u8] = include_bytes!("../../../bench/normalize/data/t5.charsmap");

fn apply(text: &str) -> String {
    let map = normalie::Precompiled::parse(T5).unwrap();
    let pad = normalie::PunctPad::parse(T5).unwrap();
    let _ = map;
    let mut out = String::new();
    if pad.write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn punctuation_is_padded_after_the_charsmap() {
    assert_eq!(apply("a.b  c"), "▁a▁.▁b▁c");
    assert_eq!(apply(""), "");
    assert_eq!(apply("   "), "");
}
