fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::Llama::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn llama_prepends_even_when_a_space_leads() {
    assert_eq!(apply("Hello world"), "▁Hello▁world");
    assert_eq!(apply(" world"), "▁▁world");
    assert_eq!(apply(""), "");
}
