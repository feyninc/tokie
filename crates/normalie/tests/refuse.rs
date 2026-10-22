#[test]
fn an_unknown_type_is_refused() {
    let err = normalie::refuse("Regex").expect_err("unknown");
    assert_eq!(err.type_name, "Regex");
    assert_eq!(err.to_string(), "no normalizer for type: Regex");
}
