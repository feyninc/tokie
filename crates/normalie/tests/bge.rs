#[test]
fn collapse_then_metaspace_keeps_a_trailing_space_as_a_mark() {
    let mut scratch = normalie::Scratch::new();
    let out = scratch.apply3(
        "aa  bb ",
        true,
        |s, at, buf| {
            let _ = (s, at, buf);
            false
        },
        |s, at, buf| normalie::CollapseSpaces::write(s, at, buf),
        |s, at, buf| normalie::Metaspace::write(s, at, buf),
    );
    assert_eq!(out, "▁aa▁bb▁");
}
