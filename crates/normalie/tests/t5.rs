const T5: &[u8] = include_bytes!("../../../bench/normalize/data/t5.charsmap");

#[test]
fn whitespace_folded_by_the_charsmap_stays_empty() {
    let map = normalie::Precompiled::parse(T5).unwrap();
    let mut scratch = normalie::Scratch::new();
    let out = scratch.apply2(
        "\u{00A0}",
        true,
        |s, at, buf| map.write(s, at, buf),
        |s, at, buf| normalie::MetaspaceWords::write(s, at, buf),
    );
    assert_eq!(out, "");
    let out = scratch.apply2(
        " \t\n",
        true,
        |s, at, buf| map.write(s, at, buf),
        |s, at, buf| normalie::MetaspaceWords::write(s, at, buf),
    );
    assert_eq!(out, "");
    let out = scratch.apply2(
        "Hello world",
        true,
        |s, at, buf| map.write(s, at, buf),
        |s, at, buf| normalie::MetaspaceWords::write(s, at, buf),
    );
    assert_eq!(out, "▁Hello▁world");
}
