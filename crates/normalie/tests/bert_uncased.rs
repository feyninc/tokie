fn apply(text: &str) -> String {
    let mut out = String::new();
    if normalie::BertUncased::write(text, true, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn bert_uncased_is_one_pass() {
    assert_eq!(apply("Hello World"), "hello world");
    assert_eq!(apply("CAFÉ"), "cafe");
    assert_eq!(apply("a中b"), "a 中 b");
    assert_eq!(apply("你好"), " 你  好 ");
    assert_eq!(apply("Ä中"), "a 中 ");
    assert_eq!(apply("กข"), "กข");
    assert_eq!(apply("ก\u{0e31}"), "ก");
    assert_eq!(apply("が"), "か");
    assert_eq!(apply("パヴゞ"), "ハウゝ");
    assert_eq!(apply("\u{3099}"), "");
    assert_eq!(apply("ア"), "ア");
    assert_eq!(apply("・"), "・");
    assert_eq!(apply("Hello\nWorld"), "hello world");
    assert_eq!(apply("\n"), " ");
    assert_eq!(apply("\u{7f}"), "");
    assert_eq!(apply("\u{378}"), "\u{378}");
    // U+F900 decomposes to U+8C48. The spaces are the CJK padding.
    assert_eq!(apply("\u{F900}"), " \u{8C48} ");
    assert_eq!(apply("\u{2F800}"), " \u{4E3D} ");
    // Extension E starts at U+2B820. The old range began at U+2B920.
    assert_eq!(apply("\u{2B820}"), " \u{2B820} ");
    assert_eq!(apply("\u{2B91F}"), " \u{2B91F} ");
    assert_eq!(apply("\u{2CEB0}"), "\u{2CEB0}");
    assert_eq!(apply("\u{3040}"), "\u{3040}");
    assert_eq!(apply("\u{3097}"), "\u{3097}");
    assert_eq!(apply("\u{3098}"), "\u{3098}");
    let mut cased = String::new();
    assert!(normalie::BertCased::write("\u{F900}", true, &mut cased));
    assert_eq!(cased, " \u{F900} ");
}

#[test]
fn bert_uncased_keeps_unassigned_kana_code_points() {
    // U+3040, U+3097, and U+3098 are unassigned. NFD and lowercase leave them
    // alone, so uncased BERT keeps them. Only U+3099 and U+309A are nonspacing
    // marks in the kana blocks, and only those are dropped.
    assert_eq!(apply("\u{3040}"), "\u{3040}");
    assert_eq!(apply("\u{3097}"), "\u{3097}");
    assert_eq!(apply("\u{3098}"), "\u{3098}");
    assert_eq!(apply("\u{3099}"), "");
    assert_eq!(apply("\u{309A}"), "");
    // The fast kana path must not swallow them inside a run either.
    assert_eq!(apply("あ\u{3040}い\u{3097}う\u{3098}"), "あ\u{3040}い\u{3097}う\u{3098}");
}
