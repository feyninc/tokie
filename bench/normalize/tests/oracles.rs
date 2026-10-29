//! Each scheme's output, locked to Hugging Face tokenizers v1.
//!
//! Where tokie or gigatoken implement the same scheme, the test requires the
//! same string. Schemes that are a different rule have their own expected value.

use normalize_bench::{metaspace_first_later, Scheme};

fn hf(scheme: Scheme, text: &str) -> String {
    scheme.hf(text).expect("hf").into_owned()
}

fn tokie(scheme: Scheme, text: &str) -> String {
    scheme.tokie(text).expect("tokie").into_owned()
}

fn giga(scheme: Scheme, text: &str) -> String {
    scheme.giga(text).expect("giga").into_owned()
}

fn expect_hf(scheme: Scheme, pairs: &[(&str, &str)]) {
    for (input, expected) in pairs {
        assert_eq!(hf(scheme, input), *expected, "{} {input:?}", scheme.name());
    }
}

fn same(scheme: Scheme, inputs: &[&str]) {
    for input in inputs {
        let hf = hf(scheme, input);
        if scheme.tokie(input).is_some() {
            assert_eq!(tokie(scheme, input), hf, "tokie {} {input:?}", scheme.name());
        }
        if scheme.giga(input).is_some() {
            assert_eq!(giga(scheme, input), hf, "giga {} {input:?}", scheme.name());
        }
        if scheme.normalie(input).is_some() {
            assert_eq!(
                scheme.normalie(input).unwrap(),
                hf,
                "normalie {} {input:?}",
                scheme.name()
            );
        }
    }
}

#[test]
fn unicode_forms() {
    expect_hf(
        Scheme::Nfc,
        &[
            ("", ""),
            ("Hello", "Hello"),
            ("café", "café"),
            ("e\u{301}", "é"),
            ("é", "é"),
        ],
    );
    expect_hf(
        Scheme::Nfd,
        &[
            ("é", "e\u{301}"),
            ("e\u{301}", "e\u{301}"),
            ("Hello", "Hello"),
        ],
    );
    expect_hf(
        Scheme::Nfkc,
        &[("ﬁ", "fi"), ("①", "1"), ("Hello", "Hello")],
    );
    expect_hf(
        Scheme::Nfkd,
        &[("ﬁ", "fi"), ("é", "e\u{301}"), ("Hello", "Hello")],
    );
    same(Scheme::Nfc, &["", "Hello", "café", "e\u{301}", "é", "中文", "a\u{0301}"]);
}

#[test]
fn nmt_removes_controls_and_folds_a_few_code_points_to_space() {
    expect_hf(
        Scheme::Nmt,
        &[
            ("Hello", "Hello"),
            ("\u{0001}a\tb", "a b"),
            ("▁x", " x"),
            ("\u{FEFF}hi", " hi"),
            ("a\u{007F}b", "ab"),
            ("a\u{000B}b", "ab"),
            ("keep\n", "keep "),
        ],
    );
}

#[test]
fn lowercase_is_unicode_casefold_of_one_character() {
    expect_hf(
        Scheme::Lowercase,
        &[
            ("", ""),
            ("HELLO", "hello"),
            ("Straße", "straße"),
            ("café", "café"),
            ("İ", "i\u{0307}"),
            ("中文", "中文"),
        ],
    );
}

#[test]
fn strip_accents_keeps_precomposed_letters() {
    expect_hf(
        Scheme::StripAccents,
        &[
            ("cafe\u{301}", "cafe"),
            ("a\u{030A}\u{0327}", "a"),
            ("café", "café"),
            ("a\u{07FD}b", "a\u{07FD}b"),
            ("a\u{1CF2}b", "ab"),
            ("Hello", "Hello"),
        ],
    );
}

#[test]
fn tokie_strip_accents_decomposes_first() {
    let pairs = [
        ("cafe\u{301}", "cafe"),
        ("café", "cafe"),
        ("CAFÉ", "CAFE"),
        ("Hello", "Hello"),
        ("a\u{030A}\u{0327}", "a"),
    ];
    for (input, expected) in pairs {
        assert_eq!(tokie(Scheme::StripAccentsNfd, input), expected, "{input:?}");
    }
    assert!(Scheme::StripAccentsNfd.hf("café").is_none());
}

#[test]
fn bert_uncased_matches_without_cjk_spacing() {
    let inputs = [
        "",
        "Hello World",
        "Hello\nWorld",
        "CAFÉ",
        "e\u{301}",
        "  spaced  ",
        "null\0here",
        "repl\u{FFFD}char",
        "ctrl\u{0007}char",
        "MiXeD",
        "straße",
    ];
    same(Scheme::BertUncased, &inputs);
    assert_eq!(hf(Scheme::BertUncased, "Hello\tWorld"), "hello world");
    // A string whose only interesting byte is tab, newline, CR, or DEL never
    // enters tokie's rewrite. HF folds the whitespace and drops DEL.
    // U+0378 is unassigned: tokie drops it, HF keeps it.
    assert_eq!(hf(Scheme::BertUncased, "\n"), " ");
    assert_eq!(tokie(Scheme::BertUncased, "\n"), "\n");
    assert_eq!(hf(Scheme::BertUncased, "\u{7f}"), "");
    assert_eq!(tokie(Scheme::BertUncased, "\u{7f}"), "\u{7f}");
    assert_eq!(hf(Scheme::BertUncased, "\u{378}"), "\u{378}");
    assert_eq!(tokie(Scheme::BertUncased, "\u{378}"), "");
    assert_eq!(hf(Scheme::BertUncasedCjk, "a中b"), "a 中 b");
    assert_eq!(hf(Scheme::BertUncasedCjk, "中文"), " 中  文 ");
    assert_ne!(hf(Scheme::BertUncased, "a中b"), hf(Scheme::BertUncasedCjk, "a中b"));
}

#[test]
fn bert_cased_and_clean_text() {
    expect_hf(
        Scheme::BertCasedCjk,
        &[
            ("Hello World", "Hello World"),
            ("Hello\nWorld", "Hello World"),
            ("CAFÉ", "CAFÉ"),
            ("a中b", "a 中 b"),
            ("null\0here", "nullhere"),
        ],
    );
    expect_hf(
        Scheme::CleanText,
        &[
            ("Hello\nWorld", "Hello World"),
            ("Hello\tWorld", "Hello World"),
            ("CAFÉ", "CAFÉ"),
            ("a\u{00A0}b", "a b"),
            ("null\0here", "nullhere"),
            ("a中b", "a中b"),
        ],
    );
    same(
        Scheme::CleanText,
        &["CAFÉ", "a\u{00A0}b", "null\0here", "a中b", ""],
    );
    // tokie's ASCII fast path does not fold a lone tab, newline, or CR.
    // HF clean text does.
    assert_eq!(hf(Scheme::CleanText, "Hello\nWorld"), "Hello World");
    assert_eq!(tokie(Scheme::CleanText, "Hello\nWorld"), "Hello\nWorld");
    assert_eq!(hf(Scheme::CleanText, "Hello\tWorld"), "Hello World");
    assert_eq!(tokie(Scheme::CleanText, "Hello\tWorld"), "Hello\tWorld");
}

#[test]
fn prepend_replace_collapse_strip() {
    expect_hf(
        Scheme::PrependMark,
        &[("", ""), ("Hello", "▁Hello"), (" world", "▁ world")],
    );
    expect_hf(
        Scheme::ReplaceSpace,
        &[("a b", "a▁b"), ("ab", "ab"), (" a ", "▁a▁"), ("", "")],
    );
    expect_hf(
        Scheme::CollapseSpaces,
        &[
            ("a  b", "a b"),
            ("a b", "a b"),
            ("a   b", "a b"),
            ("  ", " "),
            ("a\t\tb", "a\t\tb"),
        ],
    );
    expect_hf(
        Scheme::Strip,
        &[("  a  ", "a"), ("a", "a"), (" a b ", "a b"), ("\n\thi\t\n", "hi"), ("", "")],
    );
    same(Scheme::PrependMark, &["", "Hello", " world", "▁x"]);
    same(Scheme::ReplaceSpace, &["a b", "ab", " a ", ""]);
    same(Scheme::CollapseSpaces, &["a  b", "a b", "a   b", "  "]);
    same(Scheme::Strip, &["  a  ", "a", " a b ", "\n\thi\t\n", ""]);
}

#[test]
fn metaspace_schemes_match_across_the_three() {
    let inputs = ["", "aa bb cc", " aa bb", "aa\tbb  cc", "▁hello", "hello", "  "];
    expect_hf(
        Scheme::MetaspaceAlways,
        &[
            ("aa bb cc", "▁aa▁bb▁cc"),
            (" aa bb", "▁aa▁bb"),
            ("aa\tbb  cc", "▁aa\tbb▁▁cc"),
            ("", ""),
            ("▁hello", "▁hello"),
        ],
    );
    expect_hf(
        Scheme::MetaspaceNever,
        &[("aa bb cc", "aa▁bb▁cc"), (" aa bb", "▁aa▁bb"), ("", "")],
    );
    expect_hf(
        Scheme::MetaspaceFirst,
        &[("aa bb cc", "▁aa▁bb▁cc"), (" aa bb", "▁aa▁bb")],
    );
    same(Scheme::MetaspaceAlways, &inputs);
    same(Scheme::PrependThenReplace, &["Hello world", " world", "▁hello", ""]);
    assert_eq!(hf(Scheme::PrependThenReplace, " world"), "▁▁world");
    assert_eq!(hf(Scheme::PrependThenReplace, "Hello world"), "▁Hello▁world");
    same(Scheme::MetaspaceNever, &inputs);
    same(Scheme::MetaspaceFirst, &inputs);

    for input in inputs {
        let (hf, tokie, giga) = metaspace_first_later(input);
        assert_eq!(tokie, hf, "tokie later {input:?}");
        assert_eq!(giga, hf, "giga later {input:?}");
        assert_eq!(
            Scheme::MetaspaceFirst.normalie_at(input, false).unwrap(),
            hf,
            "normalie later {input:?}"
        );
    }
    assert_eq!(
        meta_drop_hf("aa bb cc"),
        "▁aa▁bb▁cc"
    );
    assert_eq!(meta_drop_hf("aa\tbb  cc"), "▁aa▁bb▁cc");
    assert_eq!(meta_drop_hf(" aa bb"), "▁aa▁bb");
    assert_eq!(meta_drop_hf(""), "");
}

fn meta_drop_hf(text: &str) -> String {
    hf(Scheme::MetaspaceDrop, text)
}

#[test]
fn byte_level_keeps_printable_ascii_and_shifts_the_space() {
    expect_hf(Scheme::ByteLevel, &[("Hello", "Hello"), ("Hi!", "Hi!"), (" ", "\u{0120}")]);
    assert!(Scheme::ByteLevel.tokie("Hello").is_none());
    assert!(Scheme::ByteLevel.giga("Hello").is_none());
}

#[test]
fn precompiled_t5_agrees() {
    let inputs = [
        "",
        "Hello",
        "Hello\r\nworld",
        "café",
        "ﬁ",
        "\u{00A0}",
        "a\u{0001}b",
        "  a  ",
        "中文",
        "e\u{301}",
    ];
    same(Scheme::Precompiled, &inputs);
    let chain: Vec<&str> = inputs
        .into_iter()
        .filter(|s| !matches!(*s, "\u{00A0}" | "" | "  "))
        .collect();
    same(Scheme::T5, &chain);
    // Whitespace that the charsmap folds away: HF emits nothing. tokie still
    // writes the leading ▁.
    assert_eq!(hf(Scheme::T5, "\u{00A0}"), "");
    assert_eq!(tokie(Scheme::T5, "\u{00A0}"), "▁");
    assert_eq!(hf(Scheme::T5, " \t\n"), "");
    assert_eq!(tokie(Scheme::T5, " \t\n"), "▁");
    assert_eq!(Scheme::T5.normalie("\u{00A0}").unwrap(), "");
    assert_eq!(Scheme::T5.normalie(" \t\n").unwrap(), "");
}

#[test]
fn sentencepiece_fusions_are_tokie_only() {
    assert_eq!(
        tokie(Scheme::SentencePiece, "  Hello   world\n"),
        "▁Hello▁world"
    );
    assert_eq!(
        tokie(Scheme::SentencePieceLower, "  Héllo   Wörld\n"),
        "▁hello▁world"
    );
    assert!(Scheme::SentencePiece.hf("Hello").is_none());
    assert!(Scheme::SentencePiece.giga("Hello").is_none());
    assert!(Scheme::SentencePieceLower.hf("Hello").is_none());
}

#[test]
fn shared_schemes_agree_on_every_bmp_scalar() {
    let mut missed = Vec::new();
    for cp in 0..=0xFFFF {
        let Some(ch) = char::from_u32(cp) else { continue };
        let text = ch.to_string();
        for scheme in [
            Scheme::Nfc,
            Scheme::Precompiled,
            Scheme::MetaspaceAlways,
            Scheme::ReplaceSpace,
            Scheme::PrependThenReplace,
        ] {
            let hf = hf(scheme, &text);
            if scheme.tokie(&text).is_some() && tokie(scheme, &text) != hf {
                missed.push((scheme.name(), cp, tokie(scheme, &text), hf.clone()));
            }
            if scheme.giga(&text).is_some() && giga(scheme, &text) != hf {
                missed.push((scheme.name(), cp, giga(scheme, &text), hf));
            }
            if missed.len() >= 12 {
                break;
            }
        }
        if missed.len() >= 12 {
            break;
        }
    }
    assert!(
        missed.is_empty(),
        "first BMP mismatches: {missed:?}"
    );
}

#[test]
fn every_hand_case_is_utf8_and_hf_returns_for_its_schemes() {
    for scheme in Scheme::ALL {
        for case in normalize_bench::HAND {
            if let Some(out) = scheme.hf(&case) {
                assert!(std::str::from_utf8(out.as_bytes()).is_ok());
            }
            if let Some(out) = scheme.tokie(&case) {
                let _ = out.len();
            }
            if let Some(out) = scheme.giga(&case) {
                let _ = out.len();
            }
            if let Some(out) = scheme.normalie(&case) {
                let _ = out.len();
            }
        }
    }
}

#[test]
fn normalie_matches_hf_on_hand_cases() {
    let mut missed = Vec::new();
    for scheme in Scheme::ALL {
        for case in normalize_bench::HAND {
            let Some(hf) = scheme.hf(case) else { continue };
            let Some(ours) = scheme.normalie(case) else { continue };
            if ours != hf {
                missed.push((scheme.name(), *case, ours.into_owned(), hf.into_owned()));
            }
            if missed.len() >= 16 {
                break;
            }
        }
        if missed.len() >= 16 {
            break;
        }
    }
    assert!(missed.is_empty(), "normalie hand mismatches: {missed:?}");
}

#[test]
fn normalie_matches_hf_on_every_bmp_scalar() {
    let mut missed = Vec::new();
    for cp in 0..=0xFFFF {
        let Some(ch) = char::from_u32(cp) else { continue };
        let text = ch.to_string();
        for scheme in Scheme::ALL {
            let Some(hf) = scheme.hf(&text) else { continue };
            let Some(ours) = scheme.normalie(&text) else { continue };
            if ours != hf {
                missed.push((scheme.name(), cp, ours.into_owned(), hf.into_owned()));
            }
            if missed.len() >= 12 {
                break;
            }
        }
        if missed.len() >= 12 {
            break;
        }
    }
    assert!(missed.is_empty(), "normalie BMP mismatches: {missed:?}");
}
