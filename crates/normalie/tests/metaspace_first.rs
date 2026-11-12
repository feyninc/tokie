fn apply(text: &str, at_start: bool) -> String {
    let mut out = String::new();
    if normalie::MetaspaceFirst::write(text, at_start, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn the_leading_mark_is_only_at_the_start() {
    assert_eq!(apply("aa bb cc", true), "▁aa▁bb▁cc");
    assert_eq!(apply("aa bb cc", false), "aa▁bb▁cc");
    assert_eq!(apply(" aa bb", false), "▁aa▁bb");
}

#[test]
fn a_later_segment_has_no_leading_mark() {
    assert_eq!(apply("hello", false), "hello");
}

#[test]
fn the_start_flag_is_the_only_difference() {
    // At the start these match the official tokenizers 0.23.3 `Metaspace`
    // (`prepend_scheme="first"`, `split=False`) output. Later segments match
    // the same tokenizer after `PreTokenizedString::split` on spaces.
    let cases = [
        ("hello", "▁hello", "hello"),
        ("a hello", "▁a▁hello", "a▁hello"),
        ("aa bb cc", "▁aa▁bb▁cc", "aa▁bb▁cc"),
        (" hello", "▁hello", "▁hello"),
        ("▁hello", "▁hello", "▁hello"),
        ("aa\tbb  cc", "▁aa\tbb▁▁cc", "aa\tbb▁▁cc"),
        ("\thello", "▁\thello", "\thello"),
        ("hello\nworld", "▁hello\nworld", "hello\nworld"),
        ("a  b", "▁a▁▁b", "a▁▁b"),
    ];
    for (text, at_start_expected, later_expected) in cases {
        assert_eq!(
            apply(text, true),
            at_start_expected,
            "at_start=true for {text:?}"
        );
        assert_eq!(
            apply(text, false),
            later_expected,
            "at_start=false for {text:?}"
        );
    }
}

#[test]
fn nothing_is_written_for_empty_or_already_marked_input() {
    for at_start in [true, false] {
        for text in ["", "▁", "▁hello"] {
            let mut out = String::new();
            assert!(
                !normalie::MetaspaceFirst::write(text, at_start, &mut out),
                "{text:?} at_start={at_start} should report no change"
            );
            assert!(out.is_empty(), "{text:?} at_start={at_start} wrote {out:?}");
        }
    }
}

#[test]
fn long_multiwindow_text_prefixes_only_at_the_start() {
    let x = "x".repeat(70);
    let y = "y".repeat(70);
    let z = "z".repeat(70);
    let text = format!("{x} {y} {z}");
    let start_expected = format!("▁{x}▁{y}▁{z}");
    let later = format!("{x}▁{y}▁{z}");
    assert_eq!(apply(&text, true), start_expected);
    assert_eq!(apply(&text, false), later);
    assert_eq!(apply(&text, true).matches('▁').count(), 3);
    assert_eq!(apply(&text, false).matches('▁').count(), 2);
}
