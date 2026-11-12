fn apply(text: &str, at_start: bool) -> String {
    let mut out = String::new();
    if normalie::Metaspace::write(text, at_start, &mut out) {
        out
    } else {
        text.to_string()
    }
}

#[test]
fn metaspace_always_prepends_unless_a_space_or_mark_leads() {
    assert_eq!(apply("aa bb cc", true), "▁aa▁bb▁cc");
    assert_eq!(apply(" aa bb", true), "▁aa▁bb");
    assert_eq!(apply("aa\tbb  cc", true), "▁aa\tbb▁▁cc");
    assert_eq!(apply("▁hello", true), "▁hello");
    assert_eq!(apply("", true), "");
}

#[test]
fn a_later_segment_still_gets_the_leading_mark() {
    assert_eq!(apply("hello", false), "▁hello");
}

#[test]
fn both_flags_agree_because_always_ignores_the_start() {
    // Expected strings are the official tokenizers 0.23.3 `Metaspace`
    // (`replacement="▁"`, `prepend_scheme="always"`, `split=False`) output.
    let cases = [
        ("hello", "▁hello"),
        ("a hello", "▁a▁hello"),
        ("aa bb cc", "▁aa▁bb▁cc"),
        (" hello", "▁hello"),
        ("▁hello", "▁hello"),
        ("aa\tbb  cc", "▁aa\tbb▁▁cc"),
        ("\thello", "▁\thello"),
        ("hello\nworld", "▁hello\nworld"),
        ("a  b", "▁a▁▁b"),
    ];
    for (text, expected) in cases {
        assert_eq!(apply(text, true), expected, "at_start=true for {text:?}");
        assert_eq!(apply(text, false), expected, "at_start=false for {text:?}");
    }
}

#[test]
fn nothing_is_written_for_empty_or_already_marked_input() {
    for at_start in [true, false] {
        for text in ["", "▁", "▁hello"] {
            let mut out = String::new();
            assert!(
                !normalie::Metaspace::write(text, at_start, &mut out),
                "{text:?} at_start={at_start} should report no change"
            );
            assert!(out.is_empty(), "{text:?} at_start={at_start} wrote {out:?}");
        }
    }
}

#[test]
fn long_multiwindow_text_prepends_once_and_replaces_every_space() {
    let x = "x".repeat(70);
    let y = "y".repeat(70);
    let z = "z".repeat(70);
    let text = format!("{x} {y} {z}");
    let expected = format!("▁{x}▁{y}▁{z}");
    for at_start in [true, false] {
        let got = apply(&text, at_start);
        assert_eq!(got, expected, "at_start={at_start}");
        assert_eq!(got.matches('▁').count(), 3, "one mark per segment");
    }
}
