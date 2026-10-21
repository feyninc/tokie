//! Voyage, Olmo, and Qwen pieces must match their reference regex.

use fancy_regex::Regex;
use pretokie::{Olmo, Qwen, Span, Voyage};

const VOYAGE: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const OLMO: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn snippet(text: &str, at: usize) -> String {
    let at = at.min(text.len());
    let mut start = at.saturating_sub(32);
    while start > 0 && !text.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = (at + 48).min(text.len());
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    format!("{at}: {:?}", &text[start..end])
}

fn assert_matches(
    re: &Regex,
    text: &str,
    name: &str,
    mut fill: impl FnMut(&mut [Span]) -> usize,
    pieces: usize,
) {
    let mut buf = [Span { start: 0, end: 0 }; 32];
    let mut filled = 0;
    let mut next = 0;
    let mut idx = 0usize;
    let mut covered = 0usize;

    for found in re.find_iter(text) {
        let found = found.expect("regex match");
        assert_eq!(
            found.start(),
            covered,
            "{name} regex left a gap before piece {idx} near {}",
            snippet(text, found.start())
        );
        if next == filled {
            filled = fill(&mut buf);
            next = 0;
            assert!(
                filled > 0,
                "{name} ended at piece {idx}, regex still has {:?} near {}",
                found.as_str(),
                snippet(text, covered)
            );
        }
        let span = buf[next];
        next += 1;
        let start = span.start as usize;
        let end = span.end as usize;
        let got = &text[start..end];
        assert_eq!(
            (start, got),
            (found.start(), found.as_str()),
            "{name} piece {idx} differs near {}",
            snippet(text, start)
        );
        covered = found.end();
        idx += 1;
    }

    if next < filled {
        let span = buf[next];
        panic!(
            "{name} has an extra piece at index {idx}: {:?} near {}",
            &text[span.start as usize..span.end as usize],
            snippet(text, span.start as usize)
        );
    }
    let extra = fill(&mut buf);
    assert_eq!(
        extra, 0,
        "{name} has an extra piece at index {idx} near {}",
        snippet(text, covered)
    );
    assert_eq!(covered, text.len(), "{name} regex did not cover {}", snippet(text, covered));
    assert_eq!(pieces, idx, "{name} count");
}

fn hand_cases() -> &'static [&'static str] {
    &[
        "",
        " ",
        "\n",
        "\t",
        "Hello world",
        "12345",
        "DON'T",
        "don't",
        "don'ts",
        "$hello",
        "a b",
        "a\nb",
        "a \nb",
        "a\n\nb",
        "a  b",
        "a   b",
        "x 42",
        "a!!b",
        "  hello",
        "\thello",
        "\t\thello",
        "a”\n\nb",
        "1¹23",
        "café",
        "中文",
        "a\u{00A0}b",
        "e\u{0301}",
        "!!\u{0301}",
        " \u{00A0}café\n\t123",
        "'hello",
        "O'Toole",
        " 'they",
        "!'they",
        "  't",
        "'Stress",
        "'reading",
        "  \n  ",
    ]
}

const ASCII_PROBES: &[&str] = &[
    "'s", "'S", "'t", "'T", "'d", "'D", "'m", "'M", "'re", "'RE", "'ve", "'VE", "'ll", "'LL",
    "'stress", "'STRESS", "'reading", "'llgo", "'vemore", "don'ts", "DON'T", "a'Stress",
    " 'they", "  'they", "!'they", "$$'sold", "\n'Stress", "\t'llgo", "\r\n'reading",
    "12345", "1234567890", "x 42", "x  42", "123't", "a\n\nb", "a \nb", "a\r\nb", "!!\n\nb",
    "?\n\t\r", "!\n \n", "?\n\t b",
    "a  b", "a   b", "$hello", "$$hello", "\thello", "  hello", "hello\n\nworld", " \n ",
    "''t", "a'", "' ",
];

fn generated() -> String {
    let mut state = 0x1234_5678_u32;
    let mut text = String::new();
    for _ in 0..50_000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let pick = state >> 24;
        let ch = match pick % 13 {
            0 => ' ',
            1 => '\n',
            2 => '\t',
            3 => '\'',
            4 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            5 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            6 => char::from_u32(0x00C0 + (pick % 64)).unwrap_or('é'),
            7 => '½',
            8 => '”',
            9 => char::from_u32('0' as u32 + (pick % 10)).unwrap(),
            10 => char::from_u32(0x0300 + (pick % 16)).unwrap_or('\u{0301}'),
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    text
}

fn fuzz(re: &Regex, name: &str, mut check: impl FnMut(&Regex, &str)) {
    let suffix = " word".repeat(12);
    for pad in ['a', '.'] {
        for probe in ASCII_PROBES {
            for shift in 0..96 {
                let mut text = String::new();
                text.extend(std::iter::repeat(pad).take(shift));
                text.push_str(probe);
                text.push_str(" end");
                text.push_str(&suffix);
                assert!(text.len() >= 64, "{name} probe {probe:?} never reaches the window");
                check(re, &text);
            }
        }
    }
    let hot: &[u8] = b"aA0 \n\t\r'!.$?";
    let printable: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\r\n'\"!$?.,;:-_/\\()";
    for (seed, alphabet, n) in [(0x1234_5678_u32, hot, 30_000), (0x0F1E_2D3C, printable, 30_000)] {
        let mut text = String::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            text.push(alphabet[(state >> 24) as usize % alphabet.len()] as char);
        }
        check(re, &text);
    }
}

fn bmp(re: &Regex, mut check: impl FnMut(&Regex, &str)) {
    let mut all = String::new();
    for cp in 0..=0xFFFF {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let alone = ch.to_string();
        check(re, &alone);
        all.push(ch);
    }
    check(re, &all);
}

macro_rules! sibling {
    ($hand:ident, $bmp:ident, $gen:ident, $fuzz:ident, $ty:ty, $pattern:expr, $label:literal) => {
        #[test]
        fn $hand() {
            let re = Regex::new($pattern).expect($label);
            for text in hand_cases() {
                let mut walk = <$ty>::walk(text);
                assert_matches(&re, text, $label, |buf| walk.fill(buf), <$ty>::count(text));
            }
        }

        #[test]
        fn $bmp() {
            let re = Regex::new($pattern).expect($label);
            bmp(&re, |re, text| {
                let mut walk = <$ty>::walk(text);
                assert_matches(re, text, $label, |buf| walk.fill(buf), <$ty>::count(text));
            });
        }

        #[test]
        fn $gen() {
            let re = Regex::new($pattern).expect($label);
            let text = generated();
            let mut walk = <$ty>::walk(&text);
            assert_matches(&re, &text, $label, |buf| walk.fill(buf), <$ty>::count(&text));
        }

        #[test]
        fn $fuzz() {
            let re = Regex::new($pattern).expect($label);
            fuzz(&re, $label, |re, text| {
                let mut walk = <$ty>::walk(text);
                assert_matches(re, text, $label, |buf| walk.fill(buf), <$ty>::count(text));
            });
        }
    };
}

sibling!(
    voyage_hand_cases_match_the_regex,
    voyage_every_bmp_character_matches_the_regex,
    voyage_generated_text_matches_the_regex,
    voyage_fuzzed_ascii_windows_match_the_regex,
    Voyage,
    VOYAGE,
    "voyage"
);
sibling!(
    olmo_hand_cases_match_the_regex,
    olmo_every_bmp_character_matches_the_regex,
    olmo_generated_text_matches_the_regex,
    olmo_fuzzed_ascii_windows_match_the_regex,
    Olmo,
    OLMO,
    "olmo"
);
sibling!(
    qwen_hand_cases_match_the_regex,
    qwen_every_bmp_character_matches_the_regex,
    qwen_generated_text_matches_the_regex,
    qwen_fuzzed_ascii_windows_match_the_regex,
    Qwen,
    QWEN,
    "qwen"
);
