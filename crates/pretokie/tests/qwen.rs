//! Qwen pieces must match the reference regex, token for token.
//!
//! Qwen3.5 is Voyage with a mark counted as a letter.
//! `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

use fancy_regex::Regex;
use pretokie::{Qwen, Span};

const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

fn regex() -> Regex {
    Regex::new(PATTERN).expect("qwen pattern compiles")
}

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

fn assert_matches(re: &Regex, text: &str) {
    let mut walk = Qwen::walk(text);
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
            "regex left a gap before piece {idx} near {}",
            snippet(text, found.start())
        );
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            assert!(
                filled > 0,
                "qwen ended at piece {idx}, regex still has {:?} near {}",
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
            "piece {idx} differs near {}",
            snippet(text, start)
        );
        covered = found.end();
        idx += 1;
    }

    if next < filled {
        let span = buf[next];
        panic!(
            "qwen has an extra piece at index {idx}: {:?} near {}",
            &text[span.start as usize..span.end as usize],
            snippet(text, span.start as usize)
        );
    }
    let extra = walk.fill(&mut buf);
    assert_eq!(
        extra,
        0,
        "qwen has an extra piece at index {idx} near {}",
        snippet(text, covered)
    );
    assert_eq!(covered, text.len(), "regex did not cover {}", snippet(text, covered));
    assert_eq!(Qwen::count(text), idx);
}

#[test]
fn hand_cases_match_the_regex() {
    let re = regex();
    for text in [
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
    ] {
        assert_matches(&re, text);
    }
}

#[test]
fn every_bmp_character_matches_the_regex() {
    let re = regex();
    let mut all = String::new();
    for cp in 0..=0xFFFF {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let alone = ch.to_string();
        assert_matches(&re, &alone);
        all.push(ch);
    }
    assert_matches(&re, &all);
}

#[test]
fn generated_text_matches_the_regex() {
    let re = regex();
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
    assert_matches(&re, &text);
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

#[test]
fn fuzzed_ascii_windows_match_the_regex() {
    let re = regex();
    let suffix = " word".repeat(12);
    for pad in ['a', '.'] {
        for probe in ASCII_PROBES {
            for shift in 0..96 {
                let mut text = String::new();
                text.extend(std::iter::repeat(pad).take(shift));
                text.push_str(probe);
                text.push_str(" end");
                text.push_str(&suffix);
                assert!(text.len() >= 64, "probe {probe:?} never reaches the window");
                assert_matches(&re, &text);
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
        assert_matches(&re, &text);
    }
}
