//! T5 pieces must match a run of non-whitespace, token for token.
//!
//! Whitespace is dropped. Punctuation stays on the word.

use fancy_regex::Regex;
use pretokie::{Span, T5};

const PATTERN: &str = r"\S+";

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

fn reference(re: &Regex, text: &str) -> Vec<(usize, usize)> {
    re.find_iter(text)
        .map(|found| {
            let found = found.expect("regex match");
            (found.start(), found.end())
        })
        .collect()
}

fn regex() -> Regex {
    Regex::new(PATTERN).expect("t5 pattern")
}

fn assert_matches(re: &Regex, text: &str) {
    let expected = reference(re, text);
    let mut walk = T5::walk(text);
    let mut buf = [Span { start: 0, end: 0 }; 32];
    let mut filled = 0;
    let mut next = 0;
    let mut cursor = 0usize;
    for (idx, &(start, end)) in expected.iter().enumerate() {
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            assert!(
                filled > 0,
                "ended at piece {idx}, regex still has {:?} near {}",
                &text[start..end],
                snippet(text, start)
            );
        }
        let span = buf[next];
        next += 1;
        let got_start = span.start as usize;
        let got_end = span.end as usize;
        assert!(
            text[cursor..got_start].chars().all(char::is_whitespace),
            "gap before piece {idx} is not whitespace near {}",
            snippet(text, cursor)
        );
        assert_eq!(
            (got_start, got_end),
            (start, end),
            "piece {idx} differs near {}",
            snippet(text, got_start)
        );
        cursor = got_end;
    }
    if next < filled {
        let span = buf[next];
        panic!(
            "extra piece at the end: {:?} near {}",
            &text[span.start as usize..span.end as usize],
            snippet(text, span.start as usize)
        );
    }
    let extra = walk.fill(&mut buf);
    assert_eq!(extra, 0, "extra piece near {}", snippet(text, cursor));
    assert!(
        text[cursor..].chars().all(char::is_whitespace),
        "tail is not whitespace near {}",
        snippet(text, cursor)
    );
    assert_eq!(T5::count(text), expected.len());
}

#[test]
fn hand_cases_match_the_regex() {
    let re = regex();
    for text in [
        "",
        " ",
        "   ",
        "\t\n\r",
        "hello",
        "Hello, world!",
        "Hello,  world!",
        "don't",
        "DON'T",
        "18th",
        "...",
        "a-b",
        "a\0b",
        "\0",
        " \0 ",
        "€5",
        "$5",
        "café résumé",
        "e\u{301}f",
        "\u{301}mark",
        "中文",
        "中 文",
        "漢字test",
        "¡hola!",
        "voilà ¡hola!",
        "\u{a0}word",
        "\u{0b}word",
        "a\u{2028}b",
        "a\u{200b}b",
        "\u{ad}word",
        "price: $5.99!",
        "@user #tag",
        "<|endoftext|>",
        "https://example.com/path?q=1",
        "tab\tsep\tvals",
        &" ".repeat(64),
        &" ".repeat(70),
        &"!".repeat(80),
        &"a ".repeat(40),
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
fn a_final_window_matches_the_regex() {
    let re = regex();
    let tail = "Justice.\n\nSessions and his aides";
    let pad = 64 - (tail.len() % 64);
    let text = format!("{}{tail}", "a".repeat(pad % 64));
    assert_eq!(text.len() % 64, 0);
    assert_matches(&re, &text);
}

#[test]
fn generated_text_matches_the_regex() {
    let re = regex();
    let mut state = 0x1234_5678_u32;
    let mut text = String::new();
    for _ in 0..50_000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let pick = state >> 24;
        let ch = match pick % 16 {
            0 => ' ',
            1 => '\n',
            2 => '\t',
            3 => '\'',
            4 => ',',
            5 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            6 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            7 => '’',
            8 => '€',
            9 => '½',
            10 => '”',
            11 => char::from_u32('0' as u32 + (pick % 10)).unwrap(),
            12 => char::from_u32(0x0300 + (pick % 16)).unwrap_or('\u{0301}'),
            13 => '\u{0}',
            14 => '。',
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    assert_matches(&re, &text);
}

const ASCII_PROBES: &[&str] = &[
    "hello",
    "Hello, world!",
    "don't",
    "18th",
    "...",
    "a\0b",
    "$5",
    "  hello",
    "\thello",
    "a\n\nb",
    "hello\n\nworld",
    "?\n\t b",
    "   ",
    "\0\0b",
    "end.",
    "a.b",
    "a..b",
];

#[test]
fn fuzzed_ascii_windows_match_the_regex() {
    let re = regex();
    let suffix = " word".repeat(12);
    for pad in ['a', 'A', '.', ' '] {
        for probe in ASCII_PROBES {
            for shift in 0..96 {
                let mut text = String::new();
                text.extend(std::iter::repeat(pad).take(shift));
                text.push_str(probe);
                text.push_str(" end");
                text.push_str(&suffix);
                assert!(text.len() >= 64);
                assert_matches(&re, &text);
            }
        }
    }
    let hot: &[u8] = b"aA0 \n\t\r'!.$?/";
    let printable: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\r\n'\"!$?.,;:-_/\\()\0";
    for (seed, alphabet, n) in [(0x1234_5678_u32, hot, 30_000), (0x0F1E_2D3C, printable, 30_000)]
    {
        let mut text = String::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            text.push(alphabet[(state >> 24) as usize % alphabet.len()] as char);
        }
        assert_matches(&re, &text);
    }
}
