//! o200k pieces must match the reference regex, token for token.

use fancy_regex::Regex;
use pretokie::{O200k, Span};

const O200K: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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
    let mut walk = O200k::walk(text);
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
                "ended at piece {idx}, regex still has {:?} near {}",
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
            "extra piece at index {idx}: {:?} near {}",
            &text[span.start as usize..span.end as usize],
            snippet(text, span.start as usize)
        );
    }
    let extra = walk.fill(&mut buf);
    assert_eq!(extra, 0, "extra piece at index {idx} near {}", snippet(text, covered));
    assert_eq!(covered, text.len(), "regex did not cover {}", snippet(text, covered));
    assert_eq!(O200k::count(text), idx);
}

#[test]
fn hand_cases_match_the_regex() {
    let re = Regex::new(O200K).expect("o200k pattern");
    for text in [
        "",
        " ",
        "\n",
        "camelCase",
        "HTTPResponse",
        "don't",
        "can'ts",
        "'sound",
        "12345",
        ".\n//",
        "a b",
        "x 42",
        "a\nb",
        "a \nb",
        "Hello world",
        "DON'T",
        "a..b",
        "a.b",
        "\thello",
        "  hello",
        "e\u{0301}",
        "A\u{02B0}B",
        "'\u{017F}",
        "café",
        "中文",
        "  \n  ",
    ] {
        assert_matches(&re, text);
    }
}

#[test]
fn every_bmp_character_matches_the_regex() {
    let re = Regex::new(O200K).expect("o200k pattern");
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
    let re = Regex::new(O200K).expect("o200k pattern");
    let tail = "Justice.\n\nSessions and his aides";
    let pad = 64 - (tail.len() % 64);
    let text = format!("{}{tail}", "a".repeat(pad % 64));
    assert_eq!(text.len() % 64, 0);
    assert_matches(&re, &text);
}

#[test]
fn a_trailing_run_at_the_final_window_stays_whole() {
    let re = Regex::new(O200K).expect("o200k pattern");
    let tail = "end\n\t   ";
    let pad = (64 - (tail.len() % 64)) % 64;
    let text = format!("{}{tail}", "a".repeat(pad));
    assert_eq!(text.len() % 64, 0);
    assert!(text.ends_with(' '));
    assert_matches(&re, &text);
}

#[test]
fn a_space_before_the_next_window_still_splits() {
    let re = Regex::new(O200K).expect("o200k pattern");
    let text = format!("{} b", "a".repeat(63));
    assert_eq!(text.len(), 65);
    assert_matches(&re, &text);
}

#[test]
fn generated_text_matches_the_regex() {
    let re = Regex::new(O200K).expect("o200k pattern");
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
            4 => '/',
            5 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            6 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            7 => char::from_u32(0x00C0 + (pick % 64)).unwrap_or('é'),
            8 => '½',
            9 => '”',
            10 => char::from_u32('0' as u32 + (pick % 10)).unwrap(),
            11 => char::from_u32(0x0300 + (pick % 16)).unwrap_or('\u{0301}'),
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    assert_matches(&re, &text);
}

const ASCII_PROBES: &[&str] = &[
    "camelCase",
    "HTTPResponse",
    "don't",
    "can'ts",
    "'sound",
    "DON'T",
    "a'S",
    "x'll'd",
    ".\n//",
    "./\n/",
    "12345",
    "x 42",
    "a\n\nb",
    "a \nb",
    "!!\n\nb",
    "?\n\t b",
    "a  b",
    "\thello",
    "  hello",
    "a.b",
    "a..b",
    "hello\n\nworld",
];

#[test]
fn fuzzed_ascii_windows_match_the_regex() {
    let re = Regex::new(O200K).expect("o200k pattern");
    let suffix = " word".repeat(12);
    for pad in ['a', 'A', '.'] {
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
