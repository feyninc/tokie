//! GPT-2 pieces must match the reference regex, token for token.
//!
//! The pattern is the one tiktoken and tokenizers use:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

use fancy_regex::Regex;
use pretokie::{Gpt2, Span};

const GPT2_REGEX: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn regex() -> Regex {
    Regex::new(GPT2_REGEX).expect("gpt-2 pattern compiles")
}

/// Compare `Gpt2` against `re` on `text`. Both walks must cover the text
/// without gaps, and each piece must be the same bytes.
fn assert_matches(re: &Regex, text: &str) {
    let mut walk = Gpt2::walk(text);
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
            "regex left a gap before piece {idx} in {text:?}"
        );
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            assert!(
                filled > 0,
                "gpt2 ended at piece {idx}, regex still has {:?}",
                found.as_str()
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
            "piece {idx} differs in {text:?}"
        );
        covered = found.end();
        idx += 1;
    }

    if next < filled {
        let span = buf[next];
        panic!(
            "gpt2 has an extra piece at index {idx}: {:?}",
            &text[span.start as usize..span.end as usize]
        );
    }
    let extra = walk.fill(&mut buf);
    assert_eq!(extra, 0, "gpt2 has an extra piece at index {idx}");
    assert_eq!(covered, text.len(), "regex did not cover {text:?}");
    assert_eq!(Gpt2::count(text), idx);
}

#[test]
fn hand_cases_match_the_regex() {
    let re = regex();
    let cases = [
        "",
        " ",
        "\n",
        "\t",
        "Hello world",
        "Hello world! It's a test.",
        "don't",
        "It's",
        "'s",
        "'t",
        "'m",
        "'d",
        "'ll",
        "'ve",
        "'re",
        "'S",
        "'hello",
        "'l",
        " 's",
        "  's",
        "\thello",
        "\t\thello",
        "  hello",
        "   hello",
        "hello\n\n",
        "hello  ",
        "a  b",
        "a\tb",
        "hello\n world",
        "hello\nworld",
        " \thello",
        "a!!b",
        "you?!?",
        "123 456",
        "a !",
        "123abc",
        "abc123",
        "café",
        "中文",
        "a\u{00A0}b",
        "e\u{0301}",
        " \u{00A0}café\n\t123",
    ];
    for text in cases {
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
        let ch = match pick % 10 {
            0 => ' ',
            1 => '\n',
            2 => '\t',
            3 => '\'',
            4 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            5 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            6 => char::from_u32(0x00C0 + (pick % 64)).unwrap_or('é'),
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    assert_matches(&re, &text);
}
