//! SmolLM pieces must match a digits split followed by the GPT-2 pattern.
//!
//! Each numeric character is removed first. The GPT-2 pattern then runs on
//! every remaining stretch:
//! `'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

use fancy_regex::Regex;
use pretokie::{SmolLM, Span};

const GPT2_REGEX: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

fn regex() -> Regex {
    Regex::new(GPT2_REGEX).expect("gpt-2 pattern compiles")
}

/// Pieces produced by taking each numeric character out, then matching the
/// GPT-2 pattern on the stretches in between.
fn reference<'a>(re: &Regex, text: &'a str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let ch = rest.chars().next().expect("rest is not empty");
        if ch.is_numeric() {
            let n = ch.len_utf8();
            out.push(&rest[..n]);
            rest = &rest[n..];
            continue;
        }
        let mut end = 0;
        for c in rest.chars() {
            if c.is_numeric() {
                break;
            }
            end += c.len_utf8();
        }
        let chunk = &rest[..end];
        let mut covered = 0;
        for found in re.find_iter(chunk) {
            let found = found.expect("regex match");
            assert_eq!(found.start(), covered, "regex left a gap in {chunk:?}");
            out.push(found.as_str());
            covered = found.end();
        }
        assert_eq!(covered, chunk.len(), "regex did not cover {chunk:?}");
        rest = &rest[end..];
    }
    out
}

fn assert_matches(re: &Regex, text: &str) {
    let expected = reference(re, text);
    let mut walk = SmolLM::walk(text);
    let mut buf = [Span { start: 0, end: 0 }; 32];
    let mut filled = 0;
    let mut next = 0;
    let mut covered = 0usize;

    for (idx, piece) in expected.iter().enumerate() {
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            assert!(filled > 0, "smollm ended at piece {idx}, regex still has {piece:?}");
        }
        let span = buf[next];
        next += 1;
        let start = span.start as usize;
        let end = span.end as usize;
        let got = &text[start..end];
        assert_eq!((start, got), (covered, *piece), "piece {idx} differs in {text:?}");
        covered = end;
    }

    if next < filled {
        let span = buf[next];
        panic!(
            "smollm has an extra piece at the end: {:?}",
            &text[span.start as usize..span.end as usize]
        );
    }
    let extra = walk.fill(&mut buf);
    assert_eq!(extra, 0, "smollm has an extra piece");
    assert_eq!(covered, text.len(), "pieces did not cover {text:?}");
    assert_eq!(SmolLM::count(text), expected.len());
}

#[test]
fn hand_cases_match_the_digits_split() {
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
        "12345",
        "test 123",
        "a 1 b",
        "abc\n123",
        "a\n\nb",
        "a  b",
        "a\n\n½ cup",
        "x  ½",
        "  9",
        "abc123",
        "123abc",
        "①②",
        "aⅣb",
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
fn every_bmp_character_matches_the_digits_split() {
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
fn generated_text_matches_the_digits_split() {
    let re = regex();
    let mut state = 0x1234_5678_u32;
    let mut text = String::new();
    for _ in 0..50_000 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let pick = state >> 24;
        let ch = match pick % 12 {
            0 => ' ',
            1 => '\n',
            2 => '\t',
            3 => '\'',
            4 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            5 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            6 => char::from_u32(0x00C0 + (pick % 64)).unwrap_or('é'),
            7 => '½',
            8 => '①',
            9 => char::from_u32('0' as u32 + (pick % 10)).unwrap(),
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    assert_matches(&re, &text);
}
