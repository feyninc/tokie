//! DeepSeek pieces must match the three reference splits, token for token.
//!
//! Numbers come first, in groups of up to three. A CJK or kana run is next.
//! The main pattern then cuts every remaining piece, and a span it does not
//! match stays as its own piece.

use fancy_regex::Regex;
use pretokie::{DeepSeek, Span};

const NUMBERS: &str = r"\p{N}{1,3}";
const CJK: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
const MAIN: &str = "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+";

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

fn split_isolated<'s>(re: &Regex, text: &'s str) -> Vec<&'s str> {
    let mut pieces = Vec::new();
    let mut last = 0;
    for found in re.find_iter(text) {
        let found = found.expect("regex match");
        if found.start() > last {
            pieces.push(&text[last..found.start()]);
        }
        pieces.push(found.as_str());
        last = found.end();
    }
    if last < text.len() {
        pieces.push(&text[last..]);
    }
    pieces
}

fn reference(res: &[Regex; 3], text: &str) -> Vec<String> {
    let mut pieces = vec![text.to_string()];
    for re in res {
        pieces = pieces
            .iter()
            .flat_map(|piece| split_isolated(re, piece))
            .map(str::to_string)
            .collect();
    }
    pieces
}

fn regexes() -> [Regex; 3] {
    [
        Regex::new(NUMBERS).expect("number pattern"),
        Regex::new(CJK).expect("cjk pattern"),
        Regex::new(MAIN).expect("main pattern"),
    ]
}

fn assert_matches(res: &[Regex; 3], text: &str) {
    let expected = reference(res, text);
    let mut walk = DeepSeek::walk(text);
    let mut buf = [Span { start: 0, end: 0 }; 32];
    let mut filled = 0;
    let mut next = 0;
    let mut covered = 0usize;
    for (idx, piece) in expected.iter().enumerate() {
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            assert!(
                filled > 0,
                "ended at piece {idx}, regex still has {piece:?} near {}",
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
            (covered, piece.as_str()),
            "piece {idx} differs near {}",
            snippet(text, start)
        );
        covered = end;
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
    assert_eq!(extra, 0, "extra piece near {}", snippet(text, covered));
    assert_eq!(covered, text.len(), "regex did not cover {}", snippet(text, covered));
    assert_eq!(DeepSeek::count(text), expected.len());
}

#[test]
fn hand_cases_match_the_regex() {
    let res = regexes();
    for text in [
        "",
        " ",
        "hello",
        " hello",
        "hello world",
        "  hello",
        "\thello",
        "\nhello",
        "\n\n   hello",
        "!hello",
        "!!hello",
        "?!x",
        "don't",
        "DON'T",
        "they'LL go",
        "'sound",
        "'sé",
        "'\u{017F}",
        "x'0",
        "123",
        "1234",
        "12345678",
        " 123",
        "  123",
        "a  1",
        "3rd",
        "3.14159",
        "abc123def",
        "abc 123 def",
        "١٢٣٤٥ ٦٧",
        "1٢3x",
        "hello, world!",
        "hi!\n\ndef",
        " !!!",
        " !word",
        "a-b",
        "...",
        "hello\n",
        "hello \nx",
        "hello  \n\n  ",
        "x \n\n ",
        "x  ",
        "  \n  hello",
        "\r\nhello",
        "a\r\n ",
        "a\n \n",
        "\n\n",
        "   ",
        "café",
        " café",
        "cafe\u{301} de\u{301}composed",
        "\u{301}leading mark",
        " \u{301}abc",
        "e\u{301}f",
        "\u{a0}word",
        "voilà ¡hola!",
        "日本語テスト",
        " 日本語",
        " 日本語 abc",
        "漢字test",
        "x字y",
        "あ゛が",
        "\u{3040}あ",
        "ん・カナ",
        "ガード",
        "カタカナ゠ヷヸヹ",
        "一二三１２３",
        "＄￥",
        "a\u{0}b",
        "a\u{0}\u{0}b",
        "\u{0}",
        " \u{0}",
        "\u{ad}word",
        "a\u{200b}b",
        "tab\tsep\tvals",
        "\u{0b}word",
        "a\u{2028}b",
        "price: $5.99!",
        "@user #tag",
        "<|endoftext|>",
        "https://example.com/path?q=1",
        &" ".repeat(64),
        &" ".repeat(70),
    ] {
        assert_matches(&res, text);
    }
}

#[test]
fn every_bmp_character_matches_the_regex() {
    let res = regexes();
    let mut all = String::new();
    for cp in 0..=0xFFFF {
        let Some(ch) = char::from_u32(cp) else {
            continue;
        };
        let alone = ch.to_string();
        assert_matches(&res, &alone);
        all.push(ch);
    }
    assert_matches(&res, &all);
}

#[test]
fn a_final_window_matches_the_regex() {
    let res = regexes();
    let tail = "Justice.\n\nSessions and his aides";
    let pad = 64 - (tail.len() % 64);
    let text = format!("{}{tail}", "a".repeat(pad % 64));
    assert_eq!(text.len() % 64, 0);
    assert_matches(&res, &text);
}

#[test]
fn generated_text_matches_the_regex() {
    let res = regexes();
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
            4 => '/',
            5 => char::from_u32(0x00A0 + (pick % 16)).unwrap(),
            6 => char::from_u32(0x4E00 + (pick % 80)).unwrap(),
            7 => char::from_u32(0x3040 + (pick % 96)).unwrap(),
            8 => char::from_u32(0x30A0 + (pick % 96)).unwrap(),
            9 => '½',
            10 => '”',
            11 => char::from_u32('0' as u32 + (pick % 10)).unwrap(),
            12 => char::from_u32(0x0300 + (pick % 16)).unwrap_or('\u{0301}'),
            13 => '\u{0}',
            14 => 'あ',
            _ => char::from_u32(0x20 + (pick % 95)).unwrap(),
        };
        text.push(ch);
    }
    assert_matches(&res, &text);
}

const ASCII_PROBES: &[&str] = &[
    "hello",
    "!hello",
    "!!hello",
    "don't",
    "DON'T",
    "1234",
    "  1",
    "a  1",
    "  hello",
    "\thello",
    "a\n\nb",
    "hello\n\nworld",
    ".\n//",
    "x 42",
    "a.b",
    "a..b",
    "?\n\t b",
    "   ",
    "\0\0b",
    "a\0b",
];

#[test]
fn fuzzed_ascii_windows_match_the_regex() {
    let res = regexes();
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
                assert_matches(&res, &text);
            }
        }
    }
    let hot: &[u8] = b"aA0 \n\t\r'!.$?/";
    let printable: &[u8] =
        b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\r\n'\"!$?.,;:-_/\\()\0";
    for (seed, alphabet, n) in [(0x1234_5678_u32, hot, 30_000), (0x0F1E_2D3C, printable, 30_000)] {
        let mut text = String::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            text.push(alphabet[(state >> 24) as usize % alphabet.len()] as char);
        }
        assert_matches(&res, &text);
    }
}
