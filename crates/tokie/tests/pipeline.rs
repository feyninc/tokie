use pretokie::Span;
use tokie::bpe::{Merge, merge};
use tokie::cache::{pack, Cache};
use tokie::cut::Cut;
use tokie::norm::{apply, split, Norm};
use tokie::unigram::{unigram, Piece};
use tokie::wordpiece::wordpiece;

#[test]
fn gpt2_splits_hello_world() {
    let mut spans = Vec::new();
    split(Cut::Gpt2, "Hello world", &mut spans);
    let text = "Hello world";
    let pieces: Vec<&str> = spans
        .iter()
        .map(|span| &text[span.start as usize..span.end as usize])
        .collect();
    assert_eq!(pieces, ["Hello", " world"]);
}

#[test]
fn ascii_nfc_keeps_the_original() {
    let mut a = String::new();
    let mut b = String::new();
    let out = apply(&[Norm::Nfc], "Hello", &mut a, &mut b);
    assert_eq!(out, "Hello");
    assert!(a.is_empty());
}

#[test]
fn bert_uncased_rewrites() {
    let mut a = String::new();
    let mut b = String::new();
    let out = apply(&[Norm::BertUncased], "Hello World", &mut a, &mut b);
    assert_eq!(out, "hello world");
}

#[test]
fn bert_pieces_drop_whitespace() {
    let mut a = String::new();
    let mut b = String::new();
    let text = apply(&[Norm::BertUncased], "Hello, World!", &mut a, &mut b);
    let mut spans = Vec::new();
    split(Cut::Bert, text, &mut spans);
    let pieces: Vec<&str> = spans
        .iter()
        .map(|Span { start, end }| &text[*start as usize..*end as usize])
        .collect();
    assert_eq!(pieces, ["hello", ",", "world", "!"]);
}

#[test]
fn cache_roundtrip_for_short_and_long_pieces() {
    let mut cache = Cache::new();
    let mut out = Vec::new();
    assert!(!cache.lookup(b"Hello", &mut out));
    cache.insert(b"Hello", &[1, 2]);
    assert!(cache.lookup(b"Hello", &mut out));
    assert_eq!(out, [1, 2]);

    let long = b"this piece is longer than fifteen bytes";
    assert!(pack(long).is_none());
    cache.insert(long, &[9, 8, 7, 6, 5]);
    out.clear();
    assert!(cache.lookup(long, &mut out));
    assert_eq!(out, [9, 8, 7, 6, 5]);
}

#[test]
fn bpe_merges_the_lowest_rank_first() {
    let merges = [
        Merge {
            left: u32::from(b'a'),
            right: u32::from(b'b'),
            id: 256,
            rank: 1,
        },
        Merge {
            left: 256,
            right: u32::from(b'c'),
            id: 257,
            rank: 0,
        },
    ];
    let mut out = Vec::new();
    merge(b"abc", &merges, &mut out);
    assert_eq!(out, [257]);
}

#[test]
fn wordpiece_takes_the_longest_prefix() {
    let mut out = Vec::new();
    let mut scratch = String::new();
    wordpiece(
        "hello",
        0,
        100,
        "##",
        &mut scratch,
        |piece| match piece {
            "hello" => Some(1),
            "hel" => Some(2),
            "##lo" => Some(3),
            _ => None,
        },
        &mut out,
    );
    assert_eq!(out, [1]);

    out.clear();
    wordpiece(
        "hello",
        99,
        100,
        "##",
        &mut scratch,
        |piece| match piece {
            "hel" => Some(2),
            "##lo" => Some(3),
            _ => None,
        },
        &mut out,
    );
    assert_eq!(out, [2, 3]);

    out.clear();
    wordpiece("xyz", 99, 100, "##", &mut scratch, |_| None, &mut out);
    assert_eq!(out, [99]);
}

#[test]
fn unigram_picks_the_higher_score() {
    let pieces = [
        Piece {
            bytes: b"a",
            id: 1,
            score: -2.0,
        },
        Piece {
            bytes: b"b",
            id: 2,
            score: -2.0,
        },
        Piece {
            bytes: b"ab",
            id: 3,
            score: -0.5,
        },
    ];
    let mut out = Vec::new();
    unigram(b"ab", &pieces, 0, &mut out);
    assert_eq!(out, [3]);
}
