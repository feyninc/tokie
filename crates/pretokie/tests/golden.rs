//! Golden tests: pretokie vs the HuggingFace oracle (feature `oracle`).
//!
//! Three layers, per `docs/pretokie-v2/00-oracle-engine.md`:
//! 1. `onig_semantics` — pin Oniguruma's definitions empirically (the
//!    `\s` question and friends) so every downstream assumption is grounded.
//! 2. Edge-case goldens — adversarial strings per scheme vs the oracle.
//! 3. Data-file goldens — chunked enwik8/OWT vs the oracle (skipped when
//!    `benches/data/` files are absent; CI downloads them).
//!
//! 4. Fuzz differential — short strings (scalar `Core` path) and long,
//!    mostly-ASCII strings (the SIMD `Mask` batch path).
//!
//! Every scheme matches the oracle on all layers; nothing is `#[ignore]`d.
//! The divergence classes fixed to get here are listed in
//! `docs/pretokie-v2/00-oracle-engine.md` §4b.

use pretokie::oracle::{self, Scheme};
use std::path::Path;

// ---------------------------------------------------------------------------
// 1. Oniguruma semantics pins
// ---------------------------------------------------------------------------

mod onig_semantics {
    use onig::Regex;

    /// Unanchored probe: does the pattern find a match anywhere?
    fn finds(pattern: &str, text: &str) -> bool {
        Regex::new(pattern).unwrap().find(text).is_some()
    }

    /// The `\s` question (doc §1.3.1), settled empirically: Oniguruma's
    /// `\s` under Ruby syntax + UTF-8 IS Unicode-aware — it matches
    /// Unicode White_Space, not just ASCII. Our class tables must agree.
    #[test]
    fn backslash_s_is_unicode_aware() {
        for c in [" ", "\t", "\n", "\r", "\x0b", "\x0c"] {
            assert!(finds(r"\s", c), "onig \\s should match ASCII ws {c:?}");
        }
        for (c, name) in [
            ("\u{a0}", "NBSP"),
            ("\u{2003}", "EM SPACE"),
            ("\u{2028}", "LINE SEPARATOR"),
            ("\u{3000}", "IDEOGRAPHIC SPACE"),
        ] {
            assert!(finds(r"\s", c), "onig \\s should match Unicode ws {name}");
            assert!(!finds(r"\S", c), "onig \\S should NOT match {name}");
        }
    }

    /// `\w` is Unicode-aware too (patterns use \p{L} for letters anyway).
    #[test]
    fn backslash_w_is_unicode_aware() {
        assert!(finds(r"\w", "a"));
        assert!(finds(r"\w", "0"));
        assert!(finds(r"\w", "_"));
        assert!(finds(r"\w", "é"));
        assert!(finds(r"\w", "中"));
    }

    /// Unicode categories: onig's own UCD tables.
    #[test]
    fn unicode_categories() {
        assert!(finds(r"\p{L}", "é"));
        assert!(finds(r"\p{L}", "中"));
        assert!(finds(r"\p{N}", "3"));
        assert!(finds(r"\p{N}", "½")); // No
        assert!(finds(r"\p{N}", "Ⅷ")); // Nl
        assert!(finds(r"\p{P}", "—")); // Pd
        assert!(finds(r"\p{S}", "+")); // Sm
        assert!(finds(r"\p{S}", "€")); // Sc
        assert!(finds(r"\p{M}", "\u{301}")); // combining acute
    }

    /// Case-insensitive group `(?i:...)` folding used by cl100k/o200k.
    #[test]
    fn case_insensitive_folding() {
        let re = Regex::new(r"(?i:'s|'t|'re)").unwrap();
        assert!(re.find("don'T").is_some());
        assert!(re.find("don'S").is_some());
        assert!(re.find("don'Ll").is_none()); // 'Ll not in the group
    }

    /// Lookaheads work (why we can't use the plain `regex` crate), and the
    /// greedy backtrack of `\s+(?!\S)` behaves as the patterns assume.
    #[test]
    fn lookaheads() {
        let re = Regex::new(r"\s+(?!\S)").unwrap();
        assert_eq!(re.find("  x"), Some((0, 1))); // backtracks to 1 space
        assert_eq!(re.find("   "), Some((0, 3))); // end-of-string ok
        assert_eq!(re.find("x  "), Some((1, 3)));
        assert_eq!(re.find("x"), None);
        // Zero-width negative lookahead matches at end.
        let re = Regex::new(r"(?!\S)").unwrap();
        assert_eq!(re.find("ab"), Some((2, 2)));
    }
}

// ---------------------------------------------------------------------------
// 2. Edge-case goldens
// ---------------------------------------------------------------------------

/// Adversarial strings shared by all schemes, plus per-scheme extras.
fn edge_cases(scheme: &str) -> Vec<&'static str> {
    let common: Vec<&'static str> = vec![
        "Hello, world!",
        "I'll don't they've we're it's you'd he's I'm won't o'clock",
        "d'Arce O'Brien Nava'i l'Hopital",
        "' 's ''s '''s a's 'll 've 're 'd 'm 't",
        "'s'll 't've",
        " 123 4567 89 0x1F 3.14 1,000,000",
        " leading space and trailing space ",
        "a  b   c    d",
        "abc   ",
        " \n x",
        "x\n \n",
        "a\n\nb",
        " \n\n ",
        "a\tb\t\tc",
        "a\r\nb\r\n",
        "\tand\tspaces   mixed \t \n",
        "emoji: 😒🌍🎉",
        "family: 👩‍👩‍👧‍👦 flags: 🇺🇸🇮🇳 skin: 👍🏽",
        "日本語テスト 한국어 汉字",
        "café résumé naïve straße",
        "Перо и слово,单词与符号",
        "mixed 🚀 Перо 予 done ✅",
        "a\u{a0}b\u{a0}\u{a0}x\u{2003}y\u{2028}z",
        "a\u{3000}b  c",
        "abcd",
        "quote\u{201c}smart\u{201d}quotes\u{2019}and\u{2014}dashes",
        "zero\u{200b}width\u{200b}space",
        "combining\u{301}marks\u{308}",
        "<tag attr='val'>text &amp; more</tag>",
        "supercalifragilisticexpialidocioussupercalifragilistic",
        "9",
        "x  1",
    ];
    let mut cases = common;
    match scheme {
        "cl100k" => cases.extend(vec![
            "1234567 89012345 a1234b",
            "IT'S DON'T O'BILL A'Ll",
            "text...\n\nmore text\n",
            "~abc ~123 ~ ~\nabc",
            "word 'Tis 'S",
        ]),
        "o200k" => cases.extend(vec![
            "JSONParser iPhone XMLHttpRequest parseJSON MacBookPro",
            "path/to/file //usr/bin /slash/run",
            "1234567 89012345",
            "IT'S DON'T O'BILL",
            "XylemPhloem xylemPhloem XYLEM",
        ]),
        "qwen2" => cases.extend(vec![
            "1234567890 a1b2c3",
            "IT'S DON'T O'BILL",
            "text...\n\nmore\n",
            "~abc ~123",
        ]),
        "voyage" => cases.extend(vec![
            "1234567890 a1b2c3",
            "IT'S DON'T O'BILL",
            "text...\n\nmore\n",
            "~abc ~123",
            "code: fn main() { let x = 42; } // ctrl",
        ]),
        "deepseek" => cases.extend(vec![
            "1234567 89012345",
            "var_name $price #hash @mention",
            "文字混合mixedテスト한국어",
            "café ½ Ⅷ Ⅻ",
            "(parenthesized) [bracketed] {braced}",
            "ÉCOLE École",
        ]),
        "smollm" => cases.extend(vec![
            "abc123def456 1a2b3c",
            "1,000,000 tokens",
            "half ½ dozen Ⅷ",
            "2024-01-15T00:00:00Z",
        ]),
        "bert" => cases.extend(vec![
            "Hey friend!     How are you?!?",
            "café €100 ±5 → arrow",
            "multi   spaces\ttabs\nnewlines",
            "100% pure #$%^ stuff",
            "野口里佳 Noguchi Rika",
            "（括弧）「引用」 ⁅x⁆ «a» ‹b›",
            "ctl\u{1}inside\u{7f}word vt\u{b}ff\u{c}end",
        ]),
        "falcon" => cases.extend(vec![
            "1234567 890 12 345a",
            "a1b2c3d4",
            "Hello, world! How are you?!?",
            "x  1 y22 z333",
        ]),
        _ => {}
    }
    cases
}

/// pretokie implementation for an oracle scheme name. Qwen2's pattern is
/// byte-identical to Voyage's, and tokie serves it with `Voyage`
/// (`hf.rs` detection); `pretokie::Qwen` is the Qwen3.5 variant
/// (`[\p{L}\p{M}]+` letter runs), which this pattern does not describe.
fn run_pretokie(scheme: &str, text: &str) -> Vec<Vec<u8>> {
    match scheme {
        "r50k" => pretokie::Gpt2::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "cl100k" => pretokie::Cl100k::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "o200k" => pretokie::O200k::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "qwen2" => pretokie::Voyage::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "voyage" => pretokie::Voyage::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "deepseek" => pretokie::DeepSeek::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "smollm" => pretokie::SmolLM::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "bert" => pretokie::Bert::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        // Falcon-style models are served by the regex fallback in tokie;
        // pretokie has no hand-written falcon scheme. The oracle still
        // exercises it via examples/dump.rs for goldens.
        "falcon" => Vec::new(),
        other => panic!("unknown scheme {other}"),
    }
}

/// Compare one scheme on one input; returns Err with a diff on mismatch.
fn diff_case(scheme: &Scheme, text: &str) -> Result<(), String> {
    let expected = oracle::pretokens(scheme, text);
    let got = run_pretokie(scheme.name, text);
    if expected == got {
        return Ok(());
    }
    // Find first difference for a readable message.
    let i = expected.iter().zip(got.iter()).position(|(a, b)| a != b);
    let show = |v: &[Vec<u8>]| {
        v.iter().map(|b| String::from_utf8_lossy(b).escape_debug().to_string()).collect::<Vec<_>>()
    };
    let (expected, got) = (show(&expected), show(&got));
    let ctx = |v: &[String], i: usize| {
        let lo = i.saturating_sub(2);
        v[lo..(i + 3).min(v.len())].join(", ")
    };
    Err(match i {
        Some(i) => format!(
            "first diff at pretoken {i}:\n  oracle: [{}]\n  pretokie: [{}]",
            ctx(&expected, i),
            ctx(&got, i),
        ),
        None => format!(
            "length mismatch: oracle {} pretokie {}\n  oracle head: {:?}\n  pretokie head: {:?}",
            expected.len(),
            got.len(),
            &expected[..expected.len().min(6)],
            &got[..got.len().min(6)],
        ),
    })
}

macro_rules! scheme_golden {
    ($fn_name:ident, $scheme:expr $(, $attr:meta)?) => {
        #[test]
        $(#[$attr])?
        fn $fn_name() {
            let scheme: &Scheme = $scheme;
            let mut failures = Vec::new();
            for (i, case) in edge_cases(scheme.name).iter().enumerate() {
                if let Err(diff) = diff_case(scheme, case) {
                    failures.push(format!("case {i} {case:?}: {diff}"));
                }
            }
            assert!(failures.is_empty(), "{} of {} edge cases failed:\n{}",
                failures.len(), edge_cases(scheme.name).len(), failures.join("\n"));
        }
    };
}

scheme_golden!(golden_r50k, &oracle::R50K);
scheme_golden!(golden_cl100k, &oracle::CL100K);
scheme_golden!(golden_o200k, &oracle::O200K);
scheme_golden!(golden_qwen2, &oracle::QWEN2);
scheme_golden!(golden_deepseek, &oracle::DEEPSEEK);
scheme_golden!(golden_smollm, &oracle::SMOLLM);
scheme_golden!(golden_bert, &oracle::BERT);
scheme_golden!(golden_voyage, &oracle::VOYAGE);

// ---------------------------------------------------------------------------
// 3. Data-file goldens (chunked enwik8 / OWT)
// ---------------------------------------------------------------------------

const CHUNK: usize = 1 << 20;
/// Cap on bytes per data file per scheme; 0 = unlimited (env override).
fn golden_bytes() -> usize {
    std::env::var("PRETOKIE_GOLDEN_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4 << 20)
}

/// Byte ranges of 1MB chunks backed off to UTF-8 boundaries — the same
/// algorithm as the external harness so dumps stay comparable.
fn rust_chunks(data: &[u8], chunk_bytes: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < data.len() {
        let mut end = (start + chunk_bytes).min(data.len());
        while end > start && std::str::from_utf8(&data[start..end]).is_err() {
            end -= 1;
        }
        out.push((start, end));
        start = end;
    }
    out
}

fn data_golden(scheme: &Scheme, file: &str) -> Result<(), String> {
    // Data lives at the workspace's benches/data (gitignored; CI downloads).
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/data")
        .join(file);
    let data = match std::fs::read(&path) {
        Ok(d) => d,
        Err(_) => return Err(format!("data file missing: {}", path.display())),
    };
    let cap = golden_bytes();
    let mut failures = 0u32;
    let mut chunks = 0u32;
    for (start, end) in rust_chunks(&data, CHUNK) {
        if cap > 0 && start >= cap {
            break;
        }
        let text = std::str::from_utf8(&data[start..end]).unwrap();
        chunks += 1;
        if let Err(diff) = diff_case(scheme, text) {
            failures += 1;
            if failures <= 2 {
                eprintln!("MISMATCH {} chunk {} ({}..{}): {}", scheme.name, chunks, start, end, diff);
            }
        }
    }
    if chunks == 0 {
        return Err("no chunks produced".into());
    }
    if failures > 0 {
        return Err(format!("{failures}/{chunks} chunks mismatch"));
    }
    Ok(())
}

macro_rules! data_golden_test {
    ($fn_name:ident, $scheme:expr, $file:expr $(, $attr:meta)?) => {
        #[test]
        $(#[$attr])?
        fn $fn_name() {
            match data_golden($scheme, $file) {
                Ok(()) => {}
                Err(e) if e.starts_with("data file missing") => eprintln!("skipped: {e}"),
                Err(e) => panic!("{e}"),
            }
        }
    };
}

data_golden_test!(data_r50k, &oracle::R50K, "enwik8");
data_golden_test!(data_r50k_owt, &oracle::R50K, "owt_sample.txt");
data_golden_test!(data_cl100k, &oracle::CL100K, "enwik8");
data_golden_test!(data_o200k, &oracle::O200K, "enwik8");
data_golden_test!(data_qwen2, &oracle::QWEN2, "enwik8");
data_golden_test!(data_deepseek, &oracle::DEEPSEEK, "enwik8");
data_golden_test!(data_smollm, &oracle::SMOLLM, "enwik8");
data_golden_test!(data_bert, &oracle::BERT, "enwik8");
data_golden_test!(data_voyage, &oracle::VOYAGE, "enwik8");
data_golden_test!(data_cl100k_owt, &oracle::CL100K, "owt_sample.txt");
data_golden_test!(data_o200k_owt, &oracle::O200K, "owt_sample.txt");
data_golden_test!(data_qwen2_owt, &oracle::QWEN2, "owt_sample.txt");
data_golden_test!(data_deepseek_owt, &oracle::DEEPSEEK, "owt_sample.txt");
data_golden_test!(data_smollm_owt, &oracle::SMOLLM, "owt_sample.txt");
data_golden_test!(data_bert_owt, &oracle::BERT, "owt_sample.txt");
data_golden_test!(data_voyage_owt, &oracle::VOYAGE, "owt_sample.txt");

// ---------------------------------------------------------------------------
// 4. Fuzz differential: random strings vs the oracle
// ---------------------------------------------------------------------------

/// xorshift PRNG — deterministic, no dev-dep needed.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Alphabet: ASCII class chars + unicode from categories the patterns care
/// about (letters incl. marks, numerics incl. No, whitespace incl. unicode,
/// punctuation, symbols).
const FUZZ_ALPHABET: &[&str] = &[
    "a", "B", "z", " ", "\t", "\n", "\r", "'", "s", "t", "l", "v", "e", "r", "d", "m",
    "0", "1", "9", "3", ".", ",", "!", "?", "-", "_", "~", "&", "$", "+", "=", "€",
    "é", "中", "日", "한", "Ⅷ", "½", "\u{a0}", "\u{2003}", "\u{2028}", "\u{301}",
    "😒", "🇺🇸", "\u{3000}", "λ", "Ж", "ß", "X", "Q", "ll", "ve", "re",
    // onig `\s` extras (VT, FF, NEL), case-class edge letters (Lt `ǅ`, Lm `ʰ`,
    // Lo+Other_Lowercase `ª`), Other_Alphabetic symbol `Ⓐ` (So, not \p{L}),
    // format/control chars, Ps/Pe punct.
    "\u{b}", "\u{c}", "\u{85}", "ǅ", "ʰ", "ª", "Ⓐ", "\u{200b}", "\u{ad}", "\u{1b}", "「", "）",
];

/// Mostly-ASCII alphabet for the long fuzz: keeps `Mask`'s 64-byte batches
/// on the SIMD path (unicode chars defer to `Core`) while still crossing
/// batch edges with VT/FF, apostrophes, slashes and newline runs.
const FUZZ_ASCII: &[&str] = &[
    "a", "B", "z", "Q", "s", "t", "ll", "re", "ve", "d", "m", " ", " ", " ", "\t", "\n",
    "\r", "\u{b}", "\u{c}", "'", "'", "0", "1", "9", ".", ",", "!", "-", "_", "/", "$",
    "\u{1b}", "\u{7f}",
];

/// Short strings (≤ 40 picks) hit only `Core` — `Mask` needs 65 bytes for a
/// SIMD batch; long ones (64..640 picks, 7/8 ASCII) exercise the mask algebra.
fn fuzz_one(long: bool, rng: &mut Rng) -> String {
    let len = if long { 64 + rng.below(576) } else { 1 + rng.below(40) };
    let mut s = String::new();
    for _ in 0..len {
        let alpha = if long && rng.below(8) != 0 { FUZZ_ASCII } else { FUZZ_ALPHABET };
        s.push_str(alpha[rng.below(alpha.len())]);
    }
    s
}

macro_rules! fuzz_golden {
    ($fn_name:ident, $scheme:expr, $long:expr) => {
        #[test]
        fn $fn_name() {
            let scheme: &Scheme = $scheme;
            let long: bool = $long;
            let iters: u32 = std::env::var("PRETOKIE_FUZZ_ITERS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300);
            let iters = if long { (iters / 10).max(30) } else { iters };
            let mut rng = Rng(if long { 0xD1B54A32D192ED03 } else { 0x9E3779B97F4A7C15 });
            let mut failures = Vec::new();
            for i in 0..iters {
                let s = fuzz_one(long, &mut rng);
                if let Err(diff) = diff_case(scheme, &s) {
                    failures.push(format!("iter {i} {s:?}: {diff}"));
                    if failures.len() >= 5 {
                        break;
                    }
                }
            }
            assert!(failures.is_empty(), "{} fuzz failures:\n{}", failures.len(), failures.join("\n"));
        }
    };
}

fuzz_golden!(fuzz_r50k, &oracle::R50K, false);
fuzz_golden!(fuzz_cl100k, &oracle::CL100K, false);
fuzz_golden!(fuzz_o200k, &oracle::O200K, false);
fuzz_golden!(fuzz_qwen2, &oracle::QWEN2, false);
fuzz_golden!(fuzz_deepseek, &oracle::DEEPSEEK, false);
fuzz_golden!(fuzz_smollm, &oracle::SMOLLM, false);
fuzz_golden!(fuzz_bert, &oracle::BERT, false);
fuzz_golden!(fuzz_voyage, &oracle::VOYAGE, false);

fuzz_golden!(fuzz_long_r50k, &oracle::R50K, true);
fuzz_golden!(fuzz_long_cl100k, &oracle::CL100K, true);
fuzz_golden!(fuzz_long_o200k, &oracle::O200K, true);
fuzz_golden!(fuzz_long_qwen2, &oracle::QWEN2, true);
fuzz_golden!(fuzz_long_deepseek, &oracle::DEEPSEEK, true);
fuzz_golden!(fuzz_long_smollm, &oracle::SMOLLM, true);
fuzz_golden!(fuzz_long_bert, &oracle::BERT, true);
fuzz_golden!(fuzz_long_voyage, &oracle::VOYAGE, true);
