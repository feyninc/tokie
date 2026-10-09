//! Dev-only oracle: a faithful re-implementation of HuggingFace `tokenizers`
//! pretokenization, compiled with the exact engine HF uses (Oniguruma via
//! `onig` 6.5.1, `Regex::new` = UTF-8 + `ONIG_OPTION_NONE` + Ruby syntax).
//!
//! Spec source: `docs/pretokie-v2/00-oracle-engine.md`. The engine mirrors
//! `tokenizers/src/utils/onig.rs` (`find_matches` gap/match splitting),
//! `SplitDelimiterBehavior` (`Isolated`/`Removed`/`Contiguous`), `Invert`,
//! `Sequence` composition, and the char-class stages (`Digits`,
//! `Punctuation`, `BertPreTokenizer`), including the ByteLevel byte→unicode
//! remap as a mid-sequence transform (Falcon-style digit splits operate on
//! remapped text and can split inside multi-byte chars — HF does this too).
//!
//! Output: pretokens as original **byte** slices of the input.
//!
//! Known approximation: ByteLevel `add_prefix_space=true` is applied once
//! to the whole input rather than per existing split (only reachable for
//! hypothetical aps=true + Sequence pipelines; every canonical scheme in
//! `ALL_SCHEMES` uses aps=false, where the two are identical).

use onig::Regex;
use std::borrow::Cow;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Canonical pattern strings (verbatim from HF sources)
// ---------------------------------------------------------------------------

/// gpt2 / r50k: `pre_tokenizers/byte_level.rs:43-46`
pub const R50K_PATTERN: &str =
    r#"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"#;

/// cl100k_base: Xenova/gpt-4 tokenizer.json (Split, Removed, invert)
pub const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// o200k_base: Xenova/gpt-4o tokenizer.json (Split, Removed, invert)
pub const O200K_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// qwen2: Qwen/Qwen2-7B tokenizer.json (Split, Isolated)
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// voyage (3/3-lite/code-3 all identical): voyageai/voyage-3 tokenizer.json
/// (Split, Isolated) — same pattern as qwen2, kept as its own constant so
/// the schemes' spec sources stay separable.
pub const VOYAGE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// deepseek-v3 stage 1: digits (Split, Isolated)
pub const DEEPSEEK_DIGITS_PATTERN: &str = r"\p{N}{1,3}";
/// deepseek-v3 stage 2: CJK + Kana literal ranges (Split, Isolated)
pub const DEEPSEEK_CJK_PATTERN: &str = "[\u{4e00}-\u{9fa5}\u{3040}-\u{309f}\u{30a0}-\u{30ff}]+";
/// deepseek-v3 stage 3: main pattern (Split, Isolated)
pub const DEEPSEEK_MAIN_PATTERN: &str = r##"[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"##;

/// falcon-style fallback: last Split stage
pub const FALCON_TRIPLETS_PATTERN: &str = r"[0-9][0-9][0-9]";

static RE_R50K: LazyLock<Regex> = LazyLock::new(|| Regex::new(R50K_PATTERN).unwrap());
static RE_CL100K: LazyLock<Regex> = LazyLock::new(|| Regex::new(CL100K_PATTERN).unwrap());
static RE_O200K: LazyLock<Regex> = LazyLock::new(|| Regex::new(O200K_PATTERN).unwrap());
static RE_QWEN2: LazyLock<Regex> = LazyLock::new(|| Regex::new(QWEN2_PATTERN).unwrap());
static RE_VOYAGE: LazyLock<Regex> = LazyLock::new(|| Regex::new(VOYAGE_PATTERN).unwrap());
static RE_DS_DIGITS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(DEEPSEEK_DIGITS_PATTERN).unwrap());
static RE_DS_CJK: LazyLock<Regex> = LazyLock::new(|| Regex::new(DEEPSEEK_CJK_PATTERN).unwrap());
static RE_DS_MAIN: LazyLock<Regex> = LazyLock::new(|| Regex::new(DEEPSEEK_MAIN_PATTERN).unwrap());
static RE_FALCON_TRIPLETS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(FALCON_TRIPLETS_PATTERN).unwrap());

// ---------------------------------------------------------------------------
// ByteLevel byte→unicode map (byte_level.rs:15-39)
// ---------------------------------------------------------------------------

static BYTES_CHAR: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(b'\xA1'..=b'\xAC');
    bs.extend(b'\xAE'..=b'\xFF');
    let mut cs: Vec<u32> = bs.iter().map(|&i| i as u32).collect();
    let mut n = 0;
    for b in 0..=255u8 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = ['\0'; 256];
    for (b, c) in bs.into_iter().zip(cs) {
        table[b as usize] = char::from_u32(c).unwrap();
    }
    table
});

fn remap_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| BYTES_CHAR[b as usize]).collect()
}

// ---------------------------------------------------------------------------
// Scheme specs (stage sequences, verbatim from tokenizer.json pipelines)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Behavior {
    Isolated,
    Removed,
    Contiguous,
}

#[derive(Clone, Copy)]
pub enum Stage {
    /// `Split` pre-tokenizer with a Regex pattern.
    Re {
        re: &'static LazyLock<Regex>,
        behavior: Behavior,
        invert: bool,
    },
    /// `ByteLevel` pre-tokenizer: optional space prepend, optional regex
    /// split, then the byte→unicode remap. Later stages see remapped text.
    ByteLevel {
        add_prefix_space: bool,
        use_regex: bool,
    },
    /// `Digits` pre-tokenizer (`char::is_numeric`, Rust std = Unicode N*).
    Digits { individual: bool },
    /// `Punctuation` pre-tokenizer (ascii punct ∪ Unicode P).
    Punct(Behavior),
    /// `BertPreTokenizer`: whitespace Removed, then punct Isolated.
    Bert,
}

pub struct Scheme {
    pub name: &'static str,
    pub stages: &'static [Stage],
}

pub const R50K: Scheme = Scheme {
    name: "r50k",
    stages: &[Stage::ByteLevel { add_prefix_space: false, use_regex: true }],
};

pub const CL100K: Scheme = Scheme {
    name: "cl100k",
    stages: &[
        Stage::Re { re: &RE_CL100K, behavior: Behavior::Removed, invert: true },
        Stage::ByteLevel { add_prefix_space: false, use_regex: false },
    ],
};

pub const O200K: Scheme = Scheme {
    name: "o200k",
    stages: &[
        Stage::Re { re: &RE_O200K, behavior: Behavior::Removed, invert: true },
        Stage::ByteLevel { add_prefix_space: false, use_regex: false },
    ],
};

pub const QWEN2: Scheme = Scheme {
    name: "qwen2",
    stages: &[
        Stage::Re { re: &RE_QWEN2, behavior: Behavior::Isolated, invert: false },
        Stage::ByteLevel { add_prefix_space: false, use_regex: false },
    ],
};

pub const VOYAGE: Scheme = Scheme {
    name: "voyage",
    stages: &[
        Stage::Re { re: &RE_VOYAGE, behavior: Behavior::Isolated, invert: false },
        Stage::ByteLevel { add_prefix_space: false, use_regex: false },
    ],
};

pub const DEEPSEEK: Scheme = Scheme {
    name: "deepseek",
    stages: &[
        Stage::Re { re: &RE_DS_DIGITS, behavior: Behavior::Isolated, invert: false },
        Stage::Re { re: &RE_DS_CJK, behavior: Behavior::Isolated, invert: false },
        Stage::Re { re: &RE_DS_MAIN, behavior: Behavior::Isolated, invert: false },
        Stage::ByteLevel { add_prefix_space: false, use_regex: false },
    ],
};

pub const SMOLLM: Scheme = Scheme {
    name: "smollm",
    stages: &[
        Stage::Digits { individual: true },
        Stage::ByteLevel { add_prefix_space: false, use_regex: true },
    ],
};

pub const BERT: Scheme = Scheme {
    name: "bert",
    stages: &[Stage::Bert],
};

/// Falcon-style 4-stage fallback family.
pub const FALCON: Scheme = Scheme {
    name: "falcon",
    stages: &[
        Stage::Punct(Behavior::Contiguous),
        Stage::ByteLevel { add_prefix_space: false, use_regex: true },
        Stage::Digits { individual: false },
        Stage::Re { re: &RE_FALCON_TRIPLETS, behavior: Behavior::Isolated, invert: false },
    ],
};

pub const ALL_SCHEMES: [&Scheme; 9] =
    [&R50K, &CL100K, &O200K, &QWEN2, &VOYAGE, &DEEPSEEK, &SMOLLM, &BERT, &FALCON];

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// A piece under construction. `start..end` is a byte range of the (possibly
/// space-prepended) input. `remapped` marks pieces whose content is the
/// ByteLevel-remapped text — for those, one content char == one original
/// input byte, and content coordinates are char indices.
#[derive(Clone, Copy, Debug)]
struct Piece {
    start: usize,
    end: usize,
    remapped: bool,
}

/// The piece's content in string form (byte coords if not remapped).
fn piece_text<'a>(input: &'a str, p: &Piece) -> Cow<'a, str> {
    if p.remapped {
        // Remapped content: byte-slice (post-remap stages may split inside
        // a multi-byte char) with each byte mapped to its remap char.
        Cow::Owned(remap_bytes(&input.as_bytes()[p.start..p.end]))
    } else {
        Cow::Borrowed(&input[p.start..p.end])
    }
}

/// Byte offset in `text` → content coordinate. Raw pieces: content coords
/// are byte offsets (identity). Remapped pieces: content coords are CHAR
/// indices (== original byte offsets), so a byte offset inside a multi-byte
/// char floors to the char count before it (HF splits exactly there).
fn to_content_coords(p: &Piece, text: &str, byte_offset: usize) -> usize {
    if !p.remapped {
        return byte_offset;
    }
    text.char_indices()
        .take_while(|(i, _)| *i < byte_offset)
        .count()
}

/// Apply a behavior to a `(start, end, matched)` list in content coords.
fn apply_behavior(
    splits: Vec<(usize, usize, bool)>,
    behavior: Behavior,
) -> Vec<(usize, usize)> {
    match behavior {
        Behavior::Isolated => splits
            .into_iter()
            .filter(|(s, e, _)| e > s)
            .map(|(s, e, _)| (s, e))
            .collect(),
        Behavior::Removed => splits
            .into_iter()
            .filter(|(s, e, m)| !m && e > s)
            .map(|(s, e, _)| (s, e))
            .collect(),
        Behavior::Contiguous => {
            let mut out = Vec::new();
            let mut open: Option<(usize, usize)> = None;
            for (s, e, m) in splits {
                if m {
                    match &mut open {
                        Some((_, end)) => *end = e,
                        None => open = Some((s, e)),
                    }
                } else {
                    if let Some(piece) = open.take() {
                        out.push(piece);
                    }
                    if e > s {
                        out.push((s, e));
                    }
                }
            }
            if let Some(piece) = open.take() {
                out.push(piece);
            }
            out
        }
    }
}

/// Char-class splits list only matched spans; add the gaps between them so
/// behavior application sees the full picture.
fn with_gaps(len: usize, mut matches: Vec<(usize, usize, bool)>) -> Vec<(usize, usize, bool)> {
    matches.sort_unstable_by_key(|(s, _, _)| *s);
    let mut out = Vec::with_capacity(matches.len() * 2 + 1);
    let mut prev = 0;
    for (s, e, m) in matches {
        if prev != s {
            out.push((prev, s, false));
        }
        out.push((s, e, m));
        prev = e;
    }
    if prev != len {
        out.push((prev, len, false));
    }
    out
}

/// `Pattern::find_matches` for a regex (utils/onig.rs:25-44). Offsets are
/// byte offsets into `text`.
fn find_matches(re: &Regex, text: &str) -> Vec<(usize, usize, bool)> {
    if text.is_empty() {
        return vec![(0, 0, false)];
    }
    let mut prev = 0;
    let mut out = Vec::new();
    for (s, e) in re.find_iter(text) {
        if prev != s {
            out.push((prev, s, false));
        }
        out.push((s, e, true));
        prev = e;
    }
    if prev != text.len() {
        out.push((prev, text.len(), false));
    }
    out
}

/// HF `is_bert_punc` (pre_tokenizers/bert.rs): ASCII punct or the
/// `unicode_categories` crate's `is_punctuation`, which is all seven P
/// categories (Pc Pd Ps Pe Pi Pf Po).
fn is_bert_punc(c: char) -> bool {
    use unicode_general_category::{get_general_category, GeneralCategory};
    c.is_ascii_punctuation()
        || matches!(
            get_general_category(c),
            GeneralCategory::OpenPunctuation
                | GeneralCategory::ClosePunctuation
                | GeneralCategory::ConnectorPunctuation
                | GeneralCategory::DashPunctuation
                | GeneralCategory::FinalPunctuation
                | GeneralCategory::InitialPunctuation
                | GeneralCategory::OtherPunctuation
        )
}

/// Iterate a piece's chars as (content_coord, char). Content coords are
/// byte offsets for raw pieces and char indices (== original byte offsets)
/// for remapped pieces.
fn piece_chars(input: &str, p: &Piece) -> Vec<(usize, char)> {
    let text = piece_text(input, p);
    if !p.remapped {
        text.char_indices().map(|(b, c)| (b, c)).collect()
    } else {
        text.chars().enumerate().map(|(i, c)| (i, c)).collect()
    }
}

/// Char width in content coords (1 for remapped pieces, UTF-8 len for raw).
fn char_w(p: &Piece, c: char) -> usize {
    if p.remapped { 1 } else { c.len_utf8() }
}

/// Apply one stage to one piece; returns sub-pieces in input coords.
fn apply_stage(input: &str, stage: &Stage, p: Piece) -> Vec<Piece> {
    match stage {
        Stage::Re { re, behavior, invert } => {
            let text = piece_text(input, &p);
            let splits = find_matches(re, &text);
            let splits = if *invert {
                splits.into_iter().map(|(s, e, m)| (s, e, !m)).collect()
            } else {
                splits
            };
            apply_behavior(splits, *behavior)
                .into_iter()
                .map(|(s, e)| {
                    let cs = to_content_coords(&p, &text, s);
                    let ce = to_content_coords(&p, &text, e);
                    Piece { start: p.start + cs, end: p.start + ce, remapped: p.remapped }
                })
                .collect()
        }
        Stage::ByteLevel { add_prefix_space, use_regex } => {
            let text = piece_text(input, &p);
            // Prepend happens per existing split in HF; we approximate at
            // input level (see module doc) — per-piece it only matters for
            // aps=true + Sequence pipelines, which no canonical scheme uses.
            if *use_regex {
                let splits = find_matches(&RE_R50K, &text);
                apply_behavior(splits, Behavior::Isolated)
                    .into_iter()
                    .map(|(s, e)| {
                        let cs = to_content_coords(&p, &text, s);
                        let ce = to_content_coords(&p, &text, e);
                        // This stage remaps the content.
                        Piece { start: p.start + cs, end: p.start + ce, remapped: true }
                    })
                    .collect()
            } else {
                vec![Piece { start: p.start, end: p.end, remapped: true }]
            }
        }
        Stage::Digits { individual } => {
            let behavior = if *individual { Behavior::Isolated } else { Behavior::Contiguous };
            let matches: Vec<(usize, usize, bool)> = piece_chars(input, &p)
                .into_iter()
                .filter(|(_, c)| c.is_numeric())
                .map(|(i, c)| (i, i + char_w(&p, c), true))
                .collect();
            let splits = with_gaps(p.end - p.start, matches);
            apply_behavior(splits, behavior)
                .into_iter()
                .map(|(s, e)| Piece { start: p.start + s, end: p.start + e, remapped: p.remapped })
                .collect()
        }
        Stage::Punct(behavior) => {
            let matches: Vec<(usize, usize, bool)> = piece_chars(input, &p)
                .into_iter()
                .filter(|(_, c)| is_bert_punc(*c))
                .map(|(i, c)| (i, i + char_w(&p, c), true))
                .collect();
            let splits = with_gaps(p.end - p.start, matches);
            apply_behavior(splits, *behavior)
                .into_iter()
                .map(|(s, e)| Piece { start: p.start + s, end: p.start + e, remapped: p.remapped })
                .collect()
        }
        Stage::Bert => {
            // Stage 1: whitespace Removed (raw text; BERT has no remap).
            let matches: Vec<(usize, usize, bool)> = piece_chars(input, &p)
                .into_iter()
                .filter(|(_, c)| c.is_whitespace())
                .map(|(i, c)| (i, i + char_w(&p, c), true))
                .collect();
            let splits = with_gaps(p.end - p.start, matches);
            let kept = apply_behavior(splits, Behavior::Removed);
            // Stage 2: punct Isolated per kept piece.
            let mut out = Vec::new();
            for (s, e) in kept {
                let sub = Piece { start: p.start + s, end: p.start + e, remapped: p.remapped };
                let matches: Vec<(usize, usize, bool)> = piece_chars(input, &sub)
                    .into_iter()
                    .filter(|(_, c)| is_bert_punc(*c))
                    .map(|(i, c)| (i, i + char_w(&sub, c), true))
                    .collect();
                let splits = with_gaps(sub.end - sub.start, matches);
                for (a, b) in apply_behavior(splits, Behavior::Isolated) {
                    out.push(Piece { start: sub.start + a, end: sub.start + b, remapped: false });
                }
            }
            out
        }
    }
}

/// Run the full pipeline; returns pretokens as byte slices of the input.
/// Post-remap stages can split inside multi-byte chars, hence bytes.
pub fn pretokens(scheme: &Scheme, text: &str) -> Vec<Vec<u8>> {
    // add_prefix_space approximation: single input-wide prepend (module doc).
    let needs_space = scheme.stages.iter().any(|s| matches!(s, Stage::ByteLevel { add_prefix_space: true, .. }))
        && !text.starts_with(' ');
    let input;
    let input: &str = if needs_space {
        input = format!(" {text}");
        &input
    } else {
        text
    };

    let mut pieces = vec![Piece { start: 0, end: input.len(), remapped: false }];
    for stage in scheme.stages {
        let mut next = Vec::new();
        for p in pieces {
            next.extend(apply_stage(input, stage, p));
        }
        pieces = next;
    }
    pieces
        .into_iter()
        .map(|p| input.as_bytes()[p.start..p.end].to_vec())
        .filter(|v| !v.is_empty())
        .collect()
}