//! One call per normalization scheme, for tokie, gigatoken, and Hugging Face
//! tokenizers v1.
//!
//! A `None` return means that library has no scheme for the row. Precompiled
//! rows use the charsmap embedded in `google-t5/t5-small`.

use std::borrow::Cow;
use std::sync::OnceLock;

use gigatoken_rs::normalize::{PrecompiledMap, PrependScheme};
use tk_encode::normalizers::metaspace::{MetaspaceNormalizer, PrependBehavior};
use tk_encode::normalizers::precompiled::PrecompiledNormalizer;
use tk_encode::normalizers::{
    BertNormalizer, ByteLevel, Lowercase, NFC, NFD, NFKC, NFKD, Nmt, Prepend, Replace, Strip,
    StripAccents,
};
use tk_encode::normalizers::replace::ReplacePattern;
use tk_encode::pipeline::Normalizer;
use tokie::normalizer::{MetaspacePrepend, Normalizer as TokieNormalizer};
use tokie::PrecompiledCharsmap;

const T5_CHARSMAP: &[u8] = include_bytes!("../data/t5.charsmap");

/// Hand cases plus one string of every scalar in the Basic Multilingual Plane.
pub fn cases() -> Vec<String> {
    let mut out: Vec<String> = HAND.iter().map(|s| (*s).to_string()).collect();
    out.push((0..=0xFFFF).filter_map(char::from_u32).collect());
    out
}

/// Inputs the oracles share. The BMP string is added by [`cases`].
pub const HAND: &[&str] = &[
    "",
    "Hello",
    "HELLO",
    "Hello World",
    "  Hello   world  ",
    "Hello\tworld\n",
    "Hello\r\nworld",
    "aa bb cc",
    " aa bb",
    "aa\tbb  cc",
    "café",
    "CAFÉ",
    "e\u{301}",
    "\u{e9}",
    "ﬁ",
    "İstanbul",
    "straße",
    "Straße",
    "中文",
    "a中b",
    "Hello  world",
    " hello",
    "▁hello",
    "\u{FEFF}Hello",
    "a\u{0000}b",
    "a\u{0001}b",
    "a\u{00A0}b",
    "a\u{200B}b",
    "\u{0007}bell",
    "MiXeD",
    "don't",
    "a.b  c",
    "\n\n",
    "   ",
    "cafe\u{301}",
    "a\u{030A}\u{0327}",
    "a\u{07FD}b",
    "a\u{1CF2}b",
    "null\0here",
    "repl\u{FFFD}char",
    "ctrl\u{0007}char",
    " ",
    "Hi!",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Nmt,
    Lowercase,
    StripAccents,
    StripAccentsNfd,
    BertUncased,
    BertUncasedCjk,
    BertCasedCjk,
    CleanText,
    PrependMark,
    ReplaceSpace,
    CollapseSpaces,
    Strip,
    MetaspaceAlways,
    PrependThenReplace,
    MetaspaceFirst,
    MetaspaceNever,
    MetaspaceDrop,
    ByteLevel,
    Precompiled,
    T5,
    SentencePiece,
    SentencePieceLower,
}

impl Scheme {
    pub const ALL: &[Scheme] = &[
        Scheme::Nfc,
        Scheme::Nfd,
        Scheme::Nfkc,
        Scheme::Nfkd,
        Scheme::Nmt,
        Scheme::Lowercase,
        Scheme::StripAccents,
        Scheme::StripAccentsNfd,
        Scheme::BertUncased,
        Scheme::BertUncasedCjk,
        Scheme::BertCasedCjk,
        Scheme::CleanText,
        Scheme::PrependMark,
        Scheme::ReplaceSpace,
        Scheme::CollapseSpaces,
        Scheme::Strip,
        Scheme::MetaspaceAlways,
        Scheme::PrependThenReplace,
        Scheme::MetaspaceFirst,
        Scheme::MetaspaceNever,
        Scheme::MetaspaceDrop,
        Scheme::ByteLevel,
        Scheme::Precompiled,
        Scheme::T5,
        Scheme::SentencePiece,
        Scheme::SentencePieceLower,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Scheme::Nfc => "nfc",
            Scheme::Nfd => "nfd",
            Scheme::Nfkc => "nfkc",
            Scheme::Nfkd => "nfkd",
            Scheme::Nmt => "nmt",
            Scheme::Lowercase => "lowercase",
            Scheme::StripAccents => "strip accents",
            Scheme::StripAccentsNfd => "strip accents after nfd",
            Scheme::BertUncased => "bert uncased",
            Scheme::BertUncasedCjk => "bert uncased + cjk spaces",
            Scheme::BertCasedCjk => "bert cased + cjk spaces",
            Scheme::CleanText => "clean text",
            Scheme::PrependMark => "prepend ▁",
            Scheme::ReplaceSpace => "replace space → ▁",
            Scheme::CollapseSpaces => "collapse 2+ spaces",
            Scheme::Strip => "strip whitespace",
            Scheme::MetaspaceAlways => "metaspace always",
            Scheme::PrependThenReplace => "prepend ▁ then replace spaces",
            Scheme::MetaspaceFirst => "metaspace first",
            Scheme::MetaspaceNever => "metaspace never",
            Scheme::MetaspaceDrop => "metaspace drop whitespace",
            Scheme::ByteLevel => "byte level",
            Scheme::Precompiled => "precompiled t5",
            Scheme::T5 => "t5 chain",
            Scheme::SentencePiece => "sentencepiece nfkc",
            Scheme::SentencePieceLower => "sentencepiece lowercase",
        }
    }

    pub fn hf<'a>(self, text: &'a str) -> Option<Cow<'a, str>> {
        let out = match self {
            Scheme::Nfc => NFC.normalize(text, 0).unwrap(),
            Scheme::Nfd => NFD.normalize(text, 0).unwrap(),
            Scheme::Nfkc => NFKC.normalize(text, 0).unwrap(),
            Scheme::Nfkd => NFKD.normalize(text, 0).unwrap(),
            Scheme::Nmt => Nmt.normalize(text, 0).unwrap(),
            Scheme::Lowercase => Lowercase.normalize(text, 0).unwrap(),
            Scheme::StripAccents => strip_accents_hf()
                .normalize(text, 0)
                .unwrap(),
            Scheme::StripAccentsNfd => return None,
            Scheme::BertUncased => bert_uncased().normalize(text, 0).unwrap(),
            Scheme::BertUncasedCjk => bert_uncased_cjk().normalize(text, 0).unwrap(),
            Scheme::BertCasedCjk => bert_cased_cjk().normalize(text, 0).unwrap(),
            Scheme::CleanText => clean_text_hf().normalize(text, 0).unwrap(),
            Scheme::PrependMark => prepend_hf().normalize(text, 0).unwrap(),
            Scheme::ReplaceSpace => replace_space_hf().normalize(text, 0).unwrap(),
            Scheme::CollapseSpaces => collapse_hf().normalize(text, 0).unwrap(),
            Scheme::Strip => strip_hf().normalize(text, 0).unwrap(),
            Scheme::MetaspaceAlways => meta_always().normalize(text, 0).unwrap(),
            Scheme::PrependThenReplace => {
                let prefixed = prepend_hf().normalize(text, 0).unwrap();
                let replaced = replace_space_hf().normalize(&prefixed, 0).unwrap();
                return Some(Cow::Owned(replaced.into_owned()));
            }
            Scheme::MetaspaceFirst => meta_first().normalize(text, 0).unwrap(),
            Scheme::MetaspaceNever => meta_never().normalize(text, 0).unwrap(),
            Scheme::MetaspaceDrop => meta_drop().normalize(text, 0).unwrap(),
            Scheme::ByteLevel => byte_level().normalize(text, 0).unwrap(),
            Scheme::Precompiled => hf_precompiled().normalize(text, 0).unwrap(),
            Scheme::T5 => {
                let mapped = hf_precompiled().normalize(text, 0).unwrap();
                let spaced = meta_drop().normalize(&mapped, 0).unwrap();
                return Some(Cow::Owned(spaced.into_owned()));
            }
            Scheme::SentencePiece | Scheme::SentencePieceLower => return None,
        };
        Some(out)
    }

    pub fn tokie<'a>(self, text: &'a str) -> Option<Cow<'a, str>> {
        Some(match self {
            Scheme::Nfc => TokieNormalizer::Nfc.normalize(text),
            Scheme::StripAccentsNfd => tokie::normalizer::strip_accents(text),
            Scheme::BertUncased => tokie::normalizer::bert_uncased_normalize(text),
            Scheme::CleanText => tokie::normalizer::clean_text(text),
            Scheme::ReplaceSpace => tokie::normalizer::fnr(text, " ", "▁"),
            Scheme::MetaspaceAlways => {
                tokie::normalizer::metaspace_normalize_with(text, MetaspacePrepend::IfNotSpaceLed, true)
            }
            Scheme::PrependThenReplace => {
                tokie::normalizer::metaspace_normalize_with(text, MetaspacePrepend::Unconditional, true)
            }
            Scheme::MetaspaceFirst => {
                tokie::normalizer::metaspace_normalize_with(text, MetaspacePrepend::FirstSegment, true)
            }
            Scheme::MetaspaceNever => tokie::normalizer::metaspace_replace_normalize(text),
            Scheme::Precompiled => {
                let mut out = String::with_capacity(text.len());
                tokie_charsmap().normalize_into(text, &mut out);
                Cow::Owned(out)
            }
            Scheme::T5 => tokie::normalizer::sentencepiece_precompiled_normalize(
                tokie_charsmap(),
                true,
                text,
            ),
            Scheme::SentencePiece => tokie::normalizer::sentencepiece_normalize(text),
            Scheme::SentencePieceLower => tokie::normalizer::sentencepiece_lowercase_normalize(text),
            _ => return None,
        })
    }

    pub fn giga<'a>(self, text: &'a str) -> Option<Cow<'a, str>> {
        Some(match self {
            Scheme::Nfc => gigatoken_rs::normalize::normalize_nfc(text),
            Scheme::PrependMark => gigatoken_rs::normalize::prepend(text, "▁"),
            Scheme::ReplaceSpace => gigatoken_rs::normalize::replace_literal(text, " ", "▁"),
            Scheme::CollapseSpaces => gigatoken_rs::normalize::collapse_spaces(text, " "),
            Scheme::Strip => gigatoken_rs::normalize::strip_whitespace(text, true, true),
            Scheme::MetaspaceAlways => {
                gigatoken_rs::normalize::metaspace(text, PrependScheme::Always, true)
            }
            Scheme::PrependThenReplace => {
                let prefixed = gigatoken_rs::normalize::prepend(text, "▁");
                Cow::Owned(
                    gigatoken_rs::normalize::replace_literal(&prefixed, " ", "▁").into_owned(),
                )
            }
            Scheme::MetaspaceFirst => {
                gigatoken_rs::normalize::metaspace(text, PrependScheme::First, true)
            }
            Scheme::MetaspaceNever => {
                gigatoken_rs::normalize::metaspace(text, PrependScheme::Never, true)
            }
            Scheme::Precompiled => Cow::Owned(giga_charsmap().normalize(text)),
            _ => return None,
        })
    }

    /// `normalie`. `None` means this row is not one of its formulas.
    ///
    /// `BertUncased` in this bench is the no-CJK tokie variant. normalie's
    /// `BertUncased` spaces CJK, so it answers [`Scheme::BertUncasedCjk`].
    pub fn normalie<'a>(self, text: &'a str) -> Option<Cow<'a, str>> {
        self.normalie_at(text, true)
    }

    pub fn normalie_at<'a>(self, text: &'a str, at_start: bool) -> Option<Cow<'a, str>> {
        let mut out = match self {
            Scheme::PrependMark if !text.is_empty() => String::with_capacity(text.len() + 3),
            _ => String::new(),
        };
        let changed = match self {
            Scheme::Nfc => normalie::Nfc::write(text, at_start, &mut out),
            Scheme::Nfd => normalie::Nfd::write(text, at_start, &mut out),
            Scheme::Nfkc => normalie::Nfkc::write(text, at_start, &mut out),
            Scheme::Nfkd => normalie::Nfkd::write(text, at_start, &mut out),
            Scheme::Nmt => normalie::Nmt::write(text, at_start, &mut out),
            Scheme::Lowercase => normalie::Lower::write(text, at_start, &mut out),
            Scheme::StripAccents => normalie::StripAccents::write(text, at_start, &mut out),
            Scheme::BertUncasedCjk => normalie::BertUncased::write(text, at_start, &mut out),
            Scheme::BertCasedCjk => normalie::BertCased::write(text, at_start, &mut out),
            Scheme::CleanText => normalie::Clean::write(text, at_start, &mut out),
            Scheme::PrependMark => normalie::Prepend::write(text, at_start, &mut out),
            Scheme::ReplaceSpace => normalie::ReplaceSpace::write(text, at_start, &mut out),
            Scheme::CollapseSpaces => normalie::CollapseSpaces::write(text, at_start, &mut out),
            Scheme::Strip => normalie::Strip::write(text, at_start, &mut out),
            Scheme::MetaspaceAlways => normalie::Metaspace::write(text, at_start, &mut out),
            Scheme::PrependThenReplace => normalie::Llama::write(text, at_start, &mut out),
            Scheme::MetaspaceFirst => normalie::MetaspaceFirst::write(text, at_start, &mut out),
            Scheme::MetaspaceNever => normalie::MetaspaceNever::write(text, at_start, &mut out),
            Scheme::MetaspaceDrop => normalie::MetaspaceWords::write(text, at_start, &mut out),
            Scheme::ByteLevel => normalie::ByteLevel::write(text, at_start, &mut out),
            Scheme::Precompiled => n_charsmap().write(text, at_start, &mut out),
            Scheme::T5 => {
                let mut scratch = normalie::Scratch::new();
                let mapped = scratch.apply2(
                    text,
                    at_start,
                    |s, at, buf| n_charsmap().write(s, at, buf),
                    |s, at, buf| normalie::MetaspaceWords::write(s, at, buf),
                );
                return Some(Cow::Owned(mapped.to_string()));
            }
            Scheme::SentencePieceLower => normalie::Albert::write(text, at_start, &mut out),
            Scheme::StripAccentsNfd | Scheme::BertUncased | Scheme::SentencePiece => return None,
        };
        Some(if changed {
            Cow::Owned(out)
        } else {
            Cow::Borrowed(text)
        })
    }
}

/// Metaspace `first` on a segment that does not open the sequence.
pub fn metaspace_first_later(text: &str) -> (Cow<'_, str>, Cow<'_, str>, Cow<'_, str>) {
    let hf = meta_first().normalize(text, 7).unwrap();
    let tokie = tokie::normalizer::metaspace_normalize_with(text, MetaspacePrepend::FirstSegment, false);
    let giga = gigatoken_rs::normalize::metaspace(text, PrependScheme::First, false);
    (hf, tokie, giga)
}

fn strip_accents_hf() -> &'static StripAccents {
    static CELL: OnceLock<StripAccents> = OnceLock::new();
    CELL.get_or_init(StripAccents::new)
}

fn bert_uncased() -> &'static BertNormalizer {
    static CELL: OnceLock<BertNormalizer> = OnceLock::new();
    CELL.get_or_init(|| BertNormalizer::new(true, false, None, true))
}

fn bert_uncased_cjk() -> &'static BertNormalizer {
    static CELL: OnceLock<BertNormalizer> = OnceLock::new();
    CELL.get_or_init(|| BertNormalizer::new(true, true, None, true))
}

fn bert_cased_cjk() -> &'static BertNormalizer {
    static CELL: OnceLock<BertNormalizer> = OnceLock::new();
    CELL.get_or_init(|| BertNormalizer::new(true, true, Some(false), false))
}

fn clean_text_hf() -> &'static BertNormalizer {
    static CELL: OnceLock<BertNormalizer> = OnceLock::new();
    CELL.get_or_init(|| BertNormalizer::new(true, false, Some(false), false))
}

fn prepend_hf() -> &'static Prepend {
    static CELL: OnceLock<Prepend> = OnceLock::new();
    CELL.get_or_init(|| Prepend::new("▁".to_string()))
}

fn replace_space_hf() -> &'static Replace {
    static CELL: OnceLock<Replace> = OnceLock::new();
    CELL.get_or_init(|| Replace::new(" ", "▁").unwrap())
}

fn collapse_hf() -> &'static Replace {
    static CELL: OnceLock<Replace> = OnceLock::new();
    CELL.get_or_init(|| Replace::new(ReplacePattern::Regex(" {2,}".to_string()), " ").unwrap())
}

fn strip_hf() -> &'static Strip {
    static CELL: OnceLock<Strip> = OnceLock::new();
    CELL.get_or_init(|| Strip::new(true, true))
}

fn meta_always() -> &'static MetaspaceNormalizer {
    static CELL: OnceLock<MetaspaceNormalizer> = OnceLock::new();
    CELL.get_or_init(|| MetaspaceNormalizer::new('▁', PrependBehavior::Always, false))
}

fn meta_first() -> &'static MetaspaceNormalizer {
    static CELL: OnceLock<MetaspaceNormalizer> = OnceLock::new();
    CELL.get_or_init(|| MetaspaceNormalizer::new('▁', PrependBehavior::First, false))
}

fn meta_never() -> &'static MetaspaceNormalizer {
    static CELL: OnceLock<MetaspaceNormalizer> = OnceLock::new();
    CELL.get_or_init(|| MetaspaceNormalizer::new('▁', PrependBehavior::Never, false))
}

fn meta_drop() -> &'static MetaspaceNormalizer {
    static CELL: OnceLock<MetaspaceNormalizer> = OnceLock::new();
    CELL.get_or_init(|| MetaspaceNormalizer::new('▁', PrependBehavior::Always, true))
}

fn byte_level() -> &'static ByteLevel {
    static CELL: OnceLock<ByteLevel> = OnceLock::new();
    CELL.get_or_init(ByteLevel::new)
}

fn hf_precompiled() -> &'static PrecompiledNormalizer {
    static CELL: OnceLock<PrecompiledNormalizer> = OnceLock::new();
    CELL.get_or_init(|| PrecompiledNormalizer::from_charsmap(T5_CHARSMAP).unwrap())
}

fn tokie_charsmap() -> &'static PrecompiledCharsmap {
    static CELL: OnceLock<PrecompiledCharsmap> = OnceLock::new();
    CELL.get_or_init(|| PrecompiledCharsmap::from_blob(T5_CHARSMAP).unwrap())
}

fn giga_charsmap() -> &'static PrecompiledMap {
    static CELL: OnceLock<PrecompiledMap> = OnceLock::new();
    CELL.get_or_init(|| PrecompiledMap::new(T5_CHARSMAP).unwrap())
}

fn n_charsmap() -> &'static normalie::Precompiled {
    static CELL: OnceLock<normalie::Precompiled> = OnceLock::new();
    CELL.get_or_init(|| normalie::Precompiled::parse(T5_CHARSMAP).unwrap())
}
