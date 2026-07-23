//! Zero-sized config types for each BPE pretokenizer.

use crate::core::config::*;

pub struct Gpt2Config;
impl PretokConfig for Gpt2Config {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Sensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Standalone;
    const DIGIT_MODE: DigitMode = DigitMode::Unlimited;
    const LETTER_MODE: LetterMode = LetterMode::Plain;
    const SPACE_PREFIXES_DIGITS: bool = true;
    const WS_PATTERN: WsPattern = WsPattern::Gpt2;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::None;
    const WS_EXCEPTION: WsException = WsException::None;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::SpaceOnly;
}

pub struct Cl100kConfig;
impl PretokConfig for Cl100kConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Insensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Standalone;
    const DIGIT_MODE: DigitMode = DigitMode::Chunked3;
    const LETTER_MODE: LetterMode = LetterMode::Plain;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Cl100k;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::Newlines;
    const WS_EXCEPTION: WsException = WsException::None;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::Any;
}

pub struct O200kConfig;
impl PretokConfig for O200kConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Insensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Suffix;
    const DIGIT_MODE: DigitMode = DigitMode::Chunked3;
    const LETTER_MODE: LetterMode = LetterMode::CamelCase;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Cl100k;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::NewlinesAndSlashes;
    const WS_EXCEPTION: WsException = WsException::None;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::Any;
}

pub struct VoyageConfig;
impl PretokConfig for VoyageConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Insensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Standalone;
    const DIGIT_MODE: DigitMode = DigitMode::Single;
    const LETTER_MODE: LetterMode = LetterMode::Plain;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Cl100k;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::Newlines;
    const WS_EXCEPTION: WsException = WsException::None;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::Any;
}

pub struct SmolLMConfig;
impl PretokConfig for SmolLMConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Sensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Standalone;
    const DIGIT_MODE: DigitMode = DigitMode::Single;
    const LETTER_MODE: LetterMode = LetterMode::Plain;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Gpt2;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::None;
    const WS_EXCEPTION: WsException = WsException::Digits;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::SpaceOnly;
}

pub struct DeepSeekConfig;
impl PretokConfig for DeepSeekConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Insensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::None;
    const DIGIT_MODE: DigitMode = DigitMode::Chunked3;
    const LETTER_MODE: LetterMode = LetterMode::PlainWithMarks;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Cl100k;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::Newlines;
    const WS_EXCEPTION: WsException = WsException::Cjk;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::AsciiOnly;
    const PUNCT_CLASS: PunctClass = PunctClass::PunctSymbolOnly;
}

pub struct QwenConfig;
impl PretokConfig for QwenConfig {
    const CONTRACTION_CASE: ContractionCase = ContractionCase::Insensitive;
    const CONTRACTION_MODE: ContractionMode = ContractionMode::Standalone;
    const DIGIT_MODE: DigitMode = DigitMode::Single;
    const LETTER_MODE: LetterMode = LetterMode::PlainWithMarks;
    const SPACE_PREFIXES_DIGITS: bool = false;
    const WS_PATTERN: WsPattern = WsPattern::Cl100k;
    const PUNCT_TRAILING: PunctTrailing = PunctTrailing::Newlines;
    const WS_EXCEPTION: WsException = WsException::None;
    const PUNCT_PREFIX_MODE: PunctPrefixMode = PunctPrefixMode::Any;
}
