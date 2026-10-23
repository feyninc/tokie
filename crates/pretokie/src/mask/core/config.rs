//! Configuration trait and dimension enums for BPE pretokenizers.

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ContractionCase {
    Sensitive,
    Insensitive,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ContractionMode {
    Standalone,
    Suffix,
    /// No contraction rule in the pattern (DeepSeek): apostrophe+letters is
    /// handled by the punct-prefix rule instead.
    None,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DigitMode {
    Unlimited,
    Chunked3,
    Single,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum LetterMode {
    Plain,
    PlainWithMarks,
    CamelCase,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WsPattern {
    Gpt2,
    Cl100k,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PunctTrailing {
    None,
    Newlines,
    NewlinesAndSlashes,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum WsException {
    None,
    Digits,
    Cjk,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PunctPrefixMode {
    SpaceOnly,
    Any,
    AsciiOnly,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PunctClass {
    /// Negated class like `[^\s\p{L}\p{N}]` — includes Cf/Cc/Co chars.
    NegatedAlnum,
    /// Positive class `[\p{P}\p{S}]` (DeepSeek) — excludes format/control chars.
    PunctSymbolOnly,
}

pub trait PretokConfig {
    const CONTRACTION_CASE: ContractionCase;
    const CONTRACTION_MODE: ContractionMode;
    const DIGIT_MODE: DigitMode;
    const LETTER_MODE: LetterMode;
    const SPACE_PREFIXES_DIGITS: bool;
    const WS_PATTERN: WsPattern;
    const PUNCT_TRAILING: PunctTrailing;
    const WS_EXCEPTION: WsException;
    const PUNCT_PREFIX_MODE: PunctPrefixMode;
    const PUNCT_CLASS: PunctClass = PunctClass::NegatedAlnum;
}
