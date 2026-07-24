//! Fast pre-tokenization for BPE tokenizers.
//!
//! All pretokenizers are specialized, zero-allocation iterators from the `pretokie` crate,
//! with a regex fallback for unknown patterns.
//!
//! # Example
//!
//! ```
//! use tokie::pretok::Pretokenizer;
//!
//! let pretok = Pretokenizer::gpt2();
//! let pieces: Vec<&str> = pretok.split("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//! ```

use std::sync::Arc;

pub use pretokie::Regex as RegexPretok;

/// Type of pretokenizer for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PretokType {
    None = 0,
    Gpt2 = 1,
    Cl100k = 2,
    O200k = 3,
    Bert = 4,
    Voyage = 5,
    DeepSeek = 6,
    SmolLM = 7,
    Qwen35 = 8,
}

impl PretokType {
    /// Create a `Pretokenizer` from this type.
    pub fn to_pretokenizer(self) -> Option<Pretokenizer> {
        match self {
            PretokType::None => None,
            _ => Some(Pretokenizer::from_type(self)),
        }
    }
}

// ===========================================================================
// Pretokenizer — stored once, creates fast iterators
// ===========================================================================

/// A pretokenizer factory backed by specialized `pretokie` iterators.
///
/// Stored once in the `Tokenizer` at construction time. Each call to `split()`
/// creates a new zero-allocation iterator of the correct type.
#[derive(Clone, Debug)]
pub enum Pretokenizer {
    Gpt2,
    Cl100k,
    Bert,
    O200k,
    Voyage,
    SmolLM,
    DeepSeek,
    Qwen,
    Regex(Arc<pretokie::Regex>),
}

/// Iterator returned by [`Pretokenizer::split`].
pub enum PretokenizerIter<'a> {
    Gpt2(pretokie::Gpt2<'a>),
    Cl100k(pretokie::Cl100k<'a>),
    Bert(pretokie::Bert<'a>),
    O200k(pretokie::O200k<'a>),
    Voyage(pretokie::Voyage<'a>),
    SmolLM(pretokie::SmolLM<'a>),
    DeepSeek(pretokie::DeepSeek<'a>),
    Qwen(pretokie::Qwen<'a>),
    Regex(pretokie::regex::RegexIter<'a>),
}

impl<'a> Iterator for PretokenizerIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self {
            PretokenizerIter::Gpt2(it) => it.next(),
            PretokenizerIter::Cl100k(it) => it.next(),
            PretokenizerIter::Bert(it) => it.next(),
            PretokenizerIter::O200k(it) => it.next(),
            PretokenizerIter::Voyage(it) => it.next(),
            PretokenizerIter::SmolLM(it) => it.next(),
            PretokenizerIter::DeepSeek(it) => it.next(),
            PretokenizerIter::Qwen(it) => it.next(),
            PretokenizerIter::Regex(it) => it.next(),
        }
    }
}

impl Pretokenizer {
    /// Create a `Pretokenizer` from a `PretokType`.
    pub fn from_type(ty: PretokType) -> Self {
        match ty {
            PretokType::None => Pretokenizer::Gpt2, // fallback
            PretokType::Gpt2 => Pretokenizer::Gpt2,
            PretokType::Cl100k => Pretokenizer::Cl100k,
            PretokType::Bert => Pretokenizer::Bert,
            PretokType::O200k => Pretokenizer::O200k,
            PretokType::Voyage => Pretokenizer::Voyage,
            PretokType::SmolLM => Pretokenizer::SmolLM,
            PretokType::DeepSeek => Pretokenizer::DeepSeek,
            PretokType::Qwen35 => Pretokenizer::Qwen,
        }
    }

    /// Create a regex-based pretokenizer from a compiled `pretokie::Regex`.
    pub fn from_regex(regex: pretokie::Regex) -> Self {
        Pretokenizer::Regex(Arc::new(regex))
    }

    /// Convenience constructors.
    pub fn gpt2() -> Self { Pretokenizer::Gpt2 }
    pub fn cl100k() -> Self { Pretokenizer::Cl100k }
    pub fn bert() -> Self { Pretokenizer::Bert }
    pub fn o200k() -> Self { Pretokenizer::O200k }
    pub fn voyage() -> Self { Pretokenizer::Voyage }
    pub fn smollm() -> Self { Pretokenizer::SmolLM }
    pub fn deepseek() -> Self { Pretokenizer::DeepSeek }
    pub fn qwen() -> Self { Pretokenizer::Qwen }

    /// Visit every piece of `text`, dispatching on the pretokenizer type
    /// once per call instead of once per piece: each arm runs a
    /// monomorphized loop over the concrete iterator, so the walker and
    /// the per-piece consumer inline together without the enum-iterator
    /// boundary of [`Self::split`]. This is the bulk-encode hot loop.
    #[inline]
    pub fn for_each_piece<'a, F: FnMut(&'a str)>(&'a self, text: &'a str, mut f: F) {
        match self {
            Pretokenizer::Gpt2 => for p in pretokie::Gpt2::new(text) { f(p) },
            Pretokenizer::Cl100k => for p in pretokie::Cl100k::new(text) { f(p) },
            Pretokenizer::Bert => for p in pretokie::Bert::new(text) { f(p) },
            Pretokenizer::O200k => for p in pretokie::O200k::new(text) { f(p) },
            Pretokenizer::Voyage => for p in pretokie::Voyage::new(text) { f(p) },
            Pretokenizer::SmolLM => for p in pretokie::SmolLM::new(text) { f(p) },
            Pretokenizer::DeepSeek => for p in pretokie::DeepSeek::new(text) { f(p) },
            Pretokenizer::Qwen => for p in pretokie::Qwen::new(text) { f(p) },
            Pretokenizer::Regex(r) => for p in r.split(text) { f(p) },
        }
    }

    /// Split text into pre-tokens using the fastest available implementation.
    #[inline]
    pub fn split<'a>(&'a self, text: &'a str) -> PretokenizerIter<'a> {
        match self {
            Pretokenizer::Gpt2 => PretokenizerIter::Gpt2(pretokie::Gpt2::new(text)),
            Pretokenizer::Cl100k => PretokenizerIter::Cl100k(pretokie::Cl100k::new(text)),
            Pretokenizer::Bert => PretokenizerIter::Bert(pretokie::Bert::new(text)),
            Pretokenizer::O200k => PretokenizerIter::O200k(pretokie::O200k::new(text)),
            Pretokenizer::Voyage => PretokenizerIter::Voyage(pretokie::Voyage::new(text)),
            Pretokenizer::SmolLM => PretokenizerIter::SmolLM(pretokie::SmolLM::new(text)),
            Pretokenizer::DeepSeek => PretokenizerIter::DeepSeek(pretokie::DeepSeek::new(text)),
            Pretokenizer::Qwen => PretokenizerIter::Qwen(pretokie::Qwen::new(text)),
            Pretokenizer::Regex(r) => PretokenizerIter::Regex(r.split(text)),
        }
    }
}
