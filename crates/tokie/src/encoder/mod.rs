//! Token encoders for different tokenization algorithms.
//!
//! This module provides encoder implementations:
//!
//! - [`BacktrackingBytePairEncoder`] - Fast greedy BPE with backtracking (for tiktoken-style)
//! - [`BytePairEncoder`] - Simple O(n²) BPE (fast for small inputs with pretokenization)
//! - [`WordPieceEncoder`] - WordPiece tokenization (for BERT-style)
//! - [`SentencePieceBPE`] - BPE with merge rank checking (for SentencePiece-style)
//!
//! The [`Encoder`] enum wraps these implementations for use in the Tokenizer.

mod backtracking;
mod sentencepiece;
mod simple;
mod unigram;
mod wordpiece;

pub use backtracking::{BacktrackingBytePairEncoder, EncodeIter, PretokenCache};
pub use sentencepiece::{EncodeState, SentencePieceBPE};
pub use simple::BytePairEncoder;
pub use unigram::UnigramEncoder;
pub use wordpiece::WordPieceEncoder;

use crate::types::{Split, TokenId};

/// Encoder algorithm type for serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum EncoderType {
    /// Backtracking BPE using Aho-Corasick automata.
    /// Fast O(n) for tiktoken-style tokenizers.
    #[default]
    Backtracking = 0,
    /// Simple O(n²) BPE with greedy merging.
    /// Fast for small pretokenized pieces.
    Simple = 1,
    /// WordPiece tokenization (BERT-style).
    /// Uses greedy longest-match-first.
    WordPiece = 2,
    /// SentencePiece BPE with merge rank checking.
    /// For SentencePiece-style tokenizers (Llama, Mistral, Gemma).
    SentencePiece = 3,
    /// Unigram language model tokenization (SentencePiece).
    /// Uses Viterbi DP to find optimal segmentation (T5, XLM-RoBERTa, ALBERT).
    Unigram = 4,
}

impl EncoderType {
    /// Convert from u32 (for deserialization).
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Backtracking),
            1 => Some(Self::Simple),
            2 => Some(Self::WordPiece),
            3 => Some(Self::SentencePiece),
            4 => Some(Self::Unigram),
            _ => None,
        }
    }
}

/// Unified encoder interface wrapping different tokenization implementations.
///
/// This enum allows the Tokenizer to use different encoding strategies
/// depending on the tokenizer type:
///
/// - `Backtracking`: Fast O(n) for tiktoken-style tokenizers (GPT-2, cl100k, o200k)
/// - `Simple`: Fast O(n²) for small pretokenized pieces (correct for all BPE tokenizers)
/// - `WordPiece`: Greedy longest-match for BERT-style tokenizers
/// - `SentencePiece`: BPE with merge rank checking for SentencePiece-style tokenizers
/// - `Unigram`: Viterbi DP for SentencePiece Unigram models (T5, XLM-RoBERTa, ALBERT)
#[derive(Clone)]
pub enum Encoder {
    /// Greedy matching with backtracking - fast for tiktoken-style.
    Backtracking(BacktrackingBytePairEncoder),
    /// Simple O(n²) BPE - fast for small pretokenized inputs, correct for all.
    Simple(BytePairEncoder),
    /// WordPiece tokenization - greedy longest-match for BERT-style.
    WordPiece(WordPieceEncoder),
    /// SentencePiece BPE with merge rank checking.
    SentencePiece(SentencePieceBPE),
    /// Unigram language model tokenization.
    Unigram(UnigramEncoder),
}

impl Encoder {
    /// Get the encoder type.
    pub fn encoder_type(&self) -> EncoderType {
        match self {
            Encoder::Backtracking(_) => EncoderType::Backtracking,
            Encoder::Simple(_) => EncoderType::Simple,
            Encoder::WordPiece(_) => EncoderType::WordPiece,
            Encoder::SentencePiece(_) => EncoderType::SentencePiece,
            Encoder::Unigram(_) => EncoderType::Unigram,
        }
    }

    /// Encode text into tokens.
    pub fn encode(&self, text: &[u8]) -> Vec<TokenId> {
        match self {
            Encoder::Backtracking(e) => e.encode(text),
            Encoder::Simple(e) => e.encode(text),
            Encoder::WordPiece(e) => e.encode(text),
            Encoder::SentencePiece(e) => e.encode(text),
            Encoder::Unigram(e) => e.encode(text),
        }
    }

    /// Append the encoding of one piece to `out`, using the pretoken cache
    /// where the encoder supports it (Backtracking only).
    #[inline]
    pub fn encode_into(&self, text: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        match self {
            Encoder::Backtracking(e) => e.encode_into(text, cache, out),
            _ => out.extend(self.encode(text)),
        }
    }

    /// Like [`Self::encode_into`], for a `piece` that is a subslice of `doc`
    /// (lets the Backtracking cache build its key with one masked load).
    #[inline]
    pub fn encode_piece_into(&self, doc: &[u8], piece: &[u8], cache: Option<&mut PretokenCache>, out: &mut Vec<TokenId>) {
        match self {
            Encoder::Backtracking(e) => e.encode_piece_into(doc, piece, cache, out),
            _ => out.extend(self.encode(piece)),
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Encoder::Backtracking(e) => e.vocab_size(),
            Encoder::Simple(e) => e.vocab_size(),
            Encoder::WordPiece(e) => e.vocab_size(),
            Encoder::SentencePiece(e) => e.vocab_size(),
            Encoder::Unigram(e) => e.vocab_size(),
        }
    }

    /// Get the number of base tokens.
    pub fn num_base_tokens(&self) -> usize {
        match self {
            Encoder::Backtracking(e) => e.num_base_tokens(),
            Encoder::Simple(e) => e.num_base_tokens(),
            Encoder::WordPiece(e) => e.num_base_tokens(),
            Encoder::SentencePiece(e) => e.num_base_tokens(),
            Encoder::Unigram(e) => e.num_base_tokens(),
        }
    }

    /// Get a reference to the split table.
    ///
    /// Only available for Backtracking encoder (used for serialization).
    pub fn split_table(&self) -> Option<&[Split]> {
        match self {
            Encoder::Backtracking(e) => Some(e.split_table()),
            Encoder::Simple(_) | Encoder::WordPiece(_) | Encoder::SentencePiece(_) | Encoder::Unigram(_) => None,
        }
    }

    /// Returns a streaming iterator over encoded tokens.
    ///
    /// Note: Currently only supported for Backtracking encoder.
    pub fn encode_iter<'a>(&'a self, text: &'a [u8]) -> EncoderIter<'a> {
        match self {
            Encoder::Backtracking(e) => EncoderIter::Backtracking(e.encode_iter(text)),
            Encoder::Simple(_) | Encoder::WordPiece(_) | Encoder::SentencePiece(_) | Encoder::Unigram(_) => {
                // Simple/WordPiece/SentencePiece/Unigram don't have streaming yet - collect all
                EncoderIter::Collected(self.encode(text).into_iter())
            }
        }
    }

    /// Get the underlying backtracking encoder if this is one.
    pub fn as_backtracking(&self) -> Option<&BacktrackingBytePairEncoder> {
        match self {
            Encoder::Backtracking(e) => Some(e),
            _ => None,
        }
    }

    /// Get the underlying simple encoder if this is one.
    pub fn as_simple(&self) -> Option<&BytePairEncoder> {
        match self {
            Encoder::Simple(e) => Some(e),
            _ => None,
        }
    }

    /// Get the underlying wordpiece encoder if this is one.
    pub fn as_wordpiece(&self) -> Option<&WordPieceEncoder> {
        match self {
            Encoder::WordPiece(e) => Some(e),
            _ => None,
        }
    }

    /// Get the underlying sentencepiece encoder if this is one.
    pub fn as_sentencepiece(&self) -> Option<&SentencePieceBPE> {
        match self {
            Encoder::SentencePiece(e) => Some(e),
            _ => None,
        }
    }

    /// Get the underlying unigram encoder if this is one.
    pub fn as_unigram(&self) -> Option<&UnigramEncoder> {
        match self {
            Encoder::Unigram(e) => Some(e),
            _ => None,
        }
    }

    /// Check if two tokens can appear adjacent in a valid encoding.
    pub fn is_valid_pair(&self, token1: TokenId, token2: TokenId) -> bool {
        match self {
            Encoder::Backtracking(e) => e.is_valid_pair(token1, token2),
            Encoder::Simple(e) => e.is_valid_pair(token1, token2),
            Encoder::WordPiece(e) => e.is_valid_pair(token1, token2),
            Encoder::SentencePiece(e) => e.is_valid_pair(token1, token2),
            Encoder::Unigram(e) => e.is_valid_pair(token1, token2),
        }
    }
}

/// Iterator over encoded tokens, abstracting over encoder types.
pub enum EncoderIter<'a> {
    Backtracking(EncodeIter<'a>),
    Collected(std::vec::IntoIter<TokenId>),
}

impl Iterator for EncoderIter<'_> {
    type Item = TokenId;

    fn next(&mut self) -> Option<TokenId> {
        match self {
            EncoderIter::Backtracking(iter) => iter.next(),
            EncoderIter::Collected(iter) => iter.next(),
        }
    }
}

impl std::iter::FusedIterator for EncoderIter<'_> {}
