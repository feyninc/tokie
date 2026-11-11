//! Tokenizer. Normalie rewrites the string. Pretokie splits it. This crate
//! maps each span to token ids.

pub mod bpe;
pub mod cache;
pub mod cut;
mod fold;
pub mod norm;
mod scan;
pub mod unigram;
pub mod wordpiece;

mod file;
mod prepare;
mod tokenizer;

/// Id of one token in a loaded vocab.
pub type TokenId = u32;

pub use tokenizer::{EncodePlus, Error, Plus, Specials, Tokenizer};
