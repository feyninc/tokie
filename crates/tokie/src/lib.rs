//! Tokenizer. Normalie rewrites the string. Pretokie splits it. This crate
//! maps each span to token ids.

pub mod bpe;
pub mod cache;
pub mod cut;
pub mod norm;
pub mod unigram;
pub mod wordpiece;

mod prepare;
mod tokenizer;

pub use tokenizer::{Error, Specials, Tokenizer};
