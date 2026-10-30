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

pub use tokenizer::{Error, Specials, Tokenizer};
