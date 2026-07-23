//! tokie — Fast, correct tokenizer for every HuggingFace model.
//!
//! Supports BPE, WordPiece, SentencePiece, and Unigram. 50x faster than
//! HuggingFace tokenizers, 100% token-accurate.
//!
//! # Quick Start
//!
//! ```ignore
//! use tokie::Tokenizer;
//!
//! let tokenizer = Tokenizer::from_json("tokenizer.json")?;
//!
//! // Encode returns Encoding with ids, attention_mask, type_ids
//! let enc = tokenizer.encode("Hello, world!", true);
//! println!("{:?}", enc.ids);
//!
//! // Decode back
//! let text = tokenizer.decode(&enc.ids).unwrap();
//!
//! // Save/load binary format (~10x smaller, ~5ms load)
//! tokenizer.to_file("model.tkz")?;
//! let tokenizer = Tokenizer::from_file("model.tkz")?;
//! ```
//!
//! # HuggingFace Hub
//!
//! Enable the `hf` feature to load from HuggingFace directly:
//!
//! ```toml
//! tokie = { version = "0.0.4", features = ["hf"] }
//! ```
//!
//! ```ignore
//! let tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
//! let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-3.2-1B")?;
//! ```
//!
//! # Padding & Truncation
//!
//! ```ignore
//! use tokie::{Tokenizer, TruncationParams, PaddingParams, PaddingStrategy};
//!
//! let mut tokenizer = Tokenizer::from_pretrained("bert-base-uncased")?;
//! tokenizer.enable_truncation(TruncationParams { max_length: 128, ..Default::default() });
//! tokenizer.enable_padding(PaddingParams {
//!     strategy: PaddingStrategy::Fixed(128),
//!     ..Default::default()
//! });
//!
//! let results = tokenizer.encode_batch(&["Hello!", "World"], true);
//! // All results are exactly 128 tokens
//! ```

pub mod charsmap;
mod decoder;
pub mod diff;
pub mod encoder;
pub mod hf;
#[cfg(feature = "hf")]
mod hub;
#[cfg(feature = "build")]
pub mod build;
pub mod normalizer;
pub mod padding;
mod postprocessor;
pub mod pretok;
mod pool;
mod serde;
mod tokenizer;
mod types;

pub use charsmap::{CharsmapError, PrecompiledCharsmap};
pub use encoder::{BacktrackingBytePairEncoder, BytePairEncoder, EncodeIter, Encoder, EncoderIter, EncoderType};
pub use decoder::{Decoder, DecoderType, VocabDecoder};
pub use hf::JsonLoadError;
#[cfg(feature = "hf")]
pub use hub::{FromPretrainedOptions, HubError};
#[cfg(feature = "build")]
pub use build::{BuildError, ConvertResult, VerifyResult, Mismatch};
pub use normalizer::{bert_uncased_normalize, clean_text, fnr, metaspace_normalize, strip_accents, FnrFinder, Normalizer};
pub use padding::{Encoding, PaddingParams, PaddingStrategy, PaddingDirection, TruncationParams, TruncationStrategy, TruncationDirection};
pub use postprocessor::PostProcessor;
pub use pretok::{PretokType, Pretokenizer, RegexPretok};
pub use serde::SerdeError;
pub use tokenizer::{EncodingPair, TokenCount, Tokenizer, TokenizeIter};
pub use types::TokenId;

// Backward compatibility aliases
#[doc(hidden)]
#[deprecated(since = "0.2.0", note = "Use PretokType instead")]
pub type PretokenizerType = PretokType;
