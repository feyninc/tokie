//! Fast pretokenizers for BPE tokenizers.
//!
//! Each pretokenizer is a zero-allocation, single-pass iterator over text pieces.
//!
//! # Example
//!
//! ```
//! use pretokie::Gpt2;
//!
//! let pieces: Vec<&str> = Gpt2::new("Hello world").collect();
//! assert_eq!(pieces, vec!["Hello", " world"]);
//! ```

mod core;
mod configs;
mod impls;
pub mod util;

pub use core::iter::Core;
pub use core::mask::Mask;
pub use configs::{Gpt2Config, Cl100kConfig, O200kConfig, VoyageConfig, SmolLMConfig, DeepSeekConfig, QwenConfig};

pub type Gpt2<'a> = Mask<'a, Gpt2Config>;
pub type Cl100k<'a> = Mask<'a, Cl100kConfig>;
pub type O200k<'a> = Mask<'a, O200kConfig>;
pub type Voyage<'a> = Mask<'a, VoyageConfig>;
pub type SmolLM<'a> = Mask<'a, SmolLMConfig>;
pub type DeepSeek<'a> = Mask<'a, DeepSeekConfig>;
pub type Qwen<'a> = Mask<'a, QwenConfig>;

pub use impls::bert::Bert;
#[cfg(feature = "regex")]
pub mod regex {
    //! Regex-based pretokenizer (requires `regex` feature).
    pub use crate::impls::regex::{Regex, RegexIter};
}
#[cfg(feature = "regex")]
pub use impls::regex::Regex;
