//! Post-processing for tokenizers.
//!
//! Post-processors add special tokens after encoding. Common patterns include:
//! - BERT: `[CLS] tokens [SEP]` for single, `[CLS] A [SEP] B [SEP]` for pairs
//! - LLaMA: `<|begin_of_text|> tokens` for single
//! - GPT-2: No special tokens
//!
//! # Example
//!
//! ```ignore
//! use tokie::PostProcessor;
//!
//! // BERT-style post-processor
//! let pp = PostProcessor::bert(101, 102); // CLS=101, SEP=102
//! let tokens = vec![7592]; // "hello"
//! let processed = pp.process(&tokens);
//! assert_eq!(processed, vec![101, 7592, 102]); // [CLS] hello [SEP]
//! ```

use crate::types::TokenId;

/// Post-processor configuration.
///
/// Post-processors add special tokens to encoded sequences.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PostProcessor {
    /// No post-processing (GPT-2, most BPE tokenizers).
    #[default]
    None,

    /// BERT-style: `[CLS] A [SEP]` for single, `[CLS] A [SEP] B [SEP]` for pairs.
    Bert {
        cls_token: TokenId,
        sep_token: TokenId,
    },

    /// Single prefix token (LLaMA 3 style): `<bos> A` for single.
    Prefix {
        bos_token: TokenId,
    },

    /// Generic template-based post-processor.
    /// Tokens are inserted at specified positions.
    Template {
        /// Tokens to prepend to single sequences.
        single_prefix: Vec<TokenId>,
        /// Tokens to append to single sequences.
        single_suffix: Vec<TokenId>,
        /// Tokens to prepend to the first sequence in pairs.
        pair_a_prefix: Vec<TokenId>,
        /// Tokens between sequences in pairs.
        pair_a_suffix: Vec<TokenId>,
        /// Tokens to prepend to the second sequence in pairs.
        pair_b_prefix: Vec<TokenId>,
        /// Tokens to append to the second sequence in pairs.
        pair_b_suffix: Vec<TokenId>,
    },
}

impl PostProcessor {
    /// Create a BERT-style post-processor.
    ///
    /// - Single: `[CLS] A [SEP]`
    /// - Pair: `[CLS] A [SEP] B [SEP]`
    pub fn bert(cls_token: TokenId, sep_token: TokenId) -> Self {
        Self::Bert { cls_token, sep_token }
    }

    /// Create a prefix-only post-processor (LLaMA 3 style).
    ///
    /// - Single: `<bos> A`
    /// - Pair: `<bos> A <bos> B`
    pub fn prefix(bos_token: TokenId) -> Self {
        Self::Prefix { bos_token }
    }

    /// Check if this post-processor adds any special tokens.
    pub fn is_none(&self) -> bool {
        matches!(self, PostProcessor::None)
    }

    /// Check if a token ID is a special token added by this post-processor.
    pub fn is_special_token(&self, id: TokenId) -> bool {
        match self {
            PostProcessor::None => false,
            PostProcessor::Bert { cls_token, sep_token } => id == *cls_token || id == *sep_token,
            PostProcessor::Prefix { bos_token } => id == *bos_token,
            PostProcessor::Template {
                single_prefix, single_suffix,
                pair_a_prefix, pair_a_suffix,
                pair_b_prefix, pair_b_suffix,
            } => {
                single_prefix.contains(&id) || single_suffix.contains(&id) ||
                pair_a_prefix.contains(&id) || pair_a_suffix.contains(&id) ||
                pair_b_prefix.contains(&id) || pair_b_suffix.contains(&id)
            }
        }
    }

    /// Process a single sequence by adding special tokens.
    pub fn process(&self, tokens: &[TokenId]) -> Vec<TokenId> {
        match self {
            PostProcessor::None => tokens.to_vec(),

            PostProcessor::Bert { cls_token, sep_token } => {
                let mut result = Vec::with_capacity(tokens.len() + 2);
                result.push(*cls_token);
                result.extend_from_slice(tokens);
                result.push(*sep_token);
                result
            }

            PostProcessor::Prefix { bos_token } => {
                let mut result = Vec::with_capacity(tokens.len() + 1);
                result.push(*bos_token);
                result.extend_from_slice(tokens);
                result
            }

            PostProcessor::Template {
                single_prefix,
                single_suffix,
                ..
            } => {
                let mut result = Vec::with_capacity(
                    single_prefix.len() + tokens.len() + single_suffix.len()
                );
                result.extend_from_slice(single_prefix);
                result.extend_from_slice(tokens);
                result.extend_from_slice(single_suffix);
                result
            }
        }
    }

    /// Process a pair of sequences by adding special tokens.
    ///
    /// Returns (tokens, type_ids) where type_ids indicate which sequence
    /// each token belongs to (0 for first, 1 for second).
    pub fn process_pair(
        &self,
        tokens_a: &[TokenId],
        tokens_b: &[TokenId],
    ) -> (Vec<TokenId>, Vec<u8>) {
        match self {
            PostProcessor::None => {
                let mut tokens = Vec::with_capacity(tokens_a.len() + tokens_b.len());
                tokens.extend_from_slice(tokens_a);
                tokens.extend_from_slice(tokens_b);

                let mut type_ids = vec![0u8; tokens_a.len()];
                type_ids.extend(vec![1u8; tokens_b.len()]);

                (tokens, type_ids)
            }

            PostProcessor::Bert { cls_token, sep_token } => {
                // [CLS] A [SEP] B [SEP]
                let mut tokens = Vec::with_capacity(tokens_a.len() + tokens_b.len() + 3);
                tokens.push(*cls_token);
                tokens.extend_from_slice(tokens_a);
                tokens.push(*sep_token);
                tokens.extend_from_slice(tokens_b);
                tokens.push(*sep_token);

                // Type IDs: 0 for [CLS], A, [SEP]; 1 for B, [SEP]
                let mut type_ids = vec![0u8; 1 + tokens_a.len() + 1];
                type_ids.extend(vec![1u8; tokens_b.len() + 1]);

                (tokens, type_ids)
            }

            PostProcessor::Prefix { bos_token } => {
                // <bos> A <bos> B
                let mut tokens = Vec::with_capacity(tokens_a.len() + tokens_b.len() + 2);
                tokens.push(*bos_token);
                tokens.extend_from_slice(tokens_a);
                tokens.push(*bos_token);
                tokens.extend_from_slice(tokens_b);

                let mut type_ids = vec![0u8; 1 + tokens_a.len()];
                type_ids.extend(vec![1u8; 1 + tokens_b.len()]);

                (tokens, type_ids)
            }

            PostProcessor::Template {
                pair_a_prefix,
                pair_a_suffix,
                pair_b_prefix,
                pair_b_suffix,
                ..
            } => {
                let total_len = pair_a_prefix.len()
                    + tokens_a.len()
                    + pair_a_suffix.len()
                    + pair_b_prefix.len()
                    + tokens_b.len()
                    + pair_b_suffix.len();

                let mut tokens = Vec::with_capacity(total_len);
                tokens.extend_from_slice(pair_a_prefix);
                tokens.extend_from_slice(tokens_a);
                tokens.extend_from_slice(pair_a_suffix);
                tokens.extend_from_slice(pair_b_prefix);
                tokens.extend_from_slice(tokens_b);
                tokens.extend_from_slice(pair_b_suffix);

                let type_0_len = pair_a_prefix.len() + tokens_a.len() + pair_a_suffix.len();
                let type_1_len = pair_b_prefix.len() + tokens_b.len() + pair_b_suffix.len();

                let mut type_ids = vec![0u8; type_0_len];
                type_ids.extend(vec![1u8; type_1_len]);

                (tokens, type_ids)
            }
        }
    }

    /// Get the number of special tokens added for a single sequence.
    pub fn num_special_tokens_single(&self) -> usize {
        match self {
            PostProcessor::None => 0,
            PostProcessor::Bert { .. } => 2, // [CLS] + [SEP]
            PostProcessor::Prefix { .. } => 1, // <bos>
            PostProcessor::Template { single_prefix, single_suffix, .. } => {
                single_prefix.len() + single_suffix.len()
            }
        }
    }

    /// Get the number of special tokens added for a pair of sequences.
    pub fn num_special_tokens_pair(&self) -> usize {
        match self {
            PostProcessor::None => 0,
            PostProcessor::Bert { .. } => 3, // [CLS] + [SEP] + [SEP]
            PostProcessor::Prefix { .. } => 2, // <bos> + <bos>
            PostProcessor::Template {
                pair_a_prefix,
                pair_a_suffix,
                pair_b_prefix,
                pair_b_suffix,
                ..
            } => {
                pair_a_prefix.len() + pair_a_suffix.len()
                    + pair_b_prefix.len() + pair_b_suffix.len()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_none_postprocessor() {
        let pp = PostProcessor::None;
        let tokens = vec![1, 2, 3];
        assert_eq!(pp.process(&tokens), vec![1, 2, 3]);
    }

    #[test]
    fn test_bert_single() {
        let pp = PostProcessor::bert(101, 102);
        let tokens = vec![7592]; // "hello"
        assert_eq!(pp.process(&tokens), vec![101, 7592, 102]);
    }

    #[test]
    fn test_bert_pair() {
        let pp = PostProcessor::bert(101, 102);
        let tokens_a = vec![7592]; // "hello"
        let tokens_b = vec![2088]; // "world"

        let (tokens, type_ids) = pp.process_pair(&tokens_a, &tokens_b);

        assert_eq!(tokens, vec![101, 7592, 102, 2088, 102]);
        assert_eq!(type_ids, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn test_prefix_single() {
        let pp = PostProcessor::prefix(128000);
        let tokens = vec![9906]; // "Hello"
        assert_eq!(pp.process(&tokens), vec![128000, 9906]);
    }

    #[test]
    fn test_prefix_pair() {
        let pp = PostProcessor::prefix(128000);
        let tokens_a = vec![9906]; // "Hello"
        let tokens_b = vec![4435]; // "World"

        let (tokens, type_ids) = pp.process_pair(&tokens_a, &tokens_b);

        assert_eq!(tokens, vec![128000, 9906, 128000, 4435]);
        assert_eq!(type_ids, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_num_special_tokens() {
        assert_eq!(PostProcessor::None.num_special_tokens_single(), 0);
        assert_eq!(PostProcessor::None.num_special_tokens_pair(), 0);

        let bert = PostProcessor::bert(101, 102);
        assert_eq!(bert.num_special_tokens_single(), 2);
        assert_eq!(bert.num_special_tokens_pair(), 3);

        let prefix = PostProcessor::prefix(128000);
        assert_eq!(prefix.num_special_tokens_single(), 1);
        assert_eq!(prefix.num_special_tokens_pair(), 2);
    }
}
