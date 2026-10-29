//! Core types for BPE tokenization

/// Token identifier - corresponds to position in vocabulary
pub type TokenId = u32;

/// Represents the two tokens that were merged to form a token.
/// For base tokens (single bytes), both values point to the token itself.
#[derive(Debug, Clone, Copy)]
pub struct Split {
    pub left: TokenId,
    pub right: TokenId,
}

impl Split {
    /// Create a split for a base token (points to itself)
    pub fn base(id: TokenId) -> Self {
        Self { left: id, right: id }
    }

    /// Create a split from a merge of two tokens
    pub fn merge(left: TokenId, right: TokenId) -> Self {
        Self { left, right }
    }
}
