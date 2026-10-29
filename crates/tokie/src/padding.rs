//! Padding and truncation support for tokenizer output.

use crate::types::TokenId;

/// Result of encoding text, with token IDs, attention mask, and type IDs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoding {
    /// Token IDs.
    pub ids: Vec<TokenId>,
    /// Attention mask: 1 for real tokens, 0 for padding.
    pub attention_mask: Vec<u8>,
    /// Token type IDs: 0 for first sequence, 1 for second sequence.
    pub type_ids: Vec<u8>,
    /// Byte offsets: (start, end) for each token in the original text.
    /// Only populated when encoding with offsets. Empty otherwise.
    pub offsets: Vec<(usize, usize)>,
}

impl Encoding {
    /// Create an encoding from token IDs (attention_mask all 1s, type_ids all 0s).
    pub fn from_ids(ids: Vec<TokenId>) -> Self {
        let len = ids.len();
        Self {
            ids,
            attention_mask: vec![1u8; len],
            type_ids: vec![0u8; len],
            offsets: Vec::new(),
        }
    }

    /// Create an encoding from token IDs with byte offsets.
    pub fn from_ids_with_offsets(ids: Vec<TokenId>, offsets: Vec<(usize, usize)>) -> Self {
        let len = ids.len();
        Self {
            ids,
            attention_mask: vec![1u8; len],
            type_ids: vec![0u8; len],
            offsets,
        }
    }

    /// Create an encoding from pair processing output.
    pub fn from_pair(ids: Vec<TokenId>, type_ids: Vec<u8>) -> Self {
        let len = ids.len();
        Self {
            ids,
            attention_mask: vec![1u8; len],
            type_ids,
            offsets: Vec::new(),
        }
    }

    /// Number of tokens in this encoding.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether this encoding is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

// --- Padding ---

/// How to determine the target padding length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    /// Pad to the longest sequence in the batch.
    BatchLongest,
    /// Pad to a fixed length.
    Fixed(usize),
}

/// Which side to add padding tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingDirection {
    /// Pad on the left (prepend).
    Left,
    /// Pad on the right (append).
    #[default]
    Right,
}

/// Padding configuration.
#[derive(Debug, Clone)]
pub struct PaddingParams {
    pub strategy: PaddingStrategy,
    pub direction: PaddingDirection,
    pub pad_to_multiple_of: Option<usize>,
    pub pad_id: TokenId,
    pub pad_type_id: u8,
}

impl Default for PaddingParams {
    fn default() -> Self {
        Self {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
        }
    }
}

// --- Truncation ---

/// Strategy for truncating pairs of sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationStrategy {
    /// Alternately remove from the longer sequence.
    #[default]
    LongestFirst,
    /// Only truncate the first sequence.
    OnlyFirst,
    /// Only truncate the second sequence.
    OnlySecond,
}

/// Which end to remove tokens from when truncating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationDirection {
    /// Remove tokens from the beginning.
    Left,
    /// Remove tokens from the end.
    #[default]
    Right,
}

/// Truncation configuration.
#[derive(Debug, Clone)]
pub struct TruncationParams {
    pub max_length: usize,
    pub strategy: TruncationStrategy,
    pub direction: TruncationDirection,
    pub stride: usize,
}

impl Default for TruncationParams {
    fn default() -> Self {
        Self {
            max_length: 512,
            strategy: TruncationStrategy::LongestFirst,
            direction: TruncationDirection::Right,
            stride: 0,
        }
    }
}

// --- Free functions ---

/// Truncate a token sequence in place.
pub fn truncate_ids(ids: &mut Vec<TokenId>, max_len: usize, direction: TruncationDirection) {
    if ids.len() <= max_len {
        return;
    }
    match direction {
        TruncationDirection::Right => ids.truncate(max_len),
        TruncationDirection::Left => {
            let start = ids.len() - max_len;
            ids.drain(..start);
        }
    }
}

/// Truncate a pair of token sequences to fit within `max_total` tokens.
pub fn truncate_pair(
    tokens_a: &mut Vec<TokenId>,
    tokens_b: &mut Vec<TokenId>,
    max_total: usize,
    strategy: TruncationStrategy,
    direction: TruncationDirection,
) {
    let total = tokens_a.len() + tokens_b.len();
    if total <= max_total {
        return;
    }
    let to_remove = total - max_total;

    match strategy {
        TruncationStrategy::LongestFirst => {
            // Alternately trim the longer sequence
            let mut remaining = to_remove;
            while remaining > 0 {
                if tokens_a.len() >= tokens_b.len() {
                    let trim = remaining.min(tokens_a.len().saturating_sub(tokens_b.len()).max(1));
                    let trim = trim.min(remaining);
                    match direction {
                        TruncationDirection::Right => {
                            tokens_a.truncate(tokens_a.len() - trim);
                        }
                        TruncationDirection::Left => {
                            tokens_a.drain(..trim);
                        }
                    }
                    remaining -= trim;
                } else {
                    let trim = remaining.min(tokens_b.len().saturating_sub(tokens_a.len()).max(1));
                    let trim = trim.min(remaining);
                    match direction {
                        TruncationDirection::Right => {
                            tokens_b.truncate(tokens_b.len() - trim);
                        }
                        TruncationDirection::Left => {
                            tokens_b.drain(..trim);
                        }
                    }
                    remaining -= trim;
                }
            }
        }
        TruncationStrategy::OnlyFirst => {
            truncate_ids(tokens_a, tokens_a.len().saturating_sub(to_remove), direction);
        }
        TruncationStrategy::OnlySecond => {
            truncate_ids(tokens_b, tokens_b.len().saturating_sub(to_remove), direction);
        }
    }
}

/// Pad an encoding to `target_len`.
pub fn pad_encoding(encoding: &mut Encoding, target_len: usize, params: &PaddingParams) {
    if encoding.len() >= target_len {
        return;
    }
    let pad_count = target_len - encoding.len();

    match params.direction {
        PaddingDirection::Right => {
            encoding.ids.extend(std::iter::repeat_n(params.pad_id, pad_count));
            encoding.attention_mask.extend(std::iter::repeat_n(0u8, pad_count));
            encoding.type_ids.extend(std::iter::repeat_n(params.pad_type_id, pad_count));
            if !encoding.offsets.is_empty() {
                encoding.offsets.extend(std::iter::repeat_n((0, 0), pad_count));
            }
        }
        PaddingDirection::Left => {
            let mut new_ids = vec![params.pad_id; pad_count];
            new_ids.append(&mut encoding.ids);
            encoding.ids = new_ids;

            let mut new_mask = vec![0u8; pad_count];
            new_mask.append(&mut encoding.attention_mask);
            encoding.attention_mask = new_mask;

            let mut new_type_ids = vec![params.pad_type_id; pad_count];
            new_type_ids.append(&mut encoding.type_ids);
            encoding.type_ids = new_type_ids;

            if !encoding.offsets.is_empty() {
                let mut new_offsets = vec![(0, 0); pad_count];
                new_offsets.append(&mut encoding.offsets);
                encoding.offsets = new_offsets;
            }
        }
    }
}

/// Compute the target padding length for a batch.
pub fn compute_pad_target(encodings: &[Encoding], params: &PaddingParams) -> usize {
    let max_len = match &params.strategy {
        PaddingStrategy::BatchLongest => encodings.iter().map(|e| e.len()).max().unwrap_or(0),
        PaddingStrategy::Fixed(n) => *n,
    };

    if let Some(multiple) = params.pad_to_multiple_of {
        if multiple > 0 {
            return ((max_len + multiple - 1) / multiple) * multiple;
        }
    }

    max_len
}

/// Pad a batch of encodings.
pub fn pad_batch(encodings: &mut [Encoding], params: &PaddingParams) {
    let target = compute_pad_target(encodings, params);
    for encoding in encodings.iter_mut() {
        pad_encoding(encoding, target, params);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_from_ids() {
        let enc = Encoding::from_ids(vec![1, 2, 3]);
        assert_eq!(enc.ids, vec![1, 2, 3]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1]);
        assert_eq!(enc.type_ids, vec![0, 0, 0]);
        assert_eq!(enc.len(), 3);
    }

    #[test]
    fn test_encoding_from_pair() {
        let enc = Encoding::from_pair(vec![101, 1, 102, 2, 102], vec![0, 0, 0, 1, 1]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_truncate_right() {
        let mut ids = vec![1, 2, 3, 4, 5];
        truncate_ids(&mut ids, 3, TruncationDirection::Right);
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_left() {
        let mut ids = vec![1, 2, 3, 4, 5];
        truncate_ids(&mut ids, 3, TruncationDirection::Left);
        assert_eq!(ids, vec![3, 4, 5]);
    }

    #[test]
    fn test_truncate_noop_when_short() {
        let mut ids = vec![1, 2];
        truncate_ids(&mut ids, 5, TruncationDirection::Right);
        assert_eq!(ids, vec![1, 2]);
    }

    #[test]
    fn test_truncate_pair_longest_first() {
        // a=5 tokens, b=3 tokens, max_total=6 → remove 2 from a
        let mut a = vec![1, 2, 3, 4, 5];
        let mut b = vec![10, 20, 30];
        truncate_pair(&mut a, &mut b, 6, TruncationStrategy::LongestFirst, TruncationDirection::Right);
        assert_eq!(a, vec![1, 2, 3]);
        assert_eq!(b, vec![10, 20, 30]);
    }

    #[test]
    fn test_truncate_pair_longest_first_balanced() {
        // a=5, b=5, max_total=6 → remove 4 total, alternating
        let mut a = vec![1, 2, 3, 4, 5];
        let mut b = vec![10, 20, 30, 40, 50];
        truncate_pair(&mut a, &mut b, 6, TruncationStrategy::LongestFirst, TruncationDirection::Right);
        assert_eq!(a.len() + b.len(), 6);
        assert_eq!(a, vec![1, 2, 3]);
        assert_eq!(b, vec![10, 20, 30]);
    }

    #[test]
    fn test_truncate_pair_only_first() {
        let mut a = vec![1, 2, 3, 4, 5];
        let mut b = vec![10, 20, 30];
        truncate_pair(&mut a, &mut b, 5, TruncationStrategy::OnlyFirst, TruncationDirection::Right);
        assert_eq!(a, vec![1, 2]);
        assert_eq!(b, vec![10, 20, 30]);
    }

    #[test]
    fn test_truncate_pair_only_second() {
        let mut a = vec![1, 2, 3];
        let mut b = vec![10, 20, 30, 40, 50];
        truncate_pair(&mut a, &mut b, 5, TruncationStrategy::OnlySecond, TruncationDirection::Right);
        assert_eq!(a, vec![1, 2, 3]);
        assert_eq!(b, vec![10, 20]);
    }

    #[test]
    fn test_pad_right() {
        let mut enc = Encoding::from_ids(vec![1, 2, 3]);
        let params = PaddingParams { pad_id: 0, ..Default::default() };
        pad_encoding(&mut enc, 5, &params);
        assert_eq!(enc.ids, vec![1, 2, 3, 0, 0]);
        assert_eq!(enc.attention_mask, vec![1, 1, 1, 0, 0]);
        assert_eq!(enc.type_ids, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_pad_left() {
        let mut enc = Encoding::from_ids(vec![1, 2, 3]);
        let params = PaddingParams {
            direction: PaddingDirection::Left,
            pad_id: 0,
            ..Default::default()
        };
        pad_encoding(&mut enc, 5, &params);
        assert_eq!(enc.ids, vec![0, 0, 1, 2, 3]);
        assert_eq!(enc.attention_mask, vec![0, 0, 1, 1, 1]);
    }

    #[test]
    fn test_pad_noop_when_long_enough() {
        let mut enc = Encoding::from_ids(vec![1, 2, 3]);
        let params = PaddingParams::default();
        pad_encoding(&mut enc, 2, &params);
        assert_eq!(enc.ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_pad_to_multiple_of() {
        let encodings = vec![
            Encoding::from_ids(vec![1, 2, 3]),      // len 3
            Encoding::from_ids(vec![1, 2, 3, 4, 5]), // len 5
        ];
        let params = PaddingParams {
            pad_to_multiple_of: Some(8),
            ..Default::default()
        };
        let target = compute_pad_target(&encodings, &params);
        assert_eq!(target, 8); // 5 rounded up to 8
    }

    #[test]
    fn test_pad_batch() {
        let mut encodings = vec![
            Encoding::from_ids(vec![1, 2]),
            Encoding::from_ids(vec![1, 2, 3, 4]),
            Encoding::from_ids(vec![1]),
        ];
        let params = PaddingParams { pad_id: 0, ..Default::default() };
        pad_batch(&mut encodings, &params);
        assert!(encodings.iter().all(|e| e.len() == 4));
        assert_eq!(encodings[0].attention_mask, vec![1, 1, 0, 0]);
        assert_eq!(encodings[1].attention_mask, vec![1, 1, 1, 1]);
        assert_eq!(encodings[2].attention_mask, vec![1, 0, 0, 0]);
    }

    #[test]
    fn test_pad_batch_fixed() {
        let mut encodings = vec![
            Encoding::from_ids(vec![1, 2]),
            Encoding::from_ids(vec![1, 2, 3]),
        ];
        let params = PaddingParams {
            strategy: PaddingStrategy::Fixed(6),
            pad_id: 0,
            ..Default::default()
        };
        pad_batch(&mut encodings, &params);
        assert!(encodings.iter().all(|e| e.len() == 6));
    }
}
