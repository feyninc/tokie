//! VocabDecoder for converting token IDs back to bytes.
//!
//! This module provides a standalone decoder that can be shared across
//! different encoder types (BPE, WordPiece, Unigram, etc.).
//!
//! Uses a flat buffer layout for cache-efficient decoding:
//! - All token bytes stored in a single contiguous buffer
//! - Offset array for O(1) token lookup
//! - No pointer chasing during decode

use crate::types::TokenId;

/// VocabDecoder for converting token IDs back to bytes.
///
/// Uses a flat buffer layout for optimal cache performance:
/// - `data`: All token bytes concatenated in a single allocation
/// - `offsets`: Start offset of each token (offsets\[i+1\] - offsets\[i\] = token length)
///
/// This eliminates pointer chasing and improves decode throughput.
///
/// # Example
/// ```ignore
/// let decoder = VocabDecoder::new(token_bytes);
/// let text = decoder.decode(&[100, 200, 300]);
/// ```
#[derive(Clone)]
pub struct VocabDecoder {
    /// All token bytes concatenated.
    data: Vec<u8>,
    /// Start offset of each token. Length = vocab_size + 1.
    /// Token i spans data[offsets[i]..offsets[i+1]].
    offsets: Vec<u32>,
}

impl VocabDecoder {
    /// Create a decoder from pre-built parts (used for deserialization).
    pub fn from_parts(data: Vec<u8>, offsets: Vec<u32>) -> Self {
        Self { data, offsets }
    }

    /// Get references to the internal data and offsets.
    pub fn as_parts(&self) -> (&[u8], &[u32]) {
        (&self.data, &self.offsets)
    }

    /// Create a new decoder from token byte sequences.
    ///
    /// Converts the input to a flat buffer layout for efficient decoding.
    pub fn new(token_bytes: Vec<Vec<u8>>) -> Self {
        // Compute total size
        let total_size: usize = token_bytes.iter().map(|t| t.len()).sum();

        // Build flat buffer and offsets
        let mut data = Vec::with_capacity(total_size);
        let mut offsets = Vec::with_capacity(token_bytes.len() + 1);

        for token in &token_bytes {
            offsets.push(data.len() as u32);
            data.extend_from_slice(token);
        }
        offsets.push(data.len() as u32);

        Self { data, offsets }
    }

    /// Get the vocabulary size.
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Get the byte sequence for a token.
    #[inline]
    pub fn token_to_bytes(&self, token: TokenId) -> &[u8] {
        let start = self.offsets[token as usize] as usize;
        let end = self.offsets[token as usize + 1] as usize;
        &self.data[start..end]
    }

    /// Get the length of a token in bytes.
    #[inline]
    pub fn token_len(&self, token: TokenId) -> usize {
        let start = self.offsets[token as usize];
        let end = self.offsets[token as usize + 1];
        (end - start) as usize
    }

    /// Minimum tokens to trigger parallel decoding.
    const PARALLEL_THRESHOLD: usize = 50_000;

    /// Decode a sequence of tokens back to bytes.
    ///
    /// Uses parallel decoding for large token sequences.
    pub fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        if tokens.is_empty() {
            return Vec::new();
        }

        // Use parallel decoding for large sequences
        let num_cpus = crate::tokenizer::num_cpus();

        if tokens.len() >= Self::PARALLEL_THRESHOLD && num_cpus > 1 {
            self.decode_parallel(tokens, num_cpus)
        } else {
            self.decode_sequential(tokens)
        }
    }

    /// Sequential decode for small token sequences.
    fn decode_sequential(&self, tokens: &[TokenId]) -> Vec<u8> {
        // Compute total size
        let total_size: usize = tokens
            .iter()
            .map(|&t| self.token_len(t))
            .sum();

        // Pre-allocate and copy
        let mut result = Vec::with_capacity(total_size);
        for &token in tokens {
            result.extend_from_slice(self.token_to_bytes(token));
        }
        result
    }

    /// Parallel decode for large token sequences.
    /// Each thread decodes its chunk, then results are concatenated.
    fn decode_parallel(&self, tokens: &[TokenId], num_threads: usize) -> Vec<u8> {
        use rayon::prelude::*;

        // Split tokens into chunks
        let chunk_size = (tokens.len() + num_threads - 1) / num_threads;
        let chunks: Vec<&[TokenId]> = tokens.chunks(chunk_size).collect();

        // Decode chunks in parallel on tokie's pool, each returning its own Vec
        let (data, offsets) = (&self.data, &self.offsets);
        let results: Vec<Vec<u8>> = crate::tokenizer::pool().install(|| chunks
            .into_par_iter()
            .with_max_len(1)
            .map(|chunk| {
                let size: usize = chunk
                    .iter()
                    .map(|&t| {
                        let idx = t as usize;
                        (offsets[idx + 1] - offsets[idx]) as usize
                    })
                    .sum();
                let mut result = Vec::with_capacity(size);
                for &token in chunk {
                    let t = token as usize;
                    result.extend_from_slice(&data[offsets[t] as usize..offsets[t + 1] as usize]);
                }
                result
            })
            .collect());

        // Concatenate results
        let total_size: usize = results.iter().map(|v| v.len()).sum();
        let mut output = Vec::with_capacity(total_size);
        for chunk in results {
            output.extend_from_slice(&chunk);
        }
        output
    }

    /// Decode tokens to a UTF-8 string.
    ///
    /// Returns `None` if the decoded bytes are not valid UTF-8.
    pub fn decode_to_string(&self, tokens: &[TokenId]) -> Option<String> {
        String::from_utf8(self.decode(tokens)).ok()
    }

    /// Get token bytes as a Vec for compatibility.
    ///
    /// Note: This allocates new vectors. For read-only access, use `token_to_bytes`.
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        (0..self.vocab_size())
            .map(|i| self.token_to_bytes(i as TokenId).to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_empty() {
        let decoder = VocabDecoder::new(vec![vec![b'a'], vec![b'b']]);
        assert_eq!(decoder.decode(&[]), Vec::<u8>::new());
    }

    #[test]
    fn test_decode_single() {
        let decoder = VocabDecoder::new(vec![vec![b'a'], vec![b'b'], vec![b'c']]);
        assert_eq!(decoder.decode(&[0]), b"a");
        assert_eq!(decoder.decode(&[1]), b"b");
        assert_eq!(decoder.decode(&[2]), b"c");
    }

    #[test]
    fn test_decode_multiple() {
        let decoder = VocabDecoder::new(vec![
            vec![b'H', b'e', b'l', b'l', b'o'],
            vec![b' '],
            vec![b'w', b'o', b'r', b'l', b'd'],
        ]);
        assert_eq!(decoder.decode(&[0, 1, 2]), b"Hello world");
    }

    #[test]
    fn test_decode_to_string() {
        let decoder = VocabDecoder::new(vec![vec![b'a'], vec![b'b'], vec![b'c']]);
        assert_eq!(decoder.decode_to_string(&[0, 1, 2]), Some("abc".to_string()));
    }

    #[test]
    fn test_vocab_size() {
        let decoder = VocabDecoder::new(vec![vec![b'a'], vec![b'b'], vec![b'c']]);
        assert_eq!(decoder.vocab_size(), 3);
    }

    #[test]
    fn test_token_to_bytes() {
        let decoder = VocabDecoder::new(vec![vec![b'a', b'b'], vec![b'c', b'd', b'e']]);
        assert_eq!(decoder.token_to_bytes(0), b"ab");
        assert_eq!(decoder.token_to_bytes(1), b"cde");
    }

    #[test]
    fn test_token_len() {
        let decoder = VocabDecoder::new(vec![vec![b'a'], vec![b'a', b'b'], vec![b'a', b'b', b'c']]);
        assert_eq!(decoder.token_len(0), 1);
        assert_eq!(decoder.token_len(1), 2);
        assert_eq!(decoder.token_len(2), 3);
    }

    #[test]
    fn test_parallel_decode_matches_sequential() {
        // Create decoder with some tokens
        let token_bytes: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let decoder = VocabDecoder::new(token_bytes);

        // Create a large token sequence (above parallel threshold)
        let tokens: Vec<TokenId> = (0..100_000).map(|i| (i % 256) as TokenId).collect();

        // Get results from both paths
        let sequential = decoder.decode_sequential(&tokens);
        let parallel = decoder.decode_parallel(&tokens, 4);

        assert_eq!(sequential.len(), parallel.len(), "Length mismatch");
        assert_eq!(sequential, parallel, "Content mismatch");
    }

    #[test]
    fn test_parallel_decode_chunk_boundaries() {
        // Test with varying token lengths to verify chunk boundaries are correct
        let token_bytes = vec![
            vec![b'A'],           // 1 byte
            vec![b'B', b'B'],     // 2 bytes
            vec![b'C', b'C', b'C'], // 3 bytes
        ];
        let decoder = VocabDecoder::new(token_bytes);

        // Create pattern that will span chunk boundaries
        let tokens: Vec<TokenId> = (0..60_000).map(|i| (i % 3) as TokenId).collect();

        let sequential = decoder.decode_sequential(&tokens);
        let parallel = decoder.decode_parallel(&tokens, 4);

        assert_eq!(sequential, parallel);
    }
}
