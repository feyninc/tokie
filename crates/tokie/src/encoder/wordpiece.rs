//! WordPiece encoder using Aho-Corasick automata with custom failure links.
//!
//! WordPiece tokenization (used by BERT, DistilBERT, etc.) uses greedy
//! longest-match-first (Maximum Matching) instead of BPE merge rules.
//!
//! Key differences from BPE:
//! - No merge rules - vocabulary is pre-defined
//! - Continuation tokens use "##" prefix (configurable)
//! - Unknown tokens return [UNK] token ID

use daggrs::{DoubleArrayAhoCorasick, Trie};
use foldhash::HashMap as FoldHashMap;

use super::backtracking::key_words_within;
use super::PretokenCache;
use crate::types::TokenId;

/// Default continuation prefix for WordPiece (BERT-style).
pub const DEFAULT_CONTINUATION_PREFIX: &[u8] = b"##";

/// WordPiece encoder using DAAC with custom failure links.
///
/// This encoder builds a trie/DAAC where:
/// - All vocabulary tokens are added as patterns
/// - Failure links from completed tokens point to the continuation anchor ("##")
/// - Tokenization uses greedy longest-match with anchor jumping
#[derive(Clone)]
pub struct WordPieceEncoder {
    /// The DAAC automaton with WordPiece failure links.
    matcher: DoubleArrayAhoCorasick,

    /// Token ID for unknown/untokenizable words.
    unk_token: TokenId,

    /// Continuation prefix (e.g., b"##").
    continuation_prefix: Vec<u8>,

    /// Total vocabulary size.
    vocab_size: usize,

    /// Single-byte fast path: direct array lookup (no hashing).
    byte_lut: [TokenId; 256],

    /// Multi-byte token cache for early exit.
    token_cache: FoldHashMap<Vec<u8>, TokenId>,

    /// Maximum number of characters per word before falling back to UNK.
    /// HuggingFace default is 100. Words longer than this return `[unk_token]`.
    max_input_chars_per_word: usize,
}

impl WordPieceEncoder {
    /// Create a WordPiece encoder from vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Vocabulary mapping token bytes to token IDs
    /// * `unk_token` - Token ID to return for unknown words
    /// * `continuation_prefix` - Prefix for continuation tokens (default: "##")
    pub fn from_vocab(
        vocab: &[(Vec<u8>, TokenId)],
        unk_token: TokenId,
        continuation_prefix: &[u8],
        max_input_chars_per_word: usize,
    ) -> Self {
        let mut trie = Trie::new();

        // Add all vocabulary tokens to the trie
        for (bytes, token_id) in vocab {
            trie.add(bytes, *token_id);
        }

        // Build with WordPiece mode (failure links -> anchor)
        trie.build_wordpiece(continuation_prefix);

        // Compile to DAAC for fast matching
        let matcher = trie.compile();

        // Build single-byte fast path array
        let mut byte_lut = [unk_token; 256];
        for (bytes, token_id) in vocab {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = *token_id;
            }
        }

        // Build multi-byte token cache
        let mut token_cache = FoldHashMap::default();
        for (bytes, token_id) in vocab {
            if bytes.len() <= 16 {
                token_cache.insert(bytes.clone(), *token_id);
            }
        }

        Self {
            matcher,
            unk_token,
            continuation_prefix: continuation_prefix.to_vec(),
            vocab_size: vocab.len(),
            byte_lut,
            token_cache,
            max_input_chars_per_word,
        }
    }

    /// Create a WordPiece encoder from vocabulary with default "##" prefix and max 100 chars.
    pub fn from_vocab_default(vocab: &[(Vec<u8>, TokenId)], unk_token: TokenId) -> Self {
        Self::from_vocab(vocab, unk_token, DEFAULT_CONTINUATION_PREFIX, 100)
    }

    /// Create a WordPiece encoder from pre-built components (for fast deserialization).
    ///
    /// This avoids rebuilding the DAAC from scratch.
    pub fn from_parts(
        matcher: DoubleArrayAhoCorasick,
        unk_token: TokenId,
        continuation_prefix: Vec<u8>,
        vocab_size: usize,
        token_bytes: &[Vec<u8>],
        max_input_chars_per_word: usize,
    ) -> Self {
        // Build single-byte fast path array
        let mut byte_lut = [unk_token; 256];
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() == 1 {
                byte_lut[bytes[0] as usize] = token_id as TokenId;
            }
        }

        // Build early exit cache
        let mut token_cache = FoldHashMap::default();
        for (token_id, bytes) in token_bytes.iter().enumerate() {
            if bytes.len() <= 16 {
                token_cache.insert(bytes.clone(), token_id as TokenId);
            }
        }

        Self {
            matcher,
            unk_token,
            continuation_prefix,
            vocab_size,
            byte_lut,
            token_cache,
            max_input_chars_per_word,
        }
    }

    /// Encode a word (pre-tokenized piece) into token IDs.
    ///
    /// Uses longest-match-first (MaxMatch) algorithm with rewind for WordPiece.
    /// Returns `[unk_token]` if the word cannot be fully tokenized.
    pub fn encode(&self, word: &[u8]) -> Vec<TokenId> {
        let mut out = Vec::new();
        self.encode_word_into(word, &mut out);
        out
    }

    /// Append the encoding of one pretokenized `piece` (a subslice of
    /// `doc`) to `out`, memoizing pieces in `cache` (inline entries for
    /// short few-token pieces, the long-piece cache for the rest).
    #[inline]
    pub fn encode_piece_into(
        &self,
        doc: &[u8],
        piece: &[u8],
        cache: Option<&mut PretokenCache>,
        out: &mut Vec<TokenId>,
    ) {
        match cache {
            Some(c) if !piece.is_empty() && piece.len() <= PretokenCache::KEY_MAX => {
                let (lo, hi) = key_words_within(doc, piece);
                if c.get_with_key(lo, hi, out) {
                    return;
                }
                if c.get_long(piece, out) {
                    return;
                }
                let start = out.len();
                self.encode_word_into(piece, out);
                let toks = &out[start..];
                if toks.len() <= PretokenCache::MAX_TOKENS {
                    if !toks.is_empty() {
                        c.insert_with_key(lo, hi, toks);
                    }
                } else {
                    c.insert_long(piece, toks);
                }
            }
            Some(c) if !piece.is_empty() && piece.len() <= PretokenCache::LONG_KEY_MAX => {
                if c.get_long(piece, out) {
                    return;
                }
                let start = out.len();
                self.encode_word_into(piece, out);
                c.insert_long(piece, &out[start..]);
            }
            _ => self.encode_word_into(piece, out),
        }
    }

    /// [`Self::encode`] appending into `out` (no per-word allocation).
    pub fn encode_word_into(&self, word: &[u8], out: &mut Vec<TokenId>) {
        if word.is_empty() {
            return;
        }

        // HF WordPiece: words exceeding max_input_chars_per_word → UNK
        // Count UTF-8 characters, not bytes
        if word.len() > self.max_input_chars_per_word {
            let char_count = std::str::from_utf8(word)
                .map(|s| s.chars().count())
                .unwrap_or(word.len());
            if char_count > self.max_input_chars_per_word {
                out.push(self.unk_token);
                return;
            }
        }

        // Single-byte fast path: direct array lookup (no hashing)
        if word.len() == 1 {
            out.push(self.byte_lut[word[0] as usize]);
            return;
        }

        // Early exit for cached tokens
        if let Some(&token_id) = self.token_cache.get(word) {
            out.push(token_id);
            return;
        }

        // Get anchor state for continuation matching
        let anchor = match self.matcher.anchor {
            Some(a) => a,
            None => {
                out.push(self.unk_token);
                return;
            }
        };

        // On failure the partial result is truncated back to `start` and
        // replaced by a single UNK, as the whole word is untokenizable.
        let start = out.len();
        let mut pos = 0usize;
        let mut state = self.matcher.start_state();
        let mut last_match: Option<(usize, TokenId)> = None;

        loop {
            // Inner loop: process characters until end of word or transition failure
            while pos < word.len() {
                if let Some(next_state) = self.try_transition(state, word[pos]) {
                    state = next_state;
                    pos += 1;

                    // Record match if this state has an output
                    if let Some(output) = self.matcher.outputs(state).next() {
                        last_match = Some((pos, output.pattern_id));
                    }
                } else {
                    // Transition failed - emit last match or return UNK
                    if let Some((end_pos, token_id)) = last_match.take() {
                        out.push(token_id);
                        pos = end_pos;
                        state = anchor;
                    } else {
                        out.truncate(start);
                        out.push(self.unk_token);
                        return;
                    }
                }
            }

            // End of word reached - emit pending match if any
            if let Some((end_pos, token_id)) = last_match.take() {
                out.push(token_id);

                // If match doesn't cover all remaining chars, rewind and continue
                // This handles cases like "grippe" → ["grip", "##pe"] where
                // transitions for "gripp"/"grippe" succeed without outputs
                if end_pos < word.len() {
                    pos = end_pos;
                    state = anchor;
                    continue;
                }
            }

            break;
        }

        if out.len() == start {
            out.push(self.unk_token);
        }
    }

    /// Try to transition to next state, returns None if no valid transition.
    #[inline]
    fn try_transition(&self, state: u32, byte: u8) -> Option<u32> {
        let states = &self.matcher.states;
        let current = &states[state as usize];
        let child = current.base ^ (byte as u32);

        if (child as usize) < states.len() && states[child as usize].check == state {
            Some(child)
        } else {
            None
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the unknown token ID.
    pub fn unk_token(&self) -> TokenId {
        self.unk_token
    }

    /// Get the continuation prefix.
    pub fn continuation_prefix(&self) -> &[u8] {
        &self.continuation_prefix
    }

    /// Get a reference to the underlying DAAC matcher.
    pub fn matcher(&self) -> &DoubleArrayAhoCorasick {
        &self.matcher
    }

    /// Get the maximum input characters per word.
    pub fn max_input_chars_per_word(&self) -> usize {
        self.max_input_chars_per_word
    }

    /// Check if two tokens can appear adjacent.
    /// For WordPiece, this is always true (no merge constraints).
    pub fn is_valid_pair(&self, _token1: TokenId, _token2: TokenId) -> bool {
        true
    }

    /// Number of base tokens (for compatibility - WordPiece doesn't have this concept).
    pub fn num_base_tokens(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vec<(Vec<u8>, TokenId)> {
        vec![
            (b"[UNK]".to_vec(), 0),
            (b"un".to_vec(), 1),
            (b"break".to_vec(), 2),
            (b"##break".to_vec(), 3),
            (b"##able".to_vec(), 4),
            (b"##ing".to_vec(), 5),
        ]
    }

    #[test]
    fn test_wordpiece_piece_cache_matches_uncached() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);
        let doc = b"unbreakable breaking zzz un unbreakable breaking zzz unbreakablex";
        let mut cache = PretokenCache::new();
        for _ in 0..2 {
            let (mut cached, mut plain) = (Vec::new(), Vec::new());
            for piece in doc.split(|&b| b == b' ') {
                encoder.encode_piece_into(doc, piece, Some(&mut cache), &mut cached);
                encoder.encode_piece_into(doc, piece, None, &mut plain);
                assert_eq!(encoder.encode(piece), {
                    let mut v = Vec::new();
                    encoder.encode_word_into(piece, &mut v);
                    v
                });
            }
            assert_eq!(cached, plain);
        }
    }

    #[test]
    fn test_wordpiece_single_token() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "un" should encode to [1]
        let tokens = encoder.encode(b"un");
        assert_eq!(tokens, vec![1]);
    }

    #[test]
    fn test_wordpiece_continuation() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "unbreakable" should encode to ["un", "##break", "##able"] = [1, 3, 4]
        let tokens = encoder.encode(b"unbreakable");
        assert_eq!(tokens, vec![1, 3, 4]);
    }

    #[test]
    fn test_wordpiece_unknown() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "xyz" is not in vocab, should return [UNK]
        let tokens = encoder.encode(b"xyz");
        assert_eq!(tokens, vec![0]);
    }

    #[test]
    fn test_wordpiece_empty() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        let tokens = encoder.encode(b"");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_wordpiece_partial_unknown() {
        let vocab = make_test_vocab();
        let encoder = WordPieceEncoder::from_vocab_default(&vocab, 0);

        // "unxyz" - "un" matches but "xyz" doesn't have continuation
        let tokens = encoder.encode(b"unxyz");
        assert_eq!(tokens, vec![0]); // Should return [UNK]
    }
}
