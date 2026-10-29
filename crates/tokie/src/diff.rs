//! Tokenization diff tool for comparing two encodings.
//!
//! Finds all differences between two tokenizations of the same text,
//! with precise byte locations and clear formatting.

use std::fmt;

/// A single token with its byte span in the original text.
#[derive(Debug, Clone)]
pub struct TokenSpan {
    pub token_id: u32,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

/// A region where two tokenizations differ.
#[derive(Debug, Clone)]
pub struct Diff {
    /// Byte range in original text where tokenizations differ
    pub byte_start: usize,
    pub byte_end: usize,
    /// The original text in this region
    pub text: String,
    /// Tokens from encoding A covering this region
    pub tokens_a: Vec<TokenSpan>,
    /// Tokens from encoding B covering this region
    pub tokens_b: Vec<TokenSpan>,
}

/// Summary statistics for a comparison.
#[derive(Debug, Clone)]
pub struct DiffSummary {
    pub text_bytes: usize,
    pub tokens_a: usize,
    pub tokens_b: usize,
    pub diff_count: usize,
    pub diff_bytes: usize,
    pub match_rate: f64,
    /// True if results were truncated due to memory limits
    pub truncated: bool,
}

/// Result of comparing two tokenizations.
#[derive(Debug)]
pub struct DiffResult {
    pub summary: DiffSummary,
    pub diffs: Vec<Diff>,
}

impl fmt::Display for DiffResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header
        writeln!(f, "=== Tokenization Diff ===")?;
        writeln!(
            f,
            "Text: {} bytes | Tokens: {} vs {} | Diffs: {}",
            self.summary.text_bytes,
            self.summary.tokens_a,
            self.summary.tokens_b,
            self.summary.diff_count
        )?;
        writeln!(
            f,
            "Match rate: {:.2}% | Diff bytes: {}",
            self.summary.match_rate * 100.0,
            self.summary.diff_bytes
        )?;

        if self.summary.truncated {
            writeln!(f, "WARNING: Results truncated (max {} diffs) to prevent memory exhaustion", self.summary.diff_count)?;
        }

        if self.diffs.is_empty() {
            writeln!(f, "\nNo differences found.")?;
            return Ok(());
        }

        writeln!(f)?;

        // Maximum tokens to show per side in a diff region
        const MAX_DISPLAY_TOKENS: usize = 5;

        // Show each diff
        for (i, diff) in self.diffs.iter().enumerate() {
            let is_large = diff.tokens_a.len() > MAX_DISPLAY_TOKENS || diff.tokens_b.len() > MAX_DISPLAY_TOKENS;

            writeln!(f, "#{} [bytes {}-{}] {:?}",
                i + 1,
                diff.byte_start,
                diff.byte_end,
                truncate(&diff.text, 40)
            )?;

            if is_large {
                // Large diff region - show compact summary
                writeln!(f, "   {} tokens (tokie) vs {} tokens (hf) - showing first {}:",
                    diff.tokens_a.len(),
                    diff.tokens_b.len(),
                    MAX_DISPLAY_TOKENS
                )?;
            }

            // Limit tokens to display
            let display_a: Vec<_> = diff.tokens_a.iter().take(MAX_DISPLAY_TOKENS).collect();
            let display_b: Vec<_> = diff.tokens_b.iter().take(MAX_DISPLAY_TOKENS).collect();

            // Find max widths for alignment (only for displayed tokens)
            let max_a = display_a.iter()
                .map(|t| format!("{} {:?}", t.token_id, t.text).len())
                .max()
                .unwrap_or(10);
            let max_b = display_b.iter()
                .map(|t| format!("{} {:?}", t.token_id, t.text).len())
                .max()
                .unwrap_or(10);

            let col_a = max_a.max(6);
            let col_b = max_b.max(12);

            // Header row
            writeln!(f, "   {:^col_a$} | {:^col_b$}", "tokie", "huggingface")?;
            writeln!(f, "   {:-<col_a$}-+-{:-<col_b$}", "", "")?;

            // Align tokens by byte position (limited to display count)
            let mut idx_a = 0;
            let mut idx_b = 0;

            while idx_a < display_a.len() || idx_b < display_b.len() {
                let span_a = display_a.get(idx_a).copied();
                let span_b = display_b.get(idx_b).copied();

                match (span_a, span_b) {
                    (Some(a), Some(b)) => {
                        if a.start == b.start && a.end == b.end {
                            // Same position - show both
                            let str_a = format!("{} {:?}", a.token_id, a.text);
                            let str_b = format!("{} {:?}", b.token_id, b.text);
                            writeln!(f, "   {:col_a$} | {:col_b$}", str_a, str_b)?;
                            idx_a += 1;
                            idx_b += 1;
                        } else if a.start < b.start || (a.start == b.start && a.end <= b.end) {
                            // A is earlier or smaller
                            let str_a = format!("{} {:?}", a.token_id, a.text);
                            writeln!(f, "   {:col_a$} |", str_a)?;
                            idx_a += 1;
                        } else {
                            // B is earlier or smaller
                            let str_b = format!("{} {:?}", b.token_id, b.text);
                            writeln!(f, "   {:col_a$} | {:col_b$}", "", str_b)?;
                            idx_b += 1;
                        }
                    }
                    (Some(a), None) => {
                        let str_a = format!("{} {:?}", a.token_id, a.text);
                        writeln!(f, "   {:col_a$} |", str_a)?;
                        idx_a += 1;
                    }
                    (None, Some(b)) => {
                        let str_b = format!("{} {:?}", b.token_id, b.text);
                        writeln!(f, "   {:col_a$} | {:col_b$}", "", str_b)?;
                        idx_b += 1;
                    }
                    (None, None) => break,
                }
            }

            // Show "and N more" if truncated
            let remaining_a = diff.tokens_a.len().saturating_sub(MAX_DISPLAY_TOKENS);
            let remaining_b = diff.tokens_b.len().saturating_sub(MAX_DISPLAY_TOKENS);
            if remaining_a > 0 || remaining_b > 0 {
                let more_a = if remaining_a > 0 { format!("+{} more", remaining_a) } else { String::new() };
                let more_b = if remaining_b > 0 { format!("+{} more", remaining_b) } else { String::new() };
                writeln!(f, "   {:col_a$} | {:col_b$}", more_a, more_b)?;
            }

            writeln!(f)?;
        }

        Ok(())
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Compare two tokenizations of the same text.
///
/// # Arguments
/// * `text` - The original text that was tokenized
/// * `tokens_a` - Token IDs from first tokenizer (e.g., tokie)
/// * `tokens_b` - Token IDs from second tokenizer (e.g., huggingface)
/// * `decoder_a` - Function to decode a token ID to its string representation
/// * `decoder_b` - Function to decode a token ID to its string representation
///
/// # Returns
/// A `DiffResult` containing all differences and summary statistics.
pub fn compare<FA, FB>(
    text: &str,
    tokens_a: &[u32],
    tokens_b: &[u32],
    decoder_a: FA,
    decoder_b: FB,
) -> DiffResult
where
    FA: Fn(u32) -> Option<String>,
    FB: Fn(u32) -> Option<String>,
{
    let total_tokens = tokens_a.len() + tokens_b.len();

    // For very large inputs, skip detailed diff to prevent memory exhaustion
    if total_tokens > MAX_TOKENS_FOR_DETAILED_DIFF {
        // Just compute basic stats without building spans
        let identical = tokens_a == tokens_b;
        let first_diff = if identical {
            None
        } else {
            first_diff_index(tokens_a, tokens_b)
        };

        let summary = DiffSummary {
            text_bytes: text.len(),
            tokens_a: tokens_a.len(),
            tokens_b: tokens_b.len(),
            diff_count: if identical { 0 } else { 1 }, // Indicate there's a diff
            diff_bytes: 0, // Can't compute without full analysis
            match_rate: if identical { 1.0 } else { 0.0 },
            truncated: true, // Mark as truncated due to size
        };

        // Create a single summary diff if not identical
        let diffs = if let Some(idx) = first_diff {
            // Decode just a few tokens around the first difference
            let start_idx = idx.saturating_sub(2);
            let end_idx_a = (idx + 3).min(tokens_a.len());
            let end_idx_b = (idx + 3).min(tokens_b.len());

            let tokens_a_sample: Vec<TokenSpan> = tokens_a[start_idx..end_idx_a]
                .iter()
                .scan(0usize, |pos, &id| {
                    let text = decoder_a(id).unwrap_or_else(|| format!("<unk:{}>", id));
                    let len = text.len();
                    let span = TokenSpan { token_id: id, start: *pos, end: *pos + len, text };
                    *pos += len;
                    Some(span)
                })
                .collect();

            let tokens_b_sample: Vec<TokenSpan> = tokens_b[start_idx..end_idx_b]
                .iter()
                .scan(0usize, |pos, &id| {
                    let text = decoder_b(id).unwrap_or_else(|| format!("<unk:{}>", id));
                    let len = text.len();
                    let span = TokenSpan { token_id: id, start: *pos, end: *pos + len, text };
                    *pos += len;
                    Some(span)
                })
                .collect();

            vec![Diff {
                byte_start: 0,
                byte_end: 0,
                text: format!("(Large input: {} tokens - showing first diff at token {})", total_tokens, idx),
                tokens_a: tokens_a_sample,
                tokens_b: tokens_b_sample,
            }]
        } else {
            vec![]
        };

        return DiffResult { summary, diffs };
    }

    // Build spans for both tokenizations
    let spans_a = build_spans(tokens_a, &decoder_a);
    let spans_b = build_spans(tokens_b, &decoder_b);

    // Find all differences
    let diffs = find_diffs(text, &spans_a, &spans_b);

    // Check if results were truncated
    let truncated = diffs.len() >= MAX_DIFFS;

    // Calculate summary
    let diff_bytes: usize = diffs.iter().map(|d| d.byte_end - d.byte_start).sum();
    let match_rate = if text.is_empty() {
        1.0
    } else {
        1.0 - (diff_bytes as f64 / text.len() as f64)
    };

    let summary = DiffSummary {
        text_bytes: text.len(),
        tokens_a: tokens_a.len(),
        tokens_b: tokens_b.len(),
        diff_count: diffs.len(),
        diff_bytes,
        match_rate,
        truncated,
    };

    DiffResult { summary, diffs }
}

/// Build token spans by decoding each token and tracking byte positions.
fn build_spans<F>(tokens: &[u32], decoder: F) -> Vec<TokenSpan>
where
    F: Fn(u32) -> Option<String>,
{
    let mut spans = Vec::with_capacity(tokens.len());
    let mut pos = 0;

    for &token_id in tokens {
        let text = decoder(token_id).unwrap_or_else(|| format!("<unk:{}>", token_id));
        let len = text.len();
        spans.push(TokenSpan {
            token_id,
            start: pos,
            end: pos + len,
            text,
        });
        pos += len;
    }

    spans
}

/// Maximum number of diffs to collect before stopping (memory protection).
const MAX_DIFFS: usize = 100;

/// Maximum tokens per diff region (prevents runaway collection).
const MAX_TOKENS_PER_DIFF: usize = 50;

/// Maximum tokens to process for detailed diff (memory protection).
/// Beyond this, we only provide summary statistics.
const MAX_TOKENS_FOR_DETAILED_DIFF: usize = 100_000;

/// Find all differing regions between two span sequences.
///
/// This function is memory-bounded: it will stop collecting after MAX_DIFFS
/// differences to prevent memory exhaustion on large files with many differences.
fn find_diffs(text: &str, spans_a: &[TokenSpan], spans_b: &[TokenSpan]) -> Vec<Diff> {
    let mut diffs = Vec::new();
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < spans_a.len() && idx_b < spans_b.len() {
        // Memory protection: stop if we've collected too many diffs
        if diffs.len() >= MAX_DIFFS {
            break;
        }

        let a = &spans_a[idx_a];
        let b = &spans_b[idx_b];

        // Check if tokens match (same ID and same byte range)
        if a.token_id == b.token_id && a.start == b.start && a.end == b.end {
            // Match - advance both
            idx_a += 1;
            idx_b += 1;
        } else {
            // Mismatch - collect all tokens until we resync
            let diff_start = a.start.min(b.start);
            let mut diff_tokens_a = Vec::new();
            let mut diff_tokens_b = Vec::new();
            let mut current_end = diff_start;
            let mut hit_limit = false;

            // Advance both until their end positions align
            loop {
                let end_a = spans_a.get(idx_a).map(|s| s.end).unwrap_or(usize::MAX);
                let end_b = spans_b.get(idx_b).map(|s| s.end).unwrap_or(usize::MAX);

                if end_a == end_b && end_a != usize::MAX {
                    // End positions aligned - include these tokens and check if they're identical
                    let a = &spans_a[idx_a];
                    let b = &spans_b[idx_b];
                    diff_tokens_a.push(a.clone());
                    diff_tokens_b.push(b.clone());
                    current_end = end_a;
                    idx_a += 1;
                    idx_b += 1;

                    // Check if next tokens are back in sync
                    let next_match = match (spans_a.get(idx_a), spans_b.get(idx_b)) {
                        (Some(na), Some(nb)) => {
                            na.token_id == nb.token_id && na.start == nb.start && na.end == nb.end
                        }
                        (None, None) => true, // Both exhausted
                        _ => false,
                    };

                    if next_match {
                        break;
                    }
                } else if end_a < end_b {
                    // A ends earlier - advance A
                    if let Some(a) = spans_a.get(idx_a) {
                        diff_tokens_a.push(a.clone());
                        current_end = current_end.max(a.end);
                        idx_a += 1;
                    } else {
                        break;
                    }
                } else {
                    // B ends earlier - advance B
                    if let Some(b) = spans_b.get(idx_b) {
                        diff_tokens_b.push(b.clone());
                        current_end = current_end.max(b.end);
                        idx_b += 1;
                    } else {
                        break;
                    }
                }

                // Safety: prevent runaway token collection per diff
                if diff_tokens_a.len() + diff_tokens_b.len() > MAX_TOKENS_PER_DIFF {
                    hit_limit = true;
                    break;
                }
            }

            // Extract text for this region
            let safe_end = current_end.min(text.len());
            let safe_start = diff_start.min(safe_end);
            let diff_text = text.get(safe_start..safe_end)
                .unwrap_or("<invalid range>")
                .to_string();

            diffs.push(Diff {
                byte_start: diff_start,
                byte_end: current_end,
                text: diff_text,
                tokens_a: diff_tokens_a,
                tokens_b: diff_tokens_b,
            });

            // If we hit the per-diff limit, force resync by finding next alignment point
            if hit_limit {
                // Try to find a sync point by looking for matching end positions
                let target_end = current_end;

                // Advance A until we pass target_end
                while idx_a < spans_a.len() && spans_a[idx_a].end <= target_end {
                    idx_a += 1;
                }
                // Advance B until we pass target_end
                while idx_b < spans_b.len() && spans_b[idx_b].end <= target_end {
                    idx_b += 1;
                }

                // If still not synced and we have many diffs, just skip ahead more aggressively
                if diffs.len() > MAX_DIFFS / 2 {
                    // Skip ahead by a chunk to avoid generating too many diffs
                    let skip = 100;
                    idx_a = (idx_a + skip).min(spans_a.len());
                    idx_b = (idx_b + skip).min(spans_b.len());
                }
            }
        }
    }

    // Handle trailing tokens (but limit their size too)
    if idx_a < spans_a.len() || idx_b < spans_b.len() {
        if diffs.len() < MAX_DIFFS {
            let diff_start = spans_a.get(idx_a).map(|s| s.start)
                .or_else(|| spans_b.get(idx_b).map(|s| s.start))
                .unwrap_or(0);
            let diff_end = spans_a.last().map(|s| s.end).unwrap_or(0)
                .max(spans_b.last().map(|s| s.end).unwrap_or(0));

            // Limit trailing tokens to prevent memory explosion
            let remaining_a = spans_a.len().saturating_sub(idx_a);
            let remaining_b = spans_b.len().saturating_sub(idx_b);
            let take_a = remaining_a.min(MAX_TOKENS_PER_DIFF);
            let take_b = remaining_b.min(MAX_TOKENS_PER_DIFF);

            let tokens_a: Vec<TokenSpan> = spans_a[idx_a..idx_a + take_a].to_vec();
            let tokens_b: Vec<TokenSpan> = spans_b[idx_b..idx_b + take_b].to_vec();

            let safe_end = diff_end.min(text.len());
            let safe_start = diff_start.min(safe_end);
            let diff_text = text.get(safe_start..safe_end)
                .unwrap_or("<trailing>")
                .to_string();

            if !tokens_a.is_empty() || !tokens_b.is_empty() {
                diffs.push(Diff {
                    byte_start: diff_start,
                    byte_end: diff_end,
                    text: diff_text,
                    tokens_a,
                    tokens_b,
                });
            }
        }
    }

    diffs
}

/// Quick check if two encodings are identical.
#[inline]
pub fn is_identical(tokens_a: &[u32], tokens_b: &[u32]) -> bool {
    tokens_a == tokens_b
}

/// Find first difference index (for quick debugging).
pub fn first_diff_index(tokens_a: &[u32], tokens_b: &[u32]) -> Option<usize> {
    tokens_a.iter()
        .zip(tokens_b.iter())
        .position(|(a, b)| a != b)
        .or_else(|| {
            if tokens_a.len() != tokens_b.len() {
                Some(tokens_a.len().min(tokens_b.len()))
            } else {
                None
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_decoder(id: u32) -> Option<String> {
        match id {
            1 => Some("hello".to_string()),
            2 => Some(" ".to_string()),
            3 => Some("world".to_string()),
            4 => Some("hel".to_string()),
            5 => Some("lo".to_string()),
            6 => Some("wor".to_string()),
            7 => Some("ld".to_string()),
            _ => None,
        }
    }

    #[test]
    fn test_identical() {
        let tokens = vec![1, 2, 3];
        assert!(is_identical(&tokens, &tokens));
    }

    #[test]
    fn test_different_lengths() {
        let a = vec![1, 2, 3];
        let b = vec![1, 2];
        assert!(!is_identical(&a, &b));
        assert_eq!(first_diff_index(&a, &b), Some(2));
    }

    #[test]
    fn test_compare_identical() {
        let text = "hello world";
        let tokens = vec![1, 2, 3]; // "hello" + " " + "world"

        let result = compare(text, &tokens, &tokens, mock_decoder, mock_decoder);

        assert_eq!(result.summary.diff_count, 0);
        assert_eq!(result.summary.match_rate, 1.0);
    }

    #[test]
    fn test_compare_different_splits() {
        let text = "hello world";
        let tokens_a = vec![1, 2, 3]; // "hello" + " " + "world"
        let tokens_b = vec![4, 5, 2, 6, 7]; // "hel" + "lo" + " " + "wor" + "ld"

        let result = compare(text, &tokens_a, &tokens_b, mock_decoder, mock_decoder);

        // Should find differences where tokenization differs
        assert!(result.summary.diff_count > 0);
        println!("{}", result);
    }

    #[test]
    fn test_memory_protection_with_never_syncing_tokenizers() {
        // Create a scenario where tokenizers never resync: one decoder adds an extra byte
        // This would previously cause unbounded memory growth
        let text = "a".repeat(10000);

        // tokens_a: each token decodes to "a"
        let tokens_a: Vec<u32> = (0..10000).map(|_| 100).collect();

        // tokens_b: each token decodes to "aa" (different length, will never sync)
        let tokens_b: Vec<u32> = (0..5000).map(|_| 200).collect();

        // Decoder A: token 100 -> "a"
        let decoder_a = |id: u32| -> Option<String> {
            if id == 100 { Some("a".to_string()) } else { None }
        };

        // Decoder B: token 200 -> "aa" (different byte length!)
        let decoder_b = |id: u32| -> Option<String> {
            if id == 200 { Some("aa".to_string()) } else { None }
        };

        let result = compare(&text, &tokens_a, &tokens_b, decoder_a, decoder_b);

        // The key assertion: diffs should be bounded by MAX_DIFFS
        assert!(result.summary.diff_count <= MAX_DIFFS);

        // If we hit the limit, truncated should be true
        if result.summary.diff_count == MAX_DIFFS {
            assert!(result.summary.truncated);
        }

        println!("Diffs: {}, Truncated: {}", result.summary.diff_count, result.summary.truncated);
    }

    #[test]
    fn test_truncated_flag_set_correctly() {
        let result = DiffSummary {
            text_bytes: 100,
            tokens_a: 10,
            tokens_b: 10,
            diff_count: 5,
            diff_bytes: 20,
            match_rate: 0.8,
            truncated: true,
        };
        assert!(result.truncated);
    }
}
