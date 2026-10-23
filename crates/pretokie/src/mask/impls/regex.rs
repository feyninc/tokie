//! Regex-based pretokenizer for custom and fallback patterns.
//!
//! Uses `regex-automata` for multi-pattern matching with lookahead simulation.
//! Achieves ~146 MiB/s — use hand-coded iterators (Gpt2, Cl100k, etc.) for
//! higher performance when available.
//!
//! Requires the `regex` feature: `pretokie = { version = "...", features = ["regex"] }`

use regex_automata::{meta, util::captures::Captures, Anchored, Input};

/// Regex-based pretokenizer.
///
/// Handles negative lookahead patterns by rewriting them as multiple patterns
/// with a trailing-character trim.
pub struct Regex {
    inner: meta::Regex,
    lookahead: Vec<bool>,
}

impl std::fmt::Debug for Regex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Regex")
            .field("patterns", &self.inner.pattern_len())
            .finish()
    }
}

impl Regex {
    /// Create a new pretokenizer from patterns.
    ///
    /// Each pattern is a tuple of `(pattern_str, is_lookahead)`.
    /// If `is_lookahead` is true, the last character of matches will be dropped
    /// (simulates negative lookahead like `\s+(?!\S)`).
    pub fn new(patterns: &[(&str, bool)]) -> Result<Self, meta::BuildError> {
        let pats: Vec<&str> = patterns.iter().map(|(p, _)| *p).collect();
        let lookahead: Vec<bool> = patterns.iter().map(|(_, l)| *l).collect();
        let inner = meta::Regex::new_many(&pats)?;
        Ok(Self { inner, lookahead })
    }

    /// Create a CL100K (GPT-3.5/GPT-4) compatible pretokenizer.
    pub fn cl100k() -> Self {
        Self::new(&[
            (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+$", false),
            (r"\s+\s", true),
            (r"\s+", false),
        ]).expect("valid cl100k pattern")
    }

    /// Create an O200K (GPT-4o) compatible pretokenizer.
    pub fn o200k() -> Self {
        let pat1 = [
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"\s*[\r\n]+",
            r"\s+$",
        ]
        .join("|");

        Self::new(&[(&pat1, false), (r"\s+\s", true), (r"\s+", false)])
            .expect("valid o200k pattern")
    }

    /// Create a GPT-2 compatible pretokenizer (regex version).
    ///
    /// Prefer `pretokie::mask::Gpt2` for ~3x better performance.
    pub fn gpt2() -> Self {
        Self::new(&[
            (
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+$",
                false,
            ),
            (r"\s+\s", true),
            (r"\s+", false),
        ])
        .expect("valid GPT-2 pattern")
    }

    /// Split text into pre-tokens.
    pub fn split<'a>(&'a self, text: &'a str) -> RegexIter<'a> {
        RegexIter {
            pretokenizer: self,
            text,
            pos: 0,
            caps: Captures::matches(self.inner.group_info().clone()),
        }
    }

    /// Split text and collect into a Vec.
    pub fn split_to_vec<'a>(&'a self, text: &'a str) -> Vec<&'a str> {
        self.split(text).collect()
    }
}

/// Iterator over pre-tokens from the regex pretokenizer.
pub struct RegexIter<'a> {
    pretokenizer: &'a Regex,
    text: &'a str,
    pos: usize,
    caps: Captures,
}

impl<'a> Iterator for RegexIter<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.text.len() {
            return None;
        }

        let input = Input::new(&self.text[self.pos..]).anchored(Anchored::Yes);
        self.caps.clear();
        self.pretokenizer.inner.captures(input, &mut self.caps);

        let m = self.caps.get_match()?;
        let start = self.pos;
        let mut end = self.pos + m.range().end;

        // If this is a lookahead pattern, drop the last character
        if self.pretokenizer.lookahead[m.pattern().as_usize()] {
            if let Some(last_char) = self.text[start..end].chars().next_back() {
                end -= last_char.len_utf8();
            }
        }

        // Safety: ensure we make progress
        if end <= start {
            self.pos = start + 1;
            return self.next();
        }

        self.pos = end;
        Some(&self.text[start..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cl100k_numbers() {
        let pre = Regex::cl100k();
        let tokens: Vec<_> = pre.split("12345").collect();
        assert_eq!(tokens, vec!["123", "45"]);
    }

    #[test]
    fn test_gpt2_basic() {
        let pre = Regex::gpt2();
        let tokens: Vec<_> = pre.split("Hello world").collect();
        assert_eq!(tokens, vec!["Hello", " world"]);
    }

    #[test]
    fn test_gpt2_contractions() {
        let pre = Regex::gpt2();
        let tokens: Vec<_> = pre.split("How's it going?").collect();
        assert_eq!(tokens, vec!["How", "'s", " it", " going", "?"]);
    }

    #[test]
    fn test_o200k_camelcase() {
        let o200k = Regex::o200k();
        assert_eq!(o200k.split("CamelCase").collect::<Vec<_>>(), vec!["Camel", "Case"]);
        assert_eq!(o200k.split("JSONParser").collect::<Vec<_>>(), vec!["JSONParser"]);
        assert_eq!(o200k.split("parseJSON").collect::<Vec<_>>(), vec!["parse", "JSON"]);
        assert_eq!(o200k.split("XMLHttpRequest").collect::<Vec<_>>(), vec!["XMLHttp", "Request"]);
        assert_eq!(o200k.split("don't").collect::<Vec<_>>(), vec!["don't"]);
        assert_eq!(o200k.split("Hello world").collect::<Vec<_>>(), vec!["Hello", " world"]);
    }
}
