//! An unrecognized split pattern is refused.
//!
//! The hand-written cuts live in the other files. A pattern that matches
//! none of them is an error. This module does not run it through a regex engine.

/// A split pattern that matches none of the hand-written cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedPattern<'a> {
    /// The pattern text that was refused.
    pub pattern: &'a str,
}

impl std::fmt::Display for UnsupportedPattern<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no pretokenizer for pattern: {}", self.pattern)
    }
}

impl std::error::Error for UnsupportedPattern<'_> {}

/// Refuse `pattern`.
///
/// Every call fails. A known cut is its own type, not a pattern passed here.
pub fn refuse(pattern: &str) -> Result<(), UnsupportedPattern<'_>> {
    Err(UnsupportedPattern { pattern })
}

#[cfg(test)]
mod tests {
    use super::refuse;

    #[test]
    fn an_unknown_pattern_is_refused() {
        let pattern = r"\p{L}+";
        let err = refuse(pattern).expect_err("an unknown pattern is refused");
        assert_eq!(err.pattern, pattern);
        assert_eq!(err.to_string(), "no pretokenizer for pattern: \\p{L}+");
    }
}
