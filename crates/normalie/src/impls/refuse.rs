//! An unrecognized normalizer type is refused.
//!
//! A known formula is its own type. This function does not run one.

/// A normalizer type that matches none of the formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedNormalizer<'a> {
    /// The `type` string that was refused.
    pub type_name: &'a str,
}

impl std::fmt::Display for UnsupportedNormalizer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no normalizer for type: {}", self.type_name)
    }
}

impl std::error::Error for UnsupportedNormalizer<'_> {}

/// Refuse `type_name`.
///
/// Every call fails.
pub fn refuse(type_name: &str) -> Result<(), UnsupportedNormalizer<'_>> {
    Err(UnsupportedNormalizer { type_name })
}
