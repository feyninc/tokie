//! Drop combining marks. Precomposed letters stay.
//!
//! The mark set is Unicode 9, which is what Hugging Face `StripAccents` uses.

use crate::formula::Formula;
use crate::marks::is_unicode9_mark;

pub struct StripAccents;

impl StripAccents {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for StripAccents {
    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        if text.chars().any(is_unicode9_mark) {
            None
        } else {
            Some(text.len())
        }
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        for c in text.chars() {
            if !is_unicode9_mark(c) {
                out.push(c);
            }
        }
        text.len()
    }
}
