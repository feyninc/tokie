//! Prepend `▁`, then turn spaces into `▁`, even when the text starts with a space.

use crate::formula::Formula;
use crate::impls::edit::expand_spaces;

pub struct Llama;

impl Llama {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Llama {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        expand_spaces(text, true, out);
        text.len()
    }
}
