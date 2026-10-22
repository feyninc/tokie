//! Charsmap, then pad ASCII punctuation and write metaspace.

use crate::formula::Formula;
use crate::impls::precompiled::Precompiled;

pub struct PunctPad {
    map: Precompiled,
}

impl PunctPad {
    /// Parse the charsmap this formula runs first.
    pub fn parse(blob: &[u8]) -> Result<Self, crate::CharsmapError> {
        Ok(Self {
            map: Precompiled::parse(blob)?,
        })
    }

    pub fn write(&self, text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(self, text, at_start, out)
    }
}

impl Formula for PunctPad {
    fn drive(&self, text: &str, at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        let mut mapped = String::new();
        let src = if self.map.write(text, at_start, &mut mapped) {
            mapped.as_str()
        } else {
            text
        };
        let stripped = src.trim();
        if stripped.is_empty() {
            return true;
        }
        if !stripped.starts_with('▁') {
            out.push('▁');
        }
        let mut pending = false;
        let mut emitted = false;
        for c in stripped.chars() {
            if c.is_whitespace() {
                pending = true;
            } else if c.is_ascii_punctuation() {
                if emitted {
                    out.push('▁');
                }
                out.push(c);
                emitted = true;
                pending = true;
            } else {
                if pending {
                    out.push('▁');
                    pending = false;
                }
                out.push(c);
                emitted = true;
            }
        }
        let _ = emitted;
        true
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        let _ = text;
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, _out: &mut String) -> usize {
        text.len().max(1)
    }
}
