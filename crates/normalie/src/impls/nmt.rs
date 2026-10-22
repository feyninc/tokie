//! SentencePiece NMT fold, as its own step.

use crate::formula::{Formula, Window};
use crate::scan::printable_prefix;

pub struct Nmt;

impl Nmt {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Nmt {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        for c in text.chars() {
            if let Some(mapped) = map_nmt(c) {
                out.push(mapped);
            }
        }
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], _at_start: bool) -> Option<Window> {
        let n = printable_prefix::<ISA>(bytes);
        if n == 0 { None } else { Some(Window::Copy(n)) }
    }
}

/// `None` removes the scalar. `Some` is the scalar that replaces it.
fn map_nmt(c: char) -> Option<char> {
    match c as u32 {
        0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F => None,
        0x0009 | 0x000A | 0x000C | 0x000D | 0x1680 | 0x200B..=0x200F | 0x2028 | 0x2029 | 0x2581
        | 0xFEFF | 0xFFFD => Some(' '),
        _ => Some(c),
    }
}
