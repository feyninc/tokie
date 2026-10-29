//! Unicode lowercase, one character at a time.
//!
//! ASCII `A..=Z` is the byte map. Every other letter uses `char::to_lowercase`.

use crate::formula::{Formula, Window};

pub struct Lower;

impl Lower {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Lower {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] < 0x80 {
                let start = i;
                while i < bytes.len() && bytes[i] < 0x80 {
                    i += 1;
                }
                for &b in &bytes[start..i] {
                    let b = if b.is_ascii_uppercase() { b | 0x20 } else { b };
                    out.push(b as char);
                }
            } else {
                let c = text[i..].chars().next().unwrap();
                for lower in c.to_lowercase() {
                    out.push(lower);
                }
                i += c.len_utf8();
            }
        }
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], _at_start: bool) -> Option<Window> {
        let _ = ISA;
        let mut n = 0;
        while n < bytes.len() {
            let b = bytes[n];
            if b >= 0x80 || b.is_ascii_uppercase() {
                break;
            }
            n += 1;
        }
        if n == 0 { None } else { Some(Window::Copy(n)) }
    }
}
