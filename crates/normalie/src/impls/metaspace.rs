//! Metaspace. Spaces become `▁`. Whitespace-dropping is `MetaspaceWords`.

use crate::formula::{Formula, Window};
use crate::impls::edit::expand_spaces;

const MARK: &str = "▁";
const MARK_B: [u8; 3] = [0xE2, 0x96, 0x81];

pub struct Metaspace;
pub struct MetaspaceFirst;
pub struct MetaspaceNever;
pub struct MetaspaceWords;

impl Metaspace {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl MetaspaceFirst {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl MetaspaceNever {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl MetaspaceWords {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Metaspace {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, at_start: bool, out: &mut String) -> usize {
        expand_spaces(text, at_start && needs_prepend(text.as_bytes()), out);
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], at_start: bool) -> Option<Window> {
        let _ = ISA;
        if at_start && needs_prepend(bytes) {
            return None;
        }
        copy_until_space(bytes)
    }
}

impl Formula for MetaspaceFirst {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, at_start: bool, out: &mut String) -> usize {
        expand_spaces(text, at_start && needs_prepend(text.as_bytes()), out);
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], at_start: bool) -> Option<Window> {
        let _ = ISA;
        if at_start && needs_prepend(bytes) {
            return None;
        }
        copy_until_space(bytes)
    }
}

impl Formula for MetaspaceNever {
    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        expand_spaces(text, false, out);
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], _at_start: bool) -> Option<Window> {
        let _ = ISA;
        copy_until_space(bytes)
    }
}

impl Formula for MetaspaceWords {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        let bytes = text.as_bytes();
        out.reserve(text.len() + text.len() / 8 + MARK.len());
        let mut i = 0;
        let mut in_word = false;
        let mut saw_ws = false;
        while i < bytes.len() {
            let b = bytes[i];
            if b < 0x80 {
                if is_ascii_ws(b) {
                    saw_ws = true;
                    in_word = false;
                    i += 1;
                    continue;
                }
                if !in_word {
                    out.push_str(MARK);
                    in_word = true;
                }
                let start = i;
                i += 1;
                while i < bytes.len() && bytes[i] < 0x80 && !is_ascii_ws(bytes[i]) {
                    i += 1;
                }
                out.push_str(&text[start..i]);
                continue;
            }
            let c = text[i..].chars().next().unwrap();
            let w = c.len_utf8();
            if c.is_whitespace() {
                saw_ws = true;
                in_word = false;
                i += w;
                continue;
            }
            if !in_word {
                if !text[i..].starts_with(MARK) {
                    out.push_str(MARK);
                }
                in_word = true;
            }
            out.push_str(&text[i..i + w]);
            i += w;
        }
        in_word || saw_ws
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        let _ = text;
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, _out: &mut String) -> usize {
        text.len().max(1)
    }
}

fn is_ascii_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

fn needs_prepend(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes[0] != b' ' && !bytes.starts_with(&MARK_B)
}

fn copy_until_space(bytes: &[u8]) -> Option<Window> {
    match memchr::memchr(b' ', bytes) {
        None if !bytes.is_empty() => Some(Window::Copy(bytes.len())),
        Some(0) | None => None,
        Some(k) => Some(Window::Copy(k)),
    }
}
