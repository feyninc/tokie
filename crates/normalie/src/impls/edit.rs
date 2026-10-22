//! Literal edits: prepend `▁`, replace spaces, collapse space runs, strip.

use crate::formula::{Formula, Window};

const MARK: &str = "▁";

pub struct Prepend;
pub struct ReplaceSpace;
pub struct CollapseSpaces;
pub struct Strip;

impl Prepend {
    #[inline(always)]
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Prepend {
    #[inline(always)]
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        let mark = MARK.as_bytes();
        let src = text.as_bytes();
        out.reserve(mark.len() + src.len());
        // SAFETY: `reserve` covers the mark and `text`. Both are UTF-8, and the
        // mark is a single scalar, so the joined bytes are UTF-8.
        unsafe {
            let v = out.as_mut_vec();
            let n = v.len();
            let dst = v.as_mut_ptr().add(n);
            std::ptr::copy_nonoverlapping(mark.as_ptr(), dst, mark.len());
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst.add(mark.len()), src.len());
            v.set_len(n + mark.len() + src.len());
        }
        true
    }

    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        out.push_str(MARK);
        out.push_str(text);
        text.len()
    }
}

impl ReplaceSpace {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for ReplaceSpace {
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
        match memchr::memchr(b' ', bytes) {
            None if !bytes.is_empty() => Some(Window::Copy(bytes.len())),
            Some(0) | None => None,
            Some(k) => Some(Window::Copy(k)),
        }
    }
}

impl CollapseSpaces {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for CollapseSpaces {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() || !text.contains("  ") {
            return false;
        }
        collapse_into(text, out);
        true
    }

    fn unchanged(&self, _text: &str, _at_start: bool) -> Option<usize> {
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        collapse_into(text, out);
        text.len()
    }

    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], _at_start: bool) -> Option<Window> {
        let _ = ISA;
        match memchr::memmem::find(bytes, b"  ") {
            None if !bytes.is_empty() => Some(Window::Copy(bytes.len())),
            Some(0) | None => None,
            Some(k) => Some(Window::Copy(k)),
        }
    }
}

impl Strip {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }

    /// Trim one side. `Strip::write` trims both.
    pub fn write_sides<const LEFT: bool, const RIGHT: bool>(
        text: &str,
        out: &mut String,
    ) -> bool {
        let trimmed = trim::<LEFT, RIGHT>(text);
        if trimmed.len() == text.len() {
            return false;
        }
        out.push_str(trimmed);
        true
    }
}

impl Formula for Strip {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        let trimmed = text.trim();
        if trimmed.len() == text.len() {
            return false;
        }
        out.push_str(trimmed);
        true
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        let trimmed = text.trim();
        if trimmed.len() == text.len() {
            Some(text.len())
        } else {
            None
        }
    }

    fn write_scalar(&self, text: &str, _at_start: bool, out: &mut String) -> usize {
        out.push_str(text.trim());
        text.len()
    }
}

fn trim<const LEFT: bool, const RIGHT: bool>(text: &str) -> &str {
    match (LEFT, RIGHT) {
        (true, true) => text.trim(),
        (true, false) => text.trim_start(),
        (false, true) => text.trim_end(),
        (false, false) => text,
    }
}

fn collapse_into(text: &str, out: &mut String) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b' ' {
                j += 1;
            }
            out.push(' ');
            i = j;
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b' ' {
                i += 1;
            }
            out.push_str(&text[start..i]);
        }
    }
}

pub(crate) fn expand_spaces(text: &str, prepend: bool, out: &mut String) {
    if prepend {
        out.push_str(MARK);
    }
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if let Some(at) = memchr::memchr(b' ', &bytes[i..]) {
            out.push_str(&text[i..i + at]);
            out.push_str(MARK);
            i += at + 1;
        } else {
            out.push_str(&text[i..]);
            break;
        }
    }
}
