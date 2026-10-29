//! NFKD, drop marks, drop controls, lowercase, collapse whitespace, metaspace.

use unicode_categories::UnicodeCategories;
use unicode_normalization::UnicodeNormalization;

use crate::formula::Formula;
use crate::marks::is_unicode9_mark;

const MARK: &str = "▁";

pub struct Albert;

impl Albert {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for Albert {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        if text.is_ascii() {
            return albert_ascii(text.as_bytes(), out);
        }
        let bytes = text.as_bytes();
        let mut i = 0;
        let mut gap = false;
        let mut word = false;
        while i < bytes.len() {
            if bytes[i] < 0x80 {
                let b = bytes[i];
                i += 1;
                feed_ascii(b, &mut gap, &mut word, out);
                continue;
            }
            let c = text[i..].chars().next().unwrap();
            i += c.len_utf8();
            feed_chars(c.nfkd(), &mut gap, &mut word, out);
        }
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

fn albert_ascii(bytes: &[u8], out: &mut String) -> bool {
    let mut gap = false;
    let mut word = false;
    for &b in bytes {
        feed_ascii(b, &mut gap, &mut word, out);
    }
    word || bytes.iter().any(|&b| b == b' ' || b < 0x20 || b == 0x7F)
}

#[inline(always)]
fn feed_ascii(b: u8, gap: &mut bool, word: &mut bool, out: &mut String) {
    if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' || b == 0x0C || b == 0x0B {
        if *word {
            *gap = true;
        }
        return;
    }
    if b < 0x20 || b == 0x7F {
        return;
    }
    let b = if b.is_ascii_uppercase() { b | 0x20 } else { b };
    push_letter(b as char, gap, word, out);
}

fn feed_chars(chars: impl Iterator<Item = char>, gap: &mut bool, word: &mut bool, out: &mut String) {
    for c in chars {
        if is_unicode9_mark(c) {
            continue;
        }
        if c.is_whitespace() {
            if *word {
                *gap = true;
            }
            continue;
        }
        if c == '\u{FFFD}' || c.is_other() {
            continue;
        }
        for lower in c.to_lowercase() {
            if lower.is_whitespace() {
                if *word {
                    *gap = true;
                }
                continue;
            }
            push_letter(lower, gap, word, out);
        }
    }
}

#[inline(always)]
fn push_letter(c: char, gap: &mut bool, word: &mut bool, out: &mut String) {
    if !*word {
        out.push_str(MARK);
        *word = true;
        *gap = false;
    } else if *gap {
        out.push_str(MARK);
        *gap = false;
    }
    out.push(c);
}

