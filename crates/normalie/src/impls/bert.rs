//! BERT cleaning, cased, and uncased.
//!
//! `Clean` drops controls and folds whitespace. `BertCased` also spaces CJK.
//! `BertUncased` adds NFD, drops nonspacing marks, and lowercases. One pass.

use icu_normalizer::DecomposingNormalizerBorrowed;
use unicode_categories::UnicodeCategories;

use crate::formula::Formula;

pub type Clean = Bert<true, false, false, false>;
pub type BertCased = Bert<true, true, false, false>;
pub type BertUncased = Bert<true, true, true, true>;

pub struct Bert<const CLEAN: bool, const CJK: bool, const ACCENTS: bool, const LOWER: bool>;

impl<const CLEAN: bool, const CJK: bool, const ACCENTS: bool, const LOWER: bool> Bert<CLEAN, CJK, ACCENTS, LOWER> {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl<const CLEAN: bool, const CJK: bool, const ACCENTS: bool, const LOWER: bool> Formula
    for Bert<CLEAN, CJK, ACCENTS, LOWER>
{
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        if text.is_empty() {
            return false;
        }
        if text.is_ascii() {
            if !needs_bert::<CLEAN, LOWER>(text.as_bytes()) {
                return false;
            }
            return write_ascii::<CLEAN, LOWER>(text.as_bytes(), out);
        }
        write_mixed::<CLEAN, CJK, ACCENTS, LOWER>(text, out);
        true
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        Some(text.len().max(1))
    }

    fn write_scalar(&self, text: &str, _at_start: bool, _out: &mut String) -> usize {
        text.len().max(1)
    }
}

fn needs_bert<const CLEAN: bool, const LOWER: bool>(bytes: &[u8]) -> bool {
    bytes.iter().any(|&b| {
        (CLEAN && (b < 0x20 || b == 0x7F)) || (LOWER && b.is_ascii_uppercase())
    })
}

fn write_ascii<const CLEAN: bool, const LOWER: bool>(bytes: &[u8], out: &mut String) -> bool {
    let mut changed = false;
    for &b in bytes {
        if CLEAN && b < 0x20 {
            if b == b'\t' || b == b'\n' || b == b'\r' {
                out.push(' ');
            }
            changed = true;
            continue;
        }
        if CLEAN && b == 0x7F {
            changed = true;
            continue;
        }
        if LOWER && b.is_ascii_uppercase() {
            out.push((b | 0x20) as char);
            changed = true;
        } else {
            out.push(b as char);
        }
    }
    if !changed {
        out.clear();
    }
    changed
}

fn write_mixed<const CLEAN: bool, const CJK: bool, const ACCENTS: bool, const LOWER: bool>(
    text: &str,
    out: &mut String,
) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < 0x80 {
            let start = i;
            while i < bytes.len() && bytes[i] < 0x80 {
                i += 1;
            }
            if i < bytes.len() && i > start {
                push_ascii::<CLEAN, LOWER>(&bytes[start..i - 1], out);
                i -= 1;
            } else {
                push_ascii::<CLEAN, LOWER>(&bytes[start..i], out);
                continue;
            }
        }
        let end = crate::form::island_end(&text[i..]);
        debug_assert!(end > 0);
        push_island::<CLEAN, CJK, ACCENTS, LOWER>(&text[i..i + end], out);
        i += end;
    }
}

fn push_ascii<const CLEAN: bool, const LOWER: bool>(bytes: &[u8], out: &mut String) {
    for &b in bytes {
        if CLEAN && (b < 0x20 || b == 0x7F) {
            if b == b'\t' || b == b'\n' || b == b'\r' {
                out.push(' ');
            }
            continue;
        }
        if LOWER && b.is_ascii_uppercase() {
            out.push((b | 0x20) as char);
        } else {
            out.push(b as char);
        }
    }
}

fn push_island<const CLEAN: bool, const CJK: bool, const ACCENTS: bool, const LOWER: bool>(
    island: &str,
    out: &mut String,
) {
    let mut cleaned = String::with_capacity(island.len() + 8);
    for c in island.chars() {
        if CLEAN && removes(c) {
            continue;
        }
        let c = if CLEAN && c.is_whitespace() { ' ' } else { c };
        if CJK && is_cjk(c) {
            cleaned.push(' ');
            cleaned.push(c);
            cleaned.push(' ');
        } else {
            cleaned.push(c);
        }
    }
    if !ACCENTS && !LOWER {
        out.push_str(&cleaned);
        return;
    }
    let source = if ACCENTS {
        DecomposingNormalizerBorrowed::new_nfd().normalize(&cleaned)
    } else {
        std::borrow::Cow::Borrowed(cleaned.as_str())
    };
    for c in source.chars() {
        if ACCENTS && c.is_mark_nonspacing() {
            continue;
        }
        if LOWER {
            for lower in c.to_lowercase() {
                out.push(lower);
            }
        } else {
            out.push(c);
        }
    }
}

fn removes(c: char) -> bool {
    c == '\u{FFFD}' || (c != '\t' && c != '\n' && c != '\r' && c.is_other())
}

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}
