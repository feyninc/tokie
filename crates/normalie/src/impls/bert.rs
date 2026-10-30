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
    // A unified ideograph is never a control, a mark, or a lowercase mapping,
    // and NFD leaves it in place. Emitting it directly skips the category
    // search and the normalizer, which are the whole cost of a CJK island.
    // A compatibility ideograph is still spaced here, then NFD runs on that
    // one character: U+F900 is U+8C48 after NFD, and the fast emit would
    // have kept the compatibility form.
    let mut pending = String::new();
    for c in island.chars() {
        // The basic ideograph block is the whole CJK island in real text.
        // Its body is the direct emit: no normalizer, no second range check.
        let cp = c as u32;
        if CJK && cp >= 0x4E00 && cp <= 0x9FFF {
            finish::<ACCENTS, LOWER>(&mut pending, out);
            out.push(' ');
            out.push(c);
            out.push(' ');
            continue;
        }
        if CJK && cp >= 0x3400 && other_cjk(c) {
            finish::<ACCENTS, LOWER>(&mut pending, out);
            out.push(' ');
            if ACCENTS && compat_ideograph(c) {
                pending.push(c);
                finish::<ACCENTS, LOWER>(&mut pending, out);
            } else {
                out.push(c);
            }
            out.push(' ');
            continue;
        }
        // Voiced kana decomposes to a base plus a mark, and uncased BERT drops
        // the mark. The result is the base. Doing that here keeps a Japanese
        // run off the normalizer, which was one call per voiced character.
        if ACCENTS && LOWER && CLEAN && let Some(stripped) = kana_stripped(c) {
            finish::<ACCENTS, LOWER>(&mut pending, out);
            if let Some(base) = stripped {
                out.push(base);
            }
            continue;
        }
        // A starter that NFD and lowercase leave alone is emitted as itself.
        // The category search and the normalizer stay on the characters that
        // actually change: marks, decompositions, and case folds.
        if ACCENTS && LOWER && CLEAN && is_stable(c) {
            finish::<ACCENTS, LOWER>(&mut pending, out);
            out.push(c);
            continue;
        }
        if CLEAN && removes(c) {
            continue;
        }
        let c = if CLEAN && c.is_whitespace() { ' ' } else { c };
        if ACCENTS || LOWER {
            pending.push(c);
        } else {
            out.push(c);
        }
    }
    finish::<ACCENTS, LOWER>(&mut pending, out);
}

fn finish<const ACCENTS: bool, const LOWER: bool>(pending: &mut String, out: &mut String) {
    if pending.is_empty() {
        return;
    }
    let source = if ACCENTS {
        DecomposingNormalizerBorrowed::new_nfd().normalize(pending)
    } else {
        std::borrow::Cow::Borrowed(pending.as_str())
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
    pending.clear();
}

fn is_stable(c: char) -> bool {
    let cp = c as u32;
    if cp >= 0x10000 {
        return false;
    }
    let rows = stable_rows();
    (rows[(cp >> 6) as usize] >> (cp & 63)) & 1 == 1
}

fn stable_rows() -> &'static [u64; 1024] {
    use std::sync::OnceLock;
    static ROWS: OnceLock<[u64; 1024]> = OnceLock::new();
    ROWS.get_or_init(|| {
        let normalizer = DecomposingNormalizerBorrowed::new_nfd();
        let mut rows = [0u64; 1024];
        for cp in 0u32..0x10000 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            if is_cjk(c) || removes(c) || c.is_whitespace() || c.is_mark_nonspacing() {
                continue;
            }
            let mut buf = [0u8; 4];
            let raw = c.encode_utf8(&mut buf);
            if normalizer.normalize(raw) != raw as &str {
                continue;
            }
            let mut lower = c.to_lowercase();
            if lower.next() != Some(c) || lower.next().is_some() {
                continue;
            }
            rows[(cp >> 6) as usize] |= 1u64 << (cp & 63);
        }
        rows
    })
}

fn removes(c: char) -> bool {
    c == '\u{FFFD}' || (c != '\t' && c != '\n' && c != '\r' && c.is_other())
}

/// Hiragana and katakana under uncased BERT. `None` is not in the block.
/// `Some(None)` is a combining mark, which the uncased pass drops. Anything
/// else is the character NFD plus mark-stripping plus lowercase would emit.
fn kana_stripped(c: char) -> Option<Option<char>> {
    let cp = c as u32;
    if !(0x3040..=0x30FF).contains(&cp) {
        return None;
    }
    Some(match cp {
        0x3040 | 0x3097 | 0x3098 | 0x3099 | 0x309A => None,
        0x304C => Some('\u{304B}'),
        0x304E => Some('\u{304D}'),
        0x3050 => Some('\u{304F}'),
        0x3052 => Some('\u{3051}'),
        0x3054 => Some('\u{3053}'),
        0x3056 => Some('\u{3055}'),
        0x3058 => Some('\u{3057}'),
        0x305A => Some('\u{3059}'),
        0x305C => Some('\u{305B}'),
        0x305E => Some('\u{305D}'),
        0x3060 => Some('\u{305F}'),
        0x3062 => Some('\u{3061}'),
        0x3065 => Some('\u{3064}'),
        0x3067 => Some('\u{3066}'),
        0x3069 => Some('\u{3068}'),
        0x3070 | 0x3071 => Some('\u{306F}'),
        0x3073 | 0x3074 => Some('\u{3072}'),
        0x3076 | 0x3077 => Some('\u{3075}'),
        0x3079 | 0x307A => Some('\u{3078}'),
        0x307C | 0x307D => Some('\u{307B}'),
        0x3094 => Some('\u{3046}'),
        0x309E => Some('\u{309D}'),
        0x30AC => Some('\u{30AB}'),
        0x30AE => Some('\u{30AD}'),
        0x30B0 => Some('\u{30AF}'),
        0x30B2 => Some('\u{30B1}'),
        0x30B4 => Some('\u{30B3}'),
        0x30B6 => Some('\u{30B5}'),
        0x30B8 => Some('\u{30B7}'),
        0x30BA => Some('\u{30B9}'),
        0x30BC => Some('\u{30BB}'),
        0x30BE => Some('\u{30BD}'),
        0x30C0 => Some('\u{30BF}'),
        0x30C2 => Some('\u{30C1}'),
        0x30C5 => Some('\u{30C4}'),
        0x30C7 => Some('\u{30C6}'),
        0x30C9 => Some('\u{30C8}'),
        0x30D0 | 0x30D1 => Some('\u{30CF}'),
        0x30D3 | 0x30D4 => Some('\u{30D2}'),
        0x30D6 | 0x30D7 => Some('\u{30D5}'),
        0x30D9 | 0x30DA => Some('\u{30D8}'),
        0x30DC | 0x30DD => Some('\u{30DB}'),
        0x30F4 => Some('\u{30A6}'),
        0x30F7 => Some('\u{30EF}'),
        0x30F8 => Some('\u{30F0}'),
        0x30F9 => Some('\u{30F1}'),
        0x30FA => Some('\u{30F2}'),
        0x30FE => Some('\u{30FD}'),
        _ => Some(c),
    })
}

fn is_cjk(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp) || other_cjk(c)
}

/// Ideographs outside the basic block, including compatibility ideographs.
/// The basic block is handled before this, so a Latin letter fails one
/// compare and then these ranges.
#[inline(always)]
fn other_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// Compatibility ideographs. NFD maps these onto unified ideographs.
fn compat_ideograph(c: char) -> bool {
    matches!(c as u32, 0xF900..=0xFAFF | 0x2F800..=0x2FA1F)
}
