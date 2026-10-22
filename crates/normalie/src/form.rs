//! Quick check for the four Unicode forms.
//!
//! ASCII is already in every form. NFC of a long string scans for bytes at
//! or above U+0300 and checks only those scalars. The other forms use the
//! Unicode quick-check tables, and `Maybe` is resolved with ICU.

use std::borrow::Cow;
use std::sync::OnceLock;

use icu_normalizer::{ComposingNormalizerBorrowed, DecomposingNormalizerBorrowed};
use unicode_normalization::IsNormalized;
use unicode_normalization::{is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick};

use crate::scan::ascii_prefix_scalar;

#[derive(Clone, Copy)]
pub(crate) enum Kind {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
}

pub(crate) fn is_clean(kind: Kind, text: &str) -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F and AVX-512BW were just detected.
            return unsafe { is_clean_avx512(kind, text) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            return unsafe { is_clean_avx2(kind, text) };
        }
    }
    is_clean_scalar(kind, text)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn is_clean_avx512(kind: Kind, text: &str) -> bool {
    is_clean_scan(kind, text, crate::scan::ascii_prefix::<{ crate::formula::ISA_AVX512 }>)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn is_clean_avx2(kind: Kind, text: &str) -> bool {
    is_clean_scan(kind, text, crate::scan::ascii_prefix::<{ crate::formula::ISA_AVX2 }>)
}

fn is_clean_scalar(kind: Kind, text: &str) -> bool {
    is_clean_scan(kind, text, ascii_prefix_scalar)
}

fn is_clean_scan(kind: Kind, text: &str, skip_ascii: impl Fn(&[u8]) -> usize) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() || (bytes[0] < 0x80 && skip_ascii(bytes) == bytes.len()) {
        return true;
    }
    // NFC and NFD match ICU on every BMP scalar, and that check is what a
    // long mixed string spends its time in. NFKC and NFKD do not match it
    // for a handful of scalars, so those stay on the quick-check tables.
    if matches!(kind, Kind::Nfc | Kind::Nfd) {
        return icu_clean(kind, text);
    }
    match quick(kind, text) {
        IsNormalized::Yes => true,
        IsNormalized::No => false,
        IsNormalized::Maybe => icu_clean(kind, text),
    }
}

fn quick(kind: Kind, text: &str) -> IsNormalized {
    match kind {
        Kind::Nfc => is_nfc_quick(text.chars()),
        Kind::Nfd => is_nfd_quick(text.chars()),
        Kind::Nfkc => is_nfkc_quick(text.chars()),
        Kind::Nfkd => is_nfkd_quick(text.chars()),
    }
}

fn icu_clean(kind: Kind, text: &str) -> bool {
    match kind {
        Kind::Nfc => ComposingNormalizerBorrowed::new_nfc().is_normalized(text),
        Kind::Nfkc => ComposingNormalizerBorrowed::new_nfkc().is_normalized(text),
        Kind::Nfd => DecomposingNormalizerBorrowed::new_nfd().is_normalized(text),
        Kind::Nfkd => DecomposingNormalizerBorrowed::new_nfkd().is_normalized(text),
    }
}

/// Append the normalized form of `text`. ASCII runs are copied.
pub(crate) fn emit(kind: Kind, text: &str, out: &mut String) {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let n = ascii_prefix_scalar(&bytes[i..]);
        let next_hi = bytes.get(i + n).is_some_and(|b| *b >= 0x80);
        let leave = usize::from(next_hi && n > 0);
        let take = n - leave;
        if take > 0 {
            out.push_str(&text[i..i + take]);
            i += take;
        }
        if i >= bytes.len() {
            break;
        }
        let end = island_end(&text[i..]);
        debug_assert!(end > 0);
        emit_island(kind, &text[i..i + end], out);
        i += end;
    }
}

fn emit_island(kind: Kind, island: &str, out: &mut String) {
    match kind {
        Kind::Nfc => push_cow(ComposingNormalizerBorrowed::new_nfc().normalize(island), out),
        Kind::Nfkc => push_cow(ComposingNormalizerBorrowed::new_nfkc().normalize(island), out),
        Kind::Nfd => push_cow(DecomposingNormalizerBorrowed::new_nfd().normalize(island), out),
        Kind::Nfkd => push_cow(DecomposingNormalizerBorrowed::new_nfkd().normalize(island), out),
    }
}

fn push_cow(cow: Cow<'_, str>, out: &mut String) {
    out.push_str(&cow);
}

/// End of the normalization island at the start of `text`.
pub(crate) fn island_end(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < 0x80 {
            let next_hi = bytes.get(i + 1).is_some_and(|b| *b >= 0x80);
            if i > 0 && !next_hi {
                return i;
            }
            i += 1;
            continue;
        }
        i += utf8_width(bytes[i]);
    }
    bytes.len()
}

#[inline]
fn utf8_width(b: u8) -> usize {
    if b < 0xE0 {
        2
    } else if b < 0xF0 {
        3
    } else {
        4
    }
}

/// `true` when `text` is already NFC.
///
/// Short strings go straight to ICU. A long string skips bytes below U+0300:
/// those scalars are starters and already NFC. A later starter that is already
/// NFC is skipped too. A scalar that can compose is checked with ICU, together
/// with the scalar before it.
pub(crate) fn nfc_yes(text: &str) -> bool {
    if text.len() < 256 {
        return ComposingNormalizerBorrowed::new_nfc().is_normalized(text);
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F and AVX-512BW were just detected.
            return unsafe { nfc_yes_avx512(text) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            return unsafe { nfc_yes_avx2(text) };
        }
    }
    nfc_yes_scalar(text)
}

fn nfc_starters() -> &'static [bool] {
    static TABLE: OnceLock<Vec<bool>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let nfc = ComposingNormalizerBorrowed::new_nfc();
        let mut table = vec![false; 65536];
        let mut buf = [0u8; 4];
        for cp in 0u32..65536 {
            if algorithmic_jamo(cp) {
                continue;
            }
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            if unicode_normalization::char::canonical_combining_class(c) != 0 {
                continue;
            }
            if nfc.is_normalized(c.encode_utf8(&mut buf)) {
                table[cp as usize] = true;
            }
        }
        table
    })
}

/// Hangul jamo compose with the next jamo even though both are starters.
fn algorithmic_jamo(cp: u32) -> bool {
    matches!(cp, 0x1100..=0x11FF | 0xA960..=0xA97F | 0xD7B0..=0xD7FF)
}

/// `Some(next)` when the scalar at `i` is a starter that is already NFC.
/// `i` is a lead byte `>= 0xCC` inside a valid string.
#[inline(always)]
unsafe fn skip_if_starter(bytes: &[u8], i: usize, table: &[bool]) -> Option<usize> {
    // SAFETY: `i` is a UTF-8 lead, so the rest of the scalar is in `bytes`.
    let b0 = unsafe { *bytes.get_unchecked(i) };
    let (cp, w) = if b0 < 0xE0 {
        let b1 = unsafe { *bytes.get_unchecked(i + 1) };
        ((u32::from(b0 & 0x1F) << 6) | u32::from(b1 & 0x3F), 2)
    } else if b0 < 0xF0 {
        let b1 = unsafe { *bytes.get_unchecked(i + 1) };
        let b2 = unsafe { *bytes.get_unchecked(i + 2) };
        (
            (u32::from(b0 & 0x0F) << 12) | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F),
            3,
        )
    } else {
        let b1 = unsafe { *bytes.get_unchecked(i + 1) };
        let b2 = unsafe { *bytes.get_unchecked(i + 2) };
        let b3 = unsafe { *bytes.get_unchecked(i + 3) };
        (
            (u32::from(b0 & 0x07) << 18)
                | (u32::from(b1 & 0x3F) << 12)
                | (u32::from(b2 & 0x3F) << 6)
                | u32::from(b3 & 0x3F),
            4,
        )
    };
    if w == 4 {
        return None;
    }
    // SAFETY: a 2-byte or 3-byte scalar is in the BMP, and the table covers it.
    if unsafe { *table.get_unchecked(cp as usize) } {
        Some(i + w)
    } else {
        None
    }
}

#[inline(always)]
fn prev_char(bytes: &[u8], i: usize) -> usize {
    let mut s = i - 1;
    while s > 0 && bytes[s] & 0xC0 == 0x80 {
        s -= 1;
    }
    s
}

/// Next index after a scalar the fast table cannot skip, or `None` when that
/// stretch is not NFC. `i` points at a lead byte `>= 0xCC`.
#[cold]
#[inline(never)]
fn accept_slow(bytes: &[u8], i: usize, table: &[bool]) -> Option<usize> {
    let w = utf8_width(bytes[i]);
    let start = if i == 0 { 0 } else { prev_char(bytes, i) };
    let mut end = i + w;
    while end < bytes.len() && bytes[end] >= 0xCC {
        // SAFETY: `end` is a lead byte in a valid string.
        if unsafe { skip_if_starter(bytes, end, table) }.is_some() {
            break;
        }
        end += utf8_width(bytes[end]);
    }
    // SAFETY: `start` and `end` are scalar boundaries in a valid string.
    let region = unsafe { std::str::from_utf8_unchecked(&bytes[start..end]) };
    if ComposingNormalizerBorrowed::new_nfc().is_normalized(region) {
        Some(end)
    } else {
        None
    }
}

#[inline(always)]
fn step(bytes: &[u8], i: usize, table: &[bool]) -> Option<usize> {
    // SAFETY: the caller found a lead byte `>= 0xCC` at `i`.
    if let Some(next) = unsafe { skip_if_starter(bytes, i, table) } {
        return Some(next);
    }
    accept_slow(bytes, i, table)
}

fn nfc_yes_scalar(text: &str) -> bool {
    let bytes = text.as_bytes();
    let table = nfc_starters();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] < 0xCC {
            i += 1;
            continue;
        }
        i = match step(bytes, i, table) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn nfc_yes_avx512(text: &str) -> bool {
    use std::arch::x86_64::*;
    let bytes = text.as_bytes();
    let table = nfc_starters();
    let mut i = 0;
    let thr = _mm512_set1_epi8(0xCCu8 as i8);
    while i < bytes.len() {
        if i + 64 <= bytes.len() {
            // SAFETY: `i + 64` is inside `bytes`.
            let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(i).cast()) };
            let mask = _mm512_cmpge_epu8_mask(v, thr);
            if mask == 0 {
                i += 64;
                continue;
            }
            i += mask.trailing_zeros() as usize;
        } else if bytes[i] < 0xCC {
            i += 1;
            continue;
        }
        i = match step(bytes, i, table) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn nfc_yes_avx2(text: &str) -> bool {
    use std::arch::x86_64::*;
    let bytes = text.as_bytes();
    let table = nfc_starters();
    let mut i = 0;
    let thr = _mm256_set1_epi8(0xCCu8 as i8);
    while i < bytes.len() {
        if i + 32 <= bytes.len() {
            // SAFETY: `i + 32` is inside `bytes`.
            let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i).cast()) };
            let ge = _mm256_cmpeq_epi8(_mm256_max_epu8(v, thr), v);
            let mask = _mm256_movemask_epi8(ge);
            if mask == 0 {
                i += 32;
                continue;
            }
            i += mask.trailing_zeros() as usize;
        } else if bytes[i] < 0xCC {
            i += 1;
            continue;
        }
        i = match step(bytes, i, table) {
            Some(next) => next,
            None => return false,
        };
    }
    true
}

