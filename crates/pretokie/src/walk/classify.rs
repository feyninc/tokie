//! Seven masks for a 64-byte window, inlined into the walker kernels.
//!
//! The masks are ASCII letters, ASCII digits, the space character, newlines
//! (`\n` and `\r`), other ASCII whitespace (tab, vertical tab, form feed),
//! apostrophes, and bytes at or above `0x80`. A byte such as `.` is in none
//! of them. Multibyte characters are only marked non-ASCII here.

use unicode_general_category::{get_general_category, GeneralCategory};

/// One bit per byte in a 64-byte window.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Masks {
    pub letter: u64,
    pub digit: u64,
    pub space: u64,
    pub newline: u64,
    pub other_ws: u64,
    pub apostrophe: u64,
    pub non_ascii: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Class {
    Letter,
    Number,
    Whitespace,
    Other,
}

pub(crate) fn class_of(c: char) -> Class {
    if c.is_ascii() {
        let b = c as u8;
        if is_ascii_letter(b) {
            return Class::Letter;
        }
        if is_ascii_digit(b) {
            return Class::Number;
        }
        if is_ascii_ws(b) {
            return Class::Whitespace;
        }
        return Class::Other;
    }
    if c.is_whitespace() {
        return Class::Whitespace;
    }
    match get_general_category(c) {
        GeneralCategory::UppercaseLetter
        | GeneralCategory::LowercaseLetter
        | GeneralCategory::TitlecaseLetter
        | GeneralCategory::ModifierLetter
        | GeneralCategory::OtherLetter => Class::Letter,
        GeneralCategory::DecimalNumber
        | GeneralCategory::LetterNumber
        | GeneralCategory::OtherNumber => Class::Number,
        _ => Class::Other,
    }
}

#[inline]
fn is_ascii_letter(b: u8) -> bool {
    (b | 0x20).wrapping_sub(b'a') < 26
}

#[inline]
fn is_ascii_digit(b: u8) -> bool {
    b.wrapping_sub(b'0') < 10
}

#[inline]
fn is_ascii_ws(b: u8) -> bool {
    b == b' ' || b.wrapping_sub(9) < 5
}

#[inline]
fn is_newline(b: u8) -> bool {
    b == b'\n' || b == b'\r'
}

#[inline]
fn is_other_ws(b: u8) -> bool {
    b.wrapping_sub(9) < 5 && !is_newline(b)
}

/// Classify `text[at..at + 64]`. `at + 64` must be at most `text.len()`.
///
/// Kernels call the matching classifier directly. This dispatch remains for tests.
#[cfg(test)]
pub(crate) fn classify(text: &[u8], at: usize) -> Masks {
    debug_assert!(at + 64 <= text.len());
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_available() {
            return unsafe { classify_avx512(text, at) };
        }
        if avx2_available() {
            return unsafe { classify_avx2(text, at) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { classify_neon(text, at) };
    }
    classify_scalar(text, at)
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx512_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
}

#[cfg(target_arch = "x86_64")]
pub(crate) fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

pub(crate) fn classify_scalar(text: &[u8], at: usize) -> Masks {
    debug_assert!(at + 64 <= text.len());
    let mut masks = Masks::default();
    for i in 0..64 {
        let b = text[at + i];
        let bit = 1u64 << i;
        if b >= 0x80 {
            masks.non_ascii |= bit;
            continue;
        }
        if is_ascii_letter(b) {
            masks.letter |= bit;
        } else if is_ascii_digit(b) {
            masks.digit |= bit;
        } else if b == b' ' {
            masks.space |= bit;
        } else if is_newline(b) {
            masks.newline |= bit;
        } else if is_other_ws(b) {
            masks.other_ws |= bit;
        } else if b == b'\'' {
            masks.apostrophe |= bit;
        }
    }
    masks
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
pub(crate) unsafe fn classify_avx512(text: &[u8], at: usize) -> Masks {
    use std::arch::x86_64::*;
    unsafe {
        let v = _mm512_loadu_si512(text.as_ptr().add(at) as *const _);
        let lowered = _mm512_or_si512(v, _mm512_set1_epi8(0x20));
        let letter = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(lowered, _mm512_set1_epi8(b'a' as i8)),
            _mm512_set1_epi8(25),
        );
        let digit = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(b'0' as i8)),
            _mm512_set1_epi8(9),
        );
        let space = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b' ' as i8));
        let newline = _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\n' as i8))
            | _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\r' as i8));
        let ws_range = _mm512_cmple_epu8_mask(
            _mm512_sub_epi8(v, _mm512_set1_epi8(9)),
            _mm512_set1_epi8(4),
        );
        let non_ascii = _mm512_movepi8_mask(v);
        Masks {
            letter: letter & !non_ascii,
            digit: digit & !non_ascii,
            space,
            newline,
            other_ws: ws_range & !newline,
            apostrophe: _mm512_cmpeq_epi8_mask(v, _mm512_set1_epi8(b'\'' as i8)),
            non_ascii,
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub(crate) unsafe fn classify_avx2(text: &[u8], at: usize) -> Masks {
    use std::arch::x86_64::*;
    unsafe {
        let v0 = _mm256_loadu_si256(text.as_ptr().add(at) as *const _);
        let v1 = _mm256_loadu_si256(text.as_ptr().add(at + 32) as *const _);
        let le = |v: __m256i, lim: i8| -> __m256i {
            let bound = _mm256_set1_epi8(lim);
            _mm256_cmpeq_epi8(_mm256_min_epu8(v, bound), v)
        };
        let pack = |m0: __m256i, m1: __m256i| -> u64 {
            (_mm256_movemask_epi8(m0) as u32 as u64)
                | ((_mm256_movemask_epi8(m1) as u32 as u64) << 32)
        };
        let lowered0 = _mm256_or_si256(v0, _mm256_set1_epi8(0x20));
        let lowered1 = _mm256_or_si256(v1, _mm256_set1_epi8(0x20));
        let letter = pack(
            le(
                _mm256_sub_epi8(lowered0, _mm256_set1_epi8(b'a' as i8)),
                25,
            ),
            le(
                _mm256_sub_epi8(lowered1, _mm256_set1_epi8(b'a' as i8)),
                25,
            ),
        );
        let digit = pack(
            le(_mm256_sub_epi8(v0, _mm256_set1_epi8(b'0' as i8)), 9),
            le(_mm256_sub_epi8(v1, _mm256_set1_epi8(b'0' as i8)), 9),
        );
        let space = pack(
            _mm256_cmpeq_epi8(v0, _mm256_set1_epi8(b' ' as i8)),
            _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(b' ' as i8)),
        );
        let newline = pack(
            _mm256_or_si256(
                _mm256_cmpeq_epi8(v0, _mm256_set1_epi8(b'\n' as i8)),
                _mm256_cmpeq_epi8(v0, _mm256_set1_epi8(b'\r' as i8)),
            ),
            _mm256_or_si256(
                _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(b'\n' as i8)),
                _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(b'\r' as i8)),
            ),
        );
        let ws_range = pack(
            le(_mm256_sub_epi8(v0, _mm256_set1_epi8(9)), 4),
            le(_mm256_sub_epi8(v1, _mm256_set1_epi8(9)), 4),
        );
        let non_ascii = pack(v0, v1);
        Masks {
            letter: letter & !non_ascii,
            digit: digit & !non_ascii,
            space,
            newline,
            other_ws: ws_range & !newline,
            apostrophe: pack(
                _mm256_cmpeq_epi8(v0, _mm256_set1_epi8(b'\'' as i8)),
                _mm256_cmpeq_epi8(v1, _mm256_set1_epi8(b'\'' as i8)),
            ),
            non_ascii,
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub(crate) unsafe fn classify_neon(text: &[u8], at: usize) -> Masks {
    use std::arch::aarch64::*;
    unsafe {
        let p = text.as_ptr().add(at);
        let mut letter_v = [vdupq_n_u8(0); 4];
        let mut digit_v = [vdupq_n_u8(0); 4];
        let mut space_v = [vdupq_n_u8(0); 4];
        let mut newline_v = [vdupq_n_u8(0); 4];
        let mut other_ws_v = [vdupq_n_u8(0); 4];
        let mut apostrophe_v = [vdupq_n_u8(0); 4];
        let mut non_ascii_v = [vdupq_n_u8(0); 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            letter_v[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            digit_v[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            space_v[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            newline_v[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\n')),
                vceqq_u8(v, vdupq_n_u8(b'\r')),
            );
            let ws_range = vcleq_u8(vsubq_u8(v, vdupq_n_u8(9)), vdupq_n_u8(4));
            other_ws_v[i] = vbicq_u8(ws_range, newline_v[i]);
            apostrophe_v[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
            non_ascii_v[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
        }
        let non_ascii = movemask64(
            non_ascii_v[0],
            non_ascii_v[1],
            non_ascii_v[2],
            non_ascii_v[3],
        );
        Masks {
            letter: movemask64(letter_v[0], letter_v[1], letter_v[2], letter_v[3]) & !non_ascii,
            digit: movemask64(digit_v[0], digit_v[1], digit_v[2], digit_v[3]) & !non_ascii,
            space: movemask64(space_v[0], space_v[1], space_v[2], space_v[3]),
            newline: movemask64(newline_v[0], newline_v[1], newline_v[2], newline_v[3]),
            other_ws: movemask64(other_ws_v[0], other_ws_v[1], other_ws_v[2], other_ws_v[3]),
            apostrophe: movemask64(
                apostrophe_v[0],
                apostrophe_v[1],
                apostrophe_v[2],
                apostrophe_v[3],
            ),
            non_ascii,
        }
    }
}

/// Four NEON compare vectors, 64 lanes of `0x00` or `0xFF`, packed into a `u64`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn movemask64(
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    v3: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const WEIGHTS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(WEIGHTS.as_ptr());
        let mut a0 = vandq_u8(v0, w);
        let a1 = vandq_u8(v1, w);
        let mut a2 = vandq_u8(v2, w);
        let a3 = vandq_u8(v3, w);
        a0 = vpaddq_u8(a0, a1);
        a2 = vpaddq_u8(a2, a3);
        a0 = vpaddq_u8(a0, a2);
        a0 = vpaddq_u8(a0, a0);
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_classes_are_separate() {
        let mut text = [b'.'; 64];
        text[0] = b'A';
        text[1] = b'z';
        text[2] = b'7';
        text[3] = b' ';
        text[4] = b'\t';
        text[5] = b'\n';
        text[6] = b'\r';
        text[7] = 0x0b;
        text[8] = 0x0c;
        text[9] = b'\'';
        text[10] = 0xE4;
        let m = classify_scalar(&text, 0);
        assert_eq!(m.letter, (1 << 0) | (1 << 1));
        assert_eq!(m.digit, 1 << 2);
        assert_eq!(m.space, 1 << 3);
        assert_eq!(m.newline, (1 << 5) | (1 << 6));
        assert_eq!(m.other_ws, (1 << 4) | (1 << 7) | (1 << 8));
        assert_eq!(m.apostrophe, 1 << 9);
        assert_eq!(m.non_ascii, 1 << 10);
    }

    #[test]
    fn punctuation_is_in_no_mask() {
        let text = [b'.'; 64];
        let m = classify_scalar(&text, 0);
        assert_eq!(m, Masks::default());
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_matches_scalar() {
        let mut state = 0x1234_5678_u32;
        let mut buf = [0u8; 96];
        for _ in 0..200 {
            for b in &mut buf {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *b = (state >> 24) as u8;
            }
            let scalar = classify_scalar(&buf, 0);
            assert_eq!(classify(&buf, 0), scalar);
            if avx512_available() {
                assert_eq!(unsafe { classify_avx512(&buf, 0) }, scalar);
            }
            if avx2_available() {
                assert_eq!(unsafe { classify_avx2(&buf, 0) }, scalar);
            }
        }
    }
}
