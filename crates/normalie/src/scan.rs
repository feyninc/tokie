//! Byte scans used by the kernels.
//!
//! `ISA` selects the implementation. Passing a SIMD isa from a function
//! without that instruction set is undefined.

#[cfg(target_arch = "x86_64")]
use crate::formula::{ISA_AVX2, ISA_AVX512};
#[cfg(target_arch = "aarch64")]
use crate::formula::ISA_NEON;

#[inline(always)]
pub(crate) fn ascii_prefix<const ISA: u8>(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == ISA_AVX512 {
            // SAFETY: the AVX-512 kernel is the only caller of this isa.
            return unsafe { ascii_prefix_avx512(bytes) };
        }
        if ISA == ISA_AVX2 {
            // SAFETY: the AVX2 kernel is the only caller of this isa.
            return unsafe { ascii_prefix_avx2(bytes) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_NEON {
        // SAFETY: the NEON kernel is the only caller of this isa.
        return unsafe { ascii_prefix_neon(bytes) };
    }
    let _ = ISA;
    ascii_prefix_scalar(bytes)
}

/// Bytes in `0x20..0x7F`, the printable ASCII range, stopping before the rest.
#[inline(always)]
pub(crate) fn printable_prefix<const ISA: u8>(bytes: &[u8]) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if ISA == ISA_AVX512 {
            return unsafe { printable_prefix_avx512(bytes) };
        }
        if ISA == ISA_AVX2 {
            return unsafe { printable_prefix_avx2(bytes) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    if ISA == ISA_NEON {
        return unsafe { printable_prefix_neon(bytes) };
    }
    let _ = ISA;
    printable_prefix_scalar(bytes)
}

#[inline]
pub(crate) fn ascii_prefix_scalar(bytes: &[u8]) -> usize {
    let mut i = 0;
    while i < bytes.len() && bytes[i] < 0x80 {
        i += 1;
    }
    i
}

#[inline]
fn printable_prefix_scalar(bytes: &[u8]) -> usize {
    let mut i = 0;
    while i < bytes.len() && (0x20..0x7F).contains(&bytes[i]) {
        i += 1;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn ascii_prefix_avx512(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 64 <= bytes.len() {
        // SAFETY: `i + 64` is inside `bytes`.
        let v = unsafe { _mm512_loadu_si512(bytes.as_ptr().add(i).cast()) };
        let high = _mm512_movepi8_mask(v);
        if high != 0 {
            return i + high.trailing_zeros() as usize;
        }
        i += 64;
    }
    i + ascii_prefix_scalar(&bytes[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn printable_prefix_avx512(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    unsafe {
        let lo = _mm512_set1_epi8(0x20);
        let hi = _mm512_set1_epi8(0x7F);
        while i + 64 <= bytes.len() {
            let v = _mm512_loadu_si512(bytes.as_ptr().add(i).cast());
            let below = _mm512_cmplt_epu8_mask(v, lo);
            let above = _mm512_cmpge_epu8_mask(v, hi);
            let bad = below | above;
            if bad != 0 {
                return i + bad.trailing_zeros() as usize;
            }
            i += 64;
        }
    }
    i + printable_prefix_scalar(&bytes[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ascii_prefix_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    while i + 32 <= bytes.len() {
        let v = unsafe { _mm256_loadu_si256(bytes.as_ptr().add(i).cast()) };
        let high = _mm256_movemask_epi8(v);
        if high != 0 {
            return i + high.trailing_zeros() as usize;
        }
        i += 32;
    }
    i + ascii_prefix_scalar(&bytes[i..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn printable_prefix_avx2(bytes: &[u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0;
    unsafe {
        let lo = _mm256_set1_epi8(0x1F);
        let top = _mm256_set1_epi8(0x7E);
        while i + 32 <= bytes.len() {
            let v = _mm256_loadu_si256(bytes.as_ptr().add(i).cast());
            let below = _mm256_movemask_epi8(_mm256_cmpgt_epi8(lo, v));
            let above = _mm256_movemask_epi8(_mm256_cmpgt_epi8(v, top));
            let bad = below | above;
            if bad != 0 {
                return i + bad.trailing_zeros() as usize;
            }
            i += 32;
        }
    }
    i + printable_prefix_scalar(&bytes[i..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ascii_prefix_neon(bytes: &[u8]) -> usize {
    use std::arch::aarch64::*;
    let mut i = 0;
    unsafe {
        while i + 16 <= bytes.len() {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            if vmaxvq_u8(v) >= 0x80 {
                break;
            }
            i += 16;
        }
    }
    i + ascii_prefix_scalar(&bytes[i..])
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn printable_prefix_neon(bytes: &[u8]) -> usize {
    use std::arch::aarch64::*;
    let mut i = 0;
    unsafe {
        let lo = vdupq_n_u8(0x20);
        let hi = vdupq_n_u8(0x7E);
        while i + 16 <= bytes.len() {
            let v = vld1q_u8(bytes.as_ptr().add(i));
            let ge = vcgeq_u8(v, lo);
            let le = vcleq_u8(v, hi);
            let ok = vandq_u8(ge, le);
            if vminvq_u8(ok) == 0 {
                break;
            }
            i += 16;
        }
    }
    i + printable_prefix_scalar(&bytes[i..])
}
