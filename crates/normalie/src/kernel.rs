//! One walk for every formula that does not bring its own.
//!
//! An unchanged prefix is not copied. The first change copies that prefix
//! and then writes. On x86-64 the order is AVX-512, then AVX2, then scalar.
//! On aarch64 the kernel uses NEON.

use crate::formula::{Formula, Window, ISA_SCALAR};

#[cfg(target_arch = "x86_64")]
use crate::formula::{ISA_AVX2, ISA_AVX512};
#[cfg(target_arch = "aarch64")]
use crate::formula::ISA_NEON;

pub(crate) fn apply<F: Formula + ?Sized>(
    formula: &F,
    text: &str,
    at_start: bool,
    out: &mut String,
) -> bool {
    debug_assert!(out.is_empty(), "write expects an empty buffer");
    if text.is_empty() {
        return false;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: AVX-512F and AVX-512BW were just detected.
            return unsafe { apply_avx512(formula, text, at_start, out) };
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was just detected.
            return unsafe { apply_avx2(formula, text, at_start, out) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on aarch64.
        return unsafe { apply_neon(formula, text, at_start, out) };
    }
    drive::<F, ISA_SCALAR>(formula, text, at_start, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn apply_avx512<F: Formula + ?Sized>(
    formula: &F,
    text: &str,
    at_start: bool,
    out: &mut String,
) -> bool {
    drive::<F, ISA_AVX512>(formula, text, at_start, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn apply_avx2<F: Formula + ?Sized>(
    formula: &F,
    text: &str,
    at_start: bool,
    out: &mut String,
) -> bool {
    drive::<F, ISA_AVX2>(formula, text, at_start, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_neon<F: Formula + ?Sized>(
    formula: &F,
    text: &str,
    at_start: bool,
    out: &mut String,
) -> bool {
    drive::<F, ISA_NEON>(formula, text, at_start, out)
}

#[inline(always)]
pub(crate) fn drive<F: Formula + ?Sized, const ISA: u8>(
    formula: &F,
    text: &str,
    at_start: bool,
    out: &mut String,
) -> bool {
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut started = false;
    while i < bytes.len() {
        let at = at_start && i == 0;
        if let Some(window) = formula.try_window::<ISA>(&bytes[i..], at) {
            match window {
                Window::Copy(n) => {
                    debug_assert!(n > 0 && i + n <= bytes.len());
                    debug_assert!(text.is_char_boundary(i + n));
                    if started {
                        out.push_str(&text[i..i + n]);
                    }
                    i += n;
                }
                Window::Rewrite { consumed, out: buf, len } => {
                    debug_assert!(consumed > 0 && i + consumed <= bytes.len());
                    if !started {
                        out.push_str(&text[..i]);
                        started = true;
                    }
                    let piece = std::str::from_utf8(&buf[..len]).expect("window is utf-8");
                    out.push_str(piece);
                    i += consumed;
                }
            }
            continue;
        }
        let rest = &text[i..];
        if let Some(n) = formula.unchanged(rest, at) {
            debug_assert!(n > 0 && n <= rest.len());
            debug_assert!(rest.is_char_boundary(n));
            if started {
                out.push_str(&rest[..n]);
            }
            i += n;
            continue;
        }
        if !started {
            out.reserve(text.len());
            out.push_str(&text[..i]);
            started = true;
        }
        let n = formula.write_scalar(rest, at, out);
        debug_assert!(n > 0 && n <= rest.len(), "a change consumes at least one byte");
        debug_assert!(rest.is_char_boundary(n));
        i += n;
    }
    started
}
