//! The private formula contract.
//!
//! `write_scalar` is the ground truth. `try_window` reports a copy or a
//! rewrite the kernel can emit without calling it. A copy may be longer
//! than 64 bytes. The SIMD width inside the scan is 64.

/// Scalar instruction set. Safe to pass from any caller.
pub(crate) const ISA_SCALAR: u8 = 0;
/// AVX2. Only a kernel compiled with AVX2 may pass this.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_AVX2: u8 = 1;
/// AVX-512. Only a kernel compiled with AVX-512 may pass this.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_AVX512: u8 = 2;
/// NEON. Only a kernel compiled with NEON may pass this.
#[cfg(target_arch = "aarch64")]
pub(crate) const ISA_NEON: u8 = 3;

/// What one window did to the bytes it consumed.
pub(crate) enum Window {
    /// These input bytes are already the output.
    Copy(usize),
    /// `consumed` input bytes become `out[..len]`.
    #[allow(dead_code)]
    Rewrite { consumed: usize, out: [u8; 192], len: usize },
}

/// One normalization formula.
pub(crate) trait Formula {
    /// How many leading bytes are already normalized.
    ///
    /// `None` means the next byte is part of a change. `Some(0)` is forbidden.
    fn unchanged(&self, text: &str, at_start: bool) -> Option<usize>;

    /// Write the next changed unit onto `out`. Returns the bytes consumed.
    fn write_scalar(&self, text: &str, at_start: bool, out: &mut String) -> usize;

    /// A copy or a rewrite starting at `bytes[0]`.
    ///
    /// `None` means this stretch stays on [`Formula::unchanged`]. `ISA`
    /// selects the scan baked into the caller. Passing a SIMD isa from a
    /// function without that instruction set is undefined.
    #[inline(always)]
    fn try_window<const ISA: u8>(&self, bytes: &[u8], at_start: bool) -> Option<Window> {
        let _ = (bytes, at_start);
        None
    }

    /// Run the formula. The default walks windows, then scalar units.
    fn drive(&self, text: &str, at_start: bool, out: &mut String) -> bool {
        crate::kernel::apply(self, text, at_start, out)
    }
}
