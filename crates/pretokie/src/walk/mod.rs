//! Shared piece walker.
//!
//! A pretokenizer implements [`Scheme::advance`]. When it can, it also
//! reports every piece start in one 64-byte window. Count and fill each run
//! in one kernel for the instruction set picked at the start of the call.
//! That kernel inlines the scheme's window and the matching function from
//! [`classify`]. The walker does not know the split rules.

pub(crate) mod classify;

use crate::Span;

/// Scalar classification. The SIMD kernels pass their own isa instead.
pub(crate) const ISA_SCALAR: u8 = 0;
/// AVX2 classification. Only a kernel compiled with AVX2 may pass this.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_AVX2: u8 = 1;
/// AVX-512 classification. Only a kernel compiled with AVX-512 may pass this.
#[cfg(target_arch = "x86_64")]
pub(crate) const ISA_AVX512: u8 = 2;
/// NEON classification. Only a kernel compiled with NEON may pass this.
#[cfg(target_arch = "aarch64")]
pub(crate) const ISA_NEON: u8 = 3;

/// Scalar ground truth for one pretokenizer.
pub(crate) trait Scheme {
    /// End offset of the piece that starts at `pos`.
    ///
    /// `pos` is less than `bytes.len()` and sits on a piece boundary.
    fn advance(bytes: &[u8], pos: usize) -> usize;

    /// Piece starts in the 64-byte window at `pos`.
    ///
    /// `pos` sits on a piece boundary. `None` means this piece stays on
    /// [`Scheme::advance`]. `ISA` selects the classifier baked into the
    /// caller. Passing a SIMD isa from a function without that instruction
    /// set is undefined.
    #[inline(always)]
    fn try_window<const ISA: u8>(bytes: &[u8], pos: usize) -> Option<Window> {
        let _ = (bytes, pos, ISA);
        None
    }
}

/// Piece starts for one 64-byte window.
///
/// Bit 0 is the first byte of the window. Each later set bit is another
/// piece start. When [`Window::finishes`] is set, the last of those pieces
/// ends at the end of the window. Otherwise the highest set bit is the
/// start of a piece this window cannot finish.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub starts: u64,
    pub finishes: bool,
}

/// Cursor over the pieces of one text.
pub struct Walk<'a, S> {
    bytes: &'a [u8],
    pos: usize,
    /// Absolute offset of byte 0 in [`Self::positions`].
    window_at: usize,
    /// Piece-start offsets within the current window, low to high.
    ///
    /// Neighboring entries are one piece. A 64 at the end means that piece
    /// ends at the end of the window. An empty list means no window is cached.
    positions: [u8; 65],
    /// Index of the current piece in [`Self::positions`].
    cursor: u8,
    /// Entries used in [`Self::positions`].
    len: u8,
    _scheme: core::marker::PhantomData<S>,
}

impl<'a, S: Scheme> Walk<'a, S> {
    pub(crate) fn new(text: &'a str) -> Self {
        assert!(
            text.len() <= u32::MAX as usize,
            "text is longer than 4 GiB"
        );
        Self {
            bytes: text.as_bytes(),
            pos: 0,
            window_at: 0,
            positions: [0; 65],
            cursor: 0,
            len: 0,
            _scheme: core::marker::PhantomData,
        }
    }

    /// Write the next pieces into `out`. Returns how many were written.
    /// A short count means the text is finished.
    pub fn fill(&mut self, out: &mut [Span]) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if compress_available() {
                // SAFETY: `compress_available` checked AVX-512 F, BW, and VBMI2.
                return unsafe { fill_avx512::<S>(self, out) };
            }
            if classify::avx512_available() {
                // SAFETY: AVX-512 F and BW are present. Popcnt is present on
                // every such CPU.
                return unsafe { fill_avx512_scalar_pack::<S>(self, out) };
            }
            if classify::avx2_available() {
                // SAFETY: AVX2 is present.
                return unsafe { fill_avx2::<S>(self, out) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is the baseline on aarch64.
            return unsafe { fill_neon::<S>(self, out) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        fill_inner::<S, false, ISA_SCALAR>(self, out)
    }

    /// Count pieces, and stop once the count passes `limit`.
    ///
    /// The result is the real count when the text has `limit` pieces or
    /// fewer. When another piece exists past `limit`, the result is
    /// `limit + 1` and the rest of the text is not read.
    pub fn count_upto(self, limit: usize) -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if classify::avx512_available() {
                // SAFETY: AVX-512 F and BW are present.
                return unsafe { count_avx512::<S>(self, limit) };
            }
            if classify::avx2_available() {
                // SAFETY: AVX2 is present.
                return unsafe { count_avx2::<S>(self, limit) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is the baseline on aarch64.
            return unsafe { count_neon::<S>(self, limit) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        count_inner::<S, ISA_SCALAR>(self, limit)
    }

    fn write(&mut self, out: &mut [Span], n: usize, end: usize) {
        let start = self.pos;
        debug_assert!(end > start, "advance must consume at least one byte");
        debug_assert_eq!(end, S::advance(self.bytes, start));
        out[n] = Span {
            start: start as u32,
            end: end as u32,
        };
        self.pos = end;
    }

}

/// Drain cached starts, then each new window, into `out`.
///
/// `COMPRESS` packs the mask with `vpcompressb`. That copy is only sound
/// when the caller is already running with AVX-512 VBMI2 enabled, so the
/// pack inlines into that function instead of being a separate call.
/// `ISA` selects the classifier inlined from [`Scheme::try_window`].
#[inline(always)]
fn fill_inner<S: Scheme, const COMPRESS: bool, const ISA: u8>(
    walk: &mut Walk<'_, S>,
    out: &mut [Span],
) -> usize {
    let mut n = 0;
    while n < out.len() && walk.pos < walk.bytes.len() {
        let ready = (walk.len as usize).saturating_sub(walk.cursor as usize + 1);
        if ready > 0 {
            let take = ready.min(out.len() - n);
            let base = walk.window_at;
            let cursor = walk.cursor as usize;
            write_pairs(&mut out[n..], base, &walk.positions, cursor, take);
            #[cfg(debug_assertions)]
            check_pairs::<S>(walk.bytes, base, &walk.positions, cursor, take);
            let next = cursor + take;
            walk.cursor = next as u8;
            walk.pos = base + walk.positions[next] as usize;
            n += take;
            continue;
        }
        walk.len = 0;
        let Some(window) = S::try_window::<ISA>(walk.bytes, walk.pos) else {
            let end = S::advance(walk.bytes, walk.pos);
            walk.write(out, n, end);
            n += 1;
            continue;
        };
        let mut positions = [0u8; 65];
        let mut count = if COMPRESS {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: `fill_avx512` is the only caller with `COMPRESS` set.
                unsafe { flatten_compress(window.starts, &mut positions) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!()
            }
        } else {
            flatten_scalar(window.starts, &mut positions)
        };
        if window.finishes {
            positions[count] = 64;
            count += 1;
        }
        if count < 2 {
            let end = S::advance(walk.bytes, walk.pos);
            walk.write(out, n, end);
            n += 1;
            continue;
        }
        let base = walk.pos;
        let take = (count - 1).min(out.len() - n);
        write_pairs(&mut out[n..], base, &positions, 0, take);
        #[cfg(debug_assertions)]
        check_pairs::<S>(walk.bytes, base, &positions, 0, take);
        n += take;
        walk.pos = base + positions[take] as usize;
        if take + 1 < count {
            walk.positions = positions;
            walk.window_at = base;
            walk.cursor = take as u8;
            walk.len = count as u8;
            break;
        }
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,popcnt")]
unsafe fn fill_avx512<S: Scheme>(walk: &mut Walk<'_, S>, out: &mut [Span]) -> usize {
    fill_inner::<S, true, ISA_AVX512>(walk, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
unsafe fn fill_avx512_scalar_pack<S: Scheme>(walk: &mut Walk<'_, S>, out: &mut [Span]) -> usize {
    fill_inner::<S, false, ISA_AVX512>(walk, out)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fill_avx2<S: Scheme>(walk: &mut Walk<'_, S>, out: &mut [Span]) -> usize {
    fill_inner::<S, false, ISA_AVX2>(walk, out)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fill_neon<S: Scheme>(walk: &mut Walk<'_, S>, out: &mut [Span]) -> usize {
    fill_inner::<S, false, ISA_NEON>(walk, out)
}

/// Count pieces from `walk`. `ISA` selects the inlined classifier.
#[inline(always)]
fn count_inner<S: Scheme, const ISA: u8>(mut walk: Walk<'_, S>, limit: usize) -> usize {
    let mut n: usize = 0;
    while walk.pos < walk.bytes.len() {
        if let Some(window) = S::try_window::<ISA>(walk.bytes, walk.pos) {
            let ones = window.starts.count_ones() as usize;
            let complete = if window.finishes { ones } else { ones - 1 };
            debug_assert!(complete >= 1);
            debug_assert_window::<S>(walk.bytes, walk.pos, window, complete);
            if n.saturating_add(complete) > limit {
                return limit + 1;
            }
            n += complete;
            walk.pos = if window.finishes {
                walk.pos + 64
            } else {
                let last = 63 - window.starts.leading_zeros();
                walk.pos + last as usize
            };
            continue;
        }
        let end = S::advance(walk.bytes, walk.pos);
        debug_assert!(end > walk.pos, "advance must consume at least one byte");
        walk.pos = end;
        n += 1;
        if n > limit {
            return n;
        }
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
unsafe fn count_avx512<S: Scheme>(walk: Walk<'_, S>, limit: usize) -> usize {
    count_inner::<S, ISA_AVX512>(walk, limit)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_avx2<S: Scheme>(walk: Walk<'_, S>, limit: usize) -> usize {
    count_inner::<S, ISA_AVX2>(walk, limit)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn count_neon<S: Scheme>(walk: Walk<'_, S>, limit: usize) -> usize {
    count_inner::<S, ISA_NEON>(walk, limit)
}

/// Write `take` pieces starting at `positions[cursor]`.
fn write_pairs(out: &mut [Span], base: usize, positions: &[u8], cursor: usize, take: usize) {
    debug_assert!(take <= out.len());
    debug_assert!(cursor + take < positions.len());
    for k in 0..take {
        // SAFETY: `cursor + k + 1` is inside `positions`, and `k` is inside `out`.
        unsafe {
            let start = base + *positions.get_unchecked(cursor + k) as usize;
            let end = base + *positions.get_unchecked(cursor + k + 1) as usize;
            *out.get_unchecked_mut(k) = Span {
                start: start as u32,
                end: end as u32,
            };
        }
    }
}

#[cfg(debug_assertions)]
fn check_pairs<S: Scheme>(bytes: &[u8], base: usize, positions: &[u8], cursor: usize, take: usize) {
    for k in 0..take {
        let start = base + positions[cursor + k] as usize;
        let end = base + positions[cursor + k + 1] as usize;
        debug_assert!(end > start, "advance must consume at least one byte");
        debug_assert_eq!(end, S::advance(bytes, start));
    }
}

/// Write the set-bit indexes of `bits` into `out`, low to high.
fn flatten_scalar(mut bits: u64, out: &mut [u8; 65]) -> usize {
    let mut n = 0;
    while bits != 0 {
        // SAFETY: a `u64` has at most 64 set bits, and `out` holds 65.
        unsafe {
            *out.get_unchecked_mut(n) = bits.trailing_zeros() as u8;
        }
        n += 1;
        bits &= bits.wrapping_sub(1);
    }
    n
}

#[cfg(target_arch = "x86_64")]
fn compress_available() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vbmi2")
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,popcnt")]
#[inline]
unsafe fn flatten_compress(bits: u64, out: &mut [u8; 65]) -> usize {
    use std::arch::x86_64::*;
    const IOTA: [u8; 64] = {
        let mut indexes = [0u8; 64];
        let mut i = 0;
        while i < 64 {
            indexes[i] = i as u8;
            i += 1;
        }
        indexes
    };
    unsafe {
        let iota = _mm512_loadu_si512(IOTA.as_ptr() as *const _);
        let packed = _mm512_maskz_compress_epi8(bits, iota);
        _mm512_storeu_si512(out.as_mut_ptr() as *mut _, packed);
    }
    bits.count_ones() as usize
}

/// Check each finished piece in `window` against [`Scheme::advance`].
#[cfg(debug_assertions)]
fn debug_assert_window<S: Scheme>(bytes: &[u8], pos: usize, window: Window, complete: usize) {
    let mut bits = window.starts;
    let mut prev = pos + bits.trailing_zeros() as usize;
    debug_assert_eq!(prev, pos);
    bits &= bits.wrapping_sub(1);
    for _ in 0..complete {
        let end = if bits == 0 {
            pos + 64
        } else {
            let end = pos + bits.trailing_zeros() as usize;
            bits &= bits.wrapping_sub(1);
            end
        };
        debug_assert_eq!(end, S::advance(bytes, prev), "mask piece diverged at {prev}");
        prev = end;
    }
}

#[cfg(not(debug_assertions))]
fn debug_assert_window<S: Scheme>(bytes: &[u8], pos: usize, window: Window, complete: usize) {
    let _ = (bytes, pos, window, complete);
}
