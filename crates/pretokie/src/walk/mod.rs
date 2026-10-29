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

    /// Next piece boundary at or after `pos`.
    ///
    /// The default keeps `pos`. A scheme that drops bytes, such as
    /// whitespace, skips them here. The result is `bytes.len()` when no
    /// piece remains.
    #[inline(always)]
    fn align(bytes: &[u8], pos: usize) -> usize {
        let _ = bytes;
        pos
    }

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
/// Bit 0 is the first byte of the window. Each later set bit in [`Window::starts`]
/// is another piece start. With [`Window::stops`] unset, each piece ends at
/// the next start, and [`Window::finishes`] means the last piece ends at
/// byte 64. With `stops` set, bit `k` is the exclusive end of a finished
/// piece. A piece that ends at byte 64 has no stop bit, so a gap can sit
/// between a piece's end and the next start.
///
/// When `finishes` is set, every start is a finished piece and the window
/// consumed all 64 bytes. Otherwise the highest start is where the next
/// call resumes, and that start is not a finished piece.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Window {
    pub starts: u64,
    pub stops: Option<u64>,
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
    /// Exclusive ends for a window whose pieces leave gaps.
    ///
    /// Parallel to the finished pieces in [`Self::positions`]. Unused when
    /// [`Self::gapped`] is false.
    ends: [u8; 64],
    /// The cached window reported [`Window::stops`].
    gapped: bool,
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
            ends: [0; 64],
            gapped: false,
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

    /// Write the next piece boundaries into `out` and return how many pieces
    /// were produced.
    ///
    /// `out[0]` is the absolute start of the first piece. `out[k]` for
    /// `k` in `1..=n` is the exclusive end of piece `k - 1`. `out` must
    /// hold `limit + 80` slots. A short count means the text is finished.
    pub(crate) fn bounds(&mut self, out: &mut [u32], limit: usize) -> usize {
        debug_assert!(out.len() >= limit.saturating_add(80));
        #[cfg(target_arch = "x86_64")]
        {
            if compress_available() {
                // SAFETY: `compress_available` checked AVX-512 F, BW, and VBMI2.
                return unsafe { bounds_avx512::<S>(self, out, limit) };
            }
            if classify::avx512_available() {
                // SAFETY: AVX-512 F and BW are present. Popcnt is present on
                // every such CPU.
                return unsafe { bounds_avx512_scalar::<S>(self, out, limit) };
            }
            if classify::avx2_available() {
                // SAFETY: AVX2 is present.
                return unsafe { bounds_avx2::<S>(self, out, limit) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is the baseline on aarch64.
            return unsafe { bounds_neon::<S>(self, out, limit) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        bounds_inner::<S, false, ISA_SCALAR>(self, out, limit)
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
            let ends = if walk.gapped { Some(&walk.ends[..]) } else { None };
            write_pairs(&mut out[n..], base, &walk.positions, ends, cursor, take);
            #[cfg(debug_assertions)]
            check_pairs::<S>(walk.bytes, base, &walk.positions, ends, cursor, take);
            let next = cursor + take;
            walk.cursor = next as u8;
            walk.pos = base + walk.positions[next] as usize;
            n += take;
            continue;
        }
        walk.len = 0;
        walk.gapped = false;
        walk.pos = S::align(walk.bytes, walk.pos);
        if walk.pos >= walk.bytes.len() {
            break;
        }
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
        let mut ends = [0u8; 64];
        let gapped = if let Some(stops) = window.stops {
            let pieces = count;
            let mut s = 0;
            let mut bits = stops;
            while bits != 0 {
                ends[s] = bits.trailing_zeros() as u8;
                s += 1;
                bits &= bits.wrapping_sub(1);
            }
            if window.finishes {
                positions[count] = 64;
                count += 1;
                if s + 1 == pieces {
                    ends[s] = 64;
                }
            }
            true
        } else {
            if window.finishes {
                positions[count] = 64;
                count += 1;
            }
            false
        };
        if count < 2 {
            let end = S::advance(walk.bytes, walk.pos);
            walk.write(out, n, end);
            n += 1;
            continue;
        }
        let base = walk.pos;
        let take = (count - 1).min(out.len() - n);
        let end_slice = if gapped { Some(&ends[..]) } else { None };
        write_pairs(&mut out[n..], base, &positions, end_slice, 0, take);
        #[cfg(debug_assertions)]
        check_pairs::<S>(walk.bytes, base, &positions, end_slice, 0, take);
        n += take;
        walk.pos = base + positions[take] as usize;
        if take + 1 < count {
            walk.positions = positions;
            walk.ends = ends;
            walk.gapped = gapped;
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

/// Same pieces as [`fill_inner`], written as absolute byte offsets.
///
/// The span expand stays out of this loop. A contiguous window becomes one
/// compressed store of boundary offsets; the caller packs those afterward.
#[inline(always)]
fn bounds_inner<S: Scheme, const COMPRESS: bool, const ISA: u8>(
    walk: &mut Walk<'_, S>,
    out: &mut [u32],
    limit: usize,
) -> usize {
    let mut n = 0usize;
    if limit == 0 || walk.pos >= walk.bytes.len() {
        return 0;
    }
    unsafe {
        *out.get_unchecked_mut(0) = walk.pos as u32;
    }
    while n < limit && walk.pos < walk.bytes.len() {
        walk.pos = S::align(walk.bytes, walk.pos);
        if walk.pos >= walk.bytes.len() {
            break;
        }
        debug_assert_eq!(walk.pos as u32, out[n]);
        let Some(window) = S::try_window::<ISA>(walk.bytes, walk.pos) else {
            n = push_scalar::<S>(walk, out, n);
            continue;
        };
        if window.stops.is_some() {
            n = push_scalar::<S>(walk, out, n);
            continue;
        }
        let abs = walk.pos as u32;
        // Issue the resume offset before the compress so the bit scan
        // overlaps the store, instead of waiting on it.
        let resume_off = if window.finishes {
            64
        } else {
            (63 - window.starts.leading_zeros()) as usize
        };
        let pop = if COMPRESS {
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: only `bounds_avx512` sets `COMPRESS`, and it requires
                // AVX-512 VBMI2. `out[n..n + 64]` is inside the caller's slack.
                unsafe { flatten_bounds(window.starts, abs, out.as_mut_ptr().add(n)) }
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                unreachable!()
            }
        } else {
            flatten_bounds_scalar(window.starts, abs, unsafe { out.as_mut_ptr().add(n) })
        };
        let mut boundaries = pop;
        if window.finishes {
            boundaries += 1;
            unsafe {
                *out.get_unchecked_mut(n + pop) = abs + 64;
            }
        }
        if boundaries < 2 {
            n = push_scalar::<S>(walk, out, n);
            continue;
        }
        let available = boundaries - 1;
        let take = available.min(limit - n);
        // The common path resumes on the last boundary. Reloading that lane
        // from the vector store just written does not forward, so derive it
        // from the bitset instead.
        if take == available {
            n += take;
            walk.pos += resume_off;
            debug_assert_eq!(walk.pos as u32, out[n]);
        } else {
            n += take;
            walk.pos = unsafe { *out.get_unchecked(n) } as usize;
            break;
        }
    }
    n
}

fn push_scalar<S: Scheme>(walk: &mut Walk<'_, S>, out: &mut [u32], n: usize) -> usize {
    let end = S::advance(walk.bytes, walk.pos);
    debug_assert!(end > walk.pos, "advance must consume at least one byte");
    walk.pos = end;
    unsafe {
        *out.get_unchecked_mut(n + 1) = end as u32;
    }
    n + 1
}

fn flatten_bounds_scalar(mut bits: u64, abs: u32, out: *mut u32) -> usize {
    let mut n = 0usize;
    while bits != 0 {
        let i = bits.trailing_zeros();
        unsafe {
            out.add(n).write(abs + i);
        }
        n += 1;
        bits &= bits.wrapping_sub(1);
    }
    n
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,popcnt")]
#[inline]
unsafe fn flatten_bounds(bits: u64, abs: u32, out: *mut u32) -> usize {
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
        let comp = _mm512_maskz_compress_epi8(bits, iota);
        let base = _mm512_set1_epi32(abs as i32);
        let low = _mm512_castsi512_si256(comp);
        let high = _mm512_extracti64x4_epi64::<1>(comp);
        let c0 = _mm256_castsi256_si128(low);
        let c1 = _mm256_extracti128_si256::<1>(low);
        let c2 = _mm256_castsi256_si128(high);
        let c3 = _mm256_extracti128_si256::<1>(high);
        let w0 = _mm512_add_epi32(_mm512_cvtepu8_epi32(c0), base);
        let w1 = _mm512_add_epi32(_mm512_cvtepu8_epi32(c1), base);
        let w2 = _mm512_add_epi32(_mm512_cvtepu8_epi32(c2), base);
        let w3 = _mm512_add_epi32(_mm512_cvtepu8_epi32(c3), base);
        _mm512_storeu_si512(out as *mut _, w0);
        _mm512_storeu_si512(out.add(16) as *mut _, w1);
        _mm512_storeu_si512(out.add(32) as *mut _, w2);
        _mm512_storeu_si512(out.add(48) as *mut _, w3);
    }
    bits.count_ones() as usize
}


#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2,popcnt")]
unsafe fn bounds_avx512<S: Scheme>(
    walk: &mut Walk<'_, S>,
    out: &mut [u32],
    limit: usize,
) -> usize {
    bounds_inner::<S, true, ISA_AVX512>(walk, out, limit)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,popcnt")]
unsafe fn bounds_avx512_scalar<S: Scheme>(
    walk: &mut Walk<'_, S>,
    out: &mut [u32],
    limit: usize,
) -> usize {
    bounds_inner::<S, false, ISA_AVX512>(walk, out, limit)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bounds_avx2<S: Scheme>(
    walk: &mut Walk<'_, S>,
    out: &mut [u32],
    limit: usize,
) -> usize {
    bounds_inner::<S, false, ISA_AVX2>(walk, out, limit)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn bounds_neon<S: Scheme>(
    walk: &mut Walk<'_, S>,
    out: &mut [u32],
    limit: usize,
) -> usize {
    bounds_inner::<S, false, ISA_NEON>(walk, out, limit)
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


/// Length in the high byte, then the bytes. `None` when `piece` is empty or
/// longer than 15 bytes.
#[inline(always)]
pub fn pack_piece(piece: &[u8]) -> Option<u128> {
    let n = piece.len();
    if n == 0 || n > 15 {
        return None;
    }
    let ptr = piece.as_ptr();
    let low = if (ptr as usize) & 4095 <= 4096 - 16 {
        let loaded = unsafe { (ptr as *const u128).read_unaligned() };
        let mask = (1u128 << (8 * n)) - 1;
        loaded & mask
    } else {
        let mut lanes = [0u8; 16];
        lanes[..n].copy_from_slice(piece);
        u128::from_le_bytes(lanes)
    };
    Some(low | ((n as u128) << 120))
}

#[inline(always)]
pub fn mix_piece<const CRC: bool>(key: u128) -> u64 {
    if CRC {
        crc32c(key)
    } else {
        let lo = key as u64;
        let hi = (key >> 64) as u64;
        let mut mixed = (lo ^ hi.rotate_right(25)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        mixed ^= mixed >> 32;
        mixed
    }
}

/// Same two `crc32` steps as `_mm_crc32_u64`. Written as inline asm so the
/// walk can emit them without a call. SSE4.2 is present wherever this crate
/// selects the CRC path.
#[inline(always)]
fn crc32c(key: u128) -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let mut hash = 0u64;
        core::arch::asm!(
            "crc32 {hash}, {lo}",
            "crc32 {hash}, {hi}",
            hash = inout(reg) hash,
            lo = in(reg) key as u64,
            hi = in(reg) (key >> 64) as u64,
            options(pure, nomem, nostack),
        );
        hash
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = key;
        0
    }
}

/// Count pieces from `walk`. `ISA` selects the inlined classifier.
#[inline(always)]
fn count_inner<S: Scheme, const ISA: u8>(mut walk: Walk<'_, S>, limit: usize) -> usize {
    let mut n: usize = 0;
    while walk.pos < walk.bytes.len() {
        walk.pos = S::align(walk.bytes, walk.pos);
        if walk.pos >= walk.bytes.len() {
            break;
        }
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
///
/// `ends` is set when a piece can stop before the next start. Otherwise the
/// next position entry is the end.
///
/// Scalar on purpose. A window holds about ten pieces, and the vectorized
/// expand spent as much time on its prologue as on the stores.
#[inline(always)]
fn write_pairs(
    out: &mut [Span],
    base: usize,
    positions: &[u8],
    ends: Option<&[u8]>,
    cursor: usize,
    take: usize,
) {
    debug_assert!(take <= out.len());
    debug_assert!(cursor + take < positions.len());
    let base = base as u32;
    let out = out.as_mut_ptr();
    let positions = positions.as_ptr();
    if let Some(ends) = ends {
        let ends = ends.as_ptr();
        let mut k = 0;
        while k < take {
            unsafe {
                let start = base + u32::from(*positions.add(cursor + k));
                let end = base + u32::from(*ends.add(cursor + k));
                out.add(k).write(Span { start, end });
            }
            k += 1;
        }
    } else {
        write_contiguous(out, base, positions, cursor, take);
    }
}

/// Piece `k` runs from `positions[cursor + k]` to `positions[cursor + k + 1]`.
#[inline(always)]
fn write_contiguous(out: *mut Span, base: u32, positions: *const u8, cursor: usize, take: usize) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::asm!(
            "test {take}, {take}",
            "jz 3f",
            "2:",
            "movzx {start:e}, byte ptr [{pos} + {k}]",
            "add {start:e}, {base:e}",
            "movzx {end:e}, byte ptr [{pos} + {k} + 1]",
            "add {end:e}, {base:e}",
            "mov dword ptr [{out} + {k}*8], {start:e}",
            "mov dword ptr [{out} + {k}*8 + 4], {end:e}",
            "inc {k}",
            "cmp {k}, {take}",
            "jb 2b",
            "3:",
            pos = in(reg) positions.add(cursor),
            out = in(reg) out,
            base = in(reg) base,
            take = in(reg) take,
            k = inout(reg) 0usize => _,
            start = out(reg) _,
            end = out(reg) _,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let mut k = 0;
        while k < take {
            unsafe {
                let start = base + u32::from(*positions.add(cursor + k));
                let end = base + u32::from(*positions.add(cursor + k + 1));
                out.add(k).write(Span { start, end });
            }
            k += 1;
        }
    }
}

#[cfg(debug_assertions)]
fn check_pairs<S: Scheme>(
    bytes: &[u8],
    base: usize,
    positions: &[u8],
    ends: Option<&[u8]>,
    cursor: usize,
    take: usize,
) {
    for k in 0..take {
        let start = base + positions[cursor + k] as usize;
        let end = match ends {
            Some(ends) => base + ends[cursor + k] as usize,
            None => base + positions[cursor + k + 1] as usize,
        };
        debug_assert!(end > start, "advance must consume at least one byte");
        let got = S::advance(bytes, start);
        debug_assert_eq!(
            end,
            got,
            "window {start}..{end} advance {start}..{got} around {:?}",
            bytes.get(start.saturating_sub(24)..end.max(got).saturating_add(24).min(bytes.len()))
        );
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
    if let Some(mut stops) = window.stops {
        for i in 0..complete {
            let end_rel = if stops == 0 {
                debug_assert!(window.finishes && i + 1 == complete);
                64
            } else {
                let end_rel = stops.trailing_zeros() as usize;
                stops &= stops.wrapping_sub(1);
                end_rel
            };
            let end = pos + end_rel;
            debug_assert!(end > prev, "advance must consume at least one byte");
            debug_assert_eq!(end, S::advance(bytes, prev), "mask piece diverged at {prev}");
            if bits != 0 {
                let next = pos + bits.trailing_zeros() as usize;
                debug_assert!(next >= end, "piece end passed the next start at {prev}");
                prev = next;
                bits &= bits.wrapping_sub(1);
            }
        }
        return;
    }
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

#[cfg(all(test, target_arch = "x86_64"))]
#[test]
fn crc32c_matches_the_intrinsic() {
    if !std::is_x86_feature_detected!("sse4.2") {
        return;
    }
    unsafe { crc32c_check() }
}

#[cfg(test)]
pub(crate) fn bounds_match_fill<S: Scheme>(text: &str) {
    let mut spans = vec![Span { start: 0, end: 0 }; text.len() + 1];
    let n = {
        let mut walk = Walk::<S>::new(text);
        let mut at = 0;
        let mut buf = [Span { start: 0, end: 0 }; 64];
        loop {
            let k = walk.fill(&mut buf);
            if k == 0 {
                break;
            }
            spans[at..at + k].copy_from_slice(&buf[..k]);
            at += k;
        }
        at
    };
    for limit in [1usize, 7, 64, 256] {
        let mut walk = Walk::<S>::new(text);
        let mut buf = vec![0u32; limit + 80];
        let mut got = Vec::new();
        loop {
            let k = walk.bounds(&mut buf, limit);
            if k == 0 {
                break;
            }
            for i in 0..k {
                got.push(Span {
                    start: buf[i],
                    end: buf[i + 1],
                });
            }
            if k < limit {
                break;
            }
        }
        assert_eq!(got.len(), n, "limit {limit} text len {}", text.len());
        assert_eq!(&got[..], &spans[..n], "limit {limit}");
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[target_feature(enable = "sse4.2")]
unsafe fn crc32c_check() {
    use core::arch::x86_64::_mm_crc32_u64;
    let key = 0x0102_0304_0506_0708_1112_1314_1516_1718u128;
    let intrinsic = _mm_crc32_u64(_mm_crc32_u64(0, key as u64), (key >> 64) as u64);
    assert_eq!(crc32c(key), intrinsic);
}
