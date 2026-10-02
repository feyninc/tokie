//! Mask-scanner pretokenizers: 64-byte batches are classified with SIMD
//! (NEON on aarch64; SSE2 or runtime-detected AVX2 on x86_64, see
//! [`MaskIsa`]) into per-byte class bitmasks, piece-start bits are derived with
//! shifted-mask algebra in scalar registers, and iteration pops one bit
//! per piece — no per-piece dispatch branches.
//!
//! `Mask<C>` produces byte-identical output to `Core<C>`, which remains
//! the scalar ground truth: regions the batch algebra cannot decide
//! locally — non-ASCII characters, apostrophe contractions, mixed
//! newline/space whitespace runs (CL100K pattern), DeepSeek control
//! chars, and batch-edge ambiguities — become "bad zones" whose pieces
//! are re-derived by running `Core` from the pending piece start. The
//! walker never emits a piece across an unresolved bad zone.
//!
//! On other targets (or `PRETOKIE_ISA=scalar`) every piece takes the
//! `Core` path (the walker starts with `scalar_until = usize::MAX`).

use std::marker::PhantomData;

use crate::core::config::*;
use crate::core::iter::Core;

/// End of the piece starting at `pos` per the scalar ground truth.
#[inline(always)]
fn scalar_advance<C: PretokConfig>(text: &str, pos: usize) -> usize {
    match Core::<C>::with_pos(text, pos).next() {
        Some(p) => pos + p.len(),
        None => pos,
    }
}

// =======================================================================
// Classification backend selection
// =======================================================================

/// Which byte-classification backend a [`Mask`] runs. Picked once per
/// `Mask` (see [`MaskIsa::detect`]); the batch algebra and walker are
/// monomorphized per backend, so the hot loop never re-dispatches.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskIsa {
    /// No SIMD: every piece takes the `Core` path.
    Scalar,
    /// Baseline 128-bit SIMD: NEON on aarch64, SSE2 on x86_64.
    Base,
    /// AVX2 (x86_64 only, runtime-detected).
    Avx2,
}

impl MaskIsa {
    /// Best backend for this CPU, cached. `PRETOKIE_ISA=scalar|base|avx2`
    /// overrides the choice (benchmarks / differential tests); an
    /// unsupported request falls back to the detected best.
    pub fn detect() -> MaskIsa {
        use std::sync::atomic::{AtomicU8, Ordering};
        static CACHED: AtomicU8 = AtomicU8::new(0);
        match CACHED.load(Ordering::Relaxed) {
            1 => return MaskIsa::Scalar,
            2 => return MaskIsa::Base,
            3 => return MaskIsa::Avx2,
            _ => {}
        }
        // AVX2 measured no faster than SSE2 end-to-end (classification is not
        // the bottleneck), so the baseline backend is the default and AVX2 is
        // opt-in via PRETOKIE_ISA=avx2.
        let best = if Self::Base.supported() {
            MaskIsa::Base
        } else {
            MaskIsa::Scalar
        };
        let isa = match std::env::var("PRETOKIE_ISA").as_deref() {
            Ok("scalar") => MaskIsa::Scalar,
            Ok("base") | Ok("sse2") | Ok("neon") if Self::Base.supported() => MaskIsa::Base,
            Ok("avx2") if Self::Avx2.supported() => MaskIsa::Avx2,
            _ => best,
        };
        CACHED.store(isa as u8 + 1, Ordering::Relaxed);
        isa
    }

    /// Whether this backend can run on the current CPU.
    pub fn supported(self) -> bool {
        match self {
            MaskIsa::Scalar => true,
            MaskIsa::Base => cfg!(any(target_arch = "aarch64", target_arch = "x86_64")),
            MaskIsa::Avx2 => {
                #[cfg(target_arch = "x86_64")]
                {
                    std::arch::is_x86_feature_detected!("avx2")
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    false
                }
            }
        }
    }
}

/// A byte-classification backend: fills [`AsciiMasks`] for one 64-byte
/// batch. `batch` is the per-batch entry the walker calls; backends that
/// need a `#[target_feature]` context override it with a trampoline.
trait Classify {
    fn ascii_masks(
        bytes: &[u8],
        scan: usize,
        case_masks: bool,
        slash_mask: bool,
        ctl_mask: bool,
    ) -> AsciiMasks;

    #[inline(always)]
    fn batch<C: PretokConfig>(bytes: &[u8], scan: usize) -> (u64, u64)
    where
        Self: Sized,
    {
        batch_masks::<C, Self>(bytes, scan)
    }
}

/// Baseline 128-bit backend (NEON / SSE2; both are architectural
/// baselines, so no runtime detection).
struct Base;

impl Classify for Base {
    #[inline(always)]
    fn ascii_masks(
        bytes: &[u8],
        scan: usize,
        case_masks: bool,
        slash_mask: bool,
        ctl_mask: bool,
    ) -> AsciiMasks {
        #[cfg(target_arch = "aarch64")]
        {
            neon_ascii_masks(bytes, scan, case_masks, slash_mask, ctl_mask)
        }
        #[cfg(target_arch = "x86_64")]
        {
            sse2_ascii_masks(bytes, scan, case_masks, slash_mask, ctl_mask)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = (bytes, scan, case_masks, slash_mask, ctl_mask);
            unreachable!("MaskIsa::Base is unsupported on this target")
        }
    }
}

/// AVX2 backend. Only instantiated after [`MaskIsa::Avx2`] was detected.
struct Avx2;

impl Classify for Avx2 {
    #[inline(always)]
    fn ascii_masks(
        bytes: &[u8],
        scan: usize,
        case_masks: bool,
        slash_mask: bool,
        ctl_mask: bool,
    ) -> AsciiMasks {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: an `Avx2` walker is only built when AVX2 was detected.
            unsafe { avx2_ascii_masks(bytes, scan, case_masks, slash_mask, ctl_mask) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = (bytes, scan, case_masks, slash_mask, ctl_mask);
            unreachable!("MaskIsa::Avx2 is unsupported on this target")
        }
    }

    /// Trampoline into an AVX2 codegen context. When the walker itself
    /// runs inside one (`for_each_piece`), this inlines away; from the
    /// plain iterator it is one call per 64-byte batch.
    #[inline(always)]
    fn batch<C: PretokConfig>(bytes: &[u8], scan: usize) -> (u64, u64) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: as above.
            unsafe { avx2_batch_masks::<C>(bytes, scan) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            batch_masks::<C, Self>(bytes, scan)
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_batch_masks<C: PretokConfig>(bytes: &[u8], scan: usize) -> (u64, u64) {
    batch_masks::<C, Avx2>(bytes, scan)
}

// =======================================================================
// NEON classification (aarch64)
// =======================================================================

/// One u64 mask per byte predicate for a 64-byte batch (bit i = byte
/// scan+i). These are `Core`'s byte classes, not Unicode's: whitespace
/// is exactly {space, tab, \r, \n} and everything else ASCII that is
/// not a letter/digit is "punct" (including \x0b, \x0c and controls),
/// matching `Core`'s scalar predicates.
#[derive(Clone, Copy, Default)]
struct AsciiMasks {
    /// ASCII letters.
    l: u64,
    /// ASCII digits.
    d: u64,
    /// Space (0x20).
    s: u64,
    /// Tab (0x09).
    t: u64,
    /// Newlines: \r, \n.
    n: u64,
    /// Non-ASCII bytes (>= 0x80).
    hi: u64,
    /// Apostrophes.
    ap: u64,
    /// ASCII lowercase letters (only filled when `case_masks`).
    lo: u64,
    /// Forward slashes (only filled when `slash_mask`).
    slash: u64,
    /// Bytes < 0x21 or == 0x7F (only filled when `ctl_mask`): the ASCII
    /// range that is outside DeepSeek's [\p{P}\p{S}] class once
    /// whitespace is removed.
    low_ctl: u64,
}

/// simdjson-style movemask: 4 mask vectors (64 lanes of 0x00/0xFF) -> u64,
/// bit i = lane i. The 4-`addp` reduction tree is pinned as asm: written
/// with `vpaddq_u8`, LLVM rewrites each pairwise add into a
/// uzp1/uzp2/orr triple and the 9-op form becomes 17 ops.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(unused_assignments)] // a2 is an asm scratch register
unsafe fn movemask64(
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    v3: std::arch::aarch64::uint8x16_t,
) -> u64 {
    use std::arch::aarch64::*;
    unsafe {
        const W: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
        let w = vld1q_u8(W.as_ptr());
        let mut a0 = vandq_u8(v0, w);
        let a1 = vandq_u8(v1, w);
        let mut a2 = vandq_u8(v2, w);
        let a3 = vandq_u8(v3, w);
        core::arch::asm!(
            "addp {a0:v}.16b, {a0:v}.16b, {a1:v}.16b",
            "addp {a2:v}.16b, {a2:v}.16b, {a3:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a2:v}.16b",
            "addp {a0:v}.16b, {a0:v}.16b, {a0:v}.16b",
            a0 = inout(vreg) a0,
            a1 = in(vreg) a1,
            a2 = inout(vreg) a2,
            a3 = in(vreg) a3,
            options(pure, nomem, nostack, preserves_flags),
        );
        vgetq_lane_u64::<0>(vreinterpretq_u64_u8(a0))
    }
}

/// Classify `bytes[scan..scan+64]` (requires `scan + 64 <= bytes.len()`).
/// The optional masks are computed only when the config needs them; the
/// flags are compile-time constants after monomorphization.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn neon_ascii_masks(
    bytes: &[u8],
    scan: usize,
    case_masks: bool,
    slash_mask: bool,
    ctl_mask: bool,
) -> AsciiMasks {
    use std::arch::aarch64::*;
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let zero = vdupq_n_u8(0);
        let mut l = [zero; 4];
        let mut d = [zero; 4];
        let mut s = [zero; 4];
        let mut t = [zero; 4];
        let mut n = [zero; 4];
        let mut hi = [zero; 4];
        let mut ap = [zero; 4];
        let mut lo = [zero; 4];
        let mut sl = [zero; 4];
        let mut lc = [zero; 4];
        for i in 0..4 {
            let v = vld1q_u8(p.add(16 * i));
            let lowered = vorrq_u8(v, vdupq_n_u8(0x20));
            l[i] = vcleq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(25));
            d[i] = vcleq_u8(vsubq_u8(v, vdupq_n_u8(b'0')), vdupq_n_u8(9));
            s[i] = vceqq_u8(v, vdupq_n_u8(b' '));
            t[i] = vceqq_u8(v, vdupq_n_u8(b'\t'));
            n[i] = vorrq_u8(
                vceqq_u8(v, vdupq_n_u8(b'\r')),
                vceqq_u8(v, vdupq_n_u8(b'\n')),
            );
            hi[i] = vcltzq_s8(vreinterpretq_s8_u8(v));
            ap[i] = vceqq_u8(v, vdupq_n_u8(b'\''));
            if case_masks {
                lo[i] = vandq_u8(l[i], vtstq_u8(v, vdupq_n_u8(0x20)));
            }
            if slash_mask {
                sl[i] = vceqq_u8(v, vdupq_n_u8(b'/'));
            }
            if ctl_mask {
                lc[i] = vorrq_u8(
                    vcltq_u8(v, vdupq_n_u8(0x21)),
                    vceqq_u8(v, vdupq_n_u8(0x7F)),
                );
            }
        }
        AsciiMasks {
            l: movemask64(l[0], l[1], l[2], l[3]),
            d: movemask64(d[0], d[1], d[2], d[3]),
            s: movemask64(s[0], s[1], s[2], s[3]),
            t: movemask64(t[0], t[1], t[2], t[3]),
            n: movemask64(n[0], n[1], n[2], n[3]),
            hi: movemask64(hi[0], hi[1], hi[2], hi[3]),
            ap: movemask64(ap[0], ap[1], ap[2], ap[3]),
            lo: if case_masks { movemask64(lo[0], lo[1], lo[2], lo[3]) } else { 0 },
            slash: if slash_mask { movemask64(sl[0], sl[1], sl[2], sl[3]) } else { 0 },
            low_ctl: if ctl_mask { movemask64(lc[0], lc[1], lc[2], lc[3]) } else { 0 },
        }
    }
}

// =======================================================================
// x86_64 classification (SSE2 baseline, AVX2 runtime-detected)
// =======================================================================
//
// Same predicates as `neon_ascii_masks`. x86 has no unsigned byte `<=`,
// so `x <= k` is `min_epu8(x, k) == x`; the high-bit test is the raw
// movemask of the input.

/// SSE2 classification of `bytes[scan..scan+64]` (requires
/// `scan + 64 <= bytes.len()`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn sse2_ascii_masks(
    bytes: &[u8],
    scan: usize,
    case_masks: bool,
    slash_mask: bool,
    ctl_mask: bool,
) -> AsciiMasks {
    use std::arch::x86_64::*;
    debug_assert!(scan + 64 <= bytes.len());
    let mut m = AsciiMasks::default();
    // SAFETY: SSE2 is part of the x86_64 baseline; the four 16-byte
    // unaligned loads stay inside `bytes[scan..scan+64]`.
    unsafe {
        let p = bytes.as_ptr().add(scan);
        let set = |b: u8| _mm_set1_epi8(b as i8);
        let le = |x: __m128i, k: u8| _mm_cmpeq_epi8(_mm_min_epu8(x, set(k)), x);
        for i in 0..4 {
            let v = _mm_loadu_si128(p.add(16 * i) as *const __m128i);
            let sh = 16 * i as u32;
            let mv = |x: __m128i| (_mm_movemask_epi8(x) as u32 as u64) << sh;
            let lowered = _mm_or_si128(v, set(0x20));
            let l = le(_mm_sub_epi8(lowered, set(b'a')), 25);
            m.l |= mv(l);
            m.d |= mv(le(_mm_sub_epi8(v, set(b'0')), 9));
            m.s |= mv(_mm_cmpeq_epi8(v, set(b' ')));
            m.t |= mv(_mm_cmpeq_epi8(v, set(b'\t')));
            m.n |= mv(_mm_or_si128(
                _mm_cmpeq_epi8(v, set(b'\r')),
                _mm_cmpeq_epi8(v, set(b'\n')),
            ));
            m.hi |= mv(v);
            m.ap |= mv(_mm_cmpeq_epi8(v, set(b'\'')));
            if case_masks {
                let bit = _mm_cmpeq_epi8(_mm_and_si128(v, set(0x20)), set(0x20));
                m.lo |= mv(_mm_and_si128(l, bit));
            }
            if slash_mask {
                m.slash |= mv(_mm_cmpeq_epi8(v, set(b'/')));
            }
            if ctl_mask {
                m.low_ctl |= mv(_mm_or_si128(le(v, 0x20), _mm_cmpeq_epi8(v, set(0x7F))));
            }
        }
    }
    m
}

/// AVX2 classification of `bytes[scan..scan+64]`: two 32-byte lanes.
///
/// # Safety
/// The CPU must support AVX2; `scan + 64 <= bytes.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_ascii_masks(
    bytes: &[u8],
    scan: usize,
    case_masks: bool,
    slash_mask: bool,
    ctl_mask: bool,
) -> AsciiMasks {
    use std::arch::x86_64::*;
    debug_assert!(scan + 64 <= bytes.len());
    let mut m = AsciiMasks::default();
    let p = bytes.as_ptr().add(scan);
    // Macros rather than closures: a closure is its own function and does
    // not reliably inherit this `#[target_feature]`, which would turn each
    // intrinsic into an out-of-line call.
    macro_rules! set {
        ($b:expr) => {
            _mm256_set1_epi8($b as i8)
        };
    }
    macro_rules! le {
        ($x:expr, $k:expr) => {{
            let x = $x;
            _mm256_cmpeq_epi8(_mm256_min_epu8(x, set!($k)), x)
        }};
    }
    for i in 0..2 {
        let v = _mm256_loadu_si256(p.add(32 * i) as *const __m256i);
        let sh = 32 * i as u32;
        macro_rules! mv {
            ($x:expr) => {
                (_mm256_movemask_epi8($x) as u32 as u64) << sh
            };
        }
        let lowered = _mm256_or_si256(v, set!(0x20));
        let l = le!(_mm256_sub_epi8(lowered, set!(b'a')), 25);
        m.l |= mv!(l);
        m.d |= mv!(le!(_mm256_sub_epi8(v, set!(b'0')), 9));
        m.s |= mv!(_mm256_cmpeq_epi8(v, set!(b' ')));
        m.t |= mv!(_mm256_cmpeq_epi8(v, set!(b'\t')));
        m.n |= mv!(_mm256_or_si256(
            _mm256_cmpeq_epi8(v, set!(b'\r')),
            _mm256_cmpeq_epi8(v, set!(b'\n')),
        ));
        m.hi |= mv!(v);
        m.ap |= mv!(_mm256_cmpeq_epi8(v, set!(b'\'')));
        if case_masks {
            let bit = _mm256_cmpeq_epi8(_mm256_and_si256(v, set!(0x20)), set!(0x20));
            m.lo |= mv!(_mm256_and_si256(l, bit));
        }
        if slash_mask {
            m.slash |= mv!(_mm256_cmpeq_epi8(v, set!(b'/')));
        }
        if ctl_mask {
            m.low_ctl |= mv!(_mm256_or_si256(le!(v, 0x20), _mm256_cmpeq_epi8(v, set!(0x7F))));
        }
    }
    m
}

// =======================================================================
// Bit-domain helpers (platform-independent)
// =======================================================================

/// Kogge-Stone rightward fill: propagate `seed` bits toward higher bit
/// positions through contiguous runs of `mask`.
#[inline(always)]
fn fill_right(seed: u64, mask: u64) -> u64 {
    let mut x = seed & mask;
    if x == 0 {
        return 0;
    }
    let c2 = mask & (mask << 1);
    let c4 = c2 & (c2 << 2);
    let c8 = c4 & (c4 << 4);
    let c16 = c8 & (c8 << 8);
    let c32 = c16 & (c16 << 16);
    x |= (x << 1) & mask;
    x |= (x << 2) & c2;
    x |= (x << 4) & c4;
    x |= (x << 8) & c8;
    x |= (x << 16) & c16;
    x |= (x << 32) & c32;
    x
}

/// Leftward counterpart of [`fill_right`].
#[inline(always)]
fn fill_left(seed: u64, mask: u64) -> u64 {
    let mut x = seed & mask;
    if x == 0 {
        return 0;
    }
    let c2 = mask & (mask >> 1);
    let c4 = c2 & (c2 >> 2);
    let c8 = c4 & (c4 >> 4);
    let c16 = c8 & (c8 >> 8);
    let c32 = c16 & (c16 >> 16);
    x |= (x >> 1) & mask;
    x |= (x >> 2) & c2;
    x |= (x >> 4) & c4;
    x |= (x >> 8) & c8;
    x |= (x >> 16) & c16;
    x |= (x >> 32) & c32;
    x
}

/// Fill whole `mask` runs containing a `seed` bit, in both directions.
#[inline(always)]
fn fill_both(seed: u64, mask: u64) -> u64 {
    fill_right(seed, mask) | fill_left(seed, mask)
}

/// Piece-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run
/// splits into 3-char pieces, so starts sit at run start + 3k.
#[inline(always)]
fn digit_run_splits3(d: u64) -> u64 {
    let mut b = d & !(d << 1); // run starts
    // A start at p re-arms at p+3 while the run continues: hop condition
    // c = "p..p+3 all digits". Log-doubling covers 64-bit runs in 5 steps.
    let mut c = d & (d >> 1) & (d >> 2) & (d >> 3);
    let mut sh = 3u32;
    while sh < 64 {
        b |= (b & c) << sh;
        c &= c >> sh;
        sh <<= 1;
    }
    b
}

/// Mask of the contiguous digit run starting at bit 0 (caller checks
/// `d & 1 != 0`).
#[inline(always)]
fn leading_run(d: u64) -> u64 {
    let tz = (!d).trailing_zeros();
    if tz >= 64 { u64::MAX } else { (1u64 << tz) - 1 }
}

#[inline(always)]
fn is_ascii_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

use crate::util::is_cjk_char;

// =======================================================================
// Unicode in-mask classification
// =======================================================================

/// UTF-8 sequence length from a lead byte (valid UTF-8 assumed).
#[inline(always)]
fn utf8_len(b: u8) -> usize {
    if b < 0x80 { 1 } else if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4 }
}

/// Effective class of a non-ASCII char under config `C`, mirroring
/// `Core`'s unicode predicates exactly. `Defer` marks chars whose piece
/// behavior the byte algebra cannot model:
/// - whitespace (run-global `\s+(?!\S)` / `\s*[\r\n]` bookkeeping),
/// - letters/marks under CamelCase (case-state dependent),
/// - numerics under Chunked3 (`\p{N}{1,3}` counts chars, masks count bytes),
/// - DeepSeek non-[\p{P}\p{S}] chars (the one-char letter-prefix rule).
enum UClass {
    Letter,
    Number,
    Punct,
    Defer,
}

#[inline(always)]
fn uclass<C: PretokConfig>(ch: char) -> UClass {
    use crate::util::{is_punct_or_symbol, is_unicode_letter, is_unicode_mark};
    let letter = match C::LETTER_MODE {
        LetterMode::Plain => is_unicode_letter(ch),
        LetterMode::PlainWithMarks | LetterMode::CamelCase => {
            crate::util::is_alpha(ch) || is_unicode_mark(ch)
        }
    };
    if letter {
        return if C::LETTER_MODE == LetterMode::CamelCase {
            UClass::Defer
        } else {
            UClass::Letter
        };
    }
    if ch.is_numeric() {
        return if C::DIGIT_MODE == DigitMode::Chunked3 {
            UClass::Defer
        } else {
            UClass::Number
        };
    }
    if ch.is_whitespace() {
        return UClass::Defer;
    }
    if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly {
        if is_punct_or_symbol(ch) { UClass::Punct } else { UClass::Defer }
    } else {
        UClass::Punct
    }
}

/// Per-byte class masks for a batch's non-ASCII chars: every byte of a
/// classified char carries the char's class, so byte-adjacency equals
/// char-adjacency and the u64 boundary algebra applies unchanged.
#[derive(Clone, Copy, Default)]
struct UniMasks {
    l: u64,
    d: u64,
    /// Lead bytes of Number chars (Single-mode piece starts).
    d_lead: u64,
    o: u64,
    /// Lead bytes of Punct chars (letter-absorb sources).
    o_lead: u64,
    /// Lead bytes of Letter chars (AsciiOnly prefix-run boundaries).
    l_lead: u64,
    /// Lead bytes of CJK chars (DeepSeek ws-split exception).
    cjk_lead: u64,
    /// All bytes of CJK chars (DeepSeek's pre-split runs).
    cjk: u64,
    /// Bytes only the scalar path can decide.
    defer: u64,
}

/// Classify every non-ASCII char whose lead bit is in `m` for
/// `bytes[scan..scan+64]`. A char spilling off the batch end is
/// classified via its full in-text bytes (valid UTF-8 keeps the read in
/// bounds); only its in-batch bytes get class bits.
#[inline(never)] // keep the clean ASCII path's register allocation intact
fn classify_uni<C: PretokConfig>(bytes: &[u8], scan: usize, m: u64) -> UniMasks {
    let mut u = UniMasks::default();
    let mut m = m;
    while m != 0 {
        let i = m.trailing_zeros() as usize;
        let b = bytes[scan + i];
        debug_assert!(b & 0xC0 != 0x80, "unclaimed continuation byte at bit {i}");
        let (ch, clen) = crate::util::decode_utf8(&bytes[scan + i..]);
        let lead = 1u64 << i;
        let chm = if clen >= 64 - i {
            u64::MAX << i
        } else {
            ((1u64 << clen) - 1) << i
        };
        match uclass::<C>(ch) {
            UClass::Letter => {
                u.l |= chm;
                u.l_lead |= lead;
            }
            UClass::Number => {
                u.d |= chm;
                u.d_lead |= lead;
            }
            UClass::Punct => {
                u.o |= chm;
                u.o_lead |= lead;
            }
            UClass::Defer => u.defer |= chm,
        }
        if C::WS_EXCEPTION == WsException::Cjk && is_cjk_char(ch) {
            u.cjk_lead |= lead;
            u.cjk |= chm;
        }
        m &= !chm;
    }
    u
}

// =======================================================================
// Per-batch boundary algebra, parameterized by PretokConfig
// =======================================================================

/// Piece-start (`usable`) and scalar-territory (`bad`) bitmasks for
/// `bytes[scan..scan+64]`. Bit k of `usable` = a trustworthy piece start
/// at `scan + k`; `bad` marks bytes whose boundaries `Core` re-derives,
/// and no piece is emitted across an unresolved bad zone.
#[inline(always)]
fn batch_masks<C: PretokConfig, K: Classify>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // One byte of lookahead is required for the bit-63 whitespace-split
    // test (reading a whole char there is safe: input is valid UTF-8, so
    // a char starting in-bounds is fully in-bounds).
    if scan + 65 > bytes.len() {
        return (0, u64::MAX);
    }

    let camel = C::LETTER_MODE == LetterMode::CamelCase;
    let want_slash = C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes;
    let want_ctl = C::PUNCT_CLASS == PunctClass::PunctSymbolOnly;
    let m = K::ascii_masks(bytes, scan, camel, want_slash, want_ctl);

    let sp = m.s | m.t;
    // A space prefixes letters and punct (and digits under GPT-2); a tab
    // prefixes letters only (`[^\r\n\p{L}\p{N}]?\p{L}+`), and nothing
    // under GPT-2's ` ?\p{L}+` (SpaceOnly configs).
    let tab_pfx = C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly;
    let n = m.n;
    let ws = sp | n;
    let hi = m.hi;
    let ap = m.ap;

    let mut bad = 0u64;

    // ---- carries from the char before the batch ----
    // For a non-ASCII previous char, walk back to its lead (at most 3
    // bytes), classify it, and claim any of its bytes that straddle into
    // this batch — so a batch following a unicode char keeps its fast
    // path instead of starting in a bad zone.
    let (pl, pd, psp, pws, po, plo, pn);
    let (mut claim_l, mut claim_d, mut claim_o) = (0u64, 0u64, 0u64);
    let (mut pcj, mut claim_cj) = (0u64, 0u64);
    let mut claimed = 0u64;
    let mut edge_kill = 0u64;
    if scan == 0 {
        (pl, pd, psp, pws, po, plo, pn) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    } else {
        let b = bytes[scan - 1];
        if b >= 0x80 {
            let mut j = scan - 1;
            while bytes[j] & 0xC0 == 0x80 {
                j -= 1;
            }
            let (ch, clen) = crate::util::decode_utf8(&bytes[j..]);
            let end = j + clen;
            let claim_bits = if end > scan {
                (1u64 << (end - scan)) - 1
            } else {
                0
            };
            claimed = claim_bits;
            if C::WS_EXCEPTION == WsException::Cjk && is_cjk_char(ch) {
                (pcj, claim_cj) = (1, claim_bits);
            }
            match uclass::<C>(ch) {
                UClass::Letter => {
                    claim_l = claim_bits;
                    (pl, pd, psp, pws, po, plo, pn) = (1, 0, 0, 0, 0, 0, 0);
                }
                UClass::Number => {
                    claim_d = claim_bits;
                    (pl, pd, psp, pws, po, plo, pn) = (0, 1, 0, 0, 0, 0, 0);
                }
                UClass::Punct => {
                    claim_o = claim_bits;
                    (pl, pd, psp, pws, po, plo, pn) = (0, 0, 0, 0, 1, 0, 0);
                    // Cross-edge letter absorb: a letter right after this
                    // punct char is absorbed iff the punct char itself
                    // starts a piece — decided by the char BEFORE it.
                    let letter_bit = 1u64 << (end - scan);
                    if C::PUNCT_PREFIX_MODE == PunctPrefixMode::Any
                        && (m.l | hi) & letter_bit != 0
                    {
                        if j == 0 {
                            edge_kill = letter_bit; // text-start punct absorbs
                        } else if bytes[j - 1] < 0x80 {
                            let b2 = bytes[j - 1];
                            let o2 = !crate::util::is_ascii_letter(b2)
                                && !crate::util::is_digit(b2)
                                && !is_ascii_ws_byte(b2);
                            let sp2 = b2 == b' '; // a tab never prefixes punct
                            if !o2 && !sp2 {
                                edge_kill = letter_bit; // fresh punct piece absorbs
                            }
                            // else: run member / space-prefixed → no absorb
                        } else {
                            bad |= claim_bits | letter_bit | letter_bit << 1;
                        }
                    }
                }
                UClass::Defer => {
                    bad |= claim_bits | 0b111;
                    (pl, pd, psp, pws, po, plo, pn) = (0, 0, 0, 0, 0, 0, 0);
                }
            }
        } else {
            let letter = crate::util::is_ascii_letter(b);
            let digit = crate::util::is_digit(b);
            let wsb = is_ascii_ws_byte(b);
            pl = u64::from(letter);
            pd = u64::from(digit);
            psp = u64::from(b == b' ' || b == b'\t');
            pws = u64::from(wsb);
            po = u64::from(!letter && !digit && !wsb);
            plo = u64::from(crate::util::is_lower(b));
            pn = u64::from(b == b'\n' || b == b'\r');
        }
    }

    // ---- classify this batch's non-ASCII chars ----
    let uni = if hi & !claimed != 0 {
        classify_uni::<C>(bytes, scan, hi & !claimed)
    } else {
        UniMasks::default()
    };
    let defer = uni.defer | (bad & claimed);

    // Merged per-byte effective classes: every byte of a classified char
    // carries the char's class, so the ASCII shifted-mask algebra applies
    // unchanged. Deferred bytes belong to no class.
    let l = m.l | uni.l | claim_l;
    let d = m.d | uni.d | claim_d;
    let o = !(m.l | m.d | ws | hi) | uni.o | claim_o;

    // ---- content piece-start bits ----
    let mut start = 0u64;

    // letters
    start |= l & !((l << 1) | pl);
    if camel {
        // CamelCase: an uppercase directly after a lowercase starts a piece
        let lo = m.lo;
        let up = l & !lo;
        start |= up & ((lo << 1) | plo);
    }

    // digits
    match C::DIGIT_MODE {
        DigitMode::Unlimited => start |= d & !((d << 1) | pd),
        // Single: every numeric CHAR is a piece — lead bytes only.
        DigitMode::Single => start |= m.d | uni.d_lead,
        DigitMode::Chunked3 => {
            // Unicode numerics are deferred under Chunked3, so d is pure
            // ASCII here and byte splits equal char splits.
            start |= digit_run_splits3(d);
            // A run carried in from the previous batch has unknown phase.
            if pd == 1 && d & 1 != 0 {
                bad |= leading_run(d);
            }
        }
    }

    // punct
    start |= o & !((o << 1) | po);

    // DeepSeek: a CJK run is a segment of its own — split at both edges.
    let cj = uni.cjk | claim_cj;
    if C::WS_EXCEPTION == WsException::Cjk {
        let cjp = (cj << 1) | pcj;
        start |= (cj & !cjp) | ((l | o | d) & !cj & cjp);
    }

    // ---- punct-run trailing newlines (and slashes for O200K) ----
    // '/' (O200K) is punct-class, so it extends the main punct run on
    // its own; it participates in the trailing tail only after a
    // consumed newline. The seed is therefore newline-after-punct.
    let trail = if want_slash { n | m.slash } else { n };
    let tn = if C::PUNCT_TRAILING == PunctTrailing::None {
        0
    } else {
        // (CJK punct ends its segment: "゛\n" → ゛ \n.)
        fill_right(n & (((o & !cj) << 1) | (po & !pcj)), trail)
    };
    // A punct char directly after a consumed trailing run starts a fresh
    // piece even though its predecessor byte is also punct-class.
    start |= o & (tn << 1);
    // With slashes in the tail, a trailing run reaching the batch edge is
    // ambiguous for the next batch (a leading '/' there can't tell a
    // trailing tail from a fresh punct run); defer to the scalar path.
    // Newline-only tails are safe: a leading newline emits no bit under
    // either interpretation.
    if want_slash && tn >> 63 != 0 {
        bad |= 1 << 63;
    }
    // A newline directly after a deferred char may or may not start a
    // trailing tail (the char's punct-ness is undecided). For
    // newline-only tails both interpretations emit the same bits, but a
    // slash in the chain changes whether the next punct char starts
    // fresh — and with it the single-punct letter absorb one byte later.
    if want_slash {
        let chain = fill_right(n & (defer << 1), trail);
        if chain != 0 {
            bad |= chain | chain << 1 | chain << 2;
        }
        // A leading trail run continuing from the previous batch (prev
        // char is a newline) is equally undecidable from local carries.
        if pn == 1 && trail & 1 != 0 {
            let lead = fill_right(1, trail);
            bad |= lead | lead << 1 | lead << 2;
        }
        // A '/' as the previous batch's last byte may have been a punct-run
        // member or a trailing-tail end; the distinction decides whether a
        // punct at bit 0 starts fresh (and absorbs a following letter).
        if scan > 0 && bytes[scan - 1] == b'/' {
            let lead = fill_right(1, trail);
            bad |= lead | lead << 1 | lead << 2 | 0b11;
        }
    }

    // ---- whitespace piece-start bits ----
    let ews = ws & !tn;
    let pwsb = (ews << 1) | pws;
    let run_start = ews & !pwsb;

    // Split base: which run members may take the "split before last ws
    // char" bit. GPT-2 pattern: any ws; CL100K pattern: space/tab only
    // (pure-newline runs are a single piece).
    let split_base = if C::WS_PATTERN == WsPattern::Gpt2 { ews } else { ews & !n };
    let mut split = split_base & !(ws >> 1);
    if C::WS_EXCEPTION != WsException::None {
        // SmolLM: stay whole before a numeric (incl. classified unicode
        // numerics); DeepSeek: before an ASCII digit or a CJK char.
        split &= !(d >> 1);
        if C::WS_EXCEPTION == WsException::Cjk {
            split &= !(uni.cjk_lead >> 1);
            // Unicode numerics (`½`, `¹`) are deferred under Chunked3, but
            // an earlier DeepSeek stage splits them off too, so a ws run
            // before one may stay whole: leave that run to `Core`.
            let pre = ews & (defer >> 1);
            if pre != 0 {
                bad |= fill_both(pre, ews) | pre << 1;
            }
        }
    }
    // Bit 63 needs the real lookahead char.
    if split_base >> 63 != 0 {
        let nb64 = bytes[scan + 64];
        let keep = if nb64 < 0x80 {
            !is_ascii_ws_byte(nb64)
                && !(C::WS_EXCEPTION != WsException::None && crate::util::is_digit(nb64))
        } else {
            let (ch, _) = crate::util::decode_utf8(&bytes[scan + 64..]);
            !ch.is_whitespace()
                && match C::WS_EXCEPTION {
                    WsException::None => true,
                    WsException::Digits => !ch.is_numeric(),
                    WsException::Cjk => !is_cjk_char(ch) && !ch.is_numeric(),
                }
        };
        split = (split & !(1 << 63)) | ((u64::from(keep) << 63) & split_base);
    }

    let wsb_bits = run_start | split;

    // CL100K pattern: runs mixing newlines and space/tab are scalar
    // territory (the `\s*[\r\n]` interaction is run-global).
    if C::WS_PATTERN == WsPattern::Cl100k {
        let nlr = n & !tn;
        let spr = ews & !n;
        let mut mixed = (nlr & ((spr << 1) | (spr >> 1))) | (spr & ((nlr << 1) | (nlr >> 1)));
        // Cross-batch adjacency (conservative: also covers a preceding
        // trailing-consumed newline, which local masks cannot know).
        mixed |= (nlr & 1) & psp;
        mixed |= (spr & 1) & pn;
        if mixed != 0 {
            bad |= fill_both(mixed, ews);
        }
    }

    // ---- suppression: space/tab prefixes the following content ----
    // Only a space that itself starts a piece absorbs what follows; a
    // run-interior space (possible under ws exceptions, e.g. DeepSeek's
    // "no split before CJK") does not.
    let (ps, pt) = if scan == 0 {
        (0u64, 0u64)
    } else {
        (u64::from(bytes[scan - 1] == b' '), u64::from(bytes[scan - 1] == b'\t'))
    };
    let pfx_bits = run_start | split;
    let after_sp_classes =
        l | o | if C::SPACE_PREFIXES_DIGITS { d } else { 0 };
    let mut after_sp = (((m.s & pfx_bits) << 1) | ps) & after_sp_classes;
    if tab_pfx {
        after_sp |= (((m.t & pfx_bits) << 1) | pt) & l;
    }
    after_sp &= !cj; // nothing prefixes a CJK run
    if C::WS_EXCEPTION == WsException::Cjk && psp == 1 && uni.cjk_lead & 1 != 0 {
        // Whether the previous batch's trailing space started a piece
        // depends on its own predecessor (the CJK split exception).
        bad |= 0b11;
    }

    let mut boundary = (start | wsb_bits) & !after_sp & !tn & !edge_kill;

    // ---- DeepSeek: ASCII controls are not [\p{P}\p{S}] ----
    if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly {
        let ctl = o & m.low_ctl;
        if ctl != 0 {
            // `<< 2`: after a lone control, ASCII punct prefixes the letters
            // ("\x1b[s" -> "\x1b" "[s"), a cut the punct algebra can't see.
            bad |= ctl | ctl << 1 | ctl << 2 | ctl >> 1;
        }
        // The same reach from a control in the previous batch's last two bytes.
        let is_ctl = |b: u8| (b < 0x20 && !matches!(b, b'\t' | b'\n' | b'\r')) || b == 0x7F;
        if scan >= 1 && is_ctl(bytes[scan - 1]) {
            bad |= 0b11;
        }
        if scan >= 2 && is_ctl(bytes[scan - 2]) {
            bad |= 0b1;
        }
    }

    // ---- deferred chars: scalar territory ----
    if defer != 0 {
        bad |= defer | defer << 1 | defer >> 1;
        // A deferred char inside/adjacent to an ASCII ws run poisons the
        // run's global split bookkeeping.
        let wsadj = ((defer << 1) | (defer >> 1)) & ews;
        if wsadj != 0 {
            bad |= fill_both(wsadj, ews);
        }
    }

    // ---- punct-prefix configs: single punct absorbs following letters ----
    if C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly {
        // A letter directly after a piece-starting ASCII punct char is
        // absorbed (byte- and char-adjacency coincide). DeepSeek's
        // AsciiOnly prefix takes ASCII letters only ("l'été" → l ' été).
        let absorb_l = if C::PUNCT_PREFIX_MODE == PunctPrefixMode::AsciiOnly { m.l } else { l };
        let kill = ((o & boundary) << 1) & absorb_l;
        boundary &= !kill;
        if C::PUNCT_PREFIX_MODE == PunctPrefixMode::AsciiOnly {
            // A unicode letter following an absorbed-prefix ASCII run
            // starts a fresh piece, while after a plain run it continues.
            let prefix_runs = fill_right(kill, m.l);
            boundary |= uni.l_lead & (prefix_runs << 1);
            // A leading ASCII letter run continued from the previous
            // batch has unknown prefix-ness; a unicode letter ending it
            // is scalar territory.
            if pl == 1 {
                let k = (!m.l).trailing_zeros();
                if k < 64 && uni.l_lead >> k & 1 != 0 {
                    bad |= (0b111u64 << k) >> 1;
                }
            }
        }
        // Unicode punct chars absorb across their full char length (Any
        // mode only — DeepSeek's AsciiOnly prefix excludes non-ASCII
        // punct, see deepseek_nonascii_punct_no_letter_prefix).
        if C::PUNCT_PREFIX_MODE == PunctPrefixMode::Any {
            let mut leads = uni.o_lead & boundary;
            while leads != 0 {
                let i = leads.trailing_zeros() as usize;
                leads &= leads - 1;
                let end = i + utf8_len(bytes[scan + i]);
                if end < 64 {
                    boundary &= !((1u64 << end) & l);
                } else {
                    // The absorb decision crosses the batch edge.
                    bad |= 1 << 63;
                }
            }
        }
        // A punct char at bit 63 may absorb a letter in the next batch.
        if o >> 63 != 0 {
            bad |= 1 << 63;
        }
    }

    // ---- apostrophe contractions: in-mask fixup ----
    // Only apostrophes that START a piece matter (after a word: "don|'t");
    // run-interior or space-prefixed apostrophes already carry no bit.
    if C::CONTRACTION_MODE != ContractionMode::None && ap != 0 {
        let mut cand = ap & boundary;
        let mut contr_next = 0u64;
        while cand != 0 {
            let i = cand.trailing_zeros() as usize;
            cand &= cand - 1;
            if i >= 61 {
                // Suffix reaches past the batch edge; scalar decides.
                bad |= u64::MAX << i;
                break;
            }
            if C::CONTRACTION_MODE == ContractionMode::Suffix && scan > 0 {
                // A word straddling in from the previous batch carries
                // contraction history the masks can't see ("a'll|'ve" must
                // NOT merge again; "don|'t" must). Defer when the
                // preceding letter run reaches the batch start.
                let run_to_start = ((!l).trailing_zeros() as usize) >= i;
                let pb = bytes[scan - 1];
                if run_to_start && (pl == 1 || pb == b'\'') {
                    bad |= 0b1111u64 << i;
                    continue;
                }
            }
            if C::CONTRACTION_MODE == ContractionMode::Suffix && (bad >> i) & 1 == 1 {
                // The merge decision depends on scalar territory (e.g. a
                // deferred-char word before the apostrophe); extend the
                // zone over the suffix so a chained apostrophe defers too.
                bad |= 0b1111u64 << i;
                continue;
            }
            let fold = |b: u8| {
                if C::CONTRACTION_CASE == ContractionCase::Insensitive { b | 0x20 } else { b }
            };
            let b1 = fold(bytes[scan + i + 1]);
            let k = match b1 {
                b's' | b't' | b'd' | b'm' => 2,
                b'l' if fold(bytes[scan + i + 2]) == b'l' => 3,
                b'v' if fold(bytes[scan + i + 2]) == b'e' => 3,
                b'r' if fold(bytes[scan + i + 2]) == b'e' => 3,
                _ => 0,
            };
            // Suffix mode only merges a contraction into a preceding
            // LETTER run ("don'ts" → don't s) that didn't itself just end
            // in a contraction ("a're's" → a're 's); after anything else
            // the apostrophe is a letter prefix ("5'ts" → 5 'ts).
            let prev_l = (if i == 0 { pl == 1 } else { (l >> (i - 1)) & 1 == 1 })
                && (contr_next >> i) & 1 == 0;
            if k != 0 && (C::CONTRACTION_MODE == ContractionMode::Standalone || prev_l) {
                // Contraction: the suffix letters merge into "'t"/"'ll"…
                // and the char after the suffix starts fresh — even a
                // letter ("don'ts" → don 't s).
                boundary &= !(1u64 << (i + 1));
                if C::CONTRACTION_MODE == ContractionMode::Suffix {
                    // O200K: the contraction continues the word instead.
                    boundary &= !(1u64 << i);
                }
                boundary |= 1u64 << (i + k);
                contr_next |= 1u64 << (i + k);
            } else {
                // No contraction: an immediately following letter is
                // absorbed by tokie's apostrophe-prefix ("x'y" → x 'y),
                // which SpaceOnly configs don't get from the general
                // punct-absorb rule.
                let nb1 = bytes[scan + i + 1];
                if crate::util::is_ascii_letter(nb1)
                    || (nb1 >= 0x80 && (uni.l_lead >> (i + 1)) & 1 == 1)
                {
                    boundary &= !(1u64 << (i + 1));
                }
            }
        }
    }


    // ---- Chunked3: digit runs touching a bad byte have unknown phase ----
    if C::DIGIT_MODE == DigitMode::Chunked3 {
        let seedd = bad & d;
        if seedd != 0 {
            bad |= fill_both(seedd, d);
        }
    }

    (boundary & !bad, bad)
}

// =======================================================================
// Bench-only internal hooks (not public API)
// =======================================================================

/// Stage-level access to the mask pipeline for the profiling examples.
/// Hidden from docs; semver-exempt.
#[doc(hidden)]
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub mod bench_internal {
    use crate::core::config::*;

    /// Stage (a): NEON classification + movemasks only, folded to a u64
    /// so the calls cannot be dead-code-eliminated.
    #[inline(always)]
    pub fn classify_fold<C: PretokConfig>(bytes: &[u8], scan: usize) -> u64 {
        let camel = C::LETTER_MODE == LetterMode::CamelCase;
        let want_slash = C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes;
        let want_ctl = C::PUNCT_CLASS == PunctClass::PunctSymbolOnly;
        let m = <super::Base as super::Classify>::ascii_masks(bytes, scan, camel, want_slash, want_ctl);
        m.l ^ m.d ^ m.s ^ m.t ^ m.n ^ m.hi ^ m.ap ^ m.lo ^ m.slash ^ m.low_ctl
    }

    /// Stages (a)+(b): the full per-batch boundary computation.
    #[inline(always)]
    pub fn batch_masks<C: PretokConfig>(bytes: &[u8], scan: usize) -> (u64, u64) {
        super::batch_masks::<C, super::Base>(bytes, scan)
    }

    /// The scalar ground-truth advance (bad-zone executor).
    #[inline(always)]
    pub fn scalar_advance<C: PretokConfig>(text: &str, pos: usize) -> usize {
        super::scalar_advance::<C>(text, pos)
    }
}

// =======================================================================
// The batch walker
// =======================================================================

/// Scheme-agnostic mask-scanner state: pops trusted boundary bits, walks
/// bad zones through `Core`, runs the buffer tail scalar, and precomputes
/// one batch ahead so the SIMD chain retires under the previous batch's
/// pops. Ported from gigatoken's `MaskState` (see its mask.rs docs).
struct MaskState {
    /// Start of the pending (not yet emitted) piece.
    pos: usize,
    /// Base of the next batch to scan.
    scan: usize,
    /// Base the `rem`/`batch_*` bits refer to.
    mask_base: usize,
    /// Boundary bits of the current segment (trusted, pop-ready).
    rem: u64,
    /// Full usable mask of the current batch (later segments).
    batch_usable: u64,
    /// Bad zones of the current batch not yet passed.
    batch_bad: u64,
    /// Emit pieces via `Core` while `pos < scalar_until`.
    scalar_until: usize,
    /// Eagerly computed masks for the batch at `pre_base` (usize::MAX =
    /// none).
    pre_base: usize,
    pre_usable: u64,
    pre_bad: u64,
    /// Consecutive mostly-bad batches (see [`Self::adapt`]).
    bad_streak: u32,
}

impl MaskState {
    #[inline]
    fn new(pos: usize, simd: bool) -> Self {
        let scalar_until = if simd { pos } else { usize::MAX };
        Self {
            pos,
            scan: pos,
            mask_base: pos,
            rem: 0,
            batch_usable: 0,
            batch_bad: 0,
            scalar_until,
            pre_base: usize::MAX,
            pre_usable: 0,
            pre_bad: 0,
            bad_streak: 0,
        }
    }

    /// Mostly-bad batches (e.g. CJK under o200k's letter classes) pay a
    /// full classification yet yield few trusted boundaries. From the
    /// second in a row, stay on the `Core` path for an exponentially
    /// growing window before classifying again. This only moves the
    /// mask/scalar switch point — `Core` from any piece start is exact —
    /// so output is unchanged. Call after `load_segment` on a refill.
    #[inline(always)]
    fn adapt(&mut self, bad: u64) {
        if bad.count_ones() >= 48 {
            self.bad_streak = (self.bad_streak + 1).min(6);
            if self.bad_streak >= 2 {
                let until = self.mask_base + (64usize << self.bad_streak);
                self.scalar_until = self.scalar_until.max(until);
            }
        } else {
            self.bad_streak = 0;
        }
    }

    /// Load the segment of `batch_usable` bits in [from_bit, next bad run)
    /// into `rem` and aim `scalar_until` past that bad run at the next
    /// trusted boundary (or the batch end).
    #[inline(always)]
    fn load_segment(&mut self, from_bit: u32) {
        let live = u64::MAX << from_bit;
        let seg_bad = self.batch_bad & live;
        if seg_bad == 0 {
            self.rem = self.batch_usable & live;
            self.batch_bad = 0;
        } else {
            let nb = seg_bad.trailing_zeros();
            self.rem = self.batch_usable & live & ((1u64 << nb) - 1);
            let rest = self.batch_usable & (u64::MAX << nb);
            self.scalar_until = if rest != 0 {
                self.mask_base + rest.trailing_zeros() as usize
            } else {
                self.mask_base + 64
            };
        }
        // A bit at the pending piece's own start is not an end.
        let at_start = self.pos == self.mask_base + from_bit as usize;
        self.rem &= !(u64::from(at_start) << from_bit);
    }

    /// The next piece's byte range, or None at end of input.
    #[inline(always)]
    fn next_span<C: PretokConfig, K: Classify>(&mut self, text: &str) -> Option<(usize, usize)> {
        let bytes = text.as_bytes();
        let len = bytes.len();
        loop {
            if self.rem != 0 {
                let tz = self.rem.trailing_zeros() as usize;
                let end = self.mask_base + tz;
                self.rem &= self.rem - 1;
                let start = self.pos;
                self.pos = end;
                return Some((start, end));
            }
            if self.pos < self.scalar_until {
                if self.pos >= len {
                    return None;
                }
                let start = self.pos;
                let end = scalar_advance::<C>(text, start);
                debug_assert!(end > start, "no scalar progress at {start}");
                self.pos = end;
                return Some((start, end));
            }
            // Continue with the current batch's next trusted segment
            // after a scalar gap (each batch is computed exactly once).
            if self.batch_bad != 0 && self.pos < self.mask_base + 64 {
                self.load_segment((self.pos - self.mask_base) as u32);
                continue;
            }
            self.batch_bad = 0;
            // Resume after a scalar overrun WITHOUT leaving the 64-byte
            // grid, so the precomputed next batch stays valid. Grid bits
            // below `pos` may be stale run-internal bits; they are masked
            // by the `from_bit` passed to load_segment below.
            while self.scan + 64 <= self.pos {
                self.scan += 64;
            }
            if self.scan + 64 > len {
                // Tail: scalar to the end of the buffer.
                self.scalar_until = usize::MAX;
                continue;
            }
            let (usable, bad) = if self.pre_base == self.scan {
                (self.pre_usable, self.pre_bad)
            } else {
                K::batch::<C>(bytes, self.scan)
            };
            self.mask_base = self.scan;
            self.scan += 64;
            self.batch_usable = usable;
            self.batch_bad = bad;
            // Kick off the next batch now; its SIMD chain overlaps this
            // batch's pops instead of stalling the next refill.
            if self.scan + 64 <= len {
                let (u2, b2) = K::batch::<C>(bytes, self.scan);
                self.pre_base = self.scan;
                self.pre_usable = u2;
                self.pre_bad = b2;
            } else {
                self.pre_base = usize::MAX;
            }
            // An overrun may have left `pos` inside this grid batch;
            // start from its bit so stale bits below never pop.
            if self.pos > self.mask_base {
                self.load_segment((self.pos - self.mask_base) as u32);
            } else {
                self.load_segment(0);
            }
            self.adapt(bad);
        }
    }

    /// Bulk-drain variant of [`Self::next_span`]: emits every piece to `f`
    /// as `(start, end)` byte offsets, in the same order and with the same
    /// boundaries `next_span` would yield (identical masks, `load_segment`
    /// segments, bad-zone re-derivation and tail). The only difference is
    /// that a freshly loaded run of trusted boundary bits and a scalar gap
    /// are each drained in a tight local loop instead of one piece per
    /// call — the per-piece `Iterator::next` re-entry, the pop-vs-refill
    /// branch retest and the `Option`/`&str` round-trip collapse, which on
    /// OWT/gpt2 is ~half of the single-thread pretokenize cost (the mask
    /// compute is the other half, and its one-batch-ahead precompute still
    /// overlaps these drains). This is the bulk-encode hot loop.
    #[inline(always)]
    fn for_each_span<C: PretokConfig, K: Classify, F: FnMut(usize, usize)>(&mut self, text: &str, mut f: F) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        loop {
            // Drain the whole trusted segment: `rem` holds piece-start bits
            // with no bad byte before the next segment, so every pop is a
            // real boundary. `pos` chains through the run.
            while self.rem != 0 {
                let tz = self.rem.trailing_zeros() as usize;
                let end = self.mask_base + tz;
                self.rem &= self.rem - 1;
                let start = self.pos;
                self.pos = end;
                f(start, end);
            }
            // Drain the whole scalar gap (bad zone / tail) piece by piece.
            while self.pos < self.scalar_until {
                if self.pos >= len {
                    return;
                }
                let start = self.pos;
                let end = scalar_advance::<C>(text, start);
                debug_assert!(end > start, "no scalar progress at {start}");
                self.pos = end;
                f(start, end);
            }
            // ---- refill: identical to next_span's tail ----
            if self.batch_bad != 0 && self.pos < self.mask_base + 64 {
                self.load_segment((self.pos - self.mask_base) as u32);
                continue;
            }
            self.batch_bad = 0;
            while self.scan + 64 <= self.pos {
                self.scan += 64;
            }
            if self.scan + 64 > len {
                self.scalar_until = usize::MAX;
                continue;
            }
            let (usable, bad) = if self.pre_base == self.scan {
                (self.pre_usable, self.pre_bad)
            } else {
                K::batch::<C>(bytes, self.scan)
            };
            self.mask_base = self.scan;
            self.scan += 64;
            self.batch_usable = usable;
            self.batch_bad = bad;
            if self.scan + 64 <= len {
                let (u2, b2) = K::batch::<C>(bytes, self.scan);
                self.pre_base = self.scan;
                self.pre_usable = u2;
                self.pre_bad = b2;
            } else {
                self.pre_base = usize::MAX;
            }
            if self.pos > self.mask_base {
                self.load_segment((self.pos - self.mask_base) as u32);
            } else {
                self.load_segment(0);
            }
            self.adapt(bad);
        }
    }
}

// =======================================================================
// Public iterator
// =======================================================================

/// Mask-scanner pretokenizer over `text`, byte-identical to `Core<C>`.
pub struct Mask<'a, C: PretokConfig> {
    text: &'a str,
    state: MaskState,
    isa: MaskIsa,
    _cfg: PhantomData<C>,
}

impl<'a, C: PretokConfig> Mask<'a, C> {
    #[inline]
    pub fn new(text: &'a str) -> Self {
        Self::with_isa(text, MaskIsa::detect())
    }

    /// Construct with an explicit classification backend (benchmarks and
    /// differential tests). An unsupported `isa` falls back to the
    /// detected best.
    #[doc(hidden)]
    #[inline]
    pub fn with_isa(text: &'a str, isa: MaskIsa) -> Self {
        let isa = if isa.supported() { isa } else { MaskIsa::detect() };
        Self {
            text,
            state: MaskState::new(0, isa != MaskIsa::Scalar),
            isa,
            _cfg: PhantomData,
        }
    }

    /// Visit every piece of `text` via an inline callback, byte-identical
    /// to iterating this `Mask` to exhaustion. Drains trusted boundary
    /// runs in a tight loop instead of one piece per `next()` — the
    /// bulk-encode hot path (see [`MaskState::for_each_span`]). Consumes
    /// the remaining state, so call on a freshly constructed `Mask`.
    #[inline]
    pub fn for_each_piece<F: FnMut(&'a str)>(&mut self, mut f: F) {
        let text = self.text;
        let bytes = text.as_bytes();
        let emit = |start: usize, end: usize| {
            debug_assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
            // SAFETY: spans come from ASCII-classified boundary bits or
            // Core's own piece ends; both are char boundaries.
            f(unsafe { std::str::from_utf8_unchecked(&bytes[start..end]) });
        };
        match self.isa {
            // SAFETY: `isa` is only `Avx2` when AVX2 was detected.
            #[cfg(target_arch = "x86_64")]
            MaskIsa::Avx2 => unsafe { avx2_for_each_span::<C, _>(&mut self.state, text, emit) },
            _ => self.state.for_each_span::<C, Base, _>(text, emit),
        }
    }
}

/// Runs the bulk walker inside an AVX2 codegen context so the per-batch
/// classification inlines into it.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_for_each_span<C: PretokConfig, F: FnMut(usize, usize)>(
    state: &mut MaskState,
    text: &str,
    f: F,
) {
    state.for_each_span::<C, Avx2, F>(text, f)
}

impl<'a, C: PretokConfig> Iterator for Mask<'a, C> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        let (start, end) = match self.isa {
            MaskIsa::Avx2 => self.state.next_span::<C, Avx2>(self.text)?,
            _ => self.state.next_span::<C, Base>(self.text)?,
        };
        debug_assert!(self.text.is_char_boundary(start) && self.text.is_char_boundary(end));
        // SAFETY: spans come from ASCII-classified boundary bits or Core's
        // own piece ends; both are char boundaries.
        Some(unsafe {
            std::str::from_utf8_unchecked(&self.text.as_bytes()[start..end])
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configs::*;

    /// Every backend runnable on this CPU (the scalar walker included).
    fn isas() -> impl Iterator<Item = MaskIsa> {
        [MaskIsa::Scalar, MaskIsa::Base, MaskIsa::Avx2].into_iter().filter(|i| i.supported())
    }

    fn check<C: PretokConfig>(text: &str) {
        let scalar: Vec<&str> = Core::<C>::new(text).collect();
        for isa in isas() {
            let masked: Vec<&str> = Mask::<C>::with_isa(text, isa).collect();
            assert_eq!(masked, scalar, "{isa:?}: mask vs core mismatch on {text:?}");
            // The bulk-drain callback path must match the iterator exactly.
            let mut bulk: Vec<&str> = Vec::new();
            Mask::<C>::with_isa(text, isa).for_each_piece(|p| bulk.push(p));
            assert_eq!(bulk, scalar, "{isa:?}: for_each_piece vs core mismatch on {text:?}");
        }
    }

    fn check_all(text: &str) {
        check::<Gpt2Config>(text);
        check::<Cl100kConfig>(text);
        check::<O200kConfig>(text);
        check::<VoyageConfig>(text);
        check::<SmolLMConfig>(text);
        check::<DeepSeekConfig>(text);
        check::<QwenConfig>(text);
        check::<TekkenConfig>(text);
    }

    /// Wrap short unit vectors in long ASCII padding so the mask path is
    /// actually exercised (short inputs are all tail-scalar).
    fn check_padded(case: &str) {
        let pad = "The quick brown fox jumps over the lazy dog 1234, and again. ".repeat(3);
        check_all(case);
        check_all(&format!("{pad}{case}"));
        check_all(&format!("{case}{pad}"));
        check_all(&format!("{pad}{case}{pad}"));
        // Shift the case across batch-edge alignments.
        for shift in 60..70 {
            let prefix: String = std::iter::repeat('x').take(shift).collect();
            check_all(&format!("{prefix} {case} {pad}"));
        }
    }

    #[test]
    fn unit_vectors() {
        let cases = [
            "Hello world", "Hello, world!", "test 123", "don't", "I'll", "I've",
            "we're", "DON'T", "O'Toole", "don'ts", "'Toole", "c'mon isn't they'd",
            "x 42", "a\n\nb", "a\n  b", "a  b", "a <b", "a   b", "12345",
            "$hello", "$$hello", " $hello", "a$hello", "a b", "a\nb", "a \nb",
            "CamelCase JSONParser parseJSON XMLHttpRequest", "'hello'",
            "test\t123", "\tword", "a\t\tb", "a \tb", "x\ty",
            "smart”\n\nM", "you…\nA", "”\n \nx", "”…\n\nx", "x«\n\ny", "«abc", "«»",
            "1¹23", "¹²³", "12¹", " ¹", "a ❶b", "x ½", "a\n\n½ cup", "x  ½",
            "values \u{200B}\u{200B}that", "higher \u{AD}partic", "\u{200B}école",
            "a\u{80}\u{94}b", "x «y", "«ab", "a \u{200B}b", "x\u{AD}y",
            "l\x1b[s\x1b[6n", "\x01!x", "\x01'll", "\x1bab", "\x7f\x00é", "x ❤\u{FE0F}\n", "!!\u{301}x",
            "ก\u{E31}น", "આફ્રિકા ખંડ", "l'été", ".\n\n'The",
            "abc\n123", "a 1 b", "12345 67890 a123b",
            "hello/world//x", "a!//b", "x”\n/y",
            "line one\nline two\r\nline three\n\n  indented",
            "\x01\x01!", "a\x01b", "»\x01!x",
            "money $100.99, 50% off!", "e.g. Dr. Smith's co-op",
            "日本語のテキスト and English", "русский текст тоже",
            "a\u{3000}b", "nbsp\u{A0}here", "tab\tnew\nline",
            "8\t_re_num", "\t.re", "\t8", "\tre", "a \t_x", "\r\n  \r\n  x", "\n \n",
            "aⅢ", " Ⅲ", "x ⅰ", "（BD-ROM・-R・-RE）", "・(PД", "Ｆ一", " 中文", "a  中",
            "ア゛b", "゛゛", "-中", "中-R", "=σλ", "-handâa",
        ];
        for c in cases {
            check_padded(c);
        }
    }

    #[test]
    fn empty_and_tiny() {
        for c in ["", "a", " ", "\n", "'", "é", "1"] {
            check_all(c);
        }
    }

    #[test]
    fn fuzz_differential() {
        // Weighted alphabet spanning every algebra dimension.
        let atoms: &[&str] = &[
            "a", "b", "z", "A", "Z", "Q", "e", "t", "o", "i", "n", "s",
            "0", "1", "9", "5",
            " ", " ", " ", "\t", "\n", "\r", "\n\n",
            ".", ",", "!", "$", "<", "/", "«", "»", "”", "…",
            "'", "'s", "'ll", "'re", "'T",
            "é", "ß", "ก", "\u{E31}", "中", "の", "・", "゛", "Ⅲ", "½", "¹", "❶",
            "\u{200B}", "\u{AD}", "\u{A0}", "\u{3000}", "\x01", "\x0b",
            "🎉", "👍",
        ];
        let mut state = 0x243F6A8885A308D3u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let iters = std::env::var("PRETOKIE_FUZZ_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(4000);
        for _ in 0..iters {
            let len = 40 + (next() % 300) as usize;
            let mut s = String::new();
            for _ in 0..len {
                s.push_str(atoms[(next() % atoms.len() as u64) as usize]);
            }
            check_all(&s);
        }
    }

    /// Streaming differential of `Mask` (iterator and bulk drain, every
    /// backend) against `Core` over a large text, without materializing
    /// the piece lists.
    fn diff_stream<C: PretokConfig>(label: &str, text: &str) {
        let name = std::any::type_name::<C>().rsplit("::").next().unwrap();
        for isa in isas() {
            let mut core = Core::<C>::new(text);
            let mut mask = Mask::<C>::with_isa(text, isa);
            let mut i = 0usize;
            loop {
                match (core.next(), mask.next()) {
                    (Some(a), Some(b)) if a == b => {}
                    (None, None) => break,
                    (a, b) => panic!("{label} {name} {isa:?} piece {i}: core {a:?} mask {b:?}"),
                }
                i += 1;
            }
            let mut core = Core::<C>::new(text);
            let mut j = 0usize;
            Mask::<C>::with_isa(text, isa).for_each_piece(|b| {
                let a = core.next();
                assert_eq!(a, Some(b), "{label} {name} {isa:?} for_each piece {j}");
                j += 1;
            });
            assert_eq!(core.next(), None, "{label} {name} {isa:?} for_each short");
        }
    }

    fn diff_stream_all(label: &str, text: &str) {
        diff_stream::<Gpt2Config>(label, text);
        diff_stream::<Cl100kConfig>(label, text);
        diff_stream::<O200kConfig>(label, text);
        diff_stream::<VoyageConfig>(label, text);
        diff_stream::<SmolLMConfig>(label, text);
        diff_stream::<DeepSeekConfig>(label, text);
        diff_stream::<QwenConfig>(label, text);
        diff_stream::<TekkenConfig>(label, text);
    }

    fn bench_data(name: &str) -> Option<Vec<u8>> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../benches/data");
        std::fs::read(dir.join(name)).ok()
    }

    /// 5 MB of OpenWebText (falls back to enwik8) under every backend.
    #[test]
    fn owt_sample_differential() {
        let Some(data) = bench_data("owt_sample.txt").or_else(|| bench_data("enwik8")) else {
            eprintln!("owt_sample.txt / enwik8 missing; skipping");
            return;
        };
        let text = String::from_utf8_lossy(&data[..data.len().min(5_000_000)]).into_owned();
        diff_stream_all("sample", &text);
    }

    /// Full 100 MB enwik8 plus every file under `PRETOKIE_FIXTURES`
    /// (e.g. tokbench's data/fixtures). Slow; run with
    /// `cargo test --release -p pretokie -- --ignored corpus_differential`.
    #[test]
    #[ignore]
    fn corpus_differential() {
        let mut ran = 0;
        if let Some(data) = bench_data("enwik8") {
            diff_stream_all("enwik8", &String::from_utf8_lossy(&data));
            ran += 1;
        }
        if let Ok(dir) = std::env::var("PRETOKIE_FIXTURES") {
            let mut stack = vec![std::path::PathBuf::from(dir)];
            while let Some(p) = stack.pop() {
                if p.is_dir() {
                    for e in std::fs::read_dir(&p).unwrap() {
                        stack.push(e.unwrap().path());
                    }
                } else if let Ok(data) = std::fs::read(&p) {
                    diff_stream_all(&p.display().to_string(), &String::from_utf8_lossy(&data));
                    ran += 1;
                }
            }
        }
        assert!(ran > 0, "no corpus found (benches/data/enwik8 or PRETOKIE_FIXTURES)");
        eprintln!("corpus_differential: {ran} corpora");
    }
}
