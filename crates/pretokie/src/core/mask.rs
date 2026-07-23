//! Mask-scanner pretokenizers: 64-byte batches are classified with NEON
//! into per-byte class bitmasks, piece-start bits are derived with
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
//! On non-aarch64 targets every piece takes the `Core` path (the walker
//! starts with `scalar_until = usize::MAX`).

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

#[inline(always)]
fn simd_available() -> bool {
    cfg!(target_arch = "aarch64")
}

// =======================================================================
// NEON classification (aarch64)
// =======================================================================

/// One u64 mask per byte predicate for a 64-byte batch (bit i = byte
/// scan+i). These are `Core`'s byte classes, not Unicode's: whitespace
/// is exactly {space, tab, \r, \n} and everything else ASCII that is
/// not a letter/digit is "punct" (including \x0b, \x0c and controls),
/// matching `Core`'s scalar predicates.
#[cfg(target_arch = "aarch64")]
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
fn ascii_masks(
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
// Bit-domain helpers (platform-independent)
// =======================================================================

/// Kogge-Stone rightward fill: propagate `seed` bits toward higher bit
/// positions through contiguous runs of `mask`.
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fill_both(seed: u64, mask: u64) -> u64 {
    fill_right(seed, mask) | fill_left(seed, mask)
}

/// Piece-start bits inside ASCII digit runs for `\p{N}{1,3}`: each run
/// splits into 3-char pieces, so starts sit at run start + 3k.
#[cfg(target_arch = "aarch64")]
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
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn leading_run(d: u64) -> u64 {
    let tz = (!d).trailing_zeros();
    if tz >= 64 { u64::MAX } else { (1u64 << tz) - 1 }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn is_ascii_ws_byte(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn is_cjk_char(ch: char) -> bool {
    let cp = ch as u32;
    matches!(cp, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

// =======================================================================
// Per-batch boundary algebra, parameterized by PretokConfig
// =======================================================================

/// Piece-start (`usable`) and scalar-territory (`bad`) bitmasks for
/// `bytes[scan..scan+64]`. Bit k of `usable` = a trustworthy piece start
/// at `scan + k`; `bad` marks bytes whose boundaries `Core` re-derives,
/// and no piece is emitted across an unresolved bad zone.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn batch_masks<C: PretokConfig>(bytes: &[u8], scan: usize) -> (u64, u64) {
    // One byte of lookahead is required for the bit-63 whitespace-split
    // test (reading a whole char there is safe: input is valid UTF-8, so
    // a char starting in-bounds is fully in-bounds).
    if scan + 65 > bytes.len() {
        return (0, u64::MAX);
    }

    let camel = C::LETTER_MODE == LetterMode::CamelCase;
    let want_slash = C::PUNCT_TRAILING == PunctTrailing::NewlinesAndSlashes;
    let want_ctl = C::PUNCT_CLASS == PunctClass::PunctSymbolOnly;
    let m = ascii_masks(bytes, scan, camel, want_slash, want_ctl);

    let l = m.l;
    let d = m.d;
    let sp = m.s | m.t; // `Core` lets both space and tab prefix content
    let n = m.n;
    let ws = sp | n;
    let hi = m.hi;
    let o = !(l | d | ws | hi); // ASCII punct + controls (Core's punct class)
    let ap = m.ap;

    let mut bad = 0u64;

    // ---- carries from the byte before the batch ----
    let (pl, pd, psp, pws, po, plo, pn);
    if scan == 0 {
        (pl, pd, psp, pws, po, plo, pn) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    } else {
        let b = bytes[scan - 1];
        if b >= 0x80 {
            // Unclassified previous char: boundaries near the batch start
            // are uncertain (up to the punct-prefix +2 pattern).
            bad |= 0b111;
            (pl, pd, psp, pws, po, plo, pn) = (0, 0, 0, 0, 0, 0, 0);
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
        DigitMode::Single => start |= d,
        DigitMode::Chunked3 => {
            start |= digit_run_splits3(d);
            // A run carried in from the previous batch has unknown phase.
            if pd == 1 && d & 1 != 0 {
                bad |= leading_run(d);
            }
        }
    }

    // punct
    start |= o & !((o << 1) | po);

    // ---- punct-run trailing newlines (and slashes for O200K) ----
    // '/' (O200K) is punct-class, so it extends the main punct run on
    // its own; it participates in the trailing tail only after a
    // consumed newline. The seed is therefore newline-after-punct.
    let trail = if want_slash { n | m.slash } else { n };
    let tn = if C::PUNCT_TRAILING == PunctTrailing::None {
        0
    } else {
        fill_right(n & ((o << 1) | po), trail)
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
    // A newline directly after a non-ASCII char may or may not start a
    // trailing tail (the char's punct-ness is unclassified). For
    // newline-only tails both interpretations emit the same bits, but a
    // slash in the chain changes whether the next punct char starts
    // fresh — and with it the single-punct letter absorb one byte later.
    if want_slash {
        let chain = fill_right(n & (hi << 1), trail);
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
        // SmolLM: stay whole before a digit; DeepSeek: before digit/CJK
        // (non-ASCII numerics/CJK are inside the hi bad zone anyway).
        split &= !(d >> 1);
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
                    WsException::Cjk => !is_cjk_char(ch),
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
    let after_sp_classes =
        l | o | if C::SPACE_PREFIXES_DIGITS { d } else { 0 };
    let after_sp = ((sp << 1) | psp) & after_sp_classes;

    let mut boundary = (start | wsb_bits) & !after_sp & !tn;

    // ---- apostrophe contractions: scalar territory ----
    if C::CONTRACTION_MODE != ContractionMode::None && ap != 0 {
        bad |= ap | ap << 1 | ap << 2 | ap << 3;
        let edge = ap & (0b111 << 61);
        if edge != 0 {
            bad |= u64::MAX << edge.trailing_zeros();
        }
    }

    // ---- DeepSeek: ASCII controls are not [\p{P}\p{S}] ----
    if C::PUNCT_CLASS == PunctClass::PunctSymbolOnly {
        let ctl = o & m.low_ctl;
        if ctl != 0 {
            bad |= ctl | ctl << 1 | ctl >> 1;
        }
    }

    // ---- non-ASCII: scalar territory (v1: no in-mask classification) ----
    if hi != 0 {
        bad |= hi | hi << 1 | hi >> 1;
    }

    // ---- punct-prefix configs: single punct absorbs following letters ----
    if C::PUNCT_PREFIX_MODE != PunctPrefixMode::SpaceOnly {
        // A letter directly after a piece-starting punct char is absorbed.
        boundary &= !(((o & boundary) << 1) & l);
        // (hi, punct, letter): whether the punct absorbs depends on the
        // unclassified char's class.
        bad |= (hi << 2) & (o << 1) & l;
        // A punct char at bit 63 may absorb a letter in the next batch.
        if o >> 63 != 0 {
            bad |= 1 << 63;
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

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
fn batch_masks<C: PretokConfig>(_bytes: &[u8], _scan: usize) -> (u64, u64) {
    (0, u64::MAX)
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
}

impl MaskState {
    #[inline]
    fn new(pos: usize) -> Self {
        let scalar_until = if simd_available() { pos } else { usize::MAX };
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
    fn next_span<C: PretokConfig>(&mut self, text: &str) -> Option<(usize, usize)> {
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
                batch_masks::<C>(bytes, self.scan)
            };
            self.mask_base = self.scan;
            self.scan += 64;
            self.batch_usable = usable;
            self.batch_bad = bad;
            // Kick off the next batch now; its SIMD chain overlaps this
            // batch's pops instead of stalling the next refill.
            if self.scan + 64 <= len {
                let (u2, b2) = batch_masks::<C>(bytes, self.scan);
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
    _cfg: PhantomData<C>,
}

impl<'a, C: PretokConfig> Mask<'a, C> {
    #[inline]
    pub fn new(text: &'a str) -> Self {
        Self { text, state: MaskState::new(0), _cfg: PhantomData }
    }
}

impl<'a, C: PretokConfig> Iterator for Mask<'a, C> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        let (start, end) = self.state.next_span::<C>(self.text)?;
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

    fn check<C: PretokConfig>(text: &str) {
        let scalar: Vec<&str> = Core::<C>::new(text).collect();
        let masked: Vec<&str> = Mask::<C>::new(text).collect();
        assert_eq!(masked, scalar, "mask vs core mismatch on {text:?}");
    }

    fn check_all(text: &str) {
        check::<Gpt2Config>(text);
        check::<Cl100kConfig>(text);
        check::<O200kConfig>(text);
        check::<VoyageConfig>(text);
        check::<SmolLMConfig>(text);
        check::<DeepSeekConfig>(text);
        check::<QwenConfig>(text);
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
            "ก\u{E31}น", "આફ્રિકા ખંડ", "l'été", ".\n\n'The",
            "abc\n123", "a 1 b", "12345 67890 a123b",
            "hello/world//x", "a!//b", "x”\n/y",
            "line one\nline two\r\nline three\n\n  indented",
            "\x01\x01!", "a\x01b", "»\x01!x",
            "money $100.99, 50% off!", "e.g. Dr. Smith's co-op",
            "日本語のテキスト and English", "русский текст тоже",
            "a\u{3000}b", "nbsp\u{A0}here", "tab\tnew\nline",
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
            "é", "ß", "ก", "\u{E31}", "中", "の", "½", "¹", "❶",
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

    #[test]
    fn owt_sample_differential() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("benches/data/owt_sample.txt");
        let Ok(data) = std::fs::read(&path) else {
            eprintln!("owt_sample.txt missing; skipping");
            return;
        };
        let text = String::from_utf8_lossy(&data[..data.len().min(5_000_000)]).into_owned();
        macro_rules! diff {
            ($cfg:ty) => {{
                let mut core = Core::<$cfg>::new(&text);
                let mut mask = Mask::<$cfg>::new(&text);
                let mut i = 0usize;
                loop {
                    match (core.next(), mask.next()) {
                        (Some(a), Some(b)) => assert_eq!(
                            a, b,
                            "{} piece {i}: core {:?} mask {:?}",
                            stringify!($cfg), a, b
                        ),
                        (None, None) => break,
                        (a, b) => panic!(
                            "{} piece {i}: core {:?} mask {:?}",
                            stringify!($cfg), a, b
                        ),
                    }
                    i += 1;
                }
            }};
        }
        diff!(Gpt2Config);
        diff!(Cl100kConfig);
        diff!(O200kConfig);
        diff!(VoyageConfig);
        diff!(SmolLMConfig);
        diff!(DeepSeekConfig);
        diff!(QwenConfig);
    }
}
