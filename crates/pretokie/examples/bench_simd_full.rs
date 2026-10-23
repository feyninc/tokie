//! Full SIMD pretokenizer prototype for GPT-2.
//!
//! Architecture:
//!   Phase 1 (SIMD): classify 16 bytes, find transitions, write boundary offsets
//!   Phase 2 (scalar fixup): handle contractions + multi-space edges
//!   Phase 3: yield pieces between boundaries
//!
//! Usage: cargo run -p pretokie --example bench_simd_full --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::is_ascii_letter;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Byte classes
const CL: u8 = 0; // letter
const CD: u8 = 1; // digit
const CS: u8 = 2; // space
const CN: u8 = 3; // newline
const CA: u8 = 4; // apostrophe
const CH: u8 = 5; // high byte
const CO: u8 = 6; // other punct

static CLASS_LUT: [u8; 256] = {
    let mut t = [CO; 256];
    let mut i = b'a';
    while i <= b'z' { t[i as usize] = CL; i += 1; }
    i = b'A';
    while i <= b'Z' { t[i as usize] = CL; i += 1; }
    i = b'0';
    while i <= b'9' { t[i as usize] = CD; i += 1; }
    t[b' ' as usize] = CS;
    t[b'\n' as usize] = CN;
    t[b'\r' as usize] = CN;
    t[b'\'' as usize] = CA;
    i = 0x80;
    loop { t[i as usize] = CH; if i == 0xFF { break; } i += 1; }
    t
};

// ---------------------------------------------------------------------------
// SIMD helpers
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn classify_16(ptr: *const u8) -> uint8x16_t {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let letter_offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(letter_offset, vdupq_n_u8(26));
    let digit_offset = vsubq_u8(chunk, vdupq_n_u8(b'0'));
    let is_digit = vcltq_u8(digit_offset, vdupq_n_u8(10));
    let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
    let is_nl = vorrq_u8(
        vceqq_u8(chunk, vdupq_n_u8(b'\n')),
        vceqq_u8(chunk, vdupq_n_u8(b'\r')),
    );
    let is_apos = vceqq_u8(chunk, vdupq_n_u8(b'\''));
    let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));

    let mut r = vdupq_n_u8(CO);
    r = vbslq_u8(is_high, vdupq_n_u8(CH), r);
    r = vbslq_u8(is_apos, vdupq_n_u8(CA), r);
    r = vbslq_u8(is_nl, vdupq_n_u8(CN), r);
    r = vbslq_u8(is_space, vdupq_n_u8(CS), r);
    r = vbslq_u8(is_digit, vdupq_n_u8(CD), r);
    r = vbslq_u8(is_letter, vdupq_n_u8(CL), r);
    r
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_movemask(v: uint8x16_t) -> u16 {
    static POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powers = vld1q_u8(POWERS.as_ptr());
    let bits = vandq_u8(v, powers);
    let lo = vaddv_u8(vget_low_u8(bits)) as u16;
    let hi = vaddv_u8(vget_high_u8(bits)) as u16;
    lo | (hi << 8)
}

// ---------------------------------------------------------------------------
// Phase 1: SIMD boundary detection → Vec<u32> of boundary offsets
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn find_boundaries_simd(bytes: &[u8]) -> Vec<u32> {
    let len = bytes.len();
    if len == 0 {
        return vec![];
    }

    // Worst case: every byte is a boundary
    let mut boundaries = Vec::with_capacity(len / 3);
    boundaries.push(0u32); // first byte always starts a piece

    let mut pos = 1usize;
    let mut prev_last_class = CLASS_LUT[bytes[0] as usize];

    unsafe {
        while pos + 16 <= len {
            let curr_classes = classify_16(bytes.as_ptr().add(pos));

            // Build prev_classes: [prev_last_class, curr[0], curr[1], ..., curr[14]]
            let prev_last_vec = vdupq_n_u8(prev_last_class);
            let prev_classes = vextq_u8(prev_last_vec, curr_classes, 15);

            // Transitions: positions where class changes
            let transitions = vmvnq_u8(vceqq_u8(curr_classes, prev_classes));

            // Suppress merge rules:
            // (CS→CL), (CS→CD), (CS→CN), (CN→CS), (CN→CN)
            let prev_is_space = vceqq_u8(prev_classes, vdupq_n_u8(CS));
            let prev_is_newline = vceqq_u8(prev_classes, vdupq_n_u8(CN));
            let curr_is_letter = vceqq_u8(curr_classes, vdupq_n_u8(CL));
            let curr_is_digit = vceqq_u8(curr_classes, vdupq_n_u8(CD));
            let curr_is_newline = vceqq_u8(curr_classes, vdupq_n_u8(CN));
            let curr_is_space = vceqq_u8(curr_classes, vdupq_n_u8(CS));

            let suppress_space = vandq_u8(
                prev_is_space,
                vorrq_u8(curr_is_letter, vorrq_u8(curr_is_digit, curr_is_newline)),
            );
            let suppress_newline = vandq_u8(
                prev_is_newline,
                vorrq_u8(curr_is_space, curr_is_newline),
            );
            let suppress = vorrq_u8(suppress_space, suppress_newline);

            let real = vandq_u8(transitions, vmvnq_u8(suppress));

            // Extract boundary positions from bitmask
            let mut mask = neon_movemask(real);
            while mask != 0 {
                let bit = mask.trailing_zeros() as u32;
                boundaries.push((pos as u32) + bit);
                mask &= mask - 1; // clear lowest set bit
            }

            prev_last_class = vgetq_lane_u8(curr_classes, 15);
            pos += 16;
        }
    }

    // Scalar tail
    for i in pos..len {
        let curr_class = CLASS_LUT[bytes[i] as usize];
        if curr_class != prev_last_class {
            let suppress = matches!(
                (prev_last_class, curr_class),
                (CS, CL) | (CS, CD) | (CS, CN) | (CN, CS) | (CN, CN)
            );
            if !suppress {
                boundaries.push(i as u32);
            }
        }
        prev_last_class = curr_class;
    }

    boundaries
}

// ---------------------------------------------------------------------------
// Phase 2: Scalar fixup for contractions + multi-space
// ---------------------------------------------------------------------------

fn fixup_boundaries(bytes: &[u8], boundaries: &mut Vec<u32>) {
    let len = bytes.len();
    let mut extra = Vec::new();
    let mut remove = Vec::new();

    // Pass 1: find contractions and merge apostrophe boundaries
    // A contraction like "'t" appears as two boundaries: one at apostrophe, one at 't'.
    // We need to remove the boundary at 't' so "'t" is one piece.
    // But non-contraction "'hello" should keep both boundaries.
    for i in 0..boundaries.len() {
        let pos = boundaries[i] as usize;
        if pos < len && bytes[pos] == b'\'' {
            // Check if this is a contraction
            let rem = len - pos;
            if rem >= 2 {
                let b1 = bytes[pos + 1];
                let is_contraction_2 = matches!(b1, b's' | b't' | b'd' | b'm')
                    && (rem == 2 || !is_ascii_letter(bytes[pos + 2]));

                let is_contraction_3 = rem >= 3 && {
                    let b2 = bytes[pos + 2];
                    (b1 == b'l' && b2 == b'l')
                        || (b1 == b'v' && b2 == b'e')
                        || (b1 == b'r' && b2 == b'e')
                };

                if is_contraction_2 || is_contraction_3 {
                    // Remove the next boundary (which would split the contraction)
                    if i + 1 < boundaries.len() {
                        let next_pos = boundaries[i + 1] as usize;
                        let end = if is_contraction_3 { pos + 3 } else { pos + 2 };
                        if next_pos > pos && next_pos <= end {
                            remove.push(i + 1);
                        }
                    }
                }
            }
        }
    }

    // Pass 2: handle multi-space ("a  b" should be ["a", " ", " b"])
    // When we have consecutive spaces, each space except the last one
    // (which prefixes the next word) should be its own piece.
    // The SIMD pass doesn't create boundaries between same-class bytes.
    for i in 0..boundaries.len() {
        let pos = boundaries[i] as usize;
        if pos < len && bytes[pos] == b' ' {
            // Count consecutive spaces from this boundary
            let mut end = pos + 1;
            while end < len && bytes[end] == b' ' {
                end += 1;
            }
            let space_count = end - pos;
            if space_count > 1 {
                // Need a boundary at each space EXCEPT:
                // - The last space if it prefixes a letter/digit (already handled by SIMD)
                // Add boundaries at pos+1, pos+2, ..., pos+space_count-1
                for j in 1..space_count {
                    let bp = (pos + j) as u32;
                    // Check the boundary doesn't already exist
                    extra.push(bp);
                }
            }
        }
    }

    // Apply removals (in reverse to preserve indices)
    remove.sort_unstable();
    remove.dedup();
    for &idx in remove.iter().rev() {
        if idx < boundaries.len() {
            boundaries.remove(idx);
        }
    }

    // Apply additions
    if !extra.is_empty() {
        boundaries.extend_from_slice(&extra);
        boundaries.sort_unstable();
        boundaries.dedup();
    }
}

// ---------------------------------------------------------------------------
// Full pipeline: SIMD boundaries → fixup → count pieces
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn count_simd_full(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut boundaries = find_boundaries_simd(bytes);
    fixup_boundaries(bytes, &mut boundaries);
    boundaries.len()
}

#[cfg(target_arch = "aarch64")]
fn count_simd_no_fixup(text: &str) -> usize {
    let bytes = text.as_bytes();
    let boundaries = find_boundaries_simd(bytes);
    boundaries.len()
}

// ---------------------------------------------------------------------------
// Benchmark: Phase 1 only (boundary detection throughput)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn bench_phase1_throughput(text: &str) -> usize {
    let bytes = text.as_bytes();
    let boundaries = find_boundaries_simd(bytes);
    boundaries.len()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    let iters = 20;

    // Reference
    let ref_count = Gpt2::new(&text).count();
    println!("Reference (scalar iter):  {ref_count} pieces");

    #[cfg(target_arch = "aarch64")]
    {
        let phase1_count = count_simd_no_fixup(&text);
        println!("Phase 1 (SIMD, no fix):   {phase1_count} pieces (delta: {})", phase1_count as i64 - ref_count as i64);

        let full_count = count_simd_full(&text);
        println!("Full (SIMD + fixup):      {full_count} pieces (delta: {})", full_count as i64 - ref_count as i64);
    }

    println!();

    // Benchmark: scalar iter
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Scalar iter:       {t:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark: Phase 1 only (SIMD boundary detection)
    #[cfg(target_arch = "aarch64")]
    {
        let _ = bench_phase1_throughput(&text);
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = bench_phase1_throughput(&text); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Phase 1 (SIMD):    {t:>8.1} MB/s  ({c} boundaries)");
    }

    // Benchmark: full pipeline (SIMD + fixup)
    #[cfg(target_arch = "aarch64")]
    {
        let _ = count_simd_full(&text);
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = count_simd_full(&text); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Full pipeline:     {t:>8.1} MB/s  ({c} pieces)");
    }
}
