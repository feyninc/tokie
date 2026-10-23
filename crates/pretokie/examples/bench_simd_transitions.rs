//! Prototype: SIMD transition-based piece counting for GPT-2.
//!
//! Instead of processing one piece at a time (dispatch per piece),
//! classify 16 bytes at once and find all transitions (piece boundaries)
//! via SIMD comparison. No per-piece branch dispatch.
//!
//! Usage: cargo run -p pretokie --example bench_simd_transitions --release

use pretokie::mask::Gpt2;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Byte classes (must fit in a byte, order doesn't matter for correctness)
const CL: u8 = 0; // ASCII letter
const CD: u8 = 1; // digit
const CS: u8 = 2; // space
const CN: u8 = 3; // newline (\r, \n)
const CA: u8 = 4; // apostrophe
const CH: u8 = 5; // high byte (>= 0x80)
const CO: u8 = 6; // other ASCII punct

static CLASS_LUT: [u8; 256] = {
    let mut t = [CO; 256];
    let mut i = b'a';
    while i <= b'z' {
        t[i as usize] = CL;
        i += 1;
    }
    i = b'A';
    while i <= b'Z' {
        t[i as usize] = CL;
        i += 1;
    }
    i = b'0';
    while i <= b'9' {
        t[i as usize] = CD;
        i += 1;
    }
    t[b' ' as usize] = CS;
    t[b'\n' as usize] = CN;
    t[b'\r' as usize] = CN;
    t[b'\'' as usize] = CA;
    i = 0x80;
    loop {
        t[i as usize] = CH;
        if i == 0xFF {
            break;
        }
        i += 1;
    }
    t
};

// ---------------------------------------------------------------------------
// Approach 1: Scalar transition counting (baseline for Phase 2)
// ---------------------------------------------------------------------------

fn count_scalar_transitions(text: &str) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return 0;
    }

    let mut count = 1usize; // first byte always starts a piece
    let mut prev_class = CLASS_LUT[bytes[0] as usize];

    for i in 1..len {
        let curr_class = CLASS_LUT[bytes[i] as usize];
        if curr_class != prev_class {
            // Raw transition — apply merge rules
            let is_boundary = match (prev_class, curr_class) {
                (CS, CL) => false, // space prefixes letter
                (CS, CD) => false, // space prefixes digit
                (CS, CN) | (CN, CS) | (CN, CN) => false, // whitespace runs
                (CA, CL) | (CA, CD) => false, // apostrophe+letter = contraction or punct+letter
                _ => true,
            };
            if is_boundary {
                count += 1;
            }
        }
        prev_class = curr_class;
    }

    count
}

// ---------------------------------------------------------------------------
// Approach 2: SIMD classify + SIMD transition detection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
unsafe fn classify_16(ptr: *const u8, lut_ptr: *const u8) -> uint8x16_t {
    // Classify 16 bytes using the LUT.
    // We split the LUT into 16 × 16-byte tables and use vtbl for lookup.
    // But vtbl only handles 4-bit indices (0..15).
    //
    // Alternative: scalar classify into a buffer, then load as NEON.
    // For prototyping, let's do scalar classify since the LUT approach
    // with vtbl requires splitting the 256-entry table into nibble lookups.
    let mut buf = [0u8; 16];
    for i in 0..16 {
        buf[i] = *lut_ptr.add(*ptr.add(i) as usize);
    }
    vld1q_u8(buf.as_ptr())
}

#[cfg(target_arch = "aarch64")]
unsafe fn classify_16_fast(ptr: *const u8) -> uint8x16_t {
    // SIMD classify without LUT: use arithmetic.
    // This is faster than LUT for the common classes.
    let chunk = vld1q_u8(ptr);

    // Letters: (b | 0x20) - 'a' < 26
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let letter_offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(letter_offset, vdupq_n_u8(26)); // 0xFF if letter

    // Digits: b - '0' < 10
    let digit_offset = vsubq_u8(chunk, vdupq_n_u8(b'0'));
    let is_digit = vcltq_u8(digit_offset, vdupq_n_u8(10));

    // Space: b == ' '
    let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));

    // Newline: b == '\n' || b == '\r'
    let is_nl = vorrq_u8(
        vceqq_u8(chunk, vdupq_n_u8(b'\n')),
        vceqq_u8(chunk, vdupq_n_u8(b'\r')),
    );

    // Apostrophe: b == '\''
    let is_apos = vceqq_u8(chunk, vdupq_n_u8(b'\''));

    // High: b >= 0x80
    let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));

    // Assign classes by priority (later wins via bitwise select):
    // Start with CO (6), then overwrite with detected classes.
    let mut result = vdupq_n_u8(CO);
    result = vbslq_u8(is_high, vdupq_n_u8(CH), result);
    result = vbslq_u8(is_apos, vdupq_n_u8(CA), result);
    result = vbslq_u8(is_nl, vdupq_n_u8(CN), result);
    result = vbslq_u8(is_space, vdupq_n_u8(CS), result);
    result = vbslq_u8(is_digit, vdupq_n_u8(CD), result);
    result = vbslq_u8(is_letter, vdupq_n_u8(CL), result);

    result
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn neon_popcount_u8(v: uint8x16_t) -> u32 {
    // Count bytes that are 0xFF (each 0xFF byte = 1 boundary)
    // Shift right by 7 to get 0 or 1 per lane, then horizontal sum.
    let bits = vshrq_n_u8(v, 7);
    vaddvq_u8(bits) as u32
}

#[cfg(target_arch = "aarch64")]
fn count_simd_transitions(text: &str) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 {
        return 0;
    }

    let mut count = 1usize; // first byte starts a piece
    let mut pos = 0;

    // We need the class of the last byte of the previous chunk
    // to compare with the first byte of the current chunk.
    let mut prev_last_class = CLASS_LUT[bytes[0] as usize];

    // Process first byte separately (already counted)
    pos = 1;

    unsafe {
        // Merge rule masks: we want to suppress certain transitions.
        // A transition is: class[i] != class[i-1].
        // We suppress: (prev=CS, curr=CL), (prev=CS, curr=CD),
        //              (prev=CS, curr=CN), (prev=CN, curr=CS), (prev=CN, curr=CN)
        //              (prev=CA, curr=CL), (prev=CA, curr=CD)

        while pos + 16 <= len {
            let curr_classes = classify_16_fast(bytes.as_ptr().add(pos));

            // Previous classes: shift curr_classes right by 1, inserting prev_last_class
            // prev_classes[0] = prev_last_class, prev_classes[1..15] = curr_classes[0..14]
            let prev_last_vec = vdupq_n_u8(prev_last_class);
            let prev_classes = vextq_u8(prev_last_vec, curr_classes, 15);

            // Find transitions: curr != prev
            let transitions = vmvnq_u8(vceqq_u8(curr_classes, prev_classes));

            // Now apply merge rules: suppress certain (prev, curr) pairs.
            // For each transition position, check if (prev_class, curr_class) should be merged.

            // prev == CS
            let prev_is_space = vceqq_u8(prev_classes, vdupq_n_u8(CS));
            // curr == CL or curr == CD
            let curr_is_letter = vceqq_u8(curr_classes, vdupq_n_u8(CL));
            let curr_is_digit = vceqq_u8(curr_classes, vdupq_n_u8(CD));
            let curr_is_newline = vceqq_u8(curr_classes, vdupq_n_u8(CN));
            // Suppress: (CS→CL), (CS→CD), (CS→CN)
            let suppress_space = vandq_u8(
                prev_is_space,
                vorrq_u8(curr_is_letter, vorrq_u8(curr_is_digit, curr_is_newline)),
            );

            // prev == CN
            let prev_is_newline = vceqq_u8(prev_classes, vdupq_n_u8(CN));
            // Suppress: (CN→CS), (CN→CN)
            let curr_is_space = vceqq_u8(curr_classes, vdupq_n_u8(CS));
            let suppress_newline = vandq_u8(
                prev_is_newline,
                vorrq_u8(curr_is_space, curr_is_newline),
            );

            // prev == CA (apostrophe)
            let prev_is_apos = vceqq_u8(prev_classes, vdupq_n_u8(CA));
            // Suppress: (CA→CL), (CA→CD) — contraction or punct-prefix
            let suppress_apos = vandq_u8(
                prev_is_apos,
                vorrq_u8(curr_is_letter, curr_is_digit),
            );

            // Combined suppression mask
            let suppress = vorrq_u8(suppress_space, vorrq_u8(suppress_newline, suppress_apos));

            // Real boundaries: transitions AND NOT suppressed
            let real_boundaries = vandq_u8(transitions, vmvnq_u8(suppress));

            // Count boundaries in this chunk
            count += neon_popcount_u8(real_boundaries) as usize;

            // Save last class for next iteration
            prev_last_class = vgetq_lane_u8(curr_classes, 15);
            pos += 16;
        }
    }

    // Scalar tail for remaining bytes
    for i in pos..len {
        let curr_class = CLASS_LUT[bytes[i] as usize];
        if curr_class != prev_last_class {
            let is_boundary = match (prev_last_class, curr_class) {
                (CS, CL) | (CS, CD) | (CS, CN) => false,
                (CN, CS) | (CN, CN) => false,
                (CA, CL) | (CA, CD) => false,
                _ => true,
            };
            if is_boundary {
                count += 1;
            }
        }
        prev_last_class = curr_class;
    }

    count
}

// ---------------------------------------------------------------------------
// Approach 3: Fused SIMD — classify_16_fast only (measure classify cost)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
fn bench_classify_only(text: &str) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut sum = 0u64;
    unsafe {
        while pos + 16 <= len {
            let classes = classify_16_fast(bytes.as_ptr().add(pos));
            // Prevent optimization by accumulating
            sum = sum.wrapping_add(vaddvq_u8(classes) as u64);
            pos += 16;
        }
    }
    sum as usize
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

    // Reference: scalar iterator
    let scalar_count = Gpt2::new(&text).count();
    println!("Reference (scalar iter): {scalar_count} pieces");

    // Scalar transition count
    let scalar_trans = count_scalar_transitions(&text);
    println!("Scalar transitions:      {scalar_trans} pieces");

    // SIMD transition count
    #[cfg(target_arch = "aarch64")]
    let simd_trans = count_simd_transitions(&text);
    #[cfg(not(target_arch = "aarch64"))]
    let simd_trans = 0usize;
    println!("SIMD transitions:        {simd_trans} pieces");

    if scalar_count != scalar_trans {
        println!("\nNOTE: scalar transition count differs from iterator ({scalar_trans} vs {scalar_count})");
        println!("      This is expected — transition counting is an approximation that doesn't handle");
        println!("      all GPT-2 rules (contractions, multi-space, etc.)");
    }
    if scalar_trans != simd_trans {
        println!("\nWARNING: SIMD transition count differs from scalar transition ({simd_trans} vs {scalar_trans})!");
    }
    println!();

    // Benchmark: scalar iterator
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            c = Gpt2::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("Scalar iter:        {throughput:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark: scalar transitions
    {
        let _ = count_scalar_transitions(&text);
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            c = count_scalar_transitions(&text);
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("Scalar transitions: {throughput:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark: SIMD classify only
    #[cfg(target_arch = "aarch64")]
    {
        let _ = bench_classify_only(&text);
        let start = Instant::now();
        for _ in 0..iters {
            let _ = bench_classify_only(&text);
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("SIMD classify only: {throughput:>8.1} MB/s");
    }

    // Benchmark: SIMD transitions
    #[cfg(target_arch = "aarch64")]
    {
        let _ = count_simd_transitions(&text);
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            c = count_simd_transitions(&text);
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("SIMD transitions:   {throughput:>8.1} MB/s  ({c} pieces)");
    }
}
