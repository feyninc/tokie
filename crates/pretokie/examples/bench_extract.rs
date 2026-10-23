//! Benchmark: branchless boundary extraction from SIMD transition mask.
//!
//! The SIMD transition detection runs at 3080 MB/s.
//! Boundary extraction (mask → positions) dropped it to 561 MB/s.
//! Can we extract boundaries faster with branchless techniques?
//!
//! Usage: cargo run -p pretokie --example bench_extract --release

use pretokie::mask::Gpt2;
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const GHZ: f64 = 3.5;

// ---------------------------------------------------------------------------
// SIMD classify + transition detect (reused from bench_cycles)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn classify_and_detect(
    ptr: *const u8,
    prev_last: u8,
) -> (u16, u8) {
    // Returns (boundary_mask, last_class)
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let is_letter = vcltq_u8(vsubq_u8(lowered, vdupq_n_u8(b'a')), vdupq_n_u8(26));
    let is_digit = vcltq_u8(vsubq_u8(chunk, vdupq_n_u8(b'0')), vdupq_n_u8(10));
    let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
    let is_nl = vorrq_u8(vceqq_u8(chunk, vdupq_n_u8(b'\n')), vceqq_u8(chunk, vdupq_n_u8(b'\r')));
    let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));
    let is_apos = vceqq_u8(chunk, vdupq_n_u8(b'\''));

    let mut cls = vdupq_n_u8(6);
    cls = vbslq_u8(is_high, vdupq_n_u8(5), cls);
    cls = vbslq_u8(is_apos, vdupq_n_u8(4), cls);
    cls = vbslq_u8(is_nl, vdupq_n_u8(3), cls);
    cls = vbslq_u8(is_space, vdupq_n_u8(2), cls);
    cls = vbslq_u8(is_digit, vdupq_n_u8(1), cls);
    cls = vbslq_u8(is_letter, vdupq_n_u8(0), cls);

    let prev_vec = vdupq_n_u8(prev_last);
    let shifted = vextq_u8(prev_vec, cls, 15);
    let transitions = vmvnq_u8(vceqq_u8(cls, shifted));

    // Suppress merge rules
    let prev_is_space = vceqq_u8(shifted, vdupq_n_u8(2));
    let prev_is_nl = vceqq_u8(shifted, vdupq_n_u8(3));
    let curr_is_letter = vceqq_u8(cls, vdupq_n_u8(0));
    let curr_is_digit = vceqq_u8(cls, vdupq_n_u8(1));
    let curr_is_nl = vceqq_u8(cls, vdupq_n_u8(3));
    let curr_is_space = vceqq_u8(cls, vdupq_n_u8(2));

    let suppress = vorrq_u8(
        vandq_u8(prev_is_space, vorrq_u8(curr_is_letter, vorrq_u8(curr_is_digit, curr_is_nl))),
        vandq_u8(prev_is_nl, vorrq_u8(curr_is_space, curr_is_nl)),
    );

    let real = vandq_u8(transitions, vmvnq_u8(suppress));

    // Extract bitmask
    static POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powers = vld1q_u8(POWERS.as_ptr());
    let bits = vandq_u8(real, powers);
    let lo = vaddv_u8(vget_low_u8(bits)) as u16;
    let hi = vaddv_u8(vget_high_u8(bits)) as u16;
    let mask = lo | (hi << 8);

    let last = vgetq_lane_u8(cls, 15);
    (mask, last)
}

// ---------------------------------------------------------------------------
// Extraction methods
// ---------------------------------------------------------------------------

/// Method 1: Loop with ctz (original approach — branches per boundary)
#[cfg(target_arch = "aarch64")]
fn extract_boundaries_ctz(bytes: &[u8]) -> Vec<u32> {
    let len = bytes.len();
    let mut result = Vec::with_capacity(len / 3);
    result.push(0u32);
    let mut pos = 1usize;
    let mut prev_last = 0u8; // class of first byte (letter=0 for enwik8)
    // Classify first byte
    let b0 = bytes[0];
    if (b0 | 0x20).wrapping_sub(b'a') < 26 { prev_last = 0; }
    else if b0.wrapping_sub(b'0') < 10 { prev_last = 1; }
    else if b0 == b' ' { prev_last = 2; }
    else if b0 == b'\n' || b0 == b'\r' { prev_last = 3; }
    else if b0 == b'\'' { prev_last = 4; }
    else if b0 >= 0x80 { prev_last = 5; }
    else { prev_last = 6; }

    unsafe {
        while pos + 16 <= len {
            let (mask, last) = classify_and_detect(bytes.as_ptr().add(pos), prev_last);
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros() as u32;
                result.push(pos as u32 + bit);
                m &= m - 1;
            }
            prev_last = last;
            pos += 16;
        }
    }
    // Scalar tail
    {
        static CLASS: [u8; 256] = {
            let mut t = [6u8; 256];
            let mut i = b'a'; while i <= b'z' { t[i as usize] = 0; i += 1; }
            i = b'A'; while i <= b'Z' { t[i as usize] = 0; i += 1; }
            i = b'0'; while i <= b'9' { t[i as usize] = 1; i += 1; }
            t[b' ' as usize] = 2; t[b'\n' as usize] = 3; t[b'\r' as usize] = 3;
            t[b'\'' as usize] = 4;
            i = 0x80; loop { t[i as usize] = 5; if i == 0xFF { break; } i += 1; }
            t
        };
        for i in pos..len {
            let c = CLASS[bytes[i] as usize];
            if c != prev_last {
                let suppress = matches!((prev_last, c), (2,0)|(2,1)|(2,3)|(3,2)|(3,3));
                if !suppress { result.push(i as u32); }
            }
            prev_last = c;
        }
    }
    result
}

/// Method 2: Pre-computed extraction table — for each possible 4-bit mask nibble,
/// store the offsets to write. Avoids the ctz loop.
#[cfg(target_arch = "aarch64")]
fn extract_boundaries_table(bytes: &[u8]) -> Vec<u32> {
    // For each 4-bit nibble, precompute which bit positions are set.
    // nibble_offsets[nibble] = list of set bit positions (0-3)
    // nibble_count[nibble] = number of set bits
    static NIBBLE_COUNT: [u8; 16] = [0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4];
    static NIBBLE_OFFSETS: [[u8; 4]; 16] = [
        [0,0,0,0], [0,0,0,0], [1,0,0,0], [0,1,0,0],
        [2,0,0,0], [0,2,0,0], [1,2,0,0], [0,1,2,0],
        [3,0,0,0], [0,3,0,0], [1,3,0,0], [0,1,3,0],
        [2,3,0,0], [0,2,3,0], [1,2,3,0], [0,1,2,3],
    ];

    let len = bytes.len();
    let mut result = Vec::with_capacity(len / 3);
    result.push(0u32);
    let mut pos = 1usize;

    let b0 = bytes[0];
    let mut prev_last;
    if (b0 | 0x20).wrapping_sub(b'a') < 26 { prev_last = 0u8; }
    else if b0.wrapping_sub(b'0') < 10 { prev_last = 1; }
    else if b0 == b' ' { prev_last = 2; }
    else if b0 == b'\n' || b0 == b'\r' { prev_last = 3; }
    else if b0 == b'\'' { prev_last = 4; }
    else if b0 >= 0x80 { prev_last = 5; }
    else { prev_last = 6; }

    unsafe {
        while pos + 16 <= len {
            let (mask, last) = classify_and_detect(bytes.as_ptr().add(pos), prev_last);

            if mask != 0 {
                let base = pos as u32;
                // Process low nibble
                let lo = (mask & 0xF) as usize;
                let lo_count = NIBBLE_COUNT[lo] as usize;
                let lo_off = &NIBBLE_OFFSETS[lo];
                for j in 0..lo_count {
                    result.push(base + lo_off[j] as u32);
                }
                // Nibble 1 (bits 4-7)
                let n1 = ((mask >> 4) & 0xF) as usize;
                let n1_count = NIBBLE_COUNT[n1] as usize;
                let n1_off = &NIBBLE_OFFSETS[n1];
                for j in 0..n1_count {
                    result.push(base + 4 + n1_off[j] as u32);
                }
                // Nibble 2 (bits 8-11)
                let n2 = ((mask >> 8) & 0xF) as usize;
                let n2_count = NIBBLE_COUNT[n2] as usize;
                let n2_off = &NIBBLE_OFFSETS[n2];
                for j in 0..n2_count {
                    result.push(base + 8 + n2_off[j] as u32);
                }
                // Nibble 3 (bits 12-15)
                let n3 = ((mask >> 12) & 0xF) as usize;
                let n3_count = NIBBLE_COUNT[n3] as usize;
                let n3_off = &NIBBLE_OFFSETS[n3];
                for j in 0..n3_count {
                    result.push(base + 12 + n3_off[j] as u32);
                }
            }

            prev_last = last;
            pos += 16;
        }
    }
    // Scalar tail (same as method 1)
    {
        static CLASS: [u8; 256] = {
            let mut t = [6u8; 256];
            let mut i = b'a'; while i <= b'z' { t[i as usize] = 0; i += 1; }
            i = b'A'; while i <= b'Z' { t[i as usize] = 0; i += 1; }
            i = b'0'; while i <= b'9' { t[i as usize] = 1; i += 1; }
            t[b' ' as usize] = 2; t[b'\n' as usize] = 3; t[b'\r' as usize] = 3;
            t[b'\'' as usize] = 4;
            i = 0x80; loop { t[i as usize] = 5; if i == 0xFF { break; } i += 1; }
            t
        };
        for i in pos..len {
            let c = CLASS[bytes[i] as usize];
            if c != prev_last {
                let suppress = matches!((prev_last, c), (2,0)|(2,1)|(2,3)|(3,2)|(3,3));
                if !suppress { result.push(i as u32); }
            }
            prev_last = c;
        }
    }
    result
}

/// Method 3: Raw pointer write — skip Vec bounds checks entirely
#[cfg(target_arch = "aarch64")]
fn extract_boundaries_raw(bytes: &[u8]) -> Vec<u32> {
    let len = bytes.len();
    // Worst case: every byte is a boundary
    let mut result = Vec::with_capacity(len / 2);
    unsafe { result.set_len(0); }
    let out_base: *mut u32 = result.as_mut_ptr();
    let mut out_pos = 0usize;

    // Write first boundary
    unsafe { *out_base.add(out_pos) = 0; }
    out_pos += 1;

    let mut pos = 1usize;
    let b0 = bytes[0];
    let mut prev_last;
    if (b0 | 0x20).wrapping_sub(b'a') < 26 { prev_last = 0u8; }
    else if b0.wrapping_sub(b'0') < 10 { prev_last = 1; }
    else if b0 == b' ' { prev_last = 2; }
    else if b0 == b'\n' || b0 == b'\r' { prev_last = 3; }
    else if b0 == b'\'' { prev_last = 4; }
    else if b0 >= 0x80 { prev_last = 5; }
    else { prev_last = 6; }

    unsafe {
        while pos + 16 <= len {
            let (mask, last) = classify_and_detect(bytes.as_ptr().add(pos), prev_last);
            let mut m = mask;
            let base = pos as u32;
            while m != 0 {
                let bit = m.trailing_zeros() as u32;
                *out_base.add(out_pos) = base + bit;
                out_pos += 1;
                m &= m - 1;
            }
            prev_last = last;
            pos += 16;
        }
    }
    // Scalar tail
    {
        static CLASS: [u8; 256] = {
            let mut t = [6u8; 256];
            let mut i = b'a'; while i <= b'z' { t[i as usize] = 0; i += 1; }
            i = b'A'; while i <= b'Z' { t[i as usize] = 0; i += 1; }
            i = b'0'; while i <= b'9' { t[i as usize] = 1; i += 1; }
            t[b' ' as usize] = 2; t[b'\n' as usize] = 3; t[b'\r' as usize] = 3;
            t[b'\'' as usize] = 4;
            i = 0x80; loop { t[i as usize] = 5; if i == 0xFF { break; } i += 1; }
            t
        };
        for i in pos..len {
            let c = CLASS[bytes[i] as usize];
            if c != prev_last {
                let suppress = matches!((prev_last, c), (2,0)|(2,1)|(2,3)|(3,2)|(3,3));
                if !suppress {
                    unsafe { *out_base.add(out_pos) = i as u32; }
                    out_pos += 1;
                }
            }
            prev_last = c;
        }
    }
    unsafe { result.set_len(out_pos); }
    result
}

/// Method 4: Just count boundaries (no Vec), for comparison
#[cfg(target_arch = "aarch64")]
fn count_boundaries_only(bytes: &[u8]) -> usize {
    let len = bytes.len();
    let mut count = 1usize;
    let mut pos = 1usize;

    let b0 = bytes[0];
    let mut prev_last;
    if (b0 | 0x20).wrapping_sub(b'a') < 26 { prev_last = 0u8; }
    else if b0.wrapping_sub(b'0') < 10 { prev_last = 1; }
    else if b0 == b' ' { prev_last = 2; }
    else if b0 == b'\n' || b0 == b'\r' { prev_last = 3; }
    else if b0 == b'\'' { prev_last = 4; }
    else if b0 >= 0x80 { prev_last = 5; }
    else { prev_last = 6; }

    unsafe {
        while pos + 16 <= len {
            let (mask, last) = classify_and_detect(bytes.as_ptr().add(pos), prev_last);
            count += mask.count_ones() as usize;
            prev_last = last;
            pos += 16;
        }
    }
    count
}

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let bytes = text.as_bytes();
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    let iters = 20;

    // Reference
    let ref_count = Gpt2::new(&text).count();
    println!("Reference: {ref_count} pieces\n");

    #[cfg(target_arch = "aarch64")]
    {
        // Count only (ceiling)
        {
            let _ = count_boundaries_only(bytes);
            let start = Instant::now();
            let mut c = 0;
            for _ in 0..iters { c = count_boundaries_only(bytes); }
            let elapsed = start.elapsed();
            let mbs = mb * iters as f64 / elapsed.as_secs_f64();
            let cpb = GHZ * 1e9 / (mbs * 1024.0 * 1024.0);
            println!("Count only:     {mbs:>8.1} MB/s  ({cpb:.2} cyc/B)  ({c} boundaries)");
        }

        // CTZ extraction
        {
            let r = extract_boundaries_ctz(bytes);
            let n = r.len();
            drop(r);
            let start = Instant::now();
            for _ in 0..iters {
                let r = extract_boundaries_ctz(bytes);
                std::hint::black_box(&r);
            }
            let elapsed = start.elapsed();
            let mbs = mb * iters as f64 / elapsed.as_secs_f64();
            let cpb = GHZ * 1e9 / (mbs * 1024.0 * 1024.0);
            println!("CTZ extract:    {mbs:>8.1} MB/s  ({cpb:.2} cyc/B)  ({n} boundaries)");
        }

        // Table extraction
        {
            let r = extract_boundaries_table(bytes);
            let n = r.len();
            drop(r);
            let start = Instant::now();
            for _ in 0..iters {
                let r = extract_boundaries_table(bytes);
                std::hint::black_box(&r);
            }
            let elapsed = start.elapsed();
            let mbs = mb * iters as f64 / elapsed.as_secs_f64();
            let cpb = GHZ * 1e9 / (mbs * 1024.0 * 1024.0);
            println!("Table extract:  {mbs:>8.1} MB/s  ({cpb:.2} cyc/B)  ({n} boundaries)");
        }

        // Raw pointer extraction
        {
            let r = extract_boundaries_raw(bytes);
            let n = r.len();
            drop(r);
            let start = Instant::now();
            for _ in 0..iters {
                let r = extract_boundaries_raw(bytes);
                std::hint::black_box(&r);
            }
            let elapsed = start.elapsed();
            let mbs = mb * iters as f64 / elapsed.as_secs_f64();
            let cpb = GHZ * 1e9 / (mbs * 1024.0 * 1024.0);
            println!("Raw ptr:        {mbs:>8.1} MB/s  ({cpb:.2} cyc/B)  ({n} boundaries)");
        }
    }

    // Scalar reference
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("Scalar iter:    {mbs:>8.1} MB/s  ({cpb:.2} cyc/B)  ({c} pieces)");
    }
}
