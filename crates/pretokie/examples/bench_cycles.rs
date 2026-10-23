//! Cycle-level analysis of pretokenizer bottlenecks.
//!
//! Measures cycles/byte for different components to identify
//! where the CPU is stalling.
//!
//! Usage: cargo run -p pretokie --example bench_cycles --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::{is_ascii_letter, is_digit};
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

const GHZ_ESTIMATE: f64 = 3.5; // Apple Silicon P-core ~3.5 GHz boost

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mb = len as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB ({} bytes)\n", mb, len);

    let iters = 30;

    // 1. Raw byte sum — memory bandwidth ceiling
    {
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..iters {
            for &b in bytes {
                sum = sum.wrapping_add(b as u64);
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("Raw byte sum:       {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [sum={sum}]");
    }

    // 2. Byte classify only — how fast can we classify each byte?
    {
        let start = Instant::now();
        let mut count = 0u64;
        for _ in 0..iters {
            for &b in bytes {
                if is_ascii_letter(b) { count += 1; }
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("Classify (letter?): {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [count={count}]");
    }

    // 3. Classify with state transition — how fast with a dependent branch?
    {
        let start = Instant::now();
        let mut transitions = 0u64;
        for _ in 0..iters {
            let mut prev_is_letter = false;
            for &b in bytes {
                let is_letter = is_ascii_letter(b);
                if is_letter != prev_is_letter { transitions += 1; }
                prev_is_letter = is_letter;
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("State transition:   {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [transitions={transitions}]");
    }

    // 4. Branchless classify — convert branch to arithmetic
    {
        let start = Instant::now();
        let mut transitions = 0u64;
        for _ in 0..iters {
            let mut prev_class = 0u8;
            for &b in bytes {
                let class = is_ascii_letter(b) as u8;
                transitions += (class != prev_class) as u64;
                prev_class = class;
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("Branchless trans:   {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [transitions={transitions}]");
    }

    // 5. Full 7-class branchless transition count
    {
        static CLASS: [u8; 256] = {
            let mut t = [6u8; 256]; // other
            let mut i = b'a';
            while i <= b'z' { t[i as usize] = 0; i += 1; }
            i = b'A';
            while i <= b'Z' { t[i as usize] = 0; i += 1; }
            i = b'0';
            while i <= b'9' { t[i as usize] = 1; i += 1; }
            t[b' ' as usize] = 2;
            t[b'\n' as usize] = 3;
            t[b'\r' as usize] = 3;
            t[b'\'' as usize] = 4;
            i = 0x80;
            loop { t[i as usize] = 5; if i == 0xFF { break; } i += 1; }
            t
        };

        let start = Instant::now();
        let mut transitions = 0u64;
        for _ in 0..iters {
            let mut prev = CLASS[bytes[0] as usize];
            for &b in &bytes[1..] {
                let curr = CLASS[b as usize];
                transitions += (curr != prev) as u64;
                prev = curr;
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("7-class LUT trans:  {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [transitions={transitions}]");
    }

    // 6. SIMD classify 16 bytes at a time
    #[cfg(target_arch = "aarch64")]
    {
        let start = Instant::now();
        let mut sum = 0u64;
        for _ in 0..iters {
            let mut pos = 0;
            unsafe {
                while pos + 16 <= len {
                    let chunk = vld1q_u8(bytes.as_ptr().add(pos));
                    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
                    let offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
                    let is_letter = vcltq_u8(offset, vdupq_n_u8(26));
                    sum = sum.wrapping_add(vaddvq_u8(is_letter) as u64);
                    pos += 16;
                }
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("NEON classify 16:   {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [sum={sum}]");
    }

    // 7. SIMD classify + transition detect 16 bytes
    #[cfg(target_arch = "aarch64")]
    {
        let start = Instant::now();
        let mut total_boundaries = 0u64;
        for _ in 0..iters {
            let mut pos = 0;
            let mut prev_last: u8 = 0;
            total_boundaries = 0;
            unsafe {
                while pos + 16 <= len {
                    let chunk = vld1q_u8(bytes.as_ptr().add(pos));
                    // Classify
                    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
                    let is_letter = vcltq_u8(
                        vsubq_u8(lowered, vdupq_n_u8(b'a')),
                        vdupq_n_u8(26),
                    );
                    let is_digit = vcltq_u8(
                        vsubq_u8(chunk, vdupq_n_u8(b'0')),
                        vdupq_n_u8(10),
                    );
                    let is_space = vceqq_u8(chunk, vdupq_n_u8(b' '));
                    let is_nl = vorrq_u8(
                        vceqq_u8(chunk, vdupq_n_u8(b'\n')),
                        vceqq_u8(chunk, vdupq_n_u8(b'\r')),
                    );
                    let is_high = vcgeq_u8(chunk, vdupq_n_u8(0x80));
                    let is_apos = vceqq_u8(chunk, vdupq_n_u8(b'\''));

                    let mut cls = vdupq_n_u8(6);
                    cls = vbslq_u8(is_high, vdupq_n_u8(5), cls);
                    cls = vbslq_u8(is_apos, vdupq_n_u8(4), cls);
                    cls = vbslq_u8(is_nl, vdupq_n_u8(3), cls);
                    cls = vbslq_u8(is_space, vdupq_n_u8(2), cls);
                    cls = vbslq_u8(is_digit, vdupq_n_u8(1), cls);
                    cls = vbslq_u8(is_letter, vdupq_n_u8(0), cls);

                    // Transition detect
                    let prev_vec = vdupq_n_u8(prev_last);
                    let shifted = vextq_u8(prev_vec, cls, 15);
                    let diff = vmvnq_u8(vceqq_u8(cls, shifted));

                    // Count transitions
                    total_boundaries += (vaddvq_u8(vshrq_n_u8(diff, 7))) as u64;

                    prev_last = vgetq_lane_u8(cls, 15);
                    pos += 16;
                }
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("NEON cls+trans 16:  {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [boundaries={total_boundaries}]");
    }

    // 8. SIMD transition + merge rule suppression
    #[cfg(target_arch = "aarch64")]
    {
        let start = Instant::now();
        let mut total_boundaries = 0u64;
        for _ in 0..iters {
            let mut pos = 0;
            let mut prev_last: u8 = 0;
            total_boundaries = 0;
            unsafe {
                while pos + 16 <= len {
                    let chunk = vld1q_u8(bytes.as_ptr().add(pos));
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
                    total_boundaries += vaddvq_u8(vshrq_n_u8(real, 7)) as u64;

                    prev_last = vgetq_lane_u8(cls, 15);
                    pos += 16;
                }
            }
        }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("NEON cls+trans+sup: {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [boundaries={total_boundaries}]");
    }

    // 9. Reference: full scalar pretokenizer
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let elapsed = start.elapsed();
        let mbs = mb * iters as f64 / elapsed.as_secs_f64();
        let cpb = GHZ_ESTIMATE * 1e9 / (mbs * 1024.0 * 1024.0);
        println!("Scalar GPT-2 iter:  {mbs:>8.0} MB/s  ({cpb:.2} cycles/byte)  [{c} pieces]");
    }

    // Summary
    println!("\n--- Analysis at {GHZ_ESTIMATE} GHz estimate ---");
    println!("Apple Silicon P-core can issue ~8 ops/cycle.");
    println!("At 310 MB/s → {:.1} cycles/byte → {:.0} ops/byte budget (at 8 ops/cycle)",
        GHZ_ESTIMATE * 1e9 / (310.0 * 1024.0 * 1024.0),
        8.0 * GHZ_ESTIMATE * 1e9 / (310.0 * 1024.0 * 1024.0));
    println!("At 2700 MB/s → {:.2} cycles/byte → {:.1} ops/byte (SIMD transition)",
        GHZ_ESTIMATE * 1e9 / (2700.0 * 1024.0 * 1024.0),
        8.0 * GHZ_ESTIMATE * 1e9 / (2700.0 * 1024.0 * 1024.0));
}
