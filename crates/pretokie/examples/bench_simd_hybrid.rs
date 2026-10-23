//! Hybrid SIMD pretokenizer: SIMD for common patterns, scalar for edge cases.
//!
//! The two most common piece types in GPT-2 on enwik8:
//!   1. Letter run (57% of pieces by start byte)
//!   2. Space + letter run (17% of pieces)
//! Together = 74% of pieces. SIMD accelerates both.
//!
//! Usage: cargo run -p pretokie --example bench_simd_hybrid --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::{decode_utf8, is_ascii_letter, is_digit};
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// SIMD: find first non-ASCII-letter in 16 bytes
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn first_non_letter_16(ptr: *const u8) -> usize {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(offset, vdupq_n_u8(26));

    if vminvq_u8(is_letter) == 0xFF {
        return 16;
    }

    let not_letter = vmvnq_u8(is_letter);
    let bitmask = neon_movemask(not_letter);
    bitmask.trailing_zeros() as usize
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
// Hybrid GPT-2 pretokenizer
// ---------------------------------------------------------------------------

pub struct Gpt2HybridIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Gpt2HybridIter<'a> {
    pub fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    /// SIMD-accelerated letter scan: skip 16 ASCII letters at a time.
    #[inline(always)]
    fn scan_letters(&mut self) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            while self.pos + 16 <= self.len {
                let n = first_non_letter_16(self.bytes.as_ptr().add(self.pos));
                self.pos += n;
                if n < 16 {
                    if self.pos < self.len && self.at(self.pos) >= 0x80 {
                        let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_alphabetic() { self.pos += cl; continue; }
                    }
                    return;
                }
            }
        }
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) { self.pos += 1; }
            else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_alphabetic() { self.pos += cl; } else { return; }
            } else { return; }
        }
    }

    #[inline(always)]
    fn scan_digits(&mut self) {
        while self.pos < self.len && is_digit(self.at(self.pos)) {
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn check_contraction(&self) -> usize {
        if self.pos >= self.len || self.bytes[self.pos] != b'\'' { return 0; }
        let rem = self.len - self.pos;
        if rem < 2 { return 0; }
        let b1 = self.bytes[self.pos + 1];
        if matches!(b1, b's' | b't' | b'd' | b'm') {
            if rem == 2 || !is_ascii_letter(self.bytes[self.pos + 2]) {
                return 2;
            }
        }
        if rem < 3 { return 0; }
        let b2 = self.bytes[self.pos + 2];
        if (b1 == b'l' && b2 == b'l')
            || (b1 == b'v' && b2 == b'e')
            || (b1 == b'r' && b2 == b'e')
        { return 3; }
        0
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) || is_digit(b) || b == b' '
                || b == b'\n' || b == b'\r' || b >= 0x80
            { break; }
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }

    /// Fast path: if pos starts a letter or space+letter, scan with SIMD.
    /// Returns true if handled, false to fall through to scalar dispatch.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn try_letter_piece(&mut self) -> bool {
        let b0 = self.at(self.pos);

        // Case 1: starts with ASCII letter
        if is_ascii_letter(b0) {
            self.pos += 1;
            self.scan_letters();
            return true;
        }

        // Case 2: space + ASCII letter
        if b0 == b' ' && self.pos + 1 < self.len {
            let b1 = self.at(self.pos + 1);
            if is_ascii_letter(b1) {
                self.pos += 2;
                self.scan_letters();
                return true;
            }
        }

        false
    }
}

impl<'a> Iterator for Gpt2HybridIter<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }
        let start = self.pos;

        // SIMD fast path for letter and space+letter pieces (74% of all pieces)
        #[cfg(target_arch = "aarch64")]
        {
            if self.try_letter_piece() {
                // Check for contraction after letter run
                if self.check_contraction() > 0 {
                    return Some(self.emit(start));
                }
                return Some(self.emit(start));
            }
        }

        // Scalar path for remaining 26% of pieces
        let b = self.at(self.pos);

        // Letters (fallback for short runs / non-aarch64)
        #[cfg(not(target_arch = "aarch64"))]
        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters();
            if self.check_contraction() > 0 {
                return Some(self.emit(start));
            }
            return Some(self.emit(start));
        }

        #[cfg(not(target_arch = "aarch64"))]
        if b == b' ' {
            self.pos += 1;
            if self.pos < self.len {
                let next = self.at(self.pos);
                if is_ascii_letter(next) {
                    self.pos += 1;
                    self.scan_letters();
                    if self.check_contraction() > 0 {
                        return Some(self.emit(start));
                    }
                    return Some(self.emit(start));
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos..]);
                    if ch.is_alphabetic() { self.scan_letters(); }
                } else if is_digit(next) {
                    self.pos += 1;
                    self.scan_digits();
                }
            }
            return Some(self.emit(start));
        }

        // Space with non-letter next (on aarch64, try_simd already handled space+letter)
        #[cfg(target_arch = "aarch64")]
        if b == b' ' {
            self.pos += 1;
            if self.pos < self.len {
                let next = self.at(self.pos);
                if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos..]);
                    if ch.is_alphabetic() { self.scan_letters(); }
                } else if is_digit(next) {
                    self.pos += 1;
                    self.scan_digits();
                }
            }
            return Some(self.emit(start));
        }

        if b == b'\'' {
            let clen = self.check_contraction();
            if clen > 0 { self.pos += clen; }
            else { self.pos += 1; self.scan_punct(); }
        } else if is_digit(b) {
            self.pos += 1;
            self.scan_digits();
        } else if b == b'\n' || b == b'\r' {
            self.pos += 1;
            while self.pos < self.len {
                let c = self.at(self.pos);
                if c == b'\n' || c == b'\r' || c == b' ' { self.pos += 1; }
                else { break; }
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
            self.pos += cl;
            if ch.is_alphabetic() {
                self.scan_letters();
            } else if ch.is_whitespace() {
                while self.pos < self.len {
                    let c = self.at(self.pos);
                    if c == b' ' || c == b'\n' || c == b'\r' { self.pos += 1; }
                    else if c >= 0x80 {
                        let (ch2, cl2) = decode_utf8(&self.bytes[self.pos..]);
                        if ch2.is_whitespace() { self.pos += cl2; } else { break; }
                    } else { break; }
                }
            }
        } else {
            self.pos += 1;
            self.scan_punct();
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
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

    // Correctness check
    {
        let ref_count = Gpt2::new(&text).count();
        let hybrid_count = Gpt2HybridIter::new(&text).count();
        println!("Scalar:  {ref_count} pieces");
        println!("Hybrid:  {hybrid_count} pieces");
        if ref_count != hybrid_count {
            println!("MISMATCH!");
            let mut r = Gpt2::new(&text);
            let mut h = Gpt2HybridIter::new(&text);
            for i in 0.. {
                let a = r.next();
                let b = h.next();
                if a != b {
                    println!("  at piece {i}: ref={a:?} hybrid={b:?}");
                    // show context
                    for j in 0..5 {
                        let a2 = r.next();
                        let b2 = h.next();
                        println!("  at piece {}: ref={:?} hybrid={:?}", i+1+j, a2, b2);
                    }
                    break;
                }
                if a.is_none() { break; }
            }
            return;
        }
        println!("Correctness: OK\n");
    }

    let iters = 20;

    // Benchmark scalar
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Scalar:   {t:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark hybrid
    {
        let _ = Gpt2HybridIter::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2HybridIter::new(&text).count(); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Hybrid:   {t:>8.1} MB/s  ({c} pieces)");
    }
}
