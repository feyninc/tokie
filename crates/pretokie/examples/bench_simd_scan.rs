//! Prototype: SIMD-accelerated letter scanning for GPT-2 pretokenizer.
//!
//! Tests whether NEON intrinsics can beat the scalar scan_letters loop
//! by processing 16 bytes at a time.
//!
//! Usage: cargo run -p pretokie --example bench_simd_scan --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::{decode_utf8, is_ascii_letter, is_digit};
use std::time::Instant;

// ---------------------------------------------------------------------------
// SIMD helpers (aarch64 NEON)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Count leading ASCII letters starting at `ptr`, examining up to 16 bytes.
/// Returns 0..=16.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn leading_ascii_letters_16(ptr: *const u8) -> usize {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(offset, vdupq_n_u8(26));

    // Fast path: all 16 are letters
    if vminvq_u8(is_letter) == 0xFF {
        return 16;
    }

    // Extract bitmask of non-letter positions, find first
    let not_letter = vmvnq_u8(is_letter);
    let bitmask = neon_movemask(not_letter);
    bitmask.trailing_zeros() as usize
}

/// Extract a 16-bit bitmask from a NEON mask register.
/// Bit i is set if lane i is 0xFF.
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
// SIMD GPT-2 pretokenizer
// ---------------------------------------------------------------------------

struct Gpt2SimdIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Gpt2SimdIter<'a> {
    fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            while self.pos + 16 <= self.len {
                let count = unsafe {
                    leading_ascii_letters_16(self.bytes.as_ptr().add(self.pos))
                };
                self.pos += count;
                if count < 16 {
                    if self.pos < self.len && self.at(self.pos) >= 0x80 {
                        let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_alphabetic() {
                            self.pos += cl;
                            continue;
                        }
                    }
                    return;
                }
            }
        }
        // Scalar tail
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_alphabetic() { self.pos += cl; } else { return; }
            } else {
                return;
            }
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
        {
            return 3;
        }
        0
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 {
                break;
            }
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }
}

impl<'a> Iterator for Gpt2SimdIter<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let b = self.at(self.pos);

        if is_ascii_letter(b) {
            self.pos += 1;
            self.scan_letters();
            if self.check_contraction() > 0 {
                return Some(self.emit(start));
            }
        } else if b == b' ' {
            self.pos += 1;
            if self.pos < self.len {
                let next = self.at(self.pos);
                if is_ascii_letter(next) {
                    self.pos += 1;
                    self.scan_letters();
                    if self.check_contraction() > 0 {
                        return Some(self.emit(start));
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&self.bytes[self.pos..]);
                    if ch.is_alphabetic() {
                        self.scan_letters();
                    }
                } else if is_digit(next) {
                    self.pos += 1;
                    self.scan_digits();
                }
            }
        } else if b == b'\'' {
            let clen = self.check_contraction();
            if clen > 0 {
                self.pos += clen;
            } else {
                self.pos += 1;
                self.scan_punct();
            }
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
// Variant 2: SIMD + LUT dispatch (avoid branch chain in next())
// ---------------------------------------------------------------------------

/// Byte class for initial dispatch — avoids if-else chain.
const C_LETTER: u8 = 0;
const C_SPACE: u8 = 1;
const C_APOS: u8 = 2;
const C_DIGIT: u8 = 3;
const C_NEWLINE: u8 = 4;
const C_HIGH: u8 = 5;
const C_OTHER: u8 = 6;

static BYTE_CLASS: [u8; 256] = {
    let mut t = [C_OTHER; 256];
    let mut i = b'a';
    while i <= b'z' { t[i as usize] = C_LETTER; i += 1; }
    i = b'A';
    while i <= b'Z' { t[i as usize] = C_LETTER; i += 1; }
    i = b'0';
    while i <= b'9' { t[i as usize] = C_DIGIT; i += 1; }
    t[b' ' as usize] = C_SPACE;
    t[b'\n' as usize] = C_NEWLINE;
    t[b'\r' as usize] = C_NEWLINE;
    t[b'\'' as usize] = C_APOS;
    // 0x80..=0xFF → C_HIGH
    i = 0x80;
    loop {
        t[i as usize] = C_HIGH;
        if i == 0xFF { break; }
        i += 1;
    }
    t
};

struct Gpt2SimdLutIter<'a> {
    bytes: &'a [u8],
    pos: usize,
    len: usize,
}

impl<'a> Gpt2SimdLutIter<'a> {
    fn new(text: &'a str) -> Self {
        let bytes = text.as_bytes();
        Self { bytes, pos: 0, len: bytes.len() }
    }

    #[inline(always)]
    fn at(&self, pos: usize) -> u8 {
        unsafe { *self.bytes.get_unchecked(pos) }
    }

    #[inline(always)]
    fn class(&self, pos: usize) -> u8 {
        unsafe { *BYTE_CLASS.get_unchecked(self.at(pos) as usize) }
    }

    #[inline(always)]
    fn scan_letters(&mut self) {
        #[cfg(target_arch = "aarch64")]
        {
            while self.pos + 16 <= self.len {
                let count = unsafe {
                    leading_ascii_letters_16(self.bytes.as_ptr().add(self.pos))
                };
                self.pos += count;
                if count < 16 {
                    if self.pos < self.len && self.at(self.pos) >= 0x80 {
                        let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_alphabetic() {
                            self.pos += cl;
                            continue;
                        }
                    }
                    return;
                }
            }
        }
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) {
                self.pos += 1;
            } else if b >= 0x80 {
                let (ch, cl) = decode_utf8(&self.bytes[self.pos..]);
                if ch.is_alphabetic() { self.pos += cl; } else { return; }
            } else {
                return;
            }
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
        {
            return 3;
        }
        0
    }

    #[inline(always)]
    fn scan_punct(&mut self) {
        while self.pos < self.len {
            let b = self.at(self.pos);
            if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 {
                break;
            }
            self.pos += 1;
        }
    }

    #[inline(always)]
    fn emit(&self, start: usize) -> &'a str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[start..self.pos]) }
    }
}

impl<'a> Iterator for Gpt2SimdLutIter<'a> {
    type Item = &'a str;

    #[inline(always)]
    fn next(&mut self) -> Option<&'a str> {
        if self.pos >= self.len { return None; }

        let start = self.pos;
        let cls = self.class(self.pos);

        match cls {
            C_LETTER => {
                self.pos += 1;
                self.scan_letters();
                if self.check_contraction() > 0 {
                    return Some(self.emit(start));
                }
            }
            C_SPACE => {
                self.pos += 1;
                if self.pos < self.len {
                    let nc = self.class(self.pos);
                    if nc == C_LETTER {
                        self.pos += 1;
                        self.scan_letters();
                        if self.check_contraction() > 0 {
                            return Some(self.emit(start));
                        }
                    } else if nc == C_HIGH {
                        let (ch, _) = decode_utf8(&self.bytes[self.pos..]);
                        if ch.is_alphabetic() {
                            self.scan_letters();
                        }
                    } else if nc == C_DIGIT {
                        self.pos += 1;
                        self.scan_digits();
                    }
                }
            }
            C_APOS => {
                let clen = self.check_contraction();
                if clen > 0 {
                    self.pos += clen;
                } else {
                    self.pos += 1;
                    self.scan_punct();
                }
            }
            C_DIGIT => {
                self.pos += 1;
                self.scan_digits();
            }
            C_NEWLINE => {
                self.pos += 1;
                while self.pos < self.len {
                    let c = self.at(self.pos);
                    if c == b'\n' || c == b'\r' || c == b' ' { self.pos += 1; }
                    else { break; }
                }
            }
            C_HIGH => {
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
            }
            _ /* C_OTHER */ => {
                self.pos += 1;
                self.scan_punct();
            }
        }

        debug_assert!(self.pos > start, "no progress at pos {start}");
        Some(self.emit(start))
    }
}

// ---------------------------------------------------------------------------
// Variant 3: SIMD + LUT + bulk count (no iterator overhead)
// ---------------------------------------------------------------------------

fn count_pieces_bulk(text: &str) -> usize {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0;
    let mut count = 0usize;

    #[inline(always)]
    fn at(bytes: &[u8], pos: usize) -> u8 {
        unsafe { *bytes.get_unchecked(pos) }
    }

    while pos < len {
        count += 1;
        let cls = unsafe { *BYTE_CLASS.get_unchecked(at(bytes, pos) as usize) };

        match cls {
            C_LETTER => {
                pos += 1;
                // SIMD scan letters
                #[cfg(target_arch = "aarch64")]
                {
                    while pos + 16 <= len {
                        let c = unsafe { leading_ascii_letters_16(bytes.as_ptr().add(pos)) };
                        pos += c;
                        if c < 16 {
                            if pos < len && at(bytes, pos) >= 0x80 {
                                let (ch, cl) = decode_utf8(&bytes[pos..]);
                                if ch.is_alphabetic() { pos += cl; continue; }
                            }
                            break;
                        }
                    }
                }
                while pos < len {
                    let b = at(bytes, pos);
                    if is_ascii_letter(b) { pos += 1; }
                    else if b >= 0x80 {
                        let (ch, cl) = decode_utf8(&bytes[pos..]);
                        if ch.is_alphabetic() { pos += cl; } else { break; }
                    } else { break; }
                }
                // Don't consume contraction here, just stop before it
                if pos < len && bytes[pos] == b'\'' {
                    let rem = len - pos;
                    if rem >= 2 {
                        let b1 = bytes[pos + 1];
                        if matches!(b1, b's' | b't' | b'd' | b'm') {
                            if rem == 2 || !is_ascii_letter(bytes[pos + 2]) {
                                continue;
                            }
                        }
                        if rem >= 3 {
                            let b2 = bytes[pos + 2];
                            if (b1 == b'l' && b2 == b'l')
                                || (b1 == b'v' && b2 == b'e')
                                || (b1 == b'r' && b2 == b'e')
                            {
                                continue;
                            }
                        }
                    }
                }
            }
            C_SPACE => {
                pos += 1;
                if pos < len {
                    let nc = unsafe { *BYTE_CLASS.get_unchecked(at(bytes, pos) as usize) };
                    if nc == C_LETTER {
                        pos += 1;
                        #[cfg(target_arch = "aarch64")]
                        {
                            while pos + 16 <= len {
                                let c = unsafe { leading_ascii_letters_16(bytes.as_ptr().add(pos)) };
                                pos += c;
                                if c < 16 {
                                    if pos < len && at(bytes, pos) >= 0x80 {
                                        let (ch, cl) = decode_utf8(&bytes[pos..]);
                                        if ch.is_alphabetic() { pos += cl; continue; }
                                    }
                                    break;
                                }
                            }
                        }
                        while pos < len {
                            let b = at(bytes, pos);
                            if is_ascii_letter(b) { pos += 1; }
                            else if b >= 0x80 {
                                let (ch, cl) = decode_utf8(&bytes[pos..]);
                                if ch.is_alphabetic() { pos += cl; } else { break; }
                            } else { break; }
                        }
                        // Check contraction
                        if pos < len && bytes[pos] == b'\'' {
                            let rem = len - pos;
                            if rem >= 2 {
                                let b1 = bytes[pos + 1];
                                if matches!(b1, b's' | b't' | b'd' | b'm') {
                                    if rem == 2 || !is_ascii_letter(bytes[pos + 2]) {
                                        continue;
                                    }
                                }
                                if rem >= 3 {
                                    let b2 = bytes[pos + 2];
                                    if (b1 == b'l' && b2 == b'l')
                                        || (b1 == b'v' && b2 == b'e')
                                        || (b1 == b'r' && b2 == b'e')
                                    {
                                        continue;
                                    }
                                }
                            }
                        }
                    } else if nc == C_HIGH {
                        let (ch, _) = decode_utf8(&bytes[pos..]);
                        if ch.is_alphabetic() {
                            while pos < len {
                                let b = at(bytes, pos);
                                if is_ascii_letter(b) { pos += 1; }
                                else if b >= 0x80 {
                                    let (ch, cl) = decode_utf8(&bytes[pos..]);
                                    if ch.is_alphabetic() { pos += cl; } else { break; }
                                } else { break; }
                            }
                        }
                    } else if nc == C_DIGIT {
                        pos += 1;
                        while pos < len && is_digit(at(bytes, pos)) { pos += 1; }
                    }
                }
            }
            C_APOS => {
                let rem = len - pos;
                let mut consumed = false;
                if rem >= 2 {
                    let b1 = bytes[pos + 1];
                    if matches!(b1, b's' | b't' | b'd' | b'm') {
                        if rem == 2 || !is_ascii_letter(bytes[pos + 2]) {
                            pos += 2;
                            consumed = true;
                        }
                    }
                    if !consumed && rem >= 3 {
                        let b2 = bytes[pos + 2];
                        if (b1 == b'l' && b2 == b'l')
                            || (b1 == b'v' && b2 == b'e')
                            || (b1 == b'r' && b2 == b'e')
                        {
                            pos += 3;
                            consumed = true;
                        }
                    }
                }
                if !consumed {
                    pos += 1;
                    while pos < len {
                        let b = at(bytes, pos);
                        if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 { break; }
                        pos += 1;
                    }
                }
            }
            C_DIGIT => {
                pos += 1;
                while pos < len && is_digit(at(bytes, pos)) { pos += 1; }
            }
            C_NEWLINE => {
                pos += 1;
                while pos < len {
                    let c = at(bytes, pos);
                    if c == b'\n' || c == b'\r' || c == b' ' { pos += 1; }
                    else { break; }
                }
            }
            C_HIGH => {
                let (ch, cl) = decode_utf8(&bytes[pos..]);
                pos += cl;
                if ch.is_alphabetic() {
                    while pos < len {
                        let b = at(bytes, pos);
                        if is_ascii_letter(b) { pos += 1; }
                        else if b >= 0x80 {
                            let (ch, cl) = decode_utf8(&bytes[pos..]);
                            if ch.is_alphabetic() { pos += cl; } else { break; }
                        } else { break; }
                    }
                } else if ch.is_whitespace() {
                    while pos < len {
                        let c = at(bytes, pos);
                        if c == b' ' || c == b'\n' || c == b'\r' { pos += 1; }
                        else if c >= 0x80 {
                            let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                            if ch2.is_whitespace() { pos += cl2; } else { break; }
                        } else { break; }
                    }
                }
            }
            _ => {
                pos += 1;
                while pos < len {
                    let b = at(bytes, pos);
                    if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 { break; }
                    pos += 1;
                }
            }
        }
    }

    count
}

// ---------------------------------------------------------------------------
// Main: correctness check + benchmark
// ---------------------------------------------------------------------------

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    // Correctness: compare SIMD vs scalar on full input
    {
        let scalar_count = Gpt2::new(&text).count();
        let simd_count = Gpt2SimdIter::new(&text).count();
        println!("Scalar pieces: {scalar_count}");
        println!("SIMD   pieces: {simd_count}");
        if scalar_count != simd_count {
            println!("WARNING: piece count mismatch!");
            // Find first divergence
            let mut scalar = Gpt2::new(&text);
            let mut simd = Gpt2SimdIter::new(&text);
            let mut i = 0;
            loop {
                let s = scalar.next();
                let d = simd.next();
                if s != d {
                    println!("  Divergence at piece {i}: scalar={s:?} simd={d:?}");
                    break;
                }
                if s.is_none() { break; }
                i += 1;
            }
        } else {
            println!("Correctness: OK\n");
        }
    }

    let iters = 20;

    // Benchmark scalar
    {
        let _ = Gpt2::new(&text).count(); // warmup
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = Gpt2::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("Scalar GPT-2:  {throughput:>8.1} MB/s  ({count} pieces)");
    }

    // Benchmark SIMD scan_letters
    {
        let _ = Gpt2SimdIter::new(&text).count();
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = Gpt2SimdIter::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("SIMD scan:     {throughput:>8.1} MB/s  ({count} pieces)");
    }

    // Correctness check for LUT variant
    {
        let lut_count = Gpt2SimdLutIter::new(&text).count();
        let scalar_count = Gpt2::new(&text).count();
        if lut_count != scalar_count {
            println!("\nWARNING: LUT count mismatch: {lut_count} vs {scalar_count}");
            let mut lut = Gpt2SimdLutIter::new(&text);
            let mut scalar = Gpt2::new(&text);
            for i in 0.. {
                let l = lut.next();
                let s = scalar.next();
                if l != s {
                    println!("  Divergence at {i}: lut={l:?} scalar={s:?}");
                    break;
                }
                if l.is_none() { break; }
            }
        }
    }

    // Benchmark SIMD + LUT dispatch
    {
        let _ = Gpt2SimdLutIter::new(&text).count();
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = Gpt2SimdLutIter::new(&text).count();
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("SIMD+LUT:      {throughput:>8.1} MB/s  ({count} pieces)");
    }

    // Correctness check for bulk count
    {
        let bulk_count = count_pieces_bulk(&text);
        let scalar_count = Gpt2::new(&text).count();
        if bulk_count != scalar_count {
            println!("\nWARNING: Bulk count mismatch: {bulk_count} vs {scalar_count}");
        }
    }

    // Benchmark bulk count (no iterator overhead)
    {
        let _ = count_pieces_bulk(&text);
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..iters {
            count = count_pieces_bulk(&text);
        }
        let elapsed = start.elapsed();
        let throughput = mb * iters as f64 / elapsed.as_secs_f64();
        println!("Bulk count:    {throughput:>8.1} MB/s  ({count} pieces)");
    }
}
