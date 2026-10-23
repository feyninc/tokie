//! Test if callback-based pretokenization (no Iterator overhead) is faster.
//!
//! Instead of yielding one piece at a time via next() → Option<&str>,
//! use an inline callback: for_each_piece(text, |piece| { ... }).
//! The compiler can inline the callback, eliminating per-piece function-call overhead.
//!
//! Usage: cargo run -p pretokie --example bench_callback --release

use pretokie::mask::Gpt2;
use pretokie::mask::util::{decode_utf8, is_ascii_letter, is_digit};
use std::time::Instant;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn first_non_letter_16(ptr: *const u8) -> usize {
    let chunk = vld1q_u8(ptr);
    let lowered = vorrq_u8(chunk, vdupq_n_u8(0x20));
    let offset = vsubq_u8(lowered, vdupq_n_u8(b'a'));
    let is_letter = vcltq_u8(offset, vdupq_n_u8(26));
    if vminvq_u8(is_letter) == 0xFF { return 16; }
    let not_letter = vmvnq_u8(is_letter);
    static POWERS: [u8; 16] = [1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128];
    let powers = vld1q_u8(POWERS.as_ptr());
    let bits = vandq_u8(not_letter, powers);
    let lo = vaddv_u8(vget_low_u8(bits)) as u16;
    let hi = vaddv_u8(vget_high_u8(bits)) as u16;
    let bitmask = lo | (hi << 8);
    bitmask.trailing_zeros() as usize
}

// ---------------------------------------------------------------------------
// Callback-based pretokenizer
// ---------------------------------------------------------------------------

#[inline(always)]
fn scan_letters(bytes: &[u8], mut pos: usize, len: usize) -> usize {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        while pos + 16 <= len {
            let n = first_non_letter_16(bytes.as_ptr().add(pos));
            pos += n;
            if n < 16 {
                if pos < len && *bytes.get_unchecked(pos) >= 0x80 {
                    let (ch, cl) = decode_utf8(&bytes[pos..]);
                    if ch.is_alphabetic() { pos += cl; continue; }
                }
                return pos;
            }
        }
    }
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_ascii_letter(b) { pos += 1; }
        else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&bytes[pos..]);
            if ch.is_alphabetic() { pos += cl; } else { return pos; }
        } else { return pos; }
    }
    pos
}

#[inline(always)]
fn scan_digits(bytes: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len && is_digit(unsafe { *bytes.get_unchecked(pos) }) {
        pos += 1;
    }
    pos
}

#[inline(always)]
fn check_contraction(bytes: &[u8], pos: usize, len: usize) -> usize {
    if pos >= len || bytes[pos] != b'\'' { return 0; }
    let rem = len - pos;
    if rem < 2 { return 0; }
    let b1 = bytes[pos + 1];
    if matches!(b1, b's' | b't' | b'd' | b'm') {
        if rem == 2 || !is_ascii_letter(bytes[pos + 2]) { return 2; }
    }
    if rem < 3 { return 0; }
    let b2 = bytes[pos + 2];
    if (b1 == b'l' && b2 == b'l') || (b1 == b'v' && b2 == b'e') || (b1 == b'r' && b2 == b'e') {
        return 3;
    }
    0
}

#[inline(always)]
fn scan_punct(bytes: &[u8], mut pos: usize, len: usize) -> usize {
    while pos < len {
        let b = unsafe { *bytes.get_unchecked(pos) };
        if is_ascii_letter(b) || is_digit(b) || b == b' ' || b == b'\n' || b == b'\r' || b >= 0x80 {
            break;
        }
        pos += 1;
    }
    pos
}

/// Process all pieces via inline callback. No Iterator, no Option, no per-piece dispatch overhead.
#[inline(always)]
fn for_each_piece<F: FnMut(usize, usize)>(text: &str, mut f: F) {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut pos = 0;

    while pos < len {
        let start = pos;
        let b = unsafe { *bytes.get_unchecked(pos) };

        if is_ascii_letter(b) {
            pos = scan_letters(bytes, pos + 1, len);
            if check_contraction(bytes, pos, len) > 0 {
                f(start, pos);
                continue;
            }
        } else if b == b' ' {
            pos += 1;
            if pos < len {
                let next = unsafe { *bytes.get_unchecked(pos) };
                if is_ascii_letter(next) {
                    pos = scan_letters(bytes, pos + 1, len);
                    if check_contraction(bytes, pos, len) > 0 {
                        f(start, pos);
                        continue;
                    }
                } else if next >= 0x80 {
                    let (ch, _) = decode_utf8(&bytes[pos..]);
                    if ch.is_alphabetic() {
                        pos = scan_letters(bytes, pos, len);
                    }
                } else if is_digit(next) {
                    pos = scan_digits(bytes, pos + 1, len);
                }
            }
        } else if b == b'\'' {
            let clen = check_contraction(bytes, pos, len);
            if clen > 0 {
                pos += clen;
            } else {
                pos = scan_punct(bytes, pos + 1, len);
            }
        } else if is_digit(b) {
            pos = scan_digits(bytes, pos + 1, len);
        } else if b == b'\n' || b == b'\r' {
            pos += 1;
            while pos < len {
                let c = unsafe { *bytes.get_unchecked(pos) };
                if c == b'\n' || c == b'\r' || c == b' ' { pos += 1; }
                else { break; }
            }
        } else if b >= 0x80 {
            let (ch, cl) = decode_utf8(&bytes[pos..]);
            pos += cl;
            if ch.is_alphabetic() {
                pos = scan_letters(bytes, pos, len);
            } else if ch.is_whitespace() {
                while pos < len {
                    let c = unsafe { *bytes.get_unchecked(pos) };
                    if c == b' ' || c == b'\n' || c == b'\r' { pos += 1; }
                    else if c >= 0x80 {
                        let (ch2, cl2) = decode_utf8(&bytes[pos..]);
                        if ch2.is_whitespace() { pos += cl2; } else { break; }
                    } else { break; }
                }
            }
        } else {
            pos = scan_punct(bytes, pos + 1, len);
        }

        f(start, pos);
    }
}

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    // Correctness: count with callback vs iterator
    let ref_count = Gpt2::new(&text).count();
    let mut cb_count = 0usize;
    for_each_piece(&text, |_, _| { cb_count += 1; });
    println!("Iterator:  {ref_count} pieces");
    println!("Callback:  {cb_count} pieces");
    if ref_count != cb_count {
        println!("MISMATCH! Finding divergence...");
        let mut iter = Gpt2::new(&text);
        let mut pieces = Vec::new();
        for_each_piece(&text, |s, e| { pieces.push((s, e)); });
        for (i, &(s, e)) in pieces.iter().enumerate() {
            let cb_piece = &text[s..e];
            let it_piece = iter.next();
            if it_piece != Some(cb_piece) {
                println!("  at {i}: iter={it_piece:?} cb={cb_piece:?}");
                break;
            }
        }
        return;
    }
    println!("Correctness: OK\n");

    let iters = 20;

    // Benchmark: iterator count
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters { c = Gpt2::new(&text).count(); }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Iterator:    {t:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark: callback count
    {
        let mut c = 0usize;
        for_each_piece(&text, |_, _| { c += 1; });
        c = 0;
        let start = Instant::now();
        for _ in 0..iters {
            c = 0;
            for_each_piece(&text, |_, _| { c += 1; });
        }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("Callback:    {t:>8.1} MB/s  ({c} pieces)");
    }

    // Benchmark: callback producing Vec<(u32,u32)> of boundaries
    {
        let mut bounds: Vec<(u32, u32)> = Vec::with_capacity(25_000_000);
        for_each_piece(&text, |s, e| { bounds.push((s as u32, e as u32)); });
        bounds.clear();
        let start = Instant::now();
        for _ in 0..iters {
            bounds.clear();
            for_each_piece(&text, |s, e| { bounds.push((s as u32, e as u32)); });
        }
        let t = mb * iters as f64 / start.elapsed().as_secs_f64();
        println!("CB+Vec:      {t:>8.1} MB/s  ({} pieces)", bounds.len());
    }
}
