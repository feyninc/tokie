//! Map each byte to its GPT-2 unicode character. Printable ASCII stays itself.

use crate::formula::Formula;

const ENC: [[u8; 3]; 256] = table().0;
const LEN: [u8; 256] = table().1;

pub struct ByteLevel;

impl ByteLevel {
    pub fn write(text: &str, at_start: bool, out: &mut String) -> bool {
        <Self as Formula>::drive(&Self, text, at_start, out)
    }
}

impl Formula for ByteLevel {
    fn drive(&self, text: &str, _at_start: bool, out: &mut String) -> bool {
        let bytes = text.as_bytes();
        if bytes.is_empty() || is_identity(bytes) {
            return false;
        }
        // Every non-identity byte expands to two UTF-8 bytes. Identity bytes stay one.
        out.reserve(bytes.len().saturating_mul(2));
        // SAFETY: `reserve` covers two bytes per input byte, and each write stays in that range.
        // Every table entry is the UTF-8 of one scalar.
        unsafe {
            let v = out.as_mut_vec();
            let dst = v.as_mut_ptr();
            let mut n = 0;
            let mut i = 0;
            while i < bytes.len() {
                let b = bytes[i];
                if b == b' ' {
                    *dst.add(n) = 0xC4;
                    *dst.add(n + 1) = 0xA0;
                    n += 2;
                    i += 1;
                    continue;
                }
                if (0x21..0x7F).contains(&b) {
                    *dst.add(n) = b;
                    n += 1;
                    i += 1;
                    continue;
                }
                let len = LEN[b as usize] as usize;
                std::ptr::copy_nonoverlapping(ENC[b as usize].as_ptr(), dst.add(n), len);
                n += len;
                i += 1;
            }
            v.set_len(n);
        }
        true
    }

    fn unchanged(&self, text: &str, _at_start: bool) -> Option<usize> {
        let _ = text;
        None
    }

    fn write_scalar(&self, text: &str, _at_start: bool, _out: &mut String) -> usize {
        text.len().max(1)
    }
}

fn is_identity(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|&b| (0x21..0x7F).contains(&b) || (LEN[b as usize] == 1 && ENC[b as usize][0] == b))
}

const fn encode(cp: u32) -> ([u8; 3], u8) {
    if cp < 0x80 {
        ([cp as u8, 0, 0], 1)
    } else if cp < 0x800 {
        (
            [0xC0 | (cp >> 6) as u8, 0x80 | (cp & 0x3F) as u8, 0],
            2,
        )
    } else {
        (
            [
                0xE0 | (cp >> 12) as u8,
                0x80 | ((cp >> 6) & 0x3F) as u8,
                0x80 | (cp & 0x3F) as u8,
            ],
            3,
        )
    }
}

const fn table() -> ([[u8; 3]; 256], [u8; 256]) {
    let mut enc = [[0u8; 3]; 256];
    let mut lens = [0u8; 256];
    let mut bs = [0u16; 256];
    let mut cs = [0u32; 256];
    let mut n = 0usize;
    let mut b = b'!' as u16;
    while b <= b'~' as u16 {
        bs[n] = b;
        cs[n] = b as u32;
        n += 1;
        b += 1;
    }
    b = 0xA1;
    while b <= 0xAC {
        bs[n] = b;
        cs[n] = b as u32;
        n += 1;
        b += 1;
    }
    b = 0xAE;
    while b <= 0xFF {
        bs[n] = b;
        cs[n] = b as u32;
        n += 1;
        b += 1;
    }
    let mut seen = [false; 256];
    let mut i = 0;
    while i < n {
        seen[bs[i] as usize] = true;
        i += 1;
    }
    let mut extra = 0u32;
    b = 0;
    while b < 256 {
        if !seen[b as usize] {
            bs[n] = b;
            cs[n] = 256 + extra;
            extra += 1;
            n += 1;
        }
        b += 1;
    }
    i = 0;
    while i < 256 {
        let (bytes, len) = encode(cs[i]);
        enc[bs[i] as usize] = bytes;
        lens[bs[i] as usize] = len;
        i += 1;
    }
    (enc, lens)
}
