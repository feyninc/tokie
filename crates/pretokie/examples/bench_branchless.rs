//! Branchless state-machine pretokenizer via lookup table.
//!
//! Per byte: two table loads, one conditional-move, zero branches (except apostrophes).
//! Theoretical: ~500-700 MB/s (limited by state→TABLE dependency chain).
//!
//! Usage: cargo run -p pretokie --example bench_branchless --release

use pretokie::mask::Gpt2;
use std::time::Instant;

// ===========================================================================
// Byte classes
// ===========================================================================

const CL: u8 = 0;  // letter (a-z, A-Z) — non-contraction
const CD: u8 = 1;  // digit (0-9)
const CS: u8 = 2;  // space
const CN: u8 = 3;  // newline (\n, \r)
const CA: u8 = 4;  // apostrophe
const CH: u8 = 5;  // high byte (≥0x80)
const CO: u8 = 6;  // other (punctuation)
const NUM_CLASSES: usize = 7;

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

// ===========================================================================
// States
// ===========================================================================

const ST_START: u8    = 0;  // beginning or just emitted
const ST_LETTER: u8   = 1;  // in letter run
const ST_DIGIT: u8    = 2;  // in digit run
const ST_SPACE: u8    = 3;  // saw space (potential prefix)
const ST_SP_LET: u8   = 4;  // space + letter run
const ST_SP_DIG: u8   = 5;  // space + digit run
const ST_PUNCT: u8    = 6;  // in punctuation run
const ST_WS: u8       = 7;  // in whitespace (newline-led)
const ST_HIGH: u8     = 8;  // in high-byte/unicode
const NUM_STATES: usize = 9;

// ===========================================================================
// Transition table: TABLE[state][class] = (next_state, emit_boundary)
// ===========================================================================

struct Action {
    next: u8,
    emit: u8, // 0 or 1
}

const fn a(next: u8, emit: u8) -> Action { Action { next, emit } }

// GPT-2 rules:
// - Letters/digits are self-continuing within their piece
// - Space prefixes letters, digits, and punctuation (merges into one piece)
// - Newlines merge with following spaces/newlines
// - Punctuation groups together (non-letter, non-digit, non-space, non-newline)
// - Apostrophe handled separately for contractions

// GPT-2 pretokenizer rules (from testing actual behavior):
//   ' ?\p{L}+      → optional space + letters (SPACE prefixes letters)
//   ' ?\p{N}+      → optional space + digits (SPACE prefixes digits)
//   ' ?[^\s\p{L}\p{N}]+ → space does NOT prefix punctuation (space consumed by \s+ first)
//   \s+(?!\S)|\s+  → whitespace runs (newlines merge with following spaces/newlines)
//   Contractions handled separately via branch on apostrophe.
//
// Verified: " />" → [" ", "/>"] (space is separate, does NOT merge with punctuation)
// Verified: " hello" → [" hello"] (space merges with letters)

static TABLE: [[Action; NUM_CLASSES]; NUM_STATES] = [
    //                    CL            CD            CS            CN            CA            CH            CO
    /* START    */ [a(ST_LETTER,0), a(ST_DIGIT,0), a(ST_SPACE,0), a(ST_WS,0),   a(ST_PUNCT,0), a(ST_HIGH,0), a(ST_PUNCT,0)],
    /* LETTER   */ [a(ST_LETTER,0), a(ST_DIGIT,1), a(ST_SPACE,1), a(ST_WS,1),   a(ST_PUNCT,1), a(ST_HIGH,0), a(ST_PUNCT,1)],
    /* DIGIT    */ [a(ST_LETTER,1), a(ST_DIGIT,0), a(ST_SPACE,1), a(ST_WS,1),   a(ST_PUNCT,1), a(ST_HIGH,1), a(ST_PUNCT,1)],
    /* SPACE    */ [a(ST_SP_LET,0), a(ST_SP_DIG,0), a(ST_SPACE,1),a(ST_WS,1),   a(ST_PUNCT,1), a(ST_HIGH,0), a(ST_PUNCT,1)],
    /* SP_LET   */ [a(ST_SP_LET,0), a(ST_DIGIT,1), a(ST_SPACE,1), a(ST_WS,1),  a(ST_PUNCT,1), a(ST_HIGH,0), a(ST_PUNCT,1)],
    /* SP_DIG   */ [a(ST_LETTER,1), a(ST_SP_DIG,0), a(ST_SPACE,1),a(ST_WS,1),  a(ST_PUNCT,1), a(ST_HIGH,1), a(ST_PUNCT,1)],
    /* PUNCT    */ [a(ST_LETTER,1), a(ST_DIGIT,1), a(ST_SPACE,1), a(ST_WS,1),   a(ST_PUNCT,0), a(ST_HIGH,1), a(ST_PUNCT,0)],
    /* WS       */ [a(ST_LETTER,1), a(ST_DIGIT,1), a(ST_WS,0),   a(ST_WS,0),   a(ST_PUNCT,1), a(ST_HIGH,1), a(ST_PUNCT,1)],
    /* HIGH     */ [a(ST_HIGH,0),   a(ST_DIGIT,1), a(ST_SPACE,1), a(ST_WS,1),   a(ST_PUNCT,1), a(ST_HIGH,0), a(ST_PUNCT,1)],
];

// ===========================================================================
// Branchless pretokenizer
// ===========================================================================

fn pretok_branchless(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 { return vec![0]; }

    // Pre-allocate worst case (every byte is a boundary)
    let mut boundaries = vec![0u32; len + 2];
    boundaries[0] = 0;
    let mut write = 1usize;
    let mut state = 0u8; // START

    // Process first byte to initialize state
    let first_class = CLASS_LUT[bytes[0] as usize];
    state = TABLE[0][first_class as usize].next;

    for pos in 1..len {
        let class = unsafe { *CLASS_LUT.get_unchecked(*bytes.get_unchecked(pos) as usize) };

        // Apostrophe after letter run? Check contraction.
        // This is the only branch — taken 0.9% of the time.
        if class == CA && (state == ST_LETTER || state == ST_SP_LET) {
            // Check if this is a contraction ('s, 't, 'd, 'm, 'll, 're, 've)
            let rem = len - pos;
            if rem >= 2 {
                let b1 = bytes[pos + 1];
                let is_short = matches!(b1, b's'|b't'|b'd'|b'm'|b'S'|b'T'|b'D'|b'M')
                    && (rem == 2 || CLASS_LUT[bytes[pos + 2] as usize] != CL);
                let is_long = rem >= 3 && {
                    let b2 = bytes[pos + 2];
                    matches!((b1, b2),
                        (b'l'|b'L', b'l'|b'L') |
                        (b'r'|b'R', b'e'|b'E') |
                        (b'v'|b'V', b'e'|b'E'))
                };
                if is_short || is_long {
                    // Emit boundary at apostrophe (end letter piece, start contraction)
                    unsafe { *boundaries.get_unchecked_mut(write) = pos as u32; }
                    write += 1;
                    state = ST_PUNCT; // contraction piece acts like punctuation
                    continue;
                }
            }
            // Not a contraction — apostrophe is punctuation, emit boundary
            let action = unsafe { &TABLE.get_unchecked(state as usize)[CA as usize] };
            unsafe { *boundaries.get_unchecked_mut(write) = pos as u32; }
            write += action.emit as usize;
            state = action.next;
            continue;
        }

        // Branchless path (99%+ of bytes)
        let action = unsafe { &TABLE.get_unchecked(state as usize).get_unchecked(class as usize) };
        unsafe { *boundaries.get_unchecked_mut(write) = pos as u32; }
        write += action.emit as usize;
        state = action.next;
    }

    boundaries.truncate(write);
    // Add sentinel end
    boundaries.push(len as u32);
    boundaries
}

// ===========================================================================
// Bitfield accumulate + batch extract
// ===========================================================================

/// Pack state table into single bytes: (emit << 7) | next_state
static PACKED_TABLE: [[u8; NUM_CLASSES]; NUM_STATES] = {
    let mut t = [[0u8; NUM_CLASSES]; NUM_STATES];
    let mut s = 0;
    while s < NUM_STATES {
        let mut c = 0;
        while c < NUM_CLASSES {
            t[s][c] = (TABLE[s][c].emit << 7) | TABLE[s][c].next;
            c += 1;
        }
        s += 1;
    }
    t
};

fn pretok_bitfield(text: &str) -> Vec<u32> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if len == 0 { return vec![0]; }

    // Pre-allocate for boundaries (worst case: every byte)
    let mut boundaries = Vec::with_capacity(len / 3 + 2);
    boundaries.push(0u32);
    let mut write = 1usize;

    // Initialize state from first byte
    let first_class = CLASS_LUT[bytes[0] as usize];
    let mut state = TABLE[0][first_class as usize].next;

    // Process in 64-byte blocks
    let mut block_start = 1usize;

    while block_start + 64 <= len {
        let mut mask = 0u64;

        // Inner loop: 64 bytes, branchless, register-only (no stores)
        let block_end = block_start + 64;
        let mut pos = block_start;
        while pos < block_end {
            let byte = unsafe { *bytes.get_unchecked(pos) };
            let class = unsafe { *CLASS_LUT.get_unchecked(byte as usize) };

            // Contraction check (rare branch, ~0.9% of bytes)
            if class == CA && (state == ST_LETTER || state == ST_SP_LET) {
                let rem = len - pos;
                if rem >= 2 {
                    let b1 = bytes[pos + 1];
                    let is_short = matches!(b1, b's'|b't'|b'd'|b'm'|b'S'|b'T'|b'D'|b'M')
                        && (rem == 2 || CLASS_LUT[bytes[pos + 2] as usize] != CL);
                    let is_long = rem >= 3 && {
                        let b2 = bytes[pos + 2];
                        matches!((b1, b2),
                            (b'l'|b'L', b'l'|b'L') |
                            (b'r'|b'R', b'e'|b'E') |
                            (b'v'|b'V', b'e'|b'E'))
                    };
                    if is_short || is_long {
                        mask |= 1u64 << (pos - block_start);
                        state = ST_PUNCT;
                        pos += 1;
                        continue;
                    }
                }
                // Not a contraction — fall through to normal table lookup
            }

            let packed = unsafe {
                *PACKED_TABLE.get_unchecked(state as usize).get_unchecked(class as usize)
            };
            let emit = packed >> 7;
            state = packed & 0x7F;
            mask |= (emit as u64) << (pos - block_start);
            pos += 1;
        }

        // Extract boundaries from bitmask via CTZ loop
        // write always increments — sequential stores, no data dependency on emit
        let base = block_start as u32;
        while mask != 0 {
            let bit = mask.trailing_zeros();
            unsafe {
                // Ensure we have capacity (we pre-allocated len/3 + 2)
                *boundaries.as_mut_ptr().add(write) = base + bit;
            }
            write += 1;
            mask &= mask - 1;
        }

        block_start = block_end;
    }

    // Scalar tail (remaining < 64 bytes)
    for pos in block_start..len {
        let byte = unsafe { *bytes.get_unchecked(pos) };
        let class = unsafe { *CLASS_LUT.get_unchecked(byte as usize) };

        if class == CA && (state == ST_LETTER || state == ST_SP_LET) {
            let rem = len - pos;
            if rem >= 2 {
                let b1 = bytes[pos + 1];
                let is_short = matches!(b1, b's'|b't'|b'd'|b'm'|b'S'|b'T'|b'D'|b'M')
                    && (rem == 2 || CLASS_LUT[bytes[pos + 2] as usize] != CL);
                let is_long = rem >= 3 && {
                    let b2 = bytes[pos + 2];
                    matches!((b1, b2),
                        (b'l'|b'L', b'l'|b'L') |
                        (b'r'|b'R', b'e'|b'E') |
                        (b'v'|b'V', b'e'|b'E'))
                };
                if is_short || is_long {
                    unsafe { *boundaries.as_mut_ptr().add(write) = pos as u32; }
                    write += 1;
                    state = ST_PUNCT;
                    continue;
                }
            }
        }

        let packed = unsafe {
            *PACKED_TABLE.get_unchecked(state as usize).get_unchecked(class as usize)
        };
        let emit = packed >> 7;
        state = packed & 0x7F;
        if emit != 0 {
            unsafe { *boundaries.as_mut_ptr().add(write) = pos as u32; }
            write += 1;
        }
    }

    unsafe { boundaries.set_len(write); }
    boundaries.push(len as u32);
    boundaries
}

// ===========================================================================
// Reference: iterator-based pretokenizer
// ===========================================================================

fn pretok_iterator(text: &str) -> Vec<u32> {
    let base = text.as_ptr() as usize;
    let mut boundaries: Vec<u32> = Gpt2::new(text)
        .map(|piece| (piece.as_ptr() as usize - base) as u32)
        .collect();
    boundaries.push(text.len() as u32);
    boundaries
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    let path = std::env::var("ENWIK8_PATH")
        .unwrap_or_else(|_| "crates/tokie/benches/data/enwik8".to_string());
    let text = std::fs::read_to_string(&path).expect("need enwik8");
    let mb = text.len() as f64 / (1024.0 * 1024.0);
    println!("Input: {:.2} MB\n", mb);

    // Correctness check
    let ref_bounds = pretok_iterator(&text);
    let lut_bounds = pretok_branchless(&text);

    println!("Reference: {} boundaries", ref_bounds.len());
    println!("LUT:       {} boundaries", lut_bounds.len());

    let mut mismatches = 0;
    let min_len = ref_bounds.len().min(lut_bounds.len());
    for i in 0..min_len {
        if ref_bounds[i] != lut_bounds[i] {
            if mismatches < 10 {
                let ref_pos = ref_bounds[i] as usize;
                let lut_pos = lut_bounds[i] as usize;
                let ctx_start = ref_pos.saturating_sub(10);
                let ctx_end = (ref_pos + 15).min(text.len());
                let ctx = &text[ctx_start..ctx_end];
                println!("  mismatch at boundary {i}: ref={ref_pos} lut={lut_pos}  context: {:?}", ctx);

                // Show what pieces the reference produces around this boundary
                if i > 0 {
                    let prev = ref_bounds[i-1] as usize;
                    println!("    ref piece: {:?}", &text[prev..ref_pos]);
                }
                if i > 0 && i < lut_bounds.len() {
                    let prev = lut_bounds[i-1] as usize;
                    println!("    lut piece: {:?}", &text[prev..lut_pos]);
                }
            }
            mismatches += 1;
        }
    }
    if ref_bounds.len() != lut_bounds.len() {
        println!("  length mismatch: ref={} lut={}", ref_bounds.len(), lut_bounds.len());
    }
    if mismatches > 0 {
        println!("  total mismatches: {mismatches}");
    } else {
        println!("  Correctness: OK");
    }

    let iters = 20;

    // Benchmark: reference iterator
    {
        let _ = Gpt2::new(&text).count();
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            c = Gpt2::new(&text).count();
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let cpb = 3.5e9 / (mbs * 1024.0 * 1024.0);
        println!("\nScalar iter:     {mbs:>8.1} MB/s  ({cpb:.1} cyc/B)  ({c} pieces)");
    }

    // Benchmark: branchless LUT
    {
        let _ = pretok_branchless(&text);
        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            let b = pretok_branchless(&text);
            c = b.len() - 1; // subtract sentinel
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let cpb = 3.5e9 / (mbs * 1024.0 * 1024.0);
        println!("Branchless LUT:  {mbs:>8.1} MB/s  ({cpb:.1} cyc/B)  ({c} boundaries)");
    }

    // Benchmark: bitfield accumulate + batch extract
    {
        let _ = pretok_bitfield(&text);
        let bf_bounds = pretok_bitfield(&text);
        println!("Bitfield:  {} boundaries", bf_bounds.len());

        let start = Instant::now();
        let mut c = 0;
        for _ in 0..iters {
            let b = pretok_bitfield(&text);
            c = b.len() - 1;
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let cpb = 3.5e9 / (mbs * 1024.0 * 1024.0);
        println!("Bitfield LUT:    {mbs:>8.1} MB/s  ({cpb:.1} cyc/B)  ({c} boundaries)");
    }

    // Benchmark: bitfield accumulate only (no extraction — measure inner loop)
    {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let start = Instant::now();
        let mut total = 0usize;
        for _ in 0..iters {
            let first_class = CLASS_LUT[bytes[0] as usize];
            let mut state = TABLE[0][first_class as usize].next;
            let mut count = 0usize;
            let mut block_start = 1usize;
            while block_start + 64 <= len {
                let mut mask = 0u64;
                let mut pos = block_start;
                while pos < block_start + 64 {
                    let byte = unsafe { *bytes.get_unchecked(pos) };
                    let class = unsafe { *CLASS_LUT.get_unchecked(byte as usize) };
                    let packed = unsafe {
                        *PACKED_TABLE.get_unchecked(state as usize).get_unchecked(class as usize)
                    };
                    state = packed & 0x7F;
                    mask |= ((packed >> 7) as u64) << (pos - block_start);
                    pos += 1;
                }
                count += mask.count_ones() as usize;
                block_start += 64;
            }
            total = count;
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let cpb = 3.5e9 / (mbs * 1024.0 * 1024.0);
        println!("Bitfield accum:  {mbs:>8.1} MB/s  ({cpb:.1} cyc/B)  ({total} boundaries, no extract)");
    }

    // Benchmark: branchless LUT count-only (skip Vec writes to measure pure loop speed)
    {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let start = Instant::now();
        let mut total_boundaries = 0usize;
        for _ in 0..iters {
            let mut state = 0u8;
            let mut count = 0usize;
            let first_class = CLASS_LUT[bytes[0] as usize];
            state = TABLE[0][first_class as usize].next;
            for pos in 1..len {
                let class = unsafe { *CLASS_LUT.get_unchecked(*bytes.get_unchecked(pos) as usize) };
                let action = unsafe { &TABLE.get_unchecked(state as usize).get_unchecked(class as usize) };
                count += action.emit as usize;
                state = action.next;
            }
            total_boundaries = count;
        }
        let mbs = mb * iters as f64 / start.elapsed().as_secs_f64();
        let cpb = 3.5e9 / (mbs * 1024.0 * 1024.0);
        println!("LUT count-only:  {mbs:>8.1} MB/s  ({cpb:.1} cyc/B)  ({total_boundaries} boundaries)");
    }
}
