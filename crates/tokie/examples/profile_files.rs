//! Stage breakdown for encode_files_flat on a big corpus file.
//!
//! Run: cargo run --release -p tokie --features build --example profile_files -- <file> [sep]

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).expect("usage: profile_files <file> [sep]").clone();
    let sep = args.get(2).map(String::as_str).unwrap_or("<|endoftext|>").as_bytes().to_vec();

    let tok = match std::env::var("TOKIE_JSON") {
        Ok(p) => tokie::Tokenizer::from_json(&p).unwrap(),
        Err(_) => tokie::Tokenizer::from_pretrained("openai-community/gpt2").unwrap(),
    };

    // Stage 1: read
    let t0 = Instant::now();
    let buf = std::fs::read(&path).unwrap();
    let t_read = t0.elapsed().as_secs_f64();
    let nbytes = buf.len();
    println!("read           : {:6.1} ms  ({:.0} MB/s)", t_read * 1e3, nbytes as f64 / 1e6 / t_read);

    // Stage 2: memmem split into ranges (no validation)
    let t0 = Instant::now();
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for pos in memchr::memmem::find_iter(&buf, &sep[..]) {
        if pos > start { ranges.push((start, pos)); }
        start = pos + sep.len();
    }
    if start < buf.len() { ranges.push((start, buf.len())); }
    let t_split = t0.elapsed().as_secs_f64();
    println!("memmem split   : {:6.1} ms  ({} docs)", t_split * 1e3, ranges.len());

    // Stage 3: per-doc lossy validation (Cow creation)
    let t0 = Instant::now();
    let docs: Vec<std::borrow::Cow<str>> = ranges
        .iter()
        .map(|&(s, e)| String::from_utf8_lossy(&buf[s..e]))
        .collect();
    let t_val = t0.elapsed().as_secs_f64();
    let owned = docs.iter().filter(|c| matches!(c, std::borrow::Cow::Owned(_))).count();
    println!("utf8 validate  : {:6.1} ms  ({} owned/lossy docs)", t_val * 1e3, owned);

    let t0 = Instant::now();
    let refs: Vec<&str> = docs.iter().map(|d| d.as_ref()).collect();
    println!("refs collect   : {:6.1} ms", t0.elapsed().as_secs_f64() * 1e3);

    // Stage 4: encode_batch_flat
    let t0 = Instant::now();
    let (ids, lens) = tok.encode_batch_flat(&refs, false);
    let t_enc = t0.elapsed().as_secs_f64();
    println!("encode_flat MT : {:6.1} ms  ({:.0} MB/s, {} tokens)", t_enc * 1e3, nbytes as f64 / 1e6 / t_enc, ids.len());
    drop(lens);
    drop(ids);

    // Comparison: count-only batch (no id materialization)
    let t0 = Instant::now();
    let n: usize = tok.count_tokens_batch(&refs).iter().sum();
    let t_cnt = t0.elapsed().as_secs_f64();
    println!("count batch MT : {:6.1} ms  ({:.0} MB/s, {} tokens)", t_cnt * 1e3, nbytes as f64 / 1e6 / t_cnt, n);

    drop(docs);
    drop(buf);

    // End-to-end public API
    let t0 = Instant::now();
    let (ids, offs) = tok.encode_files_flat(&[&path], &sep, false).unwrap();
    let t_all = t0.elapsed().as_secs_f64();
    println!("encode_files   : {:6.1} ms  ({:.0} MB/s, {} tokens, {} docs)", t_all * 1e3, nbytes as f64 / 1e6 / t_all, ids.len(), offs.len() - 1);
}
