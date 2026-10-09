//! Dev-only dump tool: pretokens from either pretokie or the HF oracle,
//! chunked format for cross-checking against data files and Python.
//!
//! Format: per chunk — u32 LE pretoken count, then per pretoken u32 LE
//! length + bytes. Chunking (1MB, UTF-8-boundary backed off) matches the
//! external harness and `tests/golden.rs`.
//!
//! Usage:
//!   cargo run -p pretokie --features oracle --example dump -- \
//!     --scheme r50k --impl oracle --file benches/data/enwik8 [--chunk-bytes N] [--out PATH]

use pretokie::oracle::{self, Scheme};
use std::io::{BufWriter, Write};

const DEFAULT_CHUNK: usize = 1 << 20;

fn scheme_by_name(name: &str) -> &'static Scheme {
    oracle::ALL_SCHEMES
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("unknown scheme {name:?}; expected one of {:?}", oracle::ALL_SCHEMES.iter().map(|s| s.name).collect::<Vec<_>>()))
}

fn pretokie_pretokens(scheme: &Scheme, text: &str) -> Vec<Vec<u8>> {
    match scheme.name {
        "r50k" => pretokie::Gpt2::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "cl100k" => pretokie::Cl100k::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "o200k" => pretokie::O200k::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "qwen2" => pretokie::Qwen::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "deepseek" => pretokie::DeepSeek::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "smollm" => pretokie::SmolLM::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        "bert" => pretokie::Bert::new(text).map(str::as_bytes).map(<[u8]>::to_vec).collect(),
        // Falcon-style models have no hand-written pretokie scheme.
        "falcon" => Vec::new(),
        other => panic!("no pretokie implementation for scheme {other:?}"),
    }
}

fn main() {
    let mut scheme = None;
    let mut impl_opt: Option<String> = None;
    let mut file = None;
    let mut text = None;
    let mut chunk_bytes = DEFAULT_CHUNK;
    let mut out = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let val = |i: &mut usize, flag: &str| {
            *i += 1;
            args.get(*i).unwrap_or_else(|| panic!("{flag} needs a value")).clone()
        };
        match args[i].as_str() {
            "--scheme" => scheme = Some(val(&mut i, "--scheme")),
            "--impl" => impl_opt = Some(val(&mut i, "--impl")),
            "--file" => file = Some(val(&mut i, "--file")),
            "--text" => text = Some(val(&mut i, "--text")),
            "--chunk-bytes" => chunk_bytes = val(&mut i, "--chunk-bytes").parse().unwrap(),
            "--out" => out = Some(val(&mut i, "--out")),
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }
    let impl_ = impl_opt.as_deref().unwrap_or("oracle");
    let scheme = scheme_by_name(scheme.expect("--scheme is required").as_str());
    let data: Vec<u8> = if let Some(text) = text {
        text.into_bytes()
    } else {
        std::fs::read(file.expect("--file or --text is required")).expect("cannot read file")
    };

    let writer: Box<dyn Write> = match out {
        Some(path) => Box::new(BufWriter::new(std::fs::File::create(path).unwrap())),
        None => Box::new(std::io::stdout().lock()),
    };
    let mut w = BufWriter::new(writer);

    let mut start = 0usize;
    while start < data.len() {
        let mut end = (start + chunk_bytes).min(data.len());
        while end > start && std::str::from_utf8(&data[start..end]).is_err() {
            end -= 1;
        }
        let text = std::str::from_utf8(&data[start..end]).unwrap();
        let pretokens = match impl_ {
            "oracle" => oracle::pretokens(scheme, text),
            "pretokie" => pretokie_pretokens(scheme, text),
            other => panic!("unknown impl {other:?}"),
        };
        w.write_all(&(pretokens.len() as u32).to_le_bytes()).unwrap();
        let mut buf = Vec::new();
        for pt in &pretokens {
            buf.extend_from_slice(&(pt.len() as u32).to_le_bytes());
            buf.extend_from_slice(pt);
        }
        w.write_all(&buf).unwrap();
        start = end;
    }
    w.flush().unwrap();
}