//! Throughput of one row on one file.
//!
//! One warmup, then the median of `--reps`. The whole file is one input, then
//! every newline-separated line is its own input. Rates are input megabytes
//! per second. A library that cannot load the row is blank.
//!
//! Tokie times encode when `tokenizer.json` is present, and normalize-then-split
//! when it is not. Those two numbers are not the same measurement.

use tokie_bench::hf;
use tokie_bench::rows::ROWS;
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_FILE: &str = "/home/bhavnick/Workspaces/pretok-vs-giga/data/owt_sample.txt";

fn main() {
    let mut file = None;
    let mut reps = 5usize;
    let mut max_bytes = usize::MAX;
    let mut only = None;
    let mut pieces = false;
    let mut skip_hf = false;
    let mut load = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => file = Some(args.next().expect("--file needs a value")),
            "--reps" => {
                reps = args
                    .next()
                    .expect("--reps needs a value")
                    .parse()
                    .expect("--reps must be an integer");
            }
            "--bytes" => {
                max_bytes = args
                    .next()
                    .expect("--bytes needs a value")
                    .parse()
                    .expect("--bytes must be an integer");
            }
            "--row" => only = Some(args.next().expect("--row needs a value")),
            "--pieces" => pieces = true,
            "--no-hf" => skip_hf = true,
            "--load" => load = true,
            other => panic!("unknown argument: {other}"),
        }
    }
    if load {
        time_loads(reps, only.as_deref(), skip_hf);
        return;
    }
    let path = file.unwrap_or_else(|| DEFAULT_FILE.to_string());
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
    let valid = bytes.len().min(max_bytes);
    let mut end = valid;
    while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() {
        end -= 1;
    }
    let text = std::str::from_utf8(&bytes[..end]).expect("sample is utf-8");
    let lines: Vec<&str> = text.split('\n').collect();
    let line_bytes: usize = lines.iter().map(|line| line.len()).sum();
    eprintln!(
        "file {path}  {} bytes  {} lines  {} line-bytes  reps {reps}",
        text.len(),
        lines.len(),
        line_bytes
    );
    println!("| row | tokie file | tokie lines | gigatoken file | gigatoken lines | hf file | hf lines |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: |");
    for row in ROWS {
        if only.as_ref().is_some_and(|want| row.name != want) {
            continue;
        }
        let Some(json) = tokie_bench::rows::model_path(row) else {
            let file = time_prepare(row.cut, text, reps);
            let line = time_prepare_lines(row.cut, &lines, reps);
            println!(
                "| {} | {} | {} |  |  |  |  |",
                row.name,
                rate(Some(file), text.len()),
                rate(Some(line), line_bytes)
            );
            continue;
        };
        if pieces {
            profile_pieces(&json, row.cut, text);
            continue;
        }
        eprint!("{} ... ", row.name);
        let tokie_file = time_tokie(&json, text, reps);
        let tokie_lines = time_tokie_lines(&json, &lines, reps);
        let (hf_file, hf_lines) = if skip_hf {
            (None, None)
        } else {
            match hf::load(&json) {
            Ok(tok) => match hf::ids_of(&tok, "Hello") {
                Ok(_) => (
                    Some(time_hf(&tok, text, reps)),
                    Some(time_hf_lines(&tok, &lines, reps)),
                ),
                Err(err) => {
                    eprintln!("tokenizers v1 refused {}: {err}", json.display());
                    (None, None)
                }
            },
            Err(err) => {
                eprintln!("tokenizers v1 refused {}: {err}", json.display());
                (None, None)
            }
            }
        };
        let (giga_file, giga_lines) = time_giga(row.name, &json, text, &lines, reps);
        eprintln!("done");
        println!(
            "| {} | {} | {} | {} | {} | {} | {} |",
            row.name,
            rate(Some(tokie_file), text.len()),
            rate(Some(tokie_lines), line_bytes),
            rate(giga_file, text.len()),
            rate(giga_lines, line_bytes),
            rate(hf_file, text.len()),
            rate(hf_lines, line_bytes)
        );
    }
}

fn profile_pieces(path: &std::path::Path, cut: tokie::cut::Cut, text: &str) {
    let mut model = tokie::Tokenizer::open(path).expect("tokie tokenizer");
    let mut ids = Vec::new();
    model.encode(text, &mut ids).expect("encode");
    ids.clear();
    let encode = {
        let start = Instant::now();
        model.encode(text, &mut ids).expect("encode");
        black_box(ids.len());
        start.elapsed().as_secs_f64()
    };
    let mut spans = Vec::new();
    let split = {
        let start = Instant::now();
        tokie::norm::split(cut, text, &mut spans);
        black_box(spans.len());
        start.elapsed().as_secs_f64()
    };
    let pieces = spans.len().max(1);
    let (filled, slots, addr) = model.cache_stats();
    eprintln!(
        "encode {:>8.1} MB/s   split {:>8.1} MB/s   {:.0} ns/piece   {} pieces   {} ids",
        text.len() as f64 / 1e6 / encode,
        text.len() as f64 / 1e6 / split,
        encode * 1e9 / pieces as f64,
        spans.len(),
        ids.len()
    );
    eprintln!(
        "cache filled {filled} slots {slots} bytes {} at {addr:#x} {}",
        slots * 32,
        page_of(addr)
    );
}

fn page_of(addr: usize) -> String {
    let Ok(smaps) = std::fs::read_to_string("/proc/self/smaps") else {
        return "smaps unread".to_string();
    };
    let mut hit = false;
    for line in smaps.lines() {
        if let Some((range, rest)) = line.split_once(' ') {
            if let Some((start, end)) = range.split_once('-') {
                if start.chars().all(|c| c.is_ascii_hexdigit())
                    && end.chars().all(|c| c.is_ascii_hexdigit())
                    && !rest.is_empty()
                {
                    if let (Ok(start), Ok(end)) = (
                        usize::from_str_radix(start, 16),
                        usize::from_str_radix(end, 16),
                    ) {
                        hit = addr >= start && addr < end;
                    }
                    continue;
                }
            }
        }
        if hit && line.starts_with("AnonHugePages:") {
            let kb = line.split_whitespace().nth(1).unwrap_or("?");
            return format!("anon-huge {kb} kB");
        }
    }
    "unmapped".to_string()
}

fn rate(secs: Option<f64>, bytes: usize) -> String {
    match secs {
        Some(secs) if secs > 0.0 => format!("{:.1}", bytes as f64 / 1e6 / secs),
        _ => String::new(),
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    samples[samples.len() / 2]
}

fn time_prepare(cut: tokie::cut::Cut, text: &str, reps: usize) -> f64 {
    let mut a = String::new();
    let mut b = String::new();
    let mut spans = Vec::new();
    let mut body = || {
        let normalized = tokie::norm::apply(&[], text, &mut a, &mut b);
        tokie::norm::split(cut, normalized, &mut spans);
        black_box(spans.len());
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_prepare_lines(cut: tokie::cut::Cut, lines: &[&str], reps: usize) -> f64 {
    let mut a = String::new();
    let mut b = String::new();
    let mut spans = Vec::new();
    let mut body = || {
        let mut n = 0;
        for line in lines {
            let normalized = tokie::norm::apply(&[], line, &mut a, &mut b);
            tokie::norm::split(cut, normalized, &mut spans);
            n += spans.len();
        }
        black_box(n);
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_tokie(path: &std::path::Path, text: &str, reps: usize) -> f64 {
    let mut model = tokie::Tokenizer::open(path).expect("tokie tokenizer");
    let mut ids = Vec::new();
    let mut body = || {
        ids.clear();
        model.encode(text, &mut ids).expect("encode");
        black_box(ids.len());
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_tokie_lines(path: &std::path::Path, lines: &[&str], reps: usize) -> f64 {
    let mut model = tokie::Tokenizer::open(path).expect("tokie tokenizer");
    let mut ids = Vec::new();
    let mut body = || {
        let mut n = 0;
        for line in lines {
            ids.clear();
            model.encode(line, &mut ids).expect("encode");
            n += ids.len();
        }
        black_box(n);
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_hf(tokenizer: &tk_encode::tokenizer::pipeline::PipelineTokenizer, text: &str, reps: usize) -> f64 {
    let body = || {
        let ids = hf::ids_of(tokenizer, text).expect("tokenizers v1 encode");
        black_box(ids.len());
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_hf_lines(
    tokenizer: &tk_encode::tokenizer::pipeline::PipelineTokenizer,
    lines: &[&str],
    reps: usize,
) -> f64 {
    let body = || {
        let mut n = 0;
        for line in lines {
            n += hf::ids_of(tokenizer, line).expect("tokenizers v1 encode").len();
        }
        black_box(n);
    };
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    median(samples)
}

fn time_giga(
    name: &str,
    path: &std::path::Path,
    text: &str,
    lines: &[&str],
    reps: usize,
) -> (Option<f64>, Option<f64>) {
    let mut tokenizer = match load_giga(name, path) {
        Some(tokenizer) => tokenizer,
        None => return (None, None),
    };
        let mut ids = Vec::new();
        let file = {
            let mut body = || {
                ids.clear();
                tokenizer.encode_with_added_tokens_flat(text.as_bytes(), &mut ids);
                black_box(ids.len());
            };
            body();
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                body();
                samples.push(start.elapsed().as_secs_f64());
            }
            Some(median(samples))
        };
        let line = {
            let mut body = || {
                let mut n = 0;
                for line in lines {
                    ids.clear();
                    tokenizer.encode_with_added_tokens_flat(line.as_bytes(), &mut ids);
                    n += ids.len();
                }
                black_box(n);
            };
            body();
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                let start = Instant::now();
                body();
                samples.push(start.elapsed().as_secs_f64());
            }
            Some(median(samples))
        };
        (file, line)
}

fn load_giga(
    name: &str,
    path: &std::path::Path,
) -> Option<gigatoken_rs::Tokenizer> {
    match gigatoken_rs::load_tokenizer::hf::load_hf_bpe(path) {
        Ok(tokenizer) => Some(tokenizer),
        Err(err) => {
            let Some(ranks) = tokie_bench::rows::tiktoken_path(name) else {
                eprintln!("gigatoken refused {}: {err}", path.display());
                return None;
            };
            let specials = match added_specials(path) {
                Ok(specials) => specials,
                Err(reason) => {
                    eprintln!("gigatoken refused {}: {err}; added tokens: {reason}", path.display());
                    return None;
                }
            };
            match gigatoken_rs::load_tokenizer::tiktoken::load_tiktoken(
                ranks,
                gigatoken_rs::pretokenize::PretokenizerType::Kimi,
                specials,
            ) {
                Ok(tokenizer) => Some(tokenizer),
                Err(rank_err) => {
                    eprintln!(
                        "gigatoken refused {}: {err}; tiktoken ranks: {rank_err}",
                        path.display()
                    );
                    None
                }
            }
        }
    }
}

/// One warmup, then the median of `reps`. Times are milliseconds and include
/// the read. The warmup leaves the file in the page cache.
fn time_loads(reps: usize, only: Option<&str>, skip_hf: bool) {
    println!("| row | tokie json ms | tokie .tokie ms | tokie 128MiB ms | gigatoken ms | hf ms |");
    println!("| --- | ---: | ---: | ---: | ---: | ---: |");
    for row in ROWS {
        if only.is_some_and(|want| row.name != want) {
            continue;
        }
        let Some(json) = tokie_bench::rows::model_path(row) else {
            continue;
        };
        eprint!("{} ... ", row.name);
        let json_ms = median_ms(reps, || {
            let tok = tokie::Tokenizer::open(&json).expect("tokie json");
            black_box(tok.token(0));
        });
        let saved = std::env::temp_dir().join(format!("tokie-load-{}.tokie", row.name));
        tokie::Tokenizer::open(&json)
            .expect("tokie json")
            .save(&saved)
            .expect("save .tokie");
        let tokie_ms = median_ms(reps, || {
            let tok = tokie::Tokenizer::open(&saved).expect("tokie file");
            black_box(tok.token(0));
        });
        let piece_ms = if row.name == "gpt2" {
            seed_gpt2(&json)
        } else {
            None
        };
        let piece_ms = piece_ms.map(|path| {
            median_ms(reps, || {
                let tok = tokie::Tokenizer::open(&path).expect("piece file");
                black_box(tok.token(0));
            })
        });
        let giga_ms = if load_giga(row.name, &json).is_some() {
            Some(median_ms(reps, || {
                black_box(load_giga(row.name, &json));
            }))
        } else {
            None
        };
        let hf_ms = if skip_hf {
            None
        } else {
            match hf::load(&json) {
                Ok(_) => Some(median_ms(reps, || {
                    let tok = hf::load(&json).expect("hf");
                    black_box(tok);
                })),
                Err(err) => {
                    eprintln!("tokenizers v1 refused {}: {err}", json.display());
                    None
                }
            }
        };
        eprintln!("done");
        println!(
            "| {} | {:.1} | {:.1} | {} | {} | {} |",
            row.name,
            json_ms,
            tokie_ms,
            piece_ms.map(|ms| format!("{ms:.1}")).unwrap_or_default(),
            giga_ms.map(|ms| format!("{ms:.1}")).unwrap_or_default(),
            hf_ms.map(|ms| format!("{ms:.1}")).unwrap_or_default()
        );
    }
}

fn median_ms(reps: usize, mut load: impl FnMut()) -> f64 {
    load();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        load();
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    samples[samples.len() / 2] * 1_000.0
}

fn seed_gpt2(json: &std::path::Path) -> Option<std::path::PathBuf> {
    let rank = std::path::Path::new("/home/bhavnick/.cache/tokie-fineweb/rank.bin");
    if !rank.exists() {
        return None;
    }
    eprintln!("seeding the 128 MiB gpt-2 piece list");
    let bytes = std::fs::read(rank).expect("rank file");
    if &bytes[..8] != b"FWPIECE1" {
        return None;
    }
    let mut tok = tokie::Tokenizer::open(json).expect("gpt2");
    let mut at = 8usize;
    let mut cost = 0usize;
    let budget = 128 * 1024 * 1024;
    while at + 12 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at + 8..at + 12].try_into().unwrap()) as usize;
        at += 12;
        if at + len > bytes.len() {
            break;
        }
        let piece = &bytes[at..at + len];
        at += len;
        let next = if piece.len() <= 15 { 32 } else { 32 + piece.len() };
        if cost + next > budget {
            break;
        }
        let Ok(piece) = std::str::from_utf8(piece) else {
            continue;
        };
        cost += next;
        tok.preload_piece(piece);
    }
    let path = std::env::temp_dir().join("tokie-load-gpt2-pieces.tokie");
    tok.save(&path).expect("save piece file");
    Some(path)
}

fn added_specials(path: &std::path::Path) -> Result<Vec<(String, u32)>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    let value: serde_json::Value = serde_json::from_str(&text).map_err(|err| err.to_string())?;
    let list = value
        .get("added_tokens")
        .and_then(|token| token.as_array())
        .ok_or("added_tokens is missing")?;
    let mut specials = Vec::with_capacity(list.len());
    for token in list {
        let content = token
            .get("content")
            .and_then(|item| item.as_str())
            .unwrap_or("")
            .to_string();
        let id = token
            .get("id")
            .and_then(|item| item.as_u64())
            .ok_or("added token has no id")?;
        if content.is_empty() {
            continue;
        }
        specials.push((
            content,
            u32::try_from(id).map_err(|_| format!("added token id {id} does not fit u32"))?,
        ));
    }
    Ok(specials)
}
