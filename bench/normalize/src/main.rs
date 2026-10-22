//! Throughput of one normalizer scheme on one file.
//!
//! Each library is timed on the whole file and again on every newline-separated
//! line (the newline itself is not part of the line). One warmup, then the
//! median of `--reps` passes. A library with no scheme for the row is blank.
//! Rates are input megabytes per second.

use normalize_bench::Scheme;
use std::borrow::Cow;
use std::hint::black_box;
use std::time::Instant;

const DEFAULT_FILE: &str = "/home/bhavnick/Workspaces/pretok-vs-giga/data/owt_sample.txt";

fn main() {
    let mut file = None;
    let mut reps = 5usize;
    let mut max_bytes = usize::MAX;
    let mut max_lines = usize::MAX;
    let mut only = None;
    let mut check_only = false;
    let mut no_check = false;
    let mut libs = String::from("normalie,giga,hf");
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
                assert!(reps >= 1, "--reps must be at least 1");
            }
            "--bytes" => {
                max_bytes = args
                    .next()
                    .expect("--bytes needs a value")
                    .parse()
                    .expect("--bytes must be an integer");
            }
            "--lines" => {
                max_lines = args
                    .next()
                    .expect("--lines needs a value")
                    .parse()
                    .expect("--lines must be an integer");
            }
            "--scheme" => only = Some(args.next().expect("--scheme needs a value")),
            "--check-only" => check_only = true,
            "--no-check" => no_check = true,
            "--libs" => libs = args.next().expect("--libs needs a value"),
            other => panic!("unknown argument: {other}"),
        }
    }

    let path = file.unwrap_or_else(|| DEFAULT_FILE.to_string());
    let mut bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
    bytes.truncate(max_bytes.min(bytes.len()));
    let valid = utf8_len(&bytes);
    bytes.truncate(valid);
    let text = std::str::from_utf8(&bytes).unwrap();
    let mut lines: Vec<&str> = text.split('\n').collect();
    if max_lines < lines.len() {
        lines.truncate(max_lines);
    }
    let line_bytes: usize = lines.iter().map(|line| line.len()).sum();

    eprintln!(
        "file {path}  {} bytes  {} lines  {} line-bytes  reps {reps}",
        text.len(),
        lines.len(),
        line_bytes
    );

    println!(
        "| scheme | normalie file | normalie lines | gigatoken file | gigatoken lines | hf file | hf lines | agree |"
    );
    println!("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |");

    for scheme in Scheme::ALL {
        if only
            .as_ref()
            .is_some_and(|want| !scheme.name().contains(want.as_str()))
        {
            continue;
        }
        eprint!("{} ... ", scheme.name());
        let agree = if no_check {
            String::new()
        } else {
            agreement(*scheme, text, &lines)
        };
        let (ours, giga, hf) = if check_only {
            (None, None, None)
        } else {
            (
                if libs.contains("normalie") {
                    time_lib(reps, text, &lines, |s| scheme.normalie(s))
                } else {
                    None
                },
                if libs.contains("giga") {
                    time_lib(reps, text, &lines, |s| scheme.giga(s))
                } else {
                    None
                },
                if libs.contains("hf") {
                    time_lib(reps, text, &lines, |s| scheme.hf(s))
                } else {
                    None
                },
            )
        };
        eprintln!("{agree}");
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            scheme.name(),
            rate(ours.as_ref().map(|r| r.file_secs), text.len()),
            rate(ours.as_ref().map(|r| r.line_secs), line_bytes),
            rate(giga.as_ref().map(|r| r.file_secs), text.len()),
            rate(giga.as_ref().map(|r| r.line_secs), line_bytes),
            rate(hf.as_ref().map(|r| r.file_secs), text.len()),
            rate(hf.as_ref().map(|r| r.line_secs), line_bytes),
            agree.replace('|', "/")
        );
    }
}

struct Rates {
    file_secs: f64,
    line_secs: f64,
}

fn time_lib<'a>(
    reps: usize,
    text: &'a str,
    lines: &[&'a str],
    mut run: impl FnMut(&str) -> Option<Cow<'_, str>>,
) -> Option<Rates> {
    if run(text).is_none() {
        return None;
    }
    let file_secs = median(reps, || {
        let out = run(text).unwrap();
        black_box(out.as_ptr());
        black_box(out.len());
    });
    let line_secs = median(reps, || {
        for line in lines {
            let out = run(line).unwrap();
            black_box(out.as_ptr());
            black_box(out.len());
        }
    });
    Some(Rates {
        file_secs,
        line_secs,
    })
}

fn median(reps: usize, mut body: impl FnMut()) -> f64 {
    body();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        samples.push(start.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[reps / 2]
}

fn rate(secs: Option<f64>, bytes: usize) -> String {
    match secs {
        None => String::new(),
        Some(secs) => format!("{:.1}", bytes as f64 / 1e6 / secs),
    }
}

#[derive(Clone, Copy)]
enum Kind {
    Normalie,
    Hf,
    Giga,
}

fn owned(scheme: Scheme, text: &str, kind: Kind) -> Option<String> {
    let cow = match kind {
        Kind::Hf => scheme.hf(text)?,
        Kind::Normalie => scheme.normalie(text)?,
        Kind::Giga => scheme.giga(text)?,
    };
    Some(cow.into_owned())
}

fn agreement(scheme: Scheme, text: &str, lines: &[&str]) -> String {
    let mut notes = Vec::new();
    if scheme.normalie(text).is_some() && scheme.hf(text).is_some() {
        if let Some(diff) = first_diff(scheme, text, lines, Kind::Normalie, Kind::Hf) {
            notes.push(format!("normalie≠hf {diff}"));
        }
    }
    if scheme.normalie(text).is_some() && scheme.giga(text).is_some() {
        if let Some(diff) = first_diff(scheme, text, lines, Kind::Normalie, Kind::Giga) {
            notes.push(format!("normalie≠giga {diff}"));
        }
    }
    if notes.is_empty() {
        if scheme.normalie(text).is_some() {
            "yes".to_string()
        } else {
            "—".to_string()
        }
    } else {
        notes.join("; ")
    }
}

fn first_diff(scheme: Scheme, text: &str, lines: &[&str], left: Kind, right: Kind) -> Option<String> {
    let a = owned(scheme, text, left).unwrap();
    let b = owned(scheme, text, right).unwrap();
    if a != b {
        return Some(format!("file {}", snippet(&a, &b)));
    }
    for (i, line) in lines.iter().enumerate() {
        let a = owned(scheme, line, left).unwrap();
        let b = owned(scheme, line, right).unwrap();
        if a != b {
            return Some(format!("line {i} {}", snippet(&a, &b)));
        }
    }
    None
}

fn snippet(a: &str, b: &str) -> String {
    fn cut(s: &str) -> String {
        let n: String = s.chars().take(24).collect();
        if s.chars().count() > 24 {
            format!("{n}…")
        } else {
            n
        }
    }
    format!("{:?} vs {:?}", cut(a), cut(b))
}

fn utf8_len(bytes: &[u8]) -> usize {
    let mut n = bytes.len();
    while n > 0 && std::str::from_utf8(&bytes[..n]).is_err() {
        n -= 1;
    }
    n
}
