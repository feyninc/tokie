//! Release comparison of the GPT-2 pretokenizer on one English file.
//!
//! pretokie fill writes spans in batches of 256. pretokie count only
//! popcounts boundaries. gigatoken iter yields each piece. gigatoken fill
//! also packs a cache key and prefetches it.

use gigatoken_rs::pretokenize::{
    FastCl100kPretokenizer, FastR50kPretokenizer, PretokenSpans, SpanBatch, PRETOKEN_CHUNK,
};
use pretokie::{Cl100k, Gpt2, SmolLM, Span};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    let mut file = None;
    let mut reps = 5usize;
    let mut repeat = 1usize;
    let mut max_bytes = usize::MAX;
    let mut smollm = false;
    let mut cl100k = false;
    let mut diff = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--file" => file = Some(args.next().expect("--file needs a value")),
            "--repeat" => {
                repeat = args
                    .next()
                    .expect("--repeat needs a value")
                    .parse()
                    .expect("--repeat must be an integer");
                assert!(repeat >= 1, "--repeat must be at least 1");
            }
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
            "--smollm" => smollm = true,
            "--cl100k" => cl100k = true,
            "--diff" => diff = true,
            other => panic!("unknown argument: {other}"),
        }
    }
    let path = file.expect("--file is required");
    let mut bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("cannot read {path}: {err}"));
    bytes.truncate(max_bytes.min(bytes.len()));
    let valid = utf8_len(&bytes);
    bytes.truncate(valid);
    if repeat > 1 {
        let once = bytes.len();
        bytes.resize(once * repeat, 0);
        for i in 1..repeat {
            bytes.copy_within(0..once, i * once);
        }
    }
    assert!(
        !smollm || !cl100k,
        "--smollm and --cl100k select different formulas"
    );
    if diff {
        show_diff(&bytes);
        return;
    }
    let text = std::str::from_utf8(&bytes).unwrap();
    let valid = text.len();
    let high = text.bytes().filter(|b| *b >= 0x80).count();

    eprintln!(
        "input: {:.1} MB, copies: {repeat}, non-ascii bytes: {:.2}%, reps: {reps} plus one warmup",
        valid as f64 / 1e6,
        high as f64 * 100.0 / valid as f64,
    );
    println!(
        "{:<16} {:>10} {:>10} {:>10} {:>12}",
        "impl", "min MB/s", "med MB/s", "med ms", "pieces"
    );
    let (fill_name, count_name) = if cl100k {
        ("cl100k fill", "cl100k count")
    } else if smollm {
        ("smollm fill", "smollm count")
    } else {
        ("pretokie fill", "pretokie count")
    };
    let fill_pieces = report(
        fill_name,
        measure(reps, valid, || fill_pretokie(text, smollm, cl100k)),
    );
    let count_pieces = report(count_name, measure(reps, valid, || {
        if cl100k {
            Cl100k::count(text)
        } else if smollm {
            SmolLM::count(text)
        } else {
            Gpt2::count(text)
        }
    }));
    let iter_pieces = report(
        "gigatoken iter",
        measure(reps, valid, || iter_giga(&bytes, cl100k)),
    );
    let giga_pieces = report(
        "gigatoken fill",
        measure(reps, valid, || fill_giga(&bytes, cl100k)),
    );
    if count_pieces != fill_pieces {
        eprintln!("piece count mismatch: {fill_name}={fill_pieces} {count_name}={count_pieces}");
    }
    if smollm {
        eprintln!(
            "cuts differ: smollm isolates digits, fastr50k keeps a number together ({fill_pieces} vs {giga_pieces} pieces)"
        );
    } else {
        for (name, count) in [("gigatoken iter", iter_pieces), ("gigatoken fill", giga_pieces)] {
            if count != fill_pieces {
                eprintln!("piece count mismatch: pretokie fill={fill_pieces} {name}={count}");
            }
        }
    }
}

fn report(name: &str, (rates, millis, count): (Vec<f64>, Vec<f64>, usize)) -> usize {
    let mut order: Vec<usize> = (0..rates.len()).collect();
    order.sort_by(|&a, &b| rates[a].partial_cmp(&rates[b]).unwrap());
    let mid = order[order.len() / 2];
    println!(
        "{name:<16} {:>10.1} {:>10.1} {:>10.1} {count:>12}",
        rates[order[0]],
        rates[mid],
        millis[mid],
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    count
}

fn fill_pretokie(text: &str, smollm: bool, cl100k: bool) -> usize {
    if cl100k {
        fill_cl100k(text)
    } else if smollm {
        fill_smollm(text)
    } else {
        fill_gpt2(text)
    }
}

fn fill_gpt2(text: &str) -> usize {
    let mut buf = [Span { start: 0, end: 0 }; PRETOKEN_CHUNK];
    let mut walk = Gpt2::walk(text);
    let mut n = 0;
    loop {
        let written = walk.fill(&mut buf);
        if written == 0 {
            break;
        }
        black_box(&buf[..written]);
        n += written;
    }
    n
}

fn fill_smollm(text: &str) -> usize {
    let mut buf = [Span { start: 0, end: 0 }; PRETOKEN_CHUNK];
    let mut walk = SmolLM::walk(text);
    let mut n = 0;
    loop {
        let written = walk.fill(&mut buf);
        if written == 0 {
            break;
        }
        black_box(&buf[..written]);
        n += written;
    }
    n
}

fn show_diff(bytes: &[u8]) {
    let text = std::str::from_utf8(bytes).unwrap();
    let mut walk = Cl100k::walk(text);
    let mut buf = [Span { start: 0, end: 0 }; 64];
    let mut giga = FastCl100kPretokenizer::new(bytes);
    let mut filled = 0;
    let mut next = 0;
    let mut idx = 0usize;
    loop {
        if next == filled {
            filled = walk.fill(&mut buf);
            next = 0;
            if filled == 0 {
                if giga.next().is_some() {
                    eprintln!("pretokie ended at piece {idx}, gigatoken continues");
                } else {
                    eprintln!("no difference in {idx} pieces");
                }
                return;
            }
        }
        let span = buf[next];
        next += 1;
        let Some(piece) = giga.next() else {
            eprintln!("gigatoken ended at piece {idx}");
            return;
        };
        let (start, end) = (span.start as usize, span.end as usize);
        if piece.0 != &bytes[start..end] {
            let from = start.saturating_sub(40);
            let to = (end + 40).min(bytes.len());
            eprintln!("mismatch at piece {idx}");
            eprintln!("pretokie {}..{} {:?}", start, end, &text[start..end]);
            eprintln!(
                "gigatoken {} bytes {:?}",
                piece.0.len(),
                String::from_utf8_lossy(piece.0)
            );
            eprintln!("context {:?}", &text[from..to]);
            return;
        }
        idx += 1;
    }
}

fn fill_cl100k(text: &str) -> usize {
    let mut buf = [Span { start: 0, end: 0 }; PRETOKEN_CHUNK];
    let mut walk = Cl100k::walk(text);
    let mut n = 0;
    loop {
        let written = walk.fill(&mut buf);
        if written == 0 {
            break;
        }
        black_box(&buf[..written]);
        n += written;
    }
    n
}

fn iter_giga(bytes: &[u8], cl100k: bool) -> usize {
    if cl100k {
        let mut n = 0;
        for piece in FastCl100kPretokenizer::new(bytes) {
            black_box(piece.0);
            n += 1;
        }
        n
    } else {
        let mut n = 0;
        for piece in FastR50kPretokenizer::new(bytes) {
            black_box(piece.0);
            n += 1;
        }
        n
    }
}

fn fill_giga(bytes: &[u8], cl100k: bool) -> usize {
    let mut batch = SpanBatch::new();
    let mut n = 0;
    if cl100k {
        let mut pretok = FastCl100kPretokenizer::new(bytes);
        loop {
            let written = pretok.fill_spans_keyed(&mut batch, &|_| {});
            if written == 0 {
                break;
            }
            black_box(written);
            black_box(unsafe { batch.span(0) });
            n += written;
        }
    } else {
        let mut pretok = FastR50kPretokenizer::new(bytes);
        loop {
            let written = pretok.fill_spans_keyed(&mut batch, &|_| {});
            if written == 0 {
                break;
            }
            black_box(written);
            black_box(unsafe { batch.span(0) });
            n += written;
        }
    }
    n
}

fn measure(
    reps: usize,
    bytes: usize,
    mut run: impl FnMut() -> usize,
) -> (Vec<f64>, Vec<f64>, usize) {
    let count = run();
    let mut rates = Vec::with_capacity(reps);
    let mut millis = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let got = run();
        let elapsed = start.elapsed().as_secs_f64();
        assert_eq!(got, count);
        rates.push(bytes as f64 / elapsed / 1e6);
        millis.push(elapsed * 1e3);
    }
    (rates, millis, count)
}

fn utf8_len(bytes: &[u8]) -> usize {
    let mut end = bytes.len();
    while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() {
        end -= 1;
    }
    end
}
