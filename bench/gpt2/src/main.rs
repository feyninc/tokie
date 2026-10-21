//! Release comparison of one pretokenizer on one English file.
//!
//! pretokie fill writes spans in batches of 256. pretokie count only
//! popcounts boundaries. gigatoken iter yields each piece. gigatoken fill
//! also packs a cache key and prefetches it.

use gigatoken_rs::pretokenize::fast::{
    FastKimiPretokenizer, FastNemotronPretokenizer, FastO200kPretokenizer,
};
use gigatoken_rs::pretokenize::{
    FastCl100kPretokenizer, FastDeepSeekV3Pretokenizer, FastOlmo3Pretokenizer,
    FastQwen2Pretokenizer, FastQwen35Pretokenizer, FastR50kPretokenizer, PretokenSpans, SpanBatch,
    PRETOKEN_CHUNK,
};
use pretokie::{Cl100k, DeepSeek, Gpt2, Kimi, Nemotron, O200k, Olmo, Qwen, SmolLM, Span, Voyage};
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
enum Mode {
    Gpt2,
    SmolLM,
    Cl100k,
    Voyage,
    Olmo,
    Qwen,
    O200k,
    Nemotron,
    Kimi,
    DeepSeek,
}

fn main() {
    let mut file = None;
    let mut reps = 5usize;
    let mut repeat = 1usize;
    let mut max_bytes = usize::MAX;
    let mut mode = Mode::Gpt2;
    let mut chosen = false;
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
            "--smollm" => mode = select(&mut chosen, Mode::SmolLM),
            "--cl100k" => mode = select(&mut chosen, Mode::Cl100k),
            "--voyage" => mode = select(&mut chosen, Mode::Voyage),
            "--olmo" => mode = select(&mut chosen, Mode::Olmo),
            "--qwen" => mode = select(&mut chosen, Mode::Qwen),
            "--o200k" => mode = select(&mut chosen, Mode::O200k),
            "--nemotron" => mode = select(&mut chosen, Mode::Nemotron),
            "--kimi" => mode = select(&mut chosen, Mode::Kimi),
            "--deepseek" => mode = select(&mut chosen, Mode::DeepSeek),
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
    if diff {
        show_diff(&bytes, mode);
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
    let (fill_name, count_name) = names(mode);
    let fill_pieces = match mode {
        Mode::Gpt2 => report(fill_name, measure(reps, valid, || fill_gpt2(text))),
        Mode::SmolLM => report(fill_name, measure(reps, valid, || fill_smollm(text))),
        Mode::Cl100k => report(fill_name, measure(reps, valid, || fill_cl100k(text))),
        Mode::Voyage => report(fill_name, measure(reps, valid, || fill_voyage(text))),
        Mode::Olmo => report(fill_name, measure(reps, valid, || fill_olmo(text))),
        Mode::Qwen => report(fill_name, measure(reps, valid, || fill_qwen(text))),
        Mode::O200k => report(fill_name, measure(reps, valid, || fill_o200k(text))),
        Mode::Nemotron => report(fill_name, measure(reps, valid, || fill_nemotron(text))),
        Mode::Kimi => report(fill_name, measure(reps, valid, || fill_kimi(text))),
        Mode::DeepSeek => report(fill_name, measure(reps, valid, || fill_deepseek(text))),
    };
    let count_pieces = match mode {
        Mode::Gpt2 => report(count_name, measure(reps, valid, || Gpt2::count(text))),
        Mode::SmolLM => report(count_name, measure(reps, valid, || SmolLM::count(text))),
        Mode::Cl100k => report(count_name, measure(reps, valid, || Cl100k::count(text))),
        Mode::Voyage => report(count_name, measure(reps, valid, || Voyage::count(text))),
        Mode::Olmo => report(count_name, measure(reps, valid, || Olmo::count(text))),
        Mode::Qwen => report(count_name, measure(reps, valid, || Qwen::count(text))),
        Mode::O200k => report(count_name, measure(reps, valid, || O200k::count(text))),
        Mode::Nemotron => report(count_name, measure(reps, valid, || Nemotron::count(text))),
        Mode::Kimi => report(count_name, measure(reps, valid, || Kimi::count(text))),
        Mode::DeepSeek => report(count_name, measure(reps, valid, || DeepSeek::count(text))),
    };
    let iter_pieces = match mode {
        Mode::Gpt2 | Mode::SmolLM => {
            report("gigatoken iter", measure(reps, valid, || iter_count(FastR50kPretokenizer::new(&bytes))))
        }
        Mode::Cl100k => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastCl100kPretokenizer::new(&bytes))),
        ),
        Mode::Voyage => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastQwen2Pretokenizer::new(&bytes))),
        ),
        Mode::Olmo => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastOlmo3Pretokenizer::new(&bytes))),
        ),
        Mode::Qwen => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastQwen35Pretokenizer::new(&bytes))),
        ),
        Mode::O200k => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastO200kPretokenizer::new(&bytes))),
        ),
        Mode::Nemotron => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastNemotronPretokenizer::new(&bytes))),
        ),
        Mode::Kimi => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastKimiPretokenizer::new(&bytes))),
        ),
        Mode::DeepSeek => report(
            "gigatoken iter",
            measure(reps, valid, || iter_count(FastDeepSeekV3Pretokenizer::new(&bytes))),
        ),
    };
    let giga_pieces = match mode {
        Mode::Gpt2 | Mode::SmolLM => {
            report("gigatoken fill", measure(reps, valid, || fill_spans(FastR50kPretokenizer::new(&bytes))))
        }
        Mode::Cl100k => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastCl100kPretokenizer::new(&bytes))),
        ),
        Mode::Voyage => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastQwen2Pretokenizer::new(&bytes))),
        ),
        Mode::Olmo => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastOlmo3Pretokenizer::new(&bytes))),
        ),
        Mode::Qwen => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastQwen35Pretokenizer::new(&bytes))),
        ),
        Mode::O200k => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastO200kPretokenizer::new(&bytes))),
        ),
        Mode::Nemotron => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastNemotronPretokenizer::new(&bytes))),
        ),
        Mode::Kimi => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastKimiPretokenizer::new(&bytes))),
        ),
        Mode::DeepSeek => report(
            "gigatoken fill",
            measure(reps, valid, || fill_spans(FastDeepSeekV3Pretokenizer::new(&bytes))),
        ),
    };
    if count_pieces != fill_pieces {
        eprintln!("piece count mismatch: {fill_name}={fill_pieces} {count_name}={count_pieces}");
    }
    if matches!(mode, Mode::SmolLM) {
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

fn select(chosen: &mut bool, mode: Mode) -> Mode {
    assert!(
        !*chosen,
        "pass only one of --smollm --cl100k --voyage --olmo --qwen --o200k --nemotron --kimi --deepseek"
    );
    *chosen = true;
    mode
}

fn names(mode: Mode) -> (&'static str, &'static str) {
    match mode {
        Mode::Gpt2 => ("pretokie fill", "pretokie count"),
        Mode::SmolLM => ("smollm fill", "smollm count"),
        Mode::Cl100k => ("cl100k fill", "cl100k count"),
        Mode::Voyage => ("voyage fill", "voyage count"),
        Mode::Olmo => ("olmo fill", "olmo count"),
        Mode::Qwen => ("qwen fill", "qwen count"),
        Mode::O200k => ("o200k fill", "o200k count"),
        Mode::Nemotron => ("nemotron fill", "nemotron count"),
        Mode::Kimi => ("kimi fill", "kimi count"),
        Mode::DeepSeek => ("deepseek fill", "deepseek count"),
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

#[inline(always)]
fn fill_walk(mut fill: impl FnMut(&mut [Span]) -> usize) -> usize {
    let mut buf = [Span { start: 0, end: 0 }; PRETOKEN_CHUNK];
    let mut n = 0;
    loop {
        let written = fill(&mut buf);
        if written == 0 {
            break;
        }
        black_box(&buf[..written]);
        n += written;
    }
    n
}

fn fill_gpt2(text: &str) -> usize {
    let mut walk = Gpt2::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_smollm(text: &str) -> usize {
    let mut walk = SmolLM::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_cl100k(text: &str) -> usize {
    let mut walk = Cl100k::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_voyage(text: &str) -> usize {
    let mut walk = Voyage::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_olmo(text: &str) -> usize {
    let mut walk = Olmo::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_qwen(text: &str) -> usize {
    let mut walk = Qwen::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_o200k(text: &str) -> usize {
    let mut walk = O200k::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_nemotron(text: &str) -> usize {
    let mut walk = Nemotron::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_kimi(text: &str) -> usize {
    let mut walk = Kimi::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn fill_deepseek(text: &str) -> usize {
    let mut walk = DeepSeek::walk(text);
    fill_walk(|buf| walk.fill(buf))
}

fn show_diff(bytes: &[u8], mode: Mode) {
    let text = std::str::from_utf8(bytes).unwrap();
    match mode {
        Mode::Gpt2 => {
            let mut walk = Gpt2::walk(text);
            diff_against(text, bytes, |buf| walk.fill(buf), FastR50kPretokenizer::new(bytes));
        }
        Mode::SmolLM => {
            let mut walk = SmolLM::walk(text);
            diff_against(text, bytes, |buf| walk.fill(buf), FastR50kPretokenizer::new(bytes));
        }
        Mode::Cl100k => {
            let mut walk = Cl100k::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastCl100kPretokenizer::new(bytes),
            );
        }
        Mode::Voyage => {
            let mut walk = Voyage::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastQwen2Pretokenizer::new(bytes),
            );
        }
        Mode::Olmo => {
            let mut walk = Olmo::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastOlmo3Pretokenizer::new(bytes),
            );
        }
        Mode::Qwen => {
            let mut walk = Qwen::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastQwen35Pretokenizer::new(bytes),
            );
        }
        Mode::O200k => {
            let mut walk = O200k::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastO200kPretokenizer::new(bytes),
            );
        }
        Mode::Nemotron => {
            let mut walk = Nemotron::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastNemotronPretokenizer::new(bytes),
            );
        }
        Mode::Kimi => {
            let mut walk = Kimi::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastKimiPretokenizer::new(bytes),
            );
        }
        Mode::DeepSeek => {
            let mut walk = DeepSeek::walk(text);
            diff_against(
                text,
                bytes,
                |buf| walk.fill(buf),
                FastDeepSeekV3Pretokenizer::new(bytes),
            );
        }
    }
}

fn diff_against<I, P>(text: &str, bytes: &[u8], mut fill: impl FnMut(&mut [Span]) -> usize, mut giga: I)
where
    I: Iterator<Item = P>,
    P: AsRef<[u8]>,
{
    let mut buf = [Span { start: 0, end: 0 }; 64];
    let mut filled = 0;
    let mut next = 0;
    let mut idx = 0usize;
    loop {
        if next == filled {
            filled = fill(&mut buf);
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
        if piece.as_ref() != &bytes[start..end] {
            let from = start.saturating_sub(40);
            let to = (end + 40).min(bytes.len());
            eprintln!("mismatch at piece {idx}");
            eprintln!("pretokie {}..{} {:?}", start, end, &text[start..end]);
            eprintln!(
                "gigatoken {} bytes {:?}",
                piece.as_ref().len(),
                String::from_utf8_lossy(piece.as_ref())
            );
            eprintln!("context {:?}", &text[from..to]);
            return;
        }
        idx += 1;
    }
}

#[inline(always)]
fn iter_count<I, P>(iter: I) -> usize
where
    I: Iterator<Item = P>,
    P: AsRef<[u8]>,
{
    let mut n = 0;
    for piece in iter {
        black_box(piece.as_ref());
        n += 1;
    }
    n
}

#[inline(always)]
fn fill_spans<'a>(mut pretok: impl PretokenSpans<'a>) -> usize {
    let mut batch = SpanBatch::new();
    let mut n = 0;
    loop {
        let written = pretok.fill_spans_keyed(&mut batch, &|_| {});
        if written == 0 {
            break;
        }
        black_box(written);
        black_box(unsafe { batch.span(0) });
        n += written;
    }
    n
}

fn measure(reps: usize, bytes: usize, mut run: impl FnMut() -> usize) -> (Vec<f64>, Vec<f64>, usize) {
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
