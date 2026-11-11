//! Baseline throughput harness for tokie vs gigatoken on the same OpenWebText
//! bytes and the same tokenizer.json.
//!
//! Both sides run in release builds, with the special-token template off:
//! tokie uses `encode` / `batch`, gigatoken uses its HF-BPE encode path.
//!
//! Input is fully loaded before any timed region. Every timed result is
//! consumed: token count and an FNV-1a checksum are recorded, and every repeat
//! must produce the same count and checksum. `--validate` checks tokie ids
//! against gigatoken ids exactly, per document and whole-file.
//!
//! Single-pass cold runs are made by starting a fresh process per sample
//! (`./compat-bench ... --warmup 0`) and alternating the engine order in the
//! driver script. The in-process default repeats with warm caches, matching
//! the requested warmup + >=5 repeat, medians, and ranges.

use serde_json::json;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MODELS: &[(&str, &str)] = &[
    (
        "gpt2",
        "/home/bhavnick/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
    ),
    (
        "cl100k",
        "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4/snapshots/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json",
    ),
    (
        "o200k",
        "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4o/snapshots/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json",
    ),
    (
        "qwen2",
        "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json",
    ),
    (
        "qwen3.5",
        "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json",
    ),
    (
        "deepseek",
        "/home/bhavnick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json",
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Tokie,
    Giga,
}

impl Engine {
    fn name(self) -> &'static str {
        match self {
            Engine::Tokie => "tokie",
            Engine::Giga => "gigatoken",
        }
    }
}

struct Corpus {
    docs: Vec<String>,
    joined: String,
}

struct Opts {
    engine: String,
    mode: String,
    threads: usize,
    reps: usize,
    warmup: usize,
    model: String,
    corpus_dir: PathBuf,
    json_out: Option<PathBuf>,
    validate: bool,
    label: String,
}

fn main() {
    let opts = parse_args();
    let model_path = MODELS
        .iter()
        .find(|(name, _)| *name == opts.model)
        .unwrap_or_else(|| panic!("unknown model {}", opts.model))
        .1;

    let corpus = load_corpus(&opts.corpus_dir);
    let docs: Vec<&str> = corpus.docs.iter().map(String::as_str).collect();
    let doc_bytes: usize = docs.iter().map(|doc| doc.len()).sum();
    assert_eq!(
        corpus.joined,
        docs.join("\n"),
        "corpus txt and jsonl disagree on document boundaries"
    );
    eprintln!(
        "corpus: {} docs, {} doc bytes, {} joined bytes, model {} ({})",
        docs.len(),
        doc_bytes,
        corpus.joined.len(),
        opts.model,
        model_path
    );
    eprintln!(
        "threads requested {}; rayon sees {}; cpus_allowed {}",
        opts.threads,
        rayon::current_num_threads(),
        cpus_allowed()
    );
    assert_eq!(
        rayon::current_num_threads(),
        opts.threads,
        "set RAYON_NUM_THREADS={} before starting the process",
        opts.threads
    );

    if opts.validate {
        validate(&opts.model, model_path, &corpus, opts.threads);
        return;
    }

    let engines: Vec<Engine> = match opts.engine.as_str() {
        "tokie" => vec![Engine::Tokie],
        "gigatoken" | "giga" => vec![Engine::Giga],
        "both" => vec![Engine::Tokie, Engine::Giga],
        other => panic!("unknown engine {other}"),
    };

    let mut tokie = if engines.contains(&Engine::Tokie) {
        let mut tokenizer = tokie::Tokenizer::open(model_path)
            .unwrap_or_else(|err| panic!("tokie cannot load {model_path}: {err}"));
        tokenizer.threads(opts.threads);
        Some(tokenizer)
    } else {
        None
    };
    let mut giga = if engines.contains(&Engine::Giga) {
        Some(
            gigatoken_rs::load_tokenizer::hf::load_hf_bpe(model_path)
                .unwrap_or_else(|err| panic!("gigatoken cannot load {model_path}: {err}")),
        )
    } else {
        None
    };
    let pool = gigatoken_rs::WorkerPool::new();

    let input_bytes = if opts.mode == "single" {
        corpus.joined.len()
    } else {
        doc_bytes
    };
    eprintln!(
        "mode {}; timed input {} bytes; reps {}; warmup {}; engines {:?}",
        opts.mode, input_bytes, opts.reps, opts.warmup, engines
    );

    // Warm each engine `opts.warmup` times. Alternate which engine goes
    // first so the warmup itself does not favor one side.
    let mut warmup_runs: Vec<(Engine, usize)> =
        engines.iter().map(|&engine| (engine, 0)).collect();
    for warm in 0..opts.warmup {
        let order: Vec<Engine> = if warm % 2 == 0 {
            engines.clone()
        } else {
            let mut reversed = engines.clone();
            reversed.reverse();
            reversed
        };
        for engine in order {
            let run = run_engine(
                engine,
                &opts.mode,
                &docs,
                &corpus.joined,
                tokie.as_mut(),
                giga.as_mut(),
                &pool,
            );
            if let Some((_, count)) = warmup_runs.iter_mut().find(|(side, _)| *side == engine) {
                *count += 1;
            }
            eprintln!(
                "warmup {}/{} {}: {} tokens checksum {:016x} in {:.6}s",
                warm + 1,
                opts.warmup,
                engine.name(),
                run.tokens,
                run.checksum,
                run.secs
            );
        }
    }

    let mut samples: Vec<(Engine, Sample)> = Vec::new();
    let mut expected: Vec<(Engine, u64, u64)> = Vec::new();
    for rep in 0..opts.reps {
        // Alternate which engine goes first so thermal and cache drift hit
        // both sides rather than one.
        let order: Vec<Engine> = if rep % 2 == 0 {
            engines.clone()
        } else {
            let mut reversed = engines.clone();
            reversed.reverse();
            reversed
        };
        for engine in order {
            let run = run_engine(
                engine,
                &opts.mode,
                &docs,
                &corpus.joined,
                tokie.as_mut(),
                giga.as_mut(),
                &pool,
            );
            if let Some((_, tokens, checksum)) =
                expected.iter().find(|(side, _, _)| *side == engine)
            {
                assert_eq!(
                    (*tokens, *checksum),
                    (run.tokens, run.checksum),
                    "{} produced different ids on repeat {rep}",
                    engine.name()
                );
            } else {
                expected.push((engine, run.tokens, run.checksum));
            }
            eprintln!(
                "rep {rep} {}: {} tokens checksum {:016x} {:.6}s ({:.1} MB/s)",
                engine.name(),
                run.tokens,
                run.checksum,
                run.secs,
                input_bytes as f64 / 1e6 / run.secs
            );
            samples.push((engine, run));
        }
    }

    let mut results = Vec::new();
    for &engine in &engines {
        let times: Vec<f64> = samples
            .iter()
            .filter(|(side, _)| *side == engine)
            .map(|(_, sample)| sample.secs)
            .collect();
        let tokens = samples
            .iter()
            .find(|(side, _)| *side == engine)
            .map(|(_, sample)| sample.tokens)
            .expect("engine has a sample");
        let checksum = samples
            .iter()
            .find(|(side, _)| *side == engine)
            .map(|(_, sample)| sample.checksum)
            .expect("engine has a sample");
        let rows = samples
            .iter()
            .find(|(side, _)| *side == engine)
            .map(|(_, sample)| sample.rows)
            .expect("engine has a sample");
        let (median, min, max) = median_range(&times);
        let warmup_runs = warmup_runs
            .iter()
            .find(|(side, _)| *side == engine)
            .map(|(_, count)| *count)
            .unwrap_or(0);
        results.push(json!({
            "label": opts.label,
            "engine": engine.name(),
            "model": opts.model,
            "tokenizer": model_path,
            "mode": opts.mode,
            "threads": opts.threads,
            "reps": opts.reps,
            "warmup": opts.warmup,
            "warmup_runs": warmup_runs,
            "rows": rows,
            "input_bytes": input_bytes,
            "tokens": tokens,
            "tokens_per_byte": tokens as f64 / input_bytes as f64,
            "checksum": format!("{:016x}", checksum),
            "times_sec": times,
            "median_sec": median,
            "min_sec": min,
            "max_sec": max,
            "mb_per_s": input_bytes as f64 / 1e6 / median,
            "mib_per_s": input_bytes as f64 / (1024.0 * 1024.0) / median,
            "tokens_per_s": tokens as f64 / median,
            "cpus_allowed": cpus_allowed(),
        }));
    }

    let output = json!({
        "generated_unix": unix_now(),
        "results": results,
    });
    let pretty = serde_json::to_string_pretty(&output).expect("json");
    println!("{pretty}");
    if let Some(path) = opts.json_out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("json output dir");
        }
        std::fs::write(&path, format!("{pretty}\n")).expect("json output write");
    }
}

struct Sample {
    secs: f64,
    tokens: u64,
    checksum: u64,
    rows: usize,
}

#[allow(clippy::too_many_arguments)]
fn run_engine(
    engine: Engine,
    mode: &str,
    docs: &[&str],
    joined: &str,
    tokie: Option<&mut tokie::Tokenizer>,
    giga: Option<&mut gigatoken_rs::Tokenizer>,
    pool: &gigatoken_rs::WorkerPool,
) -> Sample {
    match (engine, mode) {
        (Engine::Tokie, "single") => {
            let tokenizer = tokie.expect("tokie");
            let mut ids = Vec::new();
            let start = Instant::now();
            tokenizer.encode(joined, &mut ids).expect("tokie encode");
            black_box(ids.len());
            let secs = start.elapsed().as_secs_f64();
            let (tokens, checksum) = consume(&ids);
            Sample {
                secs,
                tokens,
                checksum,
                rows: 1,
            }
        }
        (Engine::Tokie, "batch") => {
            let tokenizer = tokie.expect("tokie");
            let mut ids = Vec::new();
            let mut ends = Vec::new();
            let start = Instant::now();
            tokenizer.batch(docs, &mut ids, &mut ends).expect("tokie batch");
            black_box(ids.len());
            let secs = start.elapsed().as_secs_f64();
            assert_eq!(ends.len(), docs.len(), "tokie batch ends");
            assert_eq!(ends.last().copied().unwrap_or(0), ids.len(), "tokie batch end");
            let (tokens, checksum) = consume(&ids);
            Sample {
                secs,
                tokens,
                checksum,
                rows: ends.len(),
            }
        }
        (Engine::Giga, "single") => {
            let tokenizer = giga.expect("gigatoken");
            let mut ids = Vec::new();
            let start = Instant::now();
            tokenizer.encode_with_added_tokens_flat(joined.as_bytes(), &mut ids);
            black_box(ids.len());
            let secs = start.elapsed().as_secs_f64();
            let (tokens, checksum) = consume(&ids);
            Sample {
                secs,
                tokens,
                checksum,
                rows: 1,
            }
        }
        (Engine::Giga, "batch") => {
            let tokenizer = giga.expect("gigatoken");
            let byte_docs: Vec<&[u8]> = docs.iter().map(|doc| doc.as_bytes()).collect();
            let start = Instant::now();
            let (ids, lens) = gigatoken_rs::encode_docs_ragged(pool, tokenizer, &byte_docs);
            black_box((&ids, &lens));
            let secs = start.elapsed().as_secs_f64();
            assert_eq!(lens.len(), docs.len(), "gigatoken batch lens");
            assert_eq!(
                lens.iter().map(|len| *len as usize).sum::<usize>(),
                ids.len(),
                "gigatoken batch lengths"
            );
            let (tokens, checksum) = consume(&ids);
            Sample {
                secs,
                tokens,
                checksum,
                rows: lens.len(),
            }
        }
        (_, other) => panic!("unknown mode {other}"),
    }
}

fn consume(ids: &[u32]) -> (u64, u64) {
    black_box(ids.as_ptr());
    let mut hash = 0xcbf29ce484222325u64;
    for &id in ids {
        hash ^= id as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (ids.len() as u64, hash)
}

fn validate(model: &str, model_path: &str, corpus: &Corpus, threads: usize) {
    let docs: Vec<&str> = corpus.docs.iter().map(String::as_str).collect();
    let mut tokie_model = tokie::Tokenizer::open(model_path)
        .unwrap_or_else(|err| panic!("tokie cannot load {model_path}: {err}"));
    let mut giga_model = gigatoken_rs::load_tokenizer::hf::load_hf_bpe(model_path)
        .unwrap_or_else(|err| panic!("gigatoken cannot load {model_path}: {err}"));

    // Whole joined file, serial paths on both sides.
    let mut tokie_ids = Vec::new();
    tokie_model
        .encode(&corpus.joined, &mut tokie_ids)
        .expect("tokie whole encode");
    let mut giga_ids = Vec::new();
    giga_model.encode_with_added_tokens_flat(corpus.joined.as_bytes(), &mut giga_ids);
    assert_eq!(
        tokie_ids, giga_ids,
        "{model}: whole-file ids differ (tokie {} vs gigatoken {})",
        tokie_ids.len(),
        giga_ids.len()
    );
    eprintln!(
        "{model}: whole file exact match, {} tokens, checksum {:016x}",
        tokie_ids.len(),
        consume(&tokie_ids).1
    );

    // Per document, serial paths on both sides.
    for (index, doc) in docs.iter().enumerate() {
        tokie_ids.clear();
        tokie_model.encode(doc, &mut tokie_ids).expect("tokie doc");
        giga_ids.clear();
        giga_model.encode_with_added_tokens_flat(doc.as_bytes(), &mut giga_ids);
        assert_eq!(
            tokie_ids, giga_ids,
            "{model}: document {index} ids differ (tokie {} vs gigatoken {})",
            tokie_ids.len(),
            giga_ids.len()
        );
    }
    eprintln!("{model}: all {} documents exact match", docs.len());

    // Batched output must equal the per-document concatenation.
    let mut batched = Vec::new();
    let mut ends = Vec::new();
    tokie_model.threads(threads);
    tokie_model
        .batch(&docs, &mut batched, &mut ends)
        .expect("tokie batch validate");
    let mut serial = Vec::new();
    tokie_ids.clear();
    for doc in &docs {
        tokie_model.encode(doc, &mut tokie_ids).expect("tokie serial");
        serial.extend_from_slice(&tokie_ids);
        tokie_ids.clear();
    }
    assert_eq!(batched, serial, "{model}: tokie batch differs from encode");

    let byte_docs: Vec<&[u8]> = docs.iter().map(|doc| doc.as_bytes()).collect();
    let pool = gigatoken_rs::WorkerPool::new();
    let (giga_batch, lens) = gigatoken_rs::encode_docs_ragged(&pool, &giga_model, &byte_docs);
    assert_eq!(
        batched, giga_batch,
        "{model}: batch ids differ (tokie {} vs gigatoken {})",
        batched.len(),
        giga_batch.len()
    );
    assert_eq!(lens.len(), docs.len());
    eprintln!(
        "{model}: batch exact match with {} threads, {} tokens, checksum {:016x}",
        threads,
        batched.len(),
        consume(&batched).1
    );
}

fn load_corpus(dir: &Path) -> Corpus {
    let jsonl = dir.join("owt_20m.jsonl");
    let txt = dir.join("owt_20m.txt");
    let text = std::fs::read_to_string(&jsonl)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", jsonl.display()));
    let mut docs = Vec::new();
    for (line, raw) in text.lines().enumerate() {
        let value: serde_json::Value = serde_json::from_str(raw)
            .unwrap_or_else(|err| panic!("{}:{}: {err}", jsonl.display(), line + 1));
        docs.push(
            value
                .get("text")
                .and_then(|text| text.as_str())
                .unwrap_or_else(|| panic!("{}:{}: no text", jsonl.display(), line + 1))
                .to_string(),
        );
    }
    let joined = std::fs::read_to_string(&txt)
        .unwrap_or_else(|err| panic!("cannot read {}: {err}", txt.display()));
    Corpus { docs, joined }
}

fn median_range(samples: &[f64]) -> (f64, f64, f64) {
    assert!(!samples.is_empty(), "no samples");
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let median = sorted[sorted.len() / 2];
    (median, sorted[0], sorted[sorted.len() - 1])
}

fn parse_args() -> Opts {
    let mut opts = Opts {
        engine: "both".to_string(),
        mode: "single".to_string(),
        threads: 1,
        reps: 5,
        warmup: 1,
        model: "gpt2".to_string(),
        corpus_dir: PathBuf::from("/tmp/opencode/tokie-compat/corpus"),
        json_out: None,
        validate: false,
        label: String::new(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().expect("flag needs a value");
        match arg.as_str() {
            "--engine" => opts.engine = value(),
            "--mode" => opts.mode = value(),
            "--threads" => opts.threads = value().parse().expect("threads"),
            "--reps" => opts.reps = value().parse().expect("reps"),
            "--warmup" => opts.warmup = value().parse().expect("warmup"),
            "--model" => opts.model = value(),
            "--corpus-dir" => opts.corpus_dir = PathBuf::from(value()),
            "--json-out" => opts.json_out = Some(PathBuf::from(value())),
            "--label" => opts.label = value(),
            "--validate" => opts.validate = true,
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(["single", "batch"].contains(&opts.mode.as_str()));
    assert!(opts.reps > 0, "--reps must be positive");
    assert!(opts.threads > 0, "--threads must be positive");
    opts
}

fn cpus_allowed() -> String {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("Cpus_allowed_list:"))
                .map(|line| line.split_whitespace().nth(1).unwrap_or("?").to_string())
        })
        .unwrap_or_else(|| "?".to_string())
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_secs()
}
