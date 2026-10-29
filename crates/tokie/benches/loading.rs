//! Benchmark tokenizer loading times
//!
//! Compares tokie's .tkz binary format against HuggingFace tokenizers JSON loading.
//!
//! Run with: cargo bench --bench loading

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::path::Path;

const MODELS_DIR: &str = "models";
const JSON_DIR: &str = "/tmp/tokenizers";

struct TokenizerSpec {
    name: &'static str,
    tkz_file: &'static str,
    json_file: &'static str,
}

const TOKENIZERS: &[TokenizerSpec] = &[
    TokenizerSpec {
        name: "bert",
        tkz_file: "bert.tkz",
        json_file: "bert_tokenizer.json",
    },
    TokenizerSpec {
        name: "gpt2",
        tkz_file: "gpt2.tkz",
        json_file: "gpt2.json",
    },
    TokenizerSpec {
        name: "llama3",
        tkz_file: "llama3.tkz",
        json_file: "llama3.json",
    },
    TokenizerSpec {
        name: "cl100k",
        tkz_file: "cl100k.tkz",
        json_file: "cl100k.json",
    },
    TokenizerSpec {
        name: "o200k",
        tkz_file: "o200k.tkz",
        json_file: "o200k.json",
    },
];

fn bench_tokie_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokie_load");

    for spec in TOKENIZERS {
        let tkz_path = format!("{}/{}", MODELS_DIR, spec.tkz_file);
        if !Path::new(&tkz_path).exists() {
            eprintln!("Skipping {} - {} not found", spec.name, tkz_path);
            continue;
        }

        let size = std::fs::metadata(&tkz_path).map(|m| m.len()).unwrap_or(0);
        group.throughput(criterion::Throughput::Bytes(size));

        group.bench_with_input(
            BenchmarkId::new("tkz", spec.name),
            &tkz_path,
            |b, path| {
                b.iter(|| tokie::Tokenizer::from_file(black_box(path)).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_hf_tokenizers_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("hf_tokenizers_load");

    for spec in TOKENIZERS {
        let json_path = format!("{}/{}", JSON_DIR, spec.json_file);
        if !Path::new(&json_path).exists() {
            eprintln!("Skipping {} - {} not found", spec.name, json_path);
            continue;
        }

        let size = std::fs::metadata(&json_path).map(|m| m.len()).unwrap_or(0);
        group.throughput(criterion::Throughput::Bytes(size));

        group.bench_with_input(
            BenchmarkId::new("json", spec.name),
            &json_path,
            |b, path| {
                b.iter(|| tokenizers::Tokenizer::from_file(black_box(path)).unwrap())
            },
        );
    }

    group.finish();
}

fn bench_loading_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("loading_comparison");

    for spec in TOKENIZERS {
        let tkz_path = format!("{}/{}", MODELS_DIR, spec.tkz_file);
        let json_path = format!("{}/{}", JSON_DIR, spec.json_file);

        let tkz_exists = Path::new(&tkz_path).exists();
        let json_exists = Path::new(&json_path).exists();

        if !tkz_exists && !json_exists {
            continue;
        }

        if tkz_exists {
            group.bench_with_input(
                BenchmarkId::new(spec.name, "tokie"),
                &tkz_path,
                |b, path| {
                    b.iter(|| tokie::Tokenizer::from_file(black_box(path)).unwrap())
                },
            );
        }

        if json_exists {
            group.bench_with_input(
                BenchmarkId::new(spec.name, "hf_tokenizers"),
                &json_path,
                |b, path| {
                    b.iter(|| tokenizers::Tokenizer::from_file(black_box(path)).unwrap())
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tokie_loading,
    bench_hf_tokenizers_loading,
    bench_loading_comparison
);
criterion_main!(benches);
