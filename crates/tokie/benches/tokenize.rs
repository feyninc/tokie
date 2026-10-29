use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::fs;
use tokie::Tokenizer;

const TOKENIZER_PATH: &str = "benches/data/gpt2_tokenizer.json";
const TEXT_PATH: &str = "benches/data/war_and_peace.txt";

fn load_tokenizer() -> Tokenizer {
    if !std::path::Path::new(TOKENIZER_PATH).exists() {
        panic!(
            "Missing {}. Download from: https://huggingface.co/openai-community/gpt2/raw/main/tokenizer.json",
            TOKENIZER_PATH
        );
    }
    Tokenizer::from_json(TOKENIZER_PATH).expect("Failed to load tokenizer")
}

fn load_text() -> String {
    if !std::path::Path::new(TEXT_PATH).exists() {
        panic!(
            "Missing {}. Download from: https://www.gutenberg.org/files/2600/2600-0.txt",
            TEXT_PATH
        );
    }
    fs::read_to_string(TEXT_PATH).expect("Failed to read text file")
}

fn bench_encode(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();

    let mut group = c.benchmark_group("encode");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("war_and_peace", |b| {
        b.iter(|| tokenizer.encode(black_box(&text), false))
    });

    group.finish();
}

fn bench_encode_with_offsets(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();

    let mut group = c.benchmark_group("encode_with_offsets");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("war_and_peace", |b| {
        b.iter(|| tokenizer.encode_with_offsets(black_box(&text), false))
    });

    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();
    let tokens = tokenizer.encode(&text, false).ids;

    let mut group = c.benchmark_group("decode");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("war_and_peace", |b| {
        b.iter(|| tokenizer.decode(black_box(&tokens)))
    });

    group.finish();
}

fn bench_count(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();

    let mut group = c.benchmark_group("count");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("war_and_peace", |b| {
        b.iter(|| tokenizer.count_tokens(black_box(&text)))
    });

    group.finish();
}

fn bench_pretokenize(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();
    let pretokenizer = tokenizer.pretokenizer().unwrap();

    let mut group = c.benchmark_group("pretokenize");
    group.throughput(Throughput::Bytes(text.len() as u64));

    group.bench_function("war_and_peace", |b| {
        b.iter(|| pretokenizer.split(black_box(&text)).count())
    });

    group.finish();
}

fn bench_bpe_only(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();
    let pretokenizer = tokenizer.pretokenizer().unwrap();

    // Pre-tokenize once, then benchmark just the BPE encoding
    let pieces: Vec<&str> = pretokenizer.split(&text).collect();
    let total_bytes: usize = pieces.iter().map(|p| p.len()).sum();

    let mut group = c.benchmark_group("bpe_only");
    group.throughput(Throughput::Bytes(total_bytes as u64));

    // Sequential - like the non-parallel path
    group.bench_function("sequential", |b| {
        b.iter(|| {
            pieces
                .iter()
                .flat_map(|piece| tokenizer.encoder().encode(black_box(piece.as_bytes())))
                .count()
        })
    });

    // Parallel - like the actual encode() path
    group.bench_function("parallel", |b| {
        b.iter(|| {
            let num_cpus = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(1);
            let chunk_size = (pieces.len() + num_cpus - 1) / num_cpus;
            std::thread::scope(|s| {
                let handles: Vec<_> = pieces
                    .chunks(chunk_size)
                    .map(|chunk| {
                        let encoder = tokenizer.encoder();
                        s.spawn(move || {
                            chunk.iter()
                                .flat_map(|piece| encoder.encode(piece.as_bytes()))
                                .count()
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum::<usize>()
            })
        })
    });

    group.finish();
}

fn bench_token_count(c: &mut Criterion) {
    let tokenizer = load_tokenizer();
    let text = load_text();

    let mut group = c.benchmark_group("token_count");

    // Small limit - should early terminate fast
    group.bench_function("limit_100", |b| {
        b.iter(|| tokenizer.token_count(black_box(&text)) > 100)
    });

    // Medium limit
    group.bench_function("limit_1000", |b| {
        b.iter(|| tokenizer.token_count(black_box(&text)) > 1000)
    });

    // Large limit - near full text
    group.bench_function("limit_100000", |b| {
        b.iter(|| tokenizer.token_count(black_box(&text)) > 100_000)
    });

    // Full count for comparison
    group.bench_function("full_count", |b| {
        b.iter(|| tokenizer.count_tokens(black_box(&text)))
    });

    group.finish();
}

criterion_group!(benches, bench_encode, bench_encode_with_offsets, bench_decode, bench_count, bench_pretokenize, bench_bpe_only, bench_token_count);
criterion_main!(benches);
