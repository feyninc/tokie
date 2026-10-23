//! Rows tokie will encode. The spec for a row is its `tokenizer.json`.

use tokie::cut::Cut;

pub struct Row {
    pub name: &'static str,
    pub file: &'static str,
    pub cut: Cut,
    /// A local `tokenizer.json` used when `data/` does not have `file`.
    pub cache: Option<&'static str>,
}

const HF: &str = "/home/bhavnick/.cache/huggingface/hub";

pub const ROWS: &[Row] = &[
    Row {
        name: "gpt2",
        file: "gpt2.json",
        cut: Cut::Gpt2,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json"
        )),
    },
    Row {
        name: "cl100k",
        file: "cl100k.json",
        cut: Cut::Cl100k,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--Xenova--gpt-4/snapshots/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json"
        )),
    },
    Row {
        name: "o200k",
        file: "o200k.json",
        cut: Cut::O200k,
        cache: Some(
            "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4o/snapshots/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json",
        ),
    },
    Row {
        name: "qwen2",
        file: "qwen2.json",
        cut: Cut::Voyage,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json"
        )),
    },
    Row {
        name: "qwen",
        file: "qwen.json",
        cut: Cut::Qwen,
        cache: Some(
            "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json",
        ),
    },
    Row {
        name: "olmo",
        file: "olmo.json",
        cut: Cut::Olmo,
        cache: Some("/home/bhavnick/.cache/tokie-tokenizers/olmo.json"),
    },
    Row {
        name: "nemotron",
        file: "nemotron.json",
        cut: Cut::Nemotron,
        cache: Some("/home/bhavnick/.cache/tokie-tokenizers/nemotron.json"),
    },
    Row {
        name: "kimi",
        file: "kimi.json",
        cut: Cut::Kimi,
        cache: Some("/home/bhavnick/.cache/tokie-tokenizers/kimi.json"),
    },
    Row {
        name: "smollm",
        file: "smollm.json",
        cut: Cut::SmolLM,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71/tokenizer.json"
        )),
    },
    Row {
        name: "deepseek",
        file: "deepseek.json",
        cut: Cut::DeepSeek,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json"
        )),
    },
    Row {
        name: "bert-uncased",
        file: "bert-uncased.json",
        cut: Cut::Bert,
        cache: Some(concat!(
            "/home/bhavnick/.cache/huggingface/hub/",
            "models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json"
        )),
    },
    Row {
        name: "bert-cased",
        file: "bert-cased.json",
        cut: Cut::Bert,
        cache: None,
    },
    Row {
        name: "t5",
        file: "t5.json",
        cut: Cut::T5,
        cache: Some(
            "/home/bhavnick/.cache/huggingface/hub/models--t5-small/snapshots/main/tokenizer.json",
        ),
    },
    Row {
        name: "albert",
        file: "albert.json",
        cut: Cut::T5,
        cache: Some(
            "/home/bhavnick/.cache/huggingface/hub/models--albert-base-v2/snapshots/main/tokenizer.json",
        ),
    },
    Row {
        name: "llama",
        file: "llama.json",
        cut: Cut::None,
        cache: Some(
            "/home/bhavnick/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/main/tokenizer.json",
        ),
    },
];

pub fn data_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

pub fn model_path(row: &Row) -> Option<std::path::PathBuf> {
    let local = data_dir().join(row.file);
    if local.is_file() {
        return Some(local);
    }
    let _ = HF;
    row.cache
        .map(std::path::PathBuf::from)
        .filter(|path| path.is_file())
}

/// Rank file for a scheme gigatoken cannot detect from `tokenizer.json`.
pub fn tiktoken_path(name: &str) -> Option<&'static str> {
    match name {
        "kimi" => Some("/home/bhavnick/.cache/tokie-tokenizers/kimi.tiktoken"),
        _ => None,
    }
}

pub const SAMPLES: &[&str] = &[
    "",
    "Hello world",
    "don't",
    "12345",
    "a12b",
    "Hello, World!",
    "<|endoftext|>",
    "The quick brown fox jumps over the lazy dog.",
    "中文",
    "cafe\u{0301}",
    "a/b",
    "Hello\nworld",
];
