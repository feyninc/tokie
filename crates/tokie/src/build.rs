//! Build tools for converting, verifying, and uploading .tkz tokenizers.
//!
//! This module is only available when the `build` feature is enabled:
//! ```toml
//! tokie = { version = "0.1", features = ["build"] }
//! ```

use std::path::Path;

use hf_hub::Repo;

use crate::hf::JsonLoadError;
use crate::serde::SerdeError;
use crate::Tokenizer;

#[derive(Debug)]
pub struct ConvertResult {
    pub vocab_size: usize,
    pub file_size_bytes: u64,
}

#[derive(Debug)]
pub struct Mismatch {
    pub text: String,
    pub tokie_ids: Vec<u32>,
    pub reference_ids: Vec<u32>,
}

#[derive(Debug)]
pub struct VerifyResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub mismatches: Vec<Mismatch>,
}

#[derive(Debug)]
pub enum BuildError {
    Download(String),
    LoadJson(JsonLoadError),
    SaveTkz(SerdeError),
    Verification { result: VerifyResult },
    Upload(String),
    HubInit(String),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Download(e) => write!(f, "download failed: {e}"),
            BuildError::LoadJson(e) => write!(f, "failed to load tokenizer.json: {e}"),
            BuildError::SaveTkz(e) => write!(f, "failed to save .tkz: {e}"),
            BuildError::Verification { result } => {
                write!(f, "verification failed: {}/{} texts mismatched", result.failed, result.total)
            }
            BuildError::Upload(e) => write!(f, "upload failed: {e}"),
            BuildError::HubInit(e) => write!(f, "HF Hub init failed: {e}"),
        }
    }
}

impl std::error::Error for BuildError {}

fn tiktoken_encoding_name(repo_id: &str) -> Option<&'static str> {
    match repo_id.to_ascii_lowercase().as_str() {
        "xenova/gpt-4" => Some("cl100k_base"),
        "xenova/gpt-4o" => Some("o200k_base"),
        "xenova/text-davinci-003" => Some("p50k_base"),
        _ => None,
    }
}

const VERIFY_TEXTS: &[&str] = &[
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models encode text into dense vector representations.",
    "Tokenization is the process of splitting text into smaller units called tokens.",
    "BGE, GTE, and E5 are popular embedding models for semantic search.",
    "The 18th century was a time of great change.",
    "user@example.com visited https://example.org/path?query=1",
    "I can't believe it's not butter! Don't you think so?",
];

/// Convert a HuggingFace repo's tokenizer.json to .tkz format.
pub fn convert(repo_id: &str, output: &Path) -> Result<ConvertResult, BuildError> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .build()
        .map_err(|e| BuildError::HubInit(e.to_string()))?;

    let repo = Repo::model(repo_id.to_string());
    let repo_api = api.repo(repo);

    let json_path = repo_api
        .get("tokenizer.json")
        .map_err(|e| BuildError::Download(e.to_string()))?;

    let tokenizer = Tokenizer::from_json(&json_path).map_err(BuildError::LoadJson)?;

    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    tokenizer.to_file(output).map_err(BuildError::SaveTkz)?;

    let file_size_bytes = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);

    Ok(ConvertResult {
        vocab_size: tokenizer.vocab_size(),
        file_size_bytes,
    })
}

/// Verify a .tkz file against a reference tokenizer backend.
///
/// Auto-detects whether to use tiktoken-rs or HF tokenizers based on repo_id.
pub fn verify(repo_id: &str, tkz_path: &Path) -> Result<VerifyResult, BuildError> {
    let tokie_tok =
        Tokenizer::from_file(tkz_path).map_err(BuildError::SaveTkz)?;

    let mut mismatches = Vec::new();
    let mut passed = 0;

    if let Some(encoding_name) = tiktoken_encoding_name(repo_id) {
        let tiktoken = match encoding_name {
            "cl100k_base" => tiktoken_rs::cl100k_base(),
            "o200k_base" => tiktoken_rs::o200k_base(),
            "p50k_base" => tiktoken_rs::p50k_base(),
            _ => unreachable!(),
        }
        .expect("failed to load tiktoken encoding");

        for text in VERIFY_TEXTS {
            let tokie_ids = tokie_tok.encode(text, false).ids;
            let ref_ids: Vec<u32> = tiktoken
                .encode_with_special_tokens(text)
                .into_iter()
                .map(|id| id as u32)
                .collect();

            if tokie_ids == ref_ids {
                passed += 1;
            } else {
                mismatches.push(Mismatch {
                    text: text.to_string(),
                    tokie_ids,
                    reference_ids: ref_ids,
                });
            }
        }
    } else {
        let hf_tok = tokenizers::Tokenizer::from_pretrained(repo_id, None)
            .map_err(|e| BuildError::Download(format!("HF tokenizer load failed: {e}")))?;

        for text in VERIFY_TEXTS {
            let tokie_ids = tokie_tok.encode(text, false).ids;
            let hf_encoding = hf_tok
                .encode(text.to_string(), false)
                .map_err(|e| BuildError::Download(format!("HF encode failed: {e}")))?;
            let ref_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            if tokie_ids == ref_ids {
                passed += 1;
            } else {
                mismatches.push(Mismatch {
                    text: text.to_string(),
                    tokie_ids,
                    reference_ids: ref_ids,
                });
            }
        }
    }

    let total = VERIFY_TEXTS.len();
    let failed = mismatches.len();

    Ok(VerifyResult {
        total,
        passed,
        failed,
        mismatches,
    })
}
