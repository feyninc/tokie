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
