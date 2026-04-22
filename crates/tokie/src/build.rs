//! Build tools for converting, verifying, and uploading .tkz tokenizers.
//!
//! This module is only available when the `build` feature is enabled:
//! ```toml
//! tokie = { version = "0.1", features = ["build"] }
//! ```

use std::path::{Path, PathBuf};

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

const ENWIK8_URL: &str = "http://mattmahoney.net/dc/enwik8.zip";

/// Download and cache enwik8. Returns the path to the cached file.
fn ensure_enwik8() -> Result<PathBuf, BuildError> {
    let cache_dir = dirs_next::cache_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("tokie");
    let enwik8_path = cache_dir.join("enwik8");

    if enwik8_path.exists() {
        return Ok(enwik8_path);
    }

    std::fs::create_dir_all(&cache_dir).ok();

    let zip_path = cache_dir.join("enwik8.zip");
    let response = ureq::get(ENWIK8_URL)
        .call()
        .map_err(|e| BuildError::Download(format!("failed to download enwik8: {e}")))?;

    let mut bytes = Vec::new();
    response
        .into_reader()
        .read_to_end(&mut bytes)
        .map_err(|e| BuildError::Download(format!("failed to read enwik8 response: {e}")))?;
    std::fs::write(&zip_path, &bytes)
        .map_err(|e| BuildError::Download(format!("failed to write enwik8.zip: {e}")))?;

    // Extract the zip
    let file = std::fs::File::open(&zip_path)
        .map_err(|e| BuildError::Download(format!("failed to open enwik8.zip: {e}")))?;
    let mut archive = zip::ZipArchive::new(file)
        .map_err(|e| BuildError::Download(format!("failed to read enwik8.zip: {e}")))?;
    let mut entry = archive
        .by_index(0)
        .map_err(|e| BuildError::Download(format!("failed to extract enwik8: {e}")))?;
    let mut content = Vec::new();
    std::io::Read::read_to_end(&mut entry, &mut content)
        .map_err(|e| BuildError::Download(format!("failed to read enwik8 entry: {e}")))?;
    std::fs::write(&enwik8_path, &content)
        .map_err(|e| BuildError::Download(format!("failed to write enwik8: {e}")))?;

    // Clean up zip
    std::fs::remove_file(&zip_path).ok();

    Ok(enwik8_path)
}

/// Split text into chunks at line boundaries, roughly `chunk_size` bytes each.
fn split_into_chunks(text: &str, chunk_size: usize) -> Vec<&str> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let target = (start + chunk_size).min(text.len());
        // Snap to a char boundary
        let target = snap_to_char_boundary(text, target);
        // Find the nearest newline at or after target to avoid splitting mid-line
        let end = if target < text.len() {
            text[target..].find('\n').map_or(target, |i| target + i + 1)
        } else {
            target
        };
        if end <= start {
            break;
        }
        chunks.push(&text[start..end]);
        start = end;
    }

    chunks
}

fn snap_to_char_boundary(text: &str, idx: usize) -> usize {
    if idx >= text.len() {
        return text.len();
    }
    // Walk backwards to find a char boundary
    let mut i = idx;
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

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

/// Verify a .tkz file against a reference tokenizer backend using enwik8.
///
/// Downloads and caches enwik8 (~100MB), splits into ~1KB chunks, and compares
/// token IDs from tokie against the reference backend (HF tokenizers or tiktoken-rs).
pub fn verify(repo_id: &str, tkz_path: &Path) -> Result<VerifyResult, BuildError> {
    let tokie_tok = Tokenizer::from_file(tkz_path).map_err(BuildError::SaveTkz)?;

    let enwik8_path = ensure_enwik8()?;
    let raw = std::fs::read(&enwik8_path)
        .map_err(|e| BuildError::Download(format!("failed to read enwik8: {e}")))?;
    let text = String::from_utf8_lossy(&raw);
    let chunks = split_into_chunks(&text, 1024);

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

        for chunk in &chunks {
            let tokie_ids = tokie_tok.encode(chunk, false).ids;
            let ref_ids: Vec<u32> = tiktoken
                .encode_with_special_tokens(chunk)
                .into_iter()
                .map(|id| id as u32)
                .collect();

            if tokie_ids == ref_ids {
                passed += 1;
            } else {
                mismatches.push(Mismatch {
                    text: chunk[..snap_to_char_boundary(chunk, 80)].to_string(),
                    tokie_ids,
                    reference_ids: ref_ids,
                });
            }
        }
    } else {
        let hf_tok = tokenizers::Tokenizer::from_pretrained(repo_id, None)
            .map_err(|e| BuildError::Download(format!("HF tokenizer load failed: {e}")))?;

        for chunk in &chunks {
            let tokie_ids = tokie_tok.encode(chunk, false).ids;
            let hf_encoding = hf_tok
                .encode(chunk.to_string(), false)
                .map_err(|e| BuildError::Download(format!("HF encode failed: {e}")))?;
            let ref_ids: Vec<u32> = hf_encoding.get_ids().to_vec();

            if tokie_ids == ref_ids {
                passed += 1;
            } else {
                mismatches.push(Mismatch {
                    text: chunk[..snap_to_char_boundary(chunk, 80)].to_string(),
                    tokie_ids,
                    reference_ids: ref_ids,
                });
            }
        }
    }

    let total = chunks.len();
    let failed = mismatches.len();

    Ok(VerifyResult {
        total,
        passed,
        failed,
        mismatches,
    })
}

/// Upload a .tkz file to the tokiers/ org on HuggingFace Hub.
///
/// Uses the HF Hub HTTP API directly (hf-hub crate is download-only).
/// If `token` is None, falls back to HF_TOKEN env var.
pub fn upload(tkz_path: &Path, tokiers_name: &str, token: Option<&str>) -> Result<(), BuildError> {
    if !tkz_path.exists() {
        return Err(BuildError::Upload(format!(
            "file not found: {}",
            tkz_path.display()
        )));
    }

    let token_str = token
        .map(|t| t.to_string())
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .ok_or_else(|| {
            BuildError::Upload(
                "no HF token found — pass --token or set HF_TOKEN env var".to_string(),
            )
        })?;

    let file_content = std::fs::read(tkz_path)
        .map_err(|e| BuildError::Upload(format!("failed to read {}: {e}", tkz_path.display())))?;

    let repo_id = format!("tokiers/{tokiers_name}");
    let url = format!(
        "https://huggingface.co/api/models/{repo_id}/upload/main/tokenizer.tkz",
    );

    let response = ureq::put(&url)
        .set("Authorization", &format!("Bearer {token_str}"))
        .set("Content-Type", "application/octet-stream")
        .send_bytes(&file_content)
        .map_err(|e| BuildError::Upload(format!("HTTP upload to {repo_id} failed: {e}")))?;

    if response.status() >= 400 {
        return Err(BuildError::Upload(format!(
            "upload returned HTTP {}: check your token has write access to tokiers/ org",
            response.status()
        )));
    }

    Ok(())
}
