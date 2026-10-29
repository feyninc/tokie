//! HuggingFace Hub integration for loading tokenizers.
//!
//! This module is only available when the `hf` feature is enabled:
//! ```toml
//! tokie = { version = "0.1", features = ["hf"] }
//! ```
//!
//! # Example
//! ```ignore
//! use tokie::Tokenizer;
//!
//! // Load from HuggingFace Hub
//! let tokenizer = Tokenizer::from_pretrained("gpt2")?;
//! let tokenizer = Tokenizer::from_pretrained("meta-llama/Llama-3.2-8B")?;
//!
//! // With options
//! let tokenizer = Tokenizer::from_pretrained_with_options(
//!     "gpt2",
//!     FromPretrainedOptions::default().revision("main"),
//! )?;
//! ```

use std::path::PathBuf;

use hf_hub::Repo;

use crate::hf::JsonLoadError;
use crate::serde::SerdeError;
use crate::Tokenizer;

/// Error type for `from_pretrained` operations.
#[derive(Debug)]
pub enum HubError {
    /// Failed to initialize the HuggingFace Hub API.
    ApiInit(hf_hub::api::sync::ApiError),
    /// Failed to download the tokenizer file.
    Download(hf_hub::api::sync::ApiError),
    /// Failed to load the tokenizer from JSON.
    Load(JsonLoadError),
    /// Failed to load the tokenizer from .tkz binary format.
    LoadBinary(SerdeError),
    /// The tokenizer.json file was not found in the repository.
    NotFound(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HubError::ApiInit(e) => write!(f, "failed to initialize HuggingFace Hub API: {}", e),
            HubError::Download(e) => write!(f, "failed to download tokenizer: {}", e),
            HubError::Load(e) => write!(f, "failed to load tokenizer: {}", e),
            HubError::LoadBinary(e) => write!(f, "failed to load .tkz tokenizer: {}", e),
            HubError::NotFound(repo) => {
                write!(f, "tokenizer not found in repository '{}'", repo)
            }
        }
    }
}

impl std::error::Error for HubError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HubError::ApiInit(e) => Some(e),
            HubError::Download(e) => Some(e),
            HubError::Load(e) => Some(e),
            HubError::LoadBinary(e) => Some(e),
            HubError::NotFound(_) => None,
        }
    }
}

impl From<JsonLoadError> for HubError {
    fn from(e: JsonLoadError) -> Self {
        HubError::Load(e)
    }
}

/// Options for `from_pretrained`.
#[derive(Debug, Clone, Default)]
pub struct FromPretrainedOptions {
    /// Git revision (branch, tag, or commit hash). Defaults to "main".
    pub revision: Option<String>,
    /// Custom cache directory. Defaults to HuggingFace cache (~/.cache/huggingface/hub).
    pub cache_dir: Option<PathBuf>,
    /// HuggingFace API token for private repositories.
    pub token: Option<String>,
}

impl FromPretrainedOptions {
    /// Set the git revision (branch, tag, or commit hash).
    pub fn revision(mut self, revision: impl Into<String>) -> Self {
        self.revision = Some(revision.into());
        self
    }

    /// Set a custom cache directory.
    pub fn cache_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.cache_dir = Some(path.into());
        self
    }

    /// Set the HuggingFace API token for private repositories.
    pub fn token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }
}

impl Tokenizer {
    /// Load a tokenizer from HuggingFace Hub.
    ///
    /// This first tries to download a `tokenizer.tkz` file (tokie's compact binary
    /// format) for faster loading. If not found, falls back to `tokenizer.json`.
    /// Files are cached locally for subsequent loads.
    ///
    /// # Arguments
    /// * `repo_id` - Repository ID (e.g., "gpt2", "meta-llama/Llama-3.2-8B")
    ///
    /// # Example
    /// ```ignore
    /// use tokie::Tokenizer;
    ///
    /// let tokenizer = Tokenizer::from_pretrained("gpt2")?;
    /// let tokens = tokenizer.encode("Hello, world!", false);
    /// ```
    pub fn from_pretrained(repo_id: impl AsRef<str>) -> Result<Self, HubError> {
        Self::from_pretrained_with_options(repo_id, FromPretrainedOptions::default())
    }

    /// Load a tokenizer from HuggingFace Hub with custom options.
    ///
    /// # Arguments
    /// * `repo_id` - Repository ID (e.g., "gpt2", "meta-llama/Llama-3.2-8B")
    /// * `options` - Configuration options (revision, cache_dir, token)
    ///
    /// # Example
    /// ```ignore
    /// use tokie::{Tokenizer, FromPretrainedOptions};
    ///
    /// let tokenizer = Tokenizer::from_pretrained_with_options(
    ///     "gpt2",
    ///     FromPretrainedOptions::default()
    ///         .revision("main")
    ///         .token("hf_xxx"),
    /// )?;
    /// ```
    pub fn from_pretrained_with_options(
        repo_id: impl AsRef<str>,
        options: FromPretrainedOptions,
    ) -> Result<Self, HubError> {
        let repo_id = repo_id.as_ref();

        // Fast path: a fully-local load with zero network round-trips.
        // If the hub disk cache already has tokenizer.json for this repo, load
        // the compiled .tkz we stored next to it (~5ms), or compile and store
        // it now. Network resolution below costs >100ms even fully warm
        // (etag checks plus a 404 probe for tokenizer.tkz).
        if options.revision.is_none() {
            let cache = match &options.cache_dir {
                Some(dir) => hf_hub::Cache::new(dir.clone()),
                None => hf_hub::Cache::default(),
            };
            let repo = hf_hub::Repo::model(repo_id.to_string());
            if let Some(local_json) = cache.repo(repo).get("tokenizer.json") {
                if let Some(tok) = load_or_build_compiled(&local_json) {
                    return Ok(tok);
                }
            }
        }

        // Build the API client
        let mut api_builder = hf_hub::api::sync::ApiBuilder::new();

        if let Some(cache_dir) = options.cache_dir {
            api_builder = api_builder.with_cache_dir(cache_dir);
        }

        if let Some(token) = options.token {
            api_builder = api_builder.with_token(Some(token));
        }

        let api = api_builder.build().map_err(HubError::ApiInit)?;

        // Build the repo reference
        let repo = if let Some(revision) = options.revision {
            Repo::with_revision(repo_id.to_string(), hf_hub::RepoType::Model, revision)
        } else {
            Repo::model(repo_id.to_string())
        };

        let repo_api = api.repo(repo);

        // Try tokenizer.tkz first (faster to load, smaller to download).
        // An unreadable .tkz (e.g. produced by a newer format version) falls
        // through to tokenizer.json instead of failing the whole load — old
        // clients must keep working when hub artifacts move ahead of them.
        if let Ok(tkz_path) = repo_api.get("tokenizer.tkz") {
            if let Ok(mut tokenizer) = Self::from_file(tkz_path) {
                // v13+ .tkz stores added tokens; older files need tokenizer.json
                if !tokenizer.added_tokens_serialized() {
                    load_added_tokens_from_json(&mut tokenizer, &repo_api);
                }
                return Ok(tokenizer);
            }
        }

        // Try pre-built .tkz from tokiers/ org (covers 60+ popular models)
        if let Some(tokiers_name) = tokiers_repo_name(repo_id) {
            let tokiers_repo = Repo::model(format!("tokiers/{tokiers_name}"));
            let tokiers_api = api.repo(tokiers_repo);
            if let Ok(tkz_path) = tokiers_api.get("tokenizer.tkz") {
                if let Ok(mut tokenizer) = Self::from_file(tkz_path) {
                    // v13+ .tkz stores added tokens; older files need tokenizer.json
                    if !tokenizer.added_tokens_serialized() {
                        load_added_tokens_from_json(&mut tokenizer, &repo_api);
                    }
                    return Ok(tokenizer);
                }
            }
        }

        // Fall back to tokenizer.json (and leave a compiled artifact behind so
        // the next load takes the fast path)
        let tokenizer_path = repo_api.get("tokenizer.json").map_err(HubError::Download)?;
        if let Some(tok) = load_or_build_compiled(&tokenizer_path) {
            return Ok(tok);
        }
        Self::from_json(tokenizer_path).map_err(HubError::Load)
    }
}

/// Added/special token metadata extracted from tokenizer.json while building
/// the compiled .tkz.
#[derive(Default)]
struct CompiledMeta {
    added: Vec<crate::tokenizer::AddedTokenSpec>,
    special: Vec<(String, crate::types::TokenId)>,
}

/// Basename of the compiled artifact cached next to a downloaded
/// tokenizer.json. The key must change whenever the artifact contents could:
/// crate version, .tkz format version, and — for builds from a git checkout —
/// a hash of this crate's sources (TOKIE_BUILD_DISCRIMINATOR, emitted by
/// build.rs; empty for crates.io builds, whose version already changes).
fn compiled_cache_basename() -> String {
    format!(
        "tokenizer.compiled-v{}-f{}{}",
        env!("CARGO_PKG_VERSION"),
        crate::serde::VERSION,
        env!("TOKIE_BUILD_DISCRIMINATOR"),
    )
}

/// TOKIE_NO_COMPILED_CACHE disables reading and writing compiled artifacts
/// (loads still work, straight from tokenizer.json). Unset, "" and "0" mean
/// enabled.
fn compiled_cache_disabled() -> bool {
    disables_compiled_cache(std::env::var("TOKIE_NO_COMPILED_CACHE").ok().as_deref())
}

fn disables_compiled_cache(value: Option<&str>) -> bool {
    matches!(value, Some(v) if !v.is_empty() && v != "0")
}

/// Load the compiled cache stored next to `json_path`, or build it from the
/// json and store it. Returns None if the json can't be loaded (caller falls
/// back to the network path). Cache writes are best-effort: a read-only cache
/// dir just means the fast path stays cold.
fn load_or_build_compiled(json_path: &std::path::Path) -> Option<Tokenizer> {
    let cache_disabled = compiled_cache_disabled();
    let tkz = json_path.with_file_name(format!("{}.tkz", compiled_cache_basename()));

    // v13 .tkz is self-contained (added/special tokens included)
    if !cache_disabled {
        if let Ok(tok) = Tokenizer::from_file(&tkz) {
            return Some(tok);
        }
    }

    // Build from json; extract added/special tokens from the same parse.
    let json_bytes = std::fs::read(json_path).ok()?;
    let mut tok = Tokenizer::from_json(json_path).ok()?;
    let mut m = CompiledMeta::default();
    if let Ok(data) = serde_json::from_slice::<serde_json::Value>(&json_bytes) {
        m.added = crate::hf::extract_added_token_specs(&data);
        m.special = m
            .added
            .iter()
            .filter(|t| t.special)
            .filter_map(|t| String::from_utf8(t.bytes.clone()).ok().map(|s| (s, t.id)))
            .collect();
    }
    if !m.added.is_empty() {
        tok.set_added_tokens(&m.added);
    }
    if !m.special.is_empty() {
        tok.set_special_tokens(m.special);
    }

    // Persist for next time (best effort, atomic-ish via temp + rename)
    if !cache_disabled {
        let tmp = tkz.with_extension("tkz.tmp");
        if tok.to_file(&tmp).is_ok() {
            let _ = std::fs::rename(&tmp, &tkz);
        }
    }

    Some(tok)
}

/// Try to load added tokens from tokenizer.json and set them on the tokenizer.
/// This is needed because .tkz format doesn't store added token info.
/// Silently does nothing if tokenizer.json isn't available or has no added tokens.
fn load_added_tokens_from_json(tokenizer: &mut Tokenizer, repo_api: &hf_hub::api::sync::ApiRepo) {
    let Ok(json_path) = repo_api.get("tokenizer.json") else { return };
    let Ok(json_bytes) = std::fs::read(&json_path) else { return };
    let Ok(data) = serde_json::from_slice::<serde_json::Value>(&json_bytes) else { return };

    let tokens = crate::hf::extract_added_token_specs(&data);
    let special: Vec<(String, crate::types::TokenId)> = tokens
        .iter()
        .filter(|t| t.special)
        .filter_map(|t| String::from_utf8(t.bytes.clone()).ok().map(|s| (s, t.id)))
        .collect();

    if !tokens.is_empty() {
        tokenizer.set_added_tokens(&tokens);
    }
    if !special.is_empty() {
        tokenizer.set_special_tokens(special);
    }
}

/// Map a HuggingFace repo ID to its pre-built tokiers/ repo name.
/// Returns None if no pre-built .tkz exists for this model.
fn tokiers_repo_name(repo_id: &str) -> Option<&'static str> {
    // Case-insensitive lookup
    let key = repo_id.to_ascii_lowercase();
    match key.as_str() {
        // Embedding models
        "alibaba-nlp/gte-qwen2-7b-instruct" => Some("gte-Qwen2-7B-instruct"),
        "baai/bge-base-en-v1.5" => Some("bge-base-en-v1.5"),
        "baai/bge-en-icl" => Some("bge-en-icl"),
        "baai/bge-large-en-v1.5" => Some("bge-large-en-v1.5"),
        "baai/bge-small-en-v1.5" => Some("bge-small-en-v1.5"),
        "cohere/cohere-embed-english-v3.0" => Some("Cohere-embed-english-v3.0"),
        "cohere/cohere-embed-english-light-v3.0" => Some("Cohere-embed-english-light-v3.0"),
        "cohere/cohere-embed-multilingual-v3.0" => Some("Cohere-embed-multilingual-v3.0"),
        "cohere/cohere-embed-multilingual-light-v3.0" => Some("Cohere-embed-multilingual-light-v3.0"),
        "intfloat/e5-small-v2" => Some("e5-small-v2"),
        "intfloat/e5-base-v2" => Some("e5-base-v2"),
        "intfloat/e5-large-v2" => Some("e5-large-v2"),
        "jinaai/jina-embeddings-v2-base-en" => Some("jina-embeddings-v2-base-en"),
        "jinaai/jina-embeddings-v2-base-code" => Some("jina-embeddings-v2-base-code"),
        "jinaai/jina-embeddings-v3" => Some("jina-embeddings-v3"),
        "jinaai/jina-embeddings-v4" => Some("jina-embeddings-v4"),
        "mixedbread-ai/mxbai-embed-large-v1" => Some("mxbai-embed-large-v1"),
        "mixedbread-ai/mxbai-embed-2d-large-v1" => Some("mxbai-embed-2d-large-v1"),
        "mixedbread-ai/mxbai-embed-xsmall-v1" => Some("mxbai-embed-xsmall-v1"),
        "mixedbread-ai/deepset-mxbai-embed-de-large-v1" => Some("deepset-mxbai-embed-de-large-v1"),
        "nomic-ai/nomic-embed-text-v1" => Some("nomic-embed-text-v1"),
        "qwen/qwen3-embedding-0.6b" => Some("Qwen3-Embedding-0.6B"),
        "qwen/qwen3-embedding-4b" => Some("Qwen3-Embedding-4B"),
        "qwen/qwen3-embedding-8b" => Some("Qwen3-Embedding-8B"),
        "sentence-transformers/all-minilm-l6-v2" => Some("all-MiniLM-L6-v2"),
        "sentence-transformers/all-minilm-l12-v2" => Some("all-MiniLM-L12-v2"),
        "sentence-transformers/all-mpnet-base-v2" => Some("all-mpnet-base-v2"),
        "thenlper/gte-small" => Some("gte-small"),
        "thenlper/gte-base" => Some("gte-base"),
        "thenlper/gte-large" => Some("gte-large"),
        "voyageai/voyage-3" => Some("voyage-3"),
        "voyageai/voyage-3-lite" => Some("voyage-3-lite"),
        "voyageai/voyage-3-large" => Some("voyage-3-large"),
        "voyageai/voyage-3.5" => Some("voyage-3.5"),
        "voyageai/voyage-3.5-lite" => Some("voyage-3.5-lite"),
        "voyageai/voyage-code-2" => Some("voyage-code-2"),
        "voyageai/voyage-code-3" => Some("voyage-code-3"),
        "voyageai/voyage-finance-2" => Some("voyage-finance-2"),
        "voyageai/voyage-law-2" => Some("voyage-law-2"),
        "voyageai/voyage-multilingual-2" => Some("voyage-multilingual-2"),
        "voyageai/voyage-multimodal-3" => Some("voyage-multimodal-3"),
        // Cross-encoders
        "cross-encoder/ms-marco-minilm-l-4-v2" => Some("ms-marco-MiniLM-L-4-v2"),
        "cross-encoder/ms-marco-minilm-l-6-v2" => Some("ms-marco-MiniLM-L-6-v2"),
        // Base models
        "bert-base-uncased" => Some("bert-base-uncased"),
        "facebookai/roberta-base" => Some("roberta-base"),
        "answerdotai/modernbert-base" => Some("ModernBERT-base"),
        "openai-community/gpt2" => Some("gpt2"),
        "xenova/gpt-4" => Some("cl100k"),
        "xenova/gpt-4o" => Some("o200k"),
        "meta-llama/llama-3.2-1b" => Some("Llama-3.2-1B"),
        "meta-llama/llama-4-scout-17b-16e" => Some("Llama-4-Scout-17B-16E"),
        "codellama/codellama-7b-hf" => Some("CodeLlama-7b-hf"),
        "mistralai/mistral-7b-v0.1" => Some("Mistral-7B-v0.1"),
        "mistralai/mistral-nemo-base-2407" => Some("Mistral-Nemo-Base-2407"),
        "mistralai/mixtral-8x7b-v0.1" => Some("Mixtral-8x7B-v0.1"),
        "microsoft/phi-2" => Some("phi-2"),
        "microsoft/phi-3-mini-4k-instruct" => Some("Phi-3-mini-4k-instruct"),
        "qwen/qwen2-7b" => Some("Qwen2-7B"),
        "google-t5/t5-base" => Some("t5-base"),
        "facebookai/xlm-roberta-base" => Some("xlm-roberta-base"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_cache_basename() {
        let base = compiled_cache_basename();
        let expected_prefix =
            format!("tokenizer.compiled-v{}-f{}", env!("CARGO_PKG_VERSION"), crate::serde::VERSION);
        assert!(base.starts_with(&expected_prefix), "unexpected basename: {base}");
        // The optional build discriminator is the only allowed suffix.
        let suffix = &base[expected_prefix.len()..];
        assert!(
            suffix.is_empty() || (suffix.starts_with("-b") && suffix.len() == 18),
            "unexpected discriminator suffix: {suffix:?}"
        );
    }

    #[test]
    fn test_disables_compiled_cache() {
        assert!(!disables_compiled_cache(None));
        assert!(!disables_compiled_cache(Some("")));
        assert!(!disables_compiled_cache(Some("0")));
        assert!(disables_compiled_cache(Some("1")));
        assert!(disables_compiled_cache(Some("true")));
    }

    #[test]
    fn test_tokiers_repo_name() {
        // Case-insensitive matching
        assert_eq!(tokiers_repo_name("BAAI/bge-base-en-v1.5"), Some("bge-base-en-v1.5"));
        assert_eq!(tokiers_repo_name("baai/bge-base-en-v1.5"), Some("bge-base-en-v1.5"));
        // Known models
        assert_eq!(tokiers_repo_name("sentence-transformers/all-MiniLM-L6-v2"), Some("all-MiniLM-L6-v2"));
        assert_eq!(tokiers_repo_name("openai-community/gpt2"), Some("gpt2"));
        assert_eq!(tokiers_repo_name("meta-llama/Llama-3.2-1B"), Some("Llama-3.2-1B"));
        // Unknown model
        assert_eq!(tokiers_repo_name("some-random/model"), None);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_from_pretrained_gpt2() {
        let tokenizer = Tokenizer::from_pretrained("gpt2").expect("Failed to load GPT-2");
        let tokens = tokenizer.encode("Hello, world!", false);
        assert!(!tokens.ids.is_empty());

        // Verify it produces expected tokens for GPT-2
        let decoded = tokenizer.decode(&tokens.ids).unwrap();
        assert_eq!(decoded, "Hello, world!");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_from_pretrained_with_revision() {
        let tokenizer = Tokenizer::from_pretrained_with_options(
            "gpt2",
            FromPretrainedOptions::default().revision("main"),
        )
        .expect("Failed to load GPT-2");

        let tokens = tokenizer.encode("Test", false);
        assert!(!tokens.is_empty());
    }
}
