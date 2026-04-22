# Build Module + CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `build` feature to the tokie crate with convert/verify/upload functions, and a `tokie-cli` crate that exposes them as `tokie convert` and `tokie verify` CLI commands.

**Architecture:** The `build` module lives in the core `tokie` crate behind a `build` feature flag that pulls in `tokenizers` and `tiktoken-rs` as optional deps. The `tokie-cli` crate is a thin clap wrapper that calls into `tokie::build::*`. The build module reuses existing `Tokenizer::from_json()` and `to_file()` for conversion, and compares against HF tokenizers / tiktoken-rs for verification.

**Tech Stack:** Rust, clap (CLI), hf-hub (HF Hub API), tokenizers (HF reference), tiktoken-rs (tiktoken reference)

---

### Task 1: Add `build` feature flag and move dependencies

**Files:**
- Modify: `crates/tokie/Cargo.toml`

- [ ] **Step 1: Update Cargo.toml — add `build` feature and move deps**

In `crates/tokie/Cargo.toml`, add the `build` feature and move `tokenizers` and `tiktoken-rs` from `[dev-dependencies]` to optional `[dependencies]`:

```toml
[features]
default = []
hf = ["hf-hub"]
build = ["hf", "dep:tokenizers", "dep:tiktoken-rs"]

[dependencies]
# ... existing deps unchanged ...
tokenizers = { version = "0.22", features = ["http"], optional = true }
tiktoken-rs = { version = "0.6", optional = true }

[dev-dependencies]
criterion = "0.5"
# tokenizers and tiktoken-rs removed from here
```

Also update examples that use `tokenizers` or `tiktoken-rs` to require the `build` feature. Find all examples that import these crates:

- `convert_and_verify` — change `required-features` to `["build"]`
- `verify_embedding_models` — change `required-features` to `["build"]`
- `bench_vs_hf` — add `required-features = ["build"]` (currently missing from Cargo.toml)
- `check_models` — add `required-features = ["build"]` if it uses tokenizers

Leave examples that only use `tokie` and `hf` unchanged.

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p tokie --features build`
Expected: Compiles with no errors.

Run: `cargo check -p tokie`
Expected: Compiles without `tokenizers` or `tiktoken-rs`.

- [ ] **Step 3: Commit**

```bash
git add crates/tokie/Cargo.toml
git commit -m "feat: add build feature flag, move tokenizers/tiktoken-rs to optional deps"
```

---

### Task 2: Create `build` module with types and `convert` function

**Files:**
- Create: `crates/tokie/src/build.rs`
- Modify: `crates/tokie/src/lib.rs:58` (add module declaration)

- [ ] **Step 1: Add module declaration to lib.rs**

In `crates/tokie/src/lib.rs`, after line 59 (`mod hub;`), add:

```rust
#[cfg(feature = "build")]
pub mod build;
```

And after line 72 (`pub use hub::{FromPretrainedOptions, HubError};`), add:

```rust
#[cfg(feature = "build")]
pub use build::{BuildError, ConvertResult, VerifyResult, Mismatch};
```

- [ ] **Step 2: Create build.rs with types and convert function**

Create `crates/tokie/src/build.rs`:

```rust
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

/// Result of a successful conversion.
#[derive(Debug)]
pub struct ConvertResult {
    pub vocab_size: usize,
    pub file_size_bytes: u64,
}

/// A single verification mismatch.
#[derive(Debug)]
pub struct Mismatch {
    pub text: String,
    pub tokie_ids: Vec<u32>,
    pub reference_ids: Vec<u32>,
}

/// Result of a verification run.
#[derive(Debug)]
pub struct VerifyResult {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub mismatches: Vec<Mismatch>,
}

/// Errors from build operations.
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

    let file_size_bytes = std::fs::metadata(output)
        .map(|m| m.len())
        .unwrap_or(0);

    Ok(ConvertResult {
        vocab_size: tokenizer.vocab_size(),
        file_size_bytes,
    })
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p tokie --features build`
Expected: Compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/tokie/src/build.rs crates/tokie/src/lib.rs
git commit -m "feat: add build module with convert function"
```

---

### Task 3: Add `verify` function

**Files:**
- Modify: `crates/tokie/src/build.rs`

- [ ] **Step 1: Add the tiktoken repo-to-encoding mapping and test texts**

Append to `crates/tokie/src/build.rs`, before the `convert` function:

```rust
/// Repos that should be verified against tiktoken-rs instead of HF tokenizers.
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
```

- [ ] **Step 2: Add the verify function**

Append to `crates/tokie/src/build.rs`:

```rust
/// Verify a .tkz file against a reference tokenizer backend.
///
/// Auto-detects whether to use tiktoken-rs or HF tokenizers based on repo_id.
pub fn verify(repo_id: &str, tkz_path: &Path) -> Result<VerifyResult, BuildError> {
    let tokie_tok = Tokenizer::from_file(tkz_path)
        .map_err(|e| BuildError::SaveTkz(e))?;

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
            let ref_ids: Vec<u32> = tiktoken.encode_with_special_tokens(text)
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
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p tokie --features build`
Expected: Compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/tokie/src/build.rs
git commit -m "feat: add verify function with auto-detected reference backend"
```

---

### Task 4: Add `upload` function

**Files:**
- Modify: `crates/tokie/src/build.rs`

- [ ] **Step 1: Add the upload function**

Append to `crates/tokie/src/build.rs`:

```rust
/// Upload a .tkz file to the tokiers/ org on HuggingFace Hub.
///
/// Uses the HF Hub HTTP API directly (hf-hub crate is download-only).
/// If `token` is None, falls back to HF_TOKEN env var.
pub fn upload(tkz_path: &Path, tokiers_name: &str, token: Option<&str>) -> Result<(), BuildError> {
    if !tkz_path.exists() {
        return Err(BuildError::Upload(format!("file not found: {}", tkz_path.display())));
    }

    let token_str = token
        .map(|t| t.to_string())
        .or_else(|| std::env::var("HF_TOKEN").ok())
        .ok_or_else(|| BuildError::Upload(
            "no HF token found — pass --token or set HF_TOKEN env var".to_string(),
        ))?;

    let file_content = std::fs::read(tkz_path)
        .map_err(|e| BuildError::Upload(format!("failed to read {}: {e}", tkz_path.display())))?;

    let repo_id = format!("tokiers/{tokiers_name}");
    let url = format!(
        "https://huggingface.co/api/models/{}/upload/main/tokenizer.tkz",
        repo_id
    );

    let response = ureq::put(&url)
        .header("Authorization", &format!("Bearer {token_str}"))
        .content_type("application/octet-stream")
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
```

Note: `ureq` is a transitive dependency via `hf-hub`. If it's not directly accessible, add `ureq = "2"` to `[dependencies]` in `crates/tokie/Cargo.toml` gated on the `build` feature.

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p tokie --features build`
Expected: Compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add crates/tokie/src/build.rs
git commit -m "feat: add upload function for tokiers/ HF org"
```

---

### Task 5: Create `tokie-cli` crate with `convert` command

**Files:**
- Create: `crates/tokie-cli/Cargo.toml`
- Create: `crates/tokie-cli/src/main.rs`

- [ ] **Step 1: Create Cargo.toml**

Create `crates/tokie-cli/Cargo.toml`:

```toml
[package]
name = "tokie-cli"
version = "0.0.1"
edition = "2024"
description = "CLI for building, verifying, and uploading tokie tokenizers"
license = "MIT OR Apache-2.0"
repository = "https://github.com/chonkie-inc/tokie"

[[bin]]
name = "tokie"
path = "src/main.rs"

[dependencies]
tokie = { path = "../tokie", features = ["hf", "build"] }
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 2: Create main.rs with convert command**

Create `crates/tokie-cli/src/main.rs`:

```rust
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "tokie", about = "Build, verify, and upload tokie tokenizers")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a HuggingFace tokenizer.json to .tkz format
    Convert {
        /// HuggingFace repo ID (e.g., "openai-community/gpt2", "meta-llama/Llama-3.2-1B")
        repo_id: String,

        /// Output path for the .tkz file
        #[arg(short, long)]
        output: PathBuf,

        /// Verify the converted .tkz against the reference tokenizer
        #[arg(short, long)]
        verify: bool,

        /// Upload to tokiers/ org on HuggingFace Hub (implies --verify)
        #[arg(short, long)]
        upload: bool,

        /// HuggingFace API token (falls back to HF_TOKEN env or cached token)
        #[arg(long)]
        token: Option<String>,
    },
    /// Verify a .tkz tokenizer against the reference backend
    Verify {
        /// HuggingFace repo ID
        repo_id: String,

        /// Path to the .tkz file (if omitted, downloads from tokiers/)
        #[arg(long)]
        tkz: Option<PathBuf>,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert { repo_id, output, verify, upload, token } => {
            cmd_convert(&repo_id, &output, verify || upload, upload, token.as_deref());
        }
        Commands::Verify { repo_id, tkz } => {
            cmd_verify(&repo_id, tkz.as_deref());
        }
    }
}

fn cmd_convert(repo_id: &str, output: &PathBuf, verify: bool, upload: bool, token: Option<&str>) {
    print!("Converting {repo_id} ... ");

    match tokie::build::convert(repo_id, output) {
        Ok(result) => {
            println!(
                "OK  vocab={} size={:.1}KB",
                result.vocab_size,
                result.file_size_bytes as f64 / 1024.0
            );
        }
        Err(e) => {
            println!("FAILED: {e}");
            std::process::exit(1);
        }
    }

    if verify {
        print!("Verifying ... ");
        match tokie::build::verify(repo_id, output) {
            Ok(result) => {
                if result.failed == 0 {
                    println!("OK  {}/{} texts pass", result.passed, result.total);
                } else {
                    println!("FAILED  {}/{} texts pass", result.passed, result.total);
                    for m in &result.mismatches {
                        println!("  MISMATCH: \"{}\"", m.text);
                        println!("    tokie: {:?}", &m.tokie_ids[..m.tokie_ids.len().min(15)]);
                        println!("    ref:   {:?}", &m.reference_ids[..m.reference_ids.len().min(15)]);
                    }
                    std::process::exit(1);
                }
            }
            Err(e) => {
                println!("FAILED: {e}");
                std::process::exit(1);
            }
        }
    }

    if upload {
        let tokiers_name = repo_id.rsplit('/').next().unwrap_or(repo_id);
        print!("Uploading to tokiers/{tokiers_name} ... ");
        match tokie::build::upload(output, tokiers_name, token) {
            Ok(()) => println!("OK"),
            Err(e) => {
                println!("FAILED: {e}");
                std::process::exit(1);
            }
        }
    }
}

fn cmd_verify(repo_id: &str, tkz: Option<&PathBuf>) {
    let tkz_path;
    let tkz_ref: &std::path::Path;

    if let Some(path) = tkz {
        tkz_ref = path;
    } else {
        // Download from tokiers/ via from_pretrained
        let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
        let tmp_path = format!("/tmp/tokie-verify-{name}.tkz");
        print!("Downloading tokiers/{name} ... ");
        let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();
        let repo = hf_hub::Repo::model(format!("tokiers/{name}"));
        match api.repo(repo).get("tokenizer.tkz") {
            Ok(p) => {
                std::fs::copy(&p, &tmp_path).unwrap();
                println!("OK");
            }
            Err(e) => {
                println!("FAILED: {e}");
                std::process::exit(1);
            }
        }
        tkz_path = PathBuf::from(tmp_path);
        tkz_ref = &tkz_path;
    }

    print!("Verifying {repo_id} ... ");
    match tokie::build::verify(repo_id, tkz_ref) {
        Ok(result) => {
            if result.failed == 0 {
                println!("OK  {}/{} texts pass", result.passed, result.total);
            } else {
                println!("FAILED  {}/{} texts pass", result.passed, result.total);
                for m in &result.mismatches {
                    println!("  MISMATCH: \"{}\"", m.text);
                    println!("    tokie: {:?}", &m.tokie_ids[..m.tokie_ids.len().min(15)]);
                    println!("    ref:   {:?}", &m.reference_ids[..m.reference_ids.len().min(15)]);
                }
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("FAILED: {e}");
            std::process::exit(1);
        }
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo build -p tokie-cli`
Expected: Compiles and produces a `tokie` binary.

- [ ] **Step 4: Commit**

```bash
git add crates/tokie-cli/
git commit -m "feat: add tokie-cli crate with convert and verify commands"
```

---

### Task 6: Integration test — convert and verify gpt2

**Files:**
- No new files — this is a manual smoke test

- [ ] **Step 1: Test convert**

Run: `cargo run -p tokie-cli -- convert openai-community/gpt2 -o /tmp/gpt2-test.tkz -v`

Expected output (approximately):
```
Converting openai-community/gpt2 ... OK  vocab=50257 size=XXX.XKB
Verifying ... OK  8/8 texts pass
```

- [ ] **Step 2: Test verify standalone**

Run: `cargo run -p tokie-cli -- verify openai-community/gpt2 --tkz /tmp/gpt2-test.tkz`

Expected output:
```
Verifying openai-community/gpt2 ... OK  8/8 texts pass
```

- [ ] **Step 3: Test verify with Hub download**

Run: `cargo run -p tokie-cli -- verify openai-community/gpt2`

Expected output:
```
Downloading tokiers/gpt2 ... OK
Verifying openai-community/gpt2 ... OK  8/8 texts pass
```

- [ ] **Step 4: Test a BERT model**

Run: `cargo run -p tokie-cli -- convert bert-base-uncased -o /tmp/bert-test.tkz -v`

Expected: Converts and verifies successfully.

- [ ] **Step 5: Test a tiktoken model (cl100k)**

Run: `cargo run -p tokie-cli -- convert Xenova/gpt-4 -o /tmp/cl100k-test.tkz -v`

Expected: Converts and verifies against tiktoken-rs.

- [ ] **Step 6: Test error case — invalid repo**

Run: `cargo run -p tokie-cli -- convert nonexistent/model -o /tmp/bad.tkz`

Expected: Prints a clear error and exits with code 1.

- [ ] **Step 7: Commit any fixes from testing**

If any bugs were found and fixed during testing, commit them:

```bash
git add -A
git commit -m "fix: address issues found during integration testing"
```

---

### Task 7: Update examples to use `build` feature

**Files:**
- Modify: `crates/tokie/Cargo.toml` (example entries)

- [ ] **Step 1: Update example required-features**

In `crates/tokie/Cargo.toml`, update all examples that use `tokenizers` or `tiktoken-rs` to require the `build` feature:

For these examples, change `required-features = ["hf"]` to `required-features = ["build"]`:
- `convert_and_verify`
- `verify_embedding_models`
- `tiktoken_compat` (if it uses tiktoken-rs directly)
- `regenerate_from_json`
- `regenerate_all_v11`

For these examples that only need `hf`, leave unchanged:
- `from_pretrained`
- `test_gpt2`
- `test_llama4`
- `test_enwik8`
- `basic_usage`
- `padding_truncation`
- `batch_processing`
- `cross_encoder`

Check each example file's imports to determine which group it belongs to. Run:

```bash
grep -l 'use tokenizers\|use tiktoken_rs' crates/tokie/examples/*.rs
```

- [ ] **Step 2: Verify all examples compile**

Run: `cargo build -p tokie --features build --examples`
Expected: All examples compile.

Run: `cargo build -p tokie --features hf --examples`
Expected: Only non-build examples compile (build-gated ones are skipped).

- [ ] **Step 3: Commit**

```bash
git add crates/tokie/Cargo.toml
git commit -m "chore: update examples to use build feature flag"
```
