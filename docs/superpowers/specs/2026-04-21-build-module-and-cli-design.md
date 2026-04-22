# tokie build module + CLI design

## Summary

Add a `build` feature to the core `tokie` crate with functions for converting `tokenizer.json` to `.tkz`, verifying correctness against reference backends, and uploading to the `tokiers/` HF org. Add a `tokie-cli` crate that wires these functions to a CLI binary named `tokie`.

## Motivation

The build pipeline is currently scattered across 18 Rust examples and ad-hoc Python scripts, with the model list duplicated in three places (`regenerate_from_json.rs`, `upload_tkz_to_hf.py`, `hub.rs::tokiers_repo_name`). There is no single command to convert, verify, and upload a tokenizer. Moving the logic into library functions makes it reusable and testable, and the CLI gives a single entry point.

## Design

### `tokie::build` module (feature-gated)

Gated behind `feature = "build"`, which implies `hf` and pulls in `tokenizers` (HuggingFace Rust crate) and `tiktoken-rs` as optional dependencies.

Three public functions:

#### `convert`

```rust
pub fn convert(repo_id: &str, output: &Path) -> Result<ConvertResult, BuildError>
```

Downloads `tokenizer.json` from the HF repo, loads it via `Tokenizer::from_json()`, saves as `.tkz` via `to_file()`. Returns `ConvertResult { vocab_size: u32, file_size_bytes: u64 }`.

#### `verify`

```rust
pub fn verify(repo_id: &str, tkz_path: &Path) -> Result<VerifyResult, BuildError>
```

Loads the `.tkz` from disk and the reference tokenizer from HF. Auto-detects the reference backend:
- tiktoken-rs for repos that map to tiktoken encodings: `Xenova/gpt-4` (cl100k_base), `Xenova/gpt-4o` (o200k_base), `Xenova/text-davinci-003` (p50k_base). These repos don't have a `tokenizer.json` that HF tokenizers can load, so tiktoken-rs is the only viable reference.
- HuggingFace `tokenizers` crate for everything else (including `openai-community/gpt2`, which has a `tokenizer.json`)

Encodes a standard set of test texts with both backends, compares token IDs. Returns `VerifyResult { total: usize, passed: usize, failed: usize, mismatches: Vec<Mismatch> }` where `Mismatch` contains the text, tokie IDs, and reference IDs.

#### `upload`

```rust
pub fn upload(tkz_path: &Path, tokiers_name: &str, token: Option<&str>) -> Result<(), BuildError>
```

Uploads the `.tkz` file to `tokiers/{tokiers_name}/tokenizer.tkz` on HF Hub. Uses the provided token, or falls back to the HF token from environment/cache. Fails with a clear error if the user lacks write permissions to the `tokiers` org.

#### Error type

```rust
pub enum BuildError {
    Download(String),
    LoadJson(JsonLoadError),
    SaveTkz(SerdeError),
    Verification { result: VerifyResult },
    Upload(String),
    MissingToken,
}
```

### `tokie-cli` crate

A thin binary crate at `crates/tokie-cli/`. Depends on `tokie = { path = "../tokie", features = ["hf", "build"] }` and `clap`.

Binary name is `tokie` (set via `[[bin]]` in Cargo.toml), so users install with `cargo install tokie-cli` and get the `tokie` command.

#### CLI surface

```
tokie convert <repo_id> -o <output.tkz> [-v] [-u]
tokie verify <repo_id> [--tkz <path>]
```

**`tokie convert <repo_id> -o <path>`**
- Calls `build::convert(repo_id, output)`
- `-v` (verify): also calls `build::verify()` after conversion. Prints pass/fail summary.
- `-u` (upload): also calls `build::upload()`. Derives tokiers name from repo_id by stripping the org prefix (e.g., `meta-llama/Llama-3.2-1B` -> `Llama-3.2-1B`). `-u` implies `-v` — unverified tokenizers are never uploaded.
- Prints vocab size, file size, and verification results to stdout.

**`tokie verify <repo_id>`**
- If `--tkz <path>` is provided, verifies that local file against the reference backend.
- If no `--tkz`, downloads the `.tkz` from `tokiers/{name}` via Hub and verifies it.
- Prints per-text pass/fail and a summary.

### Dependency changes to `tokie` crate

```toml
[features]
build = ["hf", "dep:tokenizers", "dep:tiktoken-rs"]

[dependencies]
tokenizers = { version = "0.22", features = ["http"], optional = true }
tiktoken-rs = { version = "0.6", optional = true }
```

These move from `[dev-dependencies]` to optional `[dependencies]` gated on `build`. Existing examples that use them get `required-features = ["build"]`.

### Files to change

- `crates/tokie/Cargo.toml` — add `build` feature, move deps
- `crates/tokie/src/lib.rs` — add `#[cfg(feature = "build")] pub mod build;`
- `crates/tokie/src/build.rs` — new module with `convert`, `verify`, `upload`, types
- `crates/tokie-cli/Cargo.toml` — new crate
- `crates/tokie-cli/src/main.rs` — clap CLI wiring
- Existing examples — update `required-features` or remove if fully replaced

### What this does NOT change

- `from_pretrained` resolution logic in `hub.rs` (already updated separately to remove `tokenizer.json` fallback)
- The `tokiers_repo_name()` mapping in `hub.rs` (still used by inference path)
- The `upload_tkz_to_hf.py` script (can be deprecated later)
- The `regenerate_from_json.rs` batch example (can be replaced later with `tokie convert` in a loop)
