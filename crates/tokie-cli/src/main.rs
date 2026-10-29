use std::path::PathBuf;

use clap::{Parser, Subcommand};
use tokie::build::VerifyResult;

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
    /// Manage tokie's local caches
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },
}

#[derive(Subcommand)]
enum CacheCommands {
    /// Delete compiled tokenizer artifacts (tokenizer.compiled-*.tkz) from the
    /// HuggingFace hub cache. Downloaded tokenizer.json files are untouched;
    /// artifacts are rebuilt on the next load.
    Clear {
        /// Cache directory to scan (defaults to the HuggingFace hub cache)
        #[arg(long)]
        cache_dir: Option<PathBuf>,
    },
}

const MAX_MISMATCHES_SHOWN: usize = 5;

fn print_verify_result(result: &VerifyResult) {
    if result.failed == 0 {
        println!("OK  {}/{} chunks pass", result.passed, result.total);
        return;
    }

    println!(
        "FAILED  {}/{} chunks pass ({} mismatches)",
        result.passed, result.total, result.failed
    );

    let shown = result.mismatches.len().min(MAX_MISMATCHES_SHOWN);
    for m in &result.mismatches[..shown] {
        println!("  MISMATCH: \"{}\"", m.text);

        // Find first divergence index
        let first_diff = m
            .tokie_ids
            .iter()
            .zip(m.reference_ids.iter())
            .position(|(a, b)| a != b)
            .unwrap_or(m.tokie_ids.len().min(m.reference_ids.len()));

        let start = first_diff.saturating_sub(2);
        let tokie_end = (first_diff + 5).min(m.tokie_ids.len());
        let ref_end = (first_diff + 5).min(m.reference_ids.len());

        println!(
            "    first diff at token {}: tokie[{}..{}]={:?}  ref[{}..{}]={:?}",
            first_diff,
            start,
            tokie_end,
            &m.tokie_ids[start..tokie_end],
            start,
            ref_end,
            &m.reference_ids[start..ref_end],
        );
    }

    let remaining = result.mismatches.len() - shown;
    if remaining > 0 {
        println!("  ... and {remaining} more mismatches");
    }
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Convert {
            repo_id,
            output,
            verify,
            upload,
            token,
        } => {
            cmd_convert(&repo_id, &output, verify || upload, upload, token.as_deref());
        }
        Commands::Verify { repo_id, tkz } => {
            cmd_verify(&repo_id, tkz);
        }
        Commands::Cache { command } => match command {
            CacheCommands::Clear { cache_dir } => {
                cmd_cache_clear(cache_dir);
            }
        },
    }
}

fn cmd_cache_clear(cache_dir: Option<PathBuf>) {
    let root = cache_dir.unwrap_or_else(|| hf_hub::Cache::default().path().clone());
    if !root.exists() {
        println!("cache directory {} does not exist, nothing to do", root.display());
        return;
    }

    let mut removed = 0usize;
    let mut bytes = 0u64;
    remove_compiled_artifacts(&root, &mut removed, &mut bytes);
    println!(
        "removed {removed} compiled artifact(s), freed {:.1} MiB from {}",
        bytes as f64 / (1024.0 * 1024.0),
        root.display()
    );
}

/// Recursively delete compiled tokenizer artifacts under `dir`. Symlinks are
/// never followed: compiled artifacts are always regular files (written via
/// temp + rename), and hub snapshot dirs symlink everything else to blobs.
fn remove_compiled_artifacts(dir: &std::path::Path, removed: &mut usize, bytes: &mut u64) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_dir() {
            remove_compiled_artifacts(&entry.path(), removed, bytes);
        } else if file_type.is_file() && is_compiled_artifact(&entry.file_name().to_string_lossy()) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            if std::fs::remove_file(entry.path()).is_ok() {
                *removed += 1;
                *bytes += size;
            }
        }
    }
}

fn is_compiled_artifact(name: &str) -> bool {
    name.starts_with("tokenizer.compiled-") && (name.ends_with(".tkz") || name.ends_with(".tkz.tmp"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_compiled_artifact() {
        assert!(is_compiled_artifact("tokenizer.compiled-v0.1.1-f13.tkz"));
        assert!(is_compiled_artifact("tokenizer.compiled-v0.1.1-f13-b1a2b3c4d5e6f708.tkz"));
        assert!(is_compiled_artifact("tokenizer.compiled-v0.1.1-f13.tkz.tmp"));
        assert!(!is_compiled_artifact("tokenizer.json"));
        assert!(!is_compiled_artifact("tokenizer.tkz"));
        assert!(!is_compiled_artifact("model.compiled-v1.tkz"));
    }

    #[test]
    fn test_remove_compiled_artifacts() {
        let root = std::env::temp_dir().join(format!("tokie-cache-clear-test-{}", std::process::id()));
        let snapshot = root.join("models--gpt2/snapshots/abc123");
        std::fs::create_dir_all(&snapshot).unwrap();
        let stale = snapshot.join("tokenizer.compiled-v0.1.0-f12.tkz");
        let tmp = snapshot.join("tokenizer.compiled-v0.1.0-f12.tkz.tmp");
        let json = snapshot.join("tokenizer.json");
        std::fs::write(&stale, b"stale").unwrap();
        std::fs::write(&tmp, b"tmp").unwrap();
        std::fs::write(&json, b"{}").unwrap();

        let mut removed = 0;
        let mut bytes = 0;
        remove_compiled_artifacts(&root, &mut removed, &mut bytes);

        assert_eq!(removed, 2);
        assert_eq!(bytes, 8);
        assert!(!stale.exists());
        assert!(!tmp.exists());
        assert!(json.exists(), "tokenizer.json must be untouched");

        std::fs::remove_dir_all(&root).unwrap();
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
                let failed = result.failed > 0;
                print_verify_result(&result);
                if failed {
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

fn cmd_verify(repo_id: &str, tkz: Option<PathBuf>) {
    let tkz_path = if let Some(path) = tkz {
        path
    } else {
        let name = repo_id.rsplit('/').next().unwrap_or(repo_id);
        print!("Downloading tokiers/{name} ... ");
        let api = hf_hub::api::sync::ApiBuilder::new().build().unwrap();
        let repo = hf_hub::Repo::model(format!("tokiers/{name}"));
        match api.repo(repo).get("tokenizer.tkz") {
            Ok(p) => {
                println!("OK");
                p
            }
            Err(e) => {
                println!("FAILED: {e}");
                std::process::exit(1);
            }
        }
    };

    print!("Verifying {repo_id} ... ");
    match tokie::build::verify(repo_id, &tkz_path) {
        Ok(result) => {
            let failed = result.failed > 0;
            print_verify_result(&result);
            if failed {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("FAILED: {e}");
            std::process::exit(1);
        }
    }
}
