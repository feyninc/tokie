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
                        println!(
                            "    tokie: {:?}",
                            &m.tokie_ids[..m.tokie_ids.len().min(15)]
                        );
                        println!(
                            "    ref:   {:?}",
                            &m.reference_ids[..m.reference_ids.len().min(15)]
                        );
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
            if result.failed == 0 {
                println!("OK  {}/{} texts pass", result.passed, result.total);
            } else {
                println!("FAILED  {}/{} texts pass", result.passed, result.total);
                for m in &result.mismatches {
                    println!("  MISMATCH: \"{}\"", m.text);
                    println!(
                        "    tokie: {:?}",
                        &m.tokie_ids[..m.tokie_ids.len().min(15)]
                    );
                    println!(
                        "    ref:   {:?}",
                        &m.reference_ids[..m.reference_ids.len().min(15)]
                    );
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
