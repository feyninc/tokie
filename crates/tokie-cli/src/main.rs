//! CLI for the tokie 0.2 tokenizer.
//!
//! `encode`, `wrap`, `decode`, `batch`, `save`, and `open` are the calls.
//! `ids` prints one id list per length-prefixed text so a Python test can
//! compare against this binary. `bench` times one file.

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use clap::{Parser, Subcommand};
use tokie::{Specials, Tokenizer};

#[derive(Parser)]
#[command(name = "tokie", about = "Encode and decode with a tokie 0.2 tokenizer")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Encode one text, or a whole file, and print the ids.
    Encode {
        /// `.tokie` file or `tokenizer.json`
        tokenizer: PathBuf,
        #[arg(long)]
        text: Option<String>,
        #[arg(long)]
        file: Option<PathBuf>,
        /// Apply the single-sequence template.
        #[arg(long)]
        wrap: bool,
    },
    /// Print ids for length-prefixed texts on stdin.
    Ids {
        tokenizer: PathBuf,
        #[arg(long)]
        wrap: bool,
    },
    /// Decode ids to text.
    Decode {
        tokenizer: PathBuf,
        ids: Vec<u32>,
        /// Omit specials.
        #[arg(long)]
        skip: bool,
    },
    /// Write a `.tokie` file.
    Save {
        tokenizer: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Time encode of one file. One warmup, then the median of `--reps`.
    Bench {
        tokenizer: PathBuf,
        #[arg(long)]
        file: PathBuf,
        #[arg(long, default_value_t = 5)]
        reps: usize,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::Encode {
            tokenizer,
            text,
            file,
            wrap,
        } => {
            let mut tok = Tokenizer::open(&tokenizer).map_err(|err| err.to_string())?;
            let body = match (text, file) {
                (Some(text), None) => text,
                (None, Some(path)) => std::fs::read_to_string(&path).map_err(|err| err.to_string())?,
                _ => return Err("pass --text or --file".to_string()),
            };
            let mut ids = Vec::new();
            if wrap {
                tok.wrap(&body, &mut ids).map_err(|err| err.to_string())?;
            } else {
                tok.encode(&body, &mut ids).map_err(|err| err.to_string())?;
            }
            print_ids(&ids);
            Ok(())
        }
        Commands::Ids { tokenizer, wrap } => {
            let mut tok = Tokenizer::open(&tokenizer).map_err(|err| err.to_string())?;
            let texts = read_texts(std::io::stdin())?;
            let mut ids = Vec::new();
            let stdout = std::io::stdout();
            let mut out = stdout.lock();
            for text in &texts {
                ids.clear();
                if wrap {
                    tok.wrap(text, &mut ids).map_err(|err| err.to_string())?;
                } else {
                    tok.encode(text, &mut ids).map_err(|err| err.to_string())?;
                }
                writeln!(out, "{}", ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(" "))
                    .map_err(|err| err.to_string())?;
            }
            Ok(())
        }
        Commands::Decode {
            tokenizer,
            ids,
            skip,
        } => {
            let tok = Tokenizer::open(&tokenizer).map_err(|err| err.to_string())?;
            let mut text = String::new();
            let specials = if skip { Specials::Skip } else { Specials::Keep };
            tok.decode(&ids, &mut text, specials).map_err(|err| err.to_string())?;
            println!("{text}");
            Ok(())
        }
        Commands::Save { tokenizer, output } => {
            let tok = Tokenizer::open(&tokenizer).map_err(|err| err.to_string())?;
            tok.save(&output).map_err(|err| err.to_string())
        }
        Commands::Bench {
            tokenizer,
            file,
            reps,
        } => {
            let text = std::fs::read_to_string(&file).map_err(|err| err.to_string())?;
            let mut tok = Tokenizer::open(&tokenizer).map_err(|err| err.to_string())?;
            let mut ids = Vec::new();
            tok.encode(&text, &mut ids).map_err(|err| err.to_string())?;
            let mut samples = Vec::with_capacity(reps);
            for _ in 0..reps {
                ids.clear();
                let start = Instant::now();
                tok.encode(&text, &mut ids).map_err(|err| err.to_string())?;
                samples.push(start.elapsed().as_secs_f64());
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = samples[samples.len() / 2];
            let mb = text.len() as f64 / 1_000_000.0 / median;
            println!("{mb:.1} MB/s  {} ids  {} bytes", ids.len(), text.len());
            Ok(())
        }
    }
}

fn print_ids(ids: &[u32]) {
    for (i, id) in ids.iter().enumerate() {
        if i > 0 {
            print!(" ");
        }
        print!("{id}");
    }
    println!();
}

/// Each text is a decimal byte length, a newline, that many bytes, and a newline.
fn read_texts(mut input: impl Read) -> Result<Vec<String>, String> {
    let mut bytes = Vec::new();
    input.read_to_end(&mut bytes).map_err(|err| err.to_string())?;
    let mut texts = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let nl = bytes[at..]
            .iter()
            .position(|b| *b == b'\n')
            .ok_or("ids input ended inside a length")?;
        let len: usize = std::str::from_utf8(&bytes[at..at + nl])
            .map_err(|_| "length is not utf-8".to_string())?
            .parse()
            .map_err(|_| "length is not an integer".to_string())?;
        at += nl + 1;
        if at + len > bytes.len() {
            return Err("ids input ended inside a text".to_string());
        }
        let text = std::str::from_utf8(&bytes[at..at + len]).map_err(|_| "text is not utf-8".to_string())?;
        texts.push(text.to_string());
        at += len;
        if at < bytes.len() && bytes[at] == b'\n' {
            at += 1;
        }
    }
    Ok(texts)
}

#[cfg(test)]
mod tests {
    use super::read_texts;

    #[test]
    fn length_prefixed_texts_keep_newlines() {
        let raw = b"5\nhello\n11\nHello\nworld\n0\n\n";
        let texts = read_texts(&raw[..]).unwrap();
        assert_eq!(texts, ["hello", "Hello\nworld", ""]);
    }
}
