//! Emits TOKIE_BUILD_DISCRIMINATOR for the compiled-tokenizer cache filename
//! (see hub.rs). Builds from a git checkout get a hash of src/**/*.rs so that
//! semantic changes without a version bump can't serve stale compiled
//! artifacts. Non-git builds (crates.io packages) get an empty string, so
//! released cache filenames are unchanged — CARGO_PKG_VERSION already
//! discriminates releases.

use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src");

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let discriminator = if in_git_checkout(&manifest_dir) {
        format!("-b{:016x}", hash_sources(&manifest_dir.join("src")))
    } else {
        String::new()
    };
    println!("cargo:rustc-env=TOKIE_BUILD_DISCRIMINATOR={discriminator}");
}

fn in_git_checkout(start: &Path) -> bool {
    start.ancestors().any(|dir| dir.join(".git").exists())
}

fn hash_sources(src: &Path) -> u64 {
    let mut files = Vec::new();
    collect_rs_files(src, &mut files);
    files.sort();

    // FNV-1a over sorted relative paths and file contents.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut update = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for file in &files {
        let rel = file.strip_prefix(src).unwrap_or(file);
        update(rel.to_string_lossy().as_bytes());
        update(&fs::read(file).unwrap_or_default());
    }
    hash
}

fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else { continue };
        if file_type.is_dir() {
            collect_rs_files(&path, out);
        } else if file_type.is_file() && path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}
