#!/usr/bin/env python3
"""Record the environment and identities for a compat-bench baseline."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

TOKIE = Path(__file__).resolve().parents[2]
GIGA = Path(
    os.environ.get(
        "GIGATOKEN_DIR",
        TOKIE / "scripts" / "compat_bench" / "vendor" / "gigatoken",
    )
).resolve()
ROOT = Path("/tmp/opencode/tokie-compat")

MODELS = {
    "gpt2": "/home/bhavnick/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
    "cl100k": "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4/snapshots/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json",
    "o200k": "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4o/snapshots/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json",
    "qwen2": "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json",
    "qwen3.5": "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json",
    "deepseek": "/home/bhavnick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json",
}


def run(args, cwd=None):
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False).stdout.strip()


def run_raw(args, cwd=None):
    return subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=False
    ).stdout


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repo(path):
    return {
        "path": str(path),
        "head": run(["git", "rev-parse", "HEAD"], cwd=path),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=path),
        # Keep porcelain leading spaces: the original baseline metadata lost
        # the first line's leading space to a strip() in the old collector.
        # The gate normalizes (code, path) so both forms compare correctly.
        "status": run_raw(["git", "status", "--porcelain"], cwd=path)
        .rstrip("\n")
        .splitlines(),
    }


def source_sha256(root):
    """Hash Cargo.toml, Cargo.lock, and every source file, path included."""
    files = [root / "Cargo.toml"]
    if (root / "Cargo.lock").exists():
        files.append(root / "Cargo.lock")
    files.extend(sorted((root / "src").rglob("*.rs")))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="base")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    out = Path(args.out) if args.out else ROOT / f"baseline_meta_{args.label}.json"
    if args.label == "base-corrected" and out.exists():
        parser.error(
            "refusing to overwrite the immutable historical baseline metadata; "
            "use enrich_baseline_meta.py for a separate derivative"
        )

    corpus_manifest = json.loads((ROOT / "corpus" / "manifest.json").read_text())
    binary = ROOT / "bin" / f"compat-bench-{args.label}"
    meta = {
        "label": args.label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": platform.node(),
        "kernel": platform.release(),
        "nproc": os.cpu_count(),
        "cpu_model": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            "?",
        ),
        "loadavg": Path("/proc/loadavg").read_text().strip(),
        "power_profile": run(["powerprofilesctl", "get"]),
        "scaling_governor_cpu8": Path(
            "/sys/devices/system/cpu/cpu8/cpufreq/scaling_governor"
        ).read_text().strip()
        if Path("/sys/devices/system/cpu/cpu8/cpufreq/scaling_governor").exists()
        else "?",
        "rustc": run(["rustc", "--version"]),
        "cargo": run(["cargo", "--version"]),
        "tokie": repo(TOKIE),
        "gigatoken": repo(GIGA),
        "gigatoken_version": "0.10.0",
        "gigatoken_source_sha256": source_sha256(GIGA),
        "gigatoken_manifest_path": "scripts/compat_bench/vendor/gigatoken",
        "gigatoken_symlink_target": os.readlink(TOKIE / "scripts/compat_bench/vendor/gigatoken")
        if (TOKIE / "scripts/compat_bench/vendor/gigatoken").is_symlink()
        else None,
        "gigatoken_modified_hashes": {
            name: sha256(GIGA / name)
            for name in [
                "Cargo.toml",
                "src/lib.rs",
                "src/bpe/tiktoken.rs",
                "src/bpe/sentencepiece.rs",
                "src/normalize.rs",
            ]
            if (GIGA / name).exists()
        },
        "corpus": corpus_manifest,
        "tokenizers": {
            name: {"path": path, "bytes": os.path.getsize(path), "sha256": sha256(path)}
            for name, path in MODELS.items()
        },
        "bench_binary": {
            "path": str(binary),
            "bytes": binary.stat().st_size,
            "sha256": sha256(binary),
        },
        "build": {
            "target_dir": str(ROOT / f"{args.label}-target"),
            "profile": "release",
            "lto": "fat",
            "codegen_units": 1,
            "command": (
                "PATH=<gcc-wrapper>/bin:$PATH GIGATOKEN_DIR=<checkout> "
                f"CARGO_TARGET_DIR={ROOT}/{args.label}-target "
                f"scripts/compat_bench/build.sh {args.label}"
            ),
        },
        "timing": {
            "affinity": {
                "single": "taskset -c 8",
                "batch_1t": "taskset -c 8",
                "batch_8t": "taskset -c 4-11",
            },
            "rayon_num_threads": "set to the tokie thread count before each process",
            "warm": "3 warmups / 11 reps single; 6 warmups / 15 reps batch; alternating engine order",
            "cold": "fresh process per sample, 0 warmups, 5 samples per engine, alternating process order",
        },
        "oracle_availability": {
            "cargo_test_pretokie": "cargo test -p pretokie (self-contained fixtures)",
            "cargo_test_tokie": "cargo test -p tokie (tokie_file needs cached HF tokenizers; present)",
            "bench_tokie_oracles": (
                "bench/tokie/tests/oracles.rs needs sibling gigatoken/ and "
                "tokenizers-v1/ at ../../../; neither exists in this worktree"
            ),
        },
    }
    out.write_text(json.dumps(meta, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
