#!/usr/bin/env python3
"""Create a separate enriched-before metadata derivative.

This does NOT modify the historical baseline metadata. It reads the original
JSON bytes, records their SHA-256, verifies the current gigatoken checkout
against the original recorded HEAD/status/modified-file hashes, and only then
computes a full source-tree hash for the derivative.

The computed full hash is explicitly labeled as "computed later after
verification". It is not presented as a hash that existed at baseline build
time. Original environment fields, timestamps, tokie status, corpus identity,
and binary hash are copied unchanged.

Usage:
  scripts/compat_bench/enrich_baseline_meta.py \
    --source-meta /tmp/opencode/tokie-compat/baseline_meta_base-corrected.json \
    --gigatoken-dir /home/bhavnick/Workspaces/gigatoken \
    --out /tmp/opencode/tokie-compat/baseline_meta_base-corrected.enriched.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args, cwd):
    return subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=False
    ).stdout.strip()


def run_raw(args, cwd):
    return subprocess.run(
        args, cwd=cwd, text=True, capture_output=True, check=False
    ).stdout


def source_sha256(root):
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


def parse_status(line):
    """Parse a porcelain line to (code, path), tolerating a stripped first space.

    The historical collector used stdout.strip(), which removed the leading
    space from only the first status line. Treat this as a known formatting
    defect, not as a source difference. Codes and paths must still match.
    """
    parts = line.strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    code, path = parts
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[1].strip()
    return (code, path)


def normalized_status(lines):
    parsed = [parse_status(line) for line in lines]
    return sorted(item for item in parsed if item is not None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-meta", required=True)
    parser.add_argument(
        "--gigatoken-dir",
        default=os.environ.get(
            "GIGATOKEN_DIR", "/home/bhavnick/Workspaces/gigatoken"
        ),
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_meta = Path(args.source_meta).resolve()
    out = Path(args.out).resolve()
    giga = Path(args.gigatoken_dir).resolve()
    if source_meta == out:
        parser.error("--out must differ from --source-meta; the original is immutable")
    if out.exists() and not args.force:
        parser.error(f"{out} exists; use a new path or --force for the derivative")

    original_bytes = source_meta.read_bytes()
    original_sha = sha256_bytes(original_bytes)
    meta = json.loads(original_bytes)
    recorded = meta.get("gigatoken") or {}
    if not meta.get("gigatoken"):
        parser.error("source metadata has no gigatoken identity")
    if meta.get("gigatoken_source_sha256") is not None:
        parser.error(
            "source metadata already has a full source hash; do not enrich it again"
        )

    head = run(["git", "rev-parse", "HEAD"], giga)
    status = (
        run_raw(["git", "status", "--porcelain"], giga)
        .rstrip("\n")
        .splitlines()
    )
    recorded_status = recorded.get("status") or []
    normalized_current = normalized_status(status)
    normalized_recorded = normalized_status(recorded_status)
    if head != recorded.get("head"):
        parser.error(f"gigatoken HEAD changed: current {head}, recorded {recorded.get('head')}")
    if normalized_current != normalized_recorded:
        parser.error(
            "gigatoken status changed; refusing to mix current tree with the "
            "historical baseline metadata"
        )

    hashes = meta.get("gigatoken_modified_hashes")
    if hashes is None:
        parser.error("source metadata has no modified-file hashes")
    current_hashes = {name: sha256_file(giga / name) for name in hashes}
    if current_hashes != hashes:
        parser.error("gigatoken modified-file hashes changed; refusing to enrich")

    dirty_paths = [path for _, path in normalized_current]
    dirty_set = set(dirty_paths)
    hashed_set = set(hashes)
    missing = sorted(dirty_set - hashed_set)
    extra = sorted(hashed_set - dirty_set)
    if missing or extra:
        parser.error(
            "modified-file manifest does not cover the recorded dirty status; "
            f"missing={missing} extra={extra}"
        )

    binary = meta.get("bench_binary") or {}
    binary_path = Path(binary.get("path", ""))
    if not binary_path.is_file():
        parser.error(f"preserved baseline binary is missing: {binary_path}")
    binary_sha = sha256_file(binary_path)
    if binary_sha != binary.get("sha256"):
        parser.error(
            f"preserved baseline binary hash changed: {binary_sha} != {binary.get('sha256')}"
        )

    full_source = source_sha256(giga)
    derived = dict(meta)
    derived["gigatoken_source_sha256"] = full_source
    derived["gigatoken_source_sha256_provenance"] = {
        "kind": "computed_later_after_verification",
        "not_a_historical_build_time_hash": True,
        "source_meta_path": str(source_meta),
        "source_meta_sha256": original_sha,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "verified_gigatoken_head": head,
        "recorded_gigatoken_status_raw": recorded_status,
        "verified_gigatoken_status_raw": status,
        "recorded_gigatoken_status_normalized": normalized_recorded,
        "verified_gigatoken_status_normalized": normalized_current,
        "status_comparison": {
            "raw_equal": status == recorded_status,
            "normalized_equal": normalized_current == normalized_recorded,
            "note": (
                "The original collector stripped the leading space from only "
                "the first porcelain line. Comparison uses (code, path) pairs; "
                "HEAD and every modified/untracked file hash also match."
            ),
        },
        "verified_gigatoken_modified_hashes": hashes,
        "dirty_manifest_coverage": {
            "complete": True,
            "dirty_paths": sorted(dirty_set),
            "hashed_paths": sorted(hashed_set),
            "missing": missing,
            "extra": extra,
        },
        "source_files_hashed": [
            "Cargo.toml",
            "Cargo.lock",
            "src/**/*.rs",
        ],
        "note": (
            "The original metadata is immutable and is identified by "
            "source_meta_sha256. The full source hash was computed from the "
            "current checkout only after its HEAD, dirty status, and every "
            "modified/untracked file hash matched the original record. This is "
            "an enriched derivative, not a historical baseline-build hash."
        ),
    }
    derived["enriched_from_original"] = {
        "path": str(source_meta),
        "sha256": original_sha,
        "bytes": len(original_bytes),
    }

    out.write_text(json.dumps(derived, indent=2) + "\n")
    if sha256_bytes(source_meta.read_bytes()) != original_sha:
        raise SystemExit("original metadata changed during enrichment")
    print(f"original:  {source_meta}")
    print(f"sha256:    {original_sha} (unchanged)")
    print(f"enriched:  {out}")
    print(f"source:    {full_source}")
    print("coverage:  complete; HEAD/status/modified hashes matched")


if __name__ == "__main__":
    main()
