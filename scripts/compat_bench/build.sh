#!/usr/bin/env bash
# Build the comparison harness without putting an absolute external path in
# Cargo.toml. The manifest uses the relative path vendor/gigatoken; this script
# points that symlink at GIGATOKEN_DIR and records what it used.
#
# Usage:
#   scripts/compat_bench/build.sh [label]
#
# Environment:
#   GIGATOKEN_DIR   gigatoken checkout (default: sibling of the tokie repo root)
#   CARGO_TARGET_DIR optional target directory
#   BIN_DIR          where to copy the binary (default: /tmp/opencode/tokie-compat/bin)
#
# The caller must have a C linker (cc) on PATH.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LABEL="${1:-base-corrected}"
GIGATOKEN_DIR="${GIGATOKEN_DIR:-$(cd "$REPO_ROOT/.." && pwd)/gigatoken}"
TARGET_DIR="${CARGO_TARGET_DIR:-/tmp/opencode/tokie-compat/$LABEL-target}"
BIN_DIR="${BIN_DIR:-/tmp/opencode/tokie-compat/bin}"

if [[ ! -f "$GIGATOKEN_DIR/Cargo.toml" ]]; then
  echo "GIGATOKEN_DIR does not look like a checkout: $GIGATOKEN_DIR" >&2
  echo "set GIGATOKEN_DIR=/path/to/gigatoken" >&2
  exit 1
fi

GIGATOKEN_DIR="$(cd "$GIGATOKEN_DIR" && pwd)"
mkdir -p "$SCRIPT_DIR/vendor" "$TARGET_DIR" "$BIN_DIR"
ln -sfn "$GIGATOKEN_DIR" "$SCRIPT_DIR/vendor/gigatoken"

echo "gigatoken: $GIGATOKEN_DIR -> $SCRIPT_DIR/vendor/gigatoken" >&2
echo "target:    $TARGET_DIR" >&2
CARGO_TARGET_DIR="$TARGET_DIR" cargo build --release --manifest-path "$SCRIPT_DIR/Cargo.toml"

OUT="$BIN_DIR/compat-bench-$LABEL"
cp "$TARGET_DIR/release/compat-bench" "$OUT"
echo "binary:    $OUT" >&2
sha256sum "$OUT" >&2
