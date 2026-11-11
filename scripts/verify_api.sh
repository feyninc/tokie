#!/usr/bin/env bash
# tokie API verification runner. Runs existing tests only; no repo edits.
# Usage: scripts/verify_api.sh [--out DIR] [--target-dir DIR] [--venv DIR]
# Env: TOKIE_VERIFY_OUT, TOKIE_VERIFY_TARGET, TOKIE_VERIFY_VENV, TOKIE_CC_BIN
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HUB="/home/bhavnick/.cache/huggingface/hub"
TK="/home/bhavnick/.cache/tokie-tokenizers"
OUT="${TOKIE_VERIFY_OUT:-/tmp/opencode/tokie-compat/verification}"
TARGET="${TOKIE_VERIFY_TARGET:-}"
VENV="${TOKIE_VERIFY_VENV:-}"
CC_BIN="${TOKIE_CC_BIN:-/nix/store/z4c6k0mrlkwl3s4w9ysxc8vq1wylm3ms-gcc-wrapper-15.3.0/bin}"

while (($#)); do
  case "$1" in
    -o|--out) OUT="${2:?--out needs DIR}"; shift 2 ;;
    --target-dir) TARGET="${2:?--target-dir needs DIR}"; shift 2 ;;
    --venv) VENV="${2:?--venv needs DIR}"; shift 2 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

TARGET="${TARGET:-$OUT/target}"
mkdir -p "$OUT" "$TARGET"
OUT="$(cd "$OUT" && pwd)"
TARGET="$(cd "$TARGET" && pwd)"
SUMMARY="$OUT/summary.txt"
: > "$SUMMARY"
FAIL=0
export CARGO_TERM_COLOR=never CARGO_TARGET_DIR="$TARGET"
PATH="$CC_BIN:$PATH"
export PATH
cd "$ROOT"

echo "repo=$ROOT" | tee -a "$SUMMARY"
echo "out=$OUT target=$TARGET" | tee -a "$SUMMARY"
echo "cc=$(command -v cc || echo missing)" | tee -a "$SUMMARY"

run() {
  local tag="$1"; shift
  local log="$OUT/$tag.log" rc=0
  { printf 'cwd=%s\n+ ' "$PWD"; printf '%q ' "$@"; printf '\n'; } | tee "$log"
  if ( "$@" ) 2>&1 | tee -a "$log"; then rc=0; else rc=$?; fi
  printf 'exit=%d %s\n' "$rc" "$tag" | tee -a "$log" "$SUMMARY"
  ((rc == 0)) || FAIL=1
}

check() {
  if [ ! -s "$1" ]; then echo "MISSING/EMPTY fixture: $1" >&2; MISSING=1; fi
}

MISSING=0
for p in \
  "$HUB/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json" \
  "$HUB/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json" \
  "$HUB/models--t5-small/snapshots/main/tokenizer.json" \
  "$ROOT/bench/normalize/data/t5.charsmap" \
  "$TK/kimi.tiktoken" \
  "$ROOT/../gigatoken/Cargo.toml" \
  "$ROOT/../tokenizers-v1/tokenizers/tk-encode/Cargo.toml" \
  "$ROOT/../tokenizers-v1/tokenizers/tk-serialize/Cargo.toml" \
  "$ROOT/../tokenizers-v1/tokenizers/tk-convert/Cargo.toml"
do
  check "$p"
done

while IFS='|' read -r name file cache; do
  local_path="$ROOT/bench/tokie/data/$file"
  if [ -f "$local_path" ]; then
    if [ ! -s "$local_path" ]; then echo "EMPTY local row $name: $local_path" >&2; MISSING=1; fi
  elif [ ! -s "$cache" ]; then
    echo "MISSING row $name: $local_path or $cache" >&2
    MISSING=1
  fi
done <<ROWS
gpt2|gpt2.json|$HUB/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json
cl100k|cl100k.json|$HUB/models--Xenova--gpt-4/snapshots/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json
o200k|o200k.json|$HUB/models--Xenova--gpt-4o/snapshots/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json
qwen2|qwen2.json|$HUB/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json
qwen|qwen.json|$HUB/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json
olmo|olmo.json|$TK/olmo.json
nemotron|nemotron.json|$TK/nemotron.json
kimi|kimi.json|$TK/kimi.json
smollm|smollm.json|$HUB/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71/tokenizer.json
deepseek|deepseek.json|$HUB/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json
bert-uncased|bert-uncased.json|$HUB/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json
t5|t5.json|$HUB/models--t5-small/snapshots/main/tokenizer.json
albert|albert.json|$HUB/models--albert-base-v2/snapshots/main/tokenizer.json
llama|llama.json|$HUB/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/main/tokenizer.json
ROWS

if [ -e "$ROOT/bench/tokie/data/bert-cased.json" ]; then
  echo "UNEXPECTED bert-cased fixture: baseline expects it absent" >&2
  MISSING=1
fi
if [ "$MISSING" -ne 0 ]; then
  echo "RESULT PRECHECK_FAIL" | tee -a "$SUMMARY"
  exit 2
fi

run normalie cargo test --locked -p normalie --tests
run pretokie cargo test --locked -p pretokie@0.2.0 --tests
run tokie cargo test --locked -p tokie --tests
run tokie-cli cargo test --locked -p tokie-cli --tests
run bench-oracles cargo test --locked --manifest-path bench/tokie/Cargo.toml --test oracles -- --nocapture --test-threads=1
run bench-ids cargo test --locked --manifest-path bench/tokie/Cargo.toml --test ids -- --nocapture --test-threads=1
run bench-normalize cargo test --locked --manifest-path bench/normalize/Cargo.toml --test oracles -- --nocapture --test-threads=1

checked=0
skipped=0
if [ -f "$OUT/bench-ids.log" ]; then
  checked=$(awk -F', ' '/^checked: /{n=NF} END{print n+0}' "$OUT/bench-ids.log")
  skipped=$(awk '/^skipped:$/{f=1;next} f&&/^(test |test result:|running |failures:|$)/{exit} f&&/^[A-Za-z0-9_.-]+: /{n++} END{print n+0}' "$OUT/bench-ids.log")
fi
echo "ids counts: checked=$checked/13 skipped=$skipped/2" | tee -a "$SUMMARY"
if [ "$checked" -ne 13 ] || [ "$skipped" -ne 2 ]; then
  echo "ids count audit FAILED" | tee -a "$SUMMARY"
  FAIL=1
fi

if [ -n "$VENV" ]; then
  PY="$VENV/bin/python"
  if [ ! -x "$PY" ]; then
    echo "venv python missing: $PY" >&2
    FAIL=1
  else
    run python-import "$PY" -c 'import tokie, pytest'
    run pytest "$PY" -m pytest -q -ra crates/tokie-python/tests
    pytest_skipped=$(awk 'match($0, /[0-9]+ skipped/) { n=substr($0, RSTART, RLENGTH)+0 } END { print n+0 }' "$OUT/pytest.log")
    if [ "$pytest_skipped" -ne 0 ]; then
      echo "pytest skipped=$pytest_skipped; skipped tests fail this gate" | tee -a "$SUMMARY"
      FAIL=1
    fi
  fi
fi

if ((FAIL)); then
  echo "RESULT FAIL" | tee -a "$SUMMARY"
  exit 1
fi
echo "RESULT PASS" | tee -a "$SUMMARY"
