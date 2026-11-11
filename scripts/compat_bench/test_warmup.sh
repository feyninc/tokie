#!/usr/bin/env bash
# CLI regression test for the warmup-count bug: `--warmup N` must execute N
# warmup encodes per engine, and `--warmup 0` must execute none.
#
# Usage: scripts/compat_bench/test_warmup.sh <compat-bench binary>

set -euo pipefail

BIN="${1:?usage: test_warmup.sh <compat-bench binary>}"
TMP="$(mktemp -d /tmp/opencode/tokie-compat-warmup-test.XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

run() {
  RAYON_NUM_THREADS=1 taskset -c 8 "$BIN" \
    --model gpt2 --mode single --engine both --threads 1 "$@"
}

run --warmup 3 --reps 1 --label warmup-3 --json-out "$TMP/warm3.json" \
  >"$TMP/w3.out" 2>"$TMP/w3.log"
total=$(grep -c '^warmup ' "$TMP/w3.log" || true)
tokie=$(grep -c '^warmup [0-9]*/3 tokie:' "$TMP/w3.log" || true)
giga=$(grep -c '^warmup [0-9]*/3 gigatoken:' "$TMP/w3.log" || true)
test "$total" -eq 6
test "$tokie" -eq 3
test "$giga" -eq 3

python3 - "$TMP/warm3.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["results"], data
for row in data["results"]:
    assert row["warmup"] == 3, row
    assert row["warmup_runs"] == 3, row
    assert row["reps"] == 1, row
PY

run --warmup 0 --reps 1 --label warmup-0 --json-out "$TMP/warm0.json" \
  >"$TMP/w0.out" 2>"$TMP/w0.log"
zero=$(grep -c '^warmup ' "$TMP/w0.log" || true)
test "$zero" -eq 0
python3 - "$TMP/warm0.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data["results"], data
for row in data["results"]:
    assert row["warmup"] == 0, row
    assert row["warmup_runs"] == 0, row
PY

set +e
run --warmup 0 --reps 0 >"$TMP/r0.out" 2>"$TMP/r0.log"
reps_rc=$?
set -e
test "$reps_rc" -ne 0
grep -q -- '--reps must be positive' "$TMP/r0.log"

echo "PASS: warmup 3 -> 3 events/engine; warmup 0 -> 0; reps 0 rejected"
