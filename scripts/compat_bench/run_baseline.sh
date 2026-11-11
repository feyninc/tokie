#!/usr/bin/env bash
# Reproducible baseline: tokie vs gigatoken on the same OpenWebText sample.
#
# Warm runs: one process per (model, mode, threads), warmup then N alternating
# in-process repeats; median and range are in each JSON.
# Cold: one fresh process per sample, no warmup, 5 samples per side, engines
# alternated process-by-process. This is the first-pass/corpus-scan number.
#
# Run from the worktree root after building:
#   GIGATOKEN_DIR=/path/to/gigatoken \
#     CARGO_TARGET_DIR=/tmp/opencode/tokie-compat/base-corrected-target \
#     scripts/compat_bench/build.sh base-corrected
#   scripts/compat_bench/run_baseline.sh base-corrected
#
# build.sh creates the ignored vendor/gigatoken symlink. Results go under
# /tmp/opencode/tokie-compat/results.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABEL="${1:-base}"
ROOT="/tmp/opencode/tokie-compat"
BIN="$ROOT/bin/compat-bench-$LABEL"
RESULTS="$ROOT/results/$LABEL"
mkdir -p "$ROOT/bin" "$RESULTS"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary $BIN" >&2
  echo "copy the release compat-bench into $BIN or build and copy it" >&2
  exit 1
fi

# Performance profile makes the laptop's amd-pstate governor stop ramping
# during the first repeats. Restore whatever was active on exit.
PREV_PROFILE=""
if command -v powerprofilesctl >/dev/null 2>&1; then
  PREV_PROFILE="$(powerprofilesctl get || true)"
  powerprofilesctl set performance || true
  trap 'if [[ -n "$PREV_PROFILE" ]]; then powerprofilesctl set "$PREV_PROFILE" || true; fi' EXIT
fi

models=(gpt2 cl100k qwen2 o200k qwen3.5 deepseek)
single_cpu=8
batch1_cpu=8
batch8_cpus=4-11

run_warm() {
  local model="$1" mode="$2" threads="$3" cpus="$4" warmup="$5" reps="$6"
  local out="$RESULTS/warm_${model}_${mode}_${threads}t.json"
  echo "warm model=$model mode=$mode threads=$threads cpus=$cpus warmup=$warmup reps=$reps" >&2
  RAYON_NUM_THREADS="$threads" taskset -c "$cpus" "$BIN" \
    --model "$model" --mode "$mode" --engine both --threads "$threads" \
    --warmup "$warmup" --reps "$reps" --label "warm-$LABEL" \
    --json-out "$out" >/dev/null
}

run_cold() {
  local model="$1" mode="$2" threads="$3" cpus="$4" samples="$5"
  local tag="cold_${model}_${mode}_${threads}t"
  for ((rep = 0; rep < samples; rep++)); do
    if (( rep % 2 == 0 )); then
      order=(tokie giga)
    else
      order=(giga tokie)
    fi
    for engine in "${order[@]}"; do
      local out="$RESULTS/${tag}_${engine}_r${rep}.json"
      RAYON_NUM_THREADS="$threads" taskset -c "$cpus" "$BIN" \
        --model "$model" --mode "$mode" --engine "$engine" --threads "$threads" \
        --warmup 0 --reps 1 --label "cold-$LABEL" \
        --json-out "$out" >/dev/null
    done
  done
}

for model in "${models[@]}"; do
  run_warm "$model" single 1 "$single_cpu" 3 11
  run_warm "$model" batch 1 "$batch1_cpu" 6 15
  run_warm "$model" batch 8 "$batch8_cpus" 6 15

  run_cold "$model" single 1 "$single_cpu" 5
  run_cold "$model" batch 1 "$batch1_cpu" 5
  run_cold "$model" batch 8 "$batch8_cpus" 5
done

"$SCRIPT_DIR/summarize.py" --label "$LABEL" --results "$RESULTS" \
  --out "$RESULTS/summary.json"

echo "done: $RESULTS/summary.json" >&2
