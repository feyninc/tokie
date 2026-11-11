#!/usr/bin/env python3
"""Combine initial and focused-confirmation paired summaries.

Only configurations present in both summaries are combined. The combined
statistics use the same strict sign-test policy as run_paired.py. MB/s fields
use the actual corpus input size: 20,974,828 bytes for batch and 20,979,014 for
single. Timing arrays, checksums, speedups, sign tests, and verdicts are
unchanged by this units derivation.

Usage:
  scripts/compat_bench/combine_paired.py \
    --initial /tmp/opencode/tokie-compat/paired/final-rust/summary.json \
    --confirm /tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/summary.json \
    --out /tmp/opencode/tokie-compat/paired/final-rust-confirm-deepseek/combined_summary.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_paired", HERE / "run_paired.py")
run_paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_paired)


def rows(result, side, input_bytes):
    return [
        {
            "median_sec": seconds,
            "tokens": result["tokens"],
            "checksum": result["checksum"],
            "input_bytes": input_bytes,
        }
        for seconds in result[f"{side}_secs"]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-bytes", type=int, default=20_974_828)
    parser.add_argument("--single-bytes", type=int, default=20_979_014)
    args = parser.parse_args()
    input_bytes_for = {"batch": args.batch_bytes, "single": args.single_bytes}

    initial = json.loads(Path(args.initial).read_text())
    confirm = json.loads(Path(args.confirm).read_text())
    first = {row["config"]: row for row in initial["results"]}
    second = {row["config"]: row for row in confirm["results"]}
    overlap = sorted(set(first) & set(second))
    if not overlap:
        raise SystemExit("no overlapping configurations")

    combined = []
    failed = []
    inputs = {}
    for config in overlap:
        a = first[config]
        b = second[config]
        mode = config.split(":")[2]
        if mode not in input_bytes_for:
            raise SystemExit(f"no input byte count for mode {mode!r} in {config}")
        input_bytes = input_bytes_for[mode]
        inputs[config] = input_bytes
        result = run_paired.summarize_config(
            rows(a, "before", input_bytes) + rows(b, "before", input_bytes),
            rows(a, "after", input_bytes) + rows(b, "after", input_bytes),
            alpha=args.alpha,
            noise_tolerance=None,
            bootstrap_iterations=args.bootstrap,
            bootstrap_seed=args.seed,
        )
        if not result["gate_pass"]:
            failed.append(config)
        combined.append(
            {
                "config": config,
                "initial_pairs": a["pairs"],
                "confirm_pairs": b["pairs"],
                "combined_pairs": result["pairs"],
                "input_bytes": input_bytes,
                "combined": result,
                "gate_pass": result["gate_pass"],
                "verdict": result["verdict"],
            }
        )

    output = {
        "initial": args.initial,
        "confirm": args.confirm,
        "alpha": args.alpha,
        "policy": "same strict no-statistically-supported-slowdown sign test",
        "input_bytes": input_bytes_for,
        "input_bytes_by_config": inputs,
        "failed_configs": failed,
        "gate_pass": not failed,
        "results": combined,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "overlap": overlap,
        "failed_configs": failed,
        "gate_pass": not failed,
    }, indent=2))


if __name__ == "__main__":
    main()
