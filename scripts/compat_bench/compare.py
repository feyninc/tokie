#!/usr/bin/env python3
"""Compare two separate compat-bench summary runs.

This is descriptive only. Separate runs cannot distinguish a small regression
from host drift; use run_paired.py for the authoritative before/after gate.

Each configuration is matched by (kind, model, mode, threads). The script
reports the median-time speedup of after vs before, and fails loudly if the
token count or id checksum changed, because that is a correctness regression,
not a perf result.

Usage:
  scripts/compat_bench/compare.py \
    /tmp/opencode/tokie-compat/results/base-corrected/summary.json \
    /tmp/opencode/tokie-compat/results/after/summary.json
"""

import argparse
import json
import sys


def index(summary):
    rows = {}
    for row in summary["rows"]:
        key = (row["kind"], row["model"], row["mode"], row["threads"], row["engine"])
        rows[key] = row
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    args = parser.parse_args()
    before = index(json.load(open(args.before)))
    after = index(json.load(open(args.after)))

    missing = sorted(set(before) ^ set(after))
    if missing:
        print(f"configuration sets differ: {missing}", file=sys.stderr)
        return 2

    bad = []
    print("| kind | model | mode | threads | engine | before MB/s | after MB/s | after/before time |")
    print("| --- | --- | --- | ---: | --- | ---: | ---: | ---: |")
    for key in sorted(before):
        old = before[key]
        new = after[key]
        if (old["tokens"], old["checksum"]) != (new["tokens"], new["checksum"]):
            bad.append((key, old["tokens"], old["checksum"], new["tokens"], new["checksum"]))
        ratio = old["median_sec"] / new["median_sec"]
        print(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | "
            f"{old['mb_per_s']:.1f} | {new['mb_per_s']:.1f} | {ratio:.3f} |"
        )

    if bad:
        print("\nID MISMATCH (not a valid perf comparison):", file=sys.stderr)
        for item in bad:
            print(item, file=sys.stderr)
        return 1
    print("\nAll token counts and id checksums match the baseline.")
    print("Descriptive only: use run_paired.py for the paired before/after gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
