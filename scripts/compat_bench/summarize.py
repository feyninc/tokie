#!/usr/bin/env python3
"""Aggregate compat-bench JSON files into a baseline summary and markdown."""

import argparse
import glob
import json
import re
import statistics
import time
from pathlib import Path


def median_range(values):
    ordered = sorted(values)
    return statistics.median(ordered), ordered[0], ordered[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="base")
    parser.add_argument("--results", default="/tmp/opencode/tokie-compat/results/base")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    results_dir = Path(args.results)

    # warm_<model>_<mode>_<threads>t.json
    warm_pattern = re.compile(r"warm_(.+)_(single|batch)_(\d+)t\.json$")
    cold_pattern = re.compile(r"cold_(.+)_(single|batch)_(\d+)t_(tokie|giga)_r(\d+)\.json$")

    groups = {}

    def key(kind, model, mode, threads, engine):
        return (kind, model, mode, int(threads), engine)

    for path in sorted(results_dir.glob("warm_*.json")):
        match = warm_pattern.match(path.name)
        if not match:
            continue
        model, mode, threads = match.groups()
        data = json.loads(path.read_text())
        for row in data["results"]:
            keyed = key("warm", model, mode, threads, row["engine"])
            entry = groups.setdefault(
                keyed,
                {
                    "kind": "warm",
                    "model": model,
                    "mode": mode,
                    "threads": int(threads),
                    "engine": row["engine"],
                    "samples": [],
                    "inputs": set(),
                    "tokens": set(),
                    "checksums": set(),
                    "rows": set(),
                    "cpus": set(),
                    "warmup": row["warmup"],
                    "warmup_runs": set(),
                    "files": [],
                },
            )
            entry["warmup_runs"].add(row.get("warmup_runs", 0))
            entry["samples"].extend(row["times_sec"])
            entry["inputs"].add(row["input_bytes"])
            entry["tokens"].add(row["tokens"])
            entry["checksums"].add(row["checksum"])
            entry["rows"].add(row["rows"])
            entry["cpus"].add(row["cpus_allowed"])
            entry["files"].append(path.name)

    for path in sorted(results_dir.glob("cold_*.json")):
        match = cold_pattern.match(path.name)
        if not match:
            continue
        model, mode, threads, engine, _ = match.groups()
        if engine == "giga":
            engine = "gigatoken"
        data = json.loads(path.read_text())
        row = data["results"][0]
        keyed = key("cold", model, mode, threads, engine)
        entry = groups.setdefault(
            keyed,
            {
                "kind": "cold",
                "model": model,
                "mode": mode,
                "threads": int(threads),
                "engine": engine,
                "samples": [],
                "inputs": set(),
                "tokens": set(),
                "checksums": set(),
                "rows": set(),
                "cpus": set(),
                "warmup": 0,
                "warmup_runs": set(),
                "files": [],
            },
        )
        entry["warmup_runs"].add(0)
        entry["samples"].append(row["median_sec"])
        entry["inputs"].add(row["input_bytes"])
        entry["tokens"].add(row["tokens"])
        entry["checksums"].add(row["checksum"])
        entry["rows"].add(row["rows"])
        entry["cpus"].add(row["cpus_allowed"])
        entry["files"].append(path.name)

    rows = []
    for (_kind, model, mode, threads, engine), entry in sorted(groups.items()):
        assert len(entry["inputs"]) == 1, (model, mode, engine, entry["inputs"])
        assert len(entry["tokens"]) == 1, (model, mode, engine, entry["tokens"])
        assert len(entry["checksums"]) == 1, (model, mode, engine, entry["checksums"])
        assert len(entry["rows"]) == 1
        assert len(entry["warmup_runs"]) == 1
        median, low, high = median_range(entry["samples"])
        input_bytes = next(iter(entry["inputs"]))
        rows.append(
            {
                "kind": entry["kind"],
                "model": model,
                "mode": mode,
                "threads": threads,
                "engine": engine,
                "samples": entry["samples"],
                "samples_n": len(entry["samples"]),
                "median_sec": median,
                "min_sec": low,
                "max_sec": high,
                "mb_per_s": input_bytes / 1e6 / median,
                "input_bytes": input_bytes,
                "tokens": next(iter(entry["tokens"])),
                "checksum": next(iter(entry["checksums"])),
                "rows_count": next(iter(entry["rows"])),
                "cpus_allowed": sorted(entry["cpus"]),
                "warmup": entry["warmup"],
                "warmup_runs": next(iter(entry["warmup_runs"])),
                "files": entry["files"],
            }
        )

    comparisons = []
    for kind in ("warm", "cold"):
        for model in sorted({row["model"] for row in rows}):
            for mode, threads in sorted({(row["mode"], row["threads"]) for row in rows}):
                tokie = next(
                    (
                        row
                        for row in rows
                        if row["kind"] == kind
                        and row["model"] == model
                        and row["mode"] == mode
                        and row["threads"] == threads
                        and row["engine"] == "tokie"
                    ),
                    None,
                )
                giga = next(
                    (
                        row
                        for row in rows
                        if row["kind"] == kind
                        and row["model"] == model
                        and row["mode"] == mode
                        and row["threads"] == threads
                        and row["engine"] == "gigatoken"
                    ),
                    None,
                )
                if not tokie or not giga:
                    continue
                comparisons.append(
                    {
                        "kind": kind,
                        "model": model,
                        "mode": mode,
                        "threads": threads,
                        "tokie_median_sec": tokie["median_sec"],
                        "gigatoken_median_sec": giga["median_sec"],
                        "tokie_mb_per_s": tokie["mb_per_s"],
                        "gigatoken_mb_per_s": giga["mb_per_s"],
                        "tokie_over_gigatoken": giga["median_sec"] / tokie["median_sec"],
                        "tokens_equal": tokie["tokens"] == giga["tokens"],
                        "checksum_equal": tokie["checksum"] == giga["checksum"],
                        "tokie_range_sec": [tokie["min_sec"], tokie["max_sec"]],
                        "gigatoken_range_sec": [giga["min_sec"], giga["max_sec"]],
                    }
                )

    summary = {
        "label": args.label,
        "generated_unix": int(time.time()),
        "rows": rows,
        "comparisons": comparisons,
    }
    out_path = Path(args.out) if args.out else results_dir / "summary.json"
    summary_text = json.dumps(summary, indent=2) + "\n"
    out_path.write_text(summary_text)
    print(summary_text)

    print("\n### Warm (cache-warm, in-process repeats)\n")
    print("| model | mode | threads | tokie MB/s | gigatoken MB/s | tokie/giga | tokie range ms | giga range ms |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for comp in comparisons:
        if comp["kind"] != "warm":
            continue
        print(
            f"| {comp['model']} | {comp['mode']} | {comp['threads']} | "
            f"{comp['tokie_mb_per_s']:.1f} | {comp['gigatoken_mb_per_s']:.1f} | "
            f"{comp['tokie_over_gigatoken']:.3f} | "
            f"{comp['tokie_range_sec'][0]*1e3:.1f}-{comp['tokie_range_sec'][1]*1e3:.1f} | "
            f"{comp['gigatoken_range_sec'][0]*1e3:.1f}-{comp['gigatoken_range_sec'][1]*1e3:.1f} |"
        )

    print("\n### Cold (fresh process, first pass)\n")
    print("| model | mode | threads | tokie MB/s | gigatoken MB/s | tokie/giga | tokie range ms | giga range ms |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for comp in comparisons:
        if comp["kind"] != "cold":
            continue
        print(
            f"| {comp['model']} | {comp['mode']} | {comp['threads']} | "
            f"{comp['tokie_mb_per_s']:.1f} | {comp['gigatoken_mb_per_s']:.1f} | "
            f"{comp['tokie_over_gigatoken']:.3f} | "
            f"{comp['tokie_range_sec'][0]*1e3:.1f}-{comp['tokie_range_sec'][1]*1e3:.1f} | "
            f"{comp['gigatoken_range_sec'][0]*1e3:.1f}-{comp['gigatoken_range_sec'][1]*1e3:.1f} |"
        )


if __name__ == "__main__":
    main()
