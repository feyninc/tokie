#!/usr/bin/env python3
"""Paired before/after runner for the fast tokie encode/batch paths.

This is the authoritative regression gate. It runs the saved before binary and
the candidate after binary in the same time window, alternating which one goes
first for each pair. Separate `run_baseline.sh` runs are descriptive only: they
cannot tell a small regression from host drift.

The primary acceptance rule is **no statistically supported slowdown** per
configuration, not a fixed percentage budget. For each configuration:

  * differences are paired before/after wall times
  * zero differences are ties and are dropped from the sign test
  * p_slower is the exact one-sided P(X >= slower_pairs) under Binomial(n, 1/2)
  * if p_slower < alpha, the configuration fails at any observed magnitude
  * otherwise it passes, including when the sample median is below 1 but the
    paired test cannot distinguish it from noise

`no_measurable_change` means the test did not detect a slowdown. It is not
proof that the true difference is zero. Every row reports the paired median
speedup, a bootstrap 95% CI for that median, the observed delta, and the sign
test p-value, so magnitude and uncertainty are both visible.

Warm processes run five timed encodes after their own warmup; a cold process
runs one timed encode with zero warmup. At least five process pairs (default
seven) are required. All six models, single, batch 1 thread, batch 8 threads,
warm, and cold are in the default workload: 36 rows.

A failed row must be rerun focused to confirm, for example:
  --focus cold:gpt2:batch:8 --pairs 9

The before and after builds must carry build metadata with the same local
gigatoken source identity. A full source-tree hash is compared when both sides
have one; otherwise the runner falls back to HEAD + normalized dirty status +
every modified-file hash, and rejects incomplete dirty manifests. It fails
before timing if the check is missing, incomplete, or mismatched.

Exit codes: 0 gate passed, 1 id/correctness mismatch, 2 gigatoken source-freeze
check failed, incomplete, or missing, 3 statistically supported slowdown.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_MODELS = "gpt2,cl100k,qwen2,o200k,qwen3.5,deepseek"
DEFAULT_CONFIGS = "single:1,batch:1,batch:8"
DEFAULT_KINDS = "warm,cold"
CPU_FOR = {"single:1": "8", "batch:1": "8", "batch:8": "4-11"}
ROOT = Path("/tmp/opencode/tokie-compat")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(binary, model, mode, threads, warmup, reps, label, json_out):
    return [
        "taskset",
        "-c",
        CPU_FOR[f"{mode}:{threads}"],
        str(binary),
        "--model",
        model,
        "--mode",
        mode,
        "--engine",
        "tokie",
        "--threads",
        str(threads),
        "--warmup",
        str(warmup),
        "--reps",
        str(reps),
        "--label",
        label,
        "--json-out",
        str(json_out),
    ]


def run_one(binary, model, mode, threads, warmup, reps, label, json_out, corpus_dir):
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(threads)
    cmd = command(binary, model, mode, threads, warmup, reps, label, json_out)
    cmd.extend(["--corpus-dir", str(corpus_dir)])
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}"
        )
    data = json.loads(json_out.read_text())
    rows = data["results"]
    if len(rows) != 1:
        raise RuntimeError(f"{json_out}: expected one result, got {len(rows)}")
    row = rows[0]
    if row["engine"] != "tokie":
        raise RuntimeError(f"{json_out}: expected tokie, got {row['engine']}")
    if row["warmup"] != warmup or row["warmup_runs"] != warmup:
        raise RuntimeError(
            f"{json_out}: warmup={row['warmup']} warmup_runs={row['warmup_runs']} "
            f"expected {warmup}"
        )
    if row["reps"] != reps:
        raise RuntimeError(f"{json_out}: reps={row['reps']} expected {reps}")
    return row


def sign_p_greater(k, n):
    """One-sided exact P(X >= k) for X ~ Binomial(n, 1/2)."""
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def quantile(values, q):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def bootstrap_median_ci(values, iterations, seed):
    """Percentile bootstrap 95% CI for the median paired speedup."""
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(iterations):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(median(sample))
    return quantile(medians, 0.025), quantile(medians, 0.975)


def load_build_meta(binary, explicit):
    path = Path(explicit) if explicit else Path(str(binary) + ".meta.json")
    if not path.is_file():
        return None, path
    return json.loads(path.read_text()), path


def parse_status(line):
    """Parse porcelain to (code, path), tolerating the historical strip bug."""
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


def source_coverage(meta):
    """Every dirty/untracked path in status must have a recorded file hash."""
    status = (meta.get("gigatoken") or {}).get("status")
    hashes = meta.get("gigatoken_modified_hashes")
    if status is None or hashes is None:
        return {
            "complete": False,
            "reason": "status or modified-file hashes missing",
            "paths": [],
            "hashed": [],
            "missing": [],
            "extra": [],
        }
    paths = sorted({path for _, path in normalized_status(status)})
    hashed = sorted(hashes)
    missing = sorted(set(paths) - set(hashed))
    extra = sorted(set(hashed) - set(paths))
    return {
        "complete": not missing and not extra,
        "reason": "" if not missing and not extra else "dirty manifest incomplete",
        "paths": paths,
        "hashed": hashed,
        "missing": missing,
        "extra": extra,
    }


def compare_source_meta(before_meta, after_meta, before_path, after_path):
    """Compare the local gigatoken source identity recorded by both builds.

    Full source-tree hashes are compared when both sides have them. If the
    historical before metadata has no full hash, fall back to HEAD + normalized
    dirty status + every modified-file hash, but only when the dirty manifest
    covers every status path. An incomplete manifest never falls back.
    """
    result = {
        "passed": False,
        "before_meta": str(before_path),
        "after_meta": str(after_path),
        "reason": "",
        "mode": "unavailable",
        "fallback_used": False,
        "full_source_hash_compared": False,
    }
    if before_meta is None or after_meta is None:
        result["reason"] = "missing build metadata for one or both binaries"
        return result

    before_coverage = source_coverage(before_meta)
    after_coverage = source_coverage(after_meta)
    result["coverage"] = {
        "before": before_coverage,
        "after": after_coverage,
    }
    if not before_coverage["complete"] or not after_coverage["complete"]:
        result["reason"] = (
            "incomplete dirty-file manifest/source coverage; full hashes or "
            "HEAD/status/file-hash fallback cannot be trusted"
        )
        return result

    before = before_meta.get("gigatoken") or {}
    after = after_meta.get("gigatoken") or {}
    mismatches = []
    if before.get("head") != after.get("head"):
        mismatches.append("head")
    before_status = normalized_status(before.get("status") or [])
    after_status = normalized_status(after.get("status") or [])
    if before_status != after_status:
        mismatches.append("status")
    before_hashes = before_meta.get("gigatoken_modified_hashes")
    after_hashes = after_meta.get("gigatoken_modified_hashes")
    if before_hashes != after_hashes:
        mismatches.append("gigatoken_modified_hashes")

    before_source = before_meta.get("gigatoken_source_sha256")
    after_source = after_meta.get("gigatoken_source_sha256")
    both_sources = before_source is not None and after_source is not None
    if both_sources:
        result["mode"] = "full_source_hash"
        result["full_source_hash_compared"] = True
        if before_source != after_source:
            mismatches.append("gigatoken_source_sha256")
    else:
        result["mode"] = "fallback_head_status_modified_hashes"
        result["fallback_used"] = True

    result["status_raw_equal"] = (before.get("status") == after.get("status"))
    result["before"] = {
        "head": before.get("head"),
        "status_normalized": before_status,
        "modified_hashes": before_hashes,
        "source_sha256": before_source,
    }
    result["after"] = {
        "head": after.get("head"),
        "status_normalized": after_status,
        "modified_hashes": after_hashes,
        "source_sha256": after_source,
    }
    if mismatches:
        result["reason"] = "gigatoken source differs: " + ", ".join(mismatches)
        return result
    result["passed"] = True
    result["reason"] = (
        "gigatoken source identity matches"
        if both_sources
        else "gigatoken source identity matches via HEAD + status + file hashes"
    )
    return result



def summarize_config(
    before,
    after,
    alpha,
    noise_tolerance,
    bootstrap_iterations,
    bootstrap_seed,
):
    if len(before) != len(after) or not before:
        raise RuntimeError("before/after sample counts differ")
    n = len(before)

    for side, rows in (("before", before), ("after", after)):
        tokens = {row["tokens"] for row in rows}
        checksums = {row["checksum"] for row in rows}
        if len(tokens) != 1 or len(checksums) != 1:
            raise RuntimeError(f"{side} produced inconsistent ids across pairs")

    before_tokens = before[0]["tokens"]
    before_checksum = before[0]["checksum"]
    id_equal = all(
        row["tokens"] == before_tokens and row["checksum"] == before_checksum
        for row in after
    )

    before_secs = [row["median_sec"] for row in before]
    after_secs = [row["median_sec"] for row in after]
    speedups = [b / a for b, a in zip(before_secs, after_secs)]
    after_faster = sum(a < b for b, a in zip(before_secs, after_secs))
    after_slower = sum(a > b for b, a in zip(before_secs, after_secs))
    ties = n - after_faster - after_slower

    median_speedup = median(speedups)
    ci_low, ci_high = bootstrap_median_ci(
        speedups, bootstrap_iterations, bootstrap_seed
    )
    p_faster = sign_p_greater(after_faster, n - ties)
    p_slower = sign_p_greater(after_slower, n - ties)

    if not id_equal:
        verdict = "correctness_mismatch"
    elif p_slower < alpha:
        # No percentage budget: any statistically supported slowdown fails.
        verdict = "significant_slowdown"
    elif median_speedup > 1.0 and p_faster < alpha:
        verdict = "significant_improvement"
    else:
        verdict = "no_measurable_change"

    if noise_tolerance is None:
        within_noise_tolerance = None
        noise_note = "no descriptive noise tolerance supplied"
    else:
        within_noise_tolerance = abs(median_speedup - 1.0) <= noise_tolerance
        noise_note = (
            "descriptive only; this field never changes pass/fail and never "
            "labels a percentage slowdown as accepted"
        )

    input_bytes = before[0]["input_bytes"]
    return {
        "id_equal": id_equal,
        "tokens": before_tokens,
        "checksum": before_checksum,
        "pairs": n,
        "input_bytes": input_bytes,
        "before_secs": before_secs,
        "after_secs": after_secs,
        "before_median_sec": median(before_secs),
        "after_median_sec": median(after_secs),
        "before_mb_per_s": input_bytes / 1e6 / median(before_secs),
        "after_mb_per_s": input_bytes / 1e6 / median(after_secs),
        "speedups": speedups,
        "median_speedup": median_speedup,
        "ci95_speedup_low": ci_low,
        "ci95_speedup_high": ci_high,
        "observed_delta_pct": (median_speedup - 1.0) * 100.0,
        "min_pair_speedup": min(speedups),
        "max_pair_speedup": max(speedups),
        "after_faster_pairs": after_faster,
        "after_slower_pairs": after_slower,
        "tie_pairs": ties,
        "p_after_faster": p_faster,
        "p_after_slower": p_slower,
        "verdict": verdict,
        "gate_pass": id_equal and p_slower >= alpha,
        "within_noise_tolerance": within_noise_tolerance,
        "noise_note": noise_note,
        "interpretation": (
            "no_measurable_change is not proof of zero change; it means the "
            "paired test did not detect a slowdown at this sample size"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-bin", required=True)
    parser.add_argument("--after-bin", required=True)
    parser.add_argument(
        "--before-meta",
        default="",
        help="collect_meta JSON for the before build (default: <before-bin>.meta.json)",
    )
    parser.add_argument(
        "--after-meta",
        default="",
        help="collect_meta JSON for the after build (default: <after-bin>.meta.json)",
    )
    parser.add_argument("--out", default=str(ROOT / "paired"))
    parser.add_argument("--label", default="paired")
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--kinds", default=DEFAULT_KINDS)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup-single", type=int, default=3)
    parser.add_argument("--warmup-batch", type=int, default=6)
    parser.add_argument(
        "--warm-reps",
        type=int,
        default=5,
        help="in-process timed reps per warm process; its median feeds one pair",
    )
    parser.add_argument("--cold-reps", type=int, default=1)
    parser.add_argument("--corpus-dir", default=str(ROOT / "corpus"))
    parser.add_argument(
        "--focus",
        default="",
        help="comma-separated confirmation rows: kind:model:mode:threads",
    )
    parser.add_argument(
        "--noise-tolerance",
        type=float,
        default=None,
        help=(
            "descriptive only; never affects pass/fail and never accepts a "
            "percentage slowdown as no regression"
        ),
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="bootstrap resamples for the median-speedup 95%% CI",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.pairs < 5:
        parser.error("--pairs must be at least 5")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must be in (0, 1)")
    if args.noise_tolerance is not None and args.noise_tolerance < 0:
        parser.error("--noise-tolerance must be non-negative")
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")

    before_bin = Path(args.before_bin).resolve()
    after_bin = Path(args.after_bin).resolve()
    for binary in (before_bin, after_bin):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"not an executable: {binary}")

    before_meta, before_meta_path = load_build_meta(before_bin, args.before_meta)
    after_meta, after_meta_path = load_build_meta(after_bin, args.after_meta)
    source_check = compare_source_meta(
        before_meta, after_meta, before_meta_path, after_meta_path
    )
    if not source_check["passed"]:
        print(
            f"gigatoken source-freeze check failed: {source_check['reason']}",
            file=sys.stderr,
        )
        print(
            "use --before-meta from baseline_meta_base-corrected.enriched.json "
            "and --after-meta from collect_meta.py; the gate does not run "
            "timing when the counterpart source is not frozen",
            file=sys.stderr,
        )
        return 2

    out = Path(args.out)
    raw_dir = out / "raw"
    if out.exists() and any(out.iterdir()) and not args.force:
        parser.error(f"{out} is not empty; pass --force or choose a new --out")
    raw_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)

    models = [item for item in args.models.split(",") if item]
    configs = []
    for item in args.configs.split(","):
        mode, threads = item.split(":")
        configs.append((mode, int(threads)))
    kinds = [item for item in args.kinds.split(",") if item]
    for mode, threads in configs:
        if f"{mode}:{threads}" not in CPU_FOR:
            parser.error(f"no affinity mapping for {mode}:{threads}")

    expected = [
        f"{kind}:{model}:{mode}:{threads}"
        for model in models
        for mode, threads in configs
        for kind in kinds
    ]
    focus = [item for item in args.focus.split(",") if item]
    if focus:
        unknown = sorted(set(focus) - set(expected))
        if unknown:
            parser.error(f"focus rows not in the workload: {unknown}")
        focus_set = set(focus)
    else:
        focus_set = set(expected)

    previous_profile = None
    if shutil.which("powerprofilesctl"):
        previous_profile = subprocess.run(
            ["powerprofilesctl", "get"], text=True, capture_output=True
        ).stdout.strip()
        subprocess.run(["powerprofilesctl", "set", "performance"], check=False)

    before_rows = {}
    after_rows = {}
    started = time.time()
    try:
        for model in models:
            for mode, threads in configs:
                for kind in kinds:
                    focus_key = f"{kind}:{model}:{mode}:{threads}"
                    if focus_key not in focus_set:
                        continue
                    warmup = 0 if kind == "cold" else (
                        args.warmup_single if mode == "single" else args.warmup_batch
                    )
                    reps = args.cold_reps if kind == "cold" else args.warm_reps
                    before_samples = []
                    after_samples = []
                    for pair in range(args.pairs):
                        order = (
                            (("before", before_bin), ("after", after_bin))
                            if pair % 2 == 0
                            else (("after", after_bin), ("before", before_bin))
                        )
                        for side, binary in order:
                            label = f"{args.label}-{kind}-{model}-{mode}-{threads}t-p{pair}"
                            path = raw_dir / f"{kind}_{model}_{mode}_{threads}t_p{pair}_{side}.json"
                            print(
                                f"{kind} {model} {mode} {threads}t pair {pair + 1}/{args.pairs} {side}",
                                file=sys.stderr,
                            )
                            row = run_one(
                                binary,
                                model,
                                mode,
                                threads,
                                warmup,
                                reps,
                                label,
                                path,
                                corpus_dir,
                            )
                            if side == "before":
                                before_samples.append(row)
                            else:
                                after_samples.append(row)
                    key = (kind, model, mode, threads)
                    before_rows[key] = before_samples
                    after_rows[key] = after_samples
    finally:
        if previous_profile:
            subprocess.run(["powerprofilesctl", "set", previous_profile], check=False)

    if not before_rows:
        parser.error("no configurations ran")

    results = []
    for key in before_rows:
        kind, model, mode, threads = key
        results.append(
            {
                "kind": kind,
                "model": model,
                "mode": mode,
                "threads": threads,
                "config": f"{kind}:{model}:{mode}:{threads}",
                "warmup": 0 if kind == "cold" else (
                    args.warmup_single if mode == "single" else args.warmup_batch
                ),
                "warmup_runs": before_rows[key][0]["warmup_runs"],
                "reps": before_rows[key][0]["reps"],
                "affinity": CPU_FOR[f"{mode}:{threads}"],
                **summarize_config(
                    before_rows[key],
                    after_rows[key],
                    args.alpha,
                    args.noise_tolerance,
                    args.bootstrap,
                    args.seed,
                ),
            }
        )

    failed = [row["config"] for row in results if not row["gate_pass"]]
    gate = {
        "pairs": args.pairs,
        "min_pairs": 5,
        "alpha": args.alpha,
        "policy": (
            "primary acceptance is no statistically supported slowdown per "
            "configuration: exact one-sided paired sign test on before/after "
            "times, p_slower < alpha fails at any observed magnitude. There is "
            "no fixed percentage budget. no_measurable_change is not proof of "
            "zero change."
        ),
        "noise_tolerance": args.noise_tolerance,
        "noise_tolerance_used_for_gate": False,
        "power_note": (
            "shared host and finite paired samples limit power; with 7 pairs "
            "the exact sign test needs 7/7 slower for p < 0.05. Use the focused "
            "confirmation with more pairs. no_measurable_change is not proof of "
            "zero change."
        ),
        "bootstrap_iterations": args.bootstrap,
        "bootstrap_seed": args.seed,
        "source_check": source_check,
        "passed": all(row["gate_pass"] for row in results) and source_check["passed"],
        "correctness_passed": all(row["id_equal"] for row in results),
        "failed_configs": failed,
        "confirmation_required": bool(failed),
        "confirmation": {
            "required_for": failed,
            "suggested_pairs": max(9, args.pairs),
            "focus_arg": ",".join(failed),
        },
    }
    corpus_manifest_path = corpus_dir / "manifest.json"
    corpus_manifest = (
        json.loads(corpus_manifest_path.read_text())
        if corpus_manifest_path.exists()
        else None
    )
    summary = {
        "label": args.label,
        "generated_unix": int(time.time()),
        "elapsed_sec": time.time() - started,
        "host": platform.node(),
        "before_bin": {"path": str(before_bin), "sha256": sha256(before_bin)},
        "after_bin": {"path": str(after_bin), "sha256": sha256(after_bin)},
        "before_meta": str(before_meta_path),
        "after_meta": str(after_meta_path),
        "source_check": source_check,
        "corpus_dir": str(corpus_dir),
        "corpus_manifest": corpus_manifest,
        "configs": [f"{mode}:{threads}" for mode, threads in configs],
        "affinity": {
            f"{mode}:{threads}": CPU_FOR[f"{mode}:{threads}"]
            for mode, threads in configs
        },
        "warmups": {
            "single": args.warmup_single,
            "batch": args.warmup_batch,
            "cold": 0,
        },
        "reps": {"warm": args.warm_reps, "cold": args.cold_reps},
        "kinds": kinds,
        "focus": sorted(focus_set),
        "gate": gate,
        "results": results,
    }
    summary_path = out / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_text = json.dumps(summary, indent=2) + "\n"
    summary_path.write_text(summary_text)

    print(
        "\n| kind | model | mode | threads | before MB/s | after MB/s | "
        "speedup | 95% CI | delta | p(slower) | verdict |"
    )
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in results:
        print(
            f"| {row['kind']} | {row['model']} | {row['mode']} | {row['threads']} | "
            f"{row['before_mb_per_s']:.1f} | {row['after_mb_per_s']:.1f} | "
            f"{row['median_speedup']:.3f} | "
            f"{row['ci95_speedup_low']:.3f}-{row['ci95_speedup_high']:.3f} | "
            f"{row['observed_delta_pct']:+.1f}% | {row['p_after_slower']:.3f} | "
            f"{row['verdict']} |"
        )
    print(
        f"\ngate: {'PASS' if gate['passed'] else 'FAIL'} "
        f"(pairs={gate['pairs']}, alpha={gate['alpha']}, "
        f"source_frozen={source_check['passed']}, "
        f"source_mode={source_check['mode']})"
    )
    print(
        "no_measurable_change means no statistically supported slowdown at this "
        "sample size, not proof of zero change."
    )
    if gate["confirmation_required"]:
        print(
            "focused confirmation required for: "
            + ", ".join(gate["confirmation"]["required_for"])
        )
        print(
            "rerun with --focus "
            + gate["confirmation"]["focus_arg"]
            + f" --pairs {gate['confirmation']['suggested_pairs']}"
        )
    print(f"summary: {summary_path}")

    if not gate["correctness_passed"]:
        return 1
    if not gate["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
