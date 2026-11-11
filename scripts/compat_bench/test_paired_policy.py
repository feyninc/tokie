#!/usr/bin/env python3
"""Pure policy checks for run_paired.py. No benchmark or timing is run here.

Run later with:
  python3 scripts/compat_bench/test_paired_policy.py
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_paired", HERE / "run_paired.py")
run_paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_paired)


def row(seconds):
    return {
        "median_sec": seconds,
        "tokens": 12345,
        "checksum": "deadbeefdeadbeef",
        "input_bytes": 1_000_000,
    }


def summarize(before, after, noise_tolerance=None):
    return run_paired.summarize_config(
        [row(value) for value in before],
        [row(value) for value in after],
        alpha=0.05,
        noise_tolerance=noise_tolerance,
        bootstrap_iterations=200,
        bootstrap_seed=1,
    )


def test_small_but_significant_slowdown_fails():
    # 1% slower in all five pairs: p_slower = 1/32 < 0.05, so it must fail.
    result = summarize([1.0] * 5, [1.01] * 5)
    assert result["verdict"] == "significant_slowdown", result
    assert not result["gate_pass"], result
    assert result["median_speedup"] < 1.0


def test_noise_tolerance_does_not_accept_a_slowdown():
    # The optional descriptive tolerance must not turn a significant slowdown
    # into a pass or a "no regression" label.
    result = summarize([1.0] * 5, [1.01] * 5, noise_tolerance=0.05)
    assert result["verdict"] == "significant_slowdown", result
    assert not result["gate_pass"], result
    assert result["within_noise_tolerance"] is True, result


def test_slower_median_without_significance_passes():
    # Three of five pairs slower, median below 1, p_slower = 0.5. This is
    # no measurable change, not a proven zero and not a regression.
    result = summarize(
        [1.0] * 5,
        [1.0, 1.0, 1.01, 1.02, 1.03],
    )
    assert result["median_speedup"] < 1.0, result
    assert result["p_after_slower"] >= 0.05, result
    assert result["verdict"] == "no_measurable_change", result
    assert result["gate_pass"], result


def test_consistent_improvement_passes():
    result = summarize([1.0] * 5, [0.99] * 5)
    assert result["verdict"] == "significant_improvement", result
    assert result["gate_pass"], result


if __name__ == "__main__":
    test_small_but_significant_slowdown_fails()
    test_noise_tolerance_does_not_accept_a_slowdown()
    test_slower_median_without_significance_passes()
    test_consistent_improvement_passes()
    print("PASS: strict paired gate policy checks")
