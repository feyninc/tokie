#!/usr/bin/env python3
"""Regression tests for the published-benchmark report tooling.

The chart fixtures are copies of the committed results JSON written to a
temporary directory. The canonical files under docs/benchmarks/published-0.2/
are only read. Chart tests need matplotlib; run them in the chart venv.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLISHED = HERE.parent / "docs" / "benchmarks" / "published-0.2"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(name):
    return json.loads((PUBLISHED / name).read_text())


def _write_fixture(directory, name, doc):
    path = Path(directory) / name
    path.write_text(json.dumps(doc))
    return path


bench = _load("bench_published", "bench_published.py")
try:
    charts = _load("generate_charts", "generate_charts.py")
except ModuleNotFoundError as exc:  # matplotlib is only needed for chart tests
    charts = None
    _CHART_SKIP = f"matplotlib not importable: {exc}"
else:
    _CHART_SKIP = ""


class StrictResultsOutputTests(unittest.TestCase):
    """A non-finite number must not leave a partial or invalid results file."""

    def test_finite_results_round_trip_through_the_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "results.json"
            payload = bench.write_results({"mib_s": 1.5}, out)
            self.assertEqual(json.loads(out.read_text()), {"mib_s": 1.5})
            self.assertEqual(out.read_text(), payload)

    def test_nan_leaves_an_existing_output_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.json"
            out.write_bytes(b"recorded earlier\n")
            with self.assertRaises(ValueError):
                bench.write_results(
                    {"benchmark": {"gpt2": {"decode_mib_s": float("nan")}}}, out)
            self.assertEqual(out.read_bytes(), b"recorded earlier\n")

    def test_infinity_does_not_create_an_output_or_its_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "nested" / "results.json"
            with self.assertRaises(ValueError):
                bench.write_results(
                    {"benchmark": {"gpt2": {"decode_mib_s": [float("inf")]}}}, out)
            self.assertFalse(out.exists())
            self.assertFalse(out.parent.exists())


@unittest.skipIf(charts is None, _CHART_SKIP)
class ChartThreadValidationTests(unittest.TestCase):
    def test_batch_threads_must_match_the_run_environment(self):
        doc = _read("results-8t.json")
        for ops in doc["benchmark"].values():
            for body in ops.values():
                body["threads"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "results-8t.json", doc)
            with self.assertRaises(SystemExit) as caught:
                charts.load_giga_results(path)
        self.assertIn("batch follows meta.env.RAYON_NUM_THREADS",
                      str(caught.exception))

    def test_single_thread_ops_accept_one_thread_under_an_eight_thread_env(self):
        doc = _read("results-decode-inputs.json")
        doc["meta"]["env"]["RAYON_NUM_THREADS"] = "8"
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "results-decode-inputs.json", doc)
            _, entries = charts.load_giga_results(path)
        self.assertEqual({entry["threads"] for entry in entries}, {1})
        self.assertEqual({entry["op"] for entry in entries},
                         {"decode", "decode_list"})

    def test_single_thread_ops_reject_a_batch_thread_count(self):
        doc = _read("results-decode-inputs.json")
        doc["meta"]["env"]["RAYON_NUM_THREADS"] = "8"
        for ops in doc["benchmark"].values():
            for body in ops.values():
                body["threads"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_fixture(tmp, "results-decode-inputs.json", doc)
            with self.assertRaises(SystemExit) as caught:
                charts.load_giga_results(path)
        self.assertIn("single encode/decode runs on one thread",
                      str(caught.exception))


@unittest.skipIf(charts is None, _CHART_SKIP)
class FirstPassTitleTests(unittest.TestCase):
    def test_title_uses_the_recorded_batch_thread_count(self):
        self.assertEqual(
            charts._first_pass_title({"threads": 8}),
            "First inference pass (8-thread batch, lower is better)")
        self.assertEqual(
            charts._first_pass_title({"threads": 4}),
            "First inference pass (4-thread batch, lower is better)")

    def test_canonical_first_pass_still_reads_eight_threads(self):
        facts, entries = charts.load_giga_results(PUBLISHED / "results-8t-large.json")
        spec = charts._first_pass_spec(facts, entries)
        self.assertIsNotNone(spec)
        self.assertEqual({entry["op"] for entry in spec["group"]}, {"batch"})
        self.assertEqual(charts._first_pass_title(spec["group"][0]),
                         "First inference pass (8-thread batch, lower is better)")


@unittest.skipIf(charts is None, _CHART_SKIP)
class DecodeOverrideTests(unittest.TestCase):
    def test_decode_override_drops_both_decode_variants(self):
        _, decode_entries = charts.load_giga_results(
            PUBLISHED / "results-decode-inputs.json")
        self.assertEqual({entry["op"] for entry in decode_entries},
                         {"decode", "decode_list"})
        self.assertEqual(charts._without_decode_ops(decode_entries), [])

    def test_decode_override_keeps_every_other_operation(self):
        _, base_entries = charts.load_giga_results(PUBLISHED / "results-1t.json")
        kept = charts._without_decode_ops(base_entries)
        self.assertEqual({entry["op"] for entry in kept}, {"encode", "batch"})

    def test_full_generation_with_the_override_has_no_duplicate_chart(self):
        with tempfile.TemporaryDirectory() as out:
            charts.main([
                "--hf-results", str(PUBLISHED / "hf-results.json"),
                "--giga-results", str(PUBLISHED / "results-1t.json"),
                "--giga-results", str(PUBLISHED / "results-8t.json"),
                "--giga-large-results", str(PUBLISHED / "results-8t-large.json"),
                "--decode-results", str(PUBLISHED / "results-decode-inputs.json"),
                "--output-dir", out,
            ])
            stems = sorted(path.stem for path in Path(out).glob("*.png"))
        self.assertEqual(stems, sorted([
            "benchmark_published_hf",
            "benchmark_published_gigatoken_single",
            "benchmark_published_gigatoken_decode",
            "benchmark_published_gigatoken_decode_list",
            "benchmark_published_gigatoken_batch_1t",
            "benchmark_published_gigatoken_batch_8t",
            "benchmark_published_gigatoken_batch_8t_large",
            "benchmark_published_gigatoken_first_pass",
        ]))


if __name__ == "__main__":
    unittest.main()
