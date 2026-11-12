#!/usr/bin/env python3
"""Tests for scripts/generate_benchmark_dashboard.py.

The fixture is deliberately tiny and fabricated; it is a unit-test input, not
a report of any real benchmark. Numeric tampering must be rejected, and both
layouts must render the validated data.
"""

import copy
import importlib.util
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "generate_benchmark_dashboard", HERE / "generate_benchmark_dashboard.py"
)
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)
MIB = 1024.0 * 1024.0


def _task(libs, task, threads, passes, input_bytes, decoded_bytes, counts, bases, rounds=5):
    rounds_entries = []
    for index in range(rounds):
        order = libs[index % len(libs):] + libs[:index % len(libs)]
        entry = {"round": index + 1, "order": order}
        for lib in libs:
            seconds = bases[lib] * (1.0 + 0.03 * (index - 2) / 2.0)
            entry[lib] = {"seconds": seconds, "count": counts[lib]}
        rounds_entries.append(entry)
    unit = decoded_bytes if task == "decode" else input_bytes
    count_unit = {"encode": "ids", "decode": "characters", "batch": "ids_and_documents"}[task]
    summary = {}
    for lib in libs:
        values = [round_entry[lib]["seconds"] for round_entry in rounds_entries]
        median = statistics.median(values)
        low, high = min(values), max(values)
        count_values = {
            "count_s_median": counts[lib] * passes / median,
            "count_s_min": counts[lib] * passes / high,
            "count_s_max": counts[lib] * passes / low,
        }
        summary[lib] = {
            "count_unit": count_unit,
            "seconds_median": median,
            "seconds_min": low,
            "seconds_max": high,
            "mib_s_median": unit * passes / median / MIB,
            "mib_s_min": unit * passes / high / MIB,
            "mib_s_max": unit * passes / low / MIB,
            **count_values,
        }
        if count_unit == "ids":
            summary[lib].update({key.replace("count_s", "tokens_s"): value
                                 for key, value in count_values.items()})
        elif count_unit == "characters":
            summary[lib].update({key.replace("count_s", "chars_s"): value
                                 for key, value in count_values.items()})
    return {
        "threads": threads,
        "documents": 10 if task != "batch" else 20,
        "passes_per_round": passes,
        "input_bytes_utf8": input_bytes,
        "decoded_bytes_utf8": decoded_bytes,
        "throughput_numerator": "decoded_bytes_utf8" if task == "decode" else "input_bytes_utf8",
        "count_unit": count_unit,
        "rounds": rounds_entries,
        "summary": summary,
        "post_check": {lib: {"output_ok": True, "checksum": 101 + i} for i, lib in enumerate(libs)},
        "profile": {lib: {"cpu_per_wall": 1.0} for lib in libs},
    }


def _models(labels):
    three = ["tokie", "giga", "hf"]
    two = ["tokie", "hf"]
    return {
        label: {"tokenizer_json_sha256": "0" * 64,
                "libraries": list(two if label == "gamma" else three),
                "figure": "hf_only" if label == "gamma" else "overview"}
        for label in labels
    }


def _payload(labels, benchmark, samples, tasks_note=""):
    models = _models(labels)
    return {
        "schema_version": 1,
        "meta": {
            "generated_utc": "2026-01-01T00:00:00Z",
            "artifacts": {
                "tokie": {"version": "0.2.0", "kind": "local_unreleased", "wheel_sha256": "f" * 64},
                "giga": {"version": "0.11.0", "kind": "pypi"},
                "hf": {"name": "HuggingFace tokenizers", "version": "0.23.3", "kind": "pypi"},
            },
            "cpu_model": "Test CPU",
            "cpu_count_logical": 4,
            "platform": "linux",
            "host": {"cpu_model": "Test CPU", "cpu_count": 4,
                      "mean_cur_freq_mhz": 1000.0, "load_before": [0.5]},
            "samples": samples,
            "tasks": {
                "encode": {"title": "Encode", "ordinal": 1, "unit": "MiB/s", "sample": "single"},
                "decode": {"title": "Unicode decode", "ordinal": 2, "unit": "MiB/s", "sample": "single"},
                "batch": {"title": "Batch", "ordinal": 3, "unit": "MiB/s", "sample": "batch"},
            },
        },
        "models": models,
        "validation": {
            label: {"ok": True, "first_mismatch": None, "validation_texts": 12,
                    "encode_mismatches": 0, "decode_mismatches": 0, "batch_mismatches": 0}
            for label in labels
        },
        "benchmark": benchmark,
        "status": {label: "ok" for label in labels},
        "errors": [],
    }


def _streams(label, scale, rounds=5):
    three = ["tokie", "giga", "hf"]
    encode_counts = {"tokie": 200_000, "giga": 200_000, "hf": 200_000}
    decode_counts = {"tokie": 210_000, "giga": 210_000, "hf": 210_000}
    batch_counts = {"tokie": 300_000, "giga": 300_000, "hf": 300_000}
    return {
        "encode": _task(three, "encode", 1, 2, 1_000_000, 0, encode_counts,
                        {"tokie": 0.02 * scale, "giga": 0.026 * scale, "hf": 0.4 * scale}, rounds=rounds),
        "decode": _task(three, "decode", 1, 2, 1_000_000, 1_050_000, decode_counts,
                        {"tokie": 0.03 * scale, "giga": 0.02 * scale, "hf": 0.5 * scale}, rounds=rounds),
        "batch": _task(three, "batch", 1, 2, 4_000_000, 0, batch_counts,
                       {"tokie": 0.05 * scale, "giga": 0.04 * scale, "hf": 1.0 * scale}, rounds=rounds),
    }


def fixture(rounds=5):
    """One 1-thread file (the shape every individual input has)."""
    three = ["tokie", "giga", "hf"]
    benchmark = {"alpha": _streams("alpha", 1.0, rounds), "beta": _streams("beta", 1.6, rounds)}
    return _payload(["alpha", "beta"], benchmark, {
        "single": {"documents": 10, "input_bytes_utf8": 1_000_000},
        "batch": {"documents": 20, "input_bytes_utf8": 4_000_000},
    })


def fixture_files():
    """Three files: 1T encode/decode/batch, 8T batch, and HF-only models."""
    three = ["tokie", "giga", "hf"]
    single = {
        "single": {"documents": 10, "input_bytes_utf8": 1_000_000},
        "batch": {"documents": 20, "input_bytes_utf8": 4_000_000},
    }
    batch_sample = {"batch": {"documents": 20, "input_bytes_utf8": 4_000_000}}
    one = _payload(["alpha", "beta"], {"alpha": _streams("alpha", 1.0), "beta": _streams("beta", 1.6)}, single)
    eight = _payload(["alpha", "beta"], {
        label: {"batch": _task(three, "batch", 8, 2, 4_000_000, 0,
                               {"tokie": 300_000, "giga": 300_000, "hf": 300_000},
                               {"tokie": 0.05 * scale, "giga": 0.04 * scale, "hf": 1.0 * scale})}
        for label, scale in (("alpha", 0.6), ("beta", 0.9))
    }, batch_sample)
    hf_only = _payload(["gamma"], {
        "gamma": {
            "encode": _task(["tokie", "hf"], "encode", 1, 2, 1_000_000, 1_000_000,
                            {"tokie": 200_000, "hf": 200_000}, {"tokie": 0.04, "hf": 0.8}),
            "decode": _task(["tokie", "hf"], "decode", 1, 2, 1_000_000, 1_050_000,
                            {"tokie": 210_000, "hf": 210_000}, {"tokie": 0.06, "hf": 1.0}),
        }
    }, {"single": {"documents": 10, "input_bytes_utf8": 1_000_000}})
    hf_only["validation"]["gamma"]["stats"] = {"batch_mismatches": 0}
    hf_only["validation"]["gamma"].pop("batch_mismatches", None)
    return [one, eight, hf_only]


class DashboardTests(unittest.TestCase):
    def write(self, payloads, directory):
        paths = []
        for payload in payloads if isinstance(payloads, list) else [payloads]:
            path = Path(directory) / f"results-{len(paths)}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            paths.append(path)
        return paths

    def assert_rejected(self, payload, fragment):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write(payload, directory)
            with self.assertRaises(ValueError) as caught:
                dashboard.load_dataset(paths)
            self.assertIn(fragment, str(caught.exception))

    def test_valid_fixture_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(fixture_files(), directory))
            self.assertEqual(dataset["model_order"], ["alpha", "beta", "gamma"])
            self.assertEqual(len(dataset["benchmark"]), 10)

    def test_summary_median_is_reproduced(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["summary"]["tokie"]["mib_s_median"] *= 1.2
        self.assert_rejected(payload, "does not match rounds")

    def test_gib_unit_is_rejected_before_write(self):
        payload = fixture()
        payload["meta"]["tasks"]["encode"]["unit"] = "GiB/s"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(payload, directory)[0]
            out = Path(directory) / "out"
            tables = Path(directory) / "tables.md"
            code = dashboard.main(["--results", str(path), "--output-dir", str(out),
                                   "--tables-out", str(tables)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())
            self.assertFalse(tables.exists())

    def test_stable_host_conflict_fails_before_write(self):
        for field, value in (("cpu_model", "Other CPU"), ("cpu_count_logical", 8),
                             ("platform", "darwin")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                payloads = fixture_files()
                payloads[1]["meta"][field] = value
                paths = self.write(payloads, directory)
                args = ["--output-dir", str(Path(directory) / "out"),
                        "--tables-out", str(Path(directory) / "tables.md")]
                for path in paths:
                    args.extend(["--results", str(path)])
                code = dashboard.main(args)
                self.assertEqual(code, 2)
                self.assertFalse((Path(directory) / "out").exists())
                self.assertFalse((Path(directory) / "tables.md").exists())

    def test_dynamic_host_snapshot_may_differ(self):
        payloads = fixture_files()
        payloads[1]["meta"]["host"]["mean_cur_freq_mhz"] = 9999.0
        payloads[1]["meta"]["host"]["load_before"] = [9.9, 9.8]
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(payloads, directory))
            self.assertEqual(dataset["host"]["cpu_model"], "Test CPU")

    def test_twenty_round_subtitle_uses_actual_count(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(fixture(rounds=20), directory))
            out = Path(directory) / "figures"
            dashboard.render(dataset, "faceted", str(out))
            svg = (out / "benchmark_current_overview.svg").read_text()
            self.assertIn("observed 20-round min\u2013max", svg)
            self.assertNotIn("observed 5-round min\u2013max", svg)

    def test_two_percent_summary_drift_is_rejected(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["summary"]["tokie"]["mib_s_median"] *= 1.02
        self.assert_rejected(payload, "does not match rounds")

    def test_metadata_ordinal_drives_task_order(self):
        payload = fixture()
        payload["meta"]["tasks"]["encode"]["ordinal"] = 30
        payload["meta"]["tasks"]["decode"]["ordinal"] = 10
        payload["meta"]["tasks"]["batch"]["ordinal"] = 20
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(payload, directory))
            tasks = dashboard._tasks(dataset, ["alpha"])
            self.assertEqual([task[0] for task in tasks], ["decode", "batch", "encode"])
            self.assertEqual(dashboard._task_unit(dataset, tasks[0]), "MiB/s")

    def test_nested_task_definitions_and_units_are_supported(self):
        payload = fixture()
        payload["meta"]["tasks"] = {"definitions": payload["meta"]["tasks"],
                                    "adapters": {"decode": "common uint32 ids -> Unicode"}}
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(payload, directory))
            self.assertEqual(dashboard._task_unit(dataset, ("decode", 1)), "MiB/s")
            self.assertEqual(dashboard._task_title(dataset, ("decode", 1)), "Unicode decode")

    def test_count_unit_and_alias_violations_fail_before_write(self):
        def wrong_count(payload):
            payload["benchmark"]["alpha"]["encode"]["summary"]["tokie"]["count_s_median"] *= 1.02

        def illegal_decode_tokens(payload):
            entry = payload["benchmark"]["alpha"]["decode"]["summary"]["tokie"]
            for suffix in ("median", "min", "max"):
                entry[f"tokens_s_{suffix}"] = entry[f"chars_s_{suffix}"]

        def illegal_batch_chars(payload):
            entry = payload["benchmark"]["alpha"]["batch"]["summary"]["tokie"]
            for suffix in ("median", "min", "max"):
                entry[f"chars_s_{suffix}"] = entry[f"count_s_{suffix}"]

        def wrong_summary_unit(payload):
            payload["benchmark"]["alpha"]["decode"]["summary"]["tokie"]["count_unit"] = "ids"

        cases = (wrong_count, illegal_decode_tokens, illegal_batch_chars, wrong_summary_unit)
        for index, mutate in enumerate(cases):
            with self.subTest(case=mutate.__name__), tempfile.TemporaryDirectory() as directory:
                payload = fixture()
                mutate(payload)
                path = self.write(payload, directory)[0]
                out = Path(directory) / "out"
                tables = Path(directory) / "tables.md"
                code = dashboard.main(["--results", str(path), "--output-dir", str(out),
                                       "--tables-out", str(tables)])
                self.assertEqual(code, 2)
                self.assertFalse(out.exists())
                self.assertFalse(tables.exists())

    def test_invalid_profile_does_not_block_charts(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["profile"]["tokie"]["cpu_per_wall"] = 0
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(payload, directory))
            self.assertIn("alpha", dataset["models"])

    def test_decode_zero_bytes_is_rejected(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["decode"]["decoded_bytes_utf8"] = 0
        self.assert_rejected(payload, "decoded_bytes_utf8 must be positive")

    def test_invalid_throughput_numerator_is_rejected(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["throughput_numerator"] = "characters"
        self.assert_rejected(payload, "throughput_numerator must be input_bytes_utf8 or decoded_bytes_utf8")

    def test_decode_numerator_uses_decoded_bytes(self):
        payload = fixture()
        task = payload["benchmark"]["alpha"]["decode"]
        summary = task["summary"]["tokie"]
        task["decoded_bytes_utf8"] *= 2
        for key, seconds in (("median", "seconds_median"), ("min", "seconds_max"), ("max", "seconds_min")):
            summary[f"mib_s_{key}"] = (
                task["input_bytes_utf8"] * task["passes_per_round"] / summary[seconds] / MIB
            )
        self.assert_rejected(payload, "does not match rounds")

    def test_minmax_order_is_enforced(self):
        payload = fixture()
        summary = payload["benchmark"]["alpha"]["encode"]["summary"]["tokie"]
        summary["mib_s_min"], summary["mib_s_max"] = summary["mib_s_max"], summary["mib_s_min"]
        self.assert_rejected(payload, "out of order")

    def test_zero_summary_is_rejected(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["summary"]["tokie"]["mib_s_median"] = 0
        self.assert_rejected(payload, "non-positive")

    def test_unstable_count_is_rejected(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["rounds"][2]["tokie"]["count"] += 1
        self.assert_rejected(payload, "count is not stable")

    def test_round_order_must_include_each_library_once(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["rounds"][0]["order"] = ["tokie", "giga", "giga"]
        self.assert_rejected(payload, "order must list each library once")

    def test_round_order_must_be_counterbalanced(self):
        payload = fixture()
        for round_entry in payload["benchmark"]["alpha"]["encode"]["rounds"]:
            round_entry["order"] = ["tokie", "giga", "hf"]
        self.assert_rejected(payload, "not counterbalanced")

    def test_validation_mismatch_is_rejected(self):
        payload = fixture()
        payload["validation"]["alpha"]["encode_mismatches"] = 1
        self.assert_rejected(payload, "zero *_mismatches counters")

    def test_status_failure_is_rejected(self):
        payload = fixture()
        payload["status"]["alpha"] = "failed"
        self.assert_rejected(payload, "status is not ok")

    def test_duplicate_series_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(fixture(), directory)[0]
            with self.assertRaises(ValueError) as caught:
                dashboard.load_dataset([path, path])
            self.assertIn("duplicate benchmark", str(caught.exception))

    def test_check_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(fixture(), directory)[0]
            out = Path(directory) / "out"
            tables = Path(directory) / "tables.md"
            code = dashboard.main(["--results", str(path), "--check",
                                   "--output-dir", str(out), "--tables-out", str(tables)])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            self.assertFalse(tables.exists())

    def test_validation_failure_writes_nothing(self):
        payload = fixture()
        payload["benchmark"]["alpha"]["encode"]["summary"]["hf"]["seconds_median"] *= 2
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(payload, directory)[0]
            out = Path(directory) / "out"
            tables = Path(directory) / "tables.md"
            code = dashboard.main(["--results", str(path), "--output-dir", str(out),
                                   "--tables-out", str(tables)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())
            self.assertFalse(tables.exists())

    def test_svg_is_trimmed_and_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(fixture_files(), directory))
            outputs = []
            for name in ("run1", "run2"):
                out = Path(directory) / name
                dashboard.render(dataset, "faceted", str(out))
                outputs.append(out)
            for filename in ("benchmark_current_overview.svg", "benchmark_current_hf_only.svg"):
                first = (outputs[0] / filename).read_bytes()
                second = (outputs[1] / filename).read_bytes()
                self.assertEqual(first, second)
                text = first.decode("utf-8")
                self.assertFalse(any(line != line.rstrip() for line in text.splitlines()))
            for filename in ("benchmark_current_overview.png", "benchmark_current_hf_only.png"):
                self.assertEqual((outputs[0] / filename).read_bytes(),
                                 (outputs[1] / filename).read_bytes())
            labels = [dashboard._legend_name(lib, dataset["artifacts"]) for lib in ("tokie", "giga", "hf")]
            self.assertEqual(labels, ["Tokie 0.2.0\nLOCAL, UNRELEASED", "** GigaToken\n0.11.0",
                                      "HuggingFace tokenizers\n0.23.3"])

    def test_hf_only_missing_batch_mismatch_is_n_a(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(fixture_files(), directory))
            self.assertNotIn("batch_mismatches", dataset["validation"]["gamma"])
            tables = Path(directory) / "tables.md"
            dashboard.write_tables(dataset, str(tables))
            self.assertIn("| Gamma | 0 | 0 | N/A | ok |", tables.read_text())

    def test_giga_coverage_note_only_in_overview_and_escaped_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = dashboard.load_dataset(self.write(fixture_files(), directory))
            out = Path(directory) / "figures"
            dashboard.render(dataset, "faceted", str(out))
            overview = (out / "benchmark_current_overview.svg").read_text()
            hf_only = (out / "benchmark_current_hf_only.svg").read_text()
            note = "** GigaToken: WordPiece and Unigram not supported."
            self.assertIn(note, overview)
            self.assertNotIn(note, hf_only)
            self.assertNotIn("** GigaToken", hf_only)
            tables = Path(directory) / "tables.md"
            dashboard.write_tables(dataset, str(tables))
            text = tables.read_text()
            self.assertIn(r"\*\* GigaToken 0.11.0", text)
            self.assertEqual(text.count(r"\*\* GigaToken: WordPiece and Unigram not supported."), 1)
            hf_tables = text.split("## HuggingFace-only models", 1)[1]
            self.assertNotIn(r"\*\* GigaToken", hf_tables)

    def test_both_layouts_render(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write(fixture_files(), directory)
            dataset = dashboard.load_dataset(paths)
            for layout in ("matrix", "faceted"):
                out = Path(directory) / layout
                written = dashboard.render(dataset, layout, str(out))
                self.assertEqual(len(written), 4)  # overview + HF-only, PNG + SVG
                for item in written:
                    self.assertGreater(Path(item).stat().st_size, 10_000)
                overview = (out / "benchmark_current_overview.svg").read_text()
                self.assertIn("LOCAL, UNRELEASED", overview)
                for version in ("0.2.0", "0.11.0", "0.23.3"):
                    self.assertIn(version, overview)
                self.assertIn("Giga.decode", overview)
                self.assertIn("** GigaToken", overview)
                hf_only = (out / "benchmark_current_hf_only.svg").read_text()
                self.assertNotIn("Giga.decode", hf_only)
                self.assertNotIn("** GigaToken", hf_only)
                self.assertIn("HF.decode(ids.tolist())", hf_only)
                self.assertTrue((out / "benchmark_current_hf_only.png").exists())
            tables = Path(directory) / "tables.md"
            dashboard.write_tables(dataset, str(tables))
            text = tables.read_text()
            self.assertIn("MiB/s", text)
            self.assertIn("Validation", text)
            self.assertIn("Test CPU", (Path(directory) / "matrix" / "benchmark_current_overview.svg").read_text())


if __name__ == "__main__":
    unittest.main()
