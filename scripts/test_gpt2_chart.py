#!/usr/bin/env python3
"""Tests for scripts/generate_gpt2_chart.py.

The fixture mirrors the recorded six-library schema and uses the recorded
medians for layout assertions; the round seconds are synthetic. It is a unit
test, not a benchmark report. The committed gpt2-hero.json remains
authoritative and is checked separately with --check.
"""

import contextlib
import copy
import importlib.util
import io
import json
import statistics
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "generate_gpt2_chart", HERE / "generate_gpt2_chart.py"
)
chart = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chart)
LIBS = chart.LIBS
INPUT_BYTES = 1_057_339
COUNT = 2_454_920
VALIDATION_TEXTS = 459
TOKIE_SHA = "8786f91b1aed3acbac9d8090234e55f0400f829da6ea8a2817bcd460332a1d85"
SHAS = {
    "tokie": TOKIE_SHA,
    "giga": "11992abc7f41e41e5be03c722df1c5205b5d0fcc4bcbd4fb440d2a0b97f7ac3f",
    "hf023": "376851d22bcf9d650a5c3090bb83e6cf9e895fbf0595369fa4cd43c1f69b5f87",
    "hf1": "ecea8dc0d5d08a9e855931989ca60591f7aa2c16d7573862635c15a3c132b332",
    "kitoken": "50728d5adcc3c5b4af1d71ea30e22e8a2809bf3b8a56de0df1c184a54279d13c",
    "tiktoken": "7896eea257fe497a2b7134474d909156c6744ce8da35bce88011a960e008aa0d",
}
VERSIONS = {"tokie": "0.2.0", "giga": "0.11.0", "hf023": "0.23.3",
            "hf1": "1.0.0rc4", "kitoken": "0.11.1", "tiktoken": "0.14.0"}
DISTRIBUTIONS = {"tokie": "tokie", "giga": "gigatoken", "hf023": "tokenizers",
                 "hf1": "tokenizers", "kitoken": "kitoken", "tiktoken": "tiktoken"}
WHEELS = {
    "tokie": "tokie-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl",
    "giga": "gigatoken-0.11.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "hf023": "tokenizers-0.23.3-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
    "hf1": "tokenizers-1.0.0rc4-cp310-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl",
    "kitoken": "kitoken-0.11.1-cp310-abi3-manylinux_2_28_x86_64.whl",
    "tiktoken": "tiktoken-0.14.0-cp312-cp312-manylinux_2_28_x86_64.whl",
}
APIS = {
    "tokie": "tokie.Tokenizer.open(path); threads(1); tokenizer.encode(text) -> numpy uint32 ids; len(ids) in clock",
    "giga": "gigatoken.Tokenizer(path); tokenizer.encode(text) -> numpy uint32 ids; len(ids) in clock",
    "hf023": "tokenizers.Tokenizer.from_file(path); tokenizer.encode(text, add_special_tokens=False).ids; len(ids) in clock",
    "hf1": "tokenizers.Tokenizer.from_file(path).compile(); compiled.encode(text, add_special_tokens=False).ids; len(ids) in clock",
    "kitoken": "kitoken.Kitoken.from_file(path); tokenizer.encode(text, encode_specials=True) -> list[int]; len(ids) in clock",
    "tiktoken": "tiktoken.get_encoding('gpt2'); encoding.encode(text, allowed_special={'<|endoftext|>'}) -> list[int]; len(ids) in clock",
}
POLICIES = {
    "tokie": "recognizes '<|endoftext|>' as id 50256 from the HF tokenizer.json",
    "giga": "recognizes '<|endoftext|>' as id 50256 from the HF tokenizer.json",
    "hf023": "add_special_tokens=False adds nothing; an existing '<|endoftext|>' literal still maps to 50256",
    "hf1": "add_special_tokens=False adds nothing; encode_special_tokens defaults to False, so the literal maps to 50256",
    "kitoken": "encode_specials=True explicitly, because the default False encodes '<|endoftext|>' as regular text",
    "tiktoken": "allowed_special={'<|endoftext|>'} explicitly; the default raises on that literal",
}
API_NOTES = {
    "tokie": "installed binding: Tokenizer.open(path); threads(n) only changes batch workers; encode(text) returns a uint32 numpy array.",
    "giga": "installed binding: Tokenizer(path); encode(text) returns a uint32 numpy array; no thread control on single-text encode.",
    "hf023": "installed stub: Tokenizer.encode(text, add_special_tokens=False) -> Encoding; Encoding.ids is list[int]. This is the only encode API in 0.23.3.",
    "hf1": "installed 1.0.0rc4 stub: Tokenizer.compile() -> CompiledTokenizer; encode(text, add_special_tokens=False) -> Encoding; Encoding.ids is list[int]; Encoding.ids_array returns a copy. No encode_fast or ids-only method exists, and offsets are not returned unless requested, so the documented encode(...).ids path is used.",
    "kitoken": "installed kitoken.pyi: Kitoken.from_file(path) -> Kitoken; encode(text, encode_specials=False) -> list[int]. Default False does not match the HF ids for '<|endoftext|>'; encode_specials=True is used and disclosed.",
    "tiktoken": "tiktoken.get_encoding('gpt2').encode(text, allowed_special=...) -> list[int]. The gpt2 ranks are the canonical OpenAI BPE; allowed_special is passed explicitly.",
}
MEDIANS = {
    "tokie": 0.014285807003034279,
    "giga": 0.01354518299922347,
    "hf023": 2.458796077989973,
    "hf1": 0.05940914100210648,
    "kitoken": 0.3023805330012692,
    "tiktoken": 0.4114029899938032,
}


def _orders():
    return [list(LIBS[index:]) + list(LIBS[:index]) for index in range(5)]


def _rounds(passes):
    rounds = []
    for index, order in enumerate(_orders()):
        entry = {"round": index + 1, "order": order, "libraries": {}}
        for lib in LIBS:
            seconds = MEDIANS[lib] * (1.0 + 0.03 * (index - 2) / 2.0)
            entry["libraries"][lib] = {
                "seconds": seconds, "process_seconds": seconds,
                "count": COUNT, "checksum": "fc7b9d8422783e7cd873723b32e8aac5a59ecdb69a32307bf6ce56e6b5816335",
                "mib_s": INPUT_BYTES * passes / seconds / chart.MIB,
                "cpu_wall_ratio": 0.994,
            }
        rounds.append(entry)
    return rounds


def _summary(rounds, passes):
    summary = {}
    for lib in LIBS:
        values = [round_entry["libraries"][lib]["seconds"] for round_entry in rounds]
        median, low, high = statistics.median(values), min(values), max(values)
        summary[lib] = {
            "seconds_median": median, "seconds_min": low, "seconds_max": high,
            "mib_s_median": INPUT_BYTES * passes / median / chart.MIB,
            "mib_s_min": INPUT_BYTES * passes / high / chart.MIB,
            "mib_s_max": INPUT_BYTES * passes / low / chart.MIB,
            "rounds": len(rounds), "tokens_per_pass": COUNT / passes,
            "tokens_s_median": COUNT / passes / median,
        }
    return summary


def fixture(passes=10, documents=179):
    rounds = _rounds(passes)
    libraries = {
        lib: {"distribution": DISTRIBUTIONS[lib], "version": VERSIONS[lib],
              "module_version": "1.0.0-rc.4" if lib == "hf1" else None,
              "api": APIS[lib], "special_policy": POLICIES[lib],
              "installed_native_matches_wheel": True,
              "wheel": {"file": WHEELS[lib], "sha256": SHAS[lib], "members": {}}}
        for lib in LIBS
    }
    return {
        "schema_version": 1,
        "task": "encode_one_text_to_ids",
        "model": "gpt2",
        "status": "ok",
        "errors": [],
        "generated_utc": "2026-11-12T05:27:50Z",
        "meta": {
            "claim_scope": "Test scope; not a universal fastest-tokenizer claim.",
            "python": "3.12.13", "platform": "Linux-test", "cpu_model": "AMD Ryzen AI 9 HX 370 w/ Radeon 890M",
            "cpu_count": 24, "load_before": [1.0, 1.0, 1.0],
            "host": {"governors": {"powersave": 24}, "mean_cur_freq_mhz": 1437.1},
            "corpus": {"jsonl": "/tmp/owt_20m.jsonl",
                       "jsonl_sha256": "20dc6601bee15f270b7a2c35cfde6cd2f073f8c15567c76b913453157d2c67db",
                       "jsonl_bytes": 21281368, "seed": 20261111,
                       "documents": documents, "sample_utf8_bytes": INPUT_BYTES,
                       "validation_texts": VALIDATION_TEXTS},
            "tokenizer": {"path": "/tmp/tokenizer.json",
                          "sha256": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
                          "bytes": 1355256},
            "protocol": {"threads": 1, "rounds": 5, "passes": passes, "warmup_passes": 3,
                         "round_order": "latin rotation of the six libraries",
                         "timed_region": "passes x all sample documents",
                         "outside_clock": "import, model load, warmup, checksum"},
            "libraries": libraries,
            "versions": dict(VERSIONS),
            "artifacts": {lib: {"wheel_file": WHEELS[lib], "wheel_sha256": SHAS[lib]}
                          for lib in LIBS},
            "harness": {"script": "scripts/bench_gpt2.py", "sha256": "f9de7e9804acca2d6275c28a348f21855dce699e4143f2dfd3d7cec1ec625a64",
                        "git_head": "8ba81969fbd1cb36cacf6e4d763adea194da4a7c"},
            "api_notes": dict(API_NOTES),
            "tiktoken_vocab_check": {"hf_vocab_entries": 50257, "hf_vocab_non_special_entries": 50256,
                                     "ranks_checked": 50256, "rank_mismatches": 0,
                                     "hf_specials": 1, "special_mismatches": 0,
                                     "tiktoken_n_vocab": 50257},
            "command": ["python3", "scripts/bench_gpt2.py", "--tokie-wheel",
                        "/tmp/opencode/decode-u32-after/dist/" + WHEELS["tokie"],
                        "--output", "docs/benchmarks/uint32-decode/gpt2-hero.json"],
        },
        "validation": {lib: {"ok": True, "vs": "hf023", "texts": VALIDATION_TEXTS,
                             "tokens": 253419, "mismatches": 0, "first_mismatch": None,
                             "checksum": "3f74aa9e68a7433ef98dd5103d70bd97f29c8c1b14043029610ab5932fb882b0"}
                       for lib in LIBS},
        "rounds": rounds,
        "summary": _summary(rounds, passes),
        "ranking_note": "giga has the lowest median, 5.2% below tokie; no robust winner is claimed.",
    }


class Gpt2ChartTests(unittest.TestCase):
    def write(self, payload, directory):
        path = Path(directory) / "gpt2-hero.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def assert_rejected(self, payload, fragment):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(payload, directory)
            with self.assertRaises(ValueError) as caught:
                chart.load_results(path)
            self.assertIn(fragment, str(caught.exception))

    def test_valid_fixture_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            self.assertEqual(dataset["rounds"], 5)
            self.assertEqual(dataset["documents"], 179)
            self.assertEqual(dataset["validation_texts"], 459)
            self.assertEqual(dataset["versions"]["hf1"], "1.0.0rc4")
            self.assertTrue(dataset["tokie_local"])
            self.assertEqual(len(chart._series(dataset)), 6)

    def test_recorded_medians_and_sort(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            rows = chart._series(dataset)
            self.assertEqual([row["lib"] for row in rows],
                             ["giga", "tokie", "hf1", "kitoken", "tiktoken", "hf023"])
            for row in rows:
                self.assertAlmostEqual(row["median"], INPUT_BYTES * 10 / MEDIANS[row["lib"]] / chart.MIB)

    def test_task_is_exact(self):
        payload = fixture()
        payload["task"] = "encode"
        self.assert_rejected(payload, "task must be 'encode_one_text_to_ids'")

    def test_summary_tamper_is_rejected(self):
        payload = fixture()
        payload["summary"]["tokie"]["mib_s_median"] *= 1.02
        self.assert_rejected(payload, "does not match rounds")

    def test_summary_relative_tolerance_is_1e_7(self):
        payload = fixture()
        payload["summary"]["tokie"]["mib_s_median"] *= 1.0 + 5e-8
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(payload, directory))
            self.assertIn("tokie", dataset["data"]["summary"])

    def test_nonfinite_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value):
                payload = fixture()
                payload["rounds"][0]["libraries"]["tokie"]["seconds"] = value
                self.assert_rejected(payload, "not finite")

    def test_missing_kitoken_is_rejected(self):
        payload = fixture()
        del payload["validation"]["kitoken"]
        self.assert_rejected(payload, "Kitoken is required")

    def test_unknown_library_is_rejected(self):
        payload = fixture()
        payload["summary"]["other"] = copy.deepcopy(payload["summary"]["tokie"])
        self.assert_rejected(payload, "unknown libraries")

    def test_validation_mismatch_is_rejected(self):
        payload = fixture()
        payload["validation"]["tokie"]["mismatches"] = 1
        self.assert_rejected(payload, "mismatches must be 0")

    def test_partial_validation_texts_are_rejected(self):
        payload = fixture()
        payload["validation"]["tokie"]["texts"] = 458
        self.assert_rejected(payload, "texts must be 459")

    def test_status_failure_is_rejected(self):
        payload = fixture()
        payload["status"] = "failed"
        self.assert_rejected(payload, "status must be 'ok'")

    def test_stable_hf1_version_is_rejected(self):
        payload = fixture()
        payload["meta"]["versions"]["hf1"] = "1.0.0"
        payload["meta"]["libraries"]["hf1"]["version"] = "1.0.0"
        self.assert_rejected(payload, "prerelease")

    def test_hf1_encode_fast_is_rejected(self):
        payload = fixture()
        payload["meta"]["libraries"]["hf1"]["api"] = "tokenizer.encode_fast(text).ids"
        self.assert_rejected(payload, "compile().encode")

    def test_kitoken_special_flag_is_required(self):
        payload = fixture()
        payload["meta"]["libraries"]["kitoken"]["api"] = "kitoken.Kitoken.from_file(path).encode(text)"
        self.assert_rejected(payload, "encode_specials=True")

    def test_tiktoken_special_flag_is_required(self):
        payload = fixture()
        payload["meta"]["libraries"]["tiktoken"]["api"] = "tiktoken.get_encoding('gpt2').encode(text)"
        self.assert_rejected(payload, "allowed_special")

    def test_tokie_local_evidence_is_required(self):
        payload = fixture()
        payload["meta"]["command"] = ["python3", "scripts/bench_gpt2.py"]
        self.assert_rejected(payload, "tokie local wheel evidence")

    def test_legacy_schema_is_rejected(self):
        payload = fixture()
        payload.pop("rounds")
        payload["benchmark"] = {"gpt2": {}}
        self.assert_rejected(payload, "legacy benchmark schema")

    def test_actual_passes_are_used(self):
        payload = fixture(passes=20)
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(payload, directory))
            self.assertEqual(chart._passes_text(dataset), "20")

    def test_subtitle_uses_metadata(self):
        payload = fixture(documents=42)
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(payload, directory))
            self.assertIn("42 OpenWebText documents", chart._subtitle(dataset))

    def test_giga_marker_and_footnote_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            self.assertEqual(chart._library_label("giga", dataset), "** GigaToken 0.11.0")
            self.assertEqual(chart.GIGA_FOOTNOTE,
                             "** GigaToken does not support all tokenizer types; "
                             "WordPiece and Unigram are excluded.")

    def test_check_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(fixture(), directory)
            out = Path(directory) / "out"
            table = Path(directory) / "table.md"
            code = chart.main(["--results", str(path), "--check",
                               "--output-dir", str(out), "--table-out", str(table)])
            self.assertEqual(code, 0)
            self.assertFalse(out.exists())
            self.assertFalse(table.exists())

    def test_check_reports_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(fixture(), directory)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = chart.main(["--results", str(path), "--check",
                                   "--output-dir", str(Path(directory) / "out"),
                                   "--table-out", str(Path(directory) / "table.md")])
            self.assertEqual(code, 0)
            self.assertIn("6 libraries, 5 rounds, 459 validation texts", buffer.getvalue())

    def test_invalid_input_writes_nothing(self):
        payload = fixture()
        payload["summary"]["tokie"]["mib_s_median"] *= 2
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(payload, directory)
            out = Path(directory) / "out"
            table = Path(directory) / "table.md"
            code = chart.main(["--results", str(path), "--output-dir", str(out),
                               "--table-out", str(table)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())
            self.assertFalse(table.exists())

    def test_missing_results_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            table = Path(directory) / "table.md"
            code = chart.main(["--results", str(Path(directory) / "missing.json"),
                               "--output-dir", str(out), "--table-out", str(table)])
            self.assertEqual(code, 2)
            self.assertFalse(out.exists())
            self.assertFalse(table.exists())

    def test_render_labels_versions_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            out = Path(directory) / "out"
            written = chart.render(dataset, str(out))
            self.assertEqual(len(written), 2)
            self.assertTrue((out / "benchmark_gpt2_main.png").exists())
            svg = (out / "benchmark_gpt2_main.svg").read_text(encoding="utf-8")
            for text in ("GPT-2 encoding", "1 thread", "179 OpenWebText documents",
                         "5-round min", "** GigaToken 0.11.0", "Tokie 0.2.0 (local)",
                         "HF tokenizers 0.23.3", "HF tokenizers 1.0.0rc4",
                         "Kitoken 0.11.1", "tiktoken 0.14.0",
                         "** GigaToken does not support all tokenizer types; "
                         "WordPiece and Unigram are excluded.",
                         "705.85", "744.44", "4.10", "169.73", "33.35", "24.51",
                         "is a prerelease", "local unreleased",
                         "not confidence intervals", "10 passes",
                         "APIs, special-token policies, artifacts and reproduction",
                         "gpt2-hero-table.md"):
                self.assertIn(text, svg)
            self.assertNotIn("encode_fast", svg)
            self.assertNotIn("fastest", svg.lower())

    def test_svg_is_deterministic_and_trimmed(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            first, second = Path(directory) / "one", Path(directory) / "two"
            chart.render(dataset, str(first))
            chart.render(dataset, str(second))
            for name in ("benchmark_gpt2_main.svg", "benchmark_gpt2_main.png"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
            text = (first / "benchmark_gpt2_main.svg").read_text(encoding="utf-8")
            self.assertFalse(any(line != line.rstrip() for line in text.splitlines()))

    def test_table_labels_and_footnote_are_escaped(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            table = Path(directory) / "gpt2-hero-table.md"
            chart.write_table(dataset, str(table))
            text = table.read_text(encoding="utf-8")
            for fragment in (r"\*\* GigaToken 0.11.0",
                             r"\*\* GigaToken does not support all tokenizer types; "
                             r"WordPiece and Unigram are excluded.",
                             "705.845", "744.440", "4.101", "169.731", "33.347", "24.510",
                             "459 texts, 0 mismatches", "not confidence intervals",
                             "local unreleased", "1.0.0rc4", "1,057,339",
                             "encode_specials=True", "allowed_special", "compile()",
                             "Encoding.ids", "numpy uint32", "5.2% below tokie",
                             "3f74aa9e68a7433ef98dd5103d70bd97f29c8c1b14043029610ab5932fb882b0"):
                self.assertIn(fragment, text)
            self.assertNotIn("| ** GigaToken", text)

    def test_footer_is_four_short_lines_and_table_keeps_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = chart.load_results(self.write(fixture(), directory))
            lines = chart._footer_lines(dataset)
            self.assertEqual(len(lines), 4)
            self.assertIn("10 passes/round", lines[0])
            self.assertIn("1.008 MiB input", lines[0])
            self.assertIn("not confidence intervals", lines[0])
            self.assertIn(chart.GIGA_FOOTNOTE, lines[1])
            self.assertIn("HF 1.0.0rc4 is a prerelease", lines[1])
            self.assertIn("8786f91", lines[2])
            self.assertIn("competitors: PyPI wheels", lines[2])
            self.assertIn("Ryzen AI 9 HX 370", lines[2])
            self.assertIn("Linux", lines[2])
            self.assertIn("gpt2-hero-table.md", lines[3])
            for line in lines:
                for method in ("encode_specials", "allowed_special", "compile()"):
                    self.assertNotIn(method, line)
            table = Path(directory) / "gpt2-hero-table.md"
            chart.write_table(dataset, str(table))
            text = table.read_text(encoding="utf-8")
            for method in ("encode_specials=True", "allowed_special", "compile()",
                           "Encoding.ids", "numpy uint32"):
                self.assertIn(method, text)


if __name__ == "__main__":
    unittest.main()
