#!/usr/bin/env python3
"""Cheap unit tests for the GPT-2 hero benchmark harness.

These do not import any native tokenizer library and do not run a benchmark.
    python scripts/test_bench_gpt2.py
"""

import argparse
import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_gpt2 as b


class ChecksumTest(unittest.TestCase):
    def test_stable(self):
        rows = [[1, 2, 3], [], [50256]]
        self.assertEqual(b.ids_checksum(rows), b.ids_checksum(rows))

    def test_row_boundaries_matter(self):
        self.assertNotEqual(b.ids_checksum([[1, 2], [3]]), b.ids_checksum([[1], [2, 3]]))
        self.assertNotEqual(b.ids_checksum([[1, 2, 3]]), b.ids_checksum([[1, 2], [3]]))

    def test_id_change_detected(self):
        self.assertNotEqual(b.ids_checksum([[1, 2, 3]]), b.ids_checksum([[1, 2, 4]]))


class MathTest(unittest.TestCase):
    def test_mib_per_s(self):
        self.assertAlmostEqual(b.mib_per_s(1024 * 1024, 1.0), 1.0)
        self.assertAlmostEqual(b.mib_per_s(2 * 1024 * 1024, 4.0), 0.5)

    def test_summarize(self):
        s = b.summarize_rounds([1.0, 2.0, 3.0], 1024 * 1024)
        self.assertEqual(s["seconds_median"], 2.0)
        self.assertEqual(s["seconds_min"], 1.0)
        self.assertEqual(s["seconds_max"], 3.0)
        self.assertAlmostEqual(s["mib_s_median"], 0.5)
        self.assertEqual(s["rounds"], 3)

    def test_rotate_order_balances_positions(self):
        seen = {lib: [0] * len(b.LIB_ORDER) for lib in b.LIB_ORDER}
        for r in range(len(b.LIB_ORDER)):
            order = b.rotate_order(r)
            self.assertEqual(sorted(order), sorted(b.LIB_ORDER))
            for pos, lib in enumerate(order):
                seen[lib][pos] += 1
        flat = [n for counts in seen.values() for n in counts]
        self.assertLessEqual(max(flat) - min(flat), 1)


class PinTest(unittest.TestCase):
    def test_accepts_exact_pins(self):
        observed = {
            lib: {"dist_version": pin["version"], "module_version": "1.0.0-rc.4"}
            for lib, pin in b.PINS.items()
        }
        self.assertEqual(b.check_pins(observed), [])

    def test_rejects_drift(self):
        observed = {lib: {"dist_version": pin["version"]} for lib, pin in b.PINS.items()}
        observed["kitoken"]["dist_version"] = "0.11.0"
        errors = b.check_pins(observed)
        self.assertTrue(any("kitoken" in e for e in errors))

    def test_hf1_module_version_guard(self):
        observed = {
            lib: {"dist_version": pin["version"], "module_version": "1.0.0-rc.4"}
            for lib, pin in b.PINS.items()
        }
        observed["hf1"]["module_version"] = "0.23.3"
        errors = b.check_pins(observed)
        self.assertTrue(any("hf1" in e for e in errors))

    def test_tokie_wheel_sha_is_pinned(self):
        self.assertEqual(len(b.TOKIE_WHEEL_SHA), 64)
        int(b.TOKIE_WHEEL_SHA, 16)


class ByteAlphabetTest(unittest.TestCase):
    def test_inverse_alphabet_is_complete(self):
        inv = b._byte_alphabet()
        self.assertEqual(len(inv), 256)
        self.assertEqual(len(set(inv.values())), 256)
        self.assertEqual(inv["Ġ"], 32)
        self.assertEqual(inv["Ċ"], 10)


class GoldenPreflightTest(unittest.TestCase):
    def test_reference_counts_pass(self):
        self.assertEqual(b.golden_preflight_errors(
            179, 459, 245492, 253419, b.EXPECTED_VALIDATION_CHECKSUM), [])

    def test_changed_counts_fail(self):
        errors = b.golden_preflight_errors(
            178, 458, 245491, 253418, b.EXPECTED_VALIDATION_CHECKSUM)
        self.assertEqual(len(errors), 4)

    def test_changed_checksum_fails(self):
        errors = b.golden_preflight_errors(
            179, 459, 245492, 253419, "0" * 64)
        self.assertTrue(any("checksum" in e for e in errors))


class FinishArtifactTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="gpt2-finish-"))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.canonical = self.dir / "gpt2-hero.json"

    def _finish(self, status):
        args = argparse.Namespace(output=self.canonical)
        result = {"status": status, "errors": [], "validation": {}, "meta": {"host": {}}}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = b._finish(args, result, [], {}, time.time())
        return rc, json.loads(out.getvalue())

    def test_error_preserves_canonical_and_writes_error_artifact(self):
        self.canonical.write_text("CANONICAL-SENTINEL")
        rc, brief = self._finish("error")
        self.assertEqual(self.canonical.read_text(), "CANONICAL-SENTINEL")
        artifact = self.dir / "gpt2-hero.json.error.json"
        self.assertTrue(artifact.is_file())
        self.assertEqual(json.loads(artifact.read_text())["status"], "error")
        self.assertEqual(rc, 1)
        self.assertEqual(brief["output"], str(artifact))

    def test_invalid_preserves_canonical(self):
        self.canonical.write_text("CANONICAL-SENTINEL")
        rc, _ = self._finish("invalid")
        self.assertEqual(self.canonical.read_text(), "CANONICAL-SENTINEL")
        self.assertEqual(rc, 2)

    def test_success_writes_canonical_unchanged_path(self):
        rc, brief = self._finish("ok")
        self.assertEqual(rc, 0)
        self.assertTrue(self.canonical.is_file())
        self.assertFalse((self.dir / "gpt2-hero.json.error.json").exists())
        self.assertEqual(brief["output"], str(self.canonical))


class TiktokenVocabStubTest(unittest.TestCase):
    class Encoding:
        def __init__(self, ranks, specials):
            self._mergeable_ranks = ranks
            self._special_tokens = specials
            self.n_vocab = len(ranks) + len(specials)

    def _tokenizer(self, rank_id=1, special_id=2):
        tmp = Path(tempfile.mkdtemp(prefix="gpt2-vocab-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        path = tmp / "tokenizer.json"
        path.write_text(json.dumps({
            "model": {"vocab": {"a": 0, "ab": rank_id, "<|endoftext|>": 2}},
            "added_tokens": [{"content": "<|endoftext|>", "id": 2, "special": True}],
        }))
        return path

    def test_rank_and_special_match(self):
        enc = self.Encoding({b"a": 0, b"ab": 1}, {"<|endoftext|>": 2})
        check = b._tiktoken_vocab_check(self._tokenizer(), enc)
        self.assertTrue(check["ok"])
        self.assertEqual(check["ranks_checked"], 2)
        self.assertEqual(check["rank_mismatches"], 0)
        self.assertEqual(check["special_mismatches"], 0)

    def test_rank_and_special_mismatch_is_false(self):
        enc = self.Encoding({b"a": 0, b"ab": 99}, {"<|endoftext|>": 3})
        check = b._tiktoken_vocab_check(self._tokenizer(), enc)
        self.assertFalse(check["ok"])
        self.assertEqual(check["rank_mismatches"], 1)
        self.assertEqual(check["special_mismatches"], 1)
        self.assertIsNotNone(check["first_rank_mismatch"])
        self.assertIsNotNone(check["first_special_mismatch"])


class RawProofTest(unittest.TestCase):
    """The archived production run must keep its 50256-rank proof."""

    def test_archived_tiktoken_proof(self):
        path = (Path(__file__).resolve().parent.parent
                / "docs/benchmarks/uint32-decode/gpt2-hero.json")
        raw = json.loads(path.read_text())
        check = raw["meta"]["tiktoken_vocab_check"]
        self.assertEqual(check["ranks_checked"], 50256)
        self.assertEqual(check["rank_mismatches"], 0)
        self.assertEqual(check["special_mismatches"], 0)
        self.assertEqual(raw["validation"]["hf023"]["checksum"],
                         b.EXPECTED_VALIDATION_CHECKSUM)


class PythonMapTest(unittest.TestCase):
    def test_defaults(self):
        m = b.parse_python_map([], "/usr/bin/python3")
        self.assertEqual(set(m), set(b.LIB_ORDER))
        self.assertTrue(all(v == "/usr/bin/python3" for v in m.values()))

    def test_rejects_unknown_and_duplicate(self):
        with self.assertRaises(ValueError):
            b.parse_python_map(["nope=/bin/sh"], "/usr/bin/python3")
        with self.assertRaises(ValueError):
            b.parse_python_map(["tokie=/bin/sh", "tokie=/bin/sh"], "/usr/bin/python3")

    def test_rejects_bad_spec(self):
        with self.assertRaises(ValueError):
            b.parse_python_map(["tokie"], "/usr/bin/python3")


class MismatchTest(unittest.TestCase):
    def test_no_mismatch(self):
        self.assertIsNone(b._first_mismatch([[1, 2]], [[1, 2]]))

    def test_first_id_and_length_mismatch(self):
        first = b._first_mismatch([[1, 2], [3]], [[1, 2], [4]])
        self.assertEqual(first["index"], 1)
        self.assertEqual(first["got_first12"], [3])
        first = b._first_mismatch([[1]], [[1], [2]])
        self.assertEqual(first["index"], 1)
        self.assertIn("row count", first["reason"])


if __name__ == "__main__":
    unittest.main()
