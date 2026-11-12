#!/usr/bin/env python3
"""Unit tests for summarize units and the batch count contract."""

import statistics
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_current as bc


def make_rounds(seconds, libs=("tokie",)):
    return [{"round": i + 1, "order": list(libs),
             **{lib: {"seconds": s, "count": 100} for lib in libs}}
            for i, s in enumerate(seconds)]


class SummarizeUnitTests(unittest.TestCase):
    def test_ids_emit_tokens_not_chars(self):
        out = bc.summarize(make_rounds([0.10, 0.20, 0.30]), ["tokie"],
                           1 << 20, 100, 10, "ids")["tokie"]
        self.assertEqual(out["count_unit"], "ids")
        self.assertIn("tokens_s_median", out)
        self.assertNotIn("chars_s_median", out)
        self.assertIn("count_s_median", out)

    def test_characters_emit_chars_not_tokens(self):
        out = bc.summarize(make_rounds([0.10, 0.20, 0.30]), ["tokie"],
                           1 << 20, 100, 10, "characters")["tokie"]
        self.assertEqual(out["count_unit"], "characters")
        self.assertIn("chars_s_median", out)
        self.assertNotIn("tokens_s_median", out)
        self.assertIn("count_s_median", out)

    def test_ids_and_documents_emit_neither(self):
        out = bc.summarize(make_rounds([0.10, 0.20, 0.30]), ["tokie"],
                           1 << 20, 5, 10, "ids_and_documents")["tokie"]
        self.assertEqual(out["count_unit"], "ids_and_documents")
        self.assertNotIn("tokens_s_median", out)
        self.assertNotIn("chars_s_median", out)
        self.assertIn("count_s_median", out)

    def test_twenty_rounds_median_and_range(self):
        secs = [0.50 + i * 0.01 for i in range(20)]
        out = bc.summarize(make_rounds(secs), ["tokie"], 2 << 20, 100, 10, "ids")["tokie"]
        med = statistics.median(secs)
        self.assertEqual(out["seconds_median"], med)
        self.assertEqual(out["seconds_min"], min(secs))
        self.assertEqual(out["seconds_max"], max(secs))
        self.assertAlmostEqual(out["tokens_s_median"], 100 * 10 / med)
        self.assertAlmostEqual(out["mib_s_median"], (2 << 20) * 10 / med / (1 << 20))


class BatchCountTests(unittest.TestCase):
    def test_hf_batch_count_is_ids_plus_documents(self):
        out = [[10, 11, 12], [20]]
        self.assertEqual(bc.count_batch("hf", out), 4 + 2)

    def test_hf_empty_rows_still_count_documents(self):
        self.assertEqual(bc.count_batch("hf", [[], []]), 2)
        self.assertEqual(bc.count_batch("hf", []), 0)


class BatchMismatchFieldTests(unittest.TestCase):
    def test_batch_not_run_omits_key(self):
        result = bc.batch_mismatch_result(False, 0)
        validation = {"ok": True, **result}
        stats = {"small_batch_run": False, **result}
        self.assertNotIn("batch_mismatches", validation)
        self.assertNotIn("batch_mismatches", stats)

    def test_batch_validated_zero_emits_key(self):
        result = bc.batch_mismatch_result(True, 0)
        validation = {"ok": True, **result}
        stats = {"small_batch_run": True, **result}
        self.assertEqual(validation["batch_mismatches"], 0)
        self.assertEqual(stats["batch_mismatches"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
