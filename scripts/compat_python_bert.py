#!/usr/bin/env python3
"""BERT entry point for the frozen compat_python_bench.py.

This imports the unchanged six-model driver, injects MODELS['bert'] from the
untimed c59c8c8 reference, and wraps only its output emitter so corpus plaintext
is not written.  All measurement functions are the frozen driver's.

BERT is WordPiece; gigatoken has no comparable WordPiece path, so this is a
Python before/after measurement only.  The same wrapper is used for all models
in the paired runner, with an explicit --model.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import compat_python_bench as bench  # noqa: E402

REFERENCE = Path("/tmp/opencode/tokie-compat/python-bert-reference.json")
if not REFERENCE.is_file():
    raise SystemExit(f"missing BERT reference: {REFERENCE}")
_ref = json.loads(REFERENCE.read_text())
if _ref.get("model") != "bert":
    raise SystemExit(f"{REFERENCE}: expected model bert")
bench.MODELS["bert"] = (
    _ref["path"],
    _ref["sha256"],
    (_ref["joined"]["tokens"], _ref["joined"]["fnv1a64"]),
    (_ref["batch"]["tokens"], _ref["batch"]["fnv1a64"]),
)

_original_emit = bench.emit


def compact_emit(output, path):
    """Drop only the corpus plaintext, after all timers have stopped."""
    corpus = output.get("corpus")
    if isinstance(corpus, dict):
        output["corpus"] = {
            key: value for key, value in corpus.items() if key not in ("docs", "joined")
        }
    return _original_emit(output, path)


def install_emit():
    bench.emit = compact_emit


def main():
    if "--model" not in sys.argv:
        sys.argv[1:1] = ["--model", "bert"]
    install_emit()
    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
