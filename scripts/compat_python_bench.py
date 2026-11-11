#!/usr/bin/env python3
"""Benchmark the mainline tokie Python bindings on the OpenWebText corpus.

Only the real binding surface is used: ``Tokenizer.open``, ``threads``,
``encode`` (raw uint32 ndarray), ``batch`` (ids, ends), and ``decode``.  There
is no legacy facade here.

Modes:
  scalar  one ``encode(doc)`` per document, in order
  batch   one ``batch(docs)`` call
  decode  one ``decode(joined_ids)`` call over precomputed joined-file ids

The timed region is only the binding call.  Scalar time is the sum of per-call
timers, so it measures per-call encode/PyO3 overhead but not the loop's
bookkeeping between calls; scalar and batch have different loop shapes and are
not a head-to-head end-to-end script comparison.  Every result is consumed
after the clock stops: token counts, id/end hashes, or decoded-text hash.  The
id buffer is dropped before the next timed call so the binding's pooled Vec is
reused rather than allocated under the timer.

Warmup is 3 and repeats 9 by default (minimum 3/7).  ``--cold`` uses a fresh
process, no warmup, and one timed call.  Every warm run validates first;
``--validate-only`` validates and exits.  Validation checks scalar concat ==
batch ids/ends and the Rust batch baseline; decode is checked against the
baseline id checksum and an optional decoded-text checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_DIR = Path("/tmp/opencode/tokie-compat/corpus")

# name: (tokenizer.json, sha256, (joined tokens, FNV-1a), (batch tokens, FNV-1a))
# The FNV values are the c59c8c8 Rust baseline over the same ids.
MODELS: dict[str, tuple[str, str, tuple[int, str], tuple[int, str]]] = {
    "gpt2": (
        "/home/bhavnick/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
        "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
        (4_788_850, "7110978460d2cf45"),
        (4_784_664, "902042f765b30f69"),
    ),
    "cl100k": (
        "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4/snapshots/1d9f1f1b1fae88c0e4df1dab0a397f8de6229075/tokenizer.json",
        "239eb2359f79c38497476671aaa835e01fb43d42743c612a8514a0dfa2ac93a2",
        (4_534_217, "bae6b0b829eb4ab6"),
        (4_533_313, "716868a48f5efd86"),
    ),
    "o200k": (
        "/home/bhavnick/.cache/huggingface/hub/models--Xenova--gpt-4o/snapshots/7956d98f2a83b2751a98ea7136fdf7fe6cf54e69/tokenizer.json",
        "43a3ad4618a6a938f8c2614154ea10c31ad53a62d5683ae0b0e6133575cef07e",
        (4_497_370, "722b2e371eb49fd7"),
        (4_496_498, "e777a7f3b1bd309f"),
    ),
    "qwen2": (
        "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen2-7B/snapshots/453ed1575b739b5b03ce3758b23befdb0967f40e/tokenizer.json",
        "f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8",
        (4_645_050, "42ea496ddef99cfc"),
        (4_644_146, "6ec969d6e4a01923"),
    ),
    "qwen3.5": (
        "/home/bhavnick/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/tokenizer.json",
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        (4_697_217, "b65105063c1ba14d"),
        (4_693_031, "fb0556d5ec02dbf5"),
    ),
    "deepseek": (
        "/home/bhavnick/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/snapshots/e815299b0bcbac849fa540c768ef21845365c9eb/tokenizer.json",
        "621ac2e32d0dba658404412318818aaa8ce8cda492e59830109d8da6b517fb41",
        (4_580_467, "5780fc3975562993"),
        (4_579_593, "e8b5530c9fdafa41"),
    ),
}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(array: Any) -> str:
    return sha256_bytes(array.tobytes())


def fnv1a64_u32(ids: Any) -> str:
    h = 0xCBF29CE484222325
    for value in ids.tolist():
        h = ((h ^ int(value)) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def median_range(samples: list[float]) -> tuple[float, float, float]:
    ordered = sorted(samples)
    return ordered[len(ordered) // 2], ordered[0], ordered[-1]


@dataclass(frozen=True)
class Corpus:
    docs: list[str]
    joined: str
    doc_bytes: int
    joined_bytes: int
    jsonl_sha256: str
    joined_sha256: str


def load_corpus() -> Corpus:
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    jsonl_bytes = (CORPUS_DIR / "owt_20m.jsonl").read_bytes()
    txt_bytes = (CORPUS_DIR / "owt_20m.txt").read_bytes()
    jsonl_sha = sha256_bytes(jsonl_bytes)
    joined_sha = sha256_bytes(txt_bytes)
    if jsonl_sha != manifest["jsonl_sha256"]:
        raise SystemExit("corpus jsonl sha256 mismatch")
    if joined_sha != manifest["joined_text_sha256"]:
        raise SystemExit("corpus joined sha256 mismatch")
    docs = [json.loads(line)["text"] for line in jsonl_bytes.decode().splitlines()]
    joined = txt_bytes.decode()
    doc_bytes = sum(len(doc.encode()) for doc in docs)
    if "\n".join(docs) != joined:
        raise SystemExit("corpus txt is not jsonl documents joined by one newline")
    if len(docs) != manifest["documents"] or doc_bytes != manifest["text_bytes"]:
        raise SystemExit("corpus document count/bytes mismatch")
    if len(txt_bytes) != manifest["joined_text_bytes"]:
        raise SystemExit("corpus joined bytes mismatch")
    log(f"corpus: {len(docs)} docs, {doc_bytes} doc bytes, {len(txt_bytes)} joined bytes")
    return Corpus(docs, joined, doc_bytes, len(txt_bytes), jsonl_sha, joined_sha)


def resolve_model(name: str) -> tuple[Path, str]:
    path = Path(MODELS[name][0])
    if not path.is_file():
        raise SystemExit(f"tokenizer.json not found: {path}")
    actual = sha256_file(path)
    if actual != MODELS[name][1]:
        raise SystemExit(f"{path}: sha256 {actual} != baseline {MODELS[name][1]}")
    log(f"tokenizer: {name} {path} sha256 {actual[:16]}")
    return path, actual


def binding_provenance(tokie: Any, np: Any) -> dict[str, Any]:
    module = Path(tokie.__file__).resolve()
    if module.suffix == ".so":
        binary = module
    else:
        candidates = sorted(module.parent.glob("*.so"))
        preferred = [p for p in candidates if p.name.startswith(module.stem)]
        binary = (preferred or candidates or [module])[0]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "tokie_version": importlib.metadata.version("tokie"),
        "tokie_binary": str(binary),
        "tokie_binary_sha256": sha256_file(binary),
        "tokie_binary_bytes": binary.stat().st_size,
        "cpus_allowed": ",".join(str(c) for c in sorted(os.sched_getaffinity(0))),
    }


def run_scalar(tokenizer: Any, np: Any, docs: list[str]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tokens = 0
    ends: list[int] = []
    secs = 0.0
    for doc in docs:
        start = time.perf_counter()
        ids = tokenizer.encode(doc)
        secs += time.perf_counter() - start
        count = int(ids.shape[0])
        tokens += count
        ends.append(tokens)
        digest.update(ids.tobytes())
        del ids  # return the buffer to the tokenizer pool before the next timed call
    return {
        "secs": secs,
        "tokens": tokens,
        "ids_sha256": digest.hexdigest(),
        "ends_sha256": array_sha256(np.asarray(ends, dtype=np.uint64)),
        "rows": len(docs),
    }


def run_batch(tokenizer: Any, np: Any, docs: list[str]) -> dict[str, Any]:
    del np
    start = time.perf_counter()
    ids, ends = tokenizer.batch(docs)
    secs = time.perf_counter() - start
    if len(ends) != len(docs) or (len(ends) and int(ends[-1]) != len(ids)):
        raise SystemExit("batch ends do not match the documents")
    if len(ends) > 1 and any(b < a for a, b in zip(ends.tolist(), ends.tolist()[1:])):
        raise SystemExit("batch ends are not monotonic")
    result = {
        "secs": secs,
        "tokens": int(ids.shape[0]),
        "ids_sha256": array_sha256(ids),
        "ends_sha256": array_sha256(ends),
        "rows": len(docs),
    }
    del ids, ends  # next sample starts with a free pooled buffer
    return result


def run_decode(tokenizer: Any, ids: Any) -> dict[str, Any]:
    start = time.perf_counter()
    text = tokenizer.decode(ids)
    secs = time.perf_counter() - start
    text_bytes = text.encode("utf-8")
    result = {
        "secs": secs,
        "decoded_sha256": sha256_bytes(text_bytes),
        "decoded_bytes": len(text_bytes),
    }
    del text, text_bytes
    return result


def validate_encode(tokenizer: Any, np: Any, corpus: Corpus, args: argparse.Namespace) -> dict[str, Any]:
    log("validate: scalar per-document encode")
    parts = []
    ends: list[int] = []
    total = 0
    for doc in corpus.docs:
        ids = np.ascontiguousarray(tokenizer.encode(doc), dtype=np.uint32)
        parts.append(ids)
        total += int(ids.shape[0])
        ends.append(total)
    scalar_ids = np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)
    del parts
    scalar_ends = np.asarray(ends, dtype=np.uint64)

    log(f"validate: batch encode, {args.threads} thread(s)")
    tokenizer.threads(args.threads)
    batch_ids, batch_ends = tokenizer.batch(corpus.docs)
    batch_ids = np.ascontiguousarray(batch_ids, dtype=np.uint32)
    batch_ends = np.ascontiguousarray(batch_ends, dtype=np.uint64)
    if len(batch_ends) != len(corpus.docs) or int(batch_ends[-1]) != len(batch_ids):
        raise SystemExit("batch ends length/last value mismatch")
    if not np.array_equal(batch_ends, scalar_ends):
        raise SystemExit("batch ends differ from scalar cumulative lengths")
    if not np.array_equal(batch_ids, scalar_ids):
        raise SystemExit("batch ids differ from scalar concatenation")

    expected_tokens, expected_fnv = MODELS[args.model][3]
    if len(batch_ids) != expected_tokens or fnv1a64_u32(batch_ids) != expected_fnv:
        raise SystemExit("batch ids differ from the Rust c59c8c8 baseline")
    log(f"validate: exact Rust batch baseline, {len(batch_ids)} ids")
    return {
        "mode": args.mode,
        "threads": args.threads,
        "documents": len(corpus.docs),
        "tokens": len(batch_ids),
        "ids_sha256": array_sha256(batch_ids),
        "ends_sha256": array_sha256(batch_ends),
        "scalar_batch_equal": True,
        "rust_batch_baseline_match": True,
    }


def validate_decode(tokenizer: Any, ids: Any, corpus: Corpus, args: argparse.Namespace) -> dict[str, Any]:
    log(f"reference decode: {len(ids)} joined ids")
    decoded = tokenizer.decode(ids)
    decoded_bytes = decoded.encode("utf-8")
    digest = sha256_bytes(decoded_bytes)
    matches = decoded == corpus.joined
    if args.decode_expected_sha256 and digest != args.decode_expected_sha256.lower():
        raise SystemExit("decoded text sha256 does not match --decode-expected-sha256")
    second = tokenizer.decode(ids)
    if sha256_bytes(second.encode("utf-8")) != digest:
        raise SystemExit("decode output is not stable across two calls")
    log(f"validate: {len(decoded_bytes)} utf-8 bytes, sha256 {digest[:16]}, roundtrip {matches}")
    return {
        "mode": "decode",
        "tokens": len(ids),
        "ids_sha256": array_sha256(ids),
        "decoded_sha256": digest,
        "decoded_bytes": len(decoded_bytes),
        "decoded_matches_input": matches,
        "decode_stable": True,
    }


def load_ids(tokenizer: Any, np: Any, corpus: Corpus, args: argparse.Namespace) -> tuple[Any, str]:
    path = Path(args.ids_file).expanduser()
    if path.is_file():
        ids = np.ascontiguousarray(np.load(path), dtype=np.uint32)
        if ids.ndim != 1:
            raise SystemExit(f"{path}: expected a 1-D uint32 array")
        source = f"loaded:{path}"
        log(f"decode ids: loaded {len(ids)} from {path}")
    else:
        ids = np.ascontiguousarray(tokenizer.encode(corpus.joined), dtype=np.uint32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, ids)
        source = f"computed+save:{path}"
        log(f"decode ids: computed {len(ids)}, saved {path}")
    expected_tokens, expected_fnv = MODELS[args.model][2]
    if len(ids) != expected_tokens or fnv1a64_u32(ids) != expected_fnv:
        raise SystemExit("joined ids differ from the Rust c59c8c8 baseline")
    log(f"decode ids: exact Rust baseline, {len(ids)} ids")
    return ids, source


def measure_encode(tokenizer: Any, np: Any, corpus: Corpus, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer.threads(args.threads)
    run_once = run_scalar if args.mode == "scalar" else run_batch
    expected: dict[str, Any] | None = None
    for warm in range(args.warmup):
        sample = run_once(tokenizer, np, corpus.docs)
        expected = expected or sample
        log(f"warmup {warm + 1}/{args.warmup}: {sample['tokens']} ids, {sample['secs']:.6f}s")
    times: list[float] = []
    for rep in range(args.reps):
        sample = run_once(tokenizer, np, corpus.docs)
        expected = expected or sample
        for key in ("tokens", "ids_sha256", "ends_sha256"):
            if sample[key] != expected[key]:
                raise SystemExit(f"encode {key} changed on rep {rep}")
        times.append(sample["secs"])
        log(f"rep {rep + 1}/{args.reps}: {sample['tokens']} ids, {sample['secs']:.6f}s")
    assert expected is not None
    median, minimum, maximum = median_range(times)
    return {
        "label": args.label,
        "model": args.model,
        "mode": args.mode,
        "threads": args.threads,
        "reps": args.reps,
        "warmup": args.warmup,
        "cold": args.cold,
        "rows": len(corpus.docs),
        "input_bytes": corpus.doc_bytes,
        "tokens": expected["tokens"],
        "ids_sha256": expected["ids_sha256"],
        "ends_sha256": expected["ends_sha256"],
        "times_sec": times,
        "median_sec": median,
        "min_sec": minimum,
        "max_sec": maximum,
        "mb_per_s": corpus.doc_bytes / 1e6 / median,
        "tokens_per_s": expected["tokens"] / median,
    }


def measure_decode(tokenizer: Any, corpus: Corpus, ids: Any, source: str, reference: str | None, args: argparse.Namespace) -> dict[str, Any]:
    expected = reference
    decoded_bytes = 0
    for warm in range(args.warmup):
        sample = run_decode(tokenizer, ids)
        expected = expected or sample["decoded_sha256"]
        if sample["decoded_sha256"] != expected:
            raise SystemExit("decoded text changed during warmup")
        decoded_bytes = sample["decoded_bytes"]
        log(f"warmup {warm + 1}/{args.warmup}: {decoded_bytes} bytes, {sample['secs']:.6f}s")
    times: list[float] = []
    for rep in range(args.reps):
        sample = run_decode(tokenizer, ids)
        expected = expected or sample["decoded_sha256"]
        if sample["decoded_sha256"] != expected:
            raise SystemExit(f"decoded text changed on rep {rep}")
        decoded_bytes = sample["decoded_bytes"]
        times.append(sample["secs"])
        log(f"rep {rep + 1}/{args.reps}: {decoded_bytes} bytes, {sample['secs']:.6f}s")
    assert expected is not None
    if args.decode_expected_sha256 and expected != args.decode_expected_sha256.lower():
        raise SystemExit("decoded text sha256 does not match --decode-expected-sha256")
    joined_sha = sha256_bytes(corpus.joined.encode("utf-8"))
    median, minimum, maximum = median_range(times)
    return {
        "label": args.label,
        "model": args.model,
        "mode": "decode",
        "threads": 1,
        "reps": args.reps,
        "warmup": args.warmup,
        "cold": args.cold,
        "rows": 1,
        "input_bytes": corpus.joined_bytes,
        "tokens": len(ids),
        "ids_sha256": array_sha256(ids),
        "ids_source": source,
        "decoded_sha256": expected,
        "decoded_bytes": decoded_bytes,
        "decoded_matches_input": expected == joined_sha,
        "times_sec": times,
        "median_sec": median,
        "min_sec": minimum,
        "max_sec": maximum,
        "mb_per_s": corpus.joined_bytes / 1e6 / median,
        "tokens_per_s": len(ids) / median,
        "decoded_bytes_per_s": decoded_bytes / median,
    }


def emit(output: dict[str, Any], path: str | None) -> None:
    pretty = json.dumps(output, indent=2)
    print(pretty)
    if path:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(pretty + "\n")
        log(f"wrote {p}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="tokie Python binding benchmark")
    p.add_argument("--model", choices=sorted(MODELS), default="gpt2")
    p.add_argument("--mode", choices=("scalar", "batch", "decode"), default="scalar")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=9)
    p.add_argument("--cold", action="store_true")
    p.add_argument("--label", default="")
    p.add_argument("--json-output", default=None)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--ids-file", default=None)
    p.add_argument("--decode-expected-sha256", default=None)
    args = p.parse_args(argv)

    if args.threads < 1:
        p.error("--threads must be positive")
    if args.mode == "decode":
        if args.threads != 1:
            p.error("decode is serial; use --threads 1")
        if args.ids_file is None:
            p.error("decode needs --ids-file")
    elif args.ids_file or args.decode_expected_sha256:
        p.error("decode-only flags used with an encode mode")
    if args.cold:
        args.warmup, args.reps = 0, 1
    elif not args.validate_only and (args.warmup < 3 or args.reps < 7):
        p.error("warmup must be >=3 and reps >=7 (or use --cold)")
    args.label = args.label or f"{args.mode}-{args.model}-{args.threads}t"
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = load_corpus()
    tokenizer_path, tokenizer_sha = resolve_model(args.model)

    import numpy as np
    import tokie

    provenance = binding_provenance(tokie, np)
    log(f"binding: tokie {provenance['tokie_version']} binary {provenance['tokie_binary']} sha256 {provenance['tokie_binary_sha256']}")
    tokenizer = tokie.Tokenizer.open(str(tokenizer_path))

    if args.mode in ("scalar", "batch"):
        tokenizer.threads(args.threads)
        validation = validate_encode(tokenizer, np, corpus, args) if (args.validate_only or not args.cold) else None
        if args.validate_only:
            emit({"generated_unix": int(time.time()), "label": args.label, "model": args.model,
                  "corpus": corpus.__dict__, "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_sha},
                  "python": provenance, "validation": validation}, args.json_output)
            return 0
        result = measure_encode(tokenizer, np, corpus, args)
    else:
        ids, source = load_ids(tokenizer, np, corpus, args)
        validation = validate_decode(tokenizer, ids, corpus, args) if (args.validate_only or not args.cold) else None
        if args.validate_only:
            if validation is not None:
                validation["ids_source"] = source
            emit({"generated_unix": int(time.time()), "label": args.label, "model": args.model,
                  "corpus": corpus.__dict__, "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_sha},
                  "python": provenance, "validation": validation}, args.json_output)
            return 0
        result = measure_decode(tokenizer, corpus, ids, source, validation["decoded_sha256"] if validation else None, args)

    emit({"generated_unix": int(time.time()), "label": args.label, "model": args.model,
          "corpus": corpus.__dict__, "tokenizer": {"path": str(tokenizer_path), "sha256": tokenizer_sha},
          "python": provenance, "results": [result]}, args.json_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
