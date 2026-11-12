#!/usr/bin/env python3
"""Six-library GPT-2 encode-to-ids hero benchmark.

One model (GPT-2), one task (single text -> token ids), one thread, one host.
Every library returns its public id output, and every id is checked against
tokenizers 0.23.3 on the same tokenizer.json. Timing is five globally
interleaved rounds; each library runs in a fresh child process per round, so
import, model load, and warmup happen outside the timed region.

    python scripts/bench_gpt2.py \
        --corpus /tmp/opencode/tokie-compat/corpus/owt_20m.jsonl \
        --tokenizer /path/gpt2/tokenizer.json \
        --tokie-wheel /tmp/opencode/decode-u32-after/dist/tokie-0.2.0-cp312-cp312-manylinux_2_34_x86_64.whl \
        --wheels-dir /tmp/opencode/gpt2-hero/wheels \
        --python tokie=/tmp/opencode/gpt2-hero/envs/stable/bin/python \
        --python giga=/tmp/opencode/gpt2-hero/envs/stable/bin/python \
        --python hf023=/tmp/opencode/gpt2-hero/envs/stable/bin/python \
        --python hf1=/tmp/opencode/gpt2-hero/envs/hf1/bin/python \
        --python kitoken=/tmp/opencode/gpt2-hero/envs/stable/bin/python \
        --python tiktoken=/tmp/opencode/gpt2-hero/envs/stable/bin/python \
        --output docs/benchmarks/uint32-decode/gpt2-hero.json

The parent only orchestrates. Each child is invoked as
``bench_gpt2.py --worker LIB --mode {validate,time} ...`` and writes JSON.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from bench_published import (
    EDGE_TEXTS,
    build_fuzz,
    cpu_model,
    host_snapshot,
    load_corpus,
    sha256_file,
)

SCHEMA_VERSION = 1
LIB_ORDER = ["tokie", "giga", "hf023", "hf1", "kitoken", "tiktoken"]

# Exact public versions under test, pinned; the parent rejects anything else.
PINS = {
    "tokie": {"distribution": "tokie", "version": "0.2.0"},
    "giga": {"distribution": "gigatoken", "version": "0.11.0"},
    "hf023": {"distribution": "tokenizers", "version": "0.23.3"},
    "hf1": {"distribution": "tokenizers", "version": "1.0.0rc4"},
    "kitoken": {"distribution": "kitoken", "version": "0.11.1"},
    "tiktoken": {"distribution": "tiktoken", "version": "0.14.0"},
}

WHEEL_GLOBS = {
    "giga": "gigatoken-0.11.0-*.whl",
    "hf023": "tokenizers-0.23.3-*.whl",
    "hf1": "tokenizers-1.0.0rc4-*.whl",
    "kitoken": "kitoken-0.11.1-*.whl",
    "tiktoken": "tiktoken-0.14.0-*.whl",
}

TOKIE_WHEEL_SHA = "8786f91b1aed3acbac9d8090234e55f0400f829da6ea8a2817bcd460332a1d85"
TOKENIZER_SHA = "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6"
EXPECTED_TEXTS = 179 + len(EDGE_TEXTS) + 256
MIN_SAMPLE_BYTES = 1048576
EXPECTED_SAMPLE_DOCS = 179
EXPECTED_SAMPLE_TOKENS = 245492
EXPECTED_VALIDATION_TOKENS = 253419
EXPECTED_VALIDATION_CHECKSUM = (
    "3f74aa9e68a7433ef98dd5103d70bd97f29c8c1b14043029610ab5932fb882b0"
)

API = {
    "tokie": "tokie.Tokenizer.open(path); threads(1); tokenizer.encode(text) -> numpy uint32 ids; len(ids) in clock",
    "giga": "gigatoken.Tokenizer(path); tokenizer.encode(text) -> numpy uint32 ids; len(ids) in clock",
    "hf023": "tokenizers.Tokenizer.from_file(path); tokenizer.encode(text, add_special_tokens=False).ids; len(ids) in clock",
    "hf1": "tokenizers.Tokenizer.from_file(path).compile(); compiled.encode(text, add_special_tokens=False).ids; len(ids) in clock",
    "kitoken": "kitoken.Kitoken.from_file(path); tokenizer.encode(text, encode_specials=True) -> list[int]; len(ids) in clock",
    "tiktoken": "tiktoken.get_encoding('gpt2'); encoding.encode(text, allowed_special={'<|endoftext|>'}) -> list[int]; len(ids) in clock",
}

SPECIAL_POLICY = {
    "tokie": "recognizes '<|endoftext|>' as id 50256 from the HF tokenizer.json",
    "giga": "recognizes '<|endoftext|>' as id 50256 from the HF tokenizer.json",
    "hf023": "add_special_tokens=False adds nothing; an existing '<|endoftext|>' literal still maps to 50256",
    "hf1": "add_special_tokens=False adds nothing; encode_special_tokens defaults to False, so the literal maps to 50256",
    "kitoken": "encode_specials=True explicitly, because the default False encodes '<|endoftext|>' as regular text (7 ids) and cannot match the HF ids",
    "tiktoken": "allowed_special={'<|endoftext|>'} explicitly; the default raises on that literal",
}

# A one-thread run needs every thread pool pinned, not just Rayon. numpy's
# OpenBLAS otherwise starts 24 spinning threads in the tokie/gigatoken
# children (they return numpy arrays), which burns CPU during the timed region.
THREAD_ENV = {
    "RAYON_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def ids_checksum(rows):
    """Deterministic checksum over every row and every id."""
    h = hashlib.sha256()
    for row in rows:
        h.update(len(row).to_bytes(8, "little"))
        if row:
            h.update(struct.pack("<%dI" % len(row), *row))
    return h.hexdigest()


def mib_per_s(utf8_bytes, seconds):
    return (utf8_bytes / (1024.0 * 1024.0)) / seconds


def summarize_rounds(round_seconds, utf8_bytes):
    vals = list(round_seconds)
    mibs = [mib_per_s(utf8_bytes, x) for x in vals]
    return {
        "seconds_median": statistics.median(vals),
        "seconds_min": min(vals),
        "seconds_max": max(vals),
        "mib_s_median": statistics.median(mibs),
        "mib_s_min": min(mibs),
        "mib_s_max": max(mibs),
        "rounds": len(vals),
    }


def rotate_order(round_index):
    r = round_index % len(LIB_ORDER)
    return LIB_ORDER[r:] + LIB_ORDER[:r]


def parse_python_map(specs, default):
    out = {lib: default for lib in LIB_ORDER}
    seen = set()
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise ValueError(f"bad --python value {spec!r}; expected LIB=PATH")
        if label not in LIB_ORDER:
            raise ValueError(f"unknown library in --python: {label!r}")
        if label in seen:
            raise ValueError(f"duplicate --python for {label!r}")
        if not Path(path).is_file():
            raise ValueError(f"--python {label}: not a file: {path}")
        seen.add(label)
        out[label] = path
    return out


def check_pins(observed):
    errors = []
    for lib, pin in PINS.items():
        got = observed.get(lib, {}).get("dist_version")
        if got != pin["version"]:
            errors.append(
                f"{lib}: expected {pin['distribution']}=={pin['version']}, "
                f"installed {got!r}"
            )
        if lib == "hf1" and got == pin["version"]:
            module_version = observed[lib].get("module_version")
            if not module_version or not module_version.startswith("1.0.0"):
                errors.append(f"hf1: tokenizers module reports {module_version!r}, expected 1.0.x")
    return errors


def golden_preflight_errors(n_sample, n_validation, sample_tokens,
                            validation_tokens, validation_checksum):
    """Reject a changed corpus/slice before any timing starts."""
    errors = []
    if n_sample != EXPECTED_SAMPLE_DOCS:
        errors.append(f"sample docs {n_sample} != {EXPECTED_SAMPLE_DOCS}")
    if n_validation != EXPECTED_TEXTS:
        errors.append(f"validation texts {n_validation} != {EXPECTED_TEXTS}")
    if sample_tokens != EXPECTED_SAMPLE_TOKENS:
        errors.append(f"sample ids {sample_tokens} != {EXPECTED_SAMPLE_TOKENS}")
    if validation_tokens != EXPECTED_VALIDATION_TOKENS:
        errors.append(f"validation ids {validation_tokens} != {EXPECTED_VALIDATION_TOKENS}")
    if validation_checksum != EXPECTED_VALIDATION_CHECKSUM:
        errors.append(
            f"validation checksum {validation_checksum} != {EXPECTED_VALIDATION_CHECKSUM}"
        )
    return errors


def _sample_texts(corpus, seed, max_chars):
    _docs, picked, chars = load_corpus(corpus, seed, max_chars, 256)
    sample = [text for _row, text in picked]
    validation = sample + EDGE_TEXTS + build_fuzz(seed)
    return sample, validation, chars


def _git(args):
    """Read-only git provenance; never a measurement input."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent.parent)] + list(args),
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def _load(lib, tokenizer_path):
    """Load the tokenizer and return (object, encode_fn, module). Native imports
    happen here, in a worker child, never in the parent."""
    if lib == "tokie":
        import tokie
        t = tokie.Tokenizer.open(str(tokenizer_path))
        t.threads(1)
        return t, (lambda s: t.encode(s)), tokie
    if lib == "giga":
        import gigatoken
        t = gigatoken.Tokenizer(str(tokenizer_path))
        return t, (lambda s: t.encode(s)), gigatoken
    if lib == "hf023":
        import tokenizers
        t = tokenizers.Tokenizer.from_file(str(tokenizer_path))
        return t, (lambda s: t.encode(s, add_special_tokens=False).ids), tokenizers
    if lib == "hf1":
        import tokenizers
        t = tokenizers.Tokenizer.from_file(str(tokenizer_path)).compile()
        return t, (lambda s: t.encode(s, add_special_tokens=False).ids), tokenizers
    if lib == "kitoken":
        import kitoken
        t = kitoken.Kitoken.from_file(str(tokenizer_path))
        return t, (lambda s: t.encode(s, encode_specials=True)), kitoken
    if lib == "tiktoken":
        import tiktoken
        allowed = frozenset({"<|endoftext|>"})
        t = tiktoken.get_encoding("gpt2")
        return t, (lambda s: t.encode(s, allowed_special=allowed)), tiktoken
    raise ValueError(f"unknown library {lib!r}")


def _native_files(module):
    pkg = Path(module.__file__).parent
    site = pkg.parent
    out = []
    for p in sorted(pkg.rglob("*.so")):
        out.append({
            "path": str(p.relative_to(site)),
            "sha256": sha256_file(p),
        })
    return out


def _byte_alphabet():
    """GPT-2's byte -> printable unicode map, as tokenizers stores vocab keys."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _tiktoken_vocab_check(tokenizer_path, encoding):
    """Every rank and the endoftext special must match the HF tokenizer.json."""
    inverse = _byte_alphabet()
    data = json.loads(Path(tokenizer_path).read_text())
    ranks = encoding._mergeable_ranks
    vocab = data["model"]["vocab"]
    specials = {t["content"]: t["id"] for t in data.get("added_tokens", []) if t.get("special")}
    special_ids = set(specials.values())
    checked = 0
    first = None
    for token, want_id in vocab.items():
        if want_id in special_ids:
            continue
        try:
            raw = bytes(inverse[ch] for ch in token)
        except KeyError as exc:
            first = first or {"kind": "byte_alphabet", "token": token[:40], "char": repr(exc.args[0])}
            continue
        checked += 1
        got_id = ranks.get(raw)
        if got_id != want_id and first is None:
            first = {"kind": "rank", "token": token[:40], "bytes": raw[:40].hex(),
                     "want_id": want_id, "got_id": got_id}
    special_first = None
    for text, want_id in specials.items():
        got_id = encoding._special_tokens.get(text)
        if got_id != want_id and special_first is None:
            special_first = {"text": text, "want_id": want_id, "got_id": got_id}
    return {
        "ok": first is None and special_first is None,
        "hf_vocab_entries": len(vocab),
        "hf_vocab_non_special_entries": len(vocab) - len(special_ids & set(vocab.values())),
        "ranks_checked": checked,
        "rank_mismatches": 0 if first is None else 1,
        "first_rank_mismatch": first,
        "hf_specials": len(specials),
        "special_mismatches": 0 if special_first is None else 1,
        "first_special_mismatch": special_first,
        "tiktoken_n_vocab": encoding.n_vocab,
    }


def worker_main(argv):
    p = argparse.ArgumentParser(prog="bench_gpt2.py --worker")
    p.add_argument("--worker", required=True, choices=LIB_ORDER)
    p.add_argument("--mode", required=True, choices=("validate", "time"))
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--max-chars", type=int, default=1048576)
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)

    for key, want in THREAD_ENV.items():
        if os.environ.get(key) != want:
            raise SystemExit(f"{key}={want} must be set before importing native libraries")

    lib = a.worker
    sample, validation, _chars = _sample_texts(a.corpus, a.seed, a.max_chars)
    t0 = time.perf_counter()
    inst, encode, module = _load(lib, a.tokenizer)
    load_seconds = time.perf_counter() - t0
    info = {
        "lib": lib,
        "dist_version": md.version(PINS[lib]["distribution"]),
        "module_version": getattr(module, "__version__", None),
        "import_path": module.__file__,
        "python": platform.python_version(),
        "executable": sys.executable,
        "api": API[lib],
        "special_policy": SPECIAL_POLICY[lib],
        "load_seconds": load_seconds,
        "native": _native_files(module),
    }

    if a.mode == "validate":
        rows = []
        for text in validation:
            ids = encode(text)
            rows.append([int(x) for x in ids])
        result = dict(info)
        result.update({
            "texts": len(rows),
            "tokens": sum(len(x) for x in rows),
            "seed_tokens": sum(len(x) for x in rows[:len(sample)]),
            "checksum": ids_checksum(rows),
            "ids": rows,
        })
        if lib == "tiktoken":
            result["vocab_check"] = _tiktoken_vocab_check(a.tokenizer, inst)
    else:
        # Warm the caches outside the timed region.
        for _ in range(a.warmup):
            for text in sample:
                encode(text)
        count = 0
        t0 = time.perf_counter()
        p0 = time.process_time()
        for _ in range(a.passes):
            for text in sample:
                count += len(encode(text))
        wall = time.perf_counter() - t0
        cpu = time.process_time() - p0
        # Checksum pass after the clock, so id conversion is not timed.
        rows = [[int(x) for x in encode(text)] for text in sample]
        result = dict(info)
        result.update({
            "passes": a.passes,
            "warmup_passes": a.warmup,
            "seconds": wall,
            "process_seconds": cpu,
            "count": count,
            "checksum": ids_checksum(rows),
        })

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, separators=(",", ":")))
    return 0


def _run_child(python, script, lib, mode, args, out_path):
    cmd = [
        python, str(script),
        "--worker", lib,
        "--mode", mode,
        "--corpus", str(args.corpus),
        "--tokenizer", str(args.tokenizer),
        "--seed", str(args.seed),
        "--max-chars", str(args.max_chars),
        "--passes", str(args.passes),
        "--warmup", str(args.warmup),
        "--out", str(out_path),
    ]
    env = dict(os.environ)
    env.update(THREAD_ENV)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-4000:]
        raise RuntimeError(f"{lib} {mode} child failed ({proc.returncode}): {tail}")
    try:
        return json.loads(out_path.read_text())
    except Exception as exc:
        raise RuntimeError(f"{lib} {mode} child wrote unreadable JSON: {exc}")


def parse_parent_args(argv):
    p = argparse.ArgumentParser(description="Six-library GPT-2 encode-to-ids benchmark")
    p.add_argument("--corpus", required=True, type=Path, help="OpenWebText JSONL")
    p.add_argument("--tokenizer", required=True, type=Path, help="GPT-2 tokenizer.json")
    p.add_argument("--tokie-wheel", required=True, type=Path,
                   help="local tokie 0.2.0 wheel under test")
    p.add_argument("--wheels-dir", required=True, type=Path,
                   help="directory holding the other five pinned wheels")
    p.add_argument("--python", action="append", default=[], metavar="LIB=PATH",
                   help="interpreter per library; repeat for tokie,giga,hf023,hf1,kitoken,tiktoken")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--passes", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--max-chars", type=int, default=1048576)
    p.add_argument("--expected-tokie-sha256", default=TOKIE_WHEEL_SHA)
    a = p.parse_args(argv)
    for flag, value in (("--rounds", a.rounds), ("--passes", a.passes),
                        ("--warmup", a.warmup), ("--max-chars", a.max_chars)):
        if value <= 0:
            p.error(f"{flag} must be > 0")
    for flag, value in (("--corpus", a.corpus), ("--tokenizer", a.tokenizer),
                        ("--tokie-wheel", a.tokie_wheel)):
        if not value.is_file():
            p.error(f"{flag} not found: {value}")
    if not a.wheels_dir.is_dir():
        p.error(f"--wheels-dir not found: {a.wheels_dir}")
    try:
        a.python_map = parse_python_map(a.python, sys.executable)
    except ValueError as exc:
        p.error(str(exc))
    return a


def _wheel_for(a, lib):
    if lib == "tokie":
        return a.tokie_wheel
    matches = sorted(a.wheels_dir.glob(WHEEL_GLOBS[lib]))
    if len(matches) != 1:
        raise RuntimeError(
            f"{lib}: expected exactly one wheel matching {WHEEL_GLOBS[lib]} in "
            f"{a.wheels_dir}, found {[m.name for m in matches]}"
        )
    return matches[0]


def _wheel_sha_and_members(path):
    import zipfile
    file_sha = sha256_file(path)
    members = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith((".so", ".pyd")):
                members[name] = hashlib.sha256(z.read(name)).hexdigest()
    return file_sha, members


def _first_mismatch(lib_rows, oracle_rows):
    for i, (got, want) in enumerate(zip(lib_rows, oracle_rows)):
        if got != want:
            return {
                "index": i,
                "text_preview": None,
                "got_len": len(got),
                "want_len": len(want),
                "got_first12": got[:12],
                "want_first12": want[:12],
            }
    if len(lib_rows) != len(oracle_rows):
        return {"index": min(len(lib_rows), len(oracle_rows)), "got_len": len(lib_rows),
                "want_len": len(oracle_rows), "reason": "row count differs"}
    return None


def _write_result(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return worker_main(argv)

    a = parse_parent_args(argv)
    started = time.time()
    errors = []
    result = {
        "schema_version": SCHEMA_VERSION,
        "task": "encode_one_text_to_ids",
        "model": "gpt2",
        "status": "error",
        "errors": errors,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "meta": {},
        "validation": {},
        "rounds": [],
        "summary": {},
        "ranking_note": None,
    }

    # Host snapshot before validation; never a timing claim.
    host_before = host_snapshot()
    load_before = list(os.getloadavg())
    script = Path(__file__).resolve()
    tokenizer_sha = sha256_file(a.tokenizer)
    corpus_sha = sha256_file(a.corpus)
    sample, validation_texts, chars = _sample_texts(a.corpus, a.seed, a.max_chars)
    utf8_bytes = sum(len(t.encode("utf-8")) for t in sample)

    result["meta"] = {
        "claim_scope": (
            "GPT-2 full public single-text encode to token ids on one seeded OpenWebText "
            "slice; one thread, one host, one fresh experiment. Not a universal fastest-tokenizer claim."
        ),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "load_before": load_before,
        "env": dict(THREAD_ENV),
        "host": host_before,
        "corpus": {
            "jsonl": str(a.corpus.resolve()),
            "jsonl_sha256": corpus_sha,
            "jsonl_bytes": a.corpus.stat().st_size,
            "seed": a.seed,
            "max_chars": a.max_chars,
            "documents": len(sample),
            "sample_utf8_bytes": utf8_bytes,
            "sample_chars": chars,
            "validation_texts": len(validation_texts),
            "edge_texts": len(EDGE_TEXTS),
            "fuzz_texts": 256,
        },
        "tokenizer": {
            "path": str(a.tokenizer.resolve()),
            "sha256": tokenizer_sha,
            "bytes": a.tokenizer.stat().st_size,
            "normalizer": "none",
        },
        "protocol": {
            "threads": 1,
            "rounds": a.rounds,
            "passes": a.passes,
            "warmup_passes": a.warmup,
            "round_order": "latin rotation uses 5 of 6 starting positions; partial rotation, not fully balanced",
            "timed_region": "passes x all sample documents; len(ids) and accumulation inside the clock",
            "outside_clock": "import, model load, warmup passes, and the id-checksum pass",
            "thread_pools": "Rayon, tokenizers, OpenBLAS/OMP/MKL, numexpr, and vecLib pools are all pinned to 1",
            "rounds_are": "min/max ranges over rounds, not a confidence interval",
        },
        "libraries": {},
        "versions": {lib: PINS[lib]["version"] for lib in LIB_ORDER},
        "artifacts": {},
        "harness": {},
        "api_notes": {
            "hf023": "installed stub: Tokenizer.encode(text, add_special_tokens=False) -> Encoding; Encoding.ids is list[int]. This is the only encode API in 0.23.3.",
            "hf1": "installed 1.0.0rc4 stub: Tokenizer.compile() -> CompiledTokenizer; encode(text, add_special_tokens=False) -> Encoding; Encoding.ids is list[int]; Encoding.ids_array returns a copy. No encode_fast or ids-only method exists, and offsets are not returned unless requested, so the documented encode(...).ids path is used.",
            "kitoken": "installed kitoken.pyi: Kitoken.from_file(path) -> Kitoken; encode(text, encode_specials=False) -> list[int]. Default False does not match the HF ids for '<|endoftext|>'; encode_specials=True is used and disclosed.",
            "tiktoken": "tiktoken.get_encoding('gpt2').encode(text, allowed_special=...) -> list[int]. The gpt2 ranks are the canonical OpenAI BPE; allowed_special is passed explicitly.",
            "tokie": "installed binding: Tokenizer.open(path); threads(n) only changes batch workers; encode(text) returns a uint32 numpy array.",
            "giga": "installed binding: Tokenizer(path); encode(text) returns a uint32 numpy array; no thread control on single-text encode.",
        },
    }
    result["meta"]["harness"] = {
        "script": str(script),
        "sha256": sha256_file(script),
        "helpers": {
            "bench_published.py": sha256_file(script.parent / "bench_published.py"),
        },
        "git_head": _git(("rev-parse", "HEAD")),
        "git_status_porcelain": _git(("status", "--porcelain")) or "",
    }
    tmp = Path(tempfile.mkdtemp(prefix="gpt2-hero-"))
    try:
        # ---- wheel and artifact guards -------------------------------------
        wheel_info = {}
        for lib in LIB_ORDER:
            wpath = _wheel_for(a, lib)
            file_sha, members = _wheel_sha_and_members(wpath)
            if lib == "tokie" and file_sha != a.expected_tokie_sha256:
                errors.append(
                    f"tokie wheel sha256 {file_sha} != expected {a.expected_tokie_sha256}"
                )
            wheel_info[lib] = {
                "file": wpath.name,
                "path": str(wpath.resolve()),
                "sha256": file_sha,
                "members": members,
            }
            result["meta"]["artifacts"][lib] = {
                "wheel_file": wpath.name,
                "wheel_sha256": file_sha,
                "wheel_members_sha256": members,
            }
        if tokenizer_sha != TOKENIZER_SHA:
            errors.append(f"tokenizer.json sha256 {tokenizer_sha} != expected {TOKENIZER_SHA}")
        if utf8_bytes < MIN_SAMPLE_BYTES or len(sample) != 179:
            errors.append(
                f"corpus slice is {len(sample)} docs / {utf8_bytes} bytes; "
                "expected the frozen 179-doc ~1 MiB slice"
            )
        if len(validation_texts) != EXPECTED_TEXTS:
            errors.append(
                f"validation set has {len(validation_texts)} texts; expected {EXPECTED_TEXTS}"
            )
        if errors:
            raise RuntimeError("; ".join(errors))

        # ---- validation: one child per library, HF 0.23.3 is the oracle --
        observed = {}
        for lib in LIB_ORDER:
            out = tmp / f"validate-{lib}.json"
            payload = _run_child(a.python_map[lib], script, lib, "validate", a, out)
            observed[lib] = payload
        pin_errors = check_pins(observed)
        if pin_errors:
            errors.extend(pin_errors)

        pythons = {lib: observed[lib]["python"] for lib in LIB_ORDER}
        if len(set(pythons.values())) != 1:
            errors.append(f"mixed Python versions across libraries: {pythons}")

        oracle_rows = observed["hf023"]["ids"]
        for lib in LIB_ORDER:
            payload = observed[lib]
            lib_rows = payload["ids"]
            mismatch = _first_mismatch(lib_rows, oracle_rows)
            ok = mismatch is None and len(lib_rows) == len(oracle_rows)
            first = mismatch
            if first is not None and first.get("index", 0) < len(validation_texts):
                first["text_preview"] = validation_texts[first["index"]][:120]
            result["validation"][lib] = {
                "ok": ok,
                "vs": "hf023",
                "texts": len(lib_rows),
                "tokens": sum(len(x) for x in lib_rows),
                "mismatches": 0 if ok else 1,
                "first_mismatch": first,
                "checksum": payload["checksum"],
            }
            info = observed[lib]
            wheel = wheel_info[lib]
            native = info.get("native", [])
            inst_hashes = {x["path"]: x["sha256"] for x in native}
            wheel_hashes = wheel["members"]
            native_match = bool(inst_hashes) and all(
                wheel_hashes.get(path) == sha for path, sha in inst_hashes.items()
            )
            result["meta"]["libraries"][lib] = {
                "distribution": PINS[lib]["distribution"],
                "version": info["dist_version"],
                "module_version": info["module_version"],
                "python": info["python"],
                "executable": info["executable"],
                "import_path": info["import_path"],
                "api": info["api"],
                "special_policy": info["special_policy"],
                "load_seconds": info["load_seconds"],
                "wheel": {
                    "file": wheel["file"],
                    "sha256": wheel["sha256"],
                    "members": wheel_hashes,
                },
                "installed_native": native,
                "installed_native_matches_wheel": native_match,
            }
            if not native_match:
                errors.append(f"{lib}: installed native file does not match the wheel member")

        vocab_check = observed["tiktoken"].get("vocab_check")
        if vocab_check is not None:
            result["meta"]["tiktoken_vocab_check"] = vocab_check
            if vocab_check["rank_mismatches"] or vocab_check["special_mismatches"]:
                errors.append("tiktoken: vocab or special ids differ from the HF tokenizer.json")

        validation_token_count = sum(len(x) for x in oracle_rows)
        validation_checksum = ids_checksum(oracle_rows)
        sample_token_count = sum(len(x) for x in oracle_rows[:len(sample)])
        preflight = golden_preflight_errors(
            len(sample), len(oracle_rows), sample_token_count,
            validation_token_count, validation_checksum,
        )
        result["meta"]["golden_preflight"] = {
            "ok": not preflight,
            "sample_docs": len(sample),
            "validation_texts": len(oracle_rows),
            "sample_ids": sample_token_count,
            "validation_ids": validation_token_count,
            "validation_checksum": validation_checksum,
        }
        if preflight:
            errors.extend(preflight)

        if errors:
            result["status"] = "invalid"
            return _finish(a, result, errors, host_before, started)

        # ---- five globally interleaved timing rounds -----------------------
        sample_checksum = ids_checksum(oracle_rows[:len(sample)])
        for r in range(a.rounds):
            order = rotate_order(r)
            round_entry = {"round": r + 1, "order": order, "libraries": {}}
            for lib in order:
                out = tmp / f"time-{r + 1}-{lib}.json"
                payload = _run_child(a.python_map[lib], script, lib, "time", a, out)
                want_count = sample_token_count * a.passes
                if payload["count"] != want_count:
                    errors.append(
                        f"round {r + 1} {lib}: count {payload['count']} != {want_count}"
                    )
                if payload["checksum"] != sample_checksum:
                    errors.append(f"round {r + 1} {lib}: id checksum changed")
                round_entry["libraries"][lib] = {
                    "seconds": payload["seconds"],
                    "process_seconds": payload["process_seconds"],
                    "count": payload["count"],
                    "checksum": payload["checksum"],
                    "mib_s": mib_per_s(utf8_bytes * a.passes, payload["seconds"]),
                    "cpu_wall_ratio": (
                        payload["process_seconds"] / payload["seconds"]
                        if payload["seconds"]
                        else None
                    ),
                }
            result["rounds"].append(round_entry)
        if errors:
            result["status"] = "error"
            return _finish(a, result, errors, host_before, started)

        for lib in LIB_ORDER:
            secs = [r["libraries"][lib]["seconds"] for r in result["rounds"]]
            result["summary"][lib] = summarize_rounds(secs, utf8_bytes * a.passes)
            result["summary"][lib]["tokens_per_pass"] = sample_token_count
            result["summary"][lib]["tokens_s_median"] = (
                (sample_token_count * a.passes) / result["summary"][lib]["seconds_median"]
            )

        ranked = sorted(LIB_ORDER, key=lambda k: result["summary"][k]["seconds_median"])
        best, second = ranked[0], ranked[1]
        b, s = result["summary"][best], result["summary"][second]
        gap_pct = (s["seconds_median"] - b["seconds_median"]) / s["seconds_median"] * 100.0
        if b["seconds_max"] < s["seconds_min"] and gap_pct >= 10.0:
            result["ranking_note"] = (
                f"{best} has the lowest median, {gap_pct:.1f}% below {second}, on disjoint "
                "single-run round ranges; other pairwise comparisons are left to the reader."
            )
        elif b["seconds_max"] < s["seconds_min"]:
            result["ranking_note"] = (
                f"{best} has the lowest median, {gap_pct:.1f}% below {second}; the single-run "
                "round ranges do not overlap, but the gap is close, so no robust winner is claimed."
            )
        else:
            result["ranking_note"] = (
                f"no winner claimed: {best} and {second} round ranges overlap "
                f"({b['seconds_min']:.6f}-{b['seconds_max']:.6f}s vs "
                f"{s['seconds_min']:.6f}-{s['seconds_max']:.6f}s)."
            )
        result["status"] = "ok"
        return _finish(a, result, errors, host_before, started)

    except Exception as exc:
        errors.append(f"harness: {type(exc).__name__}: {exc}")
        result["status"] = "error"
        return _finish(a, result, errors, host_before, started)
    finally:
        try:
            for p in tmp.glob("*"):
                p.unlink()
            tmp.rmdir()
        except OSError:
            pass


def _finish(a, result, errors, host_before, started):
    result["errors"] = errors
    result["meta"]["host"]["load_after"] = list(os.getloadavg())
    result["meta"]["host"]["snapshot_after"] = host_snapshot()
    result["meta"]["elapsed_seconds"] = round(time.time() - started, 3)
    result["meta"]["command"] = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
    target = a.output
    if result["status"] != "ok":
        # Never clobber the canonical recorded result with a failed replay.
        target = a.output.with_name(a.output.name + ".error.json")
    _write_result(target, result)
    brief = {
        "status": result["status"],
        "output": str(target),
        "validation": {k: v.get("ok") for k, v in result.get("validation", {}).items()},
        "errors": errors,
    }
    print(json.dumps(brief, indent=2))
    return 0 if result["status"] == "ok" else (2 if result["status"] == "invalid" else 1)


if __name__ == "__main__":
    sys.exit(main())
