#!/usr/bin/env python3
"""Compare published tokie and gigatoken on one tokenizer.json at a time.

Scope: exact encode ids, exact decode bytes, and cold/warm throughput on a
seeded OpenWebText slice. This is a per-model, per-output, per-thread-config
comparison on that slice, not a universal fastest-tokenizer claim. Both
packages must come from PyPI in a clean environment; no local checkout.

    python scripts/bench_published.py --corpus owt_20m.jsonl \
        --models gpt2=/path/gpt2/tokenizer.json --threads 1 \
        --rounds 5 --output results-1t.json

Run 8-thread batch in a separate process so RAYON_NUM_THREADS is set before
the native libraries load:

    RAYON_NUM_THREADS=8 python scripts/bench_published.py ... --threads 8 \
        --ops batch --output results-8t.json

Correctness uses the seeded sample plus fixed edge cases and deterministic
fuzz. Timing uses OpenWebText documents only; fuzz is never timed.
"""

import argparse
import cProfile
import hashlib
import importlib.metadata as md
import io
import json
import os
import pstats
import random
import re
import statistics
import sys
import time
from pathlib import Path

EDGE_TEXTS = [
    "", " ", "\n", "\t", "\r\n", "a\tb\nc\r\nd", "\x00\x01\x02 control \x07\x08",
    "line one\nline two\n\nline three", "café naïve résumé", "e\u0301 combining acute",
    "Z\u0335a\u0336l\u0337g\u0338o\u0334", "😀🎉🚀👩‍👩‍👧‍👦", "日本語のテキスト、中文文本、한국어",
    "العربية نص عربي", "mixed عربى 日本語 😀 café", "  leading and trailing  ",
    "UPPER lower MiXeD", "punctuation!?,.;:'\"()[]{}<>@#$%^&*_-+=/\\|~`",
    "numbers 0 1 42 3.14159 1e10 2026", "https://example.com/a?b=c&d=e#frag",
    "\\n is not a newline", "\u00a0non-breaking\u2003em space", "\ufeffBOM",
    "zero\u200bwidth\u200djoiner",
]
FUZZ_N = 256
FUZZ_POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!\n\t")
    + list("éüñßçøåæœ") + list("日本語中文한국어") + list("العربية") + list("😀🎉🚀🔥✨")
    + ["\u0301", "\u0335", "\u200d", "\u200b", "\u00a0", "\u3000", "\ufeff"]
    + ["<|endoftext|>", "[SEP]", "[MASK]", "</s>", "<pad>", "\x00", "\x07"]
)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, type=Path, help="OpenWebText JSONL with row/text fields")
    p.add_argument("--models", action="append", default=[], metavar="LABEL=PATH",
                   help="tokenizer.json to compare; repeat per model")
    p.add_argument("--threads", type=int, choices=(1, 8), default=1,
                   help="batch threads; single encode/decode stay on one core")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=1_048_576)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--ops", default="encode,decode,decode_list,batch",
                   help="comma-separated subset of encode,decode,decode_list,batch")
    p.add_argument("--passes", type=int, default=20, help="inner passes per timed round")
    p.add_argument("--max-docs", type=int, default=256)
    p.add_argument("--wheels-dir", type=Path, default=None,
                   help="optional downloaded-wheel directory to hash")
    p.add_argument("--allow-no-hf", action="store_true",
                   help="compare tokie vs gigatoken without the HF oracle if tokenizers cannot import")
    a = p.parse_args()
    if not a.models:
        p.error("at least one --models LABEL=PATH is required")
    for flag, value in (("--rounds", a.rounds), ("--passes", a.passes),
                        ("--max-docs", a.max_docs), ("--max-chars", a.max_chars)):
        if value <= 0:
            p.error(f"{flag} must be > 0")
    parsed = []
    for spec in a.models:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            p.error(f"bad --models value {spec!r}; expected LABEL=PATH")
        if not Path(path).is_file():
            p.error(f"tokenizer.json not found: {path}")
        parsed.append((label, Path(path)))
    labels = [label for label, _ in parsed]
    if len(set(labels)) != len(labels):
        p.error("duplicate --models labels: " + ", ".join(sorted({x for x in labels if labels.count(x) > 1})))
    a.models = parsed
    a.ops = [x.strip() for x in a.ops.split(",") if x.strip()]
    if not a.ops:
        p.error("--ops must not be empty")
    bad = set(a.ops) - {"encode", "decode", "decode_list", "batch"}
    if bad:
        p.error(f"unknown ops: {sorted(bad)}")
    return a

def load_corpus(path, seed, max_chars, max_docs):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append((rec["row"], rec["text"]))
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    picked, chars = [], 0
    for i in idx:
        row, text = docs[i]
        if picked and chars + len(text) > max_chars:
            continue
        picked.append((row, text))
        chars += len(text)
        if len(picked) >= max_docs or chars >= max_chars:
            break
    picked.sort(key=lambda x: x[0])
    return docs, picked, chars

def build_fuzz(seed):
    rng = random.Random(seed ^ 0x5EED)
    return ["".join(rng.choice(FUZZ_POOL) for _ in range(rng.randint(0, 48)))
            for _ in range(FUZZ_N)]

def cpu_model():
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None

def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def host_snapshot():
    """Read-only sysfs snapshot; never changes the governor."""
    root = Path("/sys/devices/system/cpu")
    governors, drivers, freqs = {}, {}, []
    for cpu in sorted(root.glob("cpu[0-9]*")):
        cp = cpu / "cpufreq"
        gov = read_text(cp / "scaling_governor")
        if gov:
            governors[gov] = governors.get(gov, 0) + 1
        drv = read_text(cp / "scaling_driver")
        if drv:
            drivers[drv] = drivers.get(drv, 0) + 1
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            val = read_text(cp / name)
            if val and val.isdigit():
                freqs.append(int(val))
                break
    return {
        "governors": governors,
        "scaling_drivers": drivers,
        "cpus_with_freq": len(freqs),
        "mean_cur_freq_mhz": round(sum(freqs) / len(freqs) / 1000, 1) if freqs else None,
        "min_cur_freq_mhz": round(min(freqs) / 1000, 1) if freqs else None,
        "max_cur_freq_mhz": round(max(freqs) / 1000, 1) if freqs else None,
    }

def package_meta(name):
    try:
        d = md.distribution(name)
        inst = (d.read_text("INSTALLER") or "").strip() or None
        return {"version": d.version, "installer": inst, "name": d.metadata.get("Name")}
    except Exception:
        return {"version": None, "installer": None, "name": None}

def wheel_hashes(wheels_dir):
    out = {}
    if wheels_dir is None:
        return out
    for pkg in ("tokie", "gigatoken", "tokenizers", "numpy", "awkward"):
        out[pkg] = [{"file": p.name, "sha256": sha256_file(p)}
                    for p in sorted(wheels_dir.glob(pkg.replace("-", "_") + "-*.whl"))]
    return out

def make_tokie(tokie, path, threads):
    t = tokie.Tokenizer.open(str(path))
    t.threads(threads)
    return t

def make_giga(gigatoken, path):
    return gigatoken.Tokenizer(str(path))

def giga_batch_parts(out):
    import awkward as ak
    import numpy as np
    flat = np.asarray(ak.flatten(out, axis=1), dtype=np.uint32)
    # gigatoken offsets include a leading 0; tokie ends do not.
    return flat, np.asarray(out.layout.offsets, dtype=np.uint64)[1:]

def validate(tokie, gigatoken, hf_mod, path, texts, threads):
    """Exact ids and decode bytes for tokie vs gigatoken; HF is a third oracle."""
    import numpy as np
    t = make_tokie(tokie, path, threads)
    g = make_giga(gigatoken, path)
    hf = hf_mod.Tokenizer.from_file(str(path)) if hf_mod else None
    tok_ids, hf_ids, tok_dec = [], [], []
    enc_bad = dec_bad = hf_enc = hf_dec = 0
    first = None
    for i, text in enumerate(texts):
        ti = np.asarray(t.encode(text), dtype=np.uint32)
        gi = np.asarray(g.encode(text), dtype=np.uint32)
        tok_ids.append(ti)
        if not np.array_equal(ti, gi):
            enc_bad += 1
            first = first or {"kind": "encode", "index": i, "text_preview": text[:120],
                              "tokie_len": len(ti), "giga_len": len(gi),
                              "tokie_first12": ti[:12].tolist(), "giga_first12": gi[:12].tolist()}
        tb, gb = t.decode_bytes(ti), g.decode(gi)
        tok_dec.append(tb)
        if tb != gb:
            dec_bad += 1
            first = first or {"kind": "decode", "index": i, "text_preview": text[:120],
                              "tokie_preview": tb[:120].hex(), "giga_preview": gb[:120].hex()}
        if hf is not None:
            hi = np.asarray(hf.encode(text, add_special_tokens=False).ids, dtype=np.uint32)
            hf_ids.append(hi)
            if not np.array_equal(ti, hi):
                hf_enc += 1
                first = first or {"kind": "hf_encode", "index": i, "text_preview": text[:120],
                                  "tokie_len": len(ti), "hf_len": len(hi),
                                  "tokie_first12": ti[:12].tolist(), "hf_first12": hi[:12].tolist()}
            else:
                hb = hf.decode(hi.tolist(), skip_special_tokens=False).encode("utf-8")
                if hb != tb:
                    hf_dec += 1
                    first = first or {"kind": "hf_decode", "index": i, "text_preview": text[:120],
                                      "tokie_preview": tb[:120].hex(), "hf_preview": hb[:120].hex()}
    tids, tends = t.batch(texts)
    gflat, goffs = giga_batch_parts(g.encode_batch(texts, parallel=(threads > 1)))
    batch_ok = np.array_equal(tids, gflat) and np.array_equal(tends, goffs)
    hf_batch_ok = None
    if hf is not None:
        flat = [x for row in hf_ids for x in row]
        ends, acc = [], 0
        for row in hf_ids:
            acc += len(row)
            ends.append(acc)
        flat_match = np.array_equal(tids, flat)
        ends_match = np.array_equal(tends, ends)
        hf_batch_ok = flat_match and ends_match
        if not hf_batch_ok:
            first = first or {"kind": "hf_batch", "tokie_flat_len": len(tids),
                              "hf_flat_len": len(flat), "flat_match": bool(flat_match),
                              "ends_match": bool(ends_match)}
    hf_ok = None if hf is None else (hf_enc == 0 and hf_dec == 0 and hf_batch_ok is True)
    stats = {"texts": len(texts), "tokens": int(sum(len(x) for x in tok_ids)),
             "encode_mismatches": enc_bad, "decode_mismatches": dec_bad,
             "batch_mismatches": 0 if batch_ok else 1,
             "hf_available": hf is not None, "hf_ok": hf_ok,
             "hf_encode_mismatches": hf_enc if hf else None,
             "hf_decode_mismatches": hf_dec if hf else None, "hf_batch_ok": hf_batch_ok}
    ok = enc_bad == dec_bad == 0 and batch_ok and hf_ok is not False
    return {"ok": ok, "stats": stats,
            "first_mismatch": first, "ids": tok_ids, "decoded": tok_dec,
            "batch_flat": tids, "batch_ends": tends}

def op_fns(impl, inst, op, threads):
    import numpy as np
    base = "decode" if op.startswith("decode") else op
    if base == "encode":
        return (lambda s: np.asarray(inst.encode(s), dtype=np.uint32)), "many", len
    if base == "decode":
        fn = (lambda ids: inst.decode_bytes(ids)) if impl == "tokie" else (lambda ids: inst.decode(ids))
        return fn, "many", len
    if impl == "tokie":
        return (lambda xs: inst.batch(xs)), "batch", (lambda o: len(o[0]) + len(o[1]))
    # Same quantity as tokie's len(flat)+len(ends): tokens + documents, O(1).
    return ((lambda xs: inst.encode_batch(xs, parallel=(threads > 1))), "batch",
            (lambda o: len(o.layout.content) + len(o.layout.offsets) - 1))

def cold_measure(inst, impl, op, texts, ids_list, threads, expected_count):
    fn, shape, count = op_fns(impl, inst, op, threads)
    if shape == "batch":
        t0 = time.perf_counter()
        out = fn(texts)
        first = time.perf_counter() - t0
        rest, full, got = 0.0, first, count(out)
    else:
        items = texts if op == "encode" else ids_list
        t0 = time.perf_counter()
        out = fn(items[0])
        first = time.perf_counter() - t0
        got = count(out)
        t0 = time.perf_counter()
        for x in items[1:]:
            got += count(fn(x))
        rest = time.perf_counter() - t0
        full = first + rest
    if got != expected_count:
        raise AssertionError(f"cold {op} {impl}: count {got} != {expected_count}")
    return {"first_call_s": first, "remaining_pass_s": rest,
            "cold_full_pass_s": full, "count": got}

def warmup(inst, impl, op, texts, ids_list, threads):
    fn, shape, _ = op_fns(impl, inst, op, threads)
    if shape == "batch":
        fn(texts)
    else:
        for x in (texts if op == "encode" else ids_list):
            fn(x)

def timed(inst, impl, op, texts, ids_list, threads, passes):
    fn, shape, count = op_fns(impl, inst, op, threads)
    counts, last = [], None
    t0 = time.perf_counter()
    if shape == "batch":
        for _ in range(passes):
            last = fn(texts)
            counts.append(count(last))
    else:
        items = texts if op == "encode" else ids_list
        for _ in range(passes):
            c = 0
            for x in items:
                last = fn(x)
                c += count(last)
            counts.append(c)
    return time.perf_counter() - t0, counts, last

def post_check(impl, op, last, expected):
    import numpy as np
    if op == "encode":
        return np.array_equal(last, expected["last_ids"])
    if op.startswith("decode"):
        return last == expected["last_bytes"]
    if impl == "tokie":
        flat, ends = last
        return np.array_equal(flat, expected["flat"]) and np.array_equal(ends, expected["ends"])
    flat, offs = giga_batch_parts(last)
    return np.array_equal(flat, expected["flat"]) and np.array_equal(offs, expected["ends"])

def checksum(impl, op, last):
    import numpy as np
    if op.startswith("decode"):
        return hashlib.sha256(last).hexdigest()[:16]
    if op == "batch" and impl == "giga":
        return int(giga_batch_parts(last)[0].sum(dtype=np.uint64))
    if op == "batch":
        return int(np.asarray(last[0], dtype=np.uint64).sum())
    return int(np.asarray(last, dtype=np.uint64).sum())

def profile_one(inst, impl, op, texts, ids_list, threads):
    fn, shape, _ = op_fns(impl, inst, op, threads)
    pr = cProfile.Profile()
    c0, w0 = time.process_time(), time.perf_counter()
    pr.enable()
    if shape == "batch":
        fn(texts)
    else:
        for x in (texts if op == "encode" else ids_list):
            fn(x)
    pr.disable()
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(6)
    top = []
    for line in buf.getvalue().splitlines():
        m = re.match(r"\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)", line)
        if m:
            top.append({"cumtime_s": float(m.group(4)), "func": m.group(6)})
    return {"wall_s": wall, "process_cpu_s": cpu,
            "cpu_per_wall": cpu / wall if wall else None,
            "cpu_bound": (cpu / wall > 0.95) if wall else None, "top": top[:5],
            "note": "cProfile sees Python frames only; native kernel internals are not attributed."}

def summarize(rounds, mib_bytes, total_tokens, op, passes):
    out = {}
    mib_bytes *= passes
    total_tokens *= passes
    for impl in ("tokie", "giga"):
        es = [r[impl]["seconds"] for r in rounds]
        out[impl] = {
            "seconds_median": statistics.median(es), "seconds_min": min(es), "seconds_max": max(es),
            "mib_s_median": mib_bytes / statistics.median(es) / (1 << 20),
            "mib_s_min": mib_bytes / max(es) / (1 << 20),
            "mib_s_max": mib_bytes / min(es) / (1 << 20),
            "tokens_s_median": total_tokens / statistics.median(es),
            "tokens_s_min": total_tokens / max(es), "tokens_s_max": total_tokens / min(es),
        }
    ratios = [r["giga"]["seconds"] / r["tokie"]["seconds"] for r in rounds]
    out["giga_over_tokie_time"] = {
        "median": statistics.median(ratios), "min": min(ratios), "max": max(ratios),
        "median_of_medians": out["giga"]["seconds_median"] / out["tokie"]["seconds_median"],
        "interpretation": "giga seconds / tokie seconds; below 1 means giga took less time",
    }
    return out

def bench_op(tokie, gigatoken, path, op, texts, ids_list, expected,
             input_bytes, total_tokens, decoded_bytes, args):
    instances = {"tokie": make_tokie(tokie, path, 1 if op != "batch" else args.threads),
                 "giga": make_giga(gigatoken, path)}
    # Python-list IDs are prepared once, outside every timed region.
    items_ids = [x.tolist() for x in ids_list] if op == "decode_list" else ids_list
    expected_count = {"encode": total_tokens, "decode": decoded_bytes,
                      "decode_list": decoded_bytes,
                      "batch": total_tokens + len(texts)}[op]
    cold = {impl: cold_measure(instances[impl], impl, op, texts, items_ids,
                               args.threads, expected_count) for impl in ("tokie", "giga")}
    for impl in ("tokie", "giga"):
        warmup(instances[impl], impl, op, texts, items_ids, args.threads)
    rounds, last = [], {}
    for r in range(args.rounds):
        order = ["tokie", "giga"] if r % 2 == 0 else ["giga", "tokie"]
        rec = {"round": r + 1, "order": order}
        for impl in order:
            secs, counts, out = timed(instances[impl], impl, op, texts, items_ids,
                                      args.threads, args.passes)
            if len(set(counts)) != 1 or counts[0] != expected_count:
                raise AssertionError(f"{path.name} {op} {impl}: unstable count "
                                     f"{counts[:3]} != {expected_count}")
            rec[impl] = {"seconds": secs, "passes": args.passes, "count": counts[0]}
            last[impl] = out
        rounds.append(rec)
    checks = {}
    for impl in ("tokie", "giga"):
        if not post_check(impl, op, last[impl], expected):
            raise AssertionError(f"{path.name} {op} {impl}: last output differs from validation")
        checks[impl] = {"output_ok": True, "checksum": checksum(impl, op, last[impl])}
    mib_bytes = decoded_bytes if op.startswith("decode") else input_bytes
    return {"threads": 1 if op != "batch" else args.threads,
            "passes_per_round": args.passes, "input_bytes_utf8": input_bytes,
            "total_tokens": total_tokens, "decoded_bytes_utf8": decoded_bytes,
            "ids_input": {"decode": "numpy_uint32", "decode_list": "python_list"}.get(op),
            "cold": cold, "rounds": rounds,
            "summary": summarize(rounds, mib_bytes, total_tokens, op, args.passes),
            "post_check": checks,
            "profile": {impl: profile_one(instances[impl], impl, op, texts, items_ids, args.threads)
                        for impl in ("tokie", "giga")}}

def main():
    args = parse_args()
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if args.threads == 1 else "true"
    import awkward as ak
    import numpy as np
    import tokie
    import gigatoken
    hf_mod, hf_error = None, None
    try:
        import tokenizers as hf_mod
    except Exception as exc:
        hf_error = f"{type(exc).__name__}: {exc}"

    load_before = list(os.getloadavg())
    docs, picked, picked_chars = load_corpus(args.corpus, args.seed, args.max_chars, args.max_docs)
    fuzz = build_fuzz(args.seed)
    sample_texts = [t for _, t in picked]
    validation_texts = sample_texts + EDGE_TEXTS + fuzz
    input_bytes = sum(len(t.encode("utf-8")) for t in sample_texts)
    manifest_path = args.corpus.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claim_scope": "published tokie vs published gigatoken per model, output, and thread config on a seeded OpenWebText subset",
        "first_pass_method": "single sample per operation, tokie measured first, model loading excluded",
        "python": sys.version.split()[0], "platform": sys.platform,
        "cpu_model": cpu_model(), "cpu_count": os.cpu_count(), "load_before": load_before,
        "host": host_snapshot(),
        "env": {"RAYON_NUM_THREADS": os.environ["RAYON_NUM_THREADS"],
                "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"]},
        "packages": {p: package_meta(p) for p in ("tokie", "gigatoken", "tokenizers", "numpy", "awkward")},
        "import_paths": {"tokie": tokie.__file__, "gigatoken": gigatoken.__file__,
                         "numpy": np.__file__, "awkward": ak.__file__,
                         "tokenizers": getattr(hf_mod, "__file__", None)},
        "hf_oracle": {"available": hf_mod is not None, "import_error": hf_error,
                      "required": not args.allow_no_hf, "allow_no_hf": args.allow_no_hf},
        "wheels": wheel_hashes(args.wheels_dir),
        "provenance": {
            "package_source": "public PyPI wheels installed in a clean venv; no local checkout or path install",
            "wheels_dir": str(args.wheels_dir) if args.wheels_dir else None,
        },
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "corpus": {
            "jsonl": str(args.corpus), "jsonl_sha256": sha256_file(args.corpus),
            "jsonl_bytes": args.corpus.stat().st_size, "available_documents": len(docs),
            "manifest": str(manifest_path) if manifest else None,
            "manifest_sha256": sha256_file(manifest_path) if manifest else None,
            "source": manifest.get("dataset") if manifest else None,
            "provenance": {k: manifest.get(k) for k in
                           ("dataset", "parquet", "source_sha256", "source_rows", "selection")}
            if manifest else None,
            "sampling_seed": args.seed, "max_chars": args.max_chars, "max_docs": args.max_docs,
            "sampled_documents": len(picked), "sampled_chars": picked_chars,
            "sampled_bytes_utf8": input_bytes,
            "sampled_row_range": [picked[0][0], picked[-1][0]],
            "sampled_rows_sha256": hashlib.sha256(
                ",".join(str(r) for r, _ in picked).encode()).hexdigest(),
            "edge_texts": len(EDGE_TEXTS), "fuzz_texts": len(fuzz),
            "timing_texts": len(sample_texts), "validation_texts": len(validation_texts),
        },
        "models": {},
    }
    results = {"meta": meta, "validation": {}, "benchmark": {}, "status": {}, "errors": []}
    hf_blocked = hf_mod is None and not args.allow_no_hf
    if hf_blocked:
        results["errors"].append(
            f"HF oracle unavailable ({hf_error}); pass --allow-no-hf to compare tokie vs gigatoken only")
        for label, _ in args.models:
            results["status"][label] = "hf_missing"
    for label, path in ([] if hf_blocked else args.models):
        try:
            t = make_tokie(tokie, path, args.threads)
            g = make_giga(gigatoken, path)
            meta["models"][label] = {"path": str(path), "sha256": sha256_file(path),
                                     "bytes": path.stat().st_size,
                                     "tokie_vocab_size": int(t.vocab_size),
                                     "giga_vocab_size": int(g.vocab_size)}
        except Exception as exc:
            results["status"][label] = "error"
            results["errors"].append(f"{label}: load failed: {type(exc).__name__}: {exc}")
            continue
        try:
            v = validate(tokie, gigatoken, hf_mod, path, validation_texts, args.threads)
        except Exception as exc:
            results["status"][label] = "error"
            results["errors"].append(f"{label}: validation failed: {type(exc).__name__}: {exc}")
            continue
        results["validation"][label] = {k: v[k] for k in ("ok", "stats", "first_mismatch")}
        if not v["ok"]:
            results["status"][label] = "mismatch"
            results["errors"].append(f"{label}: tokie/gigatoken mismatch; timing skipped")
            continue
        n = len(sample_texts)
        sample_ids, sample_decoded = v["ids"][:n], v["decoded"][:n]
        total_tokens = sum(len(x) for x in sample_ids)
        decoded_bytes = sum(len(x) for x in sample_decoded)
        expected = {"last_ids": sample_ids[-1], "last_bytes": sample_decoded[-1],
                    "flat": v["batch_flat"][:int(v["batch_ends"][n - 1])],
                    "ends": v["batch_ends"][:n]}
        results["benchmark"][label] = {}
        try:
            for op in args.ops:
                results["benchmark"][label][op] = bench_op(
                    tokie, gigatoken, path, op, sample_texts, sample_ids, expected,
                    input_bytes, total_tokens, decoded_bytes, args)
            results["status"][label] = "ok"
        except Exception as exc:
            results["status"][label] = "error"
            results["errors"].append(f"{label}: benchmark failed: {type(exc).__name__}: {exc}")

    meta["load_after"] = list(os.getloadavg())
    meta["limitations"] = [
        "Per-model, per-output, per-thread comparison on a seeded OpenWebText slice; not a universal fastest-tokenizer claim.",
        "Timing uses OpenWebText sample documents only. Edge cases and fuzz are correctness-only.",
        "Cold reports the first call and the first full pass from a fresh instance; tokenizer.json load time is excluded. Warm rounds exclude one warmup pass.",
        "tokie batch returns flat uint32 ids plus uint64 ends; gigatoken batch returns an awkward list array. Output shapes differ; both are counted as tokens + documents inside the timed loop.",
        "Decode input is explicit: decode uses NumPy uint32 ids; decode_list uses Python lists converted outside the timed region. Both include public binding overhead and can rank differently.",
        "HF tokenizers is required by default; --allow-no-hf is an explicit opt-out. When present, HF ids, decode, or batch mismatches fail validation.",
        "cProfile cannot see native kernel internals; cpu_per_wall only says whether the timed Python loop was CPU-bound.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({"status": results["status"],
                      "validation": {k: v["stats"] for k, v in results["validation"].items()},
                      "errors": results["errors"], "output": str(args.output)}, indent=2))
    return 1 if results["errors"] else 0

if __name__ == "__main__":
    sys.exit(main())
