#!/usr/bin/env python3
"""Warm throughput for LOCAL UNRELEASED tokie, PUBLIC gigatoken, and PUBLIC HF.

Common endpoints on one seeded OpenWebText slice:
  encode: text -> token ids
  decode: the same precomputed 1-D contiguous uint32 ids -> str
  batch:  the same documents -> ragged ids

The tokie artifact is a local wheel, not a published release. The wheel hash is
checked against --expected-tokie-wheel-sha256 and against the installed .so
before any timing. Gigatoken and tokenizers are public PyPI versions.

This is a current-state measurement, not a regression study and not a claim
about encode or batch improvements. Cold starts are not measured here.

    RAYON_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
    python scripts/bench_current.py --corpus corpus.jsonl \
        --model gpt2=/path/gpt2/tokenizer.json \
        --model qwen2=/path/qwen2/tokenizer.json \
        --model cl100k=/path/cl100k/tokenizer.json \
        --threads 1 --tasks encode,decode,batch \
        --rounds 5 --passes-encode 10 --passes-decode 10 --passes-batch 1 \
        --tokie-wheel /path/tokie.whl \
        --expected-tokie-wheel-sha256 <hex> --output results-1t.json
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import math
import os
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_published as bp  # frozen helper: load_corpus, build_fuzz, EDGE_TEXTS

LIBRARIES = ("tokie", "giga", "hf")
ORDERS_3 = [
    ["tokie", "giga", "hf"],
    ["hf", "giga", "tokie"],
    ["giga", "hf", "tokie"],
    ["tokie", "hf", "giga"],
    ["giga", "tokie", "hf"],
]
ORDERS_2 = [["tokie", "hf"], ["hf", "tokie"]]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

def model_algorithm(path):
    try:
        m = json.loads(Path(path).read_text(encoding="utf-8")).get("model", {})
    except Exception:
        return None
    if m.get("type"):
        return m["type"]
    if "merges" in m:
        return "BPE"
    if "unigram" in m:
        return "Unigram"
    if "continuing_subword_prefix" in m or "max_input_chars_per_word" in m:
        return "WordPiece"
    if "unk_id" in m:
        return "Unigram"
    return None

def package_meta(name):
    try:
        d = md.distribution(name)
        return {
            "version": d.version,
            "name": d.metadata.get("Name"),
            "installer": (d.read_text("INSTALLER") or "").strip() or None,
            "direct_url": (d.read_text("direct_url.json") or "").strip() or None,
        }
    except Exception:
        return {"version": None, "name": None, "installer": None, "direct_url": None}

def wheel_record(path):
    p = Path(path)
    return {"file": p.name, "path": str(p), "sha256": sha256_file(p),
            "bytes": p.stat().st_size}

def installed_artifacts(module, wheel):
    pkg_dir = Path(module.__file__).parent
    installed = {}
    for p in sorted(pkg_dir.rglob("*.so")):
        if p.is_file():
            installed[p.name] = sha256_file(p)
    members = {}
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            base = Path(name).name
            if base in installed:
                members[base] = hashlib.sha256(z.read(name)).hexdigest()
    return {"installed_native": installed, "wheel_native_members": members,
            "installed_matches_wheel": bool(installed) and installed == members}

def wheel_metadata_version(wheel):
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            if name.endswith(".dist-info/METADATA"):
                for line in z.read(name).decode("utf-8", "replace").splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
    return None

def wheel_hashes(wheels_dir):
    out = {}
    if wheels_dir is None:
        return out
    for pkg in ("tokie", "gigatoken", "tokenizers", "numpy", "awkward"):
        found = sorted(Path(wheels_dir).glob(pkg.replace("-", "_") + "-*.whl"))
        out[pkg] = [wheel_record(p) for p in found]
    return out

def git_state(repo):
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(repo), *cmd],
                                  capture_output=True, text=True, check=True).stdout
        except Exception as exc:
            return f"<error: {exc}>"
    status = [x for x in run("status", "--porcelain").splitlines() if x.strip()]
    paths = [x[3:] for x in status]
    return {
        "repo": str(repo),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_commit": run("rev-parse", "HEAD").strip(),
        "worktree_dirty": bool(status),
        "dirty_files": paths,
        "source_changes": [p for p in paths if p.startswith("crates/")],
        "generator_or_output_changes": [p for p in paths if not p.startswith("crates/")],
    }

def host_snapshot():
    root = Path("/sys/devices/system/cpu")
    governors, drivers, freqs = {}, {}, []
    for cpu in sorted(root.glob("cpu[0-9]*")):
        cp = cpu / "cpufreq"
        gov = bp.read_text(cp / "scaling_governor")
        if gov:
            governors[gov] = governors.get(gov, 0) + 1
        drv = bp.read_text(cp / "scaling_driver")
        if drv:
            drivers[drv] = drivers.get(drv, 0) + 1
        for name in ("scaling_cur_freq", "cpuinfo_cur_freq"):
            val = bp.read_text(cp / name)
            if val and val.isdigit():
                freqs.append(int(val))
                break
    return {
        "label": "one pre-validation snapshot; not sampled during timing and not causal proof",
        "governors": governors, "scaling_drivers": drivers,
        "cpus_with_freq": len(freqs),
        "mean_cur_freq_mhz": round(sum(freqs) / len(freqs) / 1000, 1) if freqs else None,
        "min_cur_freq_mhz": round(min(freqs) / 1000, 1) if freqs else None,
        "max_cur_freq_mhz": round(max(freqs) / 1000, 1) if freqs else None,
    }

def parse_specs(specs, flag):
    out = []
    for spec in specs:
        label, sep, path = spec.partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"bad {flag} {spec!r}; expected LABEL=PATH")
        if not Path(path).is_file():
            raise SystemExit(f"tokenizer.json not found for {label}: {path}")
        out.append((label, Path(path)))
    labels = [x for x, _ in out]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate model labels")
    return out

def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--model", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--hf-only-model", action="append", default=[], metavar="LABEL=PATH")
    p.add_argument("--threads", type=int, choices=(1, 8), default=1)
    p.add_argument("--tasks", default="encode,decode,batch")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--passes-encode", type=int, default=10)
    p.add_argument("--passes-decode", type=int, default=10)
    p.add_argument("--passes-batch", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=1_048_576)
    p.add_argument("--max-docs", type=int, default=256)
    p.add_argument("--batch-max-chars", type=int, default=16_777_216)
    p.add_argument("--batch-max-docs", type=int, default=4187)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--tokie-wheel", required=True, type=Path)
    p.add_argument("--expected-tokie-wheel-sha256", required=True)
    p.add_argument("--wheels-dir", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    for flag in ("rounds", "passes_encode", "passes_decode", "passes_batch",
                 "max_chars", "max_docs", "batch_max_chars", "batch_max_docs"):
        if getattr(a, flag) <= 0:
            p.error(f"--{flag.replace('_', '-')} must be > 0")
    a.tasks = [x.strip() for x in a.tasks.split(",") if x.strip()]
    bad = set(a.tasks) - {"encode", "decode", "batch"}
    if not a.tasks or bad:
        p.error("--tasks must be a non-empty subset of encode,decode,batch")
    if a.hf_only_model:
        if a.model:
            p.error("use either --model or --hf-only-model, not both")
        a.models = parse_specs(a.hf_only_model, "--hf-only-model")
        if len(a.models) != 2:
            p.error("HF-only runs require exactly two --hf-only-model entries")
        if "batch" in a.tasks:
            p.error("HF-only runs support --tasks encode,decode (no batch)")
        a.mode = "hf_only"
    else:
        a.models = parse_specs(a.model, "--model")
        if len(a.models) != 3:
            p.error("shared runs require exactly three --model entries")
        a.mode = "shared"
    if not Path(a.tokie_wheel).is_file():
        p.error(f"--tokie-wheel not found: {a.tokie_wheel}")
    return a

def task_libraries(mode):
    return ["tokie", "giga", "hf"] if mode == "shared" else ["tokie", "hf"]

def make_tokie(tokie, path, threads):
    t = tokie.Tokenizer.open(str(path))
    t.threads(threads)
    return t

def make_giga(gigatoken, path):
    return gigatoken.Tokenizer(str(path))

def encode_call(lib, inst, text):
    if lib in ("tokie", "giga"):
        return inst.encode(text)
    return inst.encode(text, add_special_tokens=False).ids

def decode_call(lib, inst, ids):
    if lib == "tokie":
        return inst.decode(ids)
    if lib == "giga":
        return inst.decode(ids).decode("utf-8", "replace")
    return inst.decode(ids.tolist(), skip_special_tokens=False)

def batch_call(lib, inst, texts, threads):
    if lib == "tokie":
        return inst.batch(texts)
    if lib == "giga":
        return inst.encode_batch(texts, parallel=(threads > 1))
    return [e.ids for e in inst.encode_batch(texts, add_special_tokens=False)]

def count_batch(lib, out):
    if lib == "tokie":
        return len(out[0]) + len(out[1])
    if lib == "giga":
        return len(out.layout.content) + len(out.layout.offsets) - 1
    return sum(len(x) for x in out) + len(out)

def batch_parts(lib, out, np, ak):
    if lib == "tokie":
        return np.asarray(out[0], dtype=np.uint32), np.asarray(out[1], dtype=np.uint64)
    if lib == "giga":
        flat = np.asarray(ak.flatten(out, axis=1), dtype=np.uint32)
        ends = np.asarray(out.layout.offsets, dtype=np.uint64)[1:]
        return flat, ends
    flat, ends, acc = [], [], 0
    for row in out:
        a = np.asarray(row, dtype=np.uint32)
        flat.append(a)
        acc += len(a)
        ends.append(acc)
    return np.concatenate(flat) if flat else np.asarray([], dtype=np.uint32), \
        np.asarray(ends, dtype=np.uint64)

def batch_fingerprint(flat, ends):
    return hashlib.sha256(flat.tobytes() + b":" + ends.tobytes()).hexdigest()

def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None

def mismatch(kind, lib, index, texts, **extra):
    return {"kind": kind, "library": lib, "index": index,
            "text_sha256": sha256_text(texts[index]),
            "text_bytes_utf8": len(texts[index].encode("utf-8")), **extra}

def validate_model(label, path, mode, tasks, timing_texts, validation_texts,
                   batch_texts, threads, tokie, gigatoken, hf_mod, np, ak):
    algorithm = model_algorithm(path)
    libs = task_libraries(mode)
    t = make_tokie(tokie, path, 1)
    hf = hf_mod.Tokenizer.from_file(str(path))
    g = make_giga(gigatoken, path) if "giga" in libs else None
    giga_refusal = None
    if g is None:
        try:
            probe = gigatoken.Tokenizer(str(path))
            giga_refusal = {"attempted": True, "supported": True,
                            "note": "gigatoken accepted this model; it is still not timed in HF-only mode"}
            del probe
        except Exception as exc:
            giga_refusal = {"attempted": True, "supported": False,
                            "error_type": type(exc).__name__,
                            "error": " ".join(str(exc).split())[:300]}
    ids_all, dec_expected = [], []
    enc_bad = dec_bad = giga_dec_bad = hf_enc_bad = hf_dec_bad = 0
    first = None
    for i, text in enumerate(validation_texts):
        ti = np.ascontiguousarray(np.asarray(t.encode(text), dtype=np.uint32))
        hi = np.ascontiguousarray(np.asarray(
            hf.encode(text, add_special_tokens=False).ids, dtype=np.uint32))
        ids_all.append(ti)
        if not np.array_equal(ti, hi):
            hf_enc_bad += 1
            first = first or mismatch("encode_hf", "hf", i, validation_texts,
                                      actual_len=int(len(ti)), hf_len=int(len(hi)),
                                      first_diff=first_diff(ti, hi))
        gt = None
        if g is not None:
            gt = np.ascontiguousarray(np.asarray(g.encode(text), dtype=np.uint32))
            if not np.array_equal(ti, gt):
                enc_bad += 1
                first = first or mismatch("encode_giga", "giga", i, validation_texts,
                                          tokie_len=int(len(ti)), giga_len=int(len(gt)),
                                          first_diff=first_diff(ti, gt))
        hs = hf.decode(ti.tolist(), skip_special_tokens=False)
        ts = t.decode(ti)
        if ts != hs:
            dec_bad += 1
            hf_dec_bad += 1
            first = first or mismatch("decode_tokie", "tokie", i, validation_texts,
                                      tokie_chars=len(ts), hf_chars=len(hs),
                                      tokie_sha256=sha256_text(ts), hf_sha256=sha256_text(hs))
        if gt is not None:
            gs = g.decode(ti).decode("utf-8", "replace")
            if gs != hs:
                dec_bad += 1
                giga_dec_bad += 1
                first = first or mismatch("decode_giga", "giga", i, validation_texts,
                                          giga_chars=len(gs), hf_chars=len(hs),
                                          giga_sha256=sha256_text(gs), hf_sha256=sha256_text(hs))
        dec_expected.append(hs)
    timing_ids = [np.ascontiguousarray(x) for x in ids_all[:len(timing_texts)]]
    expected_decoded = dec_expected[:len(timing_texts)]
    expected_chars = sum(len(s) for s in expected_decoded)
    expected_decoded_bytes = sum(len(s.encode("utf-8")) for s in expected_decoded)

    batch_bad = 0
    batch_stats = {"small_batch_run": False, "large_batch_run": False}
    large_flat = large_ends = None
    if "batch" in tasks and mode == "shared":
        tv = batch_call("tokie", t, validation_texts, threads)
        small_flat, small_ends = batch_parts("tokie", tv, np, ak)
        hfv = hf.encode_batch(validation_texts, add_special_tokens=False)
        hf_flat, hf_ends = batch_parts("hf", [e.ids for e in hfv], np, ak)
        same = np.array_equal(small_flat, hf_flat) and np.array_equal(small_ends, hf_ends)
        if not same:
            batch_bad += 1
            first = first or {"kind": "batch_small_hf", "library": "hf",
                              "first_diff_flat": first_diff(small_flat, hf_flat),
                              "first_diff_ends": first_diff(small_ends, hf_ends)}
        small_dec_bad = 0
        for i, (a, b) in enumerate(zip(small_ends.tolist(), [0] + small_ends.tolist()[:-1])):
            hs = hf.decode(small_flat[b:a].tolist(), skip_special_tokens=False)
            if t.decode(np.ascontiguousarray(small_flat[b:a])) != hs:
                small_dec_bad += 1
                first = first or {"kind": "batch_small_decode_tokie", "library": "tokie",
                                  "index": i}
        if small_dec_bad:
            batch_bad += 1
        batch_stats.update({"small_batch_run": True, "small_batch_tokens": int(len(small_flat)),
                            "small_batch_documents": int(len(small_ends)),
                            "small_batch_decode_mismatches": small_dec_bad})
        if g is not None:
            gv = batch_call("giga", g, validation_texts, threads)
            gf, ge = batch_parts("giga", gv, np, ak)
            if not (np.array_equal(gf, hf_flat) and np.array_equal(ge, hf_ends)):
                batch_bad += 1
                first = first or {"kind": "batch_small_giga", "library": "giga",
                                  "first_diff_flat": first_diff(gf, hf_flat),
                                  "first_diff_ends": first_diff(ge, hf_ends)}
        large = batch_call("tokie", t, batch_texts, threads)
        large_flat, large_ends = batch_parts("tokie", large, np, ak)
        hfl = hf.encode_batch(batch_texts, add_special_tokens=False)
        hf_lf, hf_le = batch_parts("hf", [e.ids for e in hfl], np, ak)
        if not (np.array_equal(large_flat, hf_lf) and np.array_equal(large_ends, hf_le)):
            batch_bad += 1
            first = first or {"kind": "batch_large_hf", "library": "hf",
                              "first_diff_flat": first_diff(large_flat, hf_lf),
                              "first_diff_ends": first_diff(large_ends, hf_le)}
        if g is not None:
            gl = batch_call("giga", g, batch_texts, threads)
            gf, ge = batch_parts("giga", gl, np, ak)
            if not (np.array_equal(gf, hf_lf) and np.array_equal(ge, hf_le)):
                batch_bad += 1
                first = first or {"kind": "batch_large_giga", "library": "giga",
                                  "first_diff_flat": first_diff(gf, hf_lf),
                                  "first_diff_ends": first_diff(ge, hf_le)}
        batch_stats.update({"large_batch_run": True,
                            "batch_documents": int(len(large_ends)),
                            "batch_tokens": int(len(large_flat)),
                            "batch_flat_sha256": batch_fingerprint(large_flat, large_ends),
                            "hf_batch_flat_sha256": batch_fingerprint(hf_lf, hf_le)})
    ok = enc_bad == dec_bad == hf_enc_bad == hf_dec_bad == batch_bad == 0
    stats = {"texts": len(validation_texts), "tokens": int(sum(len(x) for x in ids_all)),
             "timing_documents": len(timing_texts),
             "timing_tokens": int(sum(len(x) for x in timing_ids)),
             "decode_chars": expected_chars, "decoded_bytes_utf8": expected_decoded_bytes,
             "encode_mismatches_tokie_vs_hf": hf_enc_bad,
             "decode_mismatches_tokie_vs_hf": hf_dec_bad,
             "encode_mismatches_tokie_vs_giga": enc_bad if g is not None else None,
             "decode_mismatches_tokie_vs_giga": giga_dec_bad if g is not None else None,
             "batch_mismatches": batch_bad, **batch_stats}
    return {
        "ok": ok,
        "algorithm": algorithm,
        "libraries": libs,
        "giga_refusal": giga_refusal,
        "validation": {"ok": ok, "first_mismatch": first,
                       "encode_mismatches": enc_bad + hf_enc_bad,
                       "decode_mismatches": dec_bad,
                       "batch_mismatches": batch_bad,
                       "validation_texts": len(validation_texts), "stats": stats},
        "timing_ids": timing_ids,
        "expected_decoded": expected_decoded,
        "expected_chars": expected_chars,
        "expected_decoded_bytes": expected_decoded_bytes,
        "large_flat": large_flat, "large_ends": large_ends,
        "batch_tokens": int(len(large_flat)) if large_flat is not None else 0,
    }

def warm(instances, libs, task, texts, ids, batch_texts, threads):
    for lib in libs:
        if task == "encode":
            for text in texts:
                encode_call(lib, instances[lib], text)
        elif task == "decode":
            for x in ids:
                decode_call(lib, instances[lib], x)
        else:
            batch_call(lib, instances[lib], batch_texts, threads)

def timed(instances, libs, task, texts, ids, batch_texts, threads, passes):
    out = {}
    for lib in libs:
        counts, last, n = [], None, 0
        t0 = time.perf_counter()
        if task == "encode":
            for _ in range(passes):
                c = 0
                for text in texts:
                    last = encode_call(lib, instances[lib], text)
                    c += len(last)
                counts.append(c)
            n = counts[0] if counts else 0
        elif task == "decode":
            for _ in range(passes):
                c = 0
                for x in ids:
                    last = decode_call(lib, instances[lib], x)
                    c += len(last)
                counts.append(c)
            n = counts[0] if counts else 0
        else:
            for _ in range(passes):
                last = batch_call(lib, instances[lib], batch_texts, threads)
                counts.append(count_batch(lib, last))
            n = counts[0] if counts else 0
        out[lib] = {"seconds": time.perf_counter() - t0, "count": n, "passes": passes,
                    "counts": counts, "last": last}
    return out

def summarize(rounds, libs, numerator_bytes, count, passes, count_unit):
    out = {}
    for lib in libs:
        secs = [r[lib]["seconds"] for r in rounds]
        med = statistics.median(secs)
        base = {"count_unit": count_unit,
                "seconds_median": med, "seconds_min": min(secs), "seconds_max": max(secs),
                "mib_s_median": numerator_bytes * passes / med / (1 << 20),
                "mib_s_min": numerator_bytes * passes / max(secs) / (1 << 20),
                "mib_s_max": numerator_bytes * passes / min(secs) / (1 << 20),
                "count_s_median": count * passes / med,
                "count_s_min": count * passes / max(secs),
                "count_s_max": count * passes / min(secs)}
        unit = "tokens" if count_unit == "ids" else "chars" if count_unit == "characters" else None
        if unit:
            for suffix in ("median", "min", "max"):
                base[f"{unit}_s_{suffix}"] = base[f"count_s_{suffix}"]
        out[lib] = base
    return out

def post_check(lib, task, last, expected, np, ak):
    if task == "encode":
        arr = np.asarray(last, dtype=np.uint32)
        ok = np.array_equal(arr, expected["ids"])
        return {"output_ok": bool(ok), "checksum": hashlib.sha256(arr.tobytes()).hexdigest()}
    if task == "decode":
        ok = last == expected["text"]
        return {"output_ok": bool(ok), "checksum": sha256_text(last)}
    flat, ends = batch_parts(lib, last, np, ak)
    ok = np.array_equal(flat, expected["flat"]) and np.array_equal(ends, expected["ends"])
    return {"output_ok": bool(ok), "checksum": batch_fingerprint(flat, ends)}

def profile_one(instances, lib, task, texts, ids, batch_texts, threads):
    c0, w0 = time.process_time(), time.perf_counter()
    if task == "encode":
        for text in texts:
            encode_call(lib, instances[lib], text)
    elif task == "decode":
        for x in ids:
            decode_call(lib, instances[lib], x)
    else:
        batch_call(lib, instances[lib], batch_texts, threads)
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    return {"wall_s": wall, "process_cpu_s": cpu,
            "cpu_per_wall": cpu / wall if wall else None,
            "passes": 1, "cProfile": False,
            "note": "one unprofiled pass after timed rounds; native internals not attributed"}

def run_task(label, task, libs, path, timing_texts, timing_ids, expected_decoded,
             batch_texts, large_flat, large_ends, threads, args,
             tokie, gigatoken, hf_mod, np, ak):
    if task == "batch" and (large_flat is None or large_ends is None):
        raise AssertionError(f"{label}: batch timing needs validated large-batch arrays")
    if task != "batch" and not timing_ids:
        raise AssertionError(f"{label}: no timing ids")
    t = make_tokie(tokie, path, 1 if task != "batch" else threads)
    instances = {"tokie": t}
    if "giga" in libs:
        instances["giga"] = make_giga(gigatoken, path)
    if "hf" in libs:
        instances["hf"] = hf_mod.Tokenizer.from_file(str(path))
    passes = {"encode": args.passes_encode, "decode": args.passes_decode,
              "batch": args.passes_batch}[task]
    if task == "batch":
        documents = len(batch_texts)
        input_bytes = sum(len(x.encode("utf-8")) for x in batch_texts)
        count_unit = "ids_and_documents"
        expected_count = int(len(large_flat)) + int(len(large_ends))
        decoded_bytes = 0
    elif task == "decode":
        documents = len(timing_texts)
        input_bytes = sum(len(x.encode("utf-8")) for x in timing_texts)
        count_unit = "characters"
        expected_count = sum(len(s) for s in expected_decoded)
        decoded_bytes = sum(len(s.encode("utf-8")) for s in expected_decoded)
    else:
        documents = len(timing_texts)
        input_bytes = sum(len(x.encode("utf-8")) for x in timing_texts)
        count_unit = "ids"
        expected_count = sum(len(x) for x in timing_ids)
        decoded_bytes = 0
    numerator = decoded_bytes if task == "decode" else input_bytes

    warm(instances, libs, task, timing_texts, timing_ids, batch_texts, threads)
    rounds, last = [], {}
    for r in range(args.rounds):
        order = (ORDERS_3 if len(libs) == 3 else ORDERS_2)[r % (5 if len(libs) == 3 else 2)]
        rec = {"round": r + 1, "order": order}
        got = timed(instances, order, task, timing_texts, timing_ids,
                    batch_texts, threads, passes)
        for lib in libs:
            counts = got[lib]["counts"]
            if len(set(counts)) != 1:
                raise AssertionError(f"{label} {task} {lib}: unstable counts {counts[:3]}")
            if got[lib]["count"] != expected_count:
                raise AssertionError(
                    f"{label} {task} {lib}: count {got[lib]['count']} != {expected_count}")
            rec[lib] = {"seconds": got[lib]["seconds"], "count": got[lib]["count"],
                        "passes": passes}
            last[lib] = got[lib]["last"]
        rounds.append(rec)

    if task == "encode":
        expected = {"ids": np.asarray(timing_ids[-1], dtype=np.uint32)}
    elif task == "decode":
        expected = {"text": expected_decoded[-1]}
    else:
        expected = {"flat": large_flat, "ends": large_ends}
    checks = {lib: post_check(lib, task, last[lib], expected, np, ak) for lib in libs}
    if not all(x["output_ok"] for x in checks.values()):
        raise AssertionError(f"{label} {task}: post-check failed {checks}")

    return {
        "threads": 1 if task != "batch" else threads,
        "documents": documents,
        "input_bytes_utf8": input_bytes,
        "decoded_bytes_utf8": decoded_bytes,
        "throughput_numerator": "decoded_bytes_utf8" if task == "decode" else "input_bytes_utf8",
        "count_unit": count_unit, "count_includes_documents": task == "batch",
        "passes_per_round": passes,
        "rounds": rounds,
        "summary": summarize(rounds, libs, numerator, expected_count, passes, count_unit),
        "post_check": checks,
        "profile": {lib: profile_one(instances, lib, task, timing_texts, timing_ids,
                                     batch_texts, threads) for lib in libs},
        "cold": "not measured; warm throughput only",
    }

def finite_dump(obj, path):
    text = json.dumps(obj, indent=1, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")

def main():
    args = parse_args()
    # Must precede every native import.
    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true" if args.threads > 1 else "false"
    import awkward as ak
    import numpy as np
    import tokie
    import gigatoken
    import tokenizers as hf_mod

    script_path = Path(__file__).resolve()
    script_sha_start = sha256_file(script_path)
    load_before = list(os.getloadavg())
    host = host_snapshot()
    docs, picked, picked_chars = bp.load_corpus(args.corpus, args.seed, args.max_chars, args.max_docs)
    batch_docs, batch_picked, batch_chars = bp.load_corpus(
        args.corpus, args.seed, args.batch_max_chars, args.batch_max_docs)
    timing_texts = [t for _, t in picked]
    batch_texts = [t for _, t in batch_picked]
    fuzz = bp.build_fuzz(args.seed)
    validation_texts = timing_texts + bp.EDGE_TEXTS + fuzz
    input_bytes = sum(len(t.encode("utf-8")) for t in timing_texts)
    batch_bytes = sum(len(t.encode("utf-8")) for t in batch_texts)
    libs = task_libraries(args.mode)
    manifest_path = args.corpus.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None

    tokie_wheel = wheel_record(args.tokie_wheel)
    expected = args.expected_tokie_wheel_sha256.strip().lower()
    wheel_ok = tokie_wheel["sha256"].lower() == expected
    installed = installed_artifacts(tokie, args.tokie_wheel)
    meta_version = wheel_metadata_version(args.tokie_wheel)
    pkg_tokie = package_meta("tokie")
    pkg_giga = package_meta("gigatoken")
    pkg_hf = package_meta("tokenizers")
    guard_error = None
    if not wheel_ok:
        guard_error = f"tokie wheel sha256 {tokie_wheel['sha256']} != expected {expected}"
    elif not installed["installed_matches_wheel"]:
        guard_error = "installed tokie .so bytes do not match the wheel member"
    elif meta_version and pkg_tokie["version"] != meta_version:
        guard_error = f"installed tokie metadata version {pkg_tokie['version']} != wheel {meta_version}"

    wheels_dir_hashes = wheel_hashes(args.wheels_dir)
    giga_wheel = None
    for rec in wheels_dir_hashes.get("gigatoken", []):
        giga_wheel = Path(rec["path"])
        break
    giga_installed = installed_artifacts(gigatoken, giga_wheel) if giga_wheel else None
    hf_wheel = None
    for rec in wheels_dir_hashes.get("tokenizers", []):
        hf_wheel = Path(rec["path"])
        break
    hf_installed = installed_artifacts(hf_mod, hf_wheel) if hf_wheel else None
    old_tokie_wheels = wheels_dir_hashes.get("tokie", [])
    tokie_source = git_state(script_path.parents[1])
    binding_src = script_path.parents[1] / "crates/tokie-python/src/lib.rs"
    if binding_src.is_file():
        tokie_source["binding_source_sha256"] = sha256_file(binding_src)
    giga_record = wheel_record(giga_wheel) if giga_wheel else None
    hf_record = wheel_record(hf_wheel) if hf_wheel else None

    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "argv": sys.argv,
        "claim_scope": (f"current-state warm throughput on one seeded OpenWebText slice; "
                        f"local unreleased tokie wheel vs public gigatoken {pkg_giga['version']} and "
                        f"public HF tokenizers {pkg_hf['version']}; not a regression study"),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "executable": sys.executable,
        "cpu_model": bp.cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "load_before": load_before,
        "host": host,
        "env": {k: os.environ.get(k) for k in
                ("RAYON_NUM_THREADS", "TOKENIZERS_PARALLELISM", "LD_LIBRARY_PATH")},
        "artifacts": {
            "tokie": {
                "version": pkg_tokie["version"],
                "kind": "local_unreleased",
                "wheel_sha256": tokie_wheel["sha256"],
                "wheel_file": tokie_wheel["file"],
                "wheel_bytes": tokie_wheel["bytes"],
                "wheel": tokie_wheel,
                "expected_wheel_sha256": expected,
                "wheel_sha256_matches_expected": wheel_ok,
                "wheel_metadata_version": meta_version,
                "installed_metadata": pkg_tokie,
                "import_path": tokie.__file__,
                **installed,
                "provenance": ("local build installed from --tokie-wheel; the 0.2.0 version "
                               "string does not identify this artifact and it is not claimed as published"),
            },
            "giga": {
                "version": pkg_giga["version"], "kind": "pypi",
                "import_path": gigatoken.__file__,
                "wheel_sha256": giga_record["sha256"] if giga_record else None,
                "wheel": giga_record, "installed_artifacts": giga_installed,
            },
            "hf": {
                "version": pkg_hf["version"], "kind": "pypi",
                "import_path": hf_mod.__file__,
                "wheel_sha256": hf_record["sha256"] if hf_record else None,
                "wheel": hf_record, "installed_artifacts": hf_installed,
            },
            "numpy": {"version": np.__version__, "kind": "pypi", "import_path": np.__file__},
            "awkward": {"version": ak.__version__, "kind": "pypi", "import_path": ak.__file__},
        },
        "wheels_dir": str(args.wheels_dir) if args.wheels_dir else None,
        "wheels": wheels_dir_hashes,
        "tokie_wheel_in_wheels_dir_not_used": old_tokie_wheels,
        "harness": {"script_path": str(script_path), "sha256_at_start": script_sha_start},
        "tokie_source": tokie_source,
        "samples": {
            "single": {"documents": len(timing_texts), "chars": picked_chars,
                       "input_bytes_utf8": input_bytes, "max_chars": args.max_chars,
                       "max_docs": args.max_docs, "sampling_seed": args.seed,
                       "rows_sha256": sha256_text(",".join(str(r) for r, _ in picked)),
                       "row_range": [picked[0][0], picked[-1][0]] if picked else None,
                       "validation_texts": len(validation_texts),
                       "edge_texts": len(bp.EDGE_TEXTS), "fuzz_texts": len(fuzz)},
            "batch": {"documents": len(batch_texts), "chars": batch_chars,
                      "input_bytes_utf8": batch_bytes, "max_chars": args.batch_max_chars,
                      "max_docs": args.batch_max_docs, "sampling_seed": args.seed,
                      "rows_sha256": sha256_text(",".join(str(r) for r, _ in batch_picked)),
                      "row_range": [batch_picked[0][0], batch_picked[-1][0]] if batch_picked else None},
        },
        "tasks": {
            "encode": {"description": "raw UTF-8 text -> ids",
                       "adapters": {"tokie": "Tokenizer.encode(text) -> np.uint32 ids",
                                    "giga": "Tokenizer.encode(text) -> np.uint32 ids",
                                    "hf": "Tokenizer.encode(text, add_special_tokens=False).ids -> list"},
                       "counted_inside_clock": "len(ids); HF .ids materialization is included",
                       "count_unit": "ids"},
            "decode": {"description": "the same precomputed 1-D contiguous np.uint32 ids -> str",
                       "input": "one shared contiguous uint32 array per document, built outside every clock",
                       "adapters": {"tokie": "Tokenizer.decode(ids) -> str",
                                    "giga": "Tokenizer.decode(ids) -> bytes; .decode('utf-8','replace') inside clock",
                                    "hf": "Tokenizer.decode(ids.tolist(), skip_special_tokens=False); array-to-list inside clock"},
                       "counted_inside_clock": "len(str) characters; decoded UTF-8 bytes computed outside clock",
                       "count_unit": "characters", "throughput_numerator": "decoded_bytes_utf8",
                       "note": "adapters differ by library; not comparable to an earlier bytes-only decode benchmark"},
            "batch": {"description": "the same documents -> ragged ids",
                      "adapters": {"tokie": "Tokenizer.batch(texts) -> (flat uint32, ends uint64)",
                                   "giga": "Tokenizer.encode_batch(texts, parallel=n>1) -> awkward Array with leading 0 offset",
                                   "hf": "Tokenizer.encode_batch(texts, add_special_tokens=False) -> list[Encoding]; [e.ids] inside clock"},
                      "counted_inside_clock": "all ids + documents", "count_unit": "ids_and_documents",
                      "count_includes_documents": True,
                      "note": "public output types differ; Giga is not flattened inside any clock; one pass per round; batch input is the larger seeded sample"},
        },
        "timing": {
            "warmth": "warm throughput only; cold starts were not retimed and old cold data stays frozen",
            "rounds": args.rounds,
            "passes_per_round": {"encode": args.passes_encode, "decode": args.passes_decode,
                                  "batch": args.passes_batch},
            "warmup_passes_per_library": 1,
            "orders": {"three_libraries": ORDERS_3, "two_libraries": ORDERS_2},
            "single_operation_threads": 1,
            "batch_threads": args.threads,
            "env_set_before_native_imports": True,
            "profile": "one unprofiled process CPU vs wall pass per library and task after rounds; no cProfile or perf trace",
        },
        "corpus": {
            "jsonl": str(args.corpus), "jsonl_sha256": sha256_file(args.corpus),
            "jsonl_bytes": args.corpus.stat().st_size, "available_documents": len(docs),
            "manifest": str(manifest_path) if manifest else None,
            "manifest_sha256": sha256_file(manifest_path) if manifest else None,
            "manifest_embedded": manifest,
            "source": ({k: manifest.get(k) for k in
                        ("dataset", "parquet", "source_sha256", "source_rows", "selection")}
                       if manifest else None),
        },
    }
    results = {"schema_version": 1, "meta": meta, "models": {}, "validation": {},
               "benchmark": {}, "status": {}, "errors": []}
    if guard_error:
        results["errors"].append(guard_error)
        for label, _ in args.models:
            results["status"][label] = "wheel_guard_failed"
        results["meta"]["harness"].update({"sha256_at_end": sha256_file(script_path)})
        finite_dump(results, args.output)
        print(f"FAIL wheel guard: {guard_error}")
        return 1

    for label, path in args.models:
        algorithm = model_algorithm(path)
        if args.mode == "shared" and algorithm != "BPE":
            results["status"][label] = "unsupported"
            results["errors"].append(f"{label}: shared mode needs BPE, got {algorithm}")
            continue
        try:
            v = validate_model(label, path, args.mode, args.tasks, timing_texts,
                               validation_texts, batch_texts, args.threads,
                               tokie, gigatoken, hf_mod, np, ak)
        except Exception as exc:
            results["status"][label] = "error"
            results["errors"].append(f"{label}: validation raised {type(exc).__name__}: {exc}")
            results["models"][label] = {"path": str(path), "tokenizer_json": str(path),
                                        "tokenizer_json_sha256": sha256_file(path),
                                        "tokenizer_json_bytes": path.stat().st_size,
                                        "algorithm": algorithm,
                                        "libraries": libs}
            continue
        results["models"][label] = {
            "path": str(path), "tokenizer_json": str(path),
            "tokenizer_json_sha256": sha256_file(path),
            "tokenizer_json_bytes": path.stat().st_size,
            "algorithm": v["algorithm"], "libraries": libs,
            "giga_refusal": v["giga_refusal"],
        }
        results["validation"][label] = v["validation"]
        if not v["ok"]:
            results["status"][label] = "invalid"
            results["errors"].append(f"{label}: validation mismatch; timing skipped")
            continue
        try:
            bench = {}
            for task in args.tasks:
                bench[task] = run_task(
                    label, task, libs, path, timing_texts, v["timing_ids"],
                    v["expected_decoded"], batch_texts, v["large_flat"], v["large_ends"],
                    args.threads, args, tokie, gigatoken, hf_mod, np, ak)
            results["benchmark"][label] = bench
            results["status"][label] = "ok"
        except Exception as exc:
            results["status"][label] = "error"
            results["errors"].append(f"{label}: timing raised {type(exc).__name__}: {exc}")

    results["meta"]["load_after"] = list(os.getloadavg())
    script_sha_end = sha256_file(script_path)
    results["meta"]["harness"].update({"sha256_at_end": script_sha_end,
                                       "unchanged": script_sha_end == script_sha_start})
    if script_sha_end != script_sha_start:
        results["errors"].append("benchmark script changed during the run")
    if set(results["status"]) != {label for label, _ in args.models}:
        results["errors"].append("status does not cover every model")
    if any(v != "ok" for v in results["status"].values()) and not results["errors"]:
        results["errors"].append("non-ok status without an error")
    finite_dump(results, args.output)

    print(f"wrote {args.output}")
    for label in results["status"]:
        print(f"  {label}: {results['status'][label]}")
    print(f"validation errors: {len(results['errors'])}")
    return 0 if not results["errors"] and all(v == "ok" for v in results["status"].values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
