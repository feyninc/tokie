#!/usr/bin/env python3
"""Published tokie 0.2.0 encode/decode validation + single-thread throughput.

Compares the public PyPI wheel against HuggingFace `tokenizers` on the same
`tokenizer.json`, over a seeded OpenWebText sample plus deterministic edges.

Scope: correctness (exact ids, decoder output) and a ballpark throughput number.
Not a formal regression test and not a claim about older versions.
"""

import os

# Must be set before tokenizers / tokie are imported.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"

import cProfile
import hashlib
import importlib.metadata as md
import io
import json
import pstats
import random
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import tokenizers
import tokie

HERE = Path(__file__).resolve().parent
CORPUS = Path("/tmp/opencode/tokie-compat/corpus")
MANIFEST = CORPUS / "manifest.json"
JSONL = CORPUS / "owt_20m.jsonl"

SEED = 20261111
MAX_DOCS = 256
MAX_CHARS = 1_048_576  # ~1 MiB of characters
ROUNDS = 5
WARMUP = 1

MODELS = {
    "gpt2": {
        "path": "/home/bhavnick/.cache/huggingface/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e/tokenizer.json",
        "repo": "openai-community/gpt2",
        "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e",
        "algo": "BPE",
        "special": "<|endoftext|>",
    },
    "bert-base-uncased": {
        "path": "/home/bhavnick/.cache/huggingface/hub/models--google-bert--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/tokenizer.json",
        "repo": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "algo": "WordPiece",
        "special": "[SEP] [CLS] [MASK]",
    },
    "t5-small": {
        "path": "/home/bhavnick/.cache/huggingface/hub/models--t5-small/snapshots/main/tokenizer.json",
        "repo": "t5-small",
        "revision": "main (local snapshot dir; no refs file)",
        "algo": "Unigram",
        "special": "</s> <pad>",
    },
}

EDGE_TEXTS = [
    "",
    " ",
    "\n",
    "\t",
    "\r\n",
    "a\tb\nc\r\nd",
    "\x00\x01\x02 control \x07\x08",
    "line one\nline two\n\nline three",
    "café naïve résumé",
    "e\u0301 combining acute",
    "Z\u0335a\u0336l\u0337g\u0338o\u0334",
    "😀🎉🚀👩‍👩‍👧‍👦",
    "日本語のテキスト、中文文本、한국어",
    "العربية نص عربي",
    "mixed عربى 日本語 😀 café",
    "  leading and trailing  ",
    "UPPER lower MiXeD",
    "punctuation!?,.;:'\"()[]{}<>@#$%^&*_-+=/\\|~`",
    "numbers 0 1 42 3.14159 1e10 2026",
    "https://example.com/a?b=c&d=e#frag",
    "\\n is not a newline",
    "\u00a0non-breaking\u2003em space",
    "\ufeffBOM",
    "zero\u200bwidth\u200djoiner",
]

FUZZ_N = 256
FUZZ_POOL = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!\n\t")
    + list("éüñßçøåæœ")
    + list("日本語中文한국어")
    + list("العربية")
    + list("😀🎉🚀🔥✨")
    + ["\u0301", "\u0335", "\u200d", "\u200b", "\u00a0", "\u3000", "\ufeff"]
    + ["<|endoftext|>", "[SEP]", "[MASK]", "</s>", "<pad>", "\x00", "\x07"]
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_corpus():
    manifest = json.loads(MANIFEST.read_text())
    docs = []
    with open(JSONL, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append((rec["row"], rec["text"]))
    rng = random.Random(SEED)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    picked, chars = [], 0
    for i in idx:
        row, text = docs[i]
        if picked and chars + len(text) > MAX_CHARS:
            continue
        picked.append((row, text))
        chars += len(text)
        if len(picked) >= MAX_DOCS or chars >= MAX_CHARS:
            break
    picked.sort(key=lambda x: x[0])
    return manifest, docs, picked, chars


def build_fuzz():
    rng = random.Random(SEED ^ 0x5EED)
    out = []
    for _ in range(FUZZ_N):
        n = rng.randint(0, 48)
        out.append("".join(rng.choice(FUZZ_POOL) for _ in range(n)))
    return out


def to_ids(arr):
    return [int(x) for x in arr]


def validate(tok, hf, texts, label, check_roundtrip=False):
    """Return (ok, stats, first_mismatch). Exact ids, then decoder output."""
    enc_mismatch = dec_mismatch = rt_mismatch = 0
    first = None
    tok_ids, hf_ids = [], []
    total_tokens = 0
    for i, text in enumerate(texts):
        a = to_ids(tok.encode(text))
        b = list(hf.encode(text, add_special_tokens=False).ids)
        tok_ids.append(a)
        hf_ids.append(b)
        total_tokens += len(a)
        if check_roundtrip and tok.decode(a, skip=False) != text:
            rt_mismatch += 1
            if first is None:
                first = {"kind": "roundtrip", "index": i, "text_preview": text[:120]}
        if a != b:
            enc_mismatch += 1
            if first is None:
                first = {
                    "kind": "encode",
                    "index": i,
                    "text_preview": text[:120],
                    "tokie_len": len(a),
                    "hf_len": len(b),
                    "tokie_first12": a[:12],
                    "hf_first12": b[:12],
                }
            continue
        da = tok.decode(a, skip=False)
        db = hf.decode(b, skip_special_tokens=False)
        if da != db:
            dec_mismatch += 1
            if first is None:
                first = {
                    "kind": "decode",
                    "index": i,
                    "text_preview": text[:120],
                    "tokie_preview": da[:120],
                    "hf_preview": db[:120],
                }
    ok = enc_mismatch == 0 and dec_mismatch == 0 and rt_mismatch == 0
    return ok, {
        "label": label,
        "texts": len(texts),
        "tokens": total_tokens,
        "encode_mismatches": enc_mismatch,
        "decode_mismatches": dec_mismatch,
        "roundtrip_mismatches": rt_mismatch if check_roundtrip else None,
    }, first, tok_ids, hf_ids


def timed_pass(fn, items):
    t0 = time.perf_counter()
    acc = 0
    for x in items:
        acc += len(fn(x))
    return time.perf_counter() - t0, acc


def bench_model(name, cfg, texts, tok_ids, hf_ids, input_bytes):
    tok = tokie.Tokenizer.open(cfg["path"])
    hf = tokenizers.Tokenizer.from_file(cfg["path"])
    encode_tok = lambda s: tok.encode(s)
    encode_hf = lambda s: hf.encode(s, add_special_tokens=False).ids
    decode_tok = lambda ids: tok.decode(ids, skip=False)
    decode_hf = lambda ids: hf.decode(ids, skip_special_tokens=False)

    # One cold pass per impl, before warmup. Not part of the reported rounds:
    # it includes one-time table/allocator init.
    cold = {}
    for impl, fn, items in (
        ("tokie_encode", encode_tok, texts),
        ("hf_encode", encode_hf, texts),
        ("tokie_decode", decode_tok, tok_ids),
        ("hf_decode", decode_hf, hf_ids),
    ):
        t0 = time.perf_counter()
        for x in items:
            len(fn(x))
        cold[impl] = time.perf_counter() - t0

    # Warmup both, one pass each, unreported.
    for _ in range(WARMUP):
        timed_pass(encode_tok, texts)
        timed_pass(encode_hf, texts)
        timed_pass(decode_tok, tok_ids)
        timed_pass(decode_hf, hf_ids)

    rounds = []
    for r in range(ROUNDS):
        order = ["tokie", "hf"] if r % 2 == 0 else ["hf", "tokie"]
        rec = {"round": r + 1, "order": order}
        for impl in order:
            if impl == "tokie":
                e, ec = timed_pass(encode_tok, texts)
                d, dc = timed_pass(decode_tok, tok_ids)
            else:
                e, ec = timed_pass(encode_hf, texts)
                d, dc = timed_pass(decode_hf, hf_ids)
            rec[impl] = {"encode_s": e, "decode_s": d, "encode_count": ec, "decode_count": dc}
        rounds.append(rec)

    total_tokens = sum(len(x) for x in tok_ids)
    decoded_bytes = sum(len(tok.decode(x, skip=False).encode("utf-8")) for x in tok_ids)

    summary = {}
    for impl in ("tokie", "hf"):
        es = [r[impl]["encode_s"] for r in rounds]
        ds = [r[impl]["decode_s"] for r in rounds]
        summary[impl] = {
            "encode_s_median": statistics.median(es),
            "encode_s_min": min(es),
            "encode_s_max": max(es),
            "decode_s_median": statistics.median(ds),
            "decode_s_min": min(ds),
            "decode_s_max": max(ds),
            "encode_mib_s_median": input_bytes / statistics.median(es) / (1 << 20),
            "decode_tokens_s_median": total_tokens / statistics.median(ds),
            "decode_mib_s_median": decoded_bytes / statistics.median(ds) / (1 << 20),
        }
    return rounds, summary, total_tokens, decoded_bytes, cold


def profile_model(name, cfg, texts, tok_ids, hf_ids):
    """One unreported iteration per impl: cProfile + process CPU vs wall."""
    tok = tokie.Tokenizer.open(cfg["path"])
    hf = tokenizers.Tokenizer.from_file(cfg["path"])
    out = {}
    for impl, fn, items in (
        ("tokie_encode", lambda s: tok.encode(s), texts),
        ("hf_encode", lambda s: hf.encode(s, add_special_tokens=False).ids, texts),
        ("tokie_decode", lambda ids: tok.decode(ids, skip=False), tok_ids),
        ("hf_decode", lambda ids: hf.decode(ids, skip_special_tokens=False), hf_ids),
    ):
        pr = cProfile.Profile()
        c0, w0 = time.process_time(), time.perf_counter()
        pr.enable()
        acc = 0
        for x in items:
            acc += len(fn(x))
        pr.disable()
        cpu, wall = time.process_time() - c0, time.perf_counter() - w0
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(6)
        top = []
        for line in buf.getvalue().splitlines():
            m = re.match(r"\s*(\d+(?:/\d+)?)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.*)", line)
            if m:
                top.append({"cumtime_s": float(m.group(4)), "func": m.group(6)})
        out[impl] = {"wall_s": wall, "process_cpu_s": cpu, "cpu_per_wall": cpu / wall, "top": top[:5]}
    return out


def pick_samples(picked, n=3):
    """Short, plain-looking docs for the artifact. Skip obvious profanity."""
    blocked = {
        "fuck", "shit", "cunt", "nigger", "nigga", "faggot", "rape", "porn",
        "cock", "dick", "pussy", "whore", "slut", "bitch",
    }
    out = []
    for row, text in picked:
        low = text.lower()
        if any(b in low for b in blocked):
            continue
        if not (200 <= len(text) <= 900):
            continue
        out.append((row, text))
        if len(out) >= n:
            break
    return out


def preview(text, limit=90):
    flat = " ".join(text.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def main():
    load_before = os.getloadavg()
    manifest, all_docs, picked, picked_chars = load_corpus()
    fuzz = build_fuzz()

    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "packages": {
            "tokie": md.version("tokie"),
            "tokie_import_path": tokie.__file__,
            "tokie_wheel_sha256": sha256_file(HERE / "wheel" / "tokie-0.2.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"),
            "numpy": np.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "thread_env": {
            "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
            "RAYON_NUM_THREADS": os.environ["RAYON_NUM_THREADS"],
        },
        "dataset": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "jsonl": str(JSONL),
            "jsonl_sha256": manifest["jsonl_sha256"],
            "jsonl_bytes": JSONL.stat().st_size,
            "source": manifest["dataset"],
            "source_parquet": manifest["parquet"],
            "source_parquet_sha256": manifest["source_sha256"],
            "source_rows": manifest["source_rows"],
            "available_documents": len(all_docs),
            "available_row_range": [all_docs[0][0], all_docs[-1][0]],
            "provenance": "stride-15 full documents from the first parquet shard only; a prefix sample, not the whole dataset",
            "sampling_seed": SEED,
            "sampled_documents": len(picked),
            "sampled_chars": picked_chars,
            "sampled_bytes_utf8": sum(len(t.encode("utf-8")) for _, t in picked),
            "sampled_row_range": [picked[0][0], picked[-1][0]],
            "sampled_rows_sha256": sha256_bytes(",".join(str(r) for r, _ in picked).encode()),
        },
        "models": {},
        "cpu_load_before": list(load_before),
        "cpu_count": os.cpu_count(),
    }
    for name, cfg in MODELS.items():
        p = Path(cfg["path"])
        meta["models"][name] = {
            "repo": cfg["repo"],
            "revision": cfg["revision"],
            "algorithm": cfg["algo"],
            "tokenizer_json": str(p),
            "tokenizer_json_sha256": sha256_file(p),
            "tokenizer_json_bytes": p.stat().st_size,
        }

    texts = [t for _, t in picked] + EDGE_TEXTS + fuzz
    input_bytes = sum(len(t.encode("utf-8")) for t in texts)
    samples = pick_samples(picked)

    results = {"meta": meta, "validation": {}, "benchmark": {}, "profile": {}, "errors": []}
    results["samples"] = [
        {"row": row, "chars": len(text), "preview": preview(text)}
        for row, text in samples
    ]

    for name, cfg in MODELS.items():
        tok = tokie.Tokenizer.open(cfg["path"])
        hf = tokenizers.Tokenizer.from_file(cfg["path"])
        ok, stats, first, tok_ids, hf_ids = validate(
            tok, hf, texts, name, check_roundtrip=(name == "gpt2")
        )
        results["validation"][name] = {"ok": ok, "stats": stats, "first_mismatch": first}
        results["validation"][name]["samples"] = [
            {
                "row": row,
                "ids_first12": to_ids(tok.encode(text))[:12],
                "token_count": len(tok.encode(text)),
                "decoded_preview": preview(tok.decode(tok.encode(text), skip=False)),
            }
            for row, text in samples
        ]
        if not ok:
            results["errors"].append(f"{name}: validation mismatch, timing skipped")
            continue
        rounds, summary, total_tokens, decoded_bytes, cold = bench_model(
            name, cfg, texts, tok_ids, hf_ids, input_bytes
        )
        results["benchmark"][name] = {
            "texts": len(texts),
            "input_bytes_utf8": input_bytes,
            "total_tokens": total_tokens,
            "decoded_bytes_utf8": decoded_bytes,
            "cold_first_pass_s": cold,
            "rounds": rounds,
            "summary": summary,
        }
        results["profile"][name] = profile_model(name, cfg, texts, tok_ids, hf_ids)

    results["meta"]["cpu_load_after"] = list(os.getloadavg())
    results["meta"]["limitations"] = [
        "Ballpark throughput from 5 paired rounds, not a statistical or regression claim.",
        "Corpus is a seeded 179-doc slice of a stride-15 prefix of one OpenWebText parquet shard, not the whole dataset.",
        "Repeated rounds reuse the same texts; the cold first pass is recorded separately.",
        "HF timing is its public single-text API (Encoding object per call), not a kernel-only comparison.",
    ]

    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    write_summary(results)
    print(json.dumps({
        "validation": {k: v["stats"] for k, v in results["validation"].items()},
        "errors": results["errors"],
        "summary": {k: v["summary"] for k, v in results["benchmark"].items()},
    }, indent=2))


def write_summary(results):
    lines = ["# Published tokie 0.2.0 — encode/decode check + throughput", ""]
    m = results["meta"]
    lines += [
        f"- tokie `{m['packages']['tokie']}` from PyPI wheel, sha256 `{m['packages']['tokie_wheel_sha256'][:16]}…`",
        f"- HF `tokenizers` {m['packages']['tokenizers']}, numpy {m['packages']['numpy']}, Python {m['python']}",
        f"- One thread: `TOKENIZERS_PARALLELISM=false`, `RAYON_NUM_THREADS=1`",
        f"- Corpus: {m['dataset']['sampled_documents']} docs / {m['dataset']['sampled_chars']} chars, seed {m['dataset']['sampling_seed']}, rows {m['dataset']['sampled_row_range']}",
        f"- Provenance: {m['dataset']['provenance']}",
        f"- Load before/after: {m['cpu_load_before']} / {m['cpu_load_after']} on {m['cpu_count']} CPUs",
        "",
        "## Correctness",
        "",
        "| model | algo | texts | tokens | encode mismatches | decode mismatches | GPT-2 round-trip |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, v in results["validation"].items():
        s = v["stats"]
        rt = "n/a" if s.get("roundtrip_mismatches") is None else s["roundtrip_mismatches"]
        lines.append(
            f"| {name} | {MODELS[name]['algo']} | {s['texts']} | {s['tokens']} | {s['encode_mismatches']} | {s['decode_mismatches']} | {rt} |"
        )
    lines += ["", "## Throughput (median of 5 alternating paired rounds, 1 thread)", ""]
    lines += [
        "| model | impl | encode MiB/s | encode s | decode tok/s | decode MiB/s | decode s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, b in results["benchmark"].items():
        for impl in ("tokie", "hf"):
            s = b["summary"][impl]
            lines.append(
                f"| {name} | {impl} | {s['encode_mib_s_median']:.2f} | {s['encode_s_median']:.3f} | "
                f"{s['decode_tokens_s_median']:.0f} | {s['decode_mib_s_median']:.2f} | {s['decode_s_median']:.3f} |"
            )
    lines += ["", "Ranges per round are in `results.json`. These are ballpark numbers, not a formal regression claim.", ""]
    lines += [
        "## Notes",
        "",
        "- Both sides are timed through their public single-text Python API on the same `tokenizer.json`.",
        "- One warmup pass is excluded. The cold first pass is recorded in `results.json` under `cold_first_pass_s`.",
        "- The HF number includes building an `Encoding` object per call; tokie returns a NumPy array.",
        "- `cpu_per_wall` in `results.json` is ~1.00 for every timed loop, so neither side used hidden threads.",
        "- BERT/T5 normalization is lossy, so only ids and HF decoder output are compared for them. GPT-2 byte-level round-trips exactly.",
        "",
    ]
    lines += ["## Sample outputs", ""]
    for s in results.get("samples", []):
        lines.append(f"- row {s['row']} ({s['chars']} chars): {s['preview']}")
    lines.append("")
    for name, v in results["validation"].items():
        lines.append(f"**{name}**")
        for s in v.get("samples", []):
            lines.append(f"- row {s['row']}: {s['token_count']} ids, first12 `{s['ids_first12']}` → `{s['decoded_preview']}`")
        lines.append("")
    if results["errors"]:
        lines += ["## Errors", ""] + [f"- {e}" for e in results["errors"]] + [""]
    (HERE / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
