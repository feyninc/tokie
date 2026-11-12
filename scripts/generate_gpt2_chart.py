#!/usr/bin/env python3
"""Generate the six-library GPT-2 encode hero chart from the recorded JSON.

One GPT-2 row, six horizontal bars, sorted by median input MiB/s. Every number
comes from the results file; the script has no timings of its own. The loader
targets the recorded ``gpt2-hero.json`` schema: top-level ``task``
``encode_one_text_to_ids`` and ``model`` ``gpt2``; ``meta.corpus``
(documents/sample_utf8_bytes/validation_texts), ``meta.protocol`` (threads,
rounds, passes, warmup_passes), ``meta.libraries`` (distribution, version, api,
special_policy, wheel, installed native match), ``meta.versions``,
``meta.artifacts``, ``meta.host`` and optional harness/tokenizer/api_notes;
``validation``, ``rounds`` (nested ``libraries``), ``summary``, ``status``
``ok`` and empty ``errors``. Missing values fail; no bar is ever filled with a
default zero. The old ``benchmark`` matrix schema is rejected.

    python scripts/generate_gpt2_chart.py --results docs/benchmarks/uint32-decode/gpt2-hero.json
        --output-dir assets --table-out docs/benchmarks/uint32-decode/gpt2-hero-table.md
    python scripts/generate_gpt2_chart.py --results ... --check
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                     "svg.hashsalt": "tokie-gpt2-hero"})

SCHEMA_VERSION, MIB, TOL = 1, 1024.0 * 1024.0, 1e-7
TASK = "encode_one_text_to_ids"
TOKIE_LOCAL_SHA = "8786f91b1aed3acbac9d8090234e55f0400f829da6ea8a2817bcd460332a1d85"
GIGA_FOOTNOTE = ("** GigaToken does not support all tokenizer types; "
                 "WordPiece and Unigram are excluded.")
LIBS = ("tokie", "giga", "hf023", "hf1", "kitoken", "tiktoken")
FIELDS = ("seconds_median", "seconds_min", "seconds_max",
          "mib_s_median", "mib_s_min", "mib_s_max")
STEM = "benchmark_gpt2_main"
CANVAS, TEXT, MUTED, GRID = "#FFFFFF", "#111827", "#6B7280", "#E5E7EB"
COLORS = {"tokie": "#2563EB", "giga": "#D97706", "hf023": "#0F766E",
          "hf1": "#0F766E", "kitoken": "#7C3AED", "tiktoken": "#DB2777"}
DISPLAY = {"tokie": "Tokie", "giga": "GigaToken", "hf023": "HF tokenizers",
           "hf1": "HF tokenizers", "kitoken": "Kitoken", "tiktoken": "tiktoken"}


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pos(value):
    return _num(value) and float(value) > 0


def _int(value):
    return ((isinstance(value, int) and not isinstance(value, bool))
            or (isinstance(value, float) and value.is_integer()))


def _close(recorded, expected):
    return abs(float(recorded) - float(expected)) <= TOL * max(
        abs(float(recorded)), abs(float(expected)), 1e-12)


def _version(meta, lib):
    versions = meta.get("versions")
    if isinstance(versions, dict) and isinstance(versions.get(lib), str) and versions[lib].strip():
        return versions[lib].strip()
    libraries = meta.get("libraries")
    if isinstance(libraries, dict) and isinstance(libraries.get(lib), dict):
        value = libraries[lib].get("version")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _library(meta, lib):
    libraries = meta.get("libraries")
    return libraries.get(lib) if isinstance(libraries, dict) and isinstance(libraries.get(lib), dict) else {}


def _artifact(meta, lib):
    info = _library(meta, lib)
    wheel = info.get("wheel") if isinstance(info.get("wheel"), dict) else {}
    artifacts = meta.get("artifacts")
    record = artifacts.get(lib) if isinstance(artifacts, dict) and isinstance(artifacts.get(lib), dict) else {}
    return {"version": _version(meta, lib), "distribution": info.get("distribution"),
            "module_version": info.get("module_version"), "api": info.get("api"),
            "special_policy": info.get("special_policy"),
            "installed_native": info.get("installed_native"),
            "installed_native_matches_wheel": info.get("installed_native_matches_wheel"),
            "wheel_file": wheel.get("file") or record.get("wheel_file"),
            "wheel_sha256": wheel.get("sha256") or record.get("wheel_sha256"),
            "wheel_members": wheel.get("members") or record.get("wheel_members_sha256")}


def _apis(meta):
    return {lib: _artifact(meta, lib)["api"] for lib in LIBS
            if isinstance(_artifact(meta, lib)["api"], str) and _artifact(meta, lib)["api"].strip()}


def _tokie_local(meta):
    art = _artifact(meta, "tokie")
    command = meta.get("command")
    text = " ".join(str(part) for part in command) if isinstance(command, list) else str(command or "")
    return (art.get("wheel_sha256") == TOKIE_LOCAL_SHA
            and art.get("installed_native_matches_wheel") is True
            and "--tokie-wheel" in text)


def _harness_commit(meta):
    harness = meta.get("harness")
    commit = harness.get("git_head") if isinstance(harness, dict) else None
    return commit.strip() if isinstance(commit, str) and commit.strip() else None


def _host(meta):
    host = dict(meta.get("host")) if isinstance(meta.get("host"), dict) else {}
    for key in ("cpu_model", "cpu_count", "platform", "load_before"):
        if key in meta:
            host.setdefault(key, meta[key])
    if "cpu_count" in host:
        host.setdefault("cpu_count_logical", host["cpu_count"])
    return host


def _corpus_meta(meta):
    corpus = meta.get("corpus")
    return corpus if isinstance(corpus, dict) else {}


def _corpus(meta):
    corpus = _corpus_meta(meta)
    value = corpus.get("name") or corpus.get("corpus") or corpus.get("dataset")
    if isinstance(value, str) and value.strip():
        return value.strip()
    path = str(corpus.get("jsonl") or "").lower()
    return "OpenWebText" if ("owt" in path or "openwebtext" in path) else "OpenWebText"


def _input_bytes(meta):
    return _corpus_meta(meta).get("sample_utf8_bytes")


def _documents(meta):
    return _corpus_meta(meta).get("documents")


def _validation_texts(meta):
    return _corpus_meta(meta).get("validation_texts")


def _protocol(meta):
    protocol = meta.get("protocol")
    return protocol if isinstance(protocol, dict) else {}


def _threads(meta):
    return _protocol(meta).get("threads")


def _passes(meta):
    return _protocol(meta).get("passes")


def _protocol_rounds(meta):
    return _protocol(meta).get("rounds")


def _walk_numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_numbers(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_numbers(item, f"{prefix}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)


def _expected(seconds, input_bytes, passes):
    median, low, high = statistics.median(seconds), min(seconds), max(seconds)
    return {"seconds_median": median, "seconds_min": low, "seconds_max": high,
            "mib_s_median": input_bytes * passes / median / MIB,
            "mib_s_min": input_bytes * passes / high / MIB,
            "mib_s_max": input_bytes * passes / low / MIB}


def validate_data(path, data):
    """Return human-readable problems; an empty list means the file is valid."""
    errors, bad = [], lambda message: errors.append(f"{path}: {message}")
    if not isinstance(data, dict):
        return [f"{path}: top level is not an object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        bad(f"schema_version must be {SCHEMA_VERSION}")
    if data.get("task") != TASK:
        bad(f"task must be {TASK!r}")
    if data.get("model") != "gpt2":
        bad("model must be 'gpt2'")
    if "benchmark" in data and "rounds" not in data:
        bad("legacy benchmark schema is not supported; pass the new gpt2-hero.json")
    for where, value in _walk_numbers(data):
        if not math.isfinite(value):
            bad(f"{where} is not finite")

    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    if not meta:
        bad("meta must be a non-empty object")
    if not _host(meta):
        bad("meta.host or top-level host fields are required")
    corpus, protocol = _corpus_meta(meta), _protocol(meta)
    if not corpus:
        bad("meta.corpus must be an object")
    for key in ("documents", "sample_utf8_bytes", "validation_texts"):
        if not _int(corpus.get(key)) or corpus[key] <= 0:
            bad(f"meta.corpus.{key} must be a positive integer")
    for key in ("threads", "rounds", "passes", "warmup_passes"):
        if not _int(protocol.get(key)) or protocol[key] <= 0:
            bad(f"meta.protocol.{key} must be a positive integer")

    libraries = meta.get("libraries") if isinstance(meta.get("libraries"), dict) else {}
    versions = meta.get("versions") if isinstance(meta.get("versions"), dict) else {}
    artifacts = meta.get("artifacts") if isinstance(meta.get("artifacts"), dict) else {}
    if set(libraries) != set(LIBS):
        bad(f"meta.libraries must contain exactly {list(LIBS)}")
    if set(versions) != set(LIBS):
        bad(f"meta.versions must contain exactly {list(LIBS)}")
    if set(artifacts) != set(LIBS):
        bad(f"meta.artifacts must contain exactly {list(LIBS)}")
    for lib in LIBS:
        info = libraries.get(lib)
        if not isinstance(info, dict):
            bad(f"meta.libraries.{lib} is missing")
            continue
        for field in ("distribution", "version", "api", "special_policy"):
            if not isinstance(info.get(field), str) or not info[field].strip():
                bad(f"meta.libraries.{lib}.{field} is required")
        if info.get("installed_native_matches_wheel") is not True:
            bad(f"meta.libraries.{lib}.installed_native_matches_wheel must be true")
        wheel = info.get("wheel")
        if not (isinstance(wheel, dict) and isinstance(wheel.get("file"), str)
                and isinstance(wheel.get("sha256"), str) and wheel["file"].strip() and wheel["sha256"].strip()):
            bad(f"meta.libraries.{lib}.wheel needs file and sha256")
        record = artifacts.get(lib)
        if not isinstance(record, dict) or not isinstance(record.get("wheel_sha256"), str):
            bad(f"meta.artifacts.{lib}.wheel_sha256 is required")
        elif isinstance(wheel, dict) and record.get("wheel_sha256") != wheel.get("sha256"):
            bad(f"meta.artifacts.{lib}.wheel_sha256 disagrees with meta.libraries.{lib}.wheel")
        if versions.get(lib) != info.get("version"):
            bad(f"meta.versions.{lib} disagrees with meta.libraries.{lib}.version")
    if not _tokie_local(meta):
        bad("tokie local wheel evidence is missing; expected --tokie-wheel and sha256 8786f91... "
            "with installed native match")
    hf1_version, hf023_version = _version(meta, "hf1") or "", _version(meta, "hf023") or ""
    if not re.search(r"(rc|alpha|beta|dev|pre)", hf1_version, re.I):
        bad("meta.versions.hf1 must be a prerelease such as 1.0.0rc4, not stable 1.0")
    if re.search(r"(rc|alpha|beta|dev|pre)", hf023_version, re.I):
        bad("meta.versions.hf023 must be a stable release")
    apis = _apis(meta)
    if "compile()" not in apis.get("hf1", "") or "encode_fast" in apis.get("hf1", ""):
        bad("meta.libraries.hf1.api must record the compile().encode(...).ids path")
    if "encode_specials=True" not in apis.get("kitoken", ""):
        bad("meta.libraries.kitoken.api must record encode_specials=True")
    if "allowed_special" not in apis.get("tiktoken", ""):
        bad("meta.libraries.tiktoken.api must record allowed_special")
    for lib in LIBS:
        if not isinstance(apis.get(lib), str) or not apis[lib]:
            bad(f"meta.libraries.{lib}.api is required")
    vocab = meta.get("tiktoken_vocab_check")
    if isinstance(vocab, dict):
        if vocab.get("rank_mismatches") != 0 or vocab.get("special_mismatches") != 0:
            bad("meta.tiktoken_vocab_check records a rank/special mismatch")

    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    for name, section in (("validation", validation), ("summary", summary)):
        if not isinstance(data.get(name), dict):
            bad(f"{name} must be an object")
            continue
        missing, extra = [lib for lib in LIBS if lib not in section], sorted(set(section) - set(LIBS))
        if missing:
            message = f"{name} must contain all six libraries; missing {', '.join(missing)}"
            if "kitoken" in missing:
                message += (" (Kitoken is required; excluding it needs an explicit recorded "
                            "reason and is a parent decision)")
            bad(message)
        if extra:
            bad(f"{name} has unknown libraries: {', '.join(extra)}")
    expected_texts = _validation_texts(meta)
    checksums = set()
    for lib in LIBS:
        entry = validation.get(lib)
        if not isinstance(entry, dict):
            bad(f"validation.{lib} is missing")
            continue
        if entry.get("ok") is not True:
            bad(f"validation.{lib}.ok must be true")
        if not _int(entry.get("texts")) or entry["texts"] != expected_texts:
            bad(f"validation.{lib}.texts must be {expected_texts}")
        if entry.get("mismatches") != 0:
            bad(f"validation.{lib}.mismatches must be 0")
        if entry.get("first_mismatch") is not None:
            bad(f"validation.{lib}.first_mismatch must be null")
        checksum = entry.get("checksum")
        if not isinstance(checksum, str) or not checksum.strip():
            bad(f"validation.{lib}.checksum is required")
        else:
            checksums.add(checksum)
    if len(checksums) > 1:
        bad("validation checksums differ across libraries")
    if data.get("status") != "ok":
        bad("status must be 'ok'")
    if data.get("errors") not in (None, []):
        bad("top-level errors is not empty")

    rounds = data.get("rounds") if isinstance(data.get("rounds"), list) else []
    seconds = {lib: [] for lib in LIBS}
    counts = {lib: set() for lib in LIBS}
    orders = set()
    if len(rounds) < 2:
        bad("rounds must be a list with at least 2 entries")
    if _int(_protocol_rounds(meta)) and _protocol_rounds(meta) != len(rounds):
        bad("meta.protocol.rounds disagrees with the number of round entries")
    input_bytes, passes = _input_bytes(meta), _passes(meta)
    for index, round_entry in enumerate(rounds, 1):
        if not isinstance(round_entry, dict):
            bad(f"rounds[{index}] must be an object")
            continue
        order = round_entry.get("order")
        if not (isinstance(order, list) and len(order) == len(LIBS) and set(order) == set(LIBS)):
            bad(f"rounds[{index}].order must list each of the six libraries once")
        else:
            orders.add(tuple(order))
        entries = round_entry.get("libraries")
        if not isinstance(entries, dict) or set(entries) != set(LIBS):
            bad(f"rounds[{index}].libraries must contain exactly the six libraries")
            continue
        for lib in LIBS:
            entry = entries.get(lib)
            if not isinstance(entry, dict):
                bad(f"rounds[{index}].libraries.{lib} is missing")
                continue
            if not _pos(entry.get("seconds")):
                bad(f"rounds[{index}].libraries.{lib}.seconds must be positive and finite")
                continue
            seconds[lib].append(float(entry["seconds"]))
            if not _pos(entry.get("count")):
                bad(f"rounds[{index}].libraries.{lib}.count must be positive and finite")
            else:
                counts[lib].add(float(entry["count"]))
            if not isinstance(entry.get("checksum"), str) or not entry["checksum"].strip():
                bad(f"rounds[{index}].libraries.{lib}.checksum is required")
            if not _pos(entry.get("mib_s")):
                bad(f"rounds[{index}].libraries.{lib}.mib_s must be positive and finite")
            elif _int(input_bytes) and _int(passes):
                expected = input_bytes * passes / float(entry["seconds"]) / MIB
                if not _close(entry["mib_s"], expected):
                    bad(f"rounds[{index}].libraries.{lib}.mib_s does not match seconds")
    if len(rounds) >= 2 and len(orders) < 2:
        bad("round order is not interleaved/counterbalanced")
    for lib in LIBS:
        if len(counts[lib]) > 1:
            bad(f"rounds.libraries.{lib}.count is not stable across rounds")
    for lib in LIBS:
        entry = summary.get(lib)
        if not isinstance(entry, dict):
            bad(f"summary.{lib} is missing")
            continue
        if any(not _pos(entry.get(field)) for field in FIELDS):
            bad(f"summary.{lib} must have positive finite {FIELDS}")
            continue
        if not (entry["seconds_min"] <= entry["seconds_median"] <= entry["seconds_max"]):
            bad(f"summary.{lib} seconds min/median/max out of order")
        if not (entry["mib_s_min"] <= entry["mib_s_median"] <= entry["mib_s_max"]):
            bad(f"summary.{lib} MiB/s min/median/max out of order")
        if "rounds" in entry and entry["rounds"] != len(rounds):
            bad(f"summary.{lib}.rounds disagrees with rounds")
        if not seconds[lib] or not _int(input_bytes) or not _int(passes):
            continue
        for field, value in _expected(seconds[lib], input_bytes, passes).items():
            if not _close(entry[field], value):
                bad(f"summary.{lib}.{field}={entry[field]:.6g} does not match rounds ({value:.6g})")
        if "tokens_per_pass" in entry and counts[lib]:
            if not _close(entry["tokens_per_pass"], next(iter(counts[lib])) / passes):
                bad(f"summary.{lib}.tokens_per_pass disagrees with count/passes")
    return errors


def load_results(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    errors = validate_data(path, data)
    if errors:
        raise ValueError("\n".join(errors))
    meta = data["meta"]
    return {"path": str(path), "data": data, "meta": meta, "host": _host(meta),
            "corpus_meta": _corpus_meta(meta), "tokenizer": meta.get("tokenizer")
            if isinstance(meta.get("tokenizer"), dict) else {},
            "protocol": _protocol(meta), "threads": _threads(meta),
            "input_bytes": _input_bytes(meta), "documents": _documents(meta),
            "validation_texts": _validation_texts(meta), "corpus": _corpus(meta),
            "passes": _passes(meta), "rounds": len(data["rounds"]),
            "apis": _apis(meta), "harness": meta.get("harness")
            if isinstance(meta.get("harness"), dict) else {},
            "harness_commit": _harness_commit(meta), "tokie_local": _tokie_local(meta),
            "api_notes": meta.get("api_notes") if isinstance(meta.get("api_notes"), dict) else {},
            "tiktoken_vocab_check": meta.get("tiktoken_vocab_check")
            if isinstance(meta.get("tiktoken_vocab_check"), dict) else {},
            "claim_scope": meta.get("claim_scope"), "ranking_note": data.get("ranking_note"),
            "generated_utc": data.get("generated_utc"),
            "artifacts": {lib: _artifact(meta, lib) for lib in LIBS},
            "versions": {lib: _version(meta, lib) for lib in LIBS}}


def _series(dataset):
    rows = [{"lib": lib, "median": float(dataset["data"]["summary"][lib]["mib_s_median"]),
             "low": float(dataset["data"]["summary"][lib]["mib_s_min"]),
             "high": float(dataset["data"]["summary"][lib]["mib_s_max"]),
             "seconds": float(dataset["data"]["summary"][lib]["seconds_median"])} for lib in LIBS]
    return sorted(rows, key=lambda row: row["median"], reverse=True)


def _value(value):
    return f"{value:,.2f}"


def _value_exact(value):
    return f"{value:,.3f}"


def _library_label(lib, dataset):
    version = dataset["versions"][lib]
    if lib == "tokie":
        return f"{DISPLAY[lib]} {version} (local)"
    if lib == "giga":
        return f"** {DISPLAY[lib]} {version}"
    return f"{DISPLAY[lib]} {version}"


def _markdown_label(lib, dataset):
    label = _library_label(lib, dataset)
    return label.replace("**", r"\*\*", 1) if lib == "giga" else label


def _short_label(lib, dataset):
    return f"{DISPLAY[lib]} {dataset['versions'][lib]}"


def _passes_text(dataset):
    return str(dataset["passes"])


def _subtitle(dataset):
    threads = dataset["threads"]
    thread_text = "1 thread" if threads == 1 else f"{threads} threads"
    return (f"{thread_text} · same {dataset['documents']:,} {dataset['corpus']} documents "
            f"· warm cache · medians and {dataset['rounds']}-round min–max")


def _short_cpu(dataset):
    cpu = dataset["host"].get("cpu_model") or "CPU not recorded"
    return cpu.replace("AMD ", "").split(" w/")[0]


def _footer_lines(dataset):
    hf1 = dataset["versions"]["hf1"]
    sha = dataset["artifacts"]["tokie"].get("wheel_sha256") or ""
    platform = str(dataset["host"].get("platform") or "").split("-")[0] or "platform not recorded"
    return [
        f"Public single-text encode → IDs · {dataset['passes']} passes/round · "
        f"{dataset['input_bytes'] / MIB:.3f} MiB input · observed ranges, not confidence intervals.",
        f"{GIGA_FOOTNOTE} HF {hf1} is a prerelease.",
        f"Tokie: local unreleased wheel {sha[:7]}… · competitors: PyPI wheels · "
        f"{_short_cpu(dataset)} · {platform}.",
        "APIs, special-token policies, artifacts and reproduction: "
        "docs/benchmarks/uint32-decode/gpt2-hero-table.md",
    ]


def _footer(fig, dataset):
    lines = _footer_lines(dataset)
    for index, line in enumerate(lines):
        fig.text(0.012, 0.012 + 0.027 * (len(lines) - 1 - index), line,
                 fontsize=8.6, color=MUTED, ha="left", va="bottom")


def _draw(dataset):
    rows = _series(dataset)
    fig = plt.figure(figsize=(12, 5.5))
    fig.patch.set_facecolor(CANVAS)
    ax = fig.add_axes([0.235, 0.24, 0.735, 0.60])
    ax.set_facecolor(CANVAS)
    for y, row in enumerate(rows):
        color = COLORS[row["lib"]]
        ax.barh(y, row["median"], height=0.6, color=color, edgecolor="none", zorder=2)
        ax.errorbar([row["median"]], [y], xerr=[[row["median"] - row["low"]], [row["high"] - row["median"]]],
                    fmt="none", ecolor=color, elinewidth=1.4, capsize=3.0, capthick=1.2, alpha=0.9, zorder=3)
        ax.annotate(_value(row["median"]), xy=(row["median"], y), xytext=(5, 0), textcoords="offset points",
                    va="center", ha="left", color=color, fontsize=10, fontweight="bold", zorder=4, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([_library_label(row["lib"], dataset) for row in rows], fontsize=10.5, color=TEXT)
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(row["high"] for row in rows) * 1.18)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
    ax.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.95)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=2.5)
    ax.tick_params(axis="y", colors=TEXT, length=0, pad=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.spines["bottom"].set_linewidth(0.7)
    ax.set_xlabel("Input MiB/s · higher is better", fontsize=10, color=MUTED, labelpad=6)
    fig.text(0.03, 0.955, "GPT-2 encoding", fontsize=19, color=TEXT, ha="left", va="top", fontweight="bold")
    fig.text(0.03, 0.905, _subtitle(dataset), fontsize=10.5, color=MUTED, ha="left", va="top")
    _footer(fig, dataset)
    return fig


def _normalize_svg(path):
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    file.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def render(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, written = _draw(dataset), []
    for extension in ("png", "svg"):
        path = os.path.join(output_dir, f"{STEM}.{extension}")
        if extension == "svg":
            fig.savefig(path, dpi=200, bbox_inches=None, facecolor=fig.get_facecolor(), metadata={"Date": None})
            _normalize_svg(path)
        else:
            fig.savefig(path, dpi=200, bbox_inches=None, facecolor=fig.get_facecolor())
        written.append(path)
    plt.close(fig)
    return written


def write_table(dataset, path):
    rows, validation = _series(dataset), dataset["data"]["validation"]
    corpus, protocol, harness = dataset["corpus_meta"], dataset["protocol"], dataset["harness"]
    tokenizer = dataset["tokenizer"]
    mib = dataset["input_bytes"] / MIB
    lines = ["# GPT-2 encoding — six-library hero table", "",
             f"Generated by `scripts/generate_gpt2_chart.py` from `{dataset['path']}`"
             + (f" at {dataset['generated_utc']}" if dataset["generated_utc"] else "") + ". "
             "This table is authoritative for the numbers in `assets/benchmark_gpt2_main.png` and `.svg`.", "",
             f"**Task:** {TASK} · **Model:** GPT-2 · **Threads:** {dataset['threads']} · "
             f"**Sample:** {dataset['documents']:,} {dataset['corpus']} documents, "
             f"{dataset['input_bytes']:,} UTF-8 bytes ({mib:.3f} MiB) · "
             f"**Rounds:** {dataset['rounds']} · **Passes per round:** {dataset['passes']} · "
             f"**Warmup passes:** {protocol.get('warmup_passes', 'not recorded')}", "",
             "Bars are median input MiB/s; whiskers are observed min–max, not confidence intervals. "
             "Throughput uses MiB (1 MiB = 1,048,576 bytes), not decimal MB. Higher is better. "
             "One host and one corpus slice; no universal ranking or winner claim.", "",
             "| Library | Version | Input MiB/s (median) | Min–max | Median seconds | Validation |",
             "| --- | --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        lib, entry = row["lib"], validation[row["lib"]]
        lines.append(f"| {_markdown_label(lib, dataset)} | {dataset['versions'][lib]} | "
                     f"{_value_exact(row['median'])} | {_value_exact(row['low'])}–{_value_exact(row['high'])} | "
                     f"{row['seconds']:.6f} | {entry.get('texts')} texts, {entry.get('mismatches')} mismatches, "
                     f"{'ok' if entry.get('ok') else 'check'} |")
    lines += ["", GIGA_FOOTNOTE.replace("**", r"\*\*", 1), "",
              "## Validation", "", "| Library | Texts | Mismatches | Checksum | Status |",
              "| --- | ---: | ---: | --- | --- |"]
    lines += [f"| {_markdown_label(lib, dataset)} | {validation[lib].get('texts')} | "
              f"{validation[lib].get('mismatches')} | `{validation[lib].get('checksum')}` | "
              f"{'ok' if validation[lib].get('ok') else 'check'} |" for lib in LIBS]
    lines += ["", "## Ranking", "", dataset["ranking_note"] or "No ranking note recorded.", "",
              "## Provenance and calls", ""]
    for lib in LIBS:
        art = dataset["artifacts"][lib]
        label = _markdown_label(lib, dataset)
        line = (f"- **{label}** — distribution `{art.get('distribution')}`; "
                f"version `{dataset['versions'][lib]}`") if lib != "giga" else (
                f"- {label} — distribution `{art.get('distribution')}`; "
                f"version `{dataset['versions'][lib]}`")
        if art.get("module_version"):
            line += f" (module `{art['module_version']}`)"
        line += f"; wheel `{art.get('wheel_file')}` sha256 `{art.get('wheel_sha256')}`"
        line += f"; installed native matches wheel: `{art.get('installed_native_matches_wheel')}`"
        line += f"; API: `{art.get('api')}`; policy: {art.get('special_policy')}"
        lines.append(line)
    lines += ["",
              f"- Tokie local provenance: `meta.command` includes `--tokie-wheel`; wheel sha256 is "
              f"`{TOKIE_LOCAL_SHA}`; installed native matches the wheel member. Tokie is a local "
              "unreleased wheel, not the PyPI package; the competitors are PyPI wheels.",
              f"- Harness: `{harness.get('script', 'not recorded')}` sha256 "
              f"`{harness.get('sha256', 'not recorded')}`, git_head "
              f"`{harness.get('git_head', 'not recorded')}`; exact commands: "
              "`docs/benchmarks/uint32-decode/gpt2-hero-commands.txt`.",
              f"- Corpus: `{corpus.get('jsonl', 'not recorded')}` sha256 "
              f"`{corpus.get('jsonl_sha256', 'not recorded')}`, seed {corpus.get('seed', 'not recorded')}, "
              f"{corpus.get('documents', '?')} docs, {corpus.get('sample_utf8_bytes', '?')} sample UTF-8 bytes, "
              f"{corpus.get('validation_texts', '?')} validation texts.",
              f"- Tokenizer: `{tokenizer.get('path', 'not recorded')}` sha256 "
              f"`{tokenizer.get('sha256', 'not recorded')}`, {tokenizer.get('bytes', '?')} bytes.",
              "", "## API notes", ""]
    for lib in LIBS:
        note = dataset["api_notes"].get(lib)
        if note:
            label = _markdown_label(lib, dataset) if lib == "giga" else f"**{_short_label(lib, dataset)}**"
            lines.append(f"- {label}: {note}")
    vocab = dataset["tiktoken_vocab_check"]
    if vocab:
        lines += ["", "## tiktoken vocabulary check", "",
                  f"- HF vocab entries: {vocab.get('hf_vocab_entries', '?')}; "
                  f"non-special: {vocab.get('hf_vocab_non_special_entries', '?')}; "
                  f"tiktoken vocab: {vocab.get('tiktoken_n_vocab', '?')}.",
                  f"- Ranks checked: {vocab.get('ranks_checked', '?')}; rank mismatches: "
                  f"{vocab.get('rank_mismatches', '?')}; special mismatches: "
                  f"{vocab.get('special_mismatches', '?')}."]
    lines += ["", "## Host", "",
              f"- CPU: {dataset['host'].get('cpu_model') or 'not recorded'}",
              f"- Logical CPUs: {dataset['host'].get('cpu_count_logical') or 'not recorded'}",
              f"- Platform: {dataset['host'].get('platform') or 'not recorded'}",
              f"- Mean current frequency snapshot: {dataset['host'].get('mean_cur_freq_mhz', 'not recorded')} MHz",
              f"- Load before: {dataset['host'].get('load_before', 'not recorded')}",
              "", "## Method", "",
              f"- Threads: {protocol.get('threads', '?')}; rounds: {protocol.get('rounds', '?')}; "
              f"passes per round: {protocol.get('passes', '?')}; warmup passes: {protocol.get('warmup_passes', '?')}.",
              f"- Round order: {protocol.get('round_order', 'not recorded')}.",
              f"- Timed region: {protocol.get('timed_region', 'not recorded')}.",
              f"- Outside the clock: {protocol.get('outside_clock', 'not recorded')}.",
              "- Throughput is input MiB/s (1 MiB = 1,048,576 bytes), not decimal MB.",
              "- Ranges are observed min–max over the recorded rounds, not confidence intervals.",
              "", "## Claim scope", "", dataset["claim_scope"] or "No claim scope recorded."]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the six-library GPT-2 encode hero chart from the recorded JSON.",
        epilog="--check validates the results file and writes no figures or tables.")
    parser.add_argument("--results", required=True, metavar="PATH", help="gpt2-hero.json results file")
    parser.add_argument("--output-dir", default="assets", metavar="DIR", help="figure directory (default: assets)")
    parser.add_argument("--table-out", default="docs/benchmarks/uint32-decode/gpt2-hero-table.md",
                        metavar="PATH", help="markdown table path")
    parser.add_argument("--check", action="store_true", help="validate only; write no figures or tables")
    args = parser.parse_args(argv)
    try:
        dataset = load_results(args.results)
    except ValueError as exc:
        for line in str(exc).splitlines():
            print(f"error: {line}", file=sys.stderr)
        return 2
    print(f"validated {dataset['path']}: {len(LIBS)} libraries, "
          f"{dataset['rounds']} rounds, {dataset['validation_texts']} validation texts")
    if args.check:
        print("check only: no figures or tables written")
        return 0
    written = render(dataset, args.output_dir)
    if args.table_out:
        written.append(write_table(dataset, args.table_out))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
