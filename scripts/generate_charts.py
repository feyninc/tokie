#!/usr/bin/env python3
"""Generate benchmark charts for the tokie README from recorded JSON results.

The chart style is the Flexoki scheme, horizontal bars, and "Nx slower"
annotations. Every bar comes from a results file passed on the command line;
this script carries no timings of its own. A results file that is missing,
inconsistent, or fails its own validation stops generation instead of drawing
a zero bar.

Usage (results are the committed copies under docs/benchmarks/published-0.2/):
    python scripts/generate_charts.py \
        --hf-results docs/benchmarks/published-0.2/hf-results.json \
        --giga-results docs/benchmarks/published-0.2/results-1t.json \
        --giga-results docs/benchmarks/published-0.2/results-8t.json \
        --giga-large-results docs/benchmarks/published-0.2/results-8t-large.json \
        --decode-results docs/benchmarks/published-0.2/results-decode-inputs.json
"""

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Theme — Flexoki color scheme (https://stephango.com/flexoki)
# ---------------------------------------------------------------------------
BG = "#FFFCF0"           # Flexoki paper
TOKIE_COLOR = "#66800B"  # Flexoki green
GRAY_COLOR = "#B7B5AC"   # Flexoki tx-3 (muted gray)
GOLD_COLOR = "#AD8301"   # Flexoki yellow (kitoken)
TEXT_COLOR = "#100F0F"    # Flexoki tx
CAPTION_COLOR = "#878580" # Flexoki tx-2
DPI = 200
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
MIB = 1024.0 * 1024.0

# Display names for the model keys used by the published benchmarks. An
# unknown key is shown verbatim, so a new model needs no code change.
MODEL_LABELS = {
    "gpt2": "GPT-2",
    "bert-base-uncased": "BERT-base",
    "t5-small": "T5-small",
    "qwen2": "Qwen2",
    "cl100k": "cl100k",
}

# Giga result operations: title stem, chart stem, and the byte count its
# MiB/s divides by. Decode speed depends on the id container the binding
# receives, so the two decode variants are charted separately.
GIGA_OPS = {
    "encode": ("Text encoding", "single", "input"),
    "decode": ("Decoding NumPy uint32 ids", "decode", "decoded"),
    "decode_list": ("Decoding Python list ids", "decode_list", "decoded"),
    "batch": ("Batch encoding", "batch", "input"),
}


def _style(ax, title):
    ax.set_facecolor(BG)
    ax.figure.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TEXT_COLOR)
    ax.spines["bottom"].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.set_title(title, color=TEXT_COLOR, fontsize=16, fontweight="bold", pad=12)


def _save(fig, name, out_dir):
    for ext in ("png", "svg"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved {path}")
    plt.close(fig)


def _horiz_chart(title, bars, xlabel, caption, fname, out_dir, figsize=None):
    """Horizontal bar chart. bars: [(label, value, color, annotation)]"""
    n = len(bars)
    if figsize is None:
        figsize = (10, max(2.5, n * 0.85 + 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    _style(ax, title)

    labels = [b[0] for b in bars]
    values = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    annots = [b[3] for b in bars]

    y_pos = list(range(n))
    ax.barh(y_pos, values, height=0.55, color=colors, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.invert_yaxis()

    max_val = max(values)
    for i, (val, annot) in enumerate(zip(values, annots)):
        if annot:
            ax.text(val + max_val * 0.02, i, annot,
                    va="center", ha="left", color=TEXT_COLOR, fontsize=11)

    if caption:
        ax.text(1.0, -0.08, caption, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontstyle="italic",
                color=CAPTION_COLOR)

    _save(fig, fname, out_dir)


# ---------------------------------------------------------------------------
# Results loading and validation
# ---------------------------------------------------------------------------


def _die(message):
    raise SystemExit(f"generate_charts: {message}")


def _load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        _die(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        _die(f"{path} is not valid JSON: {exc}")


def _mapping(value, where):
    if not isinstance(value, dict):
        _die(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _sequence(value, where):
    if not isinstance(value, list) or not value:
        _die(f"{where}: expected a non-empty list")
    return value


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        _die(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _positive(value, where):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _die(f"{where}: expected a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        _die(f"{where}: expected a finite positive number, got {value!r}")
    return float(value)


def _count(value, where):
    # Thread counts are recorded as environment strings; counts are integers.
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _die(f"{where}: expected a positive integer, got {value!r}")
    return value


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _close(actual, expected, where, rel=1e-9):
    if abs(actual - expected) > rel * abs(expected):
        _die(f"{where}: {actual!r} does not match the recorded data ({expected!r})")
    return actual


def _slower(ratio):
    return f"{ratio:.0f}x slower" if ratio >= 10 else f"{ratio:.1f}x slower"


def _version(packages, name, where):
    entry = _mapping(packages.get(name), where)
    return _text(entry.get("version"), f"{where}.version")


def load_hf_results(path):
    """Validate the published tokie-vs-Hugging-Face results JSON.

    The timed set mixes OpenWebText documents with fixed edge and fuzz cases,
    so the caption names both counts instead of calling every input a document.
    """
    doc = _mapping(_load_json(path), path)
    meta = _mapping(doc.get("meta"), "meta")
    benchmark = _mapping(doc.get("benchmark"), "benchmark")
    validation = _mapping(doc.get("validation"), "validation")

    packages = _mapping(meta.get("packages"), "meta.packages")
    tokie_version = _text(packages.get("tokie"), "meta.packages.tokie")
    hf_version = _text(packages.get("tokenizers"), "meta.packages.tokenizers")

    thread_env = _mapping(meta.get("thread_env"), "meta.thread_env")
    threads = _count(thread_env.get("RAYON_NUM_THREADS"), "meta.thread_env.RAYON_NUM_THREADS")

    dataset = _mapping(meta.get("dataset"), "meta.dataset")
    documents = _count(dataset.get("sampled_documents"), "meta.dataset.sampled_documents")

    models = []
    texts = mib = None
    shared_bytes = None
    for model, bench in benchmark.items():
        where = f"benchmark.{model}"
        bench = _mapping(bench, where)

        model_texts = _count(bench.get("texts"), f"{where}.texts")
        input_bytes = _count(bench.get("input_bytes_utf8"), f"{where}.input_bytes_utf8")
        if texts is None:
            texts, mib, shared_bytes = model_texts, input_bytes / MIB, input_bytes
        elif (model_texts, input_bytes) != (texts, shared_bytes):
            _die(f"{where}: input size differs from the other models")

        check = _mapping(validation.get(model), f"validation.{model}")
        if check.get("ok") is not True:
            _die(f"validation.{model}.ok is not true")
        stats = _mapping(check.get("stats"), f"validation.{model}.stats")
        if stats.get("encode_mismatches") != 0:
            _die(f"validation.{model}: encode_mismatches is not 0")
        if stats.get("decode_mismatches") != 0:
            _die(f"validation.{model}: decode_mismatches is not 0")
        if check.get("first_mismatch") is not None:
            _die(f"validation.{model}: first_mismatch is set")
        if stats.get("texts") != model_texts:
            _die(f"validation.{model}.stats.texts does not match {where}.texts")

        summary = _mapping(bench.get("summary"), f"{where}.summary")
        mib_in = input_bytes / MIB
        measured = {}
        for side in ("tokie", "hf"):
            side_where = f"{where}.summary.{side}"
            side_summary = _mapping(summary.get(side), side_where)
            mib_s = _positive(side_summary.get("encode_mib_s_median"),
                              f"{side_where}.encode_mib_s_median")
            encode_s = _positive(side_summary.get("encode_s_median"),
                                f"{side_where}.encode_s_median")
            if abs(mib_in / encode_s - mib_s) > 0.005 * mib_s:
                _die(f"{side_where}: encode_mib_s_median contradicts input size and encode_s_median")
            measured[side] = mib_s

        models.append((model, measured["tokie"], measured["hf"]))

    if not models:
        _die(f"{path}: benchmark object has no models")
    if documents >= texts:
        _die(f"{path}: sampled_documents must be smaller than the timed text count")

    caption = (
        f"tokie {tokie_version} \u00b7 HF tokenizers {hf_version} \u00b7 "
        f"{threads} thread \u00b7 warm cache\n"
        f"{texts} texts: {documents} OpenWebText documents + {texts - documents} "
        f"edge/fuzz cases \u00b7 {mib:.1f} MiB"
    )
    return caption, models


def _check_giga_validation(validation, status, model):
    where = f"validation.{model}"
    if status.get(model) != "ok":
        _die(f"status.{model}: expected 'ok', got {status.get(model)!r}")
    check = _mapping(validation.get(model), where)
    if check.get("ok") is not True:
        _die(f"{where}.ok is not true")
    if check.get("first_mismatch") is not None:
        _die(f"{where}: first_mismatch is set")
    stats = _mapping(check.get("stats"), f"{where}.stats")
    for key in ("encode_mismatches", "decode_mismatches", "batch_mismatches",
                "hf_encode_mismatches", "hf_decode_mismatches"):
        if stats.get(key) != 0:
            _die(f"{where}.stats.{key} is not 0")
    if stats.get("hf_batch_ok") is not True:
        _die(f"{where}.stats.hf_batch_ok is not true")
    if stats.get("hf_available") is not None and stats.get("hf_available") is not True:
        _die(f"{where}.stats.hf_available is not true")
    if stats.get("hf_ok") is not None and stats.get("hf_ok") is not True:
        _die(f"{where}.stats.hf_ok is not true")


def _check_giga_side(summary, side, where, byte_count, passes, total_tokens, rounds):
    side_where = f"{where}.summary.{side}"
    values = _mapping(summary.get(side), side_where)
    seconds = _positive(values.get("seconds_median"), f"{side_where}.seconds_median")
    mib_s = _positive(values.get("mib_s_median"), f"{side_where}.mib_s_median")
    tokens_s = _positive(values.get("tokens_s_median"), f"{side_where}.tokens_s_median")

    round_seconds = []
    for index, entry in enumerate(rounds):
        round_side = _mapping(entry.get(side), f"{where}.rounds[{index}].{side}")
        if _count(round_side.get("passes"), f"{where}.rounds[{index}].{side}.passes") != passes:
            _die(f"{where}.rounds[{index}].{side}.passes is not {passes}")
        _count(round_side.get("count"), f"{where}.rounds[{index}].{side}.count")
        round_seconds.append(
            _positive(round_side.get("seconds"), f"{where}.rounds[{index}].{side}.seconds"))
    _close(seconds, _median(round_seconds), f"{side_where}.seconds_median")
    _close(_positive(values.get("seconds_min"), f"{side_where}.seconds_min"),
           min(round_seconds), f"{side_where}.seconds_min")
    _close(_positive(values.get("seconds_max"), f"{side_where}.seconds_max"),
           max(round_seconds), f"{side_where}.seconds_max")

    # MiB/s is bytes per pass times passes per round over the round seconds.
    _close(mib_s, byte_count * passes / seconds / MIB, f"{side_where}.mib_s_median", 1e-9)
    _close(_positive(values.get("mib_s_min"), f"{side_where}.mib_s_min"),
           byte_count * passes / max(round_seconds) / MIB, f"{side_where}.mib_s_min", 1e-9)
    _close(_positive(values.get("mib_s_max"), f"{side_where}.mib_s_max"),
           byte_count * passes / min(round_seconds) / MIB, f"{side_where}.mib_s_max", 1e-9)
    _close(tokens_s, total_tokens * passes / seconds, f"{side_where}.tokens_s_median", 1e-9)
    return round_seconds


def _check_giga_cold(body, where, op, decoded_bytes, total_tokens, documents):
    """Check the explicit first pass; model load time is not part of it."""
    cold = _mapping(body.get("cold"), f"{where}.cold")
    expected = (decoded_bytes if op in ("decode", "decode_list")
                else total_tokens if op == "encode"
                else total_tokens + documents)
    seconds = {}
    for side in ("tokie", "giga"):
        side_where = f"{where}.cold.{side}"
        values = _mapping(cold.get(side), side_where)
        first = _positive(values.get("first_call_s"), f"{side_where}.first_call_s")
        remaining = values.get("remaining_pass_s")
        if (isinstance(remaining, bool) or not isinstance(remaining, (int, float))
                or not math.isfinite(remaining) or remaining < 0):
            _die(f"{side_where}.remaining_pass_s: expected a finite non-negative number, "
                 f"got {remaining!r}")
        full = _positive(values.get("cold_full_pass_s"), f"{side_where}.cold_full_pass_s")
        _close(full, first + remaining, f"{side_where}.cold_full_pass_s")
        if _count(values.get("count"), f"{side_where}.count") != expected:
            _die(f"{side_where}.count is not the expected {expected} outputs")
        seconds[side] = full
    return seconds


def _decode_title(op, ids_input, where):
    """Decode labels name the id container the timed call actually received."""
    if ids_input is None:
        return GIGA_OPS[op][0]  # archived results predate ids_input: NumPy ids
    if not isinstance(ids_input, str):
        _die(f"{where}.ids_input: expected a string, got {ids_input!r}")
    normalized = ids_input.strip().lower().replace("-", "_").replace(" ", "_")
    if "list" in normalized:
        title = "Decoding Python list ids"
    elif any(token in normalized for token in ("numpy", "ndarray", "uint32")):
        title = "Decoding NumPy uint32 ids"
    else:
        _die(f"{where}.ids_input: unknown id input {ids_input!r}")
    expected = ("Decoding Python list ids" if op == "decode_list"
                else "Decoding NumPy uint32 ids")
    if title != expected:
        _die(f"{where}.ids_input {ids_input!r} does not match operation {op!r}")
    return title


def load_giga_results(path):
    """Validate one tokie-vs-gigatoken results JSON.

    Returns the caption pieces plus one entry per present operation. Only the
    operations recorded in the file are charted, so a batch-only 8-thread run
    never invents single-thread bars.
    """
    doc = _mapping(_load_json(path), path)
    errors = doc.get("errors")
    if not isinstance(errors, list) or errors:
        _die(f"{path}: errors must be an empty list")

    meta = _mapping(doc.get("meta"), "meta")
    benchmark = _mapping(doc.get("benchmark"), "benchmark")
    validation = _mapping(doc.get("validation"), "validation")
    status = _mapping(doc.get("status"), "status")
    if not benchmark:
        _die(f"{path}: benchmark object has no models")
    if set(benchmark) != set(validation) or set(benchmark) != set(status):
        _die(f"{path}: benchmark, validation, and status name different models")

    packages = _mapping(meta.get("packages"), "meta.packages")
    facts = {
        "tokie": _version(packages, "tokie", "meta.packages.tokie"),
        "giga": _version(packages, "gigatoken", "meta.packages.gigatoken"),
        "cpu_model": _text(meta.get("cpu_model"), "meta.cpu_model"),
        "cpu_count": _count(meta.get("cpu_count"), "meta.cpu_count"),
        "date": _text(meta.get("generated_utc"), "meta.generated_utc").split("T")[0],
    }
    if len(facts["date"]) != 10 or facts["date"][4] != "-" or facts["date"][7] != "-":
        _die(f"meta.generated_utc: expected an ISO date, got {meta.get('generated_utc')!r}")

    corpus = _mapping(meta.get("corpus"), "meta.corpus")
    facts["documents"] = _count(corpus.get("sampled_documents"), "meta.corpus.sampled_documents")
    facts["rows_sha256"] = _text(corpus.get("sampled_rows_sha256"),
                                 "meta.corpus.sampled_rows_sha256")
    facts["jsonl_sha256"] = _text(corpus.get("jsonl_sha256"), "meta.corpus.jsonl_sha256")
    timing_texts = _count(corpus.get("timing_texts"), "meta.corpus.timing_texts")
    _count(corpus.get("sampled_bytes_utf8"), "meta.corpus.sampled_bytes_utf8")
    _count(corpus.get("validation_texts"), "meta.corpus.validation_texts")
    if timing_texts != facts["documents"]:
        _die("meta.corpus.timing_texts differs from sampled_documents")

    env = _mapping(meta.get("env"), "meta.env")
    env_threads = _count(env.get("RAYON_NUM_THREADS"), "meta.env.RAYON_NUM_THREADS")

    entries = []
    for model, ops in benchmark.items():
        model_where = f"benchmark.{model}"
        ops = _mapping(ops, model_where)
        _check_giga_validation(validation, status, model)
        for op, body in ops.items():
            if op not in GIGA_OPS:
                _die(f"{model_where}: unknown operation {op!r}")
            title, stem, byte_kind = GIGA_OPS[op]
            where = f"{model_where}.{op}"
            body = _mapping(body, where)
            if op in ("decode", "decode_list"):
                title = _decode_title(op, body.get("ids_input"), where)

            threads = _count(body.get("threads"), f"{where}.threads")
            if threads != env_threads:
                _die(f"{where}.threads is {threads} but meta.env.RAYON_NUM_THREADS is {env_threads}")
            passes = _count(body.get("passes_per_round"), f"{where}.passes_per_round")
            input_bytes = _count(body.get("input_bytes_utf8"), f"{where}.input_bytes_utf8")
            decoded_bytes = _count(body.get("decoded_bytes_utf8"), f"{where}.decoded_bytes_utf8")
            total_tokens = _count(body.get("total_tokens"), f"{where}.total_tokens")
            byte_count = decoded_bytes if byte_kind == "decoded" else input_bytes

            rounds = _sequence(body.get("rounds"), f"{where}.rounds")
            check = _mapping(body.get("post_check"), f"{where}.post_check")
            checksums = {}
            for side in ("tokie", "giga"):
                side_check = _mapping(check.get(side), f"{where}.post_check.{side}")
                if side_check.get("output_ok") is not True:
                    _die(f"{where}.post_check.{side}.output_ok is not true")
                checksums[side] = side_check.get("checksum")
            if checksums["tokie"] != checksums["giga"]:
                _die(f"{where}: tokie and giga output checksums differ")

            summary = _mapping(body.get("summary"), f"{where}.summary")
            seconds = {}
            for side in ("tokie", "giga"):
                seconds[side] = _check_giga_side(
                    summary, side, where, byte_count, passes, total_tokens, rounds)

            ratio = _mapping(summary.get("giga_over_tokie_time"), f"{where}.summary.giga_over_tokie_time")
            round_ratios = [
                giga / tokie for tokie, giga in zip(seconds["tokie"], seconds["giga"])
            ]
            _close(_positive(ratio.get("median"), f"{where}.summary.giga_over_tokie_time.median"),
                   _median(round_ratios), f"{where}.summary.giga_over_tokie_time.median")

            profile = _mapping(body.get("profile"), f"{where}.profile")
            for side in ("tokie", "giga"):
                side_profile = _mapping(profile.get(side), f"{where}.profile.{side}")
                _positive(side_profile.get("cpu_per_wall"), f"{where}.profile.{side}.cpu_per_wall")

            cold_seconds = _check_giga_cold(
                body, where, op, decoded_bytes, total_tokens, facts["documents"])

            entries.append({
                "model": model,
                "op": op,
                "title": title,
                "stem": stem,
                "threads": threads,
                "tokie_mib_s": _positive(summary["tokie"].get("mib_s_median"),
                                         f"{where}.summary.tokie.mib_s_median"),
                "giga_mib_s": _positive(summary["giga"].get("mib_s_median"),
                                        f"{where}.summary.giga.mib_s_median"),
                "mib": byte_count / MIB,
                "input_bytes": input_bytes,
                "rounds": len(rounds),
                "passes": passes,
                "checksum": checksums["tokie"],
                "giga_cpu": _positive(profile["giga"].get("cpu_per_wall"),
                                      f"{where}.profile.giga.cpu_per_wall"),
                "cold_ms": {side: cold_seconds[side] * 1000.0 for side in cold_seconds},
                "byte_kind": byte_kind,
            })

    # The caption is per file, so every operation and model must agree on the
    # input size and round count it would be quoting.
    if len({entry["rounds"] for entry in entries}) != 1:
        _die(f"{path}: operations disagree on the number of rounds")
    if len({entry["input_bytes"] for entry in entries}) != 1:
        _die(f"{path}: operations disagree on the input size")

    operations = {entry["op"] for entry in entries}
    for model in benchmark:
        if {entry["op"] for entry in entries if entry["model"] == model} != operations:
            _die(f"{path}: {model} does not have the same operations as the other models")
    return facts, entries


def _giga_caption(facts, entry, cpu_per_wall):
    threads = entry["threads"]
    size = f"~{entry['mib']:.1f} MiB {'decoded' if entry['byte_kind'] == 'decoded' else 'input'}"
    lines = [
        f"tokie {facts['tokie']} vs gigatoken {facts['giga']} \u00b7 {facts['date']}",
        f"{facts['cpu_model']} \u00b7 {facts['cpu_count']} CPUs \u00b7 warm cache",
        f"{facts['documents']} OpenWebText documents, {size} \u00b7 "
        f"{entry['rounds']} rounds \u00b7 higher is better",
    ]
    if threads > 1 and cpu_per_wall <= 1.2:
        # Small batches take gigatoken's serial path; say so instead of letting
        # an 8-thread run read as a general threading claim.
        lines.append(f"gigatoken cpu/wall {cpu_per_wall:.1f} at this size")
    if entry["byte_kind"] == "decoded":
        # Decode timing includes the binding's id conversion and byte output.
        lines.append(f"{entry['passes']} passes/round \u00b7 decoded bytes out \u00b7 "
                     "full Python API incl. id input conversion, not kernel-only")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_published_hf(caption, models, out_dir):
    """Headline: tokie vs Hugging Face encode throughput, three tokenizer families."""
    bars = []
    for model, tokie_mib_s, hf_mib_s in models:
        label = MODEL_LABELS.get(model, model)
        bars.append((f"{label} \u00b7 tokie", tokie_mib_s, TOKIE_COLOR, ""))
        bars.append((f"{label} \u00b7 HF", hf_mib_s, GRAY_COLOR,
                     _slower(tokie_mib_s / hf_mib_s)))
    _horiz_chart(
        "Encoding speed (higher is better)",
        bars,
        "MiB/s",
        caption,
        "benchmark_published_hf",
        out_dir,
    )


def chart_published_gigatoken(spec, out_dir):
    """One chart for a validated tokie-vs-gigatoken operation.

    Encode, decode, and batch at one thread use the fixed stems; batch at other
    thread counts is named after the count so a later run cannot overwrite it.
    """
    group = spec["group"]
    first = group[0]
    bars = []
    for entry in group:
        label = MODEL_LABELS.get(entry["model"], entry["model"])
        bars.append((f"{label} \u00b7 tokie", entry["tokie_mib_s"],
                     TOKIE_COLOR, f"{entry['tokie_mib_s']:.1f}"))
        bars.append((f"{label} \u00b7 gigatoken", entry["giga_mib_s"],
                     GRAY_COLOR, f"{entry['giga_mib_s']:.1f}"))

    _horiz_chart(
        f"{first['title']} ({first['threads']} "
        f"{'thread' if first['threads'] == 1 else 'threads'})",
        bars,
        "MiB/s",
        _giga_caption(spec["facts"], first, max(entry["giga_cpu"] for entry in group)),
        spec["name"],
        out_dir,
    )


def _first_pass_caption(facts, entry):
    return "\n".join((
        f"tokie {facts['tokie']} vs gigatoken {facts['giga']} \u00b7 {facts['date']}",
        f"{facts['cpu_model']} \u00b7 {facts['cpu_count']} CPUs \u00b7 "
        f"{entry['threads']} threads",
        f"{facts['documents']} OpenWebText documents, ~{entry['mib']:.1f} MiB input",
        "one observation per model \u00b7 tokie measured first \u00b7 "
        "fresh instance, model loading excluded",
    ))


def chart_published_first_pass(spec, out_dir):
    """One fresh-instance first pass per model; model loading is excluded."""
    group = spec["group"]
    bars = []
    for entry in group:
        label = MODEL_LABELS.get(entry["model"], entry["model"])
        for side, color in (("tokie", TOKIE_COLOR), ("gigatoken", GRAY_COLOR)):
            key = "tokie" if side == "tokie" else "giga"
            millis = entry["cold_ms"][key]
            bars.append((f"{label} \u00b7 {side}", millis, color, f"{millis:.1f}"))

    _horiz_chart(
        "First inference pass (8-thread batch, lower is better)",
        bars,
        "ms",
        _first_pass_caption(spec["facts"], group[0]),
        spec["name"],
        out_dir,
    )


def _giga_chart_specs(facts, entries, suffix=""):
    """Group one file's entries into one chart per operation and thread count."""
    groups = {}
    for entry in entries:
        groups.setdefault((entry["op"], entry["threads"]), []).append(entry)

    specs = []
    for (op, threads), group in groups.items():
        first = group[0]
        if op == "batch":
            name = f"benchmark_published_gigatoken_batch_{threads}t"
        else:
            name = f"benchmark_published_gigatoken_{first['stem']}"
        specs.append({"name": name + suffix, "facts": facts, "group": group})
    return specs


def _first_pass_spec(facts, entries):
    """First-pass chart comes from the large batch file, not the warm rounds."""
    batch = [entry for entry in entries if entry["op"] == "batch"]
    if not batch:
        return None
    return {
        "name": "benchmark_published_gigatoken_first_pass",
        "facts": facts,
        "group": batch,
    }


def _check_decode_override(base_files, facts, entries, where):
    """Let the supplemental file replace decode only, on the same corpus."""
    operations = sorted({entry["op"] for entry in entries})
    if operations != ["decode", "decode_list"]:
        _die(f"{where}: expected the decode and decode_list operations, got {operations}")
    models = {entry["model"] for entry in entries}
    threads = {entry["threads"] for entry in entries}
    if len(threads) != 1:
        _die(f"{where}: decode and decode_list disagree on the thread count")
    for model in models:
        by_op = {entry["op"]: entry["checksum"] for entry in entries if entry["model"] == model}
        if by_op["decode"] != by_op["decode_list"]:
            _die(f"{where}: decode and decode_list checksums differ for {model}")

    for path, base_facts, base_entries in base_files:
        if not any(entry["op"] == "decode" for entry in base_entries):
            continue
        for key, label in (("tokie", "tokie version"), ("giga", "gigatoken version"),
                           ("documents", "sampled_documents"),
                           ("rows_sha256", "sampled_rows_sha256"),
                           ("jsonl_sha256", "jsonl_sha256")):
            if base_facts[key] != facts[key]:
                _die(f"{where}: {label} differs from {path}")
        if {entry["model"] for entry in base_entries} != models:
            _die(f"{where}: model set differs from {path}")
        base_decode = [entry for entry in base_entries if entry["op"] == "decode"]
        if {entry["threads"] for entry in base_decode} != threads:
            _die(f"{where}: thread count differs from {path}")
        if {entry["input_bytes"] for entry in base_decode} != \
                {entry["input_bytes"] for entry in entries}:
            _die(f"{where}: input size differs from {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-results", metavar="PATH",
                        help="published tokie-vs-Hugging-Face results.json")
    parser.add_argument("--giga-results", metavar="PATH", action="append", default=[],
                        help="tokie-vs-gigatoken results.json (repeatable)")
    parser.add_argument("--giga-large-results", metavar="PATH", action="append", default=[],
                        help="large-input tokie-vs-gigatoken results.json (repeatable; "
                             "charts get a '_large' stem and feed the first-pass chart)")
    parser.add_argument("--decode-results", metavar="PATH",
                        help="supplemental decode ids results.json (NumPy and list); "
                             "replaces the decode chart from --giga-results")
    parser.add_argument("--output-dir", metavar="DIR", default=ASSETS,
                        help="directory for the PNG and SVG files (default: assets/)")
    args = parser.parse_args(argv)

    if not args.hf_results and not args.giga_results and \
            not args.giga_large_results and not args.decode_results:
        parser.error("pass --hf-results, --giga-results, --giga-large-results "
                     "and/or --decode-results")

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Validate every input before writing anything, so one bad file cannot
    # leave a half-updated set of charts behind.
    hf_data = None
    if args.hf_results:
        hf_data = load_hf_results(args.hf_results)

    plans = []
    written = {}

    def add(kind, spec, path):
        if spec["name"] in written:
            _die(f"{path}: duplicate chart {spec['name']} "
                 f"(already produced from {written[spec['name']]})")
        written[spec["name"]] = path
        plans.append((kind, spec))

    base_files = []
    for path in args.giga_results:
        facts, entries = load_giga_results(path)
        base_files.append((path, facts, entries))

    decode_plan = None
    if args.decode_results:
        facts, entries = load_giga_results(args.decode_results)
        _check_decode_override(base_files, facts, entries, args.decode_results)
        decode_plan = (facts, entries)

    for path, facts, entries in base_files:
        if decode_plan is not None:
            # The supplemental file is the explicit replacement for decode.
            entries = [entry for entry in entries if entry["op"] != "decode"]
        for spec in _giga_chart_specs(facts, entries):
            add("op", spec, path)

    if decode_plan is not None:
        for spec in _giga_chart_specs(*decode_plan):
            add("op", spec, args.decode_results)

    for path in args.giga_large_results:
        facts, entries = load_giga_results(path)
        for spec in _giga_chart_specs(facts, entries, suffix="_large"):
            add("op", spec, path)
        spec = _first_pass_spec(facts, entries)
        if spec is not None:
            add("first_pass", spec, path)

    print("Generating benchmark charts...")
    if hf_data is not None:
        chart_published_hf(hf_data[0], hf_data[1], out_dir)
    for kind, spec in plans:
        if kind == "first_pass":
            chart_published_first_pass(spec, out_dir)
        else:
            chart_published_gigatoken(spec, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
