#!/usr/bin/env python3
"""Generate throughput figures and tables from validated tokie benchmark JSON.

Every number comes from the ``--results`` files; this script has no timings or
per-library data of its own. Two structurally different layouts are available:
a model x task matrix (default) and faceted task panels. Models with a
GigaToken series form the overview figure; HuggingFace-only models form an
optional second figure. ``--check`` validates the same inputs and writes
nothing.

    python scripts/generate_benchmark_dashboard.py --results FILE [--results FILE ...]
        [--layout matrix|faceted] [--output-dir assets]
        [--tables-out docs/benchmarks/uint32-decode/tables.md] [--check]
"""

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.lines import Line2D

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "path",
                     "svg.hashsalt": "tokie-benchmark-dashboard"})

SCHEMA_VERSION = 1
MIB = 1024.0 * 1024.0
TOL = 1e-7
CANVAS, TEXT, MUTED, GRID = "#FFFFFF", "#111827", "#6B7280", "#E5E7EB"
NA_FILL, NA_TEXT, PREVIEW = "#F3F4F6", "#9CA3AF", "#B45309"
LIB_ORDER = ("tokie", "giga", "hf")
LIB_STYLE = {"tokie": ("#2563EB", "o"), "giga": ("#D97706", "s"), "hf": ("#059669", "^")}
TASK_ORDER = (("encode", 1), ("decode", 1), ("batch", 1), ("batch", 8))
TASK_TITLES = {"encode": "Encode", "decode": "Unicode decode", "batch": "Batch"}
MODEL_DISPLAY = {"gpt2": "GPT-2", "qwen2": "Qwen2", "cl100k": "cl100k", "cl100k_base": "cl100k",
                 "bert-base-uncased": "BERT-base", "t5-small": "T5-small"}
RESULT_STEMS = {"overview": "benchmark_current_overview", "hf_only": "benchmark_current_hf_only"}
SUMMARY_FIELDS = ("seconds_median", "seconds_min", "seconds_max",
                  "mib_s_median", "mib_s_min", "mib_s_max")


def _num(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _pos(value):
    return _num(value) and float(value) > 0


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _close(recorded, expected, tol=TOL):
    return abs(float(recorded) - float(expected)) <= tol * max(abs(float(recorded)), abs(float(expected)), 1e-12)


def _flat_validation(entry):
    flat = dict(entry)
    if isinstance(entry.get("stats"), dict):
        flat.update(entry["stats"])
    return flat


def _definitions(tasks):
    """Return task definitions, accepting either a flat map or a definitions map."""
    if not isinstance(tasks, dict):
        return {}
    nested = tasks.get("definitions")
    if isinstance(nested, dict):
        return nested
    return {key: value for key, value in tasks.items() if key in TASK_TITLES and isinstance(value, dict)}


def validate_file(path, data):
    """Return human-readable problems; an empty list means the file is valid."""
    errors = []

    def bad(message):
        errors.append(f"{path}: {message}")

    if not isinstance(data, dict):
        return [f"{path}: top level is not an object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        bad(f"schema_version must be {SCHEMA_VERSION}")
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for key in ("artifacts", "host", "samples", "tasks"):
        if not (isinstance(meta.get(key), dict) and meta[key]):
            bad(f"meta.{key} must be a non-empty object")
    for task, definition in _definitions(meta.get("tasks")).items():
        if not isinstance(definition, dict):
            bad(f"meta.tasks.{task} must be an object")
            continue
        if "ordinal" in definition and not _is_int(definition["ordinal"]):
            bad(f"meta.tasks.{task}.ordinal must be an integer")
        if "unit" in definition and not (isinstance(definition["unit"], str) and definition["unit"].strip()):
            bad(f"meta.tasks.{task}.unit must be a non-empty string")
    artifacts = meta.get("artifacts") if isinstance(meta.get("artifacts"), dict) else {}
    models = data.get("models") if isinstance(data.get("models"), dict) else {}
    if not models:
        bad("models must be a non-empty object")
    used = set()
    for label, model in models.items():
        if not isinstance(model, dict):
            bad(f"models.{label} must be an object")
            continue
        libs = model.get("libraries")
        if not (isinstance(libs, list) and libs):
            bad(f"models.{label}.libraries must be a non-empty list")
            continue
        if len(set(libs)) != len(libs) or any(lib not in LIB_ORDER for lib in libs):
            bad(f"models.{label}.libraries must be unique ids from {list(LIB_ORDER)}")
        if not model.get("tokenizer_json_sha256"):
            bad(f"models.{label}.tokenizer_json_sha256 is required")
        if "ordinal" in model and not _is_int(model["ordinal"]):
            bad(f"models.{label}.ordinal must be an integer")
        used.update(libs)
    for lib in sorted(used):
        if not isinstance(artifacts.get(lib), dict):
            bad(f"meta.artifacts.{lib} is required by models")
    if isinstance(artifacts.get("tokie"), dict) and not artifacts["tokie"].get("kind"):
        bad("meta.artifacts.tokie.kind is required")

    validation = data.get("validation") if isinstance(data.get("validation"), dict) else {}
    for label in models:
        entry = validation.get(label)
        if not isinstance(entry, dict):
            bad(f"validation.{label} is missing")
            continue
        if entry.get("ok") is not True or entry.get("first_mismatch") is not None:
            bad(f"validation.{label} is not ok or has a first_mismatch")
        flat = _flat_validation(entry)
        counters = [key for key in flat if str(key).endswith("_mismatches")]
        if not counters or any(flat[key] != 0 for key in counters):
            bad(f"validation.{label} needs zero *_mismatches counters")
        if "validation_texts" in entry and not entry["validation_texts"]:
            bad(f"validation.{label}.validation_texts is empty")

    benchmark = data.get("benchmark") if isinstance(data.get("benchmark"), dict) else {}
    if not benchmark:
        bad("benchmark must be a non-empty object")
    for label, model in models.items():
        libs = model.get("libraries") if isinstance(model, dict) else None
        if not (isinstance(libs, list) and libs):
            continue
        tasks = benchmark.get(label)
        if not (isinstance(tasks, dict) and tasks):
            bad(f"benchmark.{label} is missing")
            continue
        for task, entry in tasks.items():
            where = f"benchmark.{label}.{task}"
            if task not in TASK_TITLES:
                bad(f"{where} is not encode, decode, or batch")
                continue
            if not isinstance(entry, dict):
                bad(f"{where} must be an object")
                continue
            passes, threads = entry.get("passes_per_round"), entry.get("threads")
            count_unit = entry.get("count_unit")
            if count_unit not in ("ids", "characters", "ids_and_documents"):
                bad(f"{where}.count_unit must be ids, characters, or ids_and_documents")
            if threads not in (1, 8):
                bad(f"{where}.threads must be 1 or 8")
            if not (_is_int(passes) and passes > 0):
                bad(f"{where}.passes_per_round must be a positive integer")
            input_bytes, decoded_bytes = entry.get("input_bytes_utf8"), entry.get("decoded_bytes_utf8")
            if not (_is_int(input_bytes) and input_bytes > 0):
                bad(f"{where}.input_bytes_utf8 must be a positive integer")
            numerator = entry.get("throughput_numerator")
            if numerator is None:
                numerator = "decoded_bytes_utf8" if task == "decode" else "input_bytes_utf8"
            if numerator not in ("input_bytes_utf8", "decoded_bytes_utf8"):
                bad(f"{where}.throughput_numerator must be input_bytes_utf8 or decoded_bytes_utf8")
                numerator = "input_bytes_utf8"
            if numerator == "decoded_bytes_utf8" and not (_is_int(decoded_bytes) and decoded_bytes > 0):
                bad(f"{where}.decoded_bytes_utf8 must be positive when it is the throughput numerator")
            unit_bytes = decoded_bytes if numerator == "decoded_bytes_utf8" else input_bytes
            rounds = entry.get("rounds") if isinstance(entry.get("rounds"), list) else []
            if len(rounds) < 2:
                bad(f"{where}.rounds needs at least 2 entries")
            seconds = {lib: [] for lib in libs}
            counts = {lib: set() for lib in libs}
            for index, round_entry in enumerate(rounds, 1):
                if not isinstance(round_entry, dict):
                    bad(f"{where}.rounds[{index}] must be an object")
                    continue
                order = round_entry.get("order")
                if not (isinstance(order, list) and len(order) == len(libs) and set(order) == set(libs)):
                    bad(f"{where}.rounds[{index}].order must list each library once")
                for lib in libs:
                    sample = round_entry.get(lib)
                    if not isinstance(sample, dict):
                        bad(f"{where}.rounds[{index}].{lib} is missing")
                        continue
                    if _pos(sample.get("seconds")):
                        seconds[lib].append(float(sample["seconds"]))
                    else:
                        bad(f"{where}.rounds[{index}].{lib}.seconds must be positive")
                    if _is_int(sample.get("count")) and sample["count"] > 0:
                        counts[lib].add(sample["count"])
                    else:
                        bad(f"{where}.rounds[{index}].{lib}.count must be positive")
                    if "passes" in sample and sample["passes"] != passes:
                        bad(f"{where}.rounds[{index}].{lib}.passes disagrees")
            if len(rounds) >= 2 and len({tuple(entry.get("order", [])) for entry in rounds
                                        if isinstance(entry, dict)}) == 1:
                bad(f"{where}.round order is not counterbalanced")
            for lib in libs:
                if len(counts[lib]) > 1:
                    bad(f"{where}.{lib} count is not stable across rounds")
            summary = entry.get("summary") if isinstance(entry.get("summary"), dict) else {}
            for lib in libs:
                recorded = summary.get(lib)
                if not isinstance(recorded, dict):
                    bad(f"{where}.summary.{lib} is missing")
                    continue
                if any(not _pos(recorded.get(key)) for key in SUMMARY_FIELDS):
                    bad(f"{where}.summary.{lib} has non-positive {SUMMARY_FIELDS}")
                    continue
                if not seconds[lib]:
                    continue
                median = statistics.median(seconds[lib])
                low, high = min(seconds[lib]), max(seconds[lib])
                expected = (median, low, high, unit_bytes * passes / median / MIB,
                            unit_bytes * passes / high / MIB, unit_bytes * passes / low / MIB)
                for key, value in zip(SUMMARY_FIELDS, expected):
                    if not _close(recorded[key], value):
                        bad(f"{where}.summary.{lib}.{key}={recorded[key]:.6g} does not match rounds ({value:.6g})")
                if not (recorded["seconds_min"] <= recorded["seconds_median"] <= recorded["seconds_max"]):
                    bad(f"{where}.summary.{lib} seconds min/median/max out of order")
                if not (recorded["mib_s_min"] <= recorded["mib_s_median"] <= recorded["mib_s_max"]):
                    bad(f"{where}.summary.{lib} MiB/s min/median/max out of order")
                if recorded.get("count_unit") != count_unit:
                    bad(f"{where}.summary.{lib}.count_unit must match benchmark count_unit")
                if counts[lib]:
                    count = next(iter(counts[lib]))
                    expected_count = (count * passes / median, count * passes / high,
                                      count * passes / low)
                    count_fields = ("count_s_median", "count_s_min", "count_s_max")
                    if any(not _pos(recorded.get(key)) for key in count_fields):
                        bad(f"{where}.summary.{lib} has non-positive {count_fields}")
                    else:
                        for key, value in zip(count_fields, expected_count):
                            if not _close(recorded[key], value):
                                bad(f"{where}.summary.{lib}.{key}={recorded[key]:.6g} "
                                    f"does not match rounds ({value:.6g})")
                    present = {prefix for prefix in ("tokens_s", "chars_s")
                               if any(f"{prefix}_{suffix}" in recorded
                                      for suffix in ("median", "min", "max"))}
                    expected_alias = {"ids": "tokens_s", "characters": "chars_s",
                                      "ids_and_documents": None}.get(count_unit)
                    for prefix in present - ({expected_alias} if expected_alias else set()):
                        bad(f"{where}.summary.{lib}.{prefix}_* is an illegal alias for {count_unit}")
                    if expected_alias:
                        alias_fields = tuple(f"{expected_alias}_{suffix}" for suffix in ("median", "min", "max"))
                        if any(not _pos(recorded.get(key)) for key in alias_fields):
                            bad(f"{where}.summary.{lib} is missing {expected_alias}_* for {count_unit}")
                        else:
                            for key, value in zip(alias_fields, expected_count):
                                if not _close(recorded[key], value):
                                    bad(f"{where}.summary.{lib}.{key} does not match count_s")
            post = entry.get("post_check")
            if post is not None:
                if not isinstance(post, dict):
                    bad(f"{where}.post_check must be an object")
                else:
                    for lib in libs:
                        check = post.get(lib)
                        if not (isinstance(check, dict) and check.get("output_ok") is True and "checksum" in check):
                            bad(f"{where}.post_check.{lib} needs output_ok=true and checksum")

    status = data.get("status")
    if status == "ok" or (isinstance(status, dict) and status.get("label") == "ok"):
        pass
    elif isinstance(status, dict) and status and all(
            value == "ok" for key, value in status.items() if key != "errors"):
        for label in models:
            if label not in status:
                bad(f"status.{label} is missing")
    else:
        bad("status is not ok")
    if data.get("errors") not in (None, []):
        bad("top-level errors is not empty")
    return errors


def load_dataset(paths):
    errors, payloads = [], []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        errors.extend(validate_file(path, data))
        payloads.append(data)
    if errors:
        raise ValueError("\n".join(errors))
    benchmark, models, order, validation, samples, task_meta = {}, {}, [], {}, {}, {}
    for path, data in zip(paths, payloads):
        for label, model in data["models"].items():
            if label in models:
                old = models[label]
                if old.get("tokenizer_json_sha256") != model.get("tokenizer_json_sha256"):
                    errors.append(f"{path}: models.{label} tokenizer hash conflicts with an earlier file")
                if set(old["libraries"]) != set(model["libraries"]):
                    errors.append(f"{path}: models.{label} library list conflicts with an earlier file")
            else:
                models[label] = dict(model)
                order.append(label)
        for key, value in data["meta"]["samples"].items():
            samples.setdefault(key, value)
        for task, definition in _definitions(data["meta"]["tasks"]).items():
            task_meta.setdefault(task, definition)
        validation.update(data["validation"])
        for label, tasks in data["benchmark"].items():
            for task, entry in tasks.items():
                key = (label, task, entry.get("threads"))
                if key in benchmark:
                    errors.append(f"{path}: duplicate benchmark {key} (already provided)")
                else:
                    benchmark[key] = entry
    artifacts = payloads[0]["meta"]["artifacts"]
    for path, data in zip(paths[1:], payloads[1:]):
        if json.dumps(data["meta"]["artifacts"], sort_keys=True) != json.dumps(artifacts, sort_keys=True):
            errors.append(f"{path}: meta.artifacts conflicts with the first results file")
    if errors:
        raise ValueError("\n".join(errors))
    position = {label: index for index, label in enumerate(order)}
    order.sort(key=lambda label: (models[label]["ordinal"] if _is_int(models[label].get("ordinal"))
                                  else len(order) + position[label], position[label]))
    meta = payloads[0]["meta"]
    source = meta.get("tokie_source") if isinstance(meta.get("tokie_source"), dict) else {}
    host = dict(meta["host"])
    for key in ("cpu_model", "cpu_count_logical", "load_before", "platform"):
        if key in meta:
            host.setdefault(key, meta[key])
    if "cpu_count_logical" in host and "cpu_count" not in host:
        host["cpu_count"] = host["cpu_count_logical"]
    return {"files": [str(path) for path in paths], "artifacts": artifacts, "host": host,
            "samples": samples, "generated_utc": meta.get("generated_utc"),
            "report": meta.get("report") or meta.get("report_url") or "docs/benchmarks/uint32-decode/",
            "task_meta": task_meta, "models": models, "model_order": order,
            "validation": validation, "benchmark": benchmark,
            "tokie_commit": source.get("head_commit")}


def _library_name(lib, artifacts):
    artifact, version = artifacts.get(lib, {}), artifacts.get(lib, {}).get("version")
    if lib == "tokie":
        tag = "LOCAL, UNRELEASED" if artifact.get("kind") == "local_unreleased" else "published, historical"
        return f"Tokie {version} ({tag})" if version else f"Tokie ({tag})"
    if lib == "giga":
        return f"GigaToken {version}" if version else "GigaToken"
    name = artifact.get("name") or "HuggingFace tokenizers"
    return f"{name} {version}" if version else name


def _legend_name(lib, artifacts):
    """Two-line legend labels keep full library names readable at README width."""
    artifact = artifacts.get(lib, {})
    version = artifact.get("version")
    if lib == "tokie":
        label = f"Tokie {version}" if version else "Tokie"
        tag = "LOCAL, UNRELEASED" if artifact.get("kind") == "local_unreleased" else "published, historical"
        return f"{label}\n{tag}"
    if lib == "giga":
        return f"** GigaToken\n{version}" if version else "** GigaToken"
    name = artifact.get("name") or "HuggingFace tokenizers"
    return f"{name}\n{version}" if version else name


def _model_name(dataset, label):
    model = dataset["models"][label]
    return model.get("display_name") or MODEL_DISPLAY.get(label) or label.replace("_", " ").replace("-", " ").title()


def _task_definition(dataset, task):
    definition = dataset.get("task_meta", {}).get(task, {})
    return definition if isinstance(definition, dict) else {}


def _task_title(dataset, task_key):
    definition = _task_definition(dataset, task_key[0])
    title = definition.get("title") or TASK_TITLES.get(task_key[0], task_key[0].title())
    if task_key[0] == "batch" and f"{task_key[1]}T" not in str(title):
        title = f"{title} {task_key[1]}T"
    return title


def _task_unit(dataset, task_key):
    unit = _task_definition(dataset, task_key[0]).get("unit")
    return unit.strip() if isinstance(unit, str) and unit.strip() else "MiB/s"


def _uses_decoded_bytes(dataset, task_key):
    for (label, task, threads), entry in dataset["benchmark"].items():
        if task == task_key[0] and threads == task_key[1]:
            numerator = entry.get("throughput_numerator")
            if numerator is not None:
                return numerator == "decoded_bytes_utf8"
            break
    return task_key[0] == "decode"


def _denominator_label(dataset, task_key, log=False):
    label = f"{'Decoded' if _uses_decoded_bytes(dataset, task_key) else 'Input'} {_task_unit(dataset, task_key)}"
    return f"{label} \u00b7 log" if log else label


def _task_ordinal(dataset, task_key):
    ordinal = _task_definition(dataset, task_key[0]).get("ordinal")
    if _is_int(ordinal):
        return ordinal
    return TASK_ORDER.index(task_key) if task_key in TASK_ORDER else len(TASK_ORDER)


def _tasks(dataset, labels):
    keys = {(task, threads) for (label, task, threads) in dataset["benchmark"] if label in labels}
    return sorted(keys, key=lambda key: (_task_ordinal(dataset, key), key[0], key[1]))


def _cell(dataset, label, task_key):
    entry = dataset["benchmark"].get((label, task_key[0], task_key[1]))
    if entry is None:
        return None
    return {lib: tuple(entry["summary"][lib][field] for field in ("mib_s_median", "mib_s_min", "mib_s_max"))
            for lib in dataset["models"][label]["libraries"] if lib in entry.get("summary", {})}


def _libraries(dataset, labels):
    present = set().union(*(dataset["models"][label]["libraries"] for label in labels))
    return [lib for lib in LIB_ORDER if lib in present]


def _sample(dataset, task_key, labels):
    for label in labels:
        entry = dataset["benchmark"].get((label, task_key[0], task_key[1]))
        if entry and _is_int(entry.get("documents")) and _is_int(entry.get("input_bytes_utf8")):
            return entry["documents"], entry["input_bytes_utf8"]
    sample_key = _task_definition(dataset, task_key[0]).get("sample")
    if sample_key not in ("single", "batch"):
        sample_key = "batch" if task_key[0] == "batch" else "single"
    sample = dataset["samples"].get(sample_key, {})
    docs, size = sample.get("documents"), sample.get("input_bytes_utf8")
    return (docs if _is_int(docs) and docs > 0 else None, size if _is_int(size) and size > 0 else None)


def _sample_label(dataset, task_key, labels):
    docs, size = _sample(dataset, task_key, labels)
    if not (docs and size):
        return "sample size not recorded"
    mib = size / MIB
    size_text = f"{mib:.3f} MiB" if mib < 10 else f"{mib:.2f} MiB"
    return f"{docs:,} docs · {size_text} input"


def _limits(dataset, labels, task_key):
    values = [value for label in labels if (cell := _cell(dataset, label, task_key))
              for pair in cell.values() for value in pair[1:]]
    return (min(values) / 1.35, max(values) * 2.4) if values else (1.0, 10.0)


def _value(value):
    return f"{value:,.0f}" if value >= 1000 else f"{value:.0f}" if value >= 100 else f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def _tick(value, _position):
    return "" if value <= 0 else f"{value:,.0f}" if value >= 1000 else f"{value:g}"


def _style(ax):
    ax.set_facecolor(CANVAS)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0,)))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_tick))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=range(2, 10), numticks=100))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="x", colors=MUTED, labelsize=9.2, length=2.5)
    ax.tick_params(axis="y", colors=TEXT, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D1D5DB")
        ax.spines[side].set_linewidth(0.7)
    ax.grid(axis="x", which="major", color=GRID, linewidth=0.7, alpha=0.95)
    ax.set_axisbelow(True)


def _offsets(count):
    return {1: (0.0,), 2: (0.19, -0.19)}.get(count, (0.26, 0.0, -0.26))


def _draw_range(ax, lib, values, y, font=8.6):
    median, low, high = values
    color, marker = LIB_STYLE[lib]
    ax.errorbar([median], [y], xerr=[[median - low], [high - median]], fmt="none",
                ecolor=color, elinewidth=2.2, capsize=3.8, capthick=1.6,
                alpha=0.9, zorder=3)
    ax.plot([median], [y], marker=marker, markersize=7.0, color=color,
            markeredgecolor=CANVAS, markeredgewidth=0.8, zorder=5)
    ax.text(high * 1.05, y, _value(median), color=color, fontsize=font, va="center",
            ha="left", zorder=6, clip_on=False)


def _draw_series(ax, dataset, label, task_key, libraries, base, full_missing):
    """Draw one model around ``base``; return False when the whole cell is N/A."""
    values = _cell(dataset, label, task_key)
    if values is None:
        if full_missing:
            ax.set_facecolor(NA_FILL)
            ax.grid(False)
            ax.text(0.5, base, "N/A", transform=ax.get_yaxis_transform(), ha="center", va="center",
                    color=NA_TEXT, fontsize=9.5, style="italic")
        else:
            ax.axhspan(base - 0.35, base + 0.35, color=NA_FILL, zorder=0)
            ax.text(ax.get_xlim()[0] * 1.5, base, "N/A", color=NA_TEXT, fontsize=9.5,
                    style="italic", va="center", ha="left")
        return False
    for lib, offset in zip(libraries, _offsets(len(libraries))):
        if lib in values:
            _draw_range(ax, lib, values[lib], base + offset, font=8.6 if full_missing else 8.3)
        else:
            ax.text(ax.get_xlim()[0] * 1.5, base + offset, "N/A", color=NA_TEXT,
                    fontsize=8.5, style="italic", va="center", ha="left")
    return True


def _common_unit(dataset, tasks):
    units = {_task_unit(dataset, task) for task in tasks}
    return units.pop() if len(units) == 1 else None


def _decorate(fig, dataset, labels, libraries, tasks):
    artifacts = dataset["artifacts"]
    unit = _common_unit(dataset, tasks)
    unit_text = f"{unit}, higher is better" if unit else "units labelled per task, higher is better"
    local = artifacts.get("tokie", {}).get("kind") == "local_unreleased"
    scope = ("Local unreleased Tokie vs public competitors" if local
             else "Historical published Tokie preview \u2014 not current data")
    fig.text(0.012, 0.974, "Tokenizer throughput by task", fontsize=17, color=TEXT, ha="left",
             va="top", fontweight="bold")
    fig.text(0.012, 0.938, f"{scope}  \u00b7  median + observed 5-round min\u2013max (not CI)"
             f"  \u00b7  {unit_text}  \u00b7  log scale per task; no universal winner",
             fontsize=10, color=MUTED, ha="left", va="top")
    handles = [Line2D([], [], color=LIB_STYLE[lib][0], marker=LIB_STYLE[lib][1], markersize=6.8,
                      linewidth=2.0, alpha=0.9, markeredgecolor=CANVAS, markeredgewidth=0.7)
               for lib in libraries]
    names = [_legend_name(lib, artifacts) for lib in libraries]
    fig.legend(handles, names, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=len(handles),
               frameon=False, fontsize=10.2, handlelength=1.7, columnspacing=2.8,
               handletextpad=0.6, labelspacing=0.35,
               labelcolor=[LIB_STYLE[lib][0] for lib in libraries])
    tokie = artifacts.get("tokie", {})
    if tokie.get("kind") not in (None, "local_unreleased"):
        fig.text(0.985, 0.973, "LAYOUT PREVIEW  \u00b7  historical tokie-vs-HF run  \u00b7  NOT the current benchmark",
                 ha="right", va="top", fontsize=9, color=PREVIEW, style="italic",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF7ED", edgecolor="#FDE68A", linewidth=0.6))
    _footer(fig, dataset, labels, libraries, tasks)


def _decode_adapters(libraries):
    parts = []
    if "tokie" in libraries:
        parts.append("Tokie.decode(ids)")
    if "giga" in libraries:
        parts.append("Giga.decode(ids)\u2192bytes\u2192UTF-8")
    if "hf" in libraries:
        parts.append("HF.decode(ids.tolist())")
    if len(parts) > 1:
        return ", ".join(parts[:-1]) + " and " + parts[-1]
    return parts[0] if parts else ""


def _footer(fig, dataset, labels, libraries, tasks):
    host = dataset["host"]
    cpu = host.get("cpu_model") or host.get("cpu")
    count = host.get("cpu_count")
    if isinstance(cpu, dict):
        count = cpu.get("count") or count
        cpu = cpu.get("model") or cpu.get("name")
    cpu = cpu or (f"{count} CPUs" if count else "CPU not recorded")
    if count and not str(cpu).endswith("CPUs"):
        cpu = f"{cpu} \u00b7 {count} CPUs"
    rounds = {len(dataset["benchmark"][(label, task, threads)]["rounds"])
              for label in labels for task, threads in tasks if (label, task, threads) in dataset["benchmark"]}
    rounds_text = f"{min(rounds)}\u2013{max(rounds)}" if len(rounds) > 1 else str(next(iter(rounds), "?"))
    unit = _common_unit(dataset, tasks)
    unit_text = f"{unit}, higher is better" if unit else "units per task, higher is better"
    tokie = dataset["artifacts"].get("tokie", {})
    build = dataset.get("tokie_commit") or tokie.get("commit") or tokie.get("source")
    if isinstance(build, dict):
        build = build.get("commit") or build.get("sha") or build.get("label")
    provenance = f" \u00b7 tokie {str(build)[:12]}" if build else ""
    lines = [
        f"Dot = median; range = observed min\u2013max over {rounds_text} warm paired rounds (not CI). {unit_text}; log scale within each task.",
        f"Decode: shared np.uint32 ids \u2192 Unicode; {_decode_adapters(libraries)} \u2014 conversions inside the clock.",
        f"{cpu}{provenance} \u00b7 samples above \u00b7 protocol/tables: {dataset['report']}",
    ]
    if "giga" in libraries:
        lines.append("** GigaToken: WordPiece and Unigram not supported.")
    for index, line in enumerate(lines):
        fig.text(0.011, 0.011 + 0.027 * (len(lines) - 1 - index), line, fontsize=9.2,
                 color=MUTED, ha="left", va="bottom")


def _draw_matrix(dataset, labels, tasks, output_dir):
    libraries = _libraries(dataset, labels)
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(CANVAS)
    grid = fig.add_gridspec(len(labels), len(tasks), left=0.115, right=0.985, top=0.775,
                            bottom=0.155, hspace=0.34, wspace=0.26)
    for column, task_key in enumerate(tasks):
        xlim = _limits(dataset, labels, task_key)
        for row, label in enumerate(labels):
            ax = fig.add_subplot(grid[row, column])
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.5, 0.5)
            _style(ax)
            _draw_series(ax, dataset, label, task_key, libraries, 0.0, True)
            if column == 0:
                ax.set_yticks([0.0])
                ax.set_yticklabels([_model_name(dataset, label)], fontsize=12, color=TEXT)
                ax.tick_params(axis="y", pad=7)
            else:
                ax.set_yticks([])
            if row == 0:
                ax.set_title(_task_title(dataset, task_key), fontsize=12, color=TEXT,
                             fontweight="bold", pad=20)
                ax.text(0.5, 1.004, _sample_label(dataset, task_key, labels),
                        transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3, color=MUTED)
            if row == len(labels) - 1:
                ax.set_xlabel(_denominator_label(dataset, task_key, log=True), fontsize=9,
                              color=MUTED, labelpad=3)
            else:
                ax.tick_params(axis="x", labelbottom=False)
    _decorate(fig, dataset, labels, libraries, tasks)
    return fig


def _draw_faceted(dataset, labels, tasks, output_dir):
    libraries = _libraries(dataset, labels)
    columns = min(2, len(tasks))
    rows = 1 if len(tasks) <= 2 else 2
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor(CANVAS)
    grid = fig.add_gridspec(rows, columns, left=0.105, right=0.985, top=0.775,
                            bottom=0.155, hspace=0.40, wspace=0.16)
    for index, task_key in enumerate(tasks):
        row, column = divmod(index, columns)
        ax = fig.add_subplot(grid[row, column])
        ax.set_xlim(*_limits(dataset, labels, task_key))
        ax.set_ylim(-0.58, len(labels) - 0.42)
        _style(ax)
        ax.grid(axis="y", which="major", color=GRID, linewidth=0.7)
        ax.set_yticks(range(len(labels)))
        if column == 0:
            ax.set_yticklabels([_model_name(dataset, label) for label in reversed(labels)],
                               fontsize=11, color=TEXT)
            ax.tick_params(axis="y", pad=7)
        else:
            ax.set_yticklabels([])
        for model_index, label in enumerate(labels):
            _draw_series(ax, dataset, label, task_key, libraries, len(labels) - 1 - model_index, False)
        ax.set_title(_task_title(dataset, task_key), fontsize=13, color=TEXT,
                     fontweight="bold", pad=20)
        ax.text(0.5, 1.004, _sample_label(dataset, task_key, labels),
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8.3, color=MUTED)
        ax.set_xlabel(_denominator_label(dataset, task_key, log=True), fontsize=9,
                      color=MUTED, labelpad=3)
    for index in range(len(tasks), rows * columns):
        fig.add_subplot(grid[index // columns, index % columns]).axis("off")
    _decorate(fig, dataset, labels, libraries, tasks)
    return fig


def _figure_group(dataset, label):
    """Use explicit model metadata first; fall back to Giga presence only if absent."""
    model = dataset["models"][label]
    for key in ("figure", "panel", "suite", "group"):
        value = model.get(key)
        if isinstance(value, str) and value.strip():
            text = value.lower().replace("-", "_")
            if "hf" in text or "only" in text:
                return "hf_only"
            if "overview" in text or "main" in text:
                return "overview"
    has_batch = any(key[0] == label and key[1] == "batch" for key in dataset["benchmark"])
    return "overview" if has_batch or "giga" in model["libraries"] else "hf_only"


def _groups(dataset):
    overview = [label for label in dataset["model_order"] if _figure_group(dataset, label) == "overview"]
    hf_only = [label for label in dataset["model_order"] if _figure_group(dataset, label) == "hf_only"]
    groups = []
    if overview:
        groups.append(("overview", overview))
    if hf_only:
        groups.append(("hf_only", hf_only))
    return groups or [("overview", list(dataset["model_order"]))]


def _normalize_svg(path):
    """Trim trailing whitespace; savefig has already written deterministic metadata."""
    file = Path(path)
    text = file.read_text(encoding="utf-8")
    file.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def render(dataset, layout, output_dir):
    groups = _groups(dataset)
    os.makedirs(output_dir, exist_ok=True)
    written = []
    for group, labels in groups:
        tasks = _tasks(dataset, labels)
        if not tasks:
            continue
        fig = (_draw_matrix if layout == "matrix" else _draw_faceted)(dataset, labels, tasks, output_dir)
        for extension in ("png", "svg"):
            path = os.path.join(output_dir, f"{RESULT_STEMS[group]}.{extension}")
            if extension == "svg":
                fig.savefig(path, dpi=200, bbox_inches=None, facecolor=fig.get_facecolor(),
                            metadata={"Date": None})
                _normalize_svg(path)
            else:
                fig.savefig(path, dpi=200, bbox_inches=None, facecolor=fig.get_facecolor())
            written.append(path)
        plt.close(fig)
    return written


def _table_library_name(lib, artifacts):
    name = _library_name(lib, artifacts)
    return r"\*\* " + name if lib == "giga" else name


def write_tables(dataset, path):
    lines = ["# Benchmark tables", "",
             f"Generated by `scripts/generate_benchmark_dashboard.py` from {len(dataset['files'])} results files. "
             "Throughput in the unit named per task; each cell is median [min\u2013max] over the recorded paired rounds. N/A = not measured or not supported.",
             ""]
    groups = [(("HuggingFace-only models" if key == "hf_only" else "Overview"), labels)
              for key, labels in _groups(dataset)]
    for heading, labels in groups:
        if not labels:
            continue
        libraries = _libraries(dataset, labels)
        lines.extend([f"## {heading}", ""])
        for task_key in _tasks(dataset, labels):
            sample = _sample_label(dataset, task_key, labels)
            lines.extend([f"### {_task_title(dataset, task_key)} \u2014 {task_key[1]} thread{'s' if task_key[1] != 1 else ''} \u2014 {sample} \u2014 {_denominator_label(dataset, task_key)}", "",
                          "| Model | " + " | ".join(_table_library_name(lib, dataset["artifacts"]) for lib in libraries) + " |",
                          "| --- | " + " | ".join("---:" for _ in libraries) + " |"])
            for label in labels:
                values = _cell(dataset, label, task_key)
                cells = [f"{_value(values[lib][0])} [{_value(values[lib][1])}\u2013{_value(values[lib][2])}]"
                         if values and lib in values else "N/A" for lib in libraries]
                lines.append(f"| {_model_name(dataset, label)} | " + " | ".join(cells) + " |")
            lines.append("")
        lines.extend(["### Validation", "", "| Model | Encode mismatches | Decode mismatches | Batch mismatches | Status |",
                      "| --- | ---: | ---: | ---: | --- |"])
        for label in labels:
            entry = _flat_validation(dataset["validation"].get(label, {}))
            lines.append(f"| {_model_name(dataset, label)} | {entry.get('encode_mismatches', 'N/A')} | "
                         f"{entry.get('decode_mismatches', 'N/A')} | {entry.get('batch_mismatches', 'N/A')} | "
                         f"{'ok' if entry.get('ok') else 'check'} |")
        lines.append("")
        if "giga" in libraries:
            lines.extend([r"\*\* GigaToken: WordPiece and Unigram not supported.", ""])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate benchmark figures/tables from tokie results JSON.",
        epilog="--check validates every --results file and the merged dataset, then exits without writing anything.")
    parser.add_argument("--results", action="append", required=True, metavar="PATH",
                        help="results JSON file; repeat once per file")
    parser.add_argument("--layout", choices=("matrix", "faceted"), default="faceted",
                        help="faceted: 2x2 task panels (default, README); matrix: compact model x task prototype")
    parser.add_argument("--output-dir", default="assets", metavar="DIR", help="figure directory (default: assets)")
    parser.add_argument("--tables-out", default="docs/benchmarks/uint32-decode/tables.md", metavar="PATH",
                        help="markdown tables path (default: docs/benchmarks/uint32-decode/tables.md)")
    parser.add_argument("--check", action="store_true", help="validate inputs and merged dataset only; write no figures or tables")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.results)
    except ValueError as exc:
        for line in str(exc).splitlines():
            print(f"error: {line}", file=sys.stderr)
        return 2
    print(f"validated {len(args.results)} files, {len(dataset['model_order'])} models, {len(dataset['benchmark'])} model/task/thread series")
    if args.check:
        print("check only: no figures or tables written")
        return 0
    written = render(dataset, args.layout, args.output_dir)
    if args.tables_out:
        written.append(write_tables(dataset, args.tables_out))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
