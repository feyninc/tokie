#!/usr/bin/env python3
"""Generate benchmark chart images for the tokie README.

Flexoki color scheme, horizontal bars, "Nx slower" annotations.
"""

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
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)


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


def _save(fig, name):
    for ext in ("png", "svg"):
        path = os.path.join(ASSETS, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  saved {path}")
    plt.close(fig)


def _horiz_chart(title, bars, xlabel, caption, fname, figsize=None):
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

    _save(fig, fname)


# ---------------------------------------------------------------------------
# Data (Python benchmarks, Apple M3, tokie 0.1.0, 900KB synthetic text)
#
# Encode (single string, 900KB, median of 10):
#   GPT-2:  tokie=9.42ms (95.5 MB/s), kitoken=18.5ms (48.6 MB/s), HF=209.2ms (4.3 MB/s)
#   BERT:   tokie=9.84ms (91.5 MB/s), kitoken=48.8ms (18.4 MB/s), HF=280.6ms (3.2 MB/s)
#   Gemma3: tokie=131ms  (6.9 MB/s),  HF=329.5ms (2.7 MB/s) [kitoken inaccurate]
#
# tiktoken (900KB):
#   cl100k: tokie=9.63ms, tiktoken=45.67ms  → 4.7x
#   o200k:  tokie=9.83ms, tiktoken=81.47ms  → 8.3x
#
# Loading (cl100k, from_pretrained, cached):
#   tokie (.tkz)=54.6ms, HF=173.2ms  → 3.2x
# ---------------------------------------------------------------------------


def chart_overview():
    """Hero chart: tokenization speed (GPT-2, MB/s) with all competitors."""
    _horiz_chart(
        "Tokenization speed",
        [
            ("tokie",          625.0, TOKIE_COLOR, ""),
            ("kitoken",        70.0,  GOLD_COLOR,  "8.9x slower"),
            ("tiktoken",       24.9,  GRAY_COLOR,  "25x slower"),
            ("HF tokenizers",  4.7,   GRAY_COLOR,  "133x slower"),
        ],
        "MB/s",
        "GPT-2 encoder, 900KB text, Apple M3, tokie 0.1.0",
        "benchmark",
    )


def chart_bpe():
    """BPE encoding speed (GPT-2, MB/s)."""
    _horiz_chart(
        "BPE encoding speed",
        [
            ("tokie",          625.0, TOKIE_COLOR, ""),
            ("kitoken",        70.0,  GOLD_COLOR,  "8.9x slower"),
            ("HF tokenizers",  4.7,   GRAY_COLOR,  "133x slower"),
        ],
        "MB/s",
        "GPT-2 encoder, 900KB text, Apple M3, tokie 0.1.0",
        "benchmark_bpe",
    )


def chart_wordpiece():
    """WordPiece encoding speed (BERT, MB/s)."""
    _horiz_chart(
        "WordPiece encoding speed",
        [
            ("tokie",          250.0, TOKIE_COLOR, ""),
            ("kitoken",        28.1,  GOLD_COLOR,  "8.9x slower"),
            ("HF tokenizers",  3.5,   GRAY_COLOR,  "71x slower"),
        ],
        "MB/s",
        "BERT-base-uncased, 900KB text, Apple M3, tokie 0.1.0",
        "benchmark_wordpiece",
    )


def chart_sentencepiece():
    """SentencePiece BPE encoding speed (Gemma 3, MB/s).
    No kitoken — it produces incorrect output on SentencePiece models."""
    _horiz_chart(
        "SentencePiece BPE encoding speed",
        [
            ("tokie",          7.9,   TOKIE_COLOR, ""),
            ("HF tokenizers",  3.1,   GRAY_COLOR,  "2.5x slower"),
        ],
        "MB/s",
        "Gemma 3, 900KB text, Apple M3, tokie 0.1.0",
        "benchmark_sentencepiece",
    )


def chart_unigram():
    """Unigram (SentencePiece Unigram) encoding speed (T5, MB/s).
    No kitoken — it produces incorrect output on SentencePiece models.
    tokie runs Viterbi per metaspace unit and memoizes Zipf-frequent units."""
    _horiz_chart(
        "Unigram encoding speed",
        [
            ("tokie",          97.5,  TOKIE_COLOR, ""),
            ("HF tokenizers",  3.2,   GRAY_COLOR,  "30x slower"),
        ],
        "MB/s",
        "T5-base, 900KB text, Apple M3, tokie 0.1.4",
        "benchmark_unigram",
    )


def chart_tiktoken():
    """tiktoken comparison (cl100k/o200k, ms — vertical bars, dark bg)."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    dark_bg = "#1C1B1A"  # Flexoki black
    fg = "#CECDC3"       # Flexoki tx-2 light
    border = "#575653"   # Flexoki ui-2

    ax.set_facecolor(dark_bg)
    fig.set_facecolor(dark_bg)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(border)
    ax.spines["bottom"].set_color(border)
    ax.tick_params(colors=fg, labelsize=10)
    ax.yaxis.label.set_color(fg)
    ax.set_title("OpenAI Tokenizer Speed (900 KB)",
                 color=fg, fontsize=14, fontweight="bold", pad=10)

    models = ["cl100k\n(GPT-4)", "o200k\n(GPT-4o)"]
    tokie_vals = [1.28, 1.66]
    tt_vals = [42.31, 70.72]
    speedups = ["33x faster", "42x faster"]

    x = [0, 1]
    width = 0.32
    tokie_c = "#879A39"  # Flexoki green (light variant for dark bg)
    tt_c = "#878580"     # Flexoki tx-2

    ax.bar([xi - width/2 for xi in x], tokie_vals, width,
           color=tokie_c, label="tokie", zorder=3)
    ax.bar([xi + width/2 for xi in x], tt_vals, width,
           color=tt_c, label="tiktoken", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, color=fg)
    ax.set_ylabel("Time (ms) \u2014 lower is better", fontsize=10)

    for i, sp in enumerate(speedups):
        ax.text(x[i] - width/2, tokie_vals[i] + 2, sp,
                ha="center", va="bottom",
                color=tokie_c, fontsize=10, fontweight="bold")

    ax.legend(loc="upper left", facecolor="#343331", edgecolor=border,
              labelcolor=fg, fontsize=9)

    _save(fig, "benchmark_tiktoken")


def chart_loading():
    """Tokenizer loading time (cl100k, ms). Consistent model for all."""
    _horiz_chart(
        "Tokenizer loading time",
        [
            ("tokie (.tkz)",   11.1,   TOKIE_COLOR, ""),
            ("HF tokenizers",  244.3,  GRAY_COLOR,  "22x slower"),
        ],
        "ms",
        "cl100k tokenizer, warm cached load, Apple M3, tokie 0.1.0",
        "benchmark_loading",
    )


if __name__ == "__main__":
    print("Generating benchmark charts...")
    chart_overview()
    chart_bpe()
    chart_wordpiece()
    chart_sentencepiece()
    chart_unigram()
    chart_tiktoken()
    chart_loading()
    print("Done.")
