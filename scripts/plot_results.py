#!/usr/bin/env python3
"""Produce the three MacMat charts from results/*.csv.

  python3 scripts/plot_results.py [--results results] [--out results/charts]

Charts: latency vs matrix size, tile-size sensitivity, thread scaling.
Colors are assigned per implementation / shape in a fixed order from a
colorblind-validated palette; identity is also carried by markers and direct labels.
"""
import argparse
import csv
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
IMPL_ORDER = ["naive", "reordered", "blocked", "parallel", "accelerate", "micro", "micro_parallel"]
IMPL_COLOR = {name: PALETTE[i] for i, name in enumerate(IMPL_ORDER)}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False,
})


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("M", "K", "N", "tile", "threads", "batch", "repeat"):
            r[k] = int(r[k])
        for k in ("median_ms", "q1_ms", "q3_ms", "iqr_ms", "gflops"):
            r[k] = float(r[k])
    return rows


def shape_of(r):
    return (r["M"], r["K"], r["N"])


def shape_label(s):
    return f"{s[0]}x{s[1]}x{s[2]}" if len(set(s)) > 1 else f"{s[0]}^3"


def aggregate(rows, key):
    """Median over repeats of (median_ms, q1, q3, gflops) grouped by key(row)."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = {}
    for k, rs in groups.items():
        out[k] = {f: statistics.median(r[f] for r in rs) for f in ("median_ms", "q1_ms", "q3_ms", "gflops")}
        out[k]["n"] = len(rs)
    return out


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=TEXT, fontsize=12, pad=12)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.tick_params(length=0)


def chart_latency(rows, out, source_name):
    sq = [r for r in rows if r["M"] == r["K"] == r["N"]]
    if not sq:
        return
    agg = aggregate(sq, lambda r: (r["impl"], r["M"]))
    sizes = sorted({r["M"] for r in sq})
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, impl in enumerate(IMPL_ORDER):
        xs = [n for n in sizes if (impl, n) in agg]
        if not xs:
            continue
        med = [agg[(impl, n)]["median_ms"] for n in xs]
        lo = [agg[(impl, n)]["median_ms"] - agg[(impl, n)]["q1_ms"] for n in xs]
        hi = [agg[(impl, n)]["q3_ms"] - agg[(impl, n)]["median_ms"] for n in xs]
        ax.errorbar(xs, med, yerr=[lo, hi], color=IMPL_COLOR[impl], marker=MARKERS[i], markersize=5,
                    linewidth=2, capsize=2, label=impl)
        ax.annotate(impl, (xs[-1], med[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=TEXT)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    style_axes(ax, f"Median latency vs. square size, IQR bars ({source_name})",
               "M = K = N", "latency per multiplication (ms, log)")
    ax.legend(loc="upper left")
    ax.set_xlim(right=sizes[-1] * 1.9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "latency_vs_size.png"), dpi=160)
    plt.close(fig)


def _tile_panel(ax, rows, title, xlabel):
    agg = aggregate(rows, lambda r: (shape_of(r), r["tile"]))
    shapes = sorted({shape_of(r) for r in rows}, key=lambda s: s[0] * s[1] * s[2])
    keep = [s for s in shapes if s[0] * s[1] * s[2] >= 128 ** 3][:8]
    tiles = sorted({r["tile"] for r in rows})
    for i, s in enumerate(keep):
        xs = [t for t in tiles if (s, t) in agg]
        ys = [agg[(s, t)]["gflops"] for t in xs]
        ax.plot(xs, ys, color=PALETTE[i], marker=MARKERS[i], markersize=5, linewidth=2, label=shape_label(s))
    ax.set_xscale("log", base=2)
    ax.set_xticks(tiles)
    ax.set_xticklabels([str(t) for t in tiles])
    ax.set_ylim(bottom=0)
    style_axes(ax, title, xlabel, "GFLOP/s (from median latency)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)


def chart_tiles(blocked_rows, micro_rows, out):
    if not blocked_rows:
        return
    panels = 2 if micro_rows else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 5), squeeze=False)
    _tile_panel(axes[0][0], blocked_rows, "Blocked kernel (i-k-j tiles), 1 thread", "tile edge (rows = k = cols)")
    if micro_rows:
        _tile_panel(axes[0][1], micro_rows, "Micro-kernel (4x16 register block), 1 thread", "k-block size")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "tile_sensitivity.png"), dpi=160)
    plt.close(fig)


def _thread_panel(ax, rows, title):
    agg = aggregate(rows, lambda r: (shape_of(r), r["threads"]))
    shapes = sorted({shape_of(r) for r in rows}, key=lambda s: s[0] * s[1] * s[2])
    threads = sorted({r["threads"] for r in rows})
    ax.plot(threads, threads, color=MUTED, linestyle="--", linewidth=1, label="ideal")
    ax.axvspan(4, threads[-1] + 0.5, color=GRID, alpha=0.5, lw=0)
    ax.text(4.1, threads[-1] - 0.3, "beyond 4 P-cores", fontsize=8, color=MUTED, va="top")
    for i, s in enumerate(shapes[:8]):
        xs = [t for t in threads if (s, t) in agg]
        base = agg[(s, xs[0])]["median_ms"]
        speed = [base / agg[(s, t)]["median_ms"] for t in xs]
        ax.plot(xs, speed, color=PALETTE[i], marker=MARKERS[i], markersize=5, linewidth=2, label=shape_label(s))
        ax.annotate(shape_label(s), (xs[-1], speed[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=TEXT)
    ax.set_xticks(threads)
    ax.set_xlim(right=threads[-1] * 1.3)
    ax.set_ylim(bottom=0)
    style_axes(ax, title, "threads (1 = no thread creation)", "speedup relative to 1 thread")
    ax.legend(loc="upper left", fontsize=8)


def chart_threads(parallel_rows, micro_rows, out):
    if not parallel_rows:
        return
    panels = 2 if micro_rows else 1
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 5), squeeze=False)
    _thread_panel(axes[0][0], parallel_rows, "parallel (threaded blocked kernel)")
    if micro_rows:
        _thread_panel(axes[0][1], micro_rows, "micro_parallel (threaded micro-kernel)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "thread_scaling.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.results, "charts")
    os.makedirs(out, exist_ok=True)
    suite = read(os.path.join(a.results, "suite_best.csv"))
    name = "suite_best.csv"
    if not suite:
        suite = read(os.path.join(a.results, "suite_tile32.csv"))
        name = "suite_tile32.csv"
    suite += read(os.path.join(a.results, "suite_micro.csv"))
    chart_latency(suite, out, name + " + suite_micro.csv")
    chart_tiles(read(os.path.join(a.results, "tile_sweep.csv")),
                read(os.path.join(a.results, "micro_tile_sweep.csv")), out)
    chart_threads(read(os.path.join(a.results, "thread_scaling.csv")),
                  read(os.path.join(a.results, "micro_thread_scaling.csv")), out)
    print("charts written to", out)


if __name__ == "__main__":
    main()
