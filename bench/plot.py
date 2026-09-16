#!/usr/bin/env python3
"""Turns bench/results/history.csv (from bench/history.sh) into the SVG
figures under docs/plots/. Standard library only.

  python3 bench/plot.py [bench/results/history.csv] [docs/plots]
"""
import csv
import math
import os
import sys
from collections import defaultdict

SRC = sys.argv[1] if len(sys.argv) > 1 else "bench/results/history.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/plots"

# --- data ------------------------------------------------------------------
rows = defaultdict(dict)  # (milestone, metric) -> {x: value}
with open(SRC) as f:
    for r in csv.DictReader(f):
        try:
            rows[(r["milestone"], r["metric"])][float(r["n"])] = float(r["value"])
        except ValueError:
            pass

MILESTONES = [  # key, label, color (categorical slots 1-4; the sequential baseline is neutral)
    ("p2-parallel-Alg6", "parallel build (Alg. 6)", "#2a78d6"),
    ("p4-advance-pointers", "+ advance pointers", "#eb6834"),
    ("p4b-review-fixes", "+ review fixes", "#1baf7a"),
    ("p5-reserved-arrays", "+ reserved list arrays", "#eda100"),
    ("p5b-compact-lists", "+ compact list storage", "#e87ba4"),
]
BASELINE = ("p1-sequential-Alg2", "sequential build (Alg. 2)", "#52514e")

# --- svg helpers -------------------------------------------------------------
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"


class Fig:
    def __init__(self, w, h, title, subtitle):
        self.w, self.h = w, h
        self.parts = [
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT} font-size='13'>",
            f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>",
            f"<text x='16' y='26' font-size='16' font-weight='600' fill='{INK}'>{title}</text>",
            f"<text x='16' y='45' fill='{INK2}'>{subtitle}</text>",
        ]

    def add(self, s):
        self.parts.append(s)

    def write(self, path):
        self.parts.append("</svg>")
        with open(path, "w") as f:
            f.write("\n".join(self.parts))
        print("wrote", path)


def fmt(v):
    if v == int(v) and v < 1e4: return str(int(v))
    if v >= 100: return f"{v:.0f}"
    if v >= 10: return f"{v:.1f}"
    return f"{v:.2f}"


def nice_linear_ticks(lo, hi):
    raw = (hi - lo) / 5
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=raw)
    t = math.ceil(lo / step) * step
    ticks = []
    while t <= hi * 1.0001:
        ticks.append(round(t, 10))
        t += step
    return ticks


def nice_log_ticks(lo, hi):
    ticks = []
    e = math.floor(math.log10(lo))
    while 10 ** e <= hi * 1.0001:
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                ticks.append(v)
        e += 1
    return ticks


def line_chart(path, title, subtitle, series, xlabel, ylabel, xlog=True, ylog=True, xfmt=fmt, extra=None):
    """series: list of (label, color, dashed, [(x, y)...])"""
    W, H = 760, 440
    L, R, T, B = 64, 210, 70, 56
    fig = Fig(W, H, title, subtitle)
    xs = [x for *_, pts in series for x, _ in pts]
    ys = [y for *_, pts in series for _, y in pts]
    if extra:
        xs += [x for x, _ in extra[3]]; ys += [y for _, y in extra[3]]
    xlo, xhi, ylo, yhi = min(xs), max(xs), min(ys), max(ys)
    if xhi == xlo: xlo, xhi = (xlo / 2, xhi * 2) if xlog else (xlo - 1, xhi + 1)
    if yhi == ylo: ylo, yhi = (ylo / 2, yhi * 2) if ylog else (0, yhi * 2)
    if ylog: ylo, yhi = ylo / 1.6, yhi * 1.6
    else: ylo, yhi = 0, yhi * 1.08
    sx = lambda x: L + (W - L - R) * ((math.log10(x) - math.log10(xlo)) / (math.log10(xhi) - math.log10(xlo)) if xlog else (x - xlo) / (xhi - xlo))
    sy = lambda y: T + (H - T - B) * (1 - ((math.log10(y) - math.log10(ylo)) / (math.log10(yhi) - math.log10(ylo)) if ylog else (y - ylo) / (yhi - ylo)))
    # grid + axes
    yt = nice_log_ticks(ylo, yhi) if ylog else nice_linear_ticks(ylo, yhi)
    for v in yt:
        fig.add(f"<line x1='{L}' x2='{W-R}' y1='{sy(v):.1f}' y2='{sy(v):.1f}' stroke='{GRID}'/>")
        fig.add(f"<text x='{L-8}' y='{sy(v)+4:.1f}' text-anchor='end' fill='{INK2}' font-size='12'>{fmt(v)}</text>")
    xt = sorted(set(xs))  # the measured x values
    for v in xt:
        fig.add(f"<line x1='{sx(v):.1f}' x2='{sx(v):.1f}' y1='{T}' y2='{H-B}' stroke='{GRID}'/>")
        fig.add(f"<text x='{sx(v):.1f}' y='{H-B+18}' text-anchor='middle' fill='{INK2}' font-size='12'>{xfmt(v)}</text>")
    fig.add(f"<text x='{(L+W-R)/2:.0f}' y='{H-14}' text-anchor='middle' fill='{INK2}'>{xlabel}</text>")
    fig.add(f"<text transform='translate(18,{(T+H-B)/2:.0f}) rotate(-90)' text-anchor='middle' fill='{INK2}'>{ylabel}</text>")
    # series; direct labels are placed top-down in the order of the lines' end values
    all_series = ([extra] if extra else []) + series
    all_series.sort(key=lambda s: -sorted(s[3])[-1][1])
    label_y = []
    for label, color, dashed, pts in all_series:
        pts = sorted(pts)
        d = " ".join(f"{'M' if i == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for i, (x, y) in enumerate(pts))
        dash = " stroke-dasharray='6,5'" if dashed else ""
        fig.add(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2'{dash} stroke-linejoin='round'/>")
        for x, y in pts:
            fig.add(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{color}' stroke='{SURFACE}' stroke-width='2'/>")
        # direct label at the right end, nudged apart
        x, y = pts[-1]
        ly = sy(y)
        while any(abs(ly - o) < 15 for o in label_y):
            ly += 15
        label_y.append(ly)
        fig.add(f"<line x1='{sx(x)+6:.1f}' x2='{W-R+10}' y1='{sy(y):.1f}' y2='{ly:.1f}' stroke='{GRID}'/>")
        fig.add(f"<text x='{W-R+14}' y='{ly+4:.1f}' fill='{INK}' font-size='12'>{label} <tspan fill='{INK2}'>{fmt(y)}</tspan></text>")
    fig.write(path)


def bar_chart(path, title, subtitle, groups, unit, vfmt=fmt):
    """groups: list of (group label, [(bar label, color, value)...]); horizontal bars."""
    W = 760
    rows_n = sum(len(b) for _, b in groups)
    H = 70 + 24 * rows_n + 34 * len(groups) + 20
    L, R = 250, 150
    fig = Fig(W, H, title, subtitle)
    vmax = max(v for _, b in groups for *_, v in b)
    sx = lambda v: L + (W - L - R) * v / vmax
    y = 62
    for g, bars in groups:
        if g:
            fig.add(f"<text x='{L}' y='{y+14}' fill='{INK}' font-weight='600'>{g}</text>")
            y += 22
        for label, color, v in bars:
            fig.add(f"<rect x='{L}' y='{y+2}' width='{max(sx(v)-L, 1):.1f}' height='18' rx='3' fill='{color}'/>")
            fig.add(f"<text x='{sx(v)+8:.1f}' y='{y+15}' fill='{INK}' font-size='12'>{vfmt(v)} {unit}</text>")
            fig.add(f"<text x='{L-8}' y='{y+15}' text-anchor='end' fill='{INK2}' font-size='12'>{label}</text>")
            y += 24
        y += 12
    fig.write(path)


os.makedirs(OUT, exist_ok=True)

# 1. construction time vs n, per milestone, sequential as the baseline
series = [(lab, col, False, sorted(rows[(k, "build_seconds")].items())) for k, lab, col in MILESTONES if rows[(k, "build_seconds")]]
base = (BASELINE[1], BASELINE[2], True, sorted(rows[(BASELINE[0], "build_seconds")].items()))
line_chart(os.path.join(OUT, "build_time.svg"), "Construction time",
           "uniform points in [0,1)², L2, α = 4, 14 workers; log-log", series, "points (n)", "seconds",
           xfmt=lambda v: f"{v/1e6:g}M" if v >= 1e6 else f"{v/1e3:g}k", extra=base)

# 2. speedup over the sequential build vs workers, current tree, both modes
seq_1m = rows[(BASELINE[0], "build_seconds")].get(1e6)
if seq_1m:
    sp = []
    for key, lab, col in (("current-adv0", "binary search", "#2a78d6"), ("current-adv1", "advance pointers", "#eb6834")):
        pts = [(w, seq_1m / t) for w, t in rows[(key, "build_seconds_by_workers")].items()]
        if pts: sp.append((lab, col, False, pts))
    ws = sorted(rows[("current-adv0", "build_seconds_by_workers")])
    ideal = ("ideal", BASELINE[2], True, [(w, w) for w in ws])
    line_chart(os.path.join(OUT, "scaling.svg"), "Speedup over the sequential build",
               f"n = 1M, α = 4; sequential Alg. 2 takes {fmt(seq_1m)} s", sp, "workers", "speedup",
               xlog=False, ylog=False, xfmt=lambda v: f"{v:.0f}", extra=ideal)

# 3. query throughput, current tree, both modes
q = []
for metric, name in (("nearest_per_second", "nearest neighbor"), ("knn10_per_second", "10 nearest neighbors")):
    bars = []
    for key, lab, col in (("current-adv0", "binary search (Alg. 3)", "#2a78d6"), ("current-adv1", "advance pointers (Alg. 4)", "#eb6834")):
        v = rows[(key, metric)].get(1e6)
        if v: bars.append((lab, col, v / 1e6))
    if bars: q.append((name, bars))
bar_chart(os.path.join(OUT, "queries.svg"), "Query throughput", "n = 1M, α = 4, 100k queries answered in parallel on 14 workers", q, "M queries/s")

# 4. memory per point at n = 1M, per milestone: peak resident set, and the
# logical size of the lists where recorded
mem, logical = [], []
for k, lab, col in MILESTONES:
    v = rows[(k, "rss_bytes")].get(1e6)
    if v: mem.append((lab, col, v / 1e6 / 1024))
    v = rows[(k, "logical_bytes")].get(1e6)
    if v and k != "p4-advance-pointers":  # that milestone's bench left the pointers out of the size
        logical.append((lab, col, v / 1e6 / 1024))
groups = [("peak resident set", mem)]
if logical: groups.append(("finger lists, logical size", logical))
bar_chart(os.path.join(OUT, "memory.svg"), "Memory per point, parallel build with advance pointers",
          "n = 1M, α = 4 (≈ 50 lists per point)", groups, "KB")
