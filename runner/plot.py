"""成功率の比較グラフ（レポート・動画用の PNG）。

    python -m runner.plot outputs/main_libero_spatial outputs/main_libero_object --out docs/success.png
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analyze import load, success_table

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {"base": ("VLA only", "#2a78d6"), "monitor": ("VLA + monitor", "#eb6834")}
SUITE = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default="docs/success.png")
    args = ap.parse_args(argv)

    rows = success_table(load(args.dirs)[0])
    groups = sorted({(r["suite"], r["disturb"]) for r in rows}, key=lambda g: (g[1], g[0]))
    idx = {(r["suite"], r["disturb"], r["cond"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width, gap = 0.36, 0.02
    x = np.arange(len(groups))
    for j, cond in enumerate(("base", "monitor")):
        label, color = SERIES[cond]
        xs, ys, lo, hi = [], [], [], []
        for i, (suite, dist) in enumerate(groups):
            r = idx.get((suite, dist, cond))
            if r is None:
                continue
            xs.append(x[i] + (j - 0.5) * (width + gap))
            ys.append(r["rate"] * 100)
            lo.append(r["rate"] * 100 - r["ci95"][0] * 100)
            hi.append(r["ci95"][1] * 100 - r["rate"] * 100)
        ax.bar(xs, ys, width, color=color, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
        for xi, yi, h in zip(xs, ys, hi):
            ax.text(xi, yi + h + 1.5, f"{yi:.0f}%", ha="center", va="bottom", fontsize=8, color=INK, zorder=4)

    ax.set_xticks(x, [f"{SUITE.get(s, s)}\n{'with disturbance' if d else 'no disturbance'}" for s, d in groups],
                  fontsize=8, color=INK)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Success rate (%)", fontsize=9, color=INK2)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, 1.0), ncols=2, labelcolor=INK)
    n = max(r["n"] for r in rows)
    ax.set_title(f"Task success, same initial states (n = {n} episodes per bar, 95% CI)",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(args.out, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
