#!/usr/bin/env python3
"""
Teaser/concept figure for the project page and blog.

Story (data-grounded): the SAME token string "terrible" rendered in black vs.
green is encoded by CLIP into a representation whose VALENCE projection is more
positive for green than for red, even though the word never changed. Right panel
uses the REAL measured valence-vs-hue values from docs/static/data/hue_data.json.

Clean layout: large readable type, generous margins, no rotated/cut-off labels,
"text color" wording (not "ink").
"""
from __future__ import annotations
import colorsys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = json.load(open(ROOT / "docs/static/data/hue_data.json"))

C_BLACK = "#1f1f1f"
C_GREEN = "#%02x%02x%02x" % tuple(int(c * 255) for c in colorsys.hsv_to_rgb(120 / 360, 1.0, 0.9))
BG = "#ffffff"


def hue_rgb(h):
    return colorsys.hsv_to_rgb((h % 360) / 360, 1.0, 0.9)


def word_card(ax, cx, cy, word, color, sub):
    w, h = 2.05, 1.0
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.04",
                 fc="white", ec="#cfcfcf", lw=1.5, zorder=3))
    ax.text(cx, cy + 0.04, word, ha="center", va="center",
            fontsize=29, fontweight="bold", color=color, zorder=4, family="DejaVu Sans")
    ax.text(cx, cy - h / 2 - 0.17, sub, ha="center", va="top",
            fontsize=11, color="#666", zorder=4)


def binned_valence(word, centers, half=30.0):
    rows = DATA["data"][word]
    hues = np.array([r["hue"] for r in rows], float)
    val = np.array([r["proj"]["valence"] for r in rows], float)
    return np.array([float(val[np.abs(((hues - c + 180) % 360) - 180) <= half].mean()) for c in centers])


def main():
    W, H = 12.6, 5.0
    fig = plt.figure(figsize=(W, H), dpi=200)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    # ---- title ----
    ax.text(0.45, 4.74, "Color shifts a word's CLIP valence",
            fontsize=19, fontweight="bold", color="#1a1a2e", va="top")
    ax.text(0.45, 4.34, "The token string is unchanged; only the text color changes.",
            fontsize=12, color="#555", va="top")

    # ================= stimulus cards =================
    word_card(ax, 1.65, 2.95, "bad", C_BLACK, "black text")
    word_card(ax, 1.65, 1.50, "bad", C_GREEN, "green text")
    ax.text(1.65, 0.52, "same string, only the color changes",
            ha="center", va="center", fontsize=10, color="#888")

    # ================= encoder =================
    ax.add_patch(FancyArrowPatch((2.78, 2.95), (3.75, 2.78), arrowstyle="-|>",
                 mutation_scale=16, color=C_BLACK, lw=2.2, zorder=2))
    ax.add_patch(FancyArrowPatch((2.78, 1.50), (3.75, 2.32), arrowstyle="-|>",
                 mutation_scale=16, color=C_GREEN, lw=2.2, zorder=2))
    ax.add_patch(FancyBboxPatch((3.78, 1.95), 1.62, 1.25,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 fc="#eef1f8", ec="#8a93b5", lw=1.6, zorder=3))
    ax.text(4.59, 2.78, "CLIP", ha="center", fontsize=16, fontweight="bold", color="#3a4a8c")
    ax.text(4.59, 2.46, "image encoder", ha="center", fontsize=10.5, color="#3a4a8c")
    ax.text(4.59, 2.18, "(diagnostic)", ha="center", fontsize=9.5, style="italic", color="#8a93b5")
    ax.add_patch(FancyArrowPatch((5.45, 2.57), (6.5, 2.57), arrowstyle="-|>",
                 mutation_scale=15, color="#555", lw=1.9, zorder=2))
    ax.text(5.97, 2.82, "embedding", ha="center", fontsize=10, color="#666")

    # ================= REAL valence at 6 canonical hues =================
    cx0, cy0, cw, chh = 7.30, 1.35, 4.95, 2.35
    ax.add_patch(FancyBboxPatch((cx0 - 0.62, cy0 - 0.62), cw + 1.05, chh + 1.35,
                 boxstyle="round,pad=0.0,rounding_size=0.05", fc="#fbfbfd", ec="#dddddd", lw=1.3, zorder=1))

    word = "bad"
    centers = [0, 60, 120, 180, 240, 300]
    names = ["red", "yellow", "green", "cyan", "blue", "magenta"]
    val = binned_valence(word, centers)
    xs = cx0 + (np.arange(6) + 0.5) / 6 * cw
    vmin, vmax = val.min() - 0.015, val.max() + 0.015
    def vy(v): return cy0 + (v - vmin) / (vmax - vmin) * chh

    ax.text(cx0 + cw / 2, cy0 + chh + 0.52, 'CLIP valence of "bad" across colors',
            ha="center", fontsize=12.5, fontweight="bold", color="#333")
    ax.plot(xs, [vy(v) for v in val], color="#cdcdcd", lw=1.8, zorder=3)
    for x, v, c, nm in zip(xs, val, centers, names):
        ax.plot([x, x], [cy0, vy(v)], color="#e6e6e6", lw=1.1, zorder=2)
        big = c in (0, 120)
        ax.add_patch(Circle((x, vy(v)), 0.115 if big else 0.08, fc=hue_rgb(c),
                     ec="#222" if big else "#aaa", lw=1.8 if big else 1.1, zorder=6))
        ax.text(x, cy0 - 0.20, nm, ha="center", va="top", fontsize=10.5,
                color="#222" if big else "#9a9a9a", fontweight="bold" if big else "normal")
    # y-axis meaning (no rotated cut-off label): poles inside the panel
    ax.text(cx0 - 0.34, vy(vmax) - 0.02, "more\npositive", ha="right", va="top", fontsize=9, color="#888", linespacing=0.95)
    ax.text(cx0 - 0.34, vy(vmin) + 0.02, "more\nnegative", ha="right", va="bottom", fontsize=9, color="#888", linespacing=0.95)

    # red -> green shift
    rx, gx, ry, gy = xs[0], xs[2], vy(val[0]), vy(val[2])
    ax.add_patch(FancyArrowPatch((rx, ry), (gx, gy), connectionstyle="arc3,rad=-0.30",
                 arrowstyle="-|>", mutation_scale=16, color="#333", lw=2.0, zorder=5))
    ax.text((rx + gx) / 2, cy0 + chh + 0.12, "green reads more positive than red",
            ha="center", va="bottom", fontsize=10, color="#333", style="italic")

    fig.text(0.036, 0.045,
             "Real CLIP-ViT-L/14-336 measurements: mean valence projection at six canonical hues. "
             "Recoloring the word toward green moves it toward positive, although the string is unchanged.",
             fontsize=9, color="#8a8a8a", ha="left")

    outdir = ROOT / "docs/static/images"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "teaser_concept.png", dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(outdir / "teaser_concept.pdf", facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    print("saved", outdir / "teaser_concept.png")
    for nm, v in zip(names, val):
        print(f"  {nm:8s} valence = {v:+.4f}")


if __name__ == "__main__":
    main()
