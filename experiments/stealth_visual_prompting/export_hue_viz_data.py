#!/usr/bin/env python3
"""
Turn the raw hue-sweep CLIP embeddings into a small JSON for the project page.

Reads results/clip_ablation/hue_embedding_viz/{embeddings.npy, records.csv}
and writes a compact hue_data.json with, per word and per hue:
  - projections onto interpretable semantic axes (for tuning curves + a
    valence x emotion "semantic plane")
  - within-word PCA(2) coords (the raw-embedding trajectory for one word)
  - global PCA(2) coords (all words share one frame, for an overview)
plus per-hue stimulus colors (hsv(h,1.0,0.9) -> hex) matching the rendering.
"""
from __future__ import annotations
import argparse
import colorsys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SHOWN_AXES = [
    ("valence", "Valence", "good", "bad"),
    ("emotion", "Emotion", "happy", "sad"),
    ("safety", "Safety", "safe", "dangerous"),
    ("temperature", "Temperature", "warm", "cold"),
    ("arousal", "Arousal", "calm", "chaotic"),
]


def hsv_hex(h):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 1.0, 0.9)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def r3(x):  # round floats to keep JSON small
    return round(float(x), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="results/clip_ablation/hue_embedding_viz")
    ap.add_argument("--out", default="docs/static/data/hue_data.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    in_dir = root / args.in_dir
    emb = np.load(in_dir / "embeddings.npy")              # [N,768]
    df = pd.read_csv(in_dir / "records.csv")
    assert len(df) == emb.shape[0]

    words = list(dict.fromkeys(df["text"].tolist()))      # preserve order
    hues = sorted(df["hue"].unique().tolist())

    # global PCA: all words share one frame
    gpca = PCA(n_components=2, random_state=0).fit(emb)
    g_xy = gpca.transform(emb)
    g_var = gpca.explained_variance_ratio_.tolist()

    data = {}
    within_var = {}
    for w in words:
        idx = df.index[df["text"] == w].tolist()
        sub = df.loc[idx].sort_values("hue")
        sub_emb = emb[sub.index.to_numpy()]
        # within-word PCA (3 comps) centers on this word so the hue path is visible in 3D
        wpca = PCA(n_components=3, random_state=0).fit(sub_emb)
        w_xyz = wpca.transform(sub_emb)
        within_var[w] = [r3(v) for v in wpca.explained_variance_ratio_]
        rows = []
        for j, (_, r) in enumerate(sub.iterrows()):
            gi = r.name  # original index into emb/g_xy
            proj = {a[0]: r3(r[f"{a[0]}_projection"]) for a in SHOWN_AXES}
            rows.append({
                "hue": int(round(r["hue"])),
                "proj": proj,
                "wx": r3(w_xyz[j, 0]), "wy": r3(w_xyz[j, 1]), "wz": r3(w_xyz[j, 2]),
                "gx": r3(g_xy[gi, 0]), "gy": r3(g_xy[gi, 1]),
            })
        data[w] = rows

    # ---- pure-colour control (no word): a selectable "(colour only)" pseudo-word ----
    COLOR_KEY = "(color only)"
    pc_path = in_dir / "pure_color_fill.npy"
    if pc_path.exists():
        cemb = np.load(pc_path)                       # [72,768], hue order 0,5,...355
        axj = json.load(open(root / "results/clip_ablation/phase1/final_adjusted/semantic_axes.json"))["axes"]
        cpca = PCA(n_components=3, random_state=0).fit(cemb)
        c_xyz = cpca.transform(cemb)
        within_var[COLOR_KEY] = [r3(v) for v in cpca.explained_variance_ratio_]
        crows = []
        for j, h in enumerate(hues):
            proj = {a[0]: r3(float(np.dot(cemb[j], np.asarray(axj[a[0]])))) for a in SHOWN_AXES}
            crows.append({"hue": int(round(h)), "proj": proj,
                          "wx": r3(c_xyz[j, 0]), "wy": r3(c_xyz[j, 1]), "wz": r3(c_xyz[j, 2])})
        data[COLOR_KEY] = crows
        words = words + [COLOR_KEY]
        print(f"Added pure-colour control '{COLOR_KEY}' (PCA3 var {within_var[COLOR_KEY]})")

    def extent(vals):
        return [r3(min(vals)), r3(max(vals))]

    payload = {
        "meta": {
            "model": "openai/clip-vit-large-patch14-336",
            "rendering": "Helvetica, white bg, hsv(h,1.0,0.9), 48px, 336x336",
            "hueStep": int(round(360 / len(hues))),
            "words": words,
            "colorOnlyKey": COLOR_KEY,
            "sentimentWords": ["excellent", "great", "good", "terrible", "awful", "bad"],
            "globalPcaVar": [r3(v) for v in g_var],
            "withinWordPcaVar": within_var,
        },
        "axes": [{"key": k, "label": lbl, "pos": p, "neg": n} for (k, lbl, p, n) in SHOWN_AXES],
        "hueColors": [{"hue": int(round(h)), "hex": hsv_hex(h)} for h in hues],
        "extents": {
            "global": {"x": extent(g_xy[:, 0]), "y": extent(g_xy[:, 1])},
            "valence": extent(df["valence_projection"]),
            "emotion": extent(df["emotion_projection"]),
        },
        "data": data,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = out.stat().st_size / 1024
    print(f"Wrote {out} ({size_kb:.1f} KB) — {len(words)} words x {len(hues)} hues")
    print(f"Global PCA var: {payload['meta']['globalPcaVar']}")
    # quick sanity: valence of 'terrible' at red(0) vs green(120)
    for w in ["terrible", "good", "excellent"]:
        if w in data:
            v0 = next(r["proj"]["valence"] for r in data[w] if r["hue"] == 0)
            v120 = min(data[w], key=lambda r: abs(r["hue"] - 120))["proj"]["valence"]
            print(f"  {w}: valence hue~0(red)={v0:+.4f}  hue~120(green)={v120:+.4f}")


if __name__ == "__main__":
    main()
