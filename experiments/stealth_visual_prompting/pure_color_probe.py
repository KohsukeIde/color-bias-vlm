#!/usr/bin/env python3
"""
Pure-color control: embed plain color images (no word) across hue, and compare
to the word+color embeddings. Answers: is the hue effect a word-independent
"pure colour" signal, or a colour x word interaction?

Outputs results/clip_ablation/hue_embedding_viz/pure_color.{npy,csv} and prints
an alignment analysis vs the existing word embeddings.
"""
from __future__ import annotations
import colorsys, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.clip_handler import CLIPHandler, SemanticAxes

HUES = np.linspace(0, 360, 72, endpoint=False)
VIZ = ROOT / "results/clip_ablation/hue_embedding_viz"


def rgb(h):
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 1.0, 0.9)
    return (int(r * 255), int(g * 255), int(b * 255))


def make(h, mode):
    if mode == "fill":
        return Image.new("RGB", (336, 336), rgb(h))
    img = Image.new("RGB", (336, 336), "white")          # colored square on white (matches text-on-white layout)
    ImageDraw.Draw(img).rectangle([108, 138, 228, 198], fill=rgb(h))
    return img


def main():
    clip = CLIPHandler("openai/clip-vit-large-patch14-336", model_type="hf", device="mps")
    axes = SemanticAxes(clip); axes.load_axes(str(VIZ.parent / "phase1/final_adjusted/semantic_axes.json"))
    AX = ["valence", "emotion", "safety", "temperature"]

    out_emb = {}
    for mode in ["fill", "square"]:
        emb = clip.encode_images([make(h, mode) for h in HUES])     # [72,768] L2-normed
        out_emb[mode] = emb
        proj = {a: emb @ axes.get_axis(a) for a in AX}
        df = pd.DataFrame({"hue": HUES, **{f"{a}_projection": proj[a] for a in AX}})
        df.to_csv(VIZ / f"pure_color_{mode}.csv", index=False)
        np.save(VIZ / f"pure_color_{mode}.npy", emb.astype(np.float32))
        print(f"\n=== mode={mode} ===")
        for a in AX:
            v = proj[a]
            i0, i120, i240 = 0, np.argmin(np.abs(HUES - 120)), np.argmin(np.abs(HUES - 240))
            print(f"  {a:11s} red(0)={v[i0]:+.4f}  green(120)={v[i120]:+.4f}  blue(240)={v[i240]:+.4f}  range[{v.min():+.3f},{v.max():+.3f}]")
        # closure + dimensionality of the pure-color loop
        from sklearn.decomposition import PCA
        p = PCA(6).fit(emb); ev = p.explained_variance_ratio_
        print(f"  loop closure cos(355,0)={float(emb[-1] @ emb[0]):.4f}; PCA var PC1-3={ev[:3].sum():.3f} PC1-6={ev[:6].sum():.3f}")

    # ---- alignment: do words' hue-variation live in the pure-colour hue subspace? ----
    print("\n=== alignment: word hue-variation vs pure-colour (fill) hue subspace ===")
    Wemb = np.load(VIZ / "embeddings.npy"); rec = pd.read_csv(VIZ / "records.csv")
    words = list(dict.fromkeys(rec["text"].tolist()))
    from sklearn.decomposition import PCA
    color = out_emb["fill"]
    color_c = color - color.mean(0)
    # k-dim pure-color hue subspace
    for k in (3, 5):
        Bk = PCA(k).fit(color_c).components_           # [k,768]
        fracs = []
        dir_cos = []
        for w in words:
            E = Wemb[rec.index[rec["text"] == w].to_numpy()]
            E = E[np.argsort(rec.loc[rec["text"] == w, "hue"].to_numpy())]
            Ec = E - E.mean(0)
            tot = (Ec ** 2).sum()
            proj = Ec @ Bk.T @ Bk                       # project word hue-variation onto colour subspace
            fracs.append(float((proj ** 2).sum() / tot))
            if k == 3:
                # direction of green-vs-red shift: word vs pure colour
                hue = np.sort(rec.loc[rec["text"] == w, "hue"].unique())
                ir, ig = 0, np.argmin(np.abs(hue - 120))
                dw = E[ig] - E[ir]; dc = color[ig] - color[ir]
                dir_cos.append(float(dw @ dc / (np.linalg.norm(dw) * np.linalg.norm(dc) + 1e-9)))
        print(f"  k={k}: mean fraction of word hue-variance inside pure-colour subspace = {np.mean(fracs):.3f} "
              f"(min {np.min(fracs):.3f} {words[int(np.argmin(fracs))]}, max {np.max(fracs):.3f} {words[int(np.argmax(fracs))]})")
        if k == 3:
            print(f"        mean cos(green-red shift: word vs pure colour) = {np.mean(dir_cos):.3f} "
                  f"(range {np.min(dir_cos):.3f}..{np.max(dir_cos):.3f})")
    print("\nInterpretation hint: high fraction + high cos => the hue effect is largely a word-INDEPENDENT pure-colour signal.")


if __name__ == "__main__":
    main()
