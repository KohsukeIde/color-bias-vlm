#!/usr/bin/env python3
"""
Hue-sweep embedding collection for the project-page interactive visualization.

Unlike phase1_data_collection.py (which discards the raw 768-d CLIP image
embeddings and keeps only scalar projections), this script PERSISTS the raw
embeddings so we can build a "sweep the hue -> watch the word's embedding move"
visualization, including a real PCA layout of CLIP embedding space.

It reuses the exact rendering (`generate_single_word_image`), the exact model
(openai/clip-vit-large-patch14-336 via CLIPHandler), and the exact semantic
axes (loaded from the existing semantic_axes.json) used to produce the paper's
a2_hue_tuning_curves figure, so the new data is consistent with the paper.

Outputs (under --output-dir):
  embeddings.npy        float32 [N, 768]  L2-normalized CLIP image embeddings
  records.csv           one row per rendered image, same order as embeddings.npy
                        cols: text, hue, saturation, brightness, font_size,
                              <axis>_projection (10 axes)
  run_config.json       parameters used for the run
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.clip_handler import CLIPHandler, SemanticAxes, create_standard_axes
# reuse the EXACT single-word renderer used by phase1
from experiments.stealth_visual_prompting.phase1_data_collection import (
    generate_single_word_image,
)

# 6 paper probe words (define the temperature/safety/valence story) +
# 4 sentiment words used by the teaser ("terrible" colored green -> positive).
DEFAULT_WORDS = [
    "warm", "cold",        # temperature axis
    "safe", "dangerous",   # safety axis
    "good", "bad",         # valence axis
    "excellent", "terrible", "great", "awful",  # sentiment teaser words
]


def main():
    ap = argparse.ArgumentParser(description="Hue-sweep CLIP embedding collection")
    ap.add_argument("--model-name", default="openai/clip-vit-large-patch14-336")
    ap.add_argument("--model-type", default="hf")
    ap.add_argument("--device", default="mps", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--output-dir", default="results/clip_ablation/hue_embedding_viz")
    ap.add_argument("--axes-json",
                    default="results/clip_ablation/phase1/final_adjusted/semantic_axes.json",
                    help="Load existing axes to stay identical to the paper; if missing, recompute.")
    ap.add_argument("--words", nargs="*", default=None)
    ap.add_argument("--hue-steps", type=int, default=72, help="number of hues in [0,360)")
    ap.add_argument("--fonts", nargs="*", type=int, default=[48])
    ap.add_argument("--saturations", nargs="*", type=float, default=[1.0])
    ap.add_argument("--brightnesses", nargs="*", type=float, default=[0.9])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="cap #images (timing test)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    words = args.words if args.words else DEFAULT_WORDS
    hues = np.linspace(0, 360, args.hue_steps, endpoint=False).tolist()

    print(f"Loading CLIP: {args.model_name} on {args.device}")
    clip = CLIPHandler(args.model_name, model_type=args.model_type, device=args.device)

    # axes: load existing to be byte-identical to the paper; fall back to recompute
    axes = SemanticAxes(clip)
    axes_path = _PROJECT_ROOT / args.axes_json
    if axes_path.exists():
        axes.load_axes(str(axes_path))
        print(f"Loaded {len(axes.axes)} semantic axes from {axes_path}")
    else:
        print(f"axes json not found at {axes_path}; recomputing standard axes")
        axes = create_standard_axes(clip)
    axis_names = list(axes.axes.keys())

    # build the full parameter grid (hue series only)
    params = []
    for w in words:
        for s in args.saturations:
            for b in args.brightnesses:
                for f in args.fonts:
                    for h in hues:
                        params.append({"text": w, "hue": float(h),
                                       "saturation": float(s), "brightness": float(b),
                                       "font_size": int(f)})
    if args.limit:
        params = params[:args.limit]
    print(f"Total images to render/encode: {len(params)}")

    all_emb = []
    records = []
    for i in tqdm(range(0, len(params), args.batch_size), desc="batches"):
        batch = params[i:i + args.batch_size]
        imgs = [generate_single_word_image(
                    word=p["text"], font_size=p["font_size"], hue=p["hue"],
                    saturation=p["saturation"], brightness=p["brightness"],
                    target_delta_e=None, image_size=(336, 336))
                for p in batch]
        emb = clip.encode_images(imgs)  # [B,768] L2-normalized
        for p, e in zip(batch, emb):
            rec = dict(p)
            for an in axis_names:
                rec[f"{an}_projection"] = float(np.dot(e, axes.get_axis(an)))
            records.append(rec)
            all_emb.append(e.astype(np.float32))

    emb_arr = np.stack(all_emb, axis=0)
    np.save(out / "embeddings.npy", emb_arr)
    df = pd.DataFrame(records)
    df.to_csv(out / "records.csv", index=False)
    with open(out / "run_config.json", "w") as fh:
        json.dump({"model_name": args.model_name, "words": words,
                   "hue_steps": args.hue_steps, "fonts": args.fonts,
                   "saturations": args.saturations, "brightnesses": args.brightnesses,
                   "axis_names": axis_names, "n": int(emb_arr.shape[0]),
                   "embedding_dim": int(emb_arr.shape[1])}, fh, indent=2)

    print(f"Saved embeddings {emb_arr.shape} -> {out/'embeddings.npy'}")
    print(f"Saved {len(df)} records -> {out/'records.csv'}")


if __name__ == "__main__":
    main()
