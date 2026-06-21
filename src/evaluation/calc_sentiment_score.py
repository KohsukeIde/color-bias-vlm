#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


POS_WORDS = {"positive", "positivity", "favorable", "favourable"}
NEG_WORDS = {"negative", "negativity", "unfavorable", "unfavourable"}


def coerce_label(text: str) -> str:
    t = text.strip().lower()
    # simple heuristic: look for explicit polarity word
    for w in POS_WORDS:
        if w in t:
            return "positive"
    for w in NEG_WORDS:
        if w in t:
            return "negative"
    # fallback
    if "neutral" in t:
        return "neutral"
    return "unknown"


def evaluate_sentiment_results(result_json_path: str) -> Dict[str, Any]:
    p = Path(result_json_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    # Attempt to infer ground truth from dataset metadata (IMDb only)
    dataset = data.get("meta", {}).get("dataset", "")

    total = 0
    correct = 0
    per_condition = {}

    for exp in data.get("experiments", []):
        gt_label = None
        if dataset == "imdb":
            # Use title convention or embedded label
            gt_label = exp.get("label") or exp.get("sentiment_label")

        for cond, resp in exp.get("vlm_outputs", {}).items():
            pred = coerce_label(resp if isinstance(resp, str) else json.dumps(resp))
            total += 1
            if gt_label in {"positive", "negative"}:
                correct += int(pred == gt_label)
                per_condition.setdefault(cond, {"n": 0, "correct": 0})
                per_condition[cond]["n"] += 1
                per_condition[cond]["correct"] += int(pred == gt_label)

    acc = correct / total if total else 0.0
    cond_stats = {k: v["correct"] / v["n"] if v["n"] else 0.0 for k, v in per_condition.items()}
    return {"dataset": dataset, "overall_acc": acc, "per_condition_acc": cond_stats, "total": total}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("result_json", type=str)
    args = ap.parse_args()
    out = evaluate_sentiment_results(args.result_json)
    print(json.dumps(out, indent=2))









