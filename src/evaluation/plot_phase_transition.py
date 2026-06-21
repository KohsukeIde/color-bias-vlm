#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_results(result_json_path: str) -> Dict[str, Any]:
    p = Path(result_json_path)
    return json.loads(p.read_text(encoding="utf-8"))


def build_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for exp in results.get('experiments', []):
        text_id = exp.get('text_id')
        label = exp.get('label', None)
        for cond, resp in exp.get('vlm_outputs', {}).items():
            rows.append({
                'text_id': text_id,
                'label': label,
                'condition': cond,
                'response': resp,
            })
    return pd.DataFrame(rows)


def parse_threshold_from_condition(cond: str) -> int:
    # e.g., threshold_positive_only_red_rgb032 -> 32
    import re
    m = re.search(r"rgb(\d+)", cond)
    return int(m.group(1)) if m else -1


def coarse_sentiment_label(text: str) -> str:
    t = (text or "").lower()
    if "positive" in t:
        return "positive"
    if "negative" in t:
        return "negative"
    if "neutral" in t:
        return "neutral"
    return "unknown"


def plot_phase_curve(df: pd.DataFrame, save_path: str) -> None:
    # Filter threshold single-sentiment only
    thr_df = df[df['condition'].str.startswith('threshold_')].copy()
    if thr_df.empty:
        print("No threshold conditions found; skipping plot.")
        return

    thr_df['rgb'] = thr_df['condition'].apply(parse_threshold_from_condition)
    thr_df = thr_df[thr_df['rgb'] >= 0]
    thr_df['pred'] = thr_df['response'].apply(coarse_sentiment_label)

    # Define phases vs ground truth if available
    def phase(row):
        gt = row.get('label', None)
        pr = row.get('pred', 'unknown')
        if gt in {'positive', 'negative'} and pr == gt:
            return 'accurate'
        if pr in {'positive', 'negative', 'neutral'} and (gt not in {'positive', 'negative'} or pr != gt):
            return 'neutral' if pr == 'neutral' else 'biased'
        return 'hallucination'

    thr_df['phase'] = thr_df.apply(phase, axis=1)

    # Aggregate per RGB
    agg = thr_df.groupby('rgb')['phase'].value_counts(normalize=True).rename('ratio').reset_index()
    pivot = agg.pivot(index='rgb', columns='phase', values='ratio').fillna(0.0)
    pivot = pivot.sort_index()

    plt.figure(figsize=(8, 5))
    for col in ['accurate', 'neutral', 'biased', 'hallucination']:
        if col in pivot.columns:
            plt.plot(pivot.index, pivot[col], marker='o', label=col)
    plt.xlabel('RGB intensity')
    plt.ylabel('ratio')
    plt.title('Phase transition over RGB')
    plt.xscale('log', base=2)
    plt.xticks(pivot.index, rotation=45)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    print(f"Saved plot: {save_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('result_json', type=str)
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    res = load_results(args.result_json)
    df = build_dataframe(res)
    out = args.out or (str(Path(args.result_json).with_suffix('')) + '_phase_curve.png')
    plot_phase_curve(df, out)









