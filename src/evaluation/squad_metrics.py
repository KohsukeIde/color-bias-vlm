#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import List, Tuple


def _normalize(text: str) -> str:
    text = text.lower()
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction: str, ground_truths: List[str]) -> int:
    pred = _normalize(prediction or "")
    for gt in ground_truths or []:
        if pred == _normalize(gt or ""):
            return 1
    return 0


def f1_score(prediction: str, ground_truths: List[str]) -> float:
    pred_tokens = (_normalize(prediction or "").split())
    if not ground_truths:
        return 0.0
    best = 0.0
    for gt in ground_truths:
        gt_tokens = _normalize(gt or "").split()
        common = {}
        for t in pred_tokens:
            if t in gt_tokens:
                common[t] = min(pred_tokens.count(t), gt_tokens.count(t))
        num_same = sum(common.values())
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:
            best = max(best, float(pred_tokens == gt_tokens))
            continue
        if num_same == 0:
            continue
        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        best = max(best, f1)
    return best


