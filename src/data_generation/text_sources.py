#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Dict

try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None


def load_custom_short(n: int, pos_words: List[str], neg_words: List[str]) -> List[Dict[str, str]]:
    import random
    random.seed(42)
    templates = [
        "The new system demonstrates {pos1} potential but {neg1} issues remain. Overall it feels {pos2} yet sometimes {neg2}.",
        "An {pos1} performance with {pos2} features, though {neg1} aspects and {neg2} usability problems exist.",
        "User feedback highlights {pos1} design and {pos2} execution, despite {neg1} reliability and {neg2} support.",
    ]
    texts = []
    for i in range(n):
        p = random.sample(pos_words, min(2, len(pos_words)))
        ng = random.sample(neg_words, min(2, len(neg_words)))
        t = templates[i % len(templates)].format(pos1=p[0], pos2=p[-1], neg1=ng[0], neg2=ng[-1])
        texts.append({"title": f"Custom Short #{i:03d}", "content": t})
    return texts


def load_custom_long(n: int, num_positive: int, num_negative: int, pos_words: List[str], neg_words: List[str]) -> List[Dict[str, str]]:
    import random
    random.seed(42)

    templates = [
        (
            "The new system demonstrates {pos1} potential with a {pos2} design architecture. "
            "However, the initial rollout revealed {neg1} implementation issues and {neg2} user experience problems. "
            "While the core functionality remains {pos3}, backend performance shows {neg3} bottlenecks. "
            "The interface design is {pos4} yet suffers from {neg4} navigation flows. "
            "Overall, this {pos5} platform contains {neg5} fundamental limitations."
        ),
        (
            "Our comprehensive review identified {pos1} performance metrics and {pos2} feature implementations. "
            "Unfortunately, customer support proved {neg1} and documentation remains {neg2}. "
            "The build quality demonstrates {pos3} engineering, representing {pos4} technical achievement, "
            "yet software integration feels {neg3} with {neg4} compatibility issues. "
            "This {pos5} solution faces {neg5} deployment challenges."
        ),
        (
            "Analysis reveals {pos1} innovation combined with {pos2} technical execution. "
            "Despite these strengths, users report {neg1} reliability problems and {neg2} workflow disruptions. "
            "The development team delivered {pos3} functionality while creating {neg3} maintenance burdens. "
            "User feedback highlights {pos4} interface elements alongside {neg4} performance bottlenecks. "
            "This {pos5} product exhibits {neg5} scalability constraints."
        ),
    ]

    def pad(lst: List[str], target: int) -> List[str]:
        if len(lst) >= target:
            return lst[:target]
        return (lst * (target // max(1, len(lst)) + 1))[:target]

    texts: List[Dict[str, str]] = []
    for i in range(n):
        template = templates[i % len(templates)]
        sp = pad(random.sample(pos_words, min(num_positive, len(pos_words))), 5)
        sn = pad(random.sample(neg_words, min(num_negative, len(neg_words))), 5)
        content = template.format(pos1=sp[0], pos2=sp[1], pos3=sp[2], pos4=sp[3], pos5=sp[4],
                                  neg1=sn[0], neg2=sn[1], neg3=sn[2], neg4=sn[3], neg5=sn[4])
        texts.append({"title": f"Controlled Sentiment Analysis #{i+1:02d}", "content": content})
    return texts


def load_imdb(n: int) -> List[Dict[str, str]]:
    if load_dataset is None:
        raise RuntimeError("datasets not installed; pip install datasets")
    ds = load_dataset("imdb", split="train")
    pos = ds.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(max(1, n // 2)))
    neg = ds.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(n - len(pos)))
    merged = list(pos) + list(neg)
    return [{"title": f"IMDb Review #{i:03d}", "content": m["text"][:2000], "label": "positive" if int(m["label"]) == 1 else "negative"} for i, m in enumerate(merged)]


def load_squad(n: int) -> List[Dict[str, str]]:
    if load_dataset is None:
        raise RuntimeError("datasets not installed; pip install datasets")
    ds = load_dataset("squad", split="train").shuffle(seed=42).select(range(n))
    return [{"title": f"Q: {ex['question']}", "content": ex["context"][:2000], "qa": {"question": ex["question"], "answers": ex.get("answers", {}).get("text", [])}} for ex in ds]


def load_wikipedia(n: int) -> List[Dict[str, str]]:
    if load_dataset is None:
        raise RuntimeError("datasets not installed; pip install datasets")
    ds = load_dataset("wikipedia", "20220301.en", split="train").shuffle(seed=42)
    texts: List[Dict[str, str]] = []
    for i, ex in enumerate(ds):
        content = ex.get("text", "")
        if not content:
            continue
        texts.append({"title": f"Wikipedia #{i:03d}", "content": content[:2000]})
        if len(texts) >= n:
            break
    return texts



