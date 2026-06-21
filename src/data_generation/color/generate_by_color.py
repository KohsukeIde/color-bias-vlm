#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
import re
import random

from PIL import ImageFont
import argparse

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None  # optional at runtime

try:
    import torch
except ImportError:
    torch = None

try:
    from scipy.spatial.distance import cosine
except ImportError:
    cosine = None

try:
    import numpy as np
except ImportError:
    np = None

# Allow direct execution: add project root to sys.path (the parent of `src`)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.image_utils import render_document_image, choose_font


# Default word lists (can be overridden via config)
DEFAULT_POSITIVE_WORDS = [
    'excellent', 'brilliant', 'fantastic', 'amazing', 'superb',
    'wonderful', 'remarkable', 'outstanding', 'magnificent', 'exceptional'
]
DEFAULT_NEGATIVE_WORDS = [
    'terrible', 'awful', 'horrible', 'disappointing', 'dreadful',
    'useless', 'appalling', 'frustrating', 'devastating', 'miserable'
]


@dataclass
class ColorExperimentConfig:
    dataset_name: str = "custom"
    task_name: str = "sentiment"
    image_size: Tuple[int, int] = (800, 600)
    rgb_threshold_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 255)
    colors: Tuple[str, ...] = ("red", "green", "blue", "yellow", "cyan", "magenta")
    intensities: Tuple[str, ...] = ("subtle", "mild", "strong")
    aa_mode: str = "aa_on"  # or "aa_off"
    font_family: str = "DroidSans"
    title_font_size: int = 40
    body_font_size: int = 28
    include_threshold_singles: bool = False
    include_bipolar: bool = True
    include_contrast: bool = False  # New: Enable contrast experiments
    contrast_delta_values: Tuple[int, ...] = (255, 128, 64, 32, 16, 8, 4, 2, 1, 0)  # Background-text difference
    target_word_categories: Dict[str, List[str]] = field(default_factory=dict)


RGB_MASKS = {
    'red':     (1, 0, 0),
    'green':   (0, 1, 0),
    'blue':    (0, 0, 1),
    'yellow':  (1, 1, 0),
    'cyan':    (0, 1, 1),
    'magenta': (1, 0, 1),
}

def _color_hex_for_channel(color: str, rgb: int) -> str:
    if color not in RGB_MASKS:
        raise ValueError(f"Unsupported color: {color}")
    r_m, g_m, b_m = RGB_MASKS[color]
    r = f"{(rgb * r_m):02x}"
    g = f"{(rgb * g_m):02x}"
    b = f"{(rgb * b_m):02x}"
    return f"#{r}{g}{b}"

def _contrast_hex_for_delta(delta: int) -> str:
    """
    コントラスト実験用のグレースケール色を計算
    背景色: 白 (255, 255, 255)
    文字色: 白から黒へのグレースケール (255-delta, 255-delta, 255-delta)
    
    Args:
        delta: 背景と文字の明度差 (0-255)
    
    Returns:
        文字色のHEX値 (例: "#808080")
    """
    text_rgb = max(0, min(255, 255 - delta))  # Clamp to [0, 255]
    return f"#{text_rgb:02x}{text_rgb:02x}{text_rgb:02x}"


def _intensity_to_hex_base(color: str, intensity: str) -> str:
    base = {
        'subtle': 0x22,
        'mild': 0x55,
        'strong': 0x88,
    }[intensity]
    return _color_hex_for_channel(color, base)


def generate_color_dataset(
    texts: List[Dict[str, str]],
    output_root: str = "data/processed/color",
    config: ColorExperimentConfig = ColorExperimentConfig(),
    raw_root: str | None = None,
) -> str:
    """Generate color-manipulated document images and a config manifest.

    Returns path to the saved configuration JSON.
    """
    out_dir = Path(output_root) / config.task_name / config.dataset_name / config.font_family / config.aa_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    title_font = choose_font([config.font_family], size=config.title_font_size)
    body_font = choose_font([config.font_family], size=config.body_font_size)

    experiments = []
    all_conditions = set()

    bipolar_pairs = [("red", "blue"), ("green", "red"), ("blue", "yellow"), ("cyan", "magenta")]

    # Collect raw entries for saving under data/raw
    raw_entries: List[Dict[str, Any]] = []

    for text_idx, td in enumerate(texts):
        title = td['title']
        content = td['content']
        text_dir = out_dir / f"text_{text_idx:03d}"
        text_dir.mkdir(parents=True, exist_ok=True)

        # Use per-text target categories if provided, else config default
        target_words = td.get('target_words', None) or config.target_word_categories

        images = {}

        # Helper: render and save one condition
        def _generate_and_save_image(base_dir: Path, condition_name: str, color_settings: Dict[str, str], baseline_text_color=None) -> str:
            base_dir.mkdir(parents=True, exist_ok=True)
            img = render_document_image(
                title=title,
                text=content,
                condition_name=condition_name,
                color_settings=color_settings,
                target_words=target_words,
                image_size=config.image_size,
                title_font=title_font,
                body_font=body_font,
                aa_mode=config.aa_mode,
                baseline_text_color=baseline_text_color,
                font_family_name=config.font_family,
                auto_fit_body=True,
                vertical_center=True,
                center_align=False,
            )
            path = base_dir / f"{condition_name}.png"
            img.save(path.as_posix())
            return str(path)

        # Always add black-text baseline once
        base_dir = text_dir / "baseline"
        cond = "baseline_black"
        images[cond] = _generate_and_save_image(
            base_dir=base_dir,
            condition_name=cond,
            color_settings={'positive': '#000000', 'negative': '#000000', 'neutral': '#000000'},
            baseline_text_color=(0, 0, 0),
        )
        all_conditions.add(cond)

        # Note: contrast threshold variants are handled in the Contrast axis; omit here

        # 全てのカテゴリに対して色付けループを実行
        for category_key in target_words.keys():
            for color in config.colors:
                for intensity in config.intensities:
                    cond = f"{category_key}_only_{color}_{intensity}"
                    color_settings = {cat: '#000000' for cat in target_words.keys()}
                    color_settings[category_key] = _intensity_to_hex_base(color, intensity)

                    inten_dir = text_dir / intensity
                    images[cond] = _generate_and_save_image(
                        base_dir=inten_dir,
                        condition_name=cond,
                        color_settings=color_settings,
                    )
                    all_conditions.add(cond)

        # Threshold experiments for single-sentiment per channel — optional
        if config.include_threshold_singles:
            thr_dir = text_dir / "threshold"
            for category_key in target_words.keys():
                for color in config.colors:
                    for rgb in config.rgb_threshold_values:
                        cond = f"threshold_{category_key}_only_{color}_rgb{rgb:03d}"
                        hex_c = _color_hex_for_channel(color, rgb)
                        color_settings = {cat: '#000000' for cat in target_words.keys()}
                        color_settings[category_key] = hex_c
                        images[cond] = _generate_and_save_image(
                            base_dir=thr_dir,
                            condition_name=cond,
                            color_settings=color_settings,
                        )
                        all_conditions.add(cond)

        # Bipolar combinations (pos vs neg) — align with stealth_prompt
        if config.include_bipolar and len(list(target_words.keys())) >= 2:
            cat_keys = list(target_words.keys())
            cat1, cat2 = cat_keys[0], cat_keys[1]
            for pos_color, neg_color in bipolar_pairs:
                for intensity in config.intensities:
                    cond = f"bipolar_{cat1}_{pos_color}_vs_{cat2}_{neg_color}_{intensity}"
                    color_settings = {cat: '#000000' for cat in target_words.keys()}
                    color_settings[cat1] = _intensity_to_hex_base(pos_color, intensity)
                    color_settings[cat2] = _intensity_to_hex_base(neg_color, intensity)
                    inten_dir = text_dir / intensity
                    images[cond] = _generate_and_save_image(
                        base_dir=inten_dir,
                        condition_name=cond,
                        color_settings=color_settings,
                    )
                    all_conditions.add(cond)

        # Contrast experiments - grayscale text on white background
        if config.include_contrast:
            contrast_dir = text_dir / "contrast"
            for delta in config.contrast_delta_values:
                cond = f"contrast_delta_{delta:03d}"
                text_color = _contrast_hex_for_delta(delta)
                # All text in the same grayscale color (no word-specific coloring)
                color_settings = {cat: text_color for cat in target_words.keys()} if target_words else {}
                images[cond] = _generate_and_save_image(
                    base_dir=contrast_dir,
                    condition_name=cond,
                    color_settings=color_settings,
                    baseline_text_color=text_color,  # Set baseline text color for non-targeted words
                )
                all_conditions.add(cond)

        exp_entry = {
            'text_id': text_idx,
            'title': title,
            'content': content,
            'target_word_categories': target_words,
            'images': images,
        }
        # Optional metadata passthrough (for evaluation)
        if 'label' in td:
            exp_entry['label'] = td['label']
        if 'qa' in td:
            exp_entry['qa'] = td['qa']

        experiments.append(exp_entry)

        raw_entry = {
            'text_id': text_idx,
            'title': title,
            'content': content,
        }
        if 'label' in exp_entry:
            raw_entry['label'] = exp_entry['label']
        if 'qa' in exp_entry:
            raw_entry['qa'] = exp_entry['qa']
        raw_entries.append(raw_entry)

    # Determine raw dir and save raw texts as JSONL
    if raw_root is None:
        # Derive .../data/raw from output_root if possible
        out_path = Path(output_root).resolve()
        parts = list(out_path.parts)
        if 'data' in parts:
            data_idx = parts.index('data')
            base_data = Path(*parts[:data_idx + 1])
            raw_dir = base_data / 'raw' / config.dataset_name
        else:
            raw_dir = Path('data/raw') / config.dataset_name
    else:
        raw_dir = Path(raw_root) / config.dataset_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl_path = raw_dir / 'texts.jsonl'
    with open(raw_jsonl_path.as_posix(), 'w', encoding='utf-8') as f:
        for obj in raw_entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    manifest = {
        'dataset_name': config.dataset_name,
        'task_name': config.task_name,
        'aa_mode': config.aa_mode,
        'font_family': config.font_family,
        'image_size': list(config.image_size),
        'rgb_threshold_values': list(config.rgb_threshold_values),
        'colors': list(config.colors),
        'intensities': list(config.intensities),
        'conditions': sorted(all_conditions),
        'experiments': experiments,
        'target_word_categories_default': config.target_word_categories,
        'raw_texts_path': raw_jsonl_path.as_posix(),
    }

    cfg_path = (out_dir / "experiment_config.json").as_posix()
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return cfg_path


def find_decoys_simple(
    context: str,
    answer_tokens: set,
    n: int = 5
) -> List[str]:
    """
    簡単なルールベースでデコイ候補を特定する（CLIP embeddingが利用できない場合のフォールバック）
    """
    context_tokens = set(re.findall(r'\b\w+\b', context.lower()))
    
    # Stop words to exclude
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'or', 'but', 'if', 'this', 'they',
        'we', 'you', 'have', 'had', 'his', 'her', 'him', 'them', 'their'
    }
    
    # 候補単語（正解とストップワード以外）
    potential_decoys = list(context_tokens - answer_tokens - stop_words)
    
    # 長さや頻度でフィルタリング（適度な長さの名詞らしい単語を優先）
    filtered_decoys = [word for word in potential_decoys if 3 <= len(word) <= 15]
    
    # ランダムに選択
    random.shuffle(filtered_decoys)
    return filtered_decoys[:n]


def find_decoys_with_clip(
    context: str,
    answer_tokens: set,
    embeddings_dict: dict,
    n: int = 5
) -> Dict[str, List[str]]:
    """
    文脈から、正解トークンと意味的に類似したデコイをCLIP embeddingを用いて特定する
    
    Returns:
        Dict with keys:
        - "top_1": [最も類似度が高い1つのデコイ]
        - "top_n": [類似度上位n個のデコイ]
    """
    # より確実なインポート状態チェック
    try:
        from scipy.spatial.distance import cosine as cosine_func
        import numpy as np_lib
        if not embeddings_dict:
            raise ImportError("No embeddings provided")
    except ImportError:
        print("WARNING: CLIP embeddings or scipy not available, falling back to simple decoy selection")
        simple_decoys = find_decoys_simple(context, answer_tokens, n)
        return {
            "top_1": simple_decoys[:1] if simple_decoys else [],
            "top_n": simple_decoys
        }
    
    # 正解単語の平均embeddingを計算
    answer_embeddings = []
    for token in answer_tokens:
        if token in embeddings_dict:
            answer_embeddings.append(embeddings_dict[token])
    
    if not answer_embeddings:
        print(f"WARNING: No embeddings found for answer tokens {answer_tokens}, falling back to simple decoy selection")
        simple_decoys = find_decoys_simple(context, answer_tokens, n)
        return {
            "top_1": simple_decoys[:1] if simple_decoys else [],
            "top_n": simple_decoys
        }
    
    avg_answer_embedding = np_lib.mean(answer_embeddings, axis=0)
    
    context_tokens = set(re.findall(r'\b\w+\b', context.lower()))
    
    # Stop words to exclude
    stop_words = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'or', 'but', 'if', 'this', 'they',
        'we', 'you', 'have', 'had', 'his', 'her', 'him', 'them', 'their'
    }
    
    # 候補単語（正解とストップワード以外）
    potential_decoys = list(context_tokens - answer_tokens - stop_words)
    
    similarities = []
    for token in potential_decoys:
        if token in embeddings_dict:
            # コサイン類似度を計算（1 - cosine distance）
            sim = 1 - cosine_func(avg_answer_embedding, embeddings_dict[token])
            similarities.append((token, sim))
    
    if not similarities:
        print("WARNING: No embeddings found for potential decoys, falling back to simple decoy selection")
        simple_decoys = find_decoys_simple(context, answer_tokens, n)
        return {
            "top_1": simple_decoys[:1] if simple_decoys else [],
            "top_n": simple_decoys
        }
    
    # 類似度が高い順にソート
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # top-1とtop-nの両方を返す
    decoy_tokens = [token for token, sim in similarities[:n]]
    return {
        "top_1": decoy_tokens[:1] if decoy_tokens else [],
        "top_n": decoy_tokens
    }


def apply_windowing_to_context(context: str, answer_start: int, window_size: int = 300) -> str:
    """
    正解が含まれるように文脈をウィンドウイング処理する
    """
    start_pos = max(0, answer_start - window_size)
    end_pos = min(len(context), answer_start + window_size)
    return context[start_pos:end_pos].strip()


def create_example_texts(n: int = 3, num_positive: int = 5, num_negative: int = 5,
                         pos_words: List[str] = None, neg_words: List[str] = None) -> List[Dict[str, str]]:
    """Generate longer custom texts comparable to stealth_prompt style.

    Each text embeds exactly num_positive and num_negative target words using
    structured multi-sentence templates to increase length and realism.
    """
    import random
    random.seed(42)

    pos_words = pos_words or list(DEFAULT_POSITIVE_WORDS)
    neg_words = neg_words or list(DEFAULT_NEGATIVE_WORDS)

    # Longer templates inspired by prior controlled texts
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

    texts: List[Dict[str, str]] = []
    for i in range(n):
        template = templates[i % len(templates)]
        selected_pos = random.sample(pos_words, min(num_positive, len(pos_words)))
        selected_neg = random.sample(neg_words, min(num_negative, len(neg_words)))
        # Ensure indices up to 5 exist (pad by cycling if fewer requested)
        def pad(lst: List[str], target: int) -> List[str]:
            if len(lst) >= target:
                return lst[:target]
            cyc = (lst * (target // max(1, len(lst)) + 1))[:target]
            return cyc

        sp = pad(selected_pos, 5)
        sn = pad(selected_neg, 5)
        content = template.format(
            pos1=sp[0], pos2=sp[1], pos3=sp[2], pos4=sp[3], pos5=sp[4],
            neg1=sn[0], neg2=sn[1], neg3=sn[2], neg4=sn[3], neg5=sn[4]
        )
        texts.append({
            'title': f'Controlled Sentiment Analysis #{i+1:02d}',
            'content': content,
        })
    return texts


def create_short_example_texts(n: int = 3, num_positive: int = 2, num_negative: int = 2,
                               pos_words: List[str] = None, neg_words: List[str] = None) -> List[Dict[str, str]]:
    """Generate shorter custom texts (earlier concise style)."""
    import random
    random.seed(42)

    pos_words = pos_words or list(DEFAULT_POSITIVE_WORDS)
    neg_words = neg_words or list(DEFAULT_NEGATIVE_WORDS)

    templates = [
        "The new system shows {pos1} potential but suffers from {neg1} issues. Overall {pos2} yet sometimes {neg2}.",
        "An {pos1} performance with {pos2} features, though {neg1} aspects and {neg2} usability problems exist.",
        "User feedback highlights {pos1} design and {pos2} execution, despite {neg1} reliability and {neg2} support.",
    ]

    texts: List[Dict[str, str]] = []
    for i in range(n):
        template = templates[i % len(templates)]
        sp = random.sample(pos_words, min(num_positive, len(pos_words)))
        sn = random.sample(neg_words, min(num_negative, len(neg_words)))
        # pad to 2 if needed
        while len(sp) < 2:
            sp += sp
        while len(sn) < 2:
            sn += sn
        content = template.format(pos1=sp[0], pos2=sp[1], neg1=sn[0], neg2=sn[1])
        texts.append({
            'title': f'Controlled Short Text #{i+1:02d}',
            'content': content,
        })
    return texts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate color-manipulated document images from various datasets")
    parser.add_argument("--dataset", type=str, default=None, choices=["custom", "imdb", "squad", "custom_short", "custom_long"], help="[Deprecated] Single dataset selector. Prefer --datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, choices=["custom_short", "custom_long", "imdb", "squad"], help="Datasets to generate together, e.g., --datasets custom_short custom_long squad")
    parser.add_argument("--all-datasets", action="store_true", default=False, help="(Deprecated) No-op; use --datasets instead")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--output-root", type=str, default="data/processed/color", help="Output root directory")
    parser.add_argument("--num-positive", type=int, default=5, help="Custom dataset: number of positive words per text")
    parser.add_argument("--num-negative", type=int, default=5, help="Custom dataset: number of negative words per text")
    parser.add_argument("--custom-style", type=str, default="long", choices=["long", "short"], help="Custom text style")
    parser.add_argument("--aa-mode", type=str, default="aa_on", choices=["aa_on", "aa_off"], help="Anti-aliasing mode")
    parser.add_argument("--font-family", type=str, default="DroidSans", help="Preferred font family name")
    parser.add_argument("--title-font-size", type=int, default=40)
    parser.add_argument("--body-font-size", type=int, default=28)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--include-threshold", action="store_true", help="Include single-sentiment threshold variants (color intensity grid)")
    parser.add_argument("--squad-exp-modes", type=str, nargs="+", 
                       default=["answer_span", "decoy_top1"], 
                       choices=["answer_span", "decoy_top1", "decoy_topn", "bipolar"],
                       help="SQuAD experiments to generate (multiple modes will create independent datasets)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parents[3]  # src/data_generation/color -> stealth_visual_prompting

    # Decide which datasets to generate
    datasets_to_generate: List[str]
    if args.datasets:
        datasets_to_generate = list(dict.fromkeys(args.datasets))
    elif args.dataset is not None:
        # Interpret single selection; allow custom style split
        if args.dataset in ("custom", "custom_short") and args.custom_style == "short":
            datasets_to_generate = ["custom_short"]
        elif args.dataset in ("custom", "custom_long") and args.custom_style == "long":
            datasets_to_generate = ["custom_long"]
        else:
            datasets_to_generate = [args.dataset]
    else:
        # Default: generate sentiment (short/long) and QA together (squad includes decoy experiments)
        datasets_to_generate = ["custom_short", "custom_long", "squad"]

    out_root = args.output_root or (base / "data/processed/color").as_posix()

    # Iterate generation per dataset
    for ds_sel in datasets_to_generate:
        # Build texts per dataset and map to task/categories
        if ds_sel == "custom_short":
            texts = create_short_example_texts(
                n=args.n_samples,
                num_positive=max(1, min(args.num_positive, 2)),
                num_negative=max(1, min(args.num_negative, 2)),
                pos_words=list(DEFAULT_POSITIVE_WORDS),
                neg_words=list(DEFAULT_NEGATIVE_WORDS),
            )
            task_name = "sentiment"
            target_categories = {
                'positive': list(DEFAULT_POSITIVE_WORDS),
                'negative': list(DEFAULT_NEGATIVE_WORDS),
            }
        elif ds_sel == "custom_long":
            texts = create_example_texts(
                n=args.n_samples,
                num_positive=args.num_positive,
                num_negative=args.num_negative,
                pos_words=list(DEFAULT_POSITIVE_WORDS),
                neg_words=list(DEFAULT_NEGATIVE_WORDS),
            )
            task_name = "sentiment"
            target_categories = {
                'positive': list(DEFAULT_POSITIVE_WORDS),
                'negative': list(DEFAULT_NEGATIVE_WORDS),
            }
        elif ds_sel == "imdb":
            if load_dataset is None:
                raise RuntimeError("datasets not installed; pip install datasets")
            ds = load_dataset("imdb", split="train")
            pos = ds.filter(lambda x: x["label"] == 1).shuffle(seed=42).select(range(max(1, args.n_samples // 2)))
            neg = ds.filter(lambda x: x["label"] == 0).shuffle(seed=42).select(range(args.n_samples - len(pos)))
            merged = list(pos) + list(neg)
            texts = [{
                "title": f"IMDb Review #{i:03d}",
                "content": m["text"][:2000],
                "label": "positive" if int(m["label"]) == 1 else "negative",
            } for i, m in enumerate(merged)]
            task_name = "sentiment"
            target_categories = {
                'positive': list(DEFAULT_POSITIVE_WORDS),
                'negative': list(DEFAULT_NEGATIVE_WORDS),
            }
        elif ds_sel == "squad":
            if load_dataset is None:
                raise RuntimeError("datasets not installed; pip install datasets")
            
            # CLIP embeddingの読み込み
            embeddings_path = base / "data" / "processed" / "squad_clip_embeddings.pt"
            squad_embeddings = {}
            
            if embeddings_path.exists() and torch is not None:
                print(f"Loading CLIP embeddings from {embeddings_path}...")
                try:
                    squad_embeddings = torch.load(embeddings_path, map_location='cpu', weights_only=False)
                    print(f"Successfully loaded {len(squad_embeddings)} embeddings.")
                except Exception as e:
                    print(f"WARNING: Failed to load CLIP embeddings ({e}). Using simple decoy selection.")
            else:
                print(f"WARNING: CLIP embeddings not found at {embeddings_path}. Using simple decoy selection.")
                if torch is None:
                    print("NOTE: To use CLIP-based decoy selection, install PyTorch and scipy.")
            
            ds = load_dataset("squad", split="train").shuffle(seed=42).select(range(args.n_samples))
            
            texts = []
            print("Processing SQuAD samples...")
            from tqdm import tqdm
            
            for ex in tqdm(ds, desc="Processing SQuAD entries"):
                ctx = ex.get("context", "")
                q = ex.get("question", "")
                answers_data = (ex.get("answers", {}) or {})
                answers = answers_data.get("text", []) or []
                answer_starts = answers_data.get("answer_start", []) or []
                
                # ウィンドウイング処理: 最初の正解の位置を基準にする
                windowed_content = ctx
                if answer_starts and len(answer_starts) > 0:
                    first_answer_start = answer_starts[0]
                    windowed_content = apply_windowing_to_context(ctx, first_answer_start, window_size=300)
                else:
                    # ウィンドウイングできない場合は先頭2000文字
                    windowed_content = ctx[:2000]
                
                # 正解単語のセットを作成
                answer_tokens = set()
                for a in answers:
                    for w in re.findall(r'\b\w+\b', (a or "").lower()):
                        if w:
                            answer_tokens.add(w)
                
                entry = {
                    "title": f"Q: {q}",
                    "content": windowed_content,
                    "qa": {"question": q, "answers": answers},
                }
                
                target_words_for_entry = {}
                if answer_tokens:
                    target_words_for_entry["answer_span"] = sorted(list(answer_tokens))
                
                # CLIP類似度でデコイを特定
                if answer_tokens:
                    decoy_results = find_decoys_with_clip(windowed_content, answer_tokens, squad_embeddings, n=5)
                    if decoy_results["top_1"]:
                        target_words_for_entry["decoy_top_1"] = decoy_results["top_1"]
                    if decoy_results["top_n"]:
                        target_words_for_entry["decoy_top_n"] = decoy_results["top_n"]
                    
                    # デバッグ情報（少数サンプル時のみ）
                    if (decoy_results["top_1"] or decoy_results["top_n"]) and len(texts) <= 10:
                        print(f"  Sample {len(texts)}: Answer={sorted(answer_tokens)}")
                        print(f"    Top-1 decoy: {decoy_results['top_1']}")
                        print(f"    Top-N decoys: {decoy_results['top_n'][:3]}...")
                
                if target_words_for_entry:
                    entry["target_words"] = target_words_for_entry
                
                texts.append(entry)
            
            task_name = "qa"
            target_categories = {}  # each sample may carry its own target_words
        else:
            texts = create_short_example_texts(
                n=args.n_samples,
                num_positive=max(1, min(args.num_positive, 2)),
                num_negative=max(1, min(args.num_negative, 2)),
                pos_words=list(DEFAULT_POSITIVE_WORDS),
                neg_words=list(DEFAULT_NEGATIVE_WORDS),
            )
            task_name = "sentiment"
            target_categories = {
                'positive': list(DEFAULT_POSITIVE_WORDS),
                'negative': list(DEFAULT_NEGATIVE_WORDS),
            }

        cfg = ColorExperimentConfig(
            dataset_name=ds_sel,
            task_name=task_name,
            image_size=(args.width, args.height),
            aa_mode=args.aa_mode,
            font_family=args.font_family,
            title_font_size=args.title_font_size,
            body_font_size=args.body_font_size,
            include_threshold_singles=args.include_threshold,
            include_bipolar=True,
            target_word_categories=target_categories,
        )

        # SQuADデータセットの場合、実験モードごとに独立したデータセットを生成
        if ds_sel == "squad":
            print(f"📊 Generated {len(texts)} master samples for SQuAD")
            
            # 実行する実験モードごとにループ
            for mode in args.squad_exp_modes:
                print(f"\n✨ Preparing SQuAD experiment for mode: '{mode}'...")
                
                # 各モード専用のテキストリストとターゲットカテゴリを準備
                exp_texts = []
                for original_text in texts:
                    # 元のテキストから情報をコピー
                    new_text = original_text.copy()
                    original_targets = original_text.get('target_words', {})
                    
                    # モードに応じて、色付け対象となるカテゴリをフィルタリング
                    new_target_words = {}
                    if mode == "answer_span" and "answer_span" in original_targets:
                        new_target_words["answer_span"] = original_targets["answer_span"]
                    elif mode == "decoy_top1" and "decoy_top_1" in original_targets:
                        new_target_words["decoy_top_1"] = original_targets["decoy_top_1"]
                    elif mode == "decoy_topn" and "decoy_top_n" in original_targets:
                        new_target_words["decoy_top_n"] = original_targets["decoy_top_n"]
                    elif mode == "bipolar" and "answer_span" in original_targets and "decoy_top_1" in original_targets:
                        # Bipolarでは正解とTop-1デコイの両方を使用
                        new_target_words["answer_span"] = original_targets["answer_span"]
                        new_target_words["decoy_top_1"] = original_targets["decoy_top_1"]
                    
                    # ターゲットが存在する場合のみ、実験リストに追加
                    if new_target_words:
                        new_text['target_words'] = new_target_words
                        exp_texts.append(new_text)

                if not exp_texts:
                    print(f"⚠️ WARNING: No valid samples for mode '{mode}'. Skipping generation.")
                    continue

                print(f"Found {len(exp_texts)} samples for mode '{mode}'")

                # モードごとに独立したConfigとデータ生成を実行
                mode_cfg = ColorExperimentConfig(
                    dataset_name=f"{ds_sel}_{mode}",  # 出力ディレクトリ名をモードごとに変更
                    task_name=task_name,
                    image_size=(args.width, args.height),
                    aa_mode=args.aa_mode,
                    font_family=args.font_family,
                    title_font_size=args.title_font_size,
                    body_font_size=args.body_font_size,
                    include_threshold_singles=args.include_threshold,
                    include_bipolar=(mode == "bipolar"),  # Bipolarモード以外ではBipolar生成を無効化
                    target_word_categories=target_categories,
                )

                cfg_path = generate_color_dataset(exp_texts, output_root=out_root, config=mode_cfg)
                print(f"✅ Saved config for mode '{mode}': {cfg_path}")
        else:
            # 非SQuADデータセットは従来通り
            cfg_path = generate_color_dataset(texts, output_root=out_root, config=cfg)
            print(f"✅ Saved config: {cfg_path}")


# Example:
# python src/data_generation/color/generate_by_color.py --datasets custom_short --n-samples 10
