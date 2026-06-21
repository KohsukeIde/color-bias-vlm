#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
from pathlib import Path
import sys
import argparse
import re
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

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

# Allow direct execution: add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.image_utils import render_document_image, choose_font
from src.data_generation.color.generate_by_color import (
    create_example_texts,
    create_short_example_texts,
    DEFAULT_POSITIVE_WORDS,
    DEFAULT_NEGATIVE_WORDS,
    find_decoys_with_clip,
)


@dataclass
class ContrastConfig:
    dataset_name: str = "custom_short"
    task_name: str = "sentiment"
    image_size: Tuple[int, int] = (800, 600)
    aa_mode: str = "aa_on"
    font_family: str = "DroidSans"
    title_font_size: int = 40
    body_font_size: int = 28
    # Contrast thresholds (text color levels)
    # ギリギリ見えるライン (128-255) を細かく設定
    rgb_threshold_values: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 160, 192, 224, 240, 250, 255)


def generate_contrast_dataset(
    texts: List[Dict[str, Any]],
    output_root: str,
    config: ContrastConfig,
) -> str:
    base_out_dir = Path(output_root) / config.task_name / config.dataset_name / config.font_family / config.aa_mode
    base_out_dir.mkdir(parents=True, exist_ok=True)

    title_font = choose_font([config.font_family], size=config.title_font_size)
    body_font = choose_font([config.font_family], size=config.body_font_size)

    experiments: List[Dict[str, Any]] = []
    all_conditions: List[str] = []

    for text_idx, td in enumerate(texts):
        title = td['title']
        content = td['content']
        
        # For sentiment tasks, use color-like structure (direct text_XXX directories)
        # For QA tasks, use task-specific structure (global/saliency subdirectories)
        if config.task_name == "sentiment":
            text_dir = base_out_dir / f"text_{text_idx:03d}"
        else:  # QA task
            text_dir = base_out_dir / "global" / f"text_{text_idx:03d}"
        
        text_dir.mkdir(parents=True, exist_ok=True)

        images: Dict[str, str] = {}
        
        # テキスト全体のコントラストを変化させる
        for rgb in config.rgb_threshold_values:
            cond = f"global_contrast_rgb_{rgb:03d}"
            img = render_document_image(
                title=title, text=content, condition_name=cond,
                target_words={}, color_settings={}, image_size=config.image_size,
                title_font=title_font, body_font=body_font, aa_mode=config.aa_mode,
                baseline_text_color=(rgb, rgb, rgb), font_family_name=config.font_family,
                auto_fit_body=True, vertical_center=True,
            )
            p = text_dir / f"{cond}.png"
            img.save(p.as_posix())
            images[cond] = p.as_posix()
            all_conditions.append(cond)

        # 実験エントリを追加
        exp_entry = {
            'text_id': text_idx,
            'title': title,
            'content': content,
            'images': images,
            'experiment_type': 'global' if config.task_name == "qa" else None
        }
        if 'label' in td:
            exp_entry['label'] = td['label']
        if 'qa' in td:
            exp_entry['qa'] = td['qa']
        if 'target_words' in td:
            exp_entry['target_word_categories'] = td['target_words']
        experiments.append(exp_entry)

    # --- 実験2: Saliency Competition (QAタスクの場合のみ) ---
    if config.task_name == "qa":
        saliency_out_dir = base_out_dir / "saliency"
        saliency_out_dir.mkdir(parents=True, exist_ok=True)
        
        for text_idx, td in enumerate(texts):
            title = td['title']
            content = td['content']
            text_dir = saliency_out_dir / f"text_{text_idx:03d}"
            text_dir.mkdir(parents=True, exist_ok=True)

            images: Dict[str, str] = {}
            target_words = td.get('target_words', {})
            answer_cat = "answer_span"
            # Top-1デコイを視覚的に顕著な競合相手として使用
            decoy_cat = "decoy_top_1" 

            if answer_cat in target_words and decoy_cat in target_words:
                # ベースラインディレクトリ
                baseline_dir = text_dir / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                
                # Answer Salientディレクトリ
                answer_dir = text_dir / "answer_salient"
                answer_dir.mkdir(parents=True, exist_ok=True)
                
                # Decoy Salientディレクトリ
                decoy_dir = text_dir / "decoy_salient"
                decoy_dir.mkdir(parents=True, exist_ok=True)
                
                # 背景となるテキストのコントラストを段階的に変化させるループ
                for bg_rgb in config.rgb_threshold_values:
                    # 顕著な単語は常に高コントラスト（黒）
                    high_contrast_hex = "#000000"
                    
                    # --- Answer Salient ---
                    cond_answer = f"saliency_answer_salient_bg_{bg_rgb:03d}"
                    # 正解を黒、他（デコイ含む）を背景色にする
                    color_settings_answer = {
                        answer_cat: high_contrast_hex,
                    }
                    img_answer = render_document_image(
                        title=title, text=content, condition_name=cond_answer,
                        target_words=target_words, color_settings=color_settings_answer,
                        image_size=config.image_size, title_font=title_font, body_font=body_font,
                        aa_mode=config.aa_mode, baseline_text_color=(bg_rgb, bg_rgb, bg_rgb),
                        font_family_name=config.font_family, auto_fit_body=True, vertical_center=True
                    )
                    p_answer = answer_dir / f"{cond_answer}.png"
                    img_answer.save(p_answer.as_posix())
                    images[cond_answer] = p_answer.as_posix()
                    all_conditions.append(cond_answer)

                    # --- Decoy Salient ---
                    cond_decoy = f"saliency_decoy_salient_bg_{bg_rgb:03d}"
                    # デコイを黒、他（正解含む）を背景色にする
                    color_settings_decoy = {
                        decoy_cat: high_contrast_hex,
                    }
                    img_decoy = render_document_image(
                        title=title, text=content, condition_name=cond_decoy,
                        target_words=target_words, color_settings=color_settings_decoy,
                        image_size=config.image_size, title_font=title_font, body_font=body_font,
                        aa_mode=config.aa_mode, baseline_text_color=(bg_rgb, bg_rgb, bg_rgb),
                        font_family_name=config.font_family, auto_fit_body=True, vertical_center=True
                    )
                    p_decoy = decoy_dir / f"{cond_decoy}.png"
                    img_decoy.save(p_decoy.as_posix())
                    images[cond_decoy] = p_decoy.as_posix()
                    all_conditions.append(cond_decoy)
                
                # ベースライン（全て高コントラスト）は最高コントラスト（rgb=0）と同じなので、
                # Global実験の結果を参照するか、必要に応じて生成
                # ここでは一応生成しておく
                cond_baseline = "saliency_baseline"
                img_baseline = render_document_image(
                    title=title, text=content, condition_name=cond_baseline, target_words={},
                    color_settings={}, image_size=config.image_size,
                    title_font=title_font, body_font=body_font, aa_mode=config.aa_mode,
                    baseline_text_color=(0, 0, 0), font_family_name=config.font_family,
                    auto_fit_body=True, vertical_center=True
                )
                p_baseline = baseline_dir / f"{cond_baseline}.png"
                img_baseline.save(p_baseline.as_posix())
                images[cond_baseline] = p_baseline.as_posix()
                all_conditions.append(cond_baseline)

            # Saliency実験のexp_entryを追加
            exp_entry = {
                'text_id': text_idx,
                'title': title,
                'content': content,
                'images': images,
                'experiment_type': 'saliency'
            }
            if 'label' in td:
                exp_entry['label'] = td['label']
            if 'qa' in td:
                exp_entry['qa'] = td['qa']
            if 'target_words' in td:
                exp_entry['target_word_categories'] = td['target_words']
            experiments.append(exp_entry)

    manifest = {
        'dataset_name': config.dataset_name,
        'task_name': config.task_name,
        'aa_mode': config.aa_mode,
        'font_family': config.font_family,
        'image_size': list(config.image_size),
        'rgb_threshold_values': list(config.rgb_threshold_values),
        'conditions': all_conditions,
        'experiments': experiments,
    }

    cfg_path = (base_out_dir / "experiment_config.json").as_posix()
    with open(cfg_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return cfg_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate contrast-based (grayscale) document images")
    ap.add_argument("--dataset", type=str, default="custom_short", choices=["custom_short", "custom_long", "squad"], help="Source dataset for contrast experiment")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--output-root", type=str, default="data/processed/contrast")
    ap.add_argument("--aa-mode", type=str, default="aa_on", choices=["aa_on", "aa_off"])
    ap.add_argument("--font-family", type=str, default="DroidSans")
    ap.add_argument("--title-font-size", type=int, default=40)
    ap.add_argument("--body-font-size", type=int, default=28)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    args = ap.parse_args()

    base = Path(__file__).resolve().parents[3]
    out_root = args.output_root or (base / "data/processed/contrast").as_posix()

    # Build texts using the same helpers as color pipeline
    if args.dataset == "custom_short":
        texts = create_short_example_texts(n=args.n_samples, num_positive=2, num_negative=2,
                                           pos_words=list(DEFAULT_POSITIVE_WORDS), neg_words=list(DEFAULT_NEGATIVE_WORDS))
        task_name = "sentiment"
    elif args.dataset == "custom_long":
        texts = create_example_texts(n=args.n_samples, num_positive=5, num_negative=5,
                                     pos_words=list(DEFAULT_POSITIVE_WORDS), neg_words=list(DEFAULT_NEGATIVE_WORDS))
        task_name = "sentiment"
    # elif args.dataset == "imdb":
    #     if load_dataset is None:
    #         raise RuntimeError("datasets not installed; pip install datasets")
    #     ds = load_dataset("imdb", split="train").shuffle(seed=42).select(range(args.n_samples))
    #     texts = [{"title": f"IMDb Review #{i:03d}", "content": r["text"][:2000], "label": "positive" if int(r["label"])==1 else "negative"} for i, r in enumerate(ds)]
    #     task_name = "sentiment"
    else:  # squad
        if load_dataset is None:
            raise RuntimeError("datasets not installed; pip install datasets")
        
        # Load CLIP embeddings for decoy selection
        base = Path(__file__).resolve().parents[4] / "stealth_visual_prompting"
        embeddings_path = base / "data/processed/squad_clip_embeddings.pt"
        print(f"Loading CLIP embeddings from {embeddings_path}...")
        
        squad_embeddings = {}
        if embeddings_path.exists() and torch is not None:
            try:
                squad_embeddings = torch.load(embeddings_path, weights_only=False)
                print(f"Successfully loaded {len(squad_embeddings)} embeddings.")
            except Exception as e:
                print(f"WARNING: Failed to load CLIP embeddings: {e}")
        else:
            print("WARNING: CLIP embeddings not found or torch not available. Using simple decoy selection.")
        
        # Process SQuAD dataset
        ds = load_dataset("squad", split="train").shuffle(seed=42).select(range(args.n_samples))
        print("Processing SQuAD samples...")
        
        texts = []
        for i, ex in enumerate(tqdm(ds, desc="Processing SQuAD entries")):
            context = ex["context"]
            question = ex["question"]
            answers = ex.get("answers", {})
            answer_texts = answers.get("text", [])
            answer_starts = answers.get("answer_start", [])
            
            if not answer_texts or not answer_starts:
                continue
                
            # Window around answer for better image generation
            answer_start = answer_starts[0]
            window_size = 600
            start_pos = max(0, answer_start - window_size // 2)
            end_pos = min(len(context), answer_start + window_size // 2)
            windowed_content = context[start_pos:end_pos]
            
            # Extract answer tokens
            answer_tokens = set()
            for ans_text in answer_texts:
                tokens = re.findall(r'\b\w+\b', ans_text.lower())
                answer_tokens.update(tokens)
            
            print(f"  Sample {i}: Answer={answer_texts}")
            
            # Find decoys using CLIP similarity
            decoy_results = find_decoys_with_clip(windowed_content, answer_tokens, squad_embeddings, n=5)
            
            print(f"    Top-1 decoy: {decoy_results.get('top_1', [])}")
            print(f"    Top-N decoys: {decoy_results.get('top_n', [])}...")
            
            # Prepare target words for differential contrast experiments
            target_words_for_entry = {}
            if answer_tokens:
                target_words_for_entry["answer_span"] = sorted(list(answer_tokens))
            if decoy_results.get('top_1'):
                target_words_for_entry["decoy_top_1"] = decoy_results['top_1']
            if decoy_results.get('top_n'):
                target_words_for_entry["decoy_clip_similar"] = decoy_results['top_n']
            
            entry = {
                "title": f"Q: {question}",
                "content": windowed_content,
                "qa": {
                    "question": question,
                    "answers": answer_texts
                },
                "target_words": target_words_for_entry
            }
            texts.append(entry)
            
        print(f"📊 Generated {len(texts)} SQuAD samples with target words")
        task_name = "qa"

    cfg = ContrastConfig(
        dataset_name=args.dataset,
        task_name=task_name,
        image_size=(args.width, args.height),
        aa_mode=args.aa_mode,
        font_family=args.font_family,
        title_font_size=args.title_font_size,
        body_font_size=args.body_font_size,
    )

    cfg_path = generate_contrast_dataset(texts, output_root=out_root, config=cfg)
    print(f"Saved config: {cfg_path}")

