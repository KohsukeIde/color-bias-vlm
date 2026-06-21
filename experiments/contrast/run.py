#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, DefaultDict, Set
import sys
import argparse
import csv
import time
import random
import numpy as np
import re

from tqdm import tqdm
from PIL import Image
from collections import defaultdict

# Allow direct execution: add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.unified_handler import VLMHandler
from src.evaluation.squad_metrics import exact_match_score, f1_score
from transformers import pipeline as hf_pipeline


def parse_forced_choice_sentiment(text: str) -> Dict[str, float]:
    """
    Parse forced-choice sentiment output (POSITIVE/NEUTRAL/NEGATIVE) 
    and convert to probability scores for compatibility with RoBERTa pipeline.
    """
    text_clean = text.strip().upper()
    
    # Direct classification mapping
    if "POSITIVE" in text_clean:
        return {"positive": 1.0, "neutral": 0.0, "negative": 0.0}
    elif "NEGATIVE" in text_clean:
        return {"positive": 0.0, "neutral": 0.0, "negative": 1.0}
    elif "NEUTRAL" in text_clean:
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}
    else:
        # Fallback: if none of the expected words found, treat as neutral
        return {"positive": 0.0, "neutral": 1.0, "negative": 0.0}


# Unified prompts for different experiment types
SENTIMENT_PROMPT = (
    "USER: <image>\n"
    "Analyze the sentiment of the document. "
    "Provide a summary of its overall tone and identify any key positive or negative themes. "
    "Please base your entire analysis only on the text visible in the image. "
    "Be specific about the emotional tone you perceive.\nASSISTANT:"
)

# Forced-choice sentiment prompt for direct classification
SENTIMENT_PROMPT_FORCED_CHOICE = (
    "USER: <image>\n"
    "Analyze the sentiment of the text in the image. "
    "Your entire response must be one of three words: POSITIVE, NEUTRAL, or NEGATIVE. "
    "Do not provide any other explanation or summary.\n"
    "Sentiment:"
)

QA_PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "Follow the example to read the text in the image and extract the short answer to the question.\n\n"
    "# Example\n"
    "Context in image: 'The Amazon rainforest is the world's largest tropical rainforest, located in South America.'\n"
    "Question: Where is the Amazon rainforest located?\n"
    "Answer: South America\n\n"
    "# Task\n"
    "Question: {question}\n"
    "Answer:"
)



def main():
    proj_root = Path(__file__).resolve().parents[2]

    ap = argparse.ArgumentParser(description="Run contrast experiment inference + metrics (resumable)")
    ap.add_argument("--manifest", type=str, required=True, help="Path to experiment_config.json to evaluate")
    ap.add_argument("--model", type=str, default=None, help="Single model name to run (preferred)")
    ap.add_argument("--models", type=str, nargs="*", default=None, help="[Deprecated] Multiple model names; use --model and call per model")
    ap.add_argument("--run-id", type=str, default=None, help="Optional run id to reuse the same result directory (enables resume)")
    ap.add_argument("--resume", action="store_true", help="Resume within the specified --run-id directory, skipping completed items")
    ap.add_argument("--enable-checkpoint", action="store_true", help="Enable periodic progress checkpointing (disabled by default)")
    ap.add_argument("--checkpoint-interval", type=int, default=0, help="Flush partial outputs/progress to disk every N items (requires --enable-checkpoint)")
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0.0 = greedy, fully deterministic)")
    ap.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (1 = greedy search)")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed for reproducibility")
    ap.add_argument("--prompt-type", type=str, choices=["free_form", "forced_choice", "both"], default="both", 
                    help="Sentiment prompt type: 'free_form' (original), 'forced_choice' (POSITIVE/NEUTRAL/NEGATIVE), or 'both' (run both types)")
    args = ap.parse_args()

    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    task_name = manifest.get('task_name', 'sentiment')
    dataset_name = manifest.get('dataset_name', 'unknown')
    # Determine model(s): prefer single-model execution
    if args.model:
        model_names = [args.model]
    elif args.models:
        if len(args.models) > 1:
            print("WARNING: Multiple models provided. It's recommended to run per-model using --model. Proceeding with the first only.", flush=True)
        model_names = [args.models[0]]
    else:
        raise SystemExit("ERROR: Please specify --model <model_id> for per-model execution")

    # Set global seeds for reproducibility (same as color experiments)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Fallback for older PyTorch versions
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize external sentiment analyzer once (reused across VLMs)
    sentiment_pipe = None
    if task_name == 'sentiment':
        try:
            print("INFO: Initializing sentiment analysis pipeline...", flush=True)
            sentiment_pipe = hf_pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            print("INFO: Sentiment pipeline loaded successfully.", flush=True)
        except Exception as e:
            print(f"WARNING: Could not load cardiffnlp model ({e}). Trying default sentiment-analysis pipeline...", flush=True)
            try:
                sentiment_pipe = hf_pipeline("sentiment-analysis")
                print("INFO: Fallback sentiment pipeline loaded.", flush=True)
            except Exception as e2:
                print(f"WARNING: Failed to initialize any sentiment pipeline ({e2}). Sentiment metrics will be skipped.", flush=True)

    # Prepare result dir
    for model_name in model_names:
        try:
            handler = VLMHandler(model_name)
        except Exception as e:
            print(f"WARNING: Skipping model '{model_name}' due to load failure: {e}", flush=True)
            continue
        model_slug = model_name.replace('/', '_')
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup result directory structure
        if dataset_name == "squad":
            # For SQuAD, determine the primary saliency condition and use appropriate folder name
            has_global = any('global' in path and 'saliency' not in path 
                           for exp in manifest['experiments'] 
                           for path in exp.get('images', {}).values())
            has_answer_salient = any('answer_salient' in path 
                                   for exp in manifest['experiments'] 
                                   for path in exp.get('images', {}).values())
            has_decoy_salient = any('decoy_salient' in path 
                                  for exp in manifest['experiments'] 
                                  for path in exp.get('images', {}).values())
            
            # Determine folder name based on what conditions exist
            if has_answer_salient and has_decoy_salient:
                folder_name = "squad_bipolar"
            elif has_answer_salient:
                folder_name = "squad_answer_span"
            elif has_decoy_salient:
                folder_name = "squad_decoy_salient"
            else:
                folder_name = "squad_global"
            
            result_dir = proj_root / "results" / "contrast" / model_slug / folder_name / f"{run_id}_{folder_name.split('_')[-1]}"
            dataset_name = folder_name  # Update dataset_name for consistency
        else:
            result_dir = proj_root / "results" / "contrast" / model_slug / dataset_name / run_id
        
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "raw_outputs").mkdir(exist_ok=True)
        (result_dir / "analysis").mkdir(exist_ok=True)
        
        print(f"📁 Output directory: {result_dir}")

        # Determine which prompt types to run
        if args.prompt_type == "both":
            prompt_types = ["free_form", "forced_choice"]
        else:
            prompt_types = [args.prompt_type]

        # Load existing outputs if resuming
        outputs = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'task_name': task_name,
            'run_id': run_id,
            'experiments': manifest['experiments'][:],  # Deep copy
            'meta': {
                'total_experiments': len(manifest['experiments']),
                'temperature': args.temperature,
                'seed': args.seed,
                'checkpoint_enabled': args.enable_checkpoint,
                'checkpoint_interval': args.checkpoint_interval,
                'prompt_types': prompt_types,
                'sentiment_prompt_free_form': SENTIMENT_PROMPT,
                'sentiment_prompt_forced_choice': SENTIMENT_PROMPT_FORCED_CHOICE,
                'qa_prompt_template': QA_PROMPT_TEMPLATE,
            }
        }

        # Resume logic
        completed = set()
        if args.resume:
            try:
                existing_file = result_dir / "raw_outputs" / "experiment_results.json"
                if existing_file.exists():
                    with open(existing_file, 'r', encoding='utf-8') as f:
                        existing = json.load(f)
                        for exp in existing.get('experiments', []):
                            text_id = exp.get('text_id')
                            if text_id is not None:
                                for cond in exp.get('vlm_outputs', {}):
                                    completed.add((text_id, cond))
                    print(f"📄 Resuming from {len(completed)} existing outputs")
            except Exception as e:
                print(f"⚠️ Could not load existing results: {e}")

        # Checkpointing setup
        do_checkpoint = args.enable_checkpoint and args.checkpoint_interval > 0
        pending_flush = 0
        def flush_progress():
            try:
                with open((result_dir / "raw_outputs" / "experiment_results.json").as_posix(), 'w', encoding='utf-8') as f:
                    json.dump(outputs, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Count total conditions for progress bar
        total = sum(len(exp.get('images', {})) for exp in outputs['experiments'])
        pbar = tqdm(total=total, desc=f"Contrast Inference ({model_slug})")
        
        # sentiment scoring (per-model aggregation, shared pipeline)
        per_condition_sentiments: DefaultDict[str, List[float]] = defaultdict(list)

        for exp in outputs['experiments']:
            vlm_out = {}
            text_id = exp.get('text_id')
            exp_file = (result_dir / "raw_outputs" / f"exp_{int(text_id):03d}.json") if (do_checkpoint and text_id is not None) else None
            # Preload existing partial outputs for the experiment if resuming
            if args.resume and exp_file and exp_file.exists():
                try:
                    with open(exp_file.as_posix(), 'r', encoding='utf-8') as ef:
                        prev = json.load(ef)
                        vlm_out = prev.get('vlm_outputs', {})
                except Exception:
                    pass

            for cond, img_path in exp.get('images', {}).items():
                key = (text_id, cond)
                if key in completed:
                    pbar.update(1)
                    continue
                try:
                    img = Image.open(img_path).convert('RGB')
                    if task_name == 'qa' and 'qa' in exp:
                        question = exp['qa']['question']
                        prompt = QA_PROMPT_TEMPLATE.format(question=question)
                    else:
                        prompt = SENTIMENT_PROMPT_FORCED_CHOICE if args.prompt_type == "forced_choice" else SENTIMENT_PROMPT
                    text = handler.generate(img, prompt, temperature=args.temperature, num_beams=args.num_beams)
                except Exception as e:
                    text = f"[ERROR] {e}"
                vlm_out[cond] = text
                # Sentiment analysis: use direct parsing for forced-choice, RoBERTa for free-form
                if args.prompt_type == "forced_choice":
                    # Direct parsing of forced-choice output
                    try:
                        lab2p = parse_forced_choice_sentiment(text)
                        pos_prob = lab2p.get("positive", 0.0)
                        neg_prob = lab2p.get("negative", 0.0)
                        signed = pos_prob - neg_prob
                        per_condition_sentiments[cond].append(signed)
                    except Exception as e:
                        print(f"Warning: Failed to parse forced-choice sentiment: {e}")
                elif sentiment_pipe is not None:
                    # Traditional RoBERTa pipeline for free-form responses
                    try:
                        preds = sentiment_pipe(text, truncation=True, return_all_scores=True)[0]
                        lab2p = { (p.get('label') or '').lower(): float(p.get('score') or 0.0) for p in preds }
                        def pick_prob(lmap, aliases):
                            for a in aliases:
                                if a in lmap:
                                    return lmap[a]
                            return 0.0
                        # Robust label mapping
                        pos_prob = pick_prob(lab2p, ["positive", "label_2", "pos", "1"])  # common variants
                        neg_prob = pick_prob(lab2p, ["negative", "label_0", "neg", "0"])  # common variants
                        if pos_prob == 0.0 and neg_prob == 0.0 and preds:
                            top = max(preds, key=lambda x: float(x.get('score') or 0.0))
                            tlab = (top.get('label') or '').lower()
                            tsc = float(top.get('score') or 0.0)
                            if 'pos' in tlab:
                                pos_prob, neg_prob = tsc, 0.0
                            elif 'neg' in tlab:
                                pos_prob, neg_prob = 0.0, tsc
                            else:
                                pos_prob, neg_prob = 0.0, 0.0
                        signed = pos_prob - neg_prob
                        per_condition_sentiments[cond].append(signed)
                    except Exception:
                        pass
                # Mark done and checkpoint periodically
                completed.add(key)
                if do_checkpoint:
                    pending_flush += 1
                    if pending_flush >= args.checkpoint_interval:
                        if exp_file is not None:
                            try:
                                with open(exp_file.as_posix(), 'w', encoding='utf-8') as ef:
                                    json.dump({
                                        'text_id': text_id,
                                        'images': exp.get('images', {}),
                                        'vlm_outputs': vlm_out,
                                    }, ef, ensure_ascii=False, indent=2)
                            except Exception:
                                pass
                        flush_progress()
                        pending_flush = 0
                pbar.update(1)
            exp['vlm_outputs'] = vlm_out
            # Flush at end of this experiment
            if do_checkpoint and exp_file is not None:
                try:
                    with open(exp_file.as_posix(), 'w', encoding='utf-8') as ef:
                        json.dump({
                            'text_id': text_id,
                            'images': exp.get('images', {}),
                            'vlm_outputs': vlm_out,
                        }, ef, ensure_ascii=False, indent=2)
                except Exception:
                    pass
            flush_progress()
        pbar.close()

        # QA metrics per condition
        if task_name == 'qa':
            cond_ems: DefaultDict[str, List[float]] = defaultdict(list)
            cond_f1s: DefaultDict[str, List[float]] = defaultdict(list)
            
            # 誘導エラー率計算用（デコイ条件）
            decoy_induced_errors: DefaultDict[str, List[bool]] = defaultdict(list)
            
            for exp in outputs['experiments']:
                gts = (exp.get('qa', {}) or {}).get('answers', [])
                target_words = exp.get('target_word_categories', {})
                
                # デコイ単語のセット（誘導エラー率計算用）
                decoy_words = set()
                for decoy_key in ['decoy_top_1', 'decoy_top_n']:
                    if decoy_key in target_words:
                        decoy_words.update([w.lower() for w in target_words[decoy_key]])
                
                for cond, pred in (exp.get('vlm_outputs') or {}).items():
                    # 通常のQA評価メトリクス
                    cond_ems[cond].append(exact_match_score(pred, gts))
                    cond_f1s[cond].append(f1_score(pred, gts))
                    
                    # デコイ条件の場合、誘導エラー率を計算
                    is_decoy_condition = (cond.startswith("decoy_top_1_only_") or 
                                        cond.startswith("decoy_top_n_only_") or
                                        cond.startswith("saliency_decoy_salient") or
                                        ("decoy" in dataset_name and cond.startswith("decoy_")))
                    if decoy_words and is_decoy_condition:
                        # VLMの回答にデコイ単語が含まれているかチェック
                        pred_words = set(re.findall(r'\b\w+\b', (pred or '').lower()))
                        is_induced_error = bool(decoy_words & pred_words)  # デコイ単語が回答に含まれる
                        decoy_induced_errors[cond].append(is_induced_error)
            
            outputs['qa_metrics_per_condition'] = {
                cond: {
                    'em_avg': (sum(v)/len(v)) if v else 0.0,
                    'f1_avg': (sum(cond_f1s.get(cond, []))/len(cond_f1s.get(cond, []))) if cond_f1s.get(cond) else 0.0,
                    'n': len(v),
                }
                for cond, v in cond_ems.items()
            }
            
            # デコイ条件がある場合、誘導エラー率メトリクスを追加
            if decoy_induced_errors:
                outputs['induced_error_metrics_per_condition'] = {
                    cond: {
                        'induced_error_rate': (sum(errors)/len(errors)) if errors else 0.0,
                        'n': len(errors),
                    }
                    for cond, errors in decoy_induced_errors.items()
                }
            
            # CSV output for QA metrics
            qa_csv_path = (result_dir / "analysis" / "qa_metrics.csv").as_posix()
            with open(qa_csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(["condition", "em_avg", "f1_avg", "n"])
                for cond, m in outputs['qa_metrics_per_condition'].items():
                    writer.writerow([cond, f"{m['em_avg']:.6f}", f"{m['f1_avg']:.6f}", m['n']])
            
            # CSV output for induced error metrics
            if decoy_induced_errors:
                induced_csv_path = (result_dir / "analysis" / "induced_error_metrics.csv").as_posix()
                with open(induced_csv_path, 'w', newline='', encoding='utf-8') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(["condition", "induced_error_rate", "n"])
                    for cond, m in outputs['induced_error_metrics_per_condition'].items():
                        writer.writerow([cond, f"{m['induced_error_rate']:.6f}", m['n']])

        # Sentiment metrics per condition with bias vs baseline
        elif task_name == 'sentiment' and per_condition_sentiments:
            baseline_key = 'global_contrast_rgb_255'  # Highest contrast (black text) as baseline
            base_vals = per_condition_sentiments.get(baseline_key, [])
            baseline_avg = (sum(base_vals)/len(base_vals)) if base_vals else None
            outputs['sentiment_metrics_per_condition'] = {}
            for cond, vals in per_condition_sentiments.items():
                avg = (sum(vals)/len(vals)) if vals else 0.0
                bias = (avg - baseline_avg) if baseline_avg is not None else None
                outputs['sentiment_metrics_per_condition'][cond] = {
                    'signed_score_avg': avg,
                    'bias_vs_baseline': bias,
                    'n': len(vals),
                }
            outputs['meta']['sentiment_scoring_mode'] = 'pos_prob_minus_neg_prob'
            # CSV
            csv_path = (result_dir / "analysis" / "sentiment_metrics.csv").as_posix()
            with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = csv.writer(cf)
                writer.writerow(["condition", "signed_score_avg", "bias_vs_baseline", "n"])
                for cond, m in outputs['sentiment_metrics_per_condition'].items():
                    bias_val = "" if m['bias_vs_baseline'] is None else f"{m['bias_vs_baseline']:.6f}"
                    writer.writerow([cond, f"{m['signed_score_avg']:.6f}", bias_val, m['n']])

        # Save per model
        out_path = result_dir / "raw_outputs" / "experiment_results.json"
        try:
            with open(out_path.as_posix(), 'w', encoding='utf-8') as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved: {out_path}")
        finally:
            pass

        # Status CSV (same format as color experiments)
        status_csv = (result_dir / "analysis" / "status.csv").as_posix()
        with open(status_csv, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(["text_id", "condition", "done"])
            for exp in outputs['experiments']:
                tid = exp.get('text_id')
                for cond in exp.get('images', {}).keys():
                    key = (tid, cond)
                    writer.writerow([tid, cond, 1 if key in completed else 0])

        print(f"✅ Completed model: {model_name}")


if __name__ == "__main__":
    main()