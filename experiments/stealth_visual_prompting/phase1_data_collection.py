#!/usr/bin/env python3
"""
【フェーズ1】網羅的データ収集スクリプト - 修正版

このスクリプトは以下を実行します：
1. color実験と同じ手法で800x600の読み取り可能な画像を生成
2. CLIP/SigLIPで画像埋め込みを取得
3. 意味軸への投影量を計算
4. VLM OCR精度（文字認識正答率）を評価
5. 全結果をCSVとして保存

使用例:
python experiments/stealth_visual_prompting/phase1_data_collection.py --model-name openai/clip-vit-large-patch14-336 --output-dir results/phase1 --enable-ocr --vlm-model llava-hf/llava-v1.6-mistral-7b-hf
"""

from __future__ import annotations
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import sys
import torch
from PIL import Image, ImageFont
import colorsys

# プロジェクトルートを追加
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.clip_handler import CLIPHandler, SemanticAxes, create_standard_axes
from src.utils.image_utils import render_document_image, choose_font


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """HSV値をHEX色に変換"""
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def calculate_contrast_color(target_delta_e: float) -> str:
    """Delta E値に基づいてグレースケール色を計算"""
    # 背景: 白 (255, 255, 255)
    # 文字色: 白からのDelta E距離に基づく.
    
    if target_delta_e == 0:
        # コントラスト0: 完全に見えない（背景と同じ色）
        return "#FFFFFF"  # 白文字 = 白背景
    
    # Delta E値を0.5-64 → RGB差 5-180にマッピング
    if target_delta_e <= 1:
        rgb_diff = int(target_delta_e * 10)  # 0.5→5, 1→10
    else:
        rgb_diff = int(10 + target_delta_e * 2.5)  # 2→15, 4→20, 8→30, etc.
    
    rgb_diff = max(0, min(180, rgb_diff))  # 0-180に制限
    text_rgb = max(75, 255 - rgb_diff)  # 最低75（濃いグレー）まで
    
    return f"#{text_rgb:02x}{text_rgb:02x}{text_rgb:02x}"


def evaluate_ocr_accuracy(image_path: str, expected_text: str, vlm_handler=None) -> tuple[float, str]:
    """VLMを使ったOCR精度評価"""
    try:
        if vlm_handler is not None:
            img = Image.open(image_path).convert('RGB')
            
            # VLMにOCRプロンプトを送信（color実験と同じ形式）
            prompt = "USER: <image>\nWhat word is written in this image? Please answer with just the word.\nASSISTANT:"
            predicted_text = vlm_handler.generate(img, prompt, temperature=0.0, num_beams=1)
            
            # 正確性判定
            predicted_clean = predicted_text.strip().lower()
            expected_clean = expected_text.strip().lower()
            
            # 完全一致または目標単語が応答に含まれていれば完全認識
            if predicted_clean == expected_clean or expected_clean in predicted_clean:
                accuracy = 1.0
            else:
                accuracy = 0.0
            
            return accuracy, predicted_text
        else:
            return 0.0, ""
            
    except Exception as e:
        print(f"VLM OCR evaluation error for {image_path}: {e}")
        return 0.0, f"ERROR: {str(e)}"


def generate_single_word_image(
    word: str,
    font_size: int,
    hue: Optional[float] = None,
    saturation: Optional[float] = None,
    brightness: Optional[float] = None,
    target_delta_e: Optional[float] = None,
    image_size: tuple = (336, 336)
) -> Image.Image:
    """
    単一単語の画像を生成（シンプル版）
    
    Args:
        word: 表示する単語
        font_size: フォントサイズ
        hue: 色相 (0-360度, Noneならコントラストモード)
        saturation: 彩度 (0-1)
        brightness: 明度 (0-1)
        target_delta_e: 目標コントラスト (Delta E)
        image_size: 画像サイズ
    """
    from PIL import ImageDraw
    
    # 画像を作成
    img = Image.new("RGB", image_size, "#FFFFFF")  # 白背景
    draw = ImageDraw.Draw(img)
    
    # フォントを準備（M1 Mac用に直接パス指定）
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = choose_font(["Helvetica", "Arial"], size=font_size)
    
    # 色設定
    if target_delta_e is not None:
        # コントラストモード: グレースケール
        text_color = calculate_contrast_color(target_delta_e)
    else:
        # 色相モード: HSV色
        text_color = hsv_to_hex(hue, saturation, brightness)
    
    # テキストのサイズを取得
    bbox = draw.textbbox((0, 0), word, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # 中央に配置
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    # テキストを描画
    draw.text((x, y), word, fill=text_color, font=font)
    
    return img


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Comprehensive data collection - Fixed")
    parser.add_argument("--model-type", choices=["hf", "openclip", "siglip"], default="hf")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--output-dir", type=str, default="results/clip_ablation/phase1/main")
    parser.add_argument("--words", nargs="*", default=None, help="Words to test (default: use standard set)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding computation")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR accuracy evaluation (requires VLM)")
    parser.add_argument("--vlm-model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", help="VLM model for OCR evaluation")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to process (for testing)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device for CLIP computation")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 単語リストを準備
    if args.words:
        test_words = args.words
    else:
        # 論文用の最強最小実行可能実験セット（3ペア6単語）
        test_words = [
            "warm", "cold",      # temperature軸
            "safe", "dangerous", # safety軸
            "good", "bad"        # valence軸
        ]
    
    print(f"Testing words: {test_words}")
    
    # CLIPハンドラーを初期化
    print(f"Loading CLIP model: {args.model_name} (type: {args.model_type}) on device: {args.device}")
    clip_handler = CLIPHandler(args.model_name, model_type=args.model_type, device=args.device)
    
    # 意味軸を作成
    print("Creating semantic axes...")
    semantic_axes = create_standard_axes(clip_handler)
    
    # 意味軸を保存
    axes_path = output_dir / "semantic_axes.json"
    semantic_axes.save_axes(str(axes_path))
    print(f"Saved semantic axes to {axes_path}")
    
    # VLMハンドラー（OCR用）
    vlm_handler = None
    if args.enable_ocr and args.vlm_model:
        try:
            from src.models.unified_handler import VLMHandler
            # MPSデバイス設定を確認
            if args.device == "mps" and torch.backends.mps.is_available():
                vlm_handler = VLMHandler(args.vlm_model, device_map="auto")  # MPSで実行
                print(f"VLM will use MPS acceleration")
            else:
                vlm_handler = VLMHandler(args.vlm_model, device_map="cpu")  # CPUフォールバック
                print(f"VLM will use CPU")
            print(f"Loaded VLM for OCR: {args.vlm_model}")
        except Exception as e:
            print(f"Warning: Failed to load VLM ({e}). OCR evaluation disabled.")
    
    # 実験パラメータを定義（CLIP 336x336に最適化）
    config = {
        "image_size": (336, 336),  # CLIPと同じサイズ
        "font_sizes": [24, 32, 40, 48, 56, 64, 72, 80],  # 24-80px (8種類)
        "hue_values": np.linspace(0, 360, 36, endpoint=False).tolist(),  # 10度刻み
        "saturation_values": [0.2, 0.5, 0.8, 1.0],
        "brightness_values": [0.3, 0.5, 0.7, 0.9],
        "contrast_delta_e_values": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 4, 8, 16]  # 相転移分析用（細かい刻み）
    }
    
    # 結果を保存するデータフレーム
    results_path = output_dir / "comprehensive_results.csv"
    
    # 既存結果の読み込み（Resume機能）
    existing_results = []
    if args.resume and results_path.exists():
        try:
            existing_df = pd.read_csv(results_path)
            existing_results = existing_df.to_dict('records')
            print(f"Resuming from {len(existing_results)} existing results")
        except Exception as e:
            print(f"Warning: Failed to load existing results ({e}). Starting fresh.")
    
    # 画像生成とデータ収集
    images_dir = output_dir / "generated_images"
    print("Generating images and collecting embeddings...")
    
    all_results = []
    all_params = []
    
    # 全パラメータ組み合わせを生成
    sample_count = 0
    for word_idx, word in enumerate(test_words):
        word_dir = images_dir / f"word_{word_idx:03d}_{word}"
        
        # 色相変化シリーズ
        for sat in config["saturation_values"]:
            for bright in config["brightness_values"]:
                for font_size in config["font_sizes"]:
                    for hue in config["hue_values"]:
                        if args.sample_size and sample_count >= args.sample_size:
                            break
                        
                        params = {
                            'text': word,
                            'font_size': font_size,
                            'hue': hue,
                            'saturation': sat,
                            'brightness': bright,
                            'target_delta_e': None,
                            'word_dir': word_dir
                        }
                        all_params.append(params)
                        sample_count += 1
                    
                    if args.sample_size and sample_count >= args.sample_size:
                        break
                if args.sample_size and sample_count >= args.sample_size:
                    break
            if args.sample_size and sample_count >= args.sample_size:
                break
        
        # コントラスト変化シリーズ（3つのベース色相）
        for base_hue in [0, 120, 240]:  # 赤、緑、青
            for font_size in config["font_sizes"]:
                for delta_e in config["contrast_delta_e_values"]:
                    if args.sample_size and sample_count >= args.sample_size:
                        break
                    
                    params = {
                        'text': word,
                        'font_size': font_size,
                        'hue': base_hue,
                        'saturation': 0.0,  # コントラストシリーズでは彩度0
                        'brightness': 0.0,  # コントラストシリーズでは明度0
                        'target_delta_e': delta_e,
                        'word_dir': word_dir
                    }
                    all_params.append(params)
                    sample_count += 1
                
                if args.sample_size and sample_count >= args.sample_size:
                    break
            if args.sample_size and sample_count >= args.sample_size:
                break
        
        if args.sample_size and sample_count >= args.sample_size:
            break
    
    print(f"Total parameter combinations: {len(all_params)}")
    
    # バッチ処理
    for i in tqdm(range(0, len(all_params), args.batch_size), desc="Processing batches"):
        batch_params = all_params[i:i + args.batch_size]
        
        # Resume機能: 既存結果をチェック
        filtered_params = []
        for params in batch_params:
            result_key = f"{params['text']}_{params['font_size']}_{params.get('hue', 0)}_{params.get('target_delta_e', 0)}"
            if not any(r.get('result_key') == result_key for r in existing_results):
                filtered_params.append(params)
        
        if not filtered_params:
            continue
        
        # 画像を生成
        batch_images = []
        for params in filtered_params:
            try:
                img = generate_single_word_image(
                    word=params['text'],
                    font_size=params['font_size'],
                    hue=params.get('hue'),
                    saturation=params.get('saturation'),
                    brightness=params.get('brightness'),
                    target_delta_e=params.get('target_delta_e'),
                    image_size=config['image_size']
                )
                batch_images.append(img)
            except Exception as e:
                print(f"Error generating image for {params}: {e}")
                batch_images.append(None)
        
        # バッチでCLIP埋め込みを取得
        valid_images = [img for img in batch_images if img is not None]
        valid_params = [params for params, img in zip(filtered_params, batch_images) if img is not None]
        
        if not valid_images:
            continue
        
        try:
            embeddings = clip_handler.encode_images(valid_images)
        except Exception as e:
            print(f"Error encoding images: {e}")
            continue
        
        # 各画像について処理
        for img, params, embedding in zip(valid_images, valid_params, embeddings):
            
            # 画像を保存
            params['word_dir'].mkdir(parents=True, exist_ok=True)
            img_filename = f"img_{len(all_results):06d}.png"
            img_path = params['word_dir'] / img_filename
            img.save(img_path)
            
            # 結果レコードを作成
            result = {
                'text': params['text'],
                'font_size': params['font_size'],
                'hue': params.get('hue'),
                'saturation': params.get('saturation'),
                'brightness': params.get('brightness'),
                'target_delta_e': params.get('target_delta_e'),
                'image_path': str(img_path),
                'result_key': f"{params['text']}_{params['font_size']}_{params.get('hue', 0)}_{params.get('target_delta_e', 0)}"
            }
            
            # 各意味軸への投影を計算
            for axis_name in semantic_axes.axes.keys():
                axis_vector = semantic_axes.get_axis(axis_name)
                projection = clip_handler.compute_semantic_projection(
                    embedding.reshape(1, -1), axis_vector
                )[0]
                result[f'{axis_name}_projection'] = float(projection)
            
            # OCR精度評価（コントラストシリーズのみ）
            if vlm_handler is not None and params.get('target_delta_e') is not None:
                ocr_accuracy, vlm_response = evaluate_ocr_accuracy(str(img_path), params['text'], vlm_handler)
                result['ocr_accuracy'] = ocr_accuracy
                result['vlm_response'] = vlm_response
                result['conditions'] = f"word={params['text']},font={params['font_size']},contrast={params.get('target_delta_e')}"
            else:
                result['ocr_accuracy'] = None  # 色相シリーズはOCR評価なし
                result['vlm_response'] = None
                result['conditions'] = f"word={params['text']},font={params['font_size']},hue={params.get('hue')}"
            
            all_results.append(result)
    
    # 結果をCSVに保存
    if all_results:
        final_results = all_results + existing_results
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(results_path, index=False)
        print(f"Saved {len(results_df)} results to {results_path}")
        
        # 統計サマリーを出力
        print("\n=== Data Collection Summary ===")
        print(f"Total samples: {len(results_df)}")
        print(f"Unique words: {results_df['text'].nunique()}")
        print(f"Font size range: {results_df['font_size'].min()} - {results_df['font_size'].max()}")
        if 'ocr_accuracy' in results_df.columns:
            print(f"Average OCR accuracy: {results_df['ocr_accuracy'].mean():.3f}")
            print(f"OCR accuracy range: {results_df['ocr_accuracy'].min():.3f} - {results_df['ocr_accuracy'].max():.3f}")
        
        # 意味軸投影の統計
        axis_cols = [col for col in results_df.columns if col.endswith('_projection')]
        if axis_cols:
            print("\nSemantic axis projections (mean ± std):")
            for col in axis_cols:
                axis_name = col.replace('_projection', '')
                mean_proj = results_df[col].mean()
                std_proj = results_df[col].std()
                print(f"  {axis_name}: {mean_proj:.3f} ± {std_proj:.3f}")
        
        # コントラストシリーズの統計
        contrast_data = results_df[results_df['target_delta_e'].notna()]
        if len(contrast_data) > 0:
            print(f"\nContrast series: {len(contrast_data)} samples")
            print(f"Delta E range: {contrast_data['target_delta_e'].min():.1f} - {contrast_data['target_delta_e'].max():.1f}")
            if 'ocr_accuracy' in contrast_data.columns:
                print(f"Contrast OCR accuracy: {contrast_data['ocr_accuracy'].mean():.3f}")
    
    print("\nPhase 1 data collection completed!")


if __name__ == "__main__":
    main()
