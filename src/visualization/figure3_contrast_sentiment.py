#!/usr/bin/env python3
"""
Figure 3: Contrast-Based Sentiment Analysis Task Bias

このスクリプトはコントラスト実験のセンチメンタルタスクバイアスを可視化します。
VLMがテキストのコントラスト（視認性）によってセンチメント判断を変化させる効果を示します。

Usage:
    python src/visualization/figure3_contrast_sentiment.py \
        --input-dir results/contrast/ \
        --output-dir figures/ \
        --models "all" \
        --datasets "custom_short,custom_long"
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import warnings
warnings.filterwarnings('ignore')

# 論文品質のプロット設定
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# モデル名の表示用短縮形
MODEL_DISPLAY_NAMES = {
    'llava-hf_llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral',
    'llava-hf_llava-v1.6-vicuna-7b-hf': 'LLaVA-Vicuna',
    'Qwen_Qwen2-VL-7B-Instruct': 'Qwen2-VL',
    'HuggingFaceM4_idefics2-8b': 'IDEFICS2'
}

# コントラスト値をRGB値から0-1スケールに変換
def extract_rgb_value(condition: str) -> int:
    """condition名からRGB値を抽出"""
    if 'global_contrast_rgb_' in condition:
        return int(condition.split('_')[-1])
    return 255  # baseline

def rgb_to_contrast_ratio(rgb_value: int) -> float:
    """RGB値をコントラスト比に変換（白背景想定）"""
    # Weber contrast: (L_background - L_text) / L_background
    # RGB 255 = 白背景, RGB 0 = 黒文字で最大コントラスト
    return (255 - rgb_value) / 255

def find_sentiment_metrics_files(input_dir: str, models: List[str], datasets: List[str]) -> Dict[str, List[str]]:
    """指定されたモデルとデータセットのsentiment_metrics.csvファイルを検索"""
    results = {}
    
    input_path = Path(input_dir)
    for model in models:
        model_files = []
        model_dir = input_path / model
        
        if not model_dir.exists():
            print(f"警告: モデルディレクトリが見つかりません: {model_dir}")
            continue
        
        for dataset in datasets:
            dataset_dirs = list(model_dir.glob(f"{dataset}/*/analysis/sentiment_metrics.csv"))
            model_files.extend([str(f) for f in dataset_dirs])
        
        if model_files:
            results[model] = model_files
        else:
            print(f"警告: {model}の{datasets}でsentiment_metrics.csvが見つかりません")
    
    return results

def load_and_process_data(file_paths: Dict[str, List[str]]) -> pd.DataFrame:
    """全ファイルを読み込み、統合データフレームを作成"""
    all_data = []
    
    for model, files in file_paths.items():
        for file_path in files:
            try:
                df = pd.read_csv(file_path)
                
                # ファイルパスから情報を抽出
                path_parts = Path(file_path).parts
                dataset = path_parts[-4]  # custom_short/custom_long
                run_id = path_parts[-3]   # timestamp
                
                # デコーディング戦略を判定
                decoding = "beam" if "beam" in run_id else "standard"
                
                # データに情報を追加
                df['model'] = model
                df['dataset'] = dataset
                df['run_id'] = run_id
                df['decoding'] = decoding
                df['file_path'] = file_path
                
                # RGB値とコントラスト比を計算
                df['rgb_value'] = df['condition'].apply(extract_rgb_value)
                df['contrast_ratio'] = df['rgb_value'].apply(rgb_to_contrast_ratio)
                
                all_data.append(df)
                
            except Exception as e:
                print(f"エラー: {file_path}の読み込みに失敗しました: {e}")
    
    if not all_data:
        raise ValueError("有効なデータファイルが見つかりませんでした")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"読み込み完了: {len(combined_df)}レコード、{combined_df['model'].nunique()}モデル")
    
    return combined_df

def create_contrast_sentiment_lineplot(df: pd.DataFrame, output_path: str, legend_path: str):
    """コントラスト vs センチメントスコアのライン・プロット（統合版）"""
    
    # 標準デコーディングのデータのみ使用
    plot_df = df[df['decoding'] == 'standard'].copy()
    
    # 1つのグラフに全モデルを表示
    plt.figure(figsize=(12, 8))
    
    models = sorted(plot_df['model'].unique())
    datasets = ['custom_short', 'custom_long']
    
    # Viridisカラーパレットを使用（A2と統一）
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    # 線種の定義
    linestyles = {
        'custom_short': '-',   # 実線
        'custom_long': '--'    # 破線
    }
    
    # マーカーの定義
    markers = {
        'custom_short': 'o',   # 円
        'custom_long': 's'     # 四角
    }
    
    for i, model in enumerate(models):
        model_data = plot_df[plot_df['model'] == model]
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        for dataset in datasets:
            dataset_data = model_data[model_data['dataset'] == dataset]
            
            if len(dataset_data) == 0:
                continue
                
            # コントラスト比でソート
            dataset_data = dataset_data.sort_values('contrast_ratio')
            
            # データセット名を短縮
            dataset_label = dataset.replace('custom_', '').title()
            
            plt.plot(dataset_data['contrast_ratio'], 
                    dataset_data['signed_score_avg'],
                    color=color,
                    linestyle=linestyles[dataset],
                    linewidth=3 if dataset == 'custom_short' else 2,
                    marker=markers[dataset],
                    markersize=6 if dataset == 'custom_short' else 4,
                    label=f"{model_display} ({dataset_label})",
                    alpha=0.9,
                    markeredgewidth=1,
                    markeredgecolor='white')
    
    plt.xlabel('Contrast Ratio (0=invisible, 1=maximum)', fontsize=16)
    plt.ylabel('Sentiment Score\n(negative ← → positive)', fontsize=16)
    plt.title('Contrast Effect on Sentiment Analysis\nAcross VLM Models', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, zorder=1)
    
    # ベースライン（中性）を強調
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1, zorder=2)
    
    # 軸の範囲設定
    plt.xlim(0, 1.05)
    
    # 軸のフォントサイズ調整
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"統合ライン・プロット保存: {output_path}")
    
    # 別途凡例を作成
    create_contrast_legend(models, datasets, legend_path)

def create_contrast_legend(models: List[str], datasets: List[str], legend_path: str):
    """コントラスト実験用の独立した凡例を作成"""
    
    # 凡例専用の図を作成
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')  # 軸を非表示
    
    # Viridisカラーパレットを使用
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    # 線種とマーカーの定義
    linestyles = {
        'custom_short': '-',   # 実線
        'custom_long': '--'    # 破線
    }
    
    markers = {
        'custom_short': 'o',   # 円
        'custom_long': 's'     # 四角
    }
    
    # 凡例用のダミープロット
    legend_elements = []
    
    for i, model in enumerate(models):
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        for dataset in datasets:
            dataset_label = dataset.replace('custom_', '').title()
            
            # ダミーライン作成
            line = plt.Line2D([0], [0], 
                            color=color,
                            linestyle=linestyles[dataset],
                            linewidth=3 if dataset == 'custom_short' else 2,
                            marker=markers[dataset],
                            markersize=8 if dataset == 'custom_short' else 6,
                            markeredgewidth=1,
                            markeredgecolor='white',
                            label=f"{model_display} ({dataset_label})")
            legend_elements.append(line)
    
    # 凡例を作成（2列表示）
    legend = ax.legend(handles=legend_elements, 
                      loc='center',
                      fontsize=12,
                      ncol=2,
                      columnspacing=2,
                      handlelength=3,
                      handletextpad=1)
    
    # 凡例のタイトル
    legend.set_title('Models and Text Types', prop={'size': 14, 'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"凡例保存: {legend_path}")

def create_contrast_bias_heatmap(df: pd.DataFrame, output_path: str):
    """モデル × コントラスト値のバイアス・ヒートマップ"""
    
    # 標準デコーディング、custom_shortデータのみ使用
    plot_df = df[(df['decoding'] == 'standard') & (df['dataset'] == 'custom_short')].copy()
    
    # ピボットテーブル作成
    heatmap_data = plot_df.pivot_table(
        values='bias_vs_baseline',
        index='model',
        columns='rgb_value',
        aggfunc='mean'
    )
    
    # モデル名を表示用に変換
    heatmap_data.index = [MODEL_DISPLAY_NAMES.get(model, model) for model in heatmap_data.index]
    
    # 図を作成
    plt.figure(figsize=(14, 6))
    
    # カラーマップ設定（赤=負のバイアス、青=正のバイアス）
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',  # 赤青反転（赤=負、青=正）
                center=0,
                cbar_kws={'label': 'Sentiment Bias vs Baseline'},
                linewidths=0.5)
    
    plt.title('Sentiment Bias by Text Contrast (RGB Value)\nCustom Short Dataset', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Text RGB Value (0=black, 255=white)', fontsize=12)
    plt.ylabel('VLM Model', fontsize=12)
    
    # X軸ラベルを読みやすく調整
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ヒートマップ保存: {output_path}")

def create_decoding_comparison(df: pd.DataFrame, output_path: str):
    """デコーディング戦略比較（Beam vs Standard）"""
    
    # custom_shortデータのみ使用
    plot_df = df[df['dataset'] == 'custom_short'].copy()
    
    # 4モデル × 2デコーディング戦略
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    models = sorted(plot_df['model'].unique())
    decoding_strategies = ['standard', 'beam']
    colors = ['#E74C3C', '#27AE60']  # 赤とグリーン
    linestyles = ['-', '--']
    
    for i, model in enumerate(models):
        ax = axes[i]
        model_data = plot_df[plot_df['model'] == model]
        
        for j, decoding in enumerate(decoding_strategies):
            decoding_data = model_data[model_data['decoding'] == decoding]
            
            if len(decoding_data) == 0:
                continue
                
            # コントラスト比でソート
            decoding_data = decoding_data.sort_values('contrast_ratio')
            
            ax.plot(decoding_data['contrast_ratio'], 
                   decoding_data['bias_vs_baseline'],
                   color=colors[j], 
                   linestyle=linestyles[j],
                   linewidth=3,
                   marker='o', 
                   markersize=5,
                   label=f"{decoding.title()} Decoding",
                   alpha=0.8)
        
        ax.set_title(f"{MODEL_DISPLAY_NAMES.get(model, model)}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Contrast Ratio', fontsize=12)
        ax.set_ylabel('Sentiment Bias vs Baseline', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # ベースライン
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"デコーディング比較保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='コントラスト・センチメンタルタスク可視化')
    parser.add_argument('--input-dir', required=True, help='結果ディレクトリのパス')
    parser.add_argument('--output-dir', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--models', default='all', 
                       help='対象モデル（カンマ区切りまたは"all"）')
    parser.add_argument('--datasets', default='custom_short,custom_long',
                       help='対象データセット（カンマ区切り）')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # モデル一覧を準備
    if args.models == 'all':
        available_models = [d.name for d in Path(args.input_dir).iterdir() if d.is_dir()]
        models = available_models
    else:
        models = [m.strip() for m in args.models.split(',')]
    
    datasets = [d.strip() for d in args.datasets.split(',')]
    
    print(f"対象モデル: {models}")
    print(f"対象データセット: {datasets}")
    
    # ファイル検索
    file_paths = find_sentiment_metrics_files(args.input_dir, models, datasets)
    
    if not file_paths:
        print("エラー: 有効なデータファイルが見つかりませんでした")
        return
    
    # データ読み込み・処理
    df = load_and_process_data(file_paths)
    
    # 可視化作成
    print("\n図表生成中...")
    
    # 1. コントラスト vs センチメントスコア（統合ライン・プロット）
    create_contrast_sentiment_lineplot(
        df, 
        output_dir / 'figure3a_contrast_sentiment_lineplot.png',
        output_dir / 'figure3a_contrast_sentiment_legend.png'
    )
    
    # 2. バイアス・ヒートマップ
    create_contrast_bias_heatmap(
        df,
        output_dir / 'figure3b_contrast_bias_heatmap.png'
    )
    
    # 3. デコーディング戦略比較
    create_decoding_comparison(
        df,
        output_dir / 'figure3c_decoding_comparison.png'
    )
    
    print(f"\n✅ 全図表生成完了: {output_dir}")
    print("生成ファイル:")
    print("  - figure3a_contrast_sentiment_lineplot.png")
    print("  - figure3b_contrast_bias_heatmap.png") 
    print("  - figure3c_decoding_comparison.png")

if __name__ == "__main__":
    main()
