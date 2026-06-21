#!/usr/bin/env python3
"""
Figure 1 Extended: Multi-Model Color Bias Comparison

複数のVLMモデルでの色彩バイアス効果を比較するための拡張版Figure 1を生成します。
モデル間での一般性と個別の特性を可視化します。

Usage:
    python src/visualization/figure1_multi_model.py \
        --input-dir results/color/ \
        --output-dir figures/ \
        --models "llava-hf_llava-v1.6-mistral-7b-hf,Qwen_Qwen2-VL-7B-Instruct" \
        --dataset custom_short
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

# モデル名の表示用短縮形
MODEL_DISPLAY_NAMES = {
    'llava-hf_llava-v1.6-mistral-7b-hf': 'LLaVA-Mistral',
    'llava-hf_llava-v1.6-vicuna-7b-hf': 'LLaVA-Vicuna',
    'Qwen_Qwen2-VL-7B-Instruct': 'Qwen2-VL',
    'HuggingFaceM4_idefics2-8b': 'IDEFICS2'
}

# 色の定義（実験で使用する実際の色）
# 読者が直感的に理解できるよう、実験条件と同じ色を使用
COLOR_MAPPING = {
    'red': '#E74C3C',      # 明確な赤（実験で使用する赤と対応）
    'green': '#27AE60',    # 明確な緑（実験で使用する緑と対応）
    'blue': '#3498DB',     # 明確な青（実験で使用する青と対応）
    'yellow': '#F39C12',   # 視認性の良い黄（実験で使用する黄と対応）
    'cyan': '#1ABC9C',     # 鮮やかなシアン（実験で使用するシアンと対応）
    'magenta': '#E91E63'   # 鮮やかなマゼンタ（実験で使用するマゼンタと対応）
}

def find_sentiment_metrics_files(input_dir: str, models: List[str], dataset: str) -> Dict[str, str]:
    """
    指定されたモデルとデータセットのsentiment_metrics.csvファイルを検索
    """
    files_found = {}
    
    for model in models:
        # パターンマッチングで最新の結果を検索
        pattern = f"{input_dir}/{model}/{dataset}/*/analysis/sentiment_metrics.csv"
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # 最新のファイルを使用（ディレクトリ名でソート）
            latest_file = sorted(matching_files)[-1]
            files_found[model] = latest_file
            print(f"Found {model}: {latest_file}")
        else:
            print(f"WARNING: No sentiment_metrics.csv found for {model}/{dataset}")
    
    return files_found

def load_and_process_multi_model_data(files_dict: Dict[str, str]) -> pd.DataFrame:
    """
    複数モデルのデータを読み込み、統合データフレームを作成
    """
    all_data = []
    
    for model, file_path in files_dict.items():
        try:
            df = pd.read_csv(file_path)
            df['model'] = model
            df['model_display'] = MODEL_DISPLAY_NAMES.get(model, model)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data files found")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 条件名を解析
    def parse_condition(condition: str) -> Dict[str, str]:
        parts = condition.split('_')
        
        if 'positive_only' in condition:
            category = 'Positive Words'
            color = parts[2]
            intensity = parts[3] if len(parts) > 3 else 'strong'
        elif 'negative_only' in condition:
            category = 'Negative Words'
            color = parts[2] 
            intensity = parts[3] if len(parts) > 3 else 'strong'
        else:
            return None  # Skip other conditions for main comparison
            
        return {
            'category': category,
            'color': color,
            'intensity': intensity
        }
    
    # 条件解析
    parsed_data = []
    for _, row in combined_df.iterrows():
        parsed = parse_condition(row['condition'])
        if parsed:
            parsed.update({
                'model': row['model'],
                'model_display': row['model_display'],
                'bias_vs_baseline': row['bias_vs_baseline'],
                'std_sentiment_score': row['std_sentiment_score'],
                'n': row['n']
            })
            parsed_data.append(parsed)
    
    return pd.DataFrame(parsed_data)

def create_multi_model_comparison(df: pd.DataFrame, output_path: str) -> str:
    """
    複数モデルの比較プロット作成
    """
    # 強度がstrongのデータのみを使用（メインの効果に集中）
    main_df = df[df['intensity'] == 'strong'].copy()
    
    # サブプロット設定（モデル数に応じて）
    n_models = len(main_df['model_display'].unique())
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8), sharey=True)
    
    if n_models == 1:
        axes = [axes]
    
    colors_to_plot = ['red', 'green', 'blue']  # 主要な色に絞る
    
    for idx, model in enumerate(sorted(main_df['model_display'].unique())):
        ax = axes[idx]
        model_data = main_df[main_df['model_display'] == model]
        
        # カテゴリごとのデータ準備
        x_pos = []
        y_values = []
        y_errors = []
        bar_colors = []
        x_labels = []
        
        pos = 0
        for category in ['Positive Words', 'Negative Words']:
            cat_data = model_data[model_data['category'] == category]
            
            for color in colors_to_plot:
                color_data = cat_data[cat_data['color'] == color]
                
                if len(color_data) > 0:
                    bias = color_data['bias_vs_baseline'].iloc[0]
                    std_err = color_data['std_sentiment_score'].iloc[0] / np.sqrt(color_data['n'].iloc[0])
                    
                    x_pos.append(pos)
                    y_values.append(bias)
                    y_errors.append(std_err)
                    bar_colors.append(COLOR_MAPPING[color])
                    x_labels.append(f'{category}\n({color.title()})')
                    
                    pos += 1
            
            pos += 0.5  # カテゴリ間のスペース
        
        # バープロット
        bars = ax.bar(x_pos, y_values, yerr=y_errors, capsize=5,
                     color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Y=0の基準線
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # 軸設定
        ax.set_title(f'{model}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
        
        if idx == 0:
            ax.set_ylabel('Sentiment Bias vs Baseline', fontsize=12, fontweight='bold')
        
        # グリッド
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
    
    # 全体タイトル
    fig.suptitle('Color-Induced Sentiment Bias Across VLM Models\n'
                 'Comparison of Cultural Color Association Effects', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # レイアウト調整
    plt.tight_layout()
    
    # 保存
    output_file = Path(output_path) / 'figure1_multi_model_comparison.png'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    return str(output_file)

def create_model_correlation_heatmap(df: pd.DataFrame, output_path: str) -> str:
    """
    モデル間のバイアス相関を示すヒートマップ作成
    """
    # 強度がstrongのデータのみ
    main_df = df[df['intensity'] == 'strong'].copy()
    
    # ピボットテーブル作成（条件×モデル）
    pivot_df = main_df.pivot_table(
        index=['category', 'color'], 
        columns='model_display',
        values='bias_vs_baseline',
        aggfunc='first'
    )
    
    # 相関行列計算
    corr_matrix = pivot_df.corr()
    
    # ヒートマップ作成
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 上三角をマスク
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8},
                ax=ax)
    
    ax.set_title('Inter-Model Correlation of Color Bias Effects\n'
                 'Pearson Correlation Coefficients', 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 保存
    output_file = Path(output_path) / 'figure1_model_correlation.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    return str(output_file)

def generate_multi_model_summary(df: pd.DataFrame, output_path: str) -> str:
    """
    複数モデルの要約統計を生成
    """
    main_df = df[df['intensity'] == 'strong'].copy()
    
    # 統計要約
    summary_stats = []
    
    for model in sorted(main_df['model_display'].unique()):
        model_data = main_df[main_df['model_display'] == model]
        
        # 主要な効果を計算
        pos_green = model_data[(model_data['category'] == 'Positive Words') & 
                              (model_data['color'] == 'green')]
        neg_red = model_data[(model_data['category'] == 'Negative Words') & 
                            (model_data['color'] == 'red')]
        
        pos_green_bias = pos_green['bias_vs_baseline'].iloc[0] if len(pos_green) > 0 else 0
        neg_red_bias = neg_red['bias_vs_baseline'].iloc[0] if len(neg_red) > 0 else 0
        
        summary_stats.append({
            'Model': model,
            'Positive_Green_Bias': pos_green_bias,
            'Negative_Red_Bias': neg_red_bias,
            'Total_Effect_Range': pos_green_bias - neg_red_bias,
            'Mean_Absolute_Bias': model_data['bias_vs_baseline'].abs().mean()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # CSV保存
    csv_file = Path(output_path) / 'figure1_multi_model_summary.csv'
    summary_df.to_csv(csv_file, index=False, float_format='%.4f')
    
    return str(csv_file)

def main():
    parser = argparse.ArgumentParser(description="Generate Multi-Model Color Bias Comparison")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Root directory containing model results")
    parser.add_argument("--output-dir", type=str, default="results/",
                       help="Output directory for figures")
    parser.add_argument("--models", type=str, required=True,
                       help="Comma-separated list of model names")
    parser.add_argument("--dataset", type=str, default="custom_short",
                       help="Dataset name to analyze")
    parser.add_argument("--create-correlation", action="store_true",
                       help="Create model correlation heatmap")
    
    args = parser.parse_args()
    
    # モデルリスト解析
    models = [m.strip() for m in args.models.split(',')]
    print(f"Analyzing models: {models}")
    
    # ファイル検索
    files_dict = find_sentiment_metrics_files(args.input_dir, models, args.dataset)
    
    if not files_dict:
        print("ERROR: No valid files found")
        return
    
    # データ読み込み・処理
    print("Loading and processing data...")
    df = load_and_process_multi_model_data(files_dict)
    print(f"Processed {len(df)} data points across {len(df['model_display'].unique())} models")
    
    # メイン比較プロット
    print("Creating multi-model comparison plot...")
    plot_file = create_multi_model_comparison(df, args.output_dir)
    print(f"Saved comparison plot: {plot_file}")
    
    # 相関ヒートマップ（オプション）
    if args.create_correlation:
        print("Creating correlation heatmap...")
        corr_file = create_model_correlation_heatmap(df, args.output_dir)
        print(f"Saved correlation heatmap: {corr_file}")
    
    # 要約統計
    print("Generating summary statistics...")
    summary_file = generate_multi_model_summary(df, args.output_dir)
    print(f"Saved summary: {summary_file}")
    
    print("Multi-model analysis completed!")

if __name__ == "__main__":
    main()

# Usage Examples:
#
# Basic multi-model comparison:
# python src/visualization/figure1_multi_model.py \
#   --input-dir results/color/ \
#   --output-dir figures/ \
#   --models "llava-hf_llava-v1.6-mistral-7b-hf,Qwen_Qwen2-VL-7B-Instruct" \
#   --dataset custom_short
#
# With correlation analysis:
# python src/visualization/figure1_multi_model.py \
#   --input-dir results/color/ \
#   --output-dir figures/ \
#   --models "llava-hf_llava-v1.6-mistral-7b-hf,llava-hf_llava-v1.6-vicuna-7b-hf,Qwen_Qwen2-VL-7B-Instruct,HuggingFaceM4_idefics2-8b" \
#   --dataset custom_short \
#   --create-correlation
