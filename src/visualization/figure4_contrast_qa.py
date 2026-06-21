#!/usr/bin/env python3
"""
Figure 4: Contrast-Based QA Task Failure Analysis

このスクリプトはコントラスト実験のQAタスク失敗分析を可視化します。
VLMがテキストのコントラスト（視認性）と視覚的サリエンシーによって
QA性能と誤答誘導がどう変化するかを示します。

Usage:
    python src/visualization/figure4_contrast_qa.py \
        --input-dir results/contrast/ \
        --output-dir figures/ \
        --models "all"
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

def extract_rgb_value(condition: str) -> int:
    """condition名からRGB値を抽出"""
    if 'global_contrast_rgb_' in condition:
        return int(condition.split('_')[-1])
    elif 'salient_bg_' in condition:
        return int(condition.split('_')[-1])
    return 255  # baseline

def rgb_to_contrast_ratio(rgb_value: int) -> float:
    """RGB値を背景可視性比に変換（RGB値が低い=暗い=見える, RGB値が高い=明るい=見えない）"""
    return (255 - rgb_value) / 255

def find_qa_metrics_files(input_dir: str, models: List[str]) -> Dict[str, List[str]]:
    """指定されたモデルのQA関連CSVファイルを検索"""
    results = {}
    
    input_path = Path(input_dir)
    for model in models:
        model_files = {'qa_metrics': [], 'induced_error': []}
        model_dir = input_path / model
        
        if not model_dir.exists():
            print(f"警告: モデルディレクトリが見つかりません: {model_dir}")
            continue
        
        # squad_bipolar データセットを検索
        qa_files = list(model_dir.glob("squad_bipolar/*/analysis/qa_metrics.csv"))
        error_files = list(model_dir.glob("squad_bipolar/*/analysis/induced_error_metrics.csv"))
        
        model_files['qa_metrics'].extend([str(f) for f in qa_files])
        model_files['induced_error'].extend([str(f) for f in error_files])
        
        if model_files['qa_metrics'] or model_files['induced_error']:
            results[model] = model_files
        else:
            print(f"警告: {model}でQAメトリクスファイルが見つかりません")
    
    return results

def load_and_process_qa_data(file_paths: Dict[str, Dict[str, List[str]]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """QAメトリクスと誘導エラーデータを読み込み、統合データフレームを作成"""
    qa_data = []
    error_data = []
    
    for model, files in file_paths.items():
        # QAメトリクス処理
        for file_path in files['qa_metrics']:
            try:
                df = pd.read_csv(file_path)
                
                # ファイルパスから情報を抽出
                path_parts = Path(file_path).parts
                run_id = path_parts[-3]  # timestamp
                
                # デコーディング戦略を判定
                decoding = "beam" if "beam" in run_id else "standard"
                
                # データに情報を追加
                df['model'] = model
                df['run_id'] = run_id
                df['decoding'] = decoding
                df['file_path'] = file_path
                
                # RGB値とコントラスト比を計算
                df['rgb_value'] = df['condition'].apply(extract_rgb_value)
                df['contrast_ratio'] = df['rgb_value'].apply(rgb_to_contrast_ratio)
                
                # 条件タイプを分類
                df['condition_type'] = df['condition'].apply(lambda x: 
                    'global_contrast' if 'global_contrast' in x else
                    'answer_salient' if 'answer_salient' in x else
                    'decoy_salient' if 'decoy_salient' in x else 'unknown')
                
                qa_data.append(df)
                
            except Exception as e:
                print(f"エラー: QAファイル {file_path}の読み込みに失敗: {e}")
        
        # 誘導エラー処理
        for file_path in files['induced_error']:
            try:
                df = pd.read_csv(file_path)
                
                # ファイルパスから情報を抽出
                path_parts = Path(file_path).parts
                run_id = path_parts[-3]
                
                decoding = "beam" if "beam" in run_id else "standard"
                
                df['model'] = model
                df['run_id'] = run_id
                df['decoding'] = decoding
                df['file_path'] = file_path
                
                df['rgb_value'] = df['condition'].apply(extract_rgb_value)
                df['contrast_ratio'] = df['rgb_value'].apply(rgb_to_contrast_ratio)
                
                error_data.append(df)
                
            except Exception as e:
                print(f"エラー: 誘導エラーファイル {file_path}の読み込みに失敗: {e}")
    
    qa_df = pd.concat(qa_data, ignore_index=True) if qa_data else pd.DataFrame()
    error_df = pd.concat(error_data, ignore_index=True) if error_data else pd.DataFrame()
    
    print(f"QAデータ読み込み完了: {len(qa_df)}レコード")
    print(f"誘導エラーデータ読み込み完了: {len(error_df)}レコード")
    
    return qa_df, error_df

def create_qa_performance_lineplot(qa_df: pd.DataFrame, output_path: str, legend_path: str):
    """QA性能 vs コントラストのライン・プロット（全条件統合版）"""
    
    # 標準デコーディングのみ使用、全条件タイプを含める
    plot_df = qa_df[qa_df['decoding'] == 'standard'].copy()
    
    if len(plot_df) == 0:
        print("警告: QA性能プロット用のデータが見つかりません")
        return
    
    # 条件タイプの表示順序と線スタイルを定義
    condition_types = ['global_contrast', 'answer_salient', 'decoy_salient']
    line_styles = {
        'global_contrast': '-',      # 実線 (テキスト全体)
        'answer_salient': '--',      # 破線 (正答強調)
        'decoy_salient': ':'         # 点線 (デコイ強調)
    }
    condition_labels = {
        'global_contrast': 'Global Contrast',
        'answer_salient': 'Answer Salient',
        'decoy_salient': 'Decoy Salient'
    }
    
    # 1つのグラフに全モデル・全条件を表示
    plt.figure(figsize=(14, 10))
    
    models = sorted(plot_df['model'].unique())
    
    # Viridisカラーパレットを使用（Figure 3と統一）
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    # 凡例用の要素を準備
    legend_elements = []
    
    for i, model in enumerate(models):
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        for condition_type in condition_types:
            model_condition_data = plot_df[
                (plot_df['model'] == model) & 
                (plot_df['condition_type'] == condition_type)
            ].copy()
            
            if len(model_condition_data) == 0:
                continue
            
            # 🔥 Highlight系条件では背景不可視（RGB >= 250）のみを除外
            if condition_type in ['answer_salient', 'decoy_salient']:
                model_condition_data = model_condition_data[
                    model_condition_data['rgb_value'] < 250
                ].copy()
                if len(model_condition_data) == 0:
                    continue
                
            # コントラスト比でソート
            model_condition_data = model_condition_data.sort_values('contrast_ratio')
            
            linestyle = line_styles[condition_type]
            
            # 最初のモデルの場合のみ条件ラベルを追加
            if i == 0:
                label = condition_labels[condition_type]
            else:
                label = None
            
            plt.plot(model_condition_data['contrast_ratio'], 
                    model_condition_data['f1_avg'],
                    color=color,
                    linestyle=linestyle,
                    linewidth=3,
                    marker='o',
                    markersize=5,
                    label=label,
                    alpha=0.8,
                    markeredgewidth=1,
                    markeredgecolor='white')
    
    # モデル専用の凡例要素を作成（色のみ）
    for i, model in enumerate(models):
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        # ダミーライン（色の参照用）
        line = plt.Line2D([0], [0], 
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=6,
                        markeredgewidth=1,
                        markeredgecolor='white',
                        label=model_display)
        legend_elements.append(line)
    
    # スタイル専用の凡例要素を作成
    for condition_type in condition_types:
        linestyle = line_styles[condition_type]
        label = condition_labels[condition_type]
        
        line = plt.Line2D([0], [0], 
                        color='black',
                        linestyle=linestyle,
                        linewidth=3,
                        label=label)
        legend_elements.append(line)
    
    plt.xlabel('Text Visibility (0=invisible, 1=maximum)', fontsize=16)
    plt.ylabel('F1 Score\n(QA Performance)', fontsize=16)
    plt.title('QA Performance vs Text Visibility\nAcross All Conditions and VLM Models', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, zorder=1)
    
    # 軸の範囲設定
    plt.xlim(0, 1.05)
    plt.ylim(0, max(plot_df['f1_avg'].max() * 1.1, 0.1))
    
    # 軸のフォントサイズ調整
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 2列の凡例を作成（モデル + 条件）
    first_legend = plt.legend(handles=legend_elements[:len(models)], 
                            title='Models', 
                            loc='upper left', 
                            fontsize=11,
                            title_fontsize=12,
                            framealpha=0.9)
    plt.gca().add_artist(first_legend)
    
    plt.legend(handles=legend_elements[len(models):], 
             title='Conditions', 
             loc='upper right', 
             fontsize=11,
             title_fontsize=12,
             framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QA性能ライン・プロット（全条件）保存: {output_path}")
    
    # 別途凡例を作成
    create_comprehensive_qa_legend(models, condition_types, line_styles, condition_labels, legend_path)

def create_comprehensive_qa_legend(models: List[str], condition_types: List[str], line_styles: Dict[str, str], condition_labels: Dict[str, str], legend_path: str):
    """全条件対応のQA実験用独立凡例を作成"""
    
    # 凡例専用の図を作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.axis('off')  # 軸を非表示
    ax2.axis('off')  # 軸を非表示
    
    # Viridisカラーパレットを使用
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    # モデル用凡例要素
    model_legend_elements = []
    for i, model in enumerate(models):
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        line = plt.Line2D([0], [0], 
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        markeredgewidth=1,
                        markeredgecolor='white',
                        label=model_display)
        model_legend_elements.append(line)
    
    # 条件用凡例要素
    condition_legend_elements = []
    for condition_type in condition_types:
        linestyle = line_styles[condition_type]
        label = condition_labels[condition_type]
        
        line = plt.Line2D([0], [0], 
                        color='black',
                        linestyle=linestyle,
                        linewidth=3,
                        label=label)
        condition_legend_elements.append(line)
    
    # モデル凡例を作成
    legend1 = ax1.legend(handles=model_legend_elements, 
                        title='VLM Models',
                        loc='center',
                        fontsize=14,
                        title_fontsize=16,
                        handlelength=3,
                        handletextpad=1)
    
    # 条件凡例を作成
    legend2 = ax2.legend(handles=condition_legend_elements, 
                        title='Condition Types',
                        loc='center',
                        fontsize=14,
                        title_fontsize=16,
                        handlelength=3,
                        handletextpad=1)
    
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QA凡例（全条件）保存: {legend_path}")

def create_qa_legend(models: List[str], legend_path: str):
    """QA実験用の独立した凡例を作成（互換性維持）"""
    
    # 凡例専用の図を作成
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')  # 軸を非表示
    
    # Viridisカラーパレットを使用
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    # 凡例用のダミープロット
    legend_elements = []
    
    for i, model in enumerate(models):
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        # ダミーライン作成
        line = plt.Line2D([0], [0], 
                        color=color,
                        linestyle='-',
                        linewidth=3,
                        marker='o',
                        markersize=8,
                        markeredgewidth=1,
                        markeredgecolor='white',
                        label=model_display)
        legend_elements.append(line)
    
    # 凡例を作成
    legend = ax.legend(handles=legend_elements, 
                      loc='center',
                      fontsize=14,
                      handlelength=3,
                      handletextpad=1)
    
    # 凡例のタイトル
    legend.set_title('VLM Models', prop={'size': 16, 'weight': 'bold'})
    
    plt.tight_layout()
    plt.savefig(legend_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QA凡例保存: {legend_path}")

def create_saliency_induced_errors(error_df: pd.DataFrame, output_path: str):
    """視覚的サリエンシーによる誘導エラー分析"""
    
    # 標準デコーディングのみ使用
    plot_df = error_df[error_df['decoding'] == 'standard'].copy()
    
    if len(plot_df) == 0:
        print("警告: 誘導エラープロット用のデータが見つかりません")
        return
    
    # 🔥 重要修正: 背景不可視条件（RGB >= 250）のみを除外
    # RGB=1-240まで含める（可視性0.059-0.996の範囲）
    plot_df = plot_df[plot_df['rgb_value'] < 250].copy()
    print(f"背景不可視条件（RGB≥250）を除外: {len(plot_df)}レコードで分析")
    
    if len(plot_df) == 0:
        print("警告: 背景可視条件のデータが見つかりません")
        return
    
    # 1つのグラフに全モデルを表示
    plt.figure(figsize=(12, 8))
    
    models = sorted(plot_df['model'].unique())
    
    # Viridisカラーパレットを使用
    import matplotlib.cm as cm
    colors = cm.viridis(np.linspace(0, 1, len(models)))
    
    for i, model in enumerate(models):
        model_data = plot_df[plot_df['model'] == model]
        color = colors[i]
        model_display = MODEL_DISPLAY_NAMES.get(model, model)
        
        if len(model_data) == 0:
            continue
            
        # コントラスト比でソート
        model_data = model_data.sort_values('contrast_ratio')
        
        plt.plot(model_data['contrast_ratio'] * 100,  # 横軸もパーセント変換
                model_data['induced_error_rate'] * 100,  # 縦軸パーセント変換
                color=color,
                linestyle='-',
                linewidth=3,
                marker='s',
                markersize=6,
                label=model_display,
                alpha=0.9,
                markeredgewidth=1,
                markeredgecolor='white')
    
    plt.xlabel('Background Text Visibility (%)\n(Decoy words always high contrast)', fontsize=16)
    plt.ylabel('Induced Error Rate (%)\n(Decoy Highlighting Effect)', fontsize=16)
    plt.title('Decoy Highlighting Effect vs Background Visibility\nSaliency-Based Visual Manipulation', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, zorder=1)
    
    # 軸の範囲設定
    plt.xlim(0, 105)  # パーセント表示に合わせて調整
    plt.ylim(0, 45)   # パーセント表示に合わせて調整
    
    # 軸のフォントサイズ調整
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 凡例
    plt.legend(fontsize=12, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"視覚的誘導エラープロット保存: {output_path}")

def create_qa_performance_heatmap(qa_df: pd.DataFrame, output_path: str):
    """QA性能ヒートマップ（モデル × コントラスト値）"""
    
    # 標準デコーディング、global_contrastのみ使用
    plot_df = qa_df[(qa_df['decoding'] == 'standard') & 
                    (qa_df['condition_type'] == 'global_contrast')].copy()
    
    if len(plot_df) == 0:
        print("警告: QA性能ヒートマップ用のデータが見つかりません")
        return
    
    # ピボットテーブル作成
    heatmap_data = plot_df.pivot_table(
        values='f1_avg',
        index='model',
        columns='rgb_value',
        aggfunc='mean'
    )
    
    # モデル名を表示用に変換
    heatmap_data.index = [MODEL_DISPLAY_NAMES.get(model, model) for model in heatmap_data.index]
    
    # 図を作成
    plt.figure(figsize=(14, 6))
    
    # カラーマップ設定（低F1=赤、高F1=緑）
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlGn',  # 赤黄緑（低から高へ）
                cbar_kws={'label': 'F1 Score (QA Performance)'},
                linewidths=0.5)
    
    plt.title('QA Performance by Text Contrast (RGB Value)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Text RGB Value (0=black, 255=white)', fontsize=12)
    plt.ylabel('VLM Model', fontsize=12)
    
    # X軸ラベルを読みやすく調整
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QA性能ヒートマップ保存: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='コントラスト・QAタスク可視化')
    parser.add_argument('--input-dir', required=True, help='結果ディレクトリのパス')
    parser.add_argument('--output-dir', required=True, help='出力ディレクトリのパス')
    parser.add_argument('--models', default='all', 
                       help='対象モデル（カンマ区切りまたは"all"）')
    
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
    
    print(f"対象モデル: {models}")
    
    # ファイル検索
    file_paths = find_qa_metrics_files(args.input_dir, models)
    
    if not file_paths:
        print("エラー: 有効なQAデータファイルが見つかりませんでした")
        return
    
    # データ読み込み・処理
    qa_df, error_df = load_and_process_qa_data(file_paths)
    
    # 可視化作成
    print("\n図表生成中...")
    
    # 1. QA性能 vs コントラスト（ライン・プロット）
    if not qa_df.empty:
        create_qa_performance_lineplot(
            qa_df, 
            output_dir / 'figure4a_qa_performance_lineplot.png',
            output_dir / 'figure4a_qa_performance_legend.png'
        )
        
        # 3. QA性能ヒートマップ
        create_qa_performance_heatmap(
            qa_df,
            output_dir / 'figure4c_qa_performance_heatmap.png'
        )
    
    # 2. 視覚的誘導エラー分析
    if not error_df.empty:
        create_saliency_induced_errors(
            error_df,
            output_dir / 'figure4b_saliency_induced_errors.png'
        )
    
    print(f"\n✅ 全図表生成完了: {output_dir}")
    print("生成ファイル:")
    if not qa_df.empty:
        print("  - figure4a_qa_performance_lineplot.png")
        print("  - figure4a_qa_performance_legend.png")
        print("  - figure4c_qa_performance_heatmap.png")
    if not error_df.empty:
        print("  - figure4b_saliency_induced_errors.png")

if __name__ == "__main__":
    main()


