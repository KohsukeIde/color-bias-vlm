#!/usr/bin/env python3
"""
新しい両プロンプト実験結果を自動的に可視化するスクリプト

使用方法:
python scripts/visualize_dual_prompt_results.py

全ての新しい実験結果を検索し、両プロンプトの比較可視化を自動生成します。
"""

import os
import sys
from pathlib import Path
import subprocess
from typing import List, Dict
import glob

def find_combined_csv_files(results_dir: Path) -> List[Dict[str, str]]:
    """
    results/ディレクトリから全ての*_combined.csvファイルを検索
    """
    pattern = str(results_dir / "**" / "*_combined.csv")
    csv_files = glob.glob(pattern, recursive=True)
    
    results = []
    for csv_path in csv_files:
        path_obj = Path(csv_path)
        
        # パスから情報を抽出: results/color/model_name/dataset/run_id/analysis/sentiment_metrics_combined.csv
        path_parts = path_obj.parts
        
        if len(path_parts) >= 6 and 'results' in path_parts:
            results_idx = path_parts.index('results')
            if results_idx + 5 < len(path_parts):
                experiment_type = path_parts[results_idx + 1]  # color or contrast
                model_name = path_parts[results_idx + 2]
                dataset_name = path_parts[results_idx + 3]
                run_id = path_parts[results_idx + 4]
                
                results.append({
                    'csv_path': csv_path,
                    'experiment_type': experiment_type,
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'run_id': run_id,
                    'output_dir': f"results/{experiment_type}/{model_name}/figures"
                })
    
    return results

def run_visualization(csv_path: str, output_dir: str, model_name: str) -> bool:
    """
    可視化スクリプトを実行（メイン + Bipolar分析）
    """
    try:
        # 出力ディレクトリを作成
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # メイン可視化コマンドを実行
        main_cmd = [
            sys.executable,
            "src/visualization/figure1_main_results.py",
            "--input-csv", csv_path,
            "--output-dir", output_dir,
            "--model-name", model_name,
            "--save-stats"
        ]
        
        print(f"🎨 Running main visualization: {' '.join(main_cmd)}")
        main_result = subprocess.run(main_cmd, capture_output=True, text=True)
        
        main_success = main_result.returncode == 0
        if not main_success:
            print(f"❌ Main visualization error: {model_name}")
            print(f"   stdout: {main_result.stdout}")
            print(f"   stderr: {main_result.stderr}")
        
        # Bipolar分析を実行
        bipolar_cmd = [
            sys.executable,
            "src/visualization/figure_bipolar_analysis.py",
            "--input-csv", csv_path,
            "--output-dir", output_dir,
            "--model-name", model_name
        ]
        
        print(f"🎨 Running bipolar analysis: {' '.join(bipolar_cmd)}")
        bipolar_result = subprocess.run(bipolar_cmd, capture_output=True, text=True)
        
        bipolar_success = bipolar_result.returncode == 0
        if not bipolar_success:
            print(f"⚠️  Bipolar analysis warning: {model_name}")
            print(f"   stdout: {bipolar_result.stdout}")
            print(f"   stderr: {bipolar_result.stderr}")
            # Bipolarは失敗してもメインが成功していればOKとする
        
        if main_success:
            status = "✅ Success"
            if bipolar_success:
                status += " (with bipolar analysis)"
            else:
                status += " (main only)"
            print(f"{status}: {model_name}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Exception for {model_name}: {e}")
        return False

def main():
    """
    メイン実行関数
    """
    print("🔍 Searching for dual prompt experiment results...")
    
    # プロジェクトルートに移動
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # 結果ファイルを検索
    results_dir = Path("results")
    csv_files = find_combined_csv_files(results_dir)
    
    if not csv_files:
        print("⚠️  No combined CSV files found. Make sure experiments have been run with dual prompt support.")
        return
    
    print(f"📊 Found {len(csv_files)} experiment results to visualize")
    
    success_count = 0
    total_count = len(csv_files)
    
    for result in csv_files:
        print(f"\n📈 Processing: {result['model_name']} - {result['dataset_name']}")
        
        # モデル名をクリーンアップ
        clean_model_name = result['model_name'].replace('_', '/').replace('-', '-')
        
        success = run_visualization(
            csv_path=result['csv_path'],
            output_dir=result['output_dir'],
            model_name=clean_model_name
        )
        
        if success:
            success_count += 1
    
    print(f"\n🎯 Visualization completed: {success_count}/{total_count} successful")
    print(f"📁 Results saved in: results/")
    
    if success_count > 0:
        print("\n📊 Generated visualizations:")
        print("  - figure1_main_color_bias.png (traditional single-prompt view)")
        print("  - figure1_dual_prompt_comparison.png (side-by-side comparison)")
        print("  - summary_statistics.csv (statistical summary)")
        print("  - figure_bipolar_heatmap_*.png (bipolar color combination heatmaps)")
        print("  - figure_bipolar_bars_*.png (bipolar color combination bar plots)")
        print("  - figure_bipolar_dual_prompt_comparison.png (bipolar prompt comparison)")
        print("  - bipolar_summary_statistics.csv (bipolar statistical summary)")

if __name__ == "__main__":
    main()
