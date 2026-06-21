#!/usr/bin/env python3
"""
ステルス・ビジュアルプロンプト研究の完全実行スクリプト

このスクリプトは研究実行計画の全フェーズを順次実行します：
- フェーズ0: モデル準備と意味軸定義
- フェーズ1: 網羅的データ収集
- フェーズ2: 可視化・分析 (A1-A4)

使用例:
python run_complete_analysis.py --output-dir results/stealth_visual_prompting --models clip siglip
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import time

# プロジェクトルートを追加
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_command(cmd: list, description: str, cwd: Path = None) -> bool:
    """コマンドを実行し、結果を表示"""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            cmd, 
            cwd=cwd or _PROJECT_ROOT,
            check=True,
            capture_output=False,  # リアルタイム出力
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ SUCCESS: {description} (took {elapsed:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ FAILED: {description}")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {description}")
        print(f"Exception: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete Stealth Visual Prompting Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Basic run with CLIP
  python run_complete_analysis.py --output-dir results/svp_analysis
  
  # Multiple models with custom words
  python run_complete_analysis.py --output-dir results/svp_analysis \\
    --models clip siglip --words safe dangerous good bad
  
  # Resume from existing data
  python run_complete_analysis.py --output-dir results/svp_analysis --resume
        """
    )
    
    # 基本設定
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for all results")
    parser.add_argument("--models", nargs="+", default=["clip"], 
                       choices=["clip", "siglip", "openclip"],
                       help="Models to analyze")
    
    # データ収集設定
    parser.add_argument("--words", nargs="*", default=None, help="Words to test (default: use standard set)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding computation")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR accuracy evaluation")
    parser.add_argument("--vlm-model", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf", 
                       help="VLM model for OCR evaluation")
    
    # 可視化設定
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample size for dimensionality reduction")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 (data collection)")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Phase 2 (visualization)")
    
    # 実行制御
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    
    args = parser.parse_args()
    
    # 出力ディレクトリを準備
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 実行ログを開始
    log_file = output_dir / f"execution_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"""
🚀 STEALTH VISUAL PROMPTING RESEARCH PIPELINE
{'='*60}
Output Directory: {output_dir.absolute()}
Models: {args.models}
Words: {'Standard set' if not args.words else args.words}
OCR Evaluation: {'Enabled' if args.enable_ocr else 'Disabled'}
Resume Mode: {'On' if args.resume else 'Off'}
Log File: {log_file}
{'='*60}
    """)
    
    if args.dry_run:
        print("🏃 DRY RUN MODE - Commands will be shown but not executed")
    
    # 実行統計
    total_phases = 0
    successful_phases = 0
    failed_phases = []
    
    # モデル別設定
    model_configs = {
        "clip": {"model_type": "hf", "model_name": "openai/clip-vit-large-patch14-336"},
        "siglip": {"model_type": "siglip", "model_name": "google/siglip-large-patch16-256"},
        "openclip": {"model_type": "openclip", "model_name": "ViT-L-14-336/openai"}
    }
    
    # 各モデルについて実行
    for model_key in args.models:
        if model_key not in model_configs:
            print(f"❌ Unknown model: {model_key}")
            continue
        
        model_config = model_configs[model_key]
        model_output_dir = output_dir / f"model_{model_key}"
        
        print(f"\n🎯 PROCESSING MODEL: {model_key}")
        print(f"   Config: {model_config}")
        print(f"   Output: {model_output_dir}")
        
        # フェーズ1: データ収集
        if not args.skip_phase1:
            total_phases += 1
            phase1_cmd = [
                sys.executable, "experiments/stealth_visual_prompting/phase1_data_collection.py",
                "--model-type", model_config["model_type"],
                "--model-name", model_config["model_name"],
                "--output-dir", str(model_output_dir / "phase1"),
                "--batch-size", str(args.batch_size)
            ]
            
            if args.words:
                phase1_cmd.extend(["--words"] + args.words)
            
            if args.enable_ocr:
                phase1_cmd.extend(["--enable-ocr", "--vlm-model", args.vlm_model])
            
            if args.resume:
                phase1_cmd.append("--resume")
            
            if args.dry_run:
                print(f"[DRY RUN] Phase 1: {' '.join(phase1_cmd)}")
            else:
                success = run_command(phase1_cmd, f"Phase 1 Data Collection ({model_key})")
                if success:
                    successful_phases += 1
                else:
                    failed_phases.append(f"Phase 1 - {model_key}")
        
        # フェーズ2: 可視化
        if not args.skip_phase2:
            # データファイルの存在確認
            data_file = model_output_dir / "phase1" / "comprehensive_results.csv"
            
            if data_file.exists() or args.dry_run:
                total_phases += 1
                phase2_cmd = [
                    sys.executable, "experiments/stealth_visual_prompting/phase2_visualization.py",
                    "--data-path", str(data_file),
                    "--output-dir", str(model_output_dir / "phase2"),
                    "--sample-size", str(args.sample_size)
                ]
                
                if args.dry_run:
                    print(f"[DRY RUN] Phase 2: {' '.join(phase2_cmd)}")
                else:
                    success = run_command(phase2_cmd, f"Phase 2 Visualization ({model_key})")
                    if success:
                        successful_phases += 1
                    else:
                        failed_phases.append(f"Phase 2 - {model_key}")
            else:
                print(f"⚠️  Skipping Phase 2 for {model_key}: Data file not found ({data_file})")
    
    # 比較分析（複数モデルの場合）
    if len(args.models) > 1 and not args.dry_run:
        total_phases += 1
        print(f"\n🔍 GENERATING CROSS-MODEL COMPARISON")
        
        # 比較分析スクリプト（簡易版）
        comparison_dir = output_dir / "comparison"
        comparison_dir.mkdir(exist_ok=True)
        
        comparison_data = {}
        for model_key in args.models:
            data_file = output_dir / f"model_{model_key}" / "phase1" / "comprehensive_results.csv"
            if data_file.exists():
                comparison_data[model_key] = str(data_file)
        
        if comparison_data:
            # 比較データを保存
            comparison_config = {
                "timestamp": datetime.now().isoformat(),
                "models": comparison_data,
                "parameters": {
                    "words": args.words,
                    "enable_ocr": args.enable_ocr,
                    "batch_size": args.batch_size
                }
            }
            
            with open(comparison_dir / "comparison_config.json", "w") as f:
                json.dump(comparison_config, f, indent=2)
            
            print(f"✅ Comparison config saved to {comparison_dir / 'comparison_config.json'}")
            successful_phases += 1
        else:
            failed_phases.append("Cross-model comparison")
    
    # 実行サマリー
    print(f"""
{'='*60}
🎉 PIPELINE EXECUTION COMPLETED
{'='*60}
Total Phases: {total_phases}
Successful: {successful_phases}
Failed: {len(failed_phases)}

Output Directory: {output_dir.absolute()}
    """)
    
    if failed_phases:
        print("❌ Failed Phases:")
        for phase in failed_phases:
            print(f"   - {phase}")
    
    if successful_phases == total_phases:
        print("🎊 ALL PHASES COMPLETED SUCCESSFULLY!")
        print("\nNext Steps:")
        print("1. Review the generated visualizations in the phase2 directories")
        print("2. Examine the A1-A4 analysis results")
        print("3. Check the cross-model comparison data (if multiple models)")
        print("4. Use the results to write your research paper!")
    
    # 結果ファイルのリスト
    print(f"\n📁 Generated Files:")
    for result_file in output_dir.rglob("*.png"):
        print(f"   📊 {result_file.relative_to(output_dir)}")
    for result_file in output_dir.rglob("*.csv"):
        print(f"   📋 {result_file.relative_to(output_dir)}")
    for result_file in output_dir.rglob("*.json"):
        print(f"   📄 {result_file.relative_to(output_dir)}")


if __name__ == "__main__":
    main()

