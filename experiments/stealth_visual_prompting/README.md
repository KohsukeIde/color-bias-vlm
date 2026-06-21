# Stealth Visual Prompting Experiments

この研究ディレクトリには、ステルス・ビジュアルプロンプトのメカニズム解明を目的とした包括的な実験実装が含まれています。

## 🎯 研究目標

VLM（Vision-Language Model）における「ステルス・ビジュアルプロンプト」現象を定量的に解明し、以下の4つの可視化手法（A1〜A4）を通じて、トップカンファレンス（CVPR, NeurIPSなど）レベルの研究成果を創出します。

### 可視化手法
- **A1**: 潜在空間マップ (UMAP/t-SNE)
- **A2**: チューニングカーブ
- **A3**: 相転移の可視化 (心理測定曲線)
- **A4**: アトリビューションマップ (Grad-CAM)

## 📁 統一ディレクトリ構造

全ての結果は `results/clip_ablation/` の下に整理されています：

```
results/clip_ablation/
├── phase1/                     # データ収集結果
│   ├── main/                   # メイン実験データ
│   ├── complete_run/           # 過去の完全実験
│   └── with_ocr/              # OCR評価付き実験
├── phase2/                     # 可視化・分析結果
│   ├── A1_latent_space/       # 潜在空間可視化
│   ├── A2_tuning_curves/      # チューニングカーブ分析
│   ├── A3_phase_transition/   # 相転移分析
│   └── A4_attribution/        # アトリビューションマップ分析
└── configs/                    # 設定ファイル
```

## 🚀 コアスクリプト

- `phase1_data_collection.py`: CLIPに最適化された336x336画像での包括的データ収集
- `phase2_visualization.py`: 完全な可視化・分析パイプライン
- `run_complete_analysis.py`: エンドツーエンド分析パイプライン

## 📋 クイック使用方法

### Phase 1: データ収集（デフォルトパス使用）

```bash
# 基本データ収集
python experiments/stealth_visual_prompting/phase1_data_collection.py

# OCR評価付きデータ収集（推奨）
python experiments/stealth_visual_prompting/phase1_data_collection.py --enable-ocr

# カスタム出力ディレクトリ
python experiments/stealth_visual_prompting/phase1_data_collection.py \
    --output-dir results/clip_ablation/phase1/custom \
    --enable-ocr
```

### Phase 2: 可視化（デフォルトパス使用）

```bash
# 完全分析パイプライン
python experiments/stealth_visual_prompting/phase2_visualization.py

# カスタムデータパス
python experiments/stealth_visual_prompting/phase2_visualization.py \
    --data-path results/clip_ablation/phase1/with_ocr/comprehensive_results.csv
```

## 🔧 技術仕様

### 画像生成最適化
- **画像サイズ**: 336x336 (CLIPの内部処理と一致)
- **フォントサイズ**: 40-120px (336x336で読み取り可能)
- **色相**: 0-360度、10度刻み
- **彩度**: 0.2, 0.5, 0.8, 1.0
- **明度**: 0.3, 0.5, 0.7, 0.9
- **コントラスト**: CIE ΔE値 0.5-128 (相転移分析用)

### 意味軸 (Semantic Axes)
標準的な10軸を提供：
- safety (safe ↔ dangerous)
- valence (good ↔ bad) 
- arousal (calm ↔ chaotic)
- temperature (warm ↔ cold)
- emotion (happy ↔ sad)
- morality (moral ↔ immoral)
- activity (active ↔ passive)
- strength (strong ↔ weak)
- size (large ↔ small)
- speed (fast ↔ slow)

## 📊 出力結果

実行完了後、以下の結果が生成されます：

### Phase 1 出力
- `comprehensive_results.csv`: 全実験データ
- `semantic_axes.json`: 意味軸定義
- `generated_images/`: 生成画像サンプル

### Phase 2 出力
- **A1**: `a1_umap_latent_space.png`, `a1_tsne_latent_space.png`
- **A2**: `a2_hue_tuning_curves.png`, `a2_fontsize_tuning_curves.png`
- **A3**: `a3_phase_transition.png`, `a3_phase_transition_data.json`
- **A4**: `a4_attribution_heatmaps.png`, `a4_influence_analysis.json`

## 🔬 研究活用のポイント

### 論文執筆への活用
- **A3の相転移分析**: 「認識不能領域での意味的バイアス」として強力な証拠
- **A1の潜在空間マップ**: 「色が意味に与える普遍的影響」を視覚的に実証
- **A2のチューニングカーブ**: 「特定色への選択的反応」を定量化

### 実験データの解釈
- **投影量 > 0**: ポジティブ方向への意味的シフト
- **投影量 < 0**: ネガティブ方向への意味的シフト
- **相転移点**: OCR精度50%となるコントラスト値
- **幻覚ゾーン**: 認識不能だが意味的バイアスが存在する領域

## 🛠️ 依存関係

```bash
# 必須パッケージ
pip install torch torchvision transformers
pip install pillow numpy pandas matplotlib seaborn
pip install scikit-learn tqdm

# 可視化拡張
pip install umap-learn  # UMAP可視化用

# OpenCLIP（オプション）
pip install open_clip_torch
```

## 📝 変更履歴

- **v2024.08**: ディレクトリ構造統一、CLIP最適化、OCR評価改善
- **v2024.07**: 初期実装、基本可視化パイプライン

---

**注意**: 全てのスクリプトはデフォルトで統一された `results/clip_ablation/` 構造を使用するよう設定されています。