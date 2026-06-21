# Visualization Scripts

論文用の図表生成スクリプト集です。実験結果から学術論文品質の可視化を自動生成します。

## スクリプト一覧

### figure1_main_results.py
**論文のメインフィギュア（Figure 1）生成**

色彩バイアス効果を示す棒グラフを作成。論文読者が最初に目にする重要な結果図です。

```bash
# 基本使用法
python src/visualization/figure1_main_results.py \
  --input-csv results/color/llava-hf_llava-v1.6-mistral-7b-hf/custom_short/20250116_120000/analysis/sentiment_metrics.csv \
  --output-dir results/color/llava-hf_llava-v1.6-mistral-7b-hf/custom_short/20250116_120000/figures/ \
  --model-name "LLaVA-v1.6-Mistral-7B"

# 統計表とキャプション付き
python src/visualization/figure1_main_results.py \
  --input-csv results/color/llava-hf_llava-v1.6-mistral-7b-hf/custom_short/20250116_120000/analysis/sentiment_metrics.csv \
  --output-dir results/color/llava-hf_llava-v1.6-mistral-7b-hf/custom_short/20250116_120000/figures/ \
  --model-name "LLaVA-v1.6-Mistral-7B" \
  --save-stats \
  --save-caption
```

**出力ファイル:**
- `figure1_main_color_bias.png` - メインの棒グラフ
- `figure1_summary_statistics.csv` - 要約統計表
- `figure1_caption.txt` - 論文用キャプション

### figure1_multi_model.py
**複数モデル比較版Figure 1**

複数のVLMモデルでの色彩バイアス効果を比較し、発見の一般性を示します。

```bash
# 複数モデル比較
python src/visualization/figure1_multi_model.py \
  --input-dir results/color/ \
  --output-dir results/figures/ \
  --models "llava-hf_llava-v1.6-mistral-7b-hf,Qwen_Qwen2-VL-7B-Instruct" \
  --dataset custom_short

# 相関分析付き
python src/visualization/figure1_multi_model.py \
  --input-dir results/color/ \
  --output-dir results/figures/ \
  --models "llava-hf_llava-v1.6-mistral-7b-hf,llava-hf_llava-v1.6-vicuna-7b-hf,Qwen_Qwen2-VL-7B-Instruct,HuggingFaceM4_idefics2-8b" \
  --dataset custom_short \
  --create-correlation
```

**出力ファイル:**
- `figure1_multi_model_comparison.png` - モデル間比較
- `figure1_model_correlation.png` - モデル相関ヒートマップ
- `figure1_multi_model_summary.csv` - 要約統計

## 論文での使用方法

### Figure 1の論文での記述例

```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{results/figures/figure1_main_color_bias.png}
\caption{Color-Induced Sentiment Bias in Vision-Language Models. 
Bar plot showing the effect of word coloring on VLM sentiment analysis...}
\label{fig:main_results}
\end{figure}
```

### 本文での言及例

```
Figure~\ref{fig:main_results} demonstrates the significant impact of color on VLM sentiment analysis. 
When positive words (excellent, brilliant, etc.) were colored green, the sentiment score increased by 
an average of 0.38 points compared to baseline (black text). Conversely, when negative words 
(terrible, awful, etc.) were colored red, sentiment scores decreased by 0.45 points on average.
```

## 技術仕様

### 画像品質設定
- **解像度**: 300 DPI（論文投稿品質）
- **フォーマット**: PNG（可逆圧縮）
- **背景**: 白色（印刷対応）
- **フォント**: Seaborn標準（可読性重視）

### 色彩設計（実験条件と同じ色を使用）
```python
# 実験で実際に使用する色と同じ色をグラフでも使用
# 読者が直感的に理解できるよう、視覚的一貫性を重視
COLOR_MAPPING = {
    'red': '#E74C3C',      # 明確な赤（実験で使用する赤と対応）
    'green': '#27AE60',    # 明確な緑（実験で使用する緑と対応）
    'blue': '#3498DB',     # 明確な青（実験で使用する青と対応）
    'yellow': '#F39C12',   # 視認性の良い黄（実験で使用する黄と対応）
    'cyan': '#1ABC9C',     # 鮮やかなシアン（実験で使用するシアンと対応）
    'magenta': '#E91E63'   # 鮮やかなマゼンタ（実験で使用するマゼンタと対応）
}
```

**実際の色を使用する利点:**
- 実験条件とグラフの色が一致し、読者が直感的に理解可能
- 「赤い文字の効果」を「赤いバー」で表現する視覚的一貫性
- 色彩心理学的な意味（赤=ネガティブ、緑=ポジティブ）が保持される
- 実験手法の透明性と再現性の向上

### 統計表示
- **エラーバー**: 標準誤差（SEM）
- **有意性**: p < 0.05レベル
- **基準線**: Y=0での点線表示

## データ要件

### 入力CSVフォーマット
`sentiment_metrics.csv`は以下の列を含む必要があります：

```csv
condition,avg_sentiment_score,bias_vs_baseline,std_sentiment_score,n
positive_only_red_subtle,-0.15,-0.23,0.12,10
positive_only_green_strong,0.45,0.38,0.08,10
negative_only_red_strong,-0.52,-0.45,0.15,10
...
```

### 必須列
- `condition`: 実験条件名
- `bias_vs_baseline`: ベースラインからのバイアス値
- `std_sentiment_score`: 標準偏差
- `n`: サンプル数

## トラブルシューティング

### よくある問題

#### 1. "No data found" エラー
```bash
# ファイルパス確認
ls -la results/color/*/custom_short/*/analysis/sentiment_metrics.csv

# データ内容確認
head results/color/llava-hf_llava-v1.6-mistral-7b-hf/custom_short/20250116_120000/analysis/sentiment_metrics.csv
```

#### 2. グラフが空白
```bash
# データの条件名確認
python -c "
import pandas as pd
df = pd.read_csv('sentiment_metrics.csv')
print(df['condition'].unique())
"
```

#### 3. フォントエラー
```bash
# matplotlib設定確認
python -c "
import matplotlib.pyplot as plt
print(plt.rcParams['font.family'])
"
```

### デバッグ用コマンド

```bash
# データ構造確認
python -c "
import pandas as pd
df = pd.read_csv('sentiment_metrics.csv')
print(df.info())
print(df.describe())
"

# 条件別データ数確認
python -c "
import pandas as pd
df = pd.read_csv('sentiment_metrics.csv')
print(df.groupby('condition').size())
"
```

## 拡張可能性

### 新しい図表の追加
1. 新しいスクリプトを`src/visualization/`に作成
2. 同様のコマンドライン引数構造を使用
3. 論文品質の設定を継承
4. READMEに使用例を追加

### カスタマイズ
- 色の変更: `COLOR_MAPPING`辞書を編集
- フォントサイズ: `fontsize`パラメータを調整
- 図のサイズ: `figsize`パラメータを変更

この可視化システムにより、実験結果から直接論文品質の図表を生成し、研究成果の効果的な伝達を支援します。
