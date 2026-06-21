# Data Generation (Color)

Color-based document image generator for multiple text sources. Produces images with targeted color manipulations for sentiment analysis and question-answering experiments.

## Quick Start

### 1. Generate SQuAD Experiments (Recommended)

For complete SQuAD experiments including CLIP-based decoy selection:

```bash
# First, precompute CLIP embeddings (one-time setup)
python scripts/precompute_clip_embeddings.py

# Generate all SQuAD experiment modes
python src/data_generation/color/generate_by_color.py \
  --datasets squad \
  --squad-exp-modes answer_span decoy_top1 decoy_topn bipolar \
  --n-samples 50
```

### 2. Generate Sentiment Experiments

```bash
python src/data_generation/color/generate_by_color.py \
  --datasets imdb \
  --n-samples 100
```

## Core Features

### Supported Datasets
- **`squad`**: Question-answering with CLIP-based semantic decoy selection
- **`imdb`**: Sentiment analysis (positive/negative words)
- **`custom_short`**: Controlled short sentiment texts
- **`custom_long`**: Controlled long sentiment texts

### SQuAD Experiment Modes
- **`answer_span`**: Colors only the correct answer words
- **`decoy_top1`**: Colors the most semantically similar incorrect word
- **`decoy_topn`**: Colors the top 5 semantically similar incorrect words  
- **`bipolar`**: Colors both correct answers and decoys simultaneously

### Color Conditions
- **Colors**: red, green, blue, yellow, cyan, magenta
- **Intensities**: subtle, mild, strong
- **Baseline**: black text (no color manipulation)
- **Bipolar**: Two categories colored simultaneously (when applicable)

## CLI Reference

### Basic Usage
```bash
python src/data_generation/color/generate_by_color.py \
  --datasets {squad|imdb|custom_short|custom_long} \
  --n-samples <NUMBER> \
  [OPTIONS]
```

### Key Arguments
- `--datasets`: Space-separated list of datasets to generate
- `--n-samples`: Number of samples per dataset (default: 50)
- `--squad-exp-modes`: SQuAD experiment types (default: answer_span decoy_top1)
- `--output-root`: Base output directory (default: data/processed/color)
- `--width`, `--height`: Image dimensions (default: 1400x1000)
- `--title-font-size`, `--body-font-size`: Font sizes (default: 40, 28)
- `--aa-mode`: Anti-aliasing setting {aa_on|aa_off} (default: aa_on)

### Advanced Options
- `--include-baseline`: Include grayscale baseline images
- `--include-threshold`: Include binary threshold images
- `--font-family`: Font to use (default: DroidSans)

## Examples

### Complete SQuAD Experiment Setup
```bash
# 1. Precompute embeddings (one-time)
python scripts/precompute_clip_embeddings.py

# 2. Generate small test dataset
python src/data_generation/color/generate_by_color.py \
  --datasets squad \
  --squad-exp-modes answer_span decoy_top1 decoy_topn bipolar \
  --n-samples 10

# 3. Generate production dataset
python src/data_generation/color/generate_by_color.py \
  --datasets squad \
  --squad-exp-modes answer_span decoy_top1 decoy_topn bipolar \
  --n-samples 200
```

### Multiple Dataset Generation
```bash
python src/data_generation/color/generate_by_color.py \
  --datasets imdb squad \
  --n-samples 100 \
  --squad-exp-modes answer_span decoy_top1
```

### High-Resolution Images
```bash
python src/data_generation/color/generate_by_color.py \
  --datasets squad \
  --width 1920 --height 1080 \
  --title-font-size 48 --body-font-size 32 \
  --n-samples 50
```

## Output Structure

### SQuAD Experiments (Independent Datasets)
```
data/processed/color/qa/
├── squad_answer_span/DroidSans/aa_on/
│   ├── experiment_config.json
│   └── text_XXX/
│       ├── baseline/baseline_black.png
│       ├── subtle/answer_span_only_red_subtle.png
│       ├── mild/answer_span_only_green_mild.png
│       └── strong/answer_span_only_blue_strong.png
├── squad_decoy_top1/DroidSans/aa_on/
│   ├── experiment_config.json
│   └── text_XXX/
│       ├── baseline/baseline_black.png
│       └── [decoy_top_1_only_COLOR_INTENSITY.png files]
├── squad_decoy_topn/DroidSans/aa_on/
│   └── [similar structure with decoy_top_n conditions]
└── squad_bipolar/DroidSans/aa_on/
    └── [bipolar combinations of answer + decoy]
```

### Sentiment Experiments
```
data/processed/color/sentiment/imdb/DroidSans/aa_on/
├── experiment_config.json
└── text_XXX/
    ├── baseline/baseline_black.png
    ├── subtle/positive_only_red_subtle.png
    ├── mild/negative_only_green_mild.png
    ├── strong/bipolar_red_green_strong.png
    └── [all color/intensity combinations]
```

## Configuration Files

Each dataset generates an `experiment_config.json` containing:
- Image paths and metadata
- Target word categories and positions
- QA pairs (for SQuAD) or sentiment labels (for IMDb)
- Color manipulation settings

### SQuAD Config Example
```json
{
  "dataset_name": "squad_answer_span",
  "task_name": "qa",
  "experiments": [
    {
      "title": "Sample Title",
      "content": "Text content with answer spans...",
      "qa": {
        "question": "What is the answer?",
        "answers": ["correct answer"]
      },
      "target_word_categories": {
        "answer_span": ["correct", "answer"]
      },
      "images": {
        "baseline_black": "path/to/baseline.png",
        "answer_span_only_red_mild": "path/to/colored.png"
      }
    }
  ]
}
```

## CLIP Embedding Setup

### Prerequisites
```bash
pip install torch transformers datasets scipy tqdm
```

### Precompute Embeddings
```bash
python scripts/precompute_clip_embeddings.py
```

This creates `data/processed/squad_clip_embeddings.pt` containing 52,000+ word embeddings for semantic similarity calculations.

## Troubleshooting

### Common Issues
1. **CLIP embeddings not found**: Run `python scripts/precompute_clip_embeddings.py`
2. **PyTorch import errors**: Ensure torch is properly installed in your environment
3. **Slow generation**: Reduce `--n-samples` for testing, use progress bars to monitor
4. **Memory issues**: Process datasets individually or reduce batch sizes

### Performance Tips
- Use `--n-samples 10` for quick testing
- Precompute embeddings once, reuse across experiments  
- Generate datasets independently to avoid memory buildup
- Monitor disk space (images can be large with high resolution)

## Research Applications

### Question Answering Analysis
- **Accuracy Impact**: Compare EM/F1 scores between `baseline_black` and `answer_span_only_*` conditions
- **Induced Error Rate**: Measure VLM misleading success with `decoy_top1_only_*` conditions
- **Semantic Confusion**: Analyze top-1 vs top-N decoy effectiveness

### Sentiment Analysis
- **Bias Measurement**: Quantify sentiment classification changes with color manipulation
- **Color Psychology**: Test different color-emotion associations across models

### Multi-modal Robustness
- **Visual Prompt Injection**: Assess VLM vulnerability to stealth visual cues
- **Cross-task Consistency**: Compare color effects across QA vs sentiment tasks
