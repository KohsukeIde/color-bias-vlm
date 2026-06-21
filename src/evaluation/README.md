### Evaluation & Visualization

Tools
- `calc_sentiment_score.py`: IMDb results → overall/per-condition accuracy (heuristic label extraction)
- `calc_cer.py`: Character Error Rate utility
- `plot_phase_transition.py`: Phase transition curve over RGB intensity (needs `threshold_*` conditions)

Examples
```
python src/evaluation/calc_sentiment_score.py results/color/<model>/<dataset>/<run_id>/raw_outputs/experiment_results.json

python src/evaluation/plot_phase_transition.py \
  results/color/<model>/<dataset>/<run_id>/raw_outputs/experiment_results.json \
  --out results/color/<model>/<dataset>/<run_id>/analysis/phase_curve.png
```

Notes
- Phase plot uses coarse labels (accurate/neutral/biased/hallucination) from predicted text and optional ground-truth label in the manifest
- Ensure `--include-threshold` when generating data to enable phase plots








