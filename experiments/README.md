### Experiments

`experiments/color/run.py` runs inference over color-manipulated images using multiple VLMs and saves raw outputs.

Workflow
1) Generate images (see `src/data_generation/README.md`).
2) Run inference (GPU node recommended):
```
python experiments/color/run.py --manifest data/processed/color/<dataset>/experiment_config.json --model <model-id>
```
3) Analyze results:
```
python src/evaluation/calc_sentiment_score.py results/color/<model>/<dataset>/<run_id>/raw_outputs/experiment_results.json
python src/evaluation/plot_phase_transition.py results/color/<model>/<dataset>/<run_id>/raw_outputs/experiment_results.json --out results/color/<model>/<dataset>/<run_id>/analysis/phase_curve.png
```

Output
```
results/color/<model>/<dataset>/<run_id>/
  ├── raw_outputs/experiment_results.json
  ├── logs/
  ├── analysis/ (CSV data etc.)
  └── figures/ (visualizations etc.)
```
