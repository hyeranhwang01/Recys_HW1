# Recys_HW1

Introduction to Recommender Systems (Fall 2026) — Homework #1

Experiments with [RecBole](https://github.com/RUCAIBox/RecBole) on the `ml-100k` dataset.

1. Neighborhood-based CF: ItemKNN / UserKNN, k ∈ {1, 5, 10, 100, 500}
2. Latent factor model: BPR, embedding size ∈ {32, 64, 128, 256}, regularization ∈ {0.01, 0.1, 0.5}
3. Item-to-item CF: SLIM (SLIMElastic), EASE vs. ItemKNN

## Setup

```bash
conda create -n recsys python=3.10 -y
conda activate recsys
pip install -r requirements.txt
```

## Run

```bash
python run_experiments.py          # all experiments -> results/results.csv
python run_experiments.py --exp knn  # or: bpr, i2i
python plot_results.py             # figures -> results/*.png, prints the (3) table
```

`ml-100k` is downloaded automatically by RecBole on the first run.
Evaluation uses RecBole defaults: random 8:1:1 split, full ranking, top-10 metrics, seed 2026.

## Notes

- UserKNN is `ItemKNN` with `knn_method: 'user'` (the `--method` flag in the
  handout is not a RecBole config key and is silently ignored).
- RecBole's `BPR` has no `reg_lambda`; `models/bpr_reg.py` adds an L2 term
  `reg_lambda * (||u||² + ||i||²)` to the BPR loss so the regularization strength
  can be swept.
- SLIM is RecBole's `SLIMElastic`; SLIM and EASE use the default hyperparameters.

## Files

```
models/bpr_reg.py     BPR with L2 regularization
run_experiments.py    runs every configuration, one CSV row per run
plot_results.py       figures for the report
results/              results.csv and figures
```
