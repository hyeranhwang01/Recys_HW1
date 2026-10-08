# Recys_HW1

Introduction to Recommender Systems (Fall 2026) — Homework #1

Experiments with [RecBole](https://github.com/RUCAIBox/RecBole) on the `ml-100k` dataset.

1. Neighborhood-based CF: ItemKNN / UserKNN, k ∈ {1, 5, 10, 100, 500}
2. Latent factor model: BPR, embedding size ∈ {32, 64, 128, 256}, regularization ∈ {0.01, 0.1, 0.5}
3. Item-to-item CF: SLIM (SLIMElastic), EASE vs. ItemKNN
