"""Draw the HW1 figures from results/results.csv.

Outputs:
    results/fig1_knn_k.png        ItemKNN vs UserKNN over k (NDCG@10, Recall@10)
    results/fig2_bpr_embedding.png BPR NDCG@10 over embedding size (reg_lambda=0.1)
    results/fig3_bpr_reg.png       BPR NDCG@10 over reg_lambda (embedding_size=64)
and prints the (3) comparison table (ItemKNN k=100 / SLIM / EASE) as markdown.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

RESULTS = os.path.join("results", "results.csv")
COLORS = {"item": "#2a78d6", "user": "#eb6834", "bpr": "#2a78d6"}
METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "axes.edgecolor": "#b3b2ad",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor="white", markeredgewidth=1.5, label=label)


def fig_knn(df):
    knn = df[df.exp == "knn"].copy()
    knn["k"] = knn["k"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, metric in zip(axes, ["ndcg@10", "recall@10"]):
        for method, name in [("item", "ItemKNN"), ("user", "UserKNN")]:
            d = knn[knn.knn_method == method].sort_values("k")
            line(ax, d["k"], d[metric], COLORS[method], name)
            ax.annotate(name, (d["k"].iloc[-1], d[metric].iloc[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=10, color="#52514e")
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 100, 500])
        ax.set_xticklabels([1, 5, 10, 100, 500])
        ax.set_xlabel("k (number of neighbors)")
        ax.set_ylabel(metric.upper())
        ax.set_xlim(right=1400)
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("ItemKNN vs UserKNN on ml-100k", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join("results", "fig1_knn_k.png"), dpi=200, bbox_inches="tight")


def fig_bpr(df, exp, xcol, xlabel, fname, title):
    d = df[df.exp == exp].copy()
    d[xcol] = d[xcol].astype(float)
    d = d.sort_values(xcol)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    line(ax, d[xcol], d["ndcg@10"], COLORS["bpr"])
    for x, y in zip(d[xcol], d["ndcg@10"]):
        ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9, color="#52514e")
    ax.set_xscale("log")
    xs = list(d[xcol])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NDCG@10")
    ax.set_title(title)
    ymin, ymax = d["ndcg@10"].min(), d["ndcg@10"].max()
    pad = max((ymax - ymin) * 0.6, 0.005)
    ax.set_ylim(ymin - pad, ymax + pad)
    fig.tight_layout()
    fig.savefig(os.path.join("results", fname), dpi=200, bbox_inches="tight")


def table_i2i(df):
    rows = [df[(df.exp == "knn") & (df.knn_method == "item") & (df.k.astype(int) == 100)].iloc[0],
            df[df.model == "SLIMElastic"].iloc[0],
            df[df.model == "EASE"].iloc[0]]
    names = ["ItemKNN (k=100)", "SLIM", "EASE"]
    print("| Model | " + " | ".join(m.upper() for m in METRICS) + " |")
    print("|---|" + "---|" * len(METRICS))
    for name, r in zip(names, rows):
        print(f"| {name} | " + " | ".join(f"{r[m]:.4f}" for m in METRICS) + " |")


def main():
    df = pd.read_csv(RESULTS)
    fig_knn(df)
    fig_bpr(df, "bpr_emb", "embedding_size", "Embedding size",
            "fig2_bpr_embedding.png", "BPR, reg_lambda = 0.1")
    fig_bpr(df, "bpr_reg", "reg_lambda", "Regularization lambda",
            "fig3_bpr_reg.png", "BPR, embedding size = 64")
    table_i2i(df)


if __name__ == "__main__":
    main()
