"""Run all HW1 experiments on ml-100k and append results to results/results.csv.

Usage:
    python run_experiments.py            # run everything not yet in the CSV
    python run_experiments.py --exp knn  # only one experiment group (knn | bpr | i2i)

Each run is one row in the CSV, so an interrupted sweep can simply be re-run.
"""

import argparse
import csv
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from recbole.config import Config  # noqa: E402
from recbole.data import create_dataset, data_preparation  # noqa: E402
from recbole.quick_start import run_recbole  # noqa: E402
from recbole.utils import get_trainer, init_logger, init_seed  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.bpr_reg import BPRReg  # noqa: E402

RESULTS = os.path.join("results", "results.csv")
METRICS = ["recall@10", "mrr@10", "ndcg@10", "hit@10", "precision@10"]
FIELDS = ["exp", "model", "k", "knn_method", "embedding_size", "reg_lambda"] + METRICS

COMMON = {
    "dataset": "ml-100k",
    "seed": 2026,
    "reproducibility": True,
    "metrics": ["Recall", "MRR", "NDCG", "Hit", "Precision"],
    "topk": [10],
    "valid_metric": "NDCG@10",
    "show_progress": False,
    "save_dataset": False,
    "save_dataloaders": False,
}

BPR_COMMON = {
    "epochs": 100,
    "stopping_step": 10,
    "train_batch_size": 2048,
    "learning_rate": 0.001,
}


def run_one(model, config_dict):
    """Built-in RecBole models go through run_recbole() as in the README / quick_start.py.
    A custom model class (BPRReg) is not accepted by run_recbole, so for that case the
    same steps as run_recbole are executed here with the class passed to Config directly."""
    config_dict = {**COMMON, **config_dict}
    if isinstance(model, str):
        result = run_recbole(model=model, dataset="ml-100k", config_dict=config_dict)
        test_result = result["test_result"]
    else:
        config = Config(model=model, config_dict=config_dict)
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        logging.getLogger().setLevel(logging.WARNING)

        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)

        init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
        net = model(config, train_data._dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, net)

        trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    return {m: float(test_result[m]) for m in METRICS}


def experiments(which):
    runs = []
    if which in ("all", "knn"):
        for method in ("item", "user"):
            for k in (1, 5, 10, 100, 500):
                runs.append(("knn", "ItemKNN", {"k": k, "knn_method": method}))
    if which in ("all", "bpr"):
        for d in (32, 64, 128, 256):
            runs.append(("bpr_emb", BPRReg, {**BPR_COMMON, "embedding_size": d, "reg_lambda": 0.1}))
        for lam in (0.01, 0.1, 0.5):
            runs.append(("bpr_reg", BPRReg, {**BPR_COMMON, "embedding_size": 64, "reg_lambda": lam}))
    if which in ("all", "i2i"):
        runs.append(("i2i", "SLIMElastic", {}))
        runs.append(("i2i", "EASE", {}))
    return runs


def row_key(row):
    return tuple(str(row.get(f, "")) for f in FIELDS[: len(FIELDS) - len(METRICS)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["all", "knn", "bpr", "i2i"])
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    done = set()
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            done = {row_key(r) for r in csv.DictReader(f)}

    for exp, model, cfg in experiments(args.exp):
        name = model if isinstance(model, str) else model.__name__
        row = {"exp": exp, "model": name}
        for f in ("k", "knn_method", "embedding_size", "reg_lambda"):
            row[f] = cfg.get(f, "")
        if row_key(row) in done:
            print(f"skip {row}")
            continue

        print(f"run  {row}", flush=True)
        row.update(run_one(model, cfg))
        print(f"     -> " + ", ".join(f"{m}={row[m]:.4f}" for m in METRICS), flush=True)

        write_header = not os.path.exists(RESULTS)
        with open(RESULTS, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
