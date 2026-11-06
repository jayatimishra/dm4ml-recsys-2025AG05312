# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""MODEL EVALUATION: comparison table + charts for all models on the test window.

Reads the test-window results produced by training (each model evaluated once,
with hyper-parameters chosen on the validation window) and builds:
  reports/model/results_table.md / .csv   all models, lift vs the popularity baseline
  reports/model/ranking_metrics.png       Precision/Recall/NDCG/HitRate @K
  reports/model/beyond_accuracy.png       Coverage and Diversity
  reports/model/metrics.json              flat metrics for `dvc metrics diff`
The comparison artifacts are also attached to the MLflow release run.
"""
from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from src.common import (MLFLOW_TRACKING_URI, REPORTS, SERIES, TOP_K, WATERMARK, get_logger, new_fig,
                        read_csv, save_fig, write_csv, write_json)

log = get_logger("evaluation")
OUT = REPORTS / "model"
K = TOP_K
LABELS = {"popularity": "Popularity (baseline)", "item_knn": "Item-KNN CF",
          "svd_mf": "SVD matrix factorisation", "hybrid": "Hybrid (SVD + content + popularity)"}
ORDER = ["popularity", "item_knn", "svd_mf", "hybrid"]


def grouped_bars(df, metrics, names, title, path):
    fig, ax = new_fig(8, 4)
    x = np.arange(len(metrics))
    w = 0.8 / len(df)
    for i, r in enumerate(df.itertuples()):
        ax.bar(x + (i - (len(df) - 1) / 2) * w, [getattr(r, m) for m in metrics], w * 0.92,
               color=SERIES[i], label=LABELS[r.model])
    ax.set_xticks(x, names)
    ax.set_title(title)
    ax.set_ylim(0, df[metrics].max().max() * 1.3)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    save_fig(fig, path)


def main() -> dict:
    test = read_csv(OUT / "test_results.csv")
    test["order"] = test.model.map({m: i for i, m in enumerate(ORDER)})
    test = test.sort_values("order").drop(columns="order").reset_index(drop=True)
    base = test.set_index("model").loc["popularity"]
    cols = [f"precision_at_{K}", f"recall_at_{K}", f"ndcg_at_{K}", f"map_at_{K}", f"hit_rate_at_{K}",
            "coverage", "diversity", "novelty"]
    table = test[["model", "params"] + cols].copy()
    table["ndcg_lift_vs_baseline"] = table[f"ndcg_at_{K}"] / base[f"ndcg_at_{K}"] - 1
    write_csv(table, OUT / "results_table.csv")

    summary = json.loads((OUT / "training_summary.json").read_text())
    winner = summary["winner"]
    head = ["Model", "Params"] + [c.replace(f"_at_{K}", f"@{K}") for c in cols] + ["NDCG lift"]
    lines = [f"<!-- {WATERMARK} -->", f"# Model evaluation results (test window, K={K})", "",
             f"Evaluated on {int(test.n_eval_users.iloc[0])} users with at least one new add-to-cart or "
             f"purchase in the 14-day test window. Selected model: **{LABELS[winner]}**.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in table.itertuples():
        name = f"**{LABELS[r.model]}**" if r.model == winner else LABELS[r.model]
        vals = [f"{getattr(r, c):.4f}" if c != "novelty" else f"{r.novelty:.2f}" for c in cols]
        lines.append(f"| {name} | `{r.params}` | " + " | ".join(vals) + f" | {r.ndcg_lift_vs_baseline:+.1%} |")
    rating = read_csv(OUT / "rating_results.csv")
    lines += ["", "## Explicit rating prediction (supplementary)", "",
              "| Model | RMSE | MAE | Test ratings |", "|---|---|---|---|"]
    lines += [f"| {r.model} | {r.rmse:.4f} | {r.mae:.4f} | {r.n_ratings} |" for r in rating.itertuples()]
    (OUT / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    grouped_bars(test, [f"precision_at_{K}", f"recall_at_{K}", f"ndcg_at_{K}", f"hit_rate_at_{K}"],
                 [f"Precision@{K}", f"Recall@{K}", f"NDCG@{K}", f"HitRate@{K}"],
                 f"Accuracy metrics on the test window (K={K})", OUT / "ranking_metrics.png")
    grouped_bars(test, ["coverage", "diversity"], ["Catalogue coverage", "Intra-list diversity"],
                 "Beyond-accuracy metrics", OUT / "beyond_accuracy.png")

    w = test.set_index("model").loc[winner]
    flat = {"winner": winner, **{f"winner_{c}": round(float(w[c]), 5) for c in cols},
            **{f"baseline_{c}": round(float(base[c]), 5) for c in cols},
            **{f"{r.model}_rmse": round(float(r.rmse), 5) for r in rating.itertuples()}}
    write_json(flat, OUT / "metrics.json")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=summary["release_run_id"]):
        for f in ["results_table.md", "ranking_metrics.png", "beyond_accuracy.png", "rating_results.csv"]:
            mlflow.log_artifact(str(OUT / f), artifact_path="evaluation")
    log.info("Evaluation table written; winner=%s ndcg@%d=%.4f vs baseline %.4f", winner, K,
             w[f"ndcg_at_{K}"], base[f"ndcg_at_{K}"])
    return flat


if __name__ == "__main__":
    main()
