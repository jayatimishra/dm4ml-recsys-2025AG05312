# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 6: MODEL TRAINING, EVALUATION, EXPERIMENT TRACKING + REGISTRY (MLflow).

Temporal protocol (no leakage, no test-set snooping):
  valid as_of  ----14d---->  train as_of  ----14d---->  serve as_of
  |  tune on this window  |  evaluate ONCE here      |
  1. Tuning:   fit on features as_of=valid, score on add_to_cart/purchase events in
               [valid, train) for items the user had NOT interacted with before.
  2. Testing:  refit each model family with its best config on as_of=train,
               evaluate once on [train, serve).
  3. Release:  refit the winning family on as_of=serve (all data), log it as an
               MLflow pyfunc model, register it and set alias 'champion'.
Every run logs params, metrics, the feature-view versions and data hashes it was
trained on (lineage), and a metrics plot.
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import tempfile
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.common import (EXPERIMENT_NAME, MLFLOW_ARTIFACTS, MLFLOW_TRACKING_URI, MODELS, PROCESSED, ROOT,
                    REGISTERED_MODEL_NAME, REPORTS, ROLL_NO, TOP_K, file_md5, get_logger, write_csv,
                    write_json)
from src.evaluation.metrics import ranking_metrics, rating_metrics
from src.training.recommenders import MODELS as ALGOS, RecommenderPyfunc
from src.training.rating_model import BiasModel, GlobalMean
from src.transformation.transform import DB as WAREHOUSE_DB
from src.features.feature_store import FeatureStore

log = get_logger("training")

SEARCH_SPACE = {
    "popularity": [{}],
    "item_knn": [{"neighbours": n} for n in (20, 50, 100, 200, 400)],
    "svd_mf": [{"factors": f} for f in (16, 32, 64)],
    "hybrid": [{"factors": f, "alpha": a, "beta": b}
               for f in (32, 64) for a in (0.1, 0.25) for b in (0.05, 0.15)],
}
PRIMARY = f"ndcg_at_{TOP_K}"


def load_inputs(fs: FeatureStore, as_of: str):
    return (fs.get_offline_snapshot("user_item_interaction", as_of),
            fs.get_offline_snapshot("item_popularity_quality", as_of),
            fs.get_offline_snapshot("user_category_affinity", as_of))


def holdout_truth(start: str, end: str, seen: pd.DataFrame) -> dict:
    with sqlite3.connect(WAREHOUSE_DB) as c:
        ev = pd.read_sql("""SELECT DISTINCT user_id, product_id FROM fact_interactions
                            WHERE event_timestamp >= ? AND event_timestamp < ?
                              AND event_type IN ('add_to_cart','purchase')""", c, params=(start, end))
    seen_pairs = set(zip(seen.user_id, seen.product_id))
    ev = ev[[(u, p) not in seen_pairs for u, p in zip(ev.user_id, ev.product_id)]]
    ev = ev[ev.user_id.isin(seen.user_id)]          # warm users only (cold start handled separately)
    return ev.groupby("user_id").product_id.apply(set).to_dict()


def lineage_tags(fs: FeatureStore, as_of: str) -> dict:
    tags = {"roll_no": ROLL_NO, "features_as_of": as_of}
    for v in ["user_item_interaction", "item_popularity_quality", "user_category_affinity"]:
        tags[f"feature_view.{v}.version"] = str(fs.view(v)["version"])
    for f in ["interactions", "users", "products"]:
        tags[f"data.processed.{f}.md5"] = file_md5(PROCESSED / "clean" / f"{f}.parquet")
    return tags


def evaluate_rating_models(fs: FeatureStore, snaps: dict) -> list:
    """RMSE/MAE for explicit ratings: train on ratings before the cutoff, test on the holdout window."""
    with sqlite3.connect(WAREHOUSE_DB) as c:
        q = "SELECT user_id, product_id, rating FROM fact_interactions WHERE rating IS NOT NULL AND {}"
        tr = pd.read_sql(q.format("event_timestamp < ?"), c, params=(snaps["train"],))
        te = pd.read_sql(q.format("event_timestamp >= ? AND event_timestamp < ?"), c,
                         params=(snaps["train"], snaps["serve"]))
    rows = []
    for model in (GlobalMean(), BiasModel(reg=5.0)):
        with mlflow.start_run(run_name=f"test-rating-{model.name}"):
            mlflow.set_tags({**lineage_tags(fs, snaps["train"]), "stage": "test", "task": "rating"})
            model.fit(tr)
            met = rating_metrics(te.rating, model.predict(te))
            mlflow.log_params({"algorithm": model.name, "reg": getattr(model, "reg", 0),
                               "n_train_ratings": len(tr)})
            mlflow.log_metrics(met)
            rows.append({"model": model.name, **met})
    write_csv(pd.DataFrame(rows), REPORTS / "model" / "rating_results.csv")
    log.info("Rating models: %s", rows)
    return rows


def main() -> dict:
    snaps = json.loads((REPORTS / "transform_summary.json").read_text())["snapshots"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACTS)
    mlflow.set_experiment(EXPERIMENT_NAME)
    fs = FeatureStore()

    # ---------------- 1. tuning on the validation window
    ui_v, it_v, af_v = load_inputs(fs, snaps["valid"])
    truth_v = holdout_truth(snaps["valid"], snaps["train"], ui_v)
    best, tuning_rows = {}, []
    with mlflow.start_run(run_name="tuning_validation_window") as parent:
        mlflow.set_tags({**lineage_tags(fs, snaps["valid"]), "stage": "tuning"})
        for algo, grid in SEARCH_SPACE.items():
            for params in grid:
                with mlflow.start_run(run_name=f"{algo}-{params}", nested=True):
                    m = ALGOS[algo](**params).fit(ui_v, it_v, af_v)
                    met = ranking_metrics(m, truth_v, TOP_K)
                    mlflow.log_params({"algorithm": algo, **params, "split": "valid"})
                    mlflow.log_metrics({k: v for k, v in met.items()})
                    tuning_rows.append({"model": algo, "params": json.dumps(params), **met})
                    if algo not in best or met[PRIMARY] > best[algo][1]:
                        best[algo] = (params, met[PRIMARY])
                    log.info("tune %-10s %-50s ndcg@10=%.4f", algo, params, met[PRIMARY])
    write_csv(pd.DataFrame(tuning_rows), REPORTS / "model" / "tuning_results.csv")

    # ---------------- 2. one-shot evaluation on the test window
    ui_t, it_t, af_t = load_inputs(fs, snaps["train"])
    truth_t = holdout_truth(snaps["train"], snaps["serve"], ui_t)
    test_rows = []
    for algo, (params, _) in best.items():
        with mlflow.start_run(run_name=f"test-{algo}"):
            mlflow.set_tags({**lineage_tags(fs, snaps["train"]), "stage": "test"})
            m = ALGOS[algo](**params).fit(ui_t, it_t, af_t)
            met = ranking_metrics(m, truth_t, TOP_K)
            mlflow.log_params({"algorithm": algo, **params, "split": "test"})
            mlflow.log_metrics(met)
            test_rows.append({"model": algo, "params": json.dumps(params), **met})
    test_df = pd.DataFrame(test_rows).sort_values(PRIMARY, ascending=False)
    write_csv(test_df, REPORTS / "model" / "test_results.csv")
    rating_rows = evaluate_rating_models(fs, snaps)
    winner = test_df.iloc[0]
    w_algo, w_params = winner.model, json.loads(winner.params)
    log.info("Winner on test: %s %s ndcg@10=%.4f", w_algo, w_params, winner[PRIMARY])

    # ---------------- 3. release: retrain on all data, log + register
    ui_s, it_s, af_s = load_inputs(fs, snaps["serve"])
    final = ALGOS[w_algo](**w_params).fit(ui_s, it_s, af_s)
    MODELS.mkdir(parents=True, exist_ok=True)
    with mlflow.start_run(run_name=f"release-{w_algo}") as run:
        mlflow.set_tags({**lineage_tags(fs, snaps["serve"]), "stage": "release",
                         "selected_on": f"test {PRIMARY}={winner[PRIMARY]:.4f}"})
        mlflow.log_params({"algorithm": w_algo, **w_params, "split": "all_data"})
        mlflow.log_metrics({f"test_{k}": float(winner[k]) for k in
                            [f"precision_at_{TOP_K}", f"recall_at_{TOP_K}", PRIMARY,
                             f"map_at_{TOP_K}", f"hit_rate_at_{TOP_K}", "coverage", "diversity", "novelty"]})
        with tempfile.TemporaryDirectory() as tmp:
            pkl = Path(tmp) / "recommender.pkl"
            with open(pkl, "wb") as f:
                pickle.dump(final, f)
            mlflow.log_artifact(str(REPORTS / "model" / "test_results.csv"))
            example = pd.DataFrame({"user_id": [float(final.user_ids[0])], "k": [5.0]})
            info = mlflow.pyfunc.log_model(
                name="model", python_model=RecommenderPyfunc(), artifacts={"model": str(pkl)},
                code_paths=[str(ROOT / "src")],
                input_example=example, registered_model_name=REGISTERED_MODEL_NAME)
        with open(MODELS / f"recommender_{ROLL_NO}.pkl", "wb") as f:   # local copy for DVC
            pickle.dump(final, f)
    client = MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'"))
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", str(version))
    client.update_model_version(REGISTERED_MODEL_NAME, str(version),
                                description=f"{w_algo} {w_params}; test {PRIMARY}={winner[PRIMARY]:.4f}; {ROLL_NO}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, str(version), "validation_status", "approved")

    summary = {"winner": w_algo, "params": w_params, "test_metrics": test_df.to_dict("records"),
               "best_tuning": {k: {"params": v[0], PRIMARY: v[1]} for k, v in best.items()},
               "registered_model": REGISTERED_MODEL_NAME, "version": version, "alias": "champion",
               "release_run_id": run.info.run_id, "n_test_users": len(truth_t),
               "n_valid_users": len(truth_v), "rating_models": rating_rows}
    write_json(summary, REPORTS / "model" / "training_summary.json")

    return summary


if __name__ == "__main__":
    main()
