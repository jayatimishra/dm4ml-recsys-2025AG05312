# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Single source of truth for the pipeline DAG, shared by all orchestrators
(Prefect flow, Airflow DAG, plain-Python runner) and mirrored by dvc.yaml."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PARAMS = yaml.safe_load((ROOT / "configs" / "params.yaml").read_text())

# task_id -> (module, upstream task_ids, retries)
DAG = {
    "simulate_sources":    ("src.ingestion.generate_sources",     [],                       1),
    "ingest":              ("src.ingestion.ingest",               ["simulate_sources"],     3),
    "validate":            ("src.validation.validate",            ["ingest"],               0),  # gate: never retry bad data
    "prepare":             ("src.preparation.prepare",            ["validate"],             1),
    "transform":           ("src.transformation.transform",       ["prepare"],              1),
    "feature_store":       ("src.features.feature_store",         ["transform"],            1),
    "pre_training_checks": ("src.validation.pre_training_checks", ["feature_store"],        0),  # gate
    "train":               ("src.training.train",                 ["pre_training_checks"],  1),
    "evaluate":            ("src.evaluation.evaluate",            ["train"],                1),
    "serve_batch":         ("src.serving.serve",                  ["evaluate"],             1),
}


def topological_order() -> list[str]:
    done, order = set(), []
    while len(order) < len(DAG):
        for t, (_, ups, _) in DAG.items():
            if t not in done and all(u in done for u in ups):
                order.append(t)
                done.add(t)
    return order


def run_task(task_id: str):
    mod = importlib.import_module(DAG[task_id][0])
    if task_id == "simulate_sources":
        return mod.main(PARAMS["source"]["end_date"])
    if task_id == "ingest":
        return mod.main(PARAMS["ingest"]["mode"])
    if task_id == "serve_batch":
        return mod.batch()
    return mod.main()
