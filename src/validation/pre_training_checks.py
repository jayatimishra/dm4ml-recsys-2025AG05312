# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""PRE-TRAINING VALIDATION with Pandera (second validation gate).

Stage 2 (validate.py) is the PRE-INGESTION gate on raw source data.
This module is the PRE-TRAINING gate on what the model will actually consume:
  1. Pandera schemas on the train/valid/test split files and feature snapshots
     (types, ranges, domains, uniqueness).
  2. Temporal leakage check: every train event is strictly before every test event,
     and the training feature snapshot (as_of) is not later than the first test event.
  3. Referential check: test users/items exist in the dimension tables.
  4. Drift check: Population Stability Index (PSI) of event-type mix and category mix
     between train and test (warn > 0.10, fail > 0.25).
Report: reports/validation/pre_training_report.json. Any failure stops the pipeline.
"""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera import Check, Column

from src.common import FEATURE_STORE, PROCESSED, REPORTS, get_logger, read_parquet, write_json
from src.transformation.transform import DB as WAREHOUSE_DB

log = get_logger("validation.pre_training")

INTERACTIONS = pa.DataFrameSchema({
    "event_id": Column(str, unique=True),
    "user_id": Column(int, Check.gt(0)),
    "product_id": Column(int, Check.in_range(1001, 1600)),
    "event_type": Column(str, Check.isin(["view", "add_to_cart", "purchase"])),
    "event_weight": Column(float, Check.isin([1.0, 3.0, 5.0])),
    "rating": Column(float, Check.in_range(1, 5), nullable=True),
    "event_timestamp": Column("datetime64[ns]"),
}, strict=False, name="interactions_split")

USER_FEATURES = pa.DataFrameSchema({
    "user_id": Column(int, unique=True),
    "u_n_events": Column(int, Check.ge(0)),
    "u_conversion_rate": Column(float, Check.in_range(0, 1), nullable=True),
    "u_avg_rating_given": Column(float, Check.in_range(1, 5), nullable=True),
    "u_days_since_last_event": Column(float, Check.ge(0), nullable=True),
    "age": Column(int, Check.in_range(13, 100)),
}, strict=False, name="user_profile_behaviour")

ITEM_FEATURES = pa.DataFrameSchema({
    "product_id": Column(int, unique=True),
    "i_n_events": Column(int, Check.ge(0)),
    "i_conversion_rate": Column(float, Check.ge(0), nullable=True),
    "i_avg_rating": Column(float, Check.in_range(1, 5), nullable=True),
    "price": Column(float, Check.gt(0)),
    "category": Column(str, Check.str_length(min_value=1)),
}, strict=False, name="item_popularity_quality")

UI_FEATURES = pa.DataFrameSchema({
    "user_id": Column(int), "product_id": Column(int),
    "ui_strength": Column(float, Check.ge(-2)),   # decayed weight >= 0, plus (rating-3) >= -2
    "ui_n_events": Column(int, Check.ge(1)),
}, strict=False, unique=["user_id", "product_id"], name="user_item_interaction")


def psi(expected: pd.Series, actual: pd.Series, eps: float = 1e-4) -> float:
    cats = expected.index.union(actual.index)
    e = expected.reindex(cats, fill_value=0) / expected.sum() + eps
    a = actual.reindex(cats, fill_value=0) / actual.sum() + eps
    return float(((a - e) * np.log(a / e)).sum())


def main() -> dict:
    snaps = json.loads((REPORTS / "transform_summary.json").read_text())["snapshots"]
    split = {k: read_parquet(PROCESSED / "splits" / f"{k}_interactions.parquet") for k in ("train", "valid", "test")}
    for df in split.values():
        df["event_timestamp"] = pd.to_datetime(df.event_timestamp).astype("datetime64[ns]")
    feats = {n: read_parquet(FEATURE_STORE / "offline" / n / f"as_of={snaps['train'][:10]}.parquet")
             for n in ("user_profile_behaviour", "item_popularity_quality", "user_item_interaction")}

    results, failed = [], False
    jobs = [(INTERACTIONS, split["train"], "train_interactions"), (INTERACTIONS, split["test"], "test_interactions"),
            (USER_FEATURES, feats["user_profile_behaviour"], "user_features@train"),
            (ITEM_FEATURES, feats["item_popularity_quality"], "item_features@train"),
            (UI_FEATURES, feats["user_item_interaction"], "user_item_features@train")]
    for schema, df, name in jobs:
        try:
            schema.validate(df, lazy=True)
            results.append({"check": f"pandera:{name}", "passed": True, "rows": len(df), "detail": "schema OK"})
        except pa.errors.SchemaErrors as err:
            failed = True
            fc = err.failure_cases
            results.append({"check": f"pandera:{name}", "passed": False, "rows": len(df),
                            "detail": fc[["column", "check", "failure_case"]].head(10).astype(str).to_dict("records")})

    tr_max, te_min = split["train"].event_timestamp.max(), split["test"].event_timestamp.min()
    ok = tr_max < te_min and pd.Timestamp(snaps["train"]) <= te_min
    failed |= not ok
    results.append({"check": "temporal_leakage", "passed": bool(ok),
                    "detail": f"train max={tr_max}, test min={te_min}, train features as_of={snaps['train']}"})

    with sqlite3.connect(WAREHOUSE_DB) as c:
        u = set(pd.read_sql("SELECT user_id FROM dim_user", c).user_id)
        p = pd.read_sql("SELECT product_id, category FROM dim_product", c)
    ok = split["test"].user_id.isin(u).all() and split["test"].product_id.isin(set(p.product_id)).all()
    failed |= not ok
    results.append({"check": "test_referential_integrity", "passed": bool(ok), "detail": "test keys in dims"})

    cat = p.set_index("product_id").category
    for name, f in [("event_type_mix", lambda d: d.event_type.value_counts()),
                    ("category_mix", lambda d: d.product_id.map(cat).value_counts())]:
        v = psi(f(split["train"]), f(split["test"]))
        level = "ok" if v <= 0.10 else ("warn" if v <= 0.25 else "fail")
        failed |= level == "fail"
        results.append({"check": f"drift_psi:{name}", "passed": level != "fail", "psi": round(v, 4),
                        "detail": level})

    report = {"gate_passed": not failed, "results": results}
    write_json(report, REPORTS / "validation" / "pre_training_report.json")
    for r in results:
        log.info("%-38s %s %s", r["check"], "PASS" if r["passed"] else "FAIL", r.get("psi", ""))
    if failed:
        raise RuntimeError("Pre-training validation FAILED, see reports/validation/pre_training_report.json")
    return report


if __name__ == "__main__":
    main()
