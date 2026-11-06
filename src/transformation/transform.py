# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 4: DATA TRANSFORMATION + STORAGE (ELT into a SQLite star schema).

1. Load processed (silver) data into the gold star schema defined in src/transformation/sql/schema.sql
   (dim_user, dim_product, dim_date, fact_interactions) with constraints enforced
   by the database (CHECK, PRIMARY KEY, REFERENCES).
2. Run the feature transformations in src/transformation/sql/features.sql INSIDE the warehouse for
   three point-in-time snapshots:
     * validation snapshot as_of = end - 2*TEST_DAYS (hyper-parameter tuning only)
     * training snapshot  as_of = end - TEST_DAYS   (model never sees the test window)
     * serving snapshot   as_of = end               (used for live recommendations)
3. Export time-based train/valid/test split files and derived datasets
   (user activity summary, item popularity, sparse user-item matrix) to data/processed/.
4. Write a data dictionary and sample analytical query results to reports/.
"""
from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd

from src.common import (WATERMARK, PROCESSED, REPORTS, ROOT, SQL_DIR, TEST_DAYS, WAREHOUSE, get_logger, read_parquet, write_parquet,
                    stamp_sqlite, write_csv, write_json)

log = get_logger("transformation")
DB = WAREHOUSE / "recsys_warehouse.db"
SPLITS = PROCESSED / "splits"
DERIVED = PROCESSED / "derived"
HALF_LIFE_DAYS = 30.0
FEATURE_TABLES = ["feat_user", "feat_item", "agg_user_item", "feat_user_category"]


def connect() -> sqlite3.Connection:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.create_function("power", 2, lambda a, b: float(a) ** float(b), deterministic=True)
    return conn


def snapshot_dates(conn) -> dict:
    mx = pd.Timestamp(conn.execute("SELECT MAX(event_timestamp) FROM fact_interactions").fetchone()[0])
    end = mx.normalize() + pd.Timedelta(days=1)
    return {"valid": (end - pd.Timedelta(days=2 * TEST_DAYS)).strftime("%Y-%m-%d %H:%M:%S"),
            "train": (end - pd.Timedelta(days=TEST_DAYS)).strftime("%Y-%m-%d %H:%M:%S"),
            "serve": end.strftime("%Y-%m-%d %H:%M:%S")}


def load_star_schema(conn) -> dict:
    conn.executescript((SQL_DIR / "schema.sql").read_text())
    users = read_parquet(PROCESSED / "clean" / "users.parquet")
    products = read_parquet(PROCESSED / "clean" / "products.parquet")
    inter = read_parquet(PROCESSED / "clean" / "interactions.parquet")

    users = users.assign(signup_date=users.signup_date.dt.strftime("%Y-%m-%d"),
                         age_imputed=users.age_imputed.astype(int),
                         is_prime_member=users.is_prime_member.astype(int))
    products = products.assign(created_at=products.created_at.dt.strftime("%Y-%m-%d %H:%M:%S"),
                               price_imputed=products.price_imputed.astype(int),
                               in_stock=products.in_stock.astype(int))
    dates = pd.date_range(inter.event_date.min(), inter.event_date.max(), freq="D")
    dim_date = pd.DataFrame({"date_key": dates.strftime("%Y-%m-%d"), "year": dates.year,
                             "month": dates.month, "day": dates.day, "day_of_week": dates.dayofweek,
                             "is_weekend": (dates.dayofweek >= 5).astype(int),
                             "week_of_year": dates.isocalendar().week.astype(int).values})
    fact = inter.assign(date_key=inter.event_date.dt.strftime("%Y-%m-%d"),
                        event_timestamp=inter.event_timestamp.dt.strftime("%Y-%m-%d %H:%M:%S"))
    fact = fact[["event_id", "user_id", "product_id", "date_key", "event_type", "event_weight",
                 "rating", "event_timestamp", "session_id", "device"]]

    for name, df in [("dim_user", users), ("dim_product", products), ("dim_date", dim_date),
                     ("fact_interactions", fact)]:
        df.to_sql(name, conn, if_exists="append", index=False, chunksize=5000)
    conn.commit()
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ["dim_user", "dim_product", "dim_date", "fact_interactions"]}
    orphans = conn.execute("PRAGMA foreign_key_check").fetchall()
    if orphans:
        raise RuntimeError(f"Referential integrity violated: {orphans[:5]}")
    log.info("Star schema loaded: %s", counts)
    return counts


def build_features(conn, as_of: str) -> dict:
    blocks = re.split(r"^-- name: (\w+)\s*$", (SQL_DIR / "features.sql").read_text(), flags=re.M)
    out = {}
    for name, body in zip(blocks[1::2], blocks[2::2]):
        query = "\n".join(l for l in body.splitlines() if not l.strip().startswith("--")).strip().rstrip(";")
        df = pd.read_sql(query, conn, params={"as_of": as_of, "half_life_days": HALF_LIFE_DAYS})
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (name,)).fetchone()
        if exists:
            conn.execute(f"DELETE FROM {name} WHERE as_of = ?", (as_of,))     # idempotent re-run
        df.to_sql(name, conn, if_exists="append", index=False)
        out[name] = len(df)
    conn.commit()
    log.info("Features materialised as_of=%s: %s", as_of, out)
    return out


ANALYTICS = {
    "top_categories_by_revenue": """
        SELECT p.category, COUNT(*) AS purchases, ROUND(SUM(p.price), 0) AS revenue_inr
        FROM fact_interactions f JOIN dim_product p USING (product_id)
        WHERE f.event_type = 'purchase' GROUP BY p.category ORDER BY revenue_inr DESC""",
    "weekly_conversion": """
        SELECT d.year || '-W' || printf('%02d', d.week_of_year) AS week,
               SUM(f.event_type = 'view') AS views, SUM(f.event_type = 'purchase') AS purchases,
               ROUND(1.0 * SUM(f.event_type = 'purchase') / SUM(f.event_type = 'view'), 4) AS conv
        FROM fact_interactions f JOIN dim_date d USING (date_key) GROUP BY week ORDER BY week""",
    "device_mix": """
        SELECT device, event_type, COUNT(*) AS n FROM fact_interactions
        GROUP BY device, event_type ORDER BY device, event_type""",
}


def export_datasets(conn, snaps: dict) -> dict:
    """Time-based train/valid/test split files + derived datasets (activity, popularity, UI matrix)."""
    from scipy import sparse

    fact = pd.read_sql("SELECT * FROM fact_interactions", conn, parse_dates=["event_timestamp"])
    v, t, e = (pd.Timestamp(snaps[k]) for k in ("valid", "train", "serve"))
    splits = {"train": fact[fact.event_timestamp < t],                      # model fitting
              "valid": fact[(fact.event_timestamp >= v) & (fact.event_timestamp < t)],  # tuning window
              "test": fact[(fact.event_timestamp >= t) & (fact.event_timestamp < e)]}   # final evaluation
    out = SPLITS
    for k, df in splits.items():
        write_parquet(df, out / f"{k}_interactions.parquet")
    info = {k: {"rows": len(df), "users": int(df.user_id.nunique()), "items": int(df.product_id.nunique()),
                "from": str(df.event_timestamp.min()), "to": str(df.event_timestamp.max())}
            for k, df in splits.items()}
    info["note"] = "valid is the last 14 days INSIDE train, used only for hyper-parameter tuning"

    serve = snaps["serve"]
    for table, name in [("feat_user", "user_activity_summary"), ("feat_item", "item_popularity"),
                        ("agg_user_item", "user_item_interactions_long")]:
        df = pd.read_sql(f"SELECT * FROM {table} WHERE as_of = ?", conn, params=(serve,))
        write_parquet(df, DERIVED / f"{name}.parquet")
    ui = pd.read_sql("SELECT user_id, product_id, ui_strength FROM agg_user_item WHERE as_of = ?",
                     conn, params=(snaps["train"],))
    users = np.sort(ui.user_id.unique())
    items = np.sort(pd.read_sql("SELECT product_id FROM dim_product", conn).product_id.values)
    m = sparse.csr_matrix((ui.ui_strength.values,
                           (np.searchsorted(users, ui.user_id), np.searchsorted(items, ui.product_id))),
                          shape=(len(users), len(items)))
    np.savez_compressed(DERIVED / "user_item_matrix_train.npz", data=m.data, indices=m.indices,
                        indptr=m.indptr, shape=np.array(m.shape), watermark=np.array(WATERMARK))  # CSR parts
    write_parquet(pd.DataFrame({"row": range(len(users)), "user_id": users}), DERIVED / "matrix_user_index.parquet")
    write_parquet(pd.DataFrame({"col": range(len(items)), "product_id": items}), DERIVED / "matrix_item_index.parquet")
    info["user_item_matrix_train"] = {"shape": list(m.shape), "nnz": int(m.nnz),
                                      "density": round(m.nnz / (m.shape[0] * m.shape[1]), 5)}
    log.info("Splits exported: %s", {k: v["rows"] for k, v in info.items() if isinstance(v, dict) and "rows" in v})
    return info


def main() -> dict:
    conn = connect()
    try:
        counts = load_star_schema(conn)
        snaps = snapshot_dates(conn)
        feats = {k: build_features(conn, v) for k, v in snaps.items()}
        splits = export_datasets(conn, snaps)
        stamp_sqlite(conn)
        for name, q in ANALYTICS.items():
            write_csv(pd.read_sql(q, conn), REPORTS / "analytics" / f"{name}.csv")
        # data dictionary straight from the catalog
        rows = []
        for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            for cid, col, typ, notnull, _, pk in conn.execute(f"PRAGMA table_info({t})"):
                rows.append({"table": t, "column": col, "type": typ or "inferred",
                             "not_null": bool(notnull), "primary_key": bool(pk)})
        write_csv(pd.DataFrame(rows), REPORTS / "data_dictionary.csv")
    finally:
        conn.close()
    result = {"warehouse": str(DB.relative_to(ROOT)), "row_counts": counts, "snapshots": snaps,
              "feature_rows": feats, "half_life_days": HALF_LIFE_DAYS, "splits": splits}
    write_json(result, REPORTS / "transform_summary.json")
    return result


if __name__ == "__main__":
    main()
