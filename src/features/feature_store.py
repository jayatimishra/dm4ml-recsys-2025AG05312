# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 5: FEATURE STORE (registry + offline store + online store).

A lightweight feature store with the same concepts as Feast:
  * Feature registry   data/features/registry.db  (feature_views + features tables):
                       name, entity, dtype, description, owner, source, version, tags.
                       A view's version is bumped automatically when its definition
                       (feature list + source SQL) changes, so models can pin a version.
  * Offline store      data/features/offline/<view>/as_of=<date>.parquet
                       full history of snapshots, used for TRAINING with a
                       point-in-time correct join (get_historical_features).
  * Online store       data/features/online.db  one latest row per entity key,
                       used for low-latency SERVING (get_online_features).
Training and serving read the same feature definitions, which removes
training/serving skew from duplicated feature code.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

from src.common import (FEATURE_STORE, REPORTS, ROLL_NO, ROOT, SQL_DIR, get_logger, read_parquet, stamp_sqlite,
                    utc_now, write_csv, write_json, write_parquet)
from src.transformation.transform import DB as WAREHOUSE_DB

log = get_logger("features")


@dataclass
class FeatureView:
    name: str
    entities: list
    source_table: str
    ttl_days: int
    description: str
    features: dict = field(default_factory=dict)   # feature -> (dtype, description)
    owner: str = f"ml-platform ({ROLL_NO})"
    tags: tuple = ()

    def definition_hash(self) -> str:
        sql = (SQL_DIR / "features.sql").read_text()
        block = sql.split(f"-- name: {self.source_table}")[1].split("-- name:")[0]
        payload = json.dumps({"f": self.features, "e": self.entities, "sql": block}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]


FEATURE_VIEWS = [
    FeatureView("user_profile_behaviour", ["user_id"], "feat_user", 30,
                "User demographics and engagement behaviour aggregated up to as_of", {
                    "u_n_events": ("int", "Total events by the user"),
                    "u_n_views": ("int", "Product views"),
                    "u_n_carts": ("int", "Add-to-cart events"),
                    "u_n_purchases": ("int", "Purchases"),
                    "u_conversion_rate": ("float", "Purchases / views"),
                    "u_avg_rating_given": ("float", "Mean explicit rating the user gave"),
                    "u_n_distinct_items": ("int", "Distinct products interacted with"),
                    "u_active_days": ("int", "Distinct active days"),
                    "u_days_since_last_event": ("float", "Recency in days at as_of"),
                    "u_tenure_days": ("float", "Days since signup"),
                    "u_avg_purchase_price": ("float", "Mean price of purchased items (INR)"),
                    "u_favourite_category": ("str", "Category with highest weighted engagement"),
                    "age": ("int", "Age (imputed when invalid)"),
                    "gender": ("str", "F/M/O/U"),
                    "city": ("str", "City"),
                    "is_prime_member": ("int", "Prime membership flag"),
                }, tags=("user", "behaviour")),
    FeatureView("item_popularity_quality", ["product_id"], "feat_item", 7,
                "Product popularity, conversion and quality signals up to as_of", {
                    "i_n_events": ("int", "Total events on the product"),
                    "i_n_views": ("int", "Views"),
                    "i_n_carts": ("int", "Add-to-cart events"),
                    "i_n_purchases": ("int", "Purchases"),
                    "i_conversion_rate": ("float", "Purchases / views"),
                    "i_avg_rating": ("float", "Mean explicit rating"),
                    "i_n_ratings": ("int", "Number of ratings"),
                    "i_n_unique_users": ("int", "Distinct users who interacted"),
                    "i_events_last_7d": ("int", "Trending signal: events in the 7 days before as_of"),
                    "i_days_since_last_event": ("float", "Days since last interaction"),
                    "i_age_days": ("float", "Days since product listing"),
                    "price": ("float", "Price in INR (cleaned)"),
                    "category": ("str", "Category"),
                    "brand": ("str", "Brand"),
                    "in_stock": ("int", "Stock flag"),
                }, tags=("item", "popularity")),
    FeatureView("user_item_interaction", ["user_id", "product_id"], "agg_user_item", 30,
                "Time-decayed implicit feedback strength per user-item pair", {
                    "ui_n_events": ("int", "Events on this pair"),
                    "ui_purchased": ("int", "1 if purchased"),
                    "ui_max_rating": ("float", "Highest explicit rating on the pair"),
                    "ui_strength": ("float", "SUM(weight * 0.5^(age/30d)) + (rating - 3)"),
                    "ui_last_event_ts": ("str", "Last event timestamp"),
                }, tags=("interaction", "label-source")),
    FeatureView("user_category_affinity", ["user_id", "category"], "feat_user_category", 30,
                "Share of the user's weighted engagement per category", {
                    "uc_affinity": ("float", "Normalised category affinity in [0,1]"),
                }, tags=("user", "content")),
]


class FeatureStore:
    def __init__(self, root=FEATURE_STORE):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.reg = sqlite3.connect(self.root / "registry.db", check_same_thread=False)
        self.reg.executescript("""
            CREATE TABLE IF NOT EXISTS feature_views (
                name TEXT PRIMARY KEY, entities TEXT, source_table TEXT, ttl_days INTEGER,
                description TEXT, owner TEXT, tags TEXT, version INTEGER, definition_hash TEXT,
                created_at TEXT, updated_at TEXT);
            CREATE TABLE IF NOT EXISTS features (
                feature_view TEXT, feature_name TEXT, dtype TEXT, description TEXT, version INTEGER,
                PRIMARY KEY (feature_view, feature_name));
            CREATE TABLE IF NOT EXISTS materializations (
                feature_view TEXT, version INTEGER, as_of TEXT, store TEXT, rows INTEGER,
                path TEXT, materialized_at TEXT);""")
        stamp_sqlite(self.reg)
        self.online = sqlite3.connect(self.root / "online.db", check_same_thread=False)
        stamp_sqlite(self.online)

    # ------------------------------------------------------------ registry
    def apply(self, views: list[FeatureView]) -> None:
        now = utc_now().isoformat()
        for v in views:
            h = v.definition_hash()
            row = self.reg.execute("SELECT version, definition_hash, created_at FROM feature_views WHERE name=?",
                                   (v.name,)).fetchone()
            version = 1 if row is None else (row[0] if row[1] == h else row[0] + 1)
            created = now if row is None else row[2]
            if row and row[1] != h:
                log.info("Definition of %s changed -> version %d", v.name, version)
            self.reg.execute("INSERT OR REPLACE INTO feature_views VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                             (v.name, json.dumps(v.entities), v.source_table, v.ttl_days, v.description,
                              v.owner, ",".join(v.tags), version, h, created, now))
            self.reg.execute("DELETE FROM features WHERE feature_view=?", (v.name,))
            self.reg.executemany("INSERT INTO features VALUES (?,?,?,?,?)",
                                 [(v.name, f, d[0], d[1], version) for f, d in v.features.items()])
        self.reg.commit()

    def view(self, name: str) -> dict:
        cur = self.reg.execute("SELECT * FROM feature_views WHERE name=?", (name,))
        cols = [c[0] for c in cur.description]
        return dict(zip(cols, cur.fetchone()))

    def list_features(self) -> pd.DataFrame:
        return pd.read_sql("""SELECT f.feature_view, f.feature_name, f.dtype, f.description, f.version,
                                     v.entities, v.ttl_days, v.source_table, v.owner
                              FROM features f JOIN feature_views v ON v.name = f.feature_view
                              ORDER BY f.feature_view, f.feature_name""", self.reg)

    # ------------------------------------------------------------ materialisation
    def materialize(self, name: str, as_of: str, online: bool = False) -> int:
        v = self.view(name)
        feats = [r[0] for r in self.reg.execute("SELECT feature_name FROM features WHERE feature_view=?", (name,))]
        keys = json.loads(v["entities"])
        with sqlite3.connect(WAREHOUSE_DB) as wh:
            df = pd.read_sql(f"SELECT as_of, {', '.join(keys + feats)} FROM {v['source_table']} WHERE as_of = ?",
                             wh, params=(as_of,))
        path = self.root / "offline" / name / f"as_of={as_of[:10]}.parquet"
        write_parquet(df, path)
        now = utc_now().isoformat()
        self.reg.execute("INSERT INTO materializations VALUES (?,?,?,?,?,?,?)",
                         (name, v["version"], as_of, "offline", len(df), str(path.relative_to(ROOT)), now))
        if online:
            df.assign(_feature_version=v["version"], _updated_at=now).to_sql(
                f"online_{name}", self.online, if_exists="replace", index=False)
            self.online.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{name} ON online_{name} ({', '.join(keys)})")
            self.online.commit()
            self.reg.execute("INSERT INTO materializations VALUES (?,?,?,?,?,?,?)",
                             (name, v["version"], as_of, "online", len(df), "online.db", now))
        self.reg.commit()
        return len(df)

    # ------------------------------------------------------------ retrieval
    def get_historical_features(self, name: str, entity_df: pd.DataFrame,
                                ts_col: str = "event_timestamp") -> pd.DataFrame:
        """Point-in-time join: each entity row gets the latest snapshot with as_of <= its timestamp."""
        v = self.view(name)
        keys = json.loads(v["entities"])
        snaps = sorted((self.root / "offline" / name).glob("as_of=*.parquet"))
        hist = pd.concat([read_parquet(p) for p in snaps], ignore_index=True)
        hist["as_of"] = pd.to_datetime(hist["as_of"])
        left = entity_df.copy()
        left["_ts"] = pd.to_datetime(left[ts_col])
        merged = pd.merge_asof(left.sort_values("_ts"), hist.sort_values("as_of"), left_on="_ts",
                               right_on="as_of", by=keys, direction="backward",
                               tolerance=pd.Timedelta(days=int(v["ttl_days"]) + 30))
        return merged.drop(columns="_ts")

    def get_offline_snapshot(self, name: str, as_of: str) -> pd.DataFrame:
        return read_parquet(self.root / "offline" / name / f"as_of={as_of[:10]}.parquet")

    def get_online_features(self, name: str, keys: dict) -> pd.DataFrame:
        where = " AND ".join(f"{k} = ?" for k in keys)
        return pd.read_sql(f"SELECT * FROM online_{name} WHERE {where}", self.online,
                           params=tuple(keys.values()))


def write_catalog(fs: "FeatureStore") -> None:
    """Human-readable feature catalog (markdown) generated from the registry."""
    reg = fs.list_features()
    lines = [f"<!-- Roll No: {ROLL_NO} -->", "# Feature catalog", "",
             "Generated from the feature registry (`data/features/registry.db`). All features are computed "
             "point-in-time in the warehouse (`src/transformation/sql/features.sql`) and served from the "
             "offline store for training and the online store for prediction.", ""]
    for v in FEATURE_VIEWS:
        meta = fs.view(v.name)
        lines += [f"## {v.name} (v{meta['version']})", "",
                  f"{v.description}. Entity key: `{', '.join(v.entities)}`. Source table: `{v.source_table}`. "
                  f"TTL: {v.ttl_days} days. Owner: {v.owner}. Definition hash: `{meta['definition_hash']}`.", "",
                  "| Feature | Type | Description | Used by |", "|---|---|---|---|"]
        for f, (dt, desc) in v.features.items():
            used = USED_BY.get(f, "serving enrichment / analysis")
            lines.append(f"| `{f}` | {dt} | {desc} | {used} |")
        lines.append("")
    (REPORTS / "feature_catalog.md").write_text("\n".join(lines), encoding="utf-8")


USED_BY = {
    "ui_strength": "all CF models (implicit matrix)", "i_n_views": "popularity baseline",
    "i_n_carts": "popularity baseline", "i_n_purchases": "popularity baseline",
    "i_events_last_7d": "popularity baseline (trending)", "category": "hybrid, diversity metric",
    "uc_affinity": "hybrid (content signal)", "price": "serving enrichment",
    "i_avg_rating": "serving enrichment", "in_stock": "serving enrichment",
    "u_favourite_category": "serving context",
}


def main() -> dict:
    snaps = json.loads((REPORTS / "transform_summary.json").read_text())["snapshots"]
    fs = FeatureStore()
    fs.apply(FEATURE_VIEWS)
    counts = {}
    for v in FEATURE_VIEWS:
        counts[v.name] = {"valid": fs.materialize(v.name, snaps["valid"]),
                          "train": fs.materialize(v.name, snaps["train"]),
                          "serve": fs.materialize(v.name, snaps["serve"], online=True)}
    reg = fs.list_features()
    write_csv(reg, REPORTS / "feature_registry.csv")
    write_catalog(fs)

    # demo: point-in-time retrieval + online lookup (evidence for the report)
    demo_entities = pd.DataFrame({"user_id": [7, 7], "event_timestamp": ["2026-10-20 10:00:00",
                                                                         "2026-11-01 09:00:00"]})
    pit = fs.get_historical_features("user_profile_behaviour", demo_entities)
    online = fs.get_online_features("user_profile_behaviour", {"user_id": 7})
    demo = {"point_in_time": pit[["user_id", "event_timestamp", "as_of", "u_n_events",
                                  "u_n_purchases"]].astype(str).to_dict("records"),
            "online": online[["user_id", "as_of", "u_n_events", "u_n_purchases",
                              "_feature_version"]].astype(str).to_dict("records")}
    out = {"views": counts, "n_features": len(reg),
           "versions": {v.name: fs.view(v.name)["version"] for v in FEATURE_VIEWS}, "demo": demo}
    write_json(out, REPORTS / "feature_store_summary.json")
    log.info("Feature store: %d features in %d views; %s", len(reg), len(FEATURE_VIEWS), counts)
    return out


if __name__ == "__main__":
    main()
