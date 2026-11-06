# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 2: DATA VALIDATION (data contracts as code + quality gate).

Every check is declared once in CONTRACTS with:
  dimension : completeness | validity | uniqueness | consistency | integrity | timeliness
  severity  : critical -> failing rows are QUARANTINED (with reason) and excluded
              warning  -> rows kept, defect flagged for repair in Stage 3
Dataset-level checks (schema, volume, freshness, null-rate) feed a QUALITY GATE:
if any gate check fails the pipeline stops before bad data reaches training.

Outputs
  data/validated/<entity>.parquet          rows that passed all critical checks
  data/errors/<entity>_rejected.csv    rejected rows + failed check ids
  reports/validation/validation_report.{json,csv,html}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.common import (QUARANTINE, RAW, REPORTS, ROLL_NO, VALIDATED, WATERMARK, get_logger,
                    read_csv, utc_now, write_csv, write_json, write_parquet)

log = get_logger("validation")
OUT = REPORTS / "validation"

EVENT_TYPES = {"view", "add_to_cart", "purchase"}
DEVICES = {"mobile", "desktop", "tablet"}
GENDERS = {"F", "M", "O"}
CATEGORIES = {"Electronics", "Books", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Toys",
              "Grocery", "Automotive", "Music", "Office", "Garden"}


def norm_event_type(s: pd.Series) -> pd.Series:
    return s.str.strip().str.lower().str.replace("-", "_", regex=False)


def to_price(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
                         .replace({"": np.nan, "None": np.nan, "nan": np.nan}), errors="coerce")


@dataclass
class Check:
    id: str
    column: str
    dimension: str
    severity: str
    description: str
    fn: Callable[[pd.DataFrame], pd.Series]  # True = row passes


# ------------------------------------------------------------------ contracts (schema as code)
def contracts(users_ids: set, product_ids: set, run_ts: pd.Timestamp) -> dict:
    ts = lambda d: pd.to_datetime(d["event_timestamp"], errors="coerce", format="%Y-%m-%d %H:%M:%S")
    rating = lambda d: pd.to_numeric(d["rating"], errors="coerce")
    age = lambda d: pd.to_numeric(d["age"], errors="coerce")
    return {
        "interactions": {
            "schema": ["event_id", "user_id", "product_id", "event_type", "rating",
                       "event_timestamp", "session_id", "device"],
            "min_rows": 1000, "max_null_rate": {"user_id": 0.01, "product_id": 0.01},
            "freshness_col": "event_timestamp", "max_staleness_days": 7,
            "checks": [
                Check("INT_001", "event_id", "completeness", "critical", "event_id is not null",
                      lambda d: d.event_id.notna()),
                Check("INT_002", "user_id", "completeness", "critical", "user_id is not null",
                      lambda d: d.user_id.notna()),
                Check("INT_003", "product_id", "validity", "critical", "product_id is an integer",
                      lambda d: pd.to_numeric(d.product_id, errors="coerce").notna()),
                Check("INT_004", "event_timestamp", "validity", "critical",
                      "event_timestamp parses as YYYY-MM-DD HH:MM:SS", lambda d: ts(d).notna()),
                Check("INT_005", "event_timestamp", "timeliness", "critical",
                      "event_timestamp is not in the future", lambda d: ~(ts(d) > run_ts)),
                Check("INT_006", "event_type", "validity", "critical",
                      "event_type maps to {view, add_to_cart, purchase} after normalisation",
                      lambda d: norm_event_type(d.event_type).isin(EVENT_TYPES)),
                Check("INT_007", "event_type", "consistency", "warning",
                      "event_type already in canonical lower_snake form",
                      lambda d: d.event_type.isin(EVENT_TYPES)),
                Check("INT_008", "rating", "validity", "warning", "rating is null or in [1,5]",
                      lambda d: rating(d).isna() | rating(d).between(1, 5)),
                Check("INT_009", "rating", "consistency", "warning",
                      "rating only present on purchase events",
                      lambda d: rating(d).isna() | (norm_event_type(d.event_type) == "purchase")),
                Check("INT_010", "user_id", "integrity", "critical",
                      "user_id exists in CRM users (FK)",
                      lambda d: pd.to_numeric(d.user_id, errors="coerce").isin(users_ids)),
                Check("INT_011", "product_id", "integrity", "critical",
                      "product_id exists in product catalog (FK)",
                      lambda d: pd.to_numeric(d.product_id, errors="coerce").isin(product_ids)),
                Check("INT_012", "device", "validity", "warning", "device in {mobile, desktop, tablet}",
                      lambda d: d.device.isin(DEVICES)),
            ],
            "key": "event_id",
        },
        "users": {
            "schema": ["user_id", "signup_date", "age", "gender", "city", "is_prime_member"],
            "min_rows": 100, "max_null_rate": {"user_id": 0.0},
            "checks": [
                Check("USR_001", "user_id", "completeness", "critical", "user_id is not null",
                      lambda d: d.user_id.notna()),
                Check("USR_002", "signup_date", "validity", "critical", "signup_date is a valid date",
                      lambda d: pd.to_datetime(d.signup_date, errors="coerce").notna()),
                Check("USR_003", "age", "completeness", "warning", "age is not null",
                      lambda d: age(d).notna()),
                Check("USR_004", "age", "validity", "warning", "age in [13, 100]",
                      lambda d: age(d).isna() | age(d).between(13, 100)),
                Check("USR_005", "gender", "validity", "warning", "gender in {F, M, O}",
                      lambda d: d.gender.isin(GENDERS)),
                Check("USR_006", "city", "completeness", "warning", "city is not null",
                      lambda d: d.city.notna()),
            ],
            "key": "user_id",
        },
        "products": {
            "schema": ["product_id", "title", "category", "brand", "price", "currency",
                       "created_at", "in_stock"],
            "min_rows": 100, "max_null_rate": {"product_id": 0.0},
            "checks": [
                Check("PRD_001", "product_id", "completeness", "critical", "product_id is not null",
                      lambda d: d.product_id.notna()),
                Check("PRD_002", "title", "completeness", "critical", "title is not null",
                      lambda d: d.title.notna()),
                Check("PRD_003", "price", "validity", "warning", "price is numeric (no currency text)",
                      lambda d: pd.to_numeric(d.price, errors="coerce").notna() | d.price.isna()),
                Check("PRD_004", "price", "completeness", "warning", "price is not null",
                      lambda d: d.price.notna()),
                Check("PRD_005", "price", "validity", "warning", "price > 0",
                      lambda d: to_price(d.price).isna() | (to_price(d.price) > 0)),
                Check("PRD_006", "category", "completeness", "warning", "category is not null",
                      lambda d: d.category.notna()),
                Check("PRD_007", "category", "consistency", "warning",
                      "category matches the controlled vocabulary exactly",
                      lambda d: d.category.isna() | d.category.isin(CATEGORIES)),
                Check("PRD_008", "currency", "validity", "warning", "currency == INR",
                      lambda d: d.currency == "INR"),
            ],
            "key": "product_id",
        },
    }


# ------------------------------------------------------------------ load raw zone
def _latest(source: str, entity: str, ext: str):
    files = sorted(p for p in (RAW / source / entity).glob(f"ingest_date=*/batch_*.{ext}")
                   if not p.name.endswith(".manifest.json"))
    if not files:
        raise FileNotFoundError(f"No raw batches for {source}/{entity}")
    return files


def load_raw() -> dict[str, pd.DataFrame]:
    # interactions: union of every incremental batch (append-only log)
    inter = pd.concat([read_csv(f, dtype=str) for f in _latest("clickstream", "interactions", "csv")],
                      ignore_index=True)
    users = read_csv(_latest("crm", "users", "csv")[-1], dtype=str)       # latest snapshot
    with open(_latest("catalog_api", "products", "json")[-1], encoding="utf-8") as f:
        products = pd.DataFrame(json.load(f)["records"])
    products["product_id"] = products["product_id"].astype("Int64")
    return {"interactions": inter, "users": users, "products": products}


# ------------------------------------------------------------------ engine
def run_entity(name: str, df: pd.DataFrame, spec: dict, run_ts) -> tuple[pd.DataFrame, list, list]:
    n0 = len(df)
    check_rows, gate = [], []

    # dataset-level: schema
    missing = [c for c in spec["schema"] if c not in df.columns]
    extra = [c for c in df.columns if c not in spec["schema"]]
    gate.append({"entity": name, "gate_check": "schema_columns_present", "passed": not missing,
                 "detail": f"missing={missing} unexpected={extra}"})
    # dataset-level: volume
    gate.append({"entity": name, "gate_check": "min_row_count", "passed": n0 >= spec["min_rows"],
                 "detail": f"rows={n0} min={spec['min_rows']}"})
    # dataset-level: null-rate thresholds on keys
    for col, thr in spec["max_null_rate"].items():
        rate = float(df[col].isna().mean())
        gate.append({"entity": name, "gate_check": f"null_rate[{col}]<={thr}",
                     "passed": rate <= thr, "detail": f"null_rate={rate:.4f}"})
    # dataset-level: freshness
    if spec.get("freshness_col"):
        mx = pd.to_datetime(df[spec["freshness_col"]], errors="coerce").max()
        stale = (run_ts - mx).days
        gate.append({"entity": name, "gate_check": f"freshness<={spec['max_staleness_days']}d",
                     "passed": stale <= spec["max_staleness_days"],
                     "detail": f"latest={mx} staleness_days={stale}", "severity": "warning"})

    # uniqueness: exact duplicate rows and duplicate keys
    key = spec["key"]
    dup_mask = df.duplicated(subset=[key], keep="first") & df[key].notna()
    check_rows.append({"entity": name, "check_id": {"users": "USR", "products": "PRD", "interactions": "INT"}[name] + "_UNQ", "column": key,
                       "dimension": "uniqueness", "severity": "critical",
                       "description": f"{key} is unique (duplicates dropped, first kept)",
                       "rows_checked": n0, "rows_failed": int(dup_mask.sum()),
                       "pass_rate": round(1 - dup_mask.mean(), 5)})
    df = df[~dup_mask].copy()

    fails = pd.Series([[] for _ in range(len(df))], index=df.index)
    critical_fail = pd.Series(False, index=df.index)
    warn_flags = pd.DataFrame(index=df.index)
    for c in spec["checks"]:
        ok = c.fn(df).fillna(False).astype(bool)
        bad = ~ok
        check_rows.append({"entity": name, "check_id": c.id, "column": c.column,
                           "dimension": c.dimension, "severity": c.severity,
                           "description": c.description, "rows_checked": len(df),
                           "rows_failed": int(bad.sum()), "pass_rate": round(float(ok.mean()), 5)})
        for i in df.index[bad]:
            fails.at[i].append(c.id)
        if c.severity == "critical":
            critical_fail |= bad
        else:
            warn_flags[f"dq_{c.id}"] = bad

    rejected = df[critical_fail].copy()
    rejected["failed_checks"] = fails[critical_fail].map(lambda x: ";".join(x))
    passed = df[~critical_fail].copy()
    passed["dq_warning_count"] = warn_flags.loc[passed.index].sum(axis=1).astype(int) if len(warn_flags.columns) else 0

    reject_rate = len(rejected) / max(len(df), 1)
    gate.append({"entity": name, "gate_check": "critical_reject_rate<=5%",
                 "passed": reject_rate <= 0.05, "detail": f"reject_rate={reject_rate:.4f}"})
    log.info("%s: in=%d dup_dropped=%d rejected=%d passed=%d", name, n0, int(dup_mask.sum()),
             len(rejected), len(passed))
    return passed, rejected, check_rows, gate, {"rows_in": n0, "duplicates_dropped": int(dup_mask.sum()),
                                                "rows_rejected": len(rejected), "rows_passed": len(passed)}


def _html(checks: pd.DataFrame, gate: pd.DataFrame, summary: dict) -> str:
    def color(v):
        return "#e8f5e9" if v else "#ffebee"
    rows = "".join(
        f"<tr style='background:{color(r.rows_failed == 0)}'><td>{r.entity}</td><td>{r.check_id}</td>"
        f"<td>{r.column}</td><td>{r.dimension}</td><td>{r.severity}</td><td>{r.description}</td>"
        f"<td>{r.rows_checked}</td><td>{r.rows_failed}</td><td>{r.pass_rate:.4f}</td></tr>"
        for r in checks.itertuples())
    grows = "".join(
        f"<tr style='background:{color(r.passed)}'><td>{r.entity}</td><td>{r.gate_check}</td>"
        f"<td>{'PASS' if r.passed else 'FAIL'}</td><td>{r.detail}</td></tr>" for r in gate.itertuples())
    srows = "".join(f"<tr><td>{k}</td>" + "".join(f"<td>{v}</td>" for v in d.values()) + "</tr>"
                    for k, d in summary["entities"].items())
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Data Validation Report {ROLL_NO}</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;margin-bottom:24px;font-size:13px}}
td,th{{border:1px solid #bbb;padding:4px 8px}}th{{background:#37474f;color:#fff}}
.wm{{position:fixed;top:40%;left:15%;font-size:72px;color:rgba(0,0,0,0.07);transform:rotate(-25deg);pointer-events:none}}</style></head>
<body><div class='wm'>{WATERMARK}</div><h1>Data Validation Report</h1>
<p><b>{WATERMARK}</b> | run at {summary['run_at']} | overall gate: <b>{'PASSED' if summary['gate_passed'] else 'FAILED'}</b></p>
<h2>Entity summary</h2><table><tr><th>entity</th><th>rows_in</th><th>duplicates_dropped</th><th>rows_rejected</th><th>rows_passed</th></tr>{srows}</table>
<h2>Quality gate (dataset-level)</h2><table><tr><th>entity</th><th>check</th><th>result</th><th>detail</th></tr>{grows}</table>
<h2>Row-level checks</h2><table><tr><th>entity</th><th>id</th><th>column</th><th>dimension</th><th>severity</th><th>description</th><th>checked</th><th>failed</th><th>pass rate</th></tr>{rows}</table>
<p style='font-size:11px'>{WATERMARK}</p></body></html>"""


def main() -> dict:
    run_ts = pd.Timestamp(utc_now().replace(tzinfo=None))
    raw = load_raw()
    # FK domains come from the reference entities' own validated keys
    u_ids = set(pd.to_numeric(raw["users"].user_id, errors="coerce").dropna().astype(int))
    p_ids = set(raw["products"].product_id.dropna().astype(int))
    specs = contracts(u_ids, p_ids, run_ts)

    all_checks, all_gate, summary = [], [], {"run_at": run_ts.isoformat(), "entities": {}}
    for name in ["users", "products", "interactions"]:
        passed, rejected, checks, gate, stats = run_entity(name, raw[name], specs[name], run_ts)
        write_parquet(passed.astype({c: "string" for c in passed.columns
                                     if passed[c].dtype == object}), VALIDATED / f"{name}.parquet")
        write_csv(rejected, QUARANTINE / f"{name}_rejected.csv")
        all_checks += checks
        all_gate += gate
        summary["entities"][name] = stats

    checks_df = pd.DataFrame(all_checks)
    gate_df = pd.DataFrame(all_gate)
    gate_df["severity"] = gate_df.get("severity", pd.Series(dtype=str)).fillna("critical")
    hard = gate_df[gate_df.severity == "critical"]
    summary["gate_passed"] = bool(hard.passed.all())
    summary["data_quality_score"] = round(float(
        (checks_df.rows_checked - checks_df.rows_failed).sum() / checks_df.rows_checked.sum()), 5)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(checks_df, OUT / "validation_checks.csv")
    write_csv(gate_df, OUT / "quality_gate.csv")
    write_json({**summary, "checks": all_checks, "gate": all_gate}, OUT / "validation_report.json")
    (OUT / "validation_report.html").write_text(_html(checks_df, gate_df, summary), encoding="utf-8")
    log.info("Validation done: DQ score=%.4f gate=%s", summary["data_quality_score"],
             "PASSED" if summary["gate_passed"] else "FAILED")
    if not summary["gate_passed"]:
        raise RuntimeError(f"Quality gate FAILED: {hard[~hard.passed].to_dict('records')}")
    return summary


if __name__ == "__main__":
    main()
