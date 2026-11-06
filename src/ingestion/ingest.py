# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 1: DATA INGESTION + RAW DATA STORAGE.

Sources
  A. Clickstream export (CSV, batch file)   -> incremental, CDC-style high-water mark on event_id
  B. CRM user export (CSV, batch file)      -> full snapshot each run (small dimension)
  C. Product catalog (REST API, JSON)       -> paginated pull with auth + retry/backoff

Raw zone layout (immutable, append-only, source format preserved = schema-on-read):
  data/raw/<source>/<entity>/ingest_date=YYYY-MM-DD/batch_<batch_id>.<ext>
  data/raw/<source>/<entity>/ingest_date=YYYY-MM-DD/batch_<batch_id>.manifest.json

Each batch gets a manifest (row count, md5, schema, mode, high-water mark) and a
row in reports/ingestion_audit.csv. Ingestion never mutates or cleans data:
that is the job of validation/preparation, so raw remains replayable.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from src.common import (RAW, REPORTS, ROLL_NO, SOURCES, WATERMARK, file_md5, get_logger,
                    read_csv, utc_now, write_csv, write_json)
from src.ingestion.mock_product_api import API_KEY, HOST, PORT, start_server

log = get_logger("ingestion.ingest")
STATE_FILE = RAW / "_state" / "ingestion_state.json"


# ------------------------------------------------------------------ helpers
def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def _save_state(state: dict) -> None:
    write_json(state, STATE_FILE)


def _partition(source: str, entity: str, batch_id: str):
    day = batch_id[:8]
    part = RAW / source / entity / f"ingest_date={day[:4]}-{day[4:6]}-{day[6:]}"
    part.mkdir(parents=True, exist_ok=True)
    return part


def _manifest(path, source, entity, batch_id, rows, columns, mode, extra=None) -> dict:
    m = {"source": source, "entity": entity, "batch_id": batch_id, "mode": mode,
         "file": path.name, "row_count": int(rows), "columns": list(columns),
         "md5": file_md5(path), "bytes": path.stat().st_size,
         "ingested_at": utc_now().isoformat(), "ingested_by": ROLL_NO, **(extra or {})}
    write_json(m, path.with_suffix(".manifest.json"))
    return m


def _audit(m: dict, status: str, duration: float) -> None:
    row = pd.DataFrame([{"batch_id": m.get("batch_id"), "source": m.get("source"),
                         "entity": m.get("entity"), "mode": m.get("mode"),
                         "rows": m.get("row_count", 0), "status": status,
                         "duration_s": round(duration, 3), "file": m.get("file"),
                         "ingested_at": utc_now().isoformat()}])
    path = REPORTS / "ingestion_audit.csv"
    prev = read_csv(path) if path.exists() else pd.DataFrame()
    write_csv(pd.concat([prev, row], ignore_index=True), path)


# ------------------------------------------------------------------ source A
def ingest_clickstream(batch_id: str, mode: str, state: dict) -> dict:
    t0 = time.time()
    src = SOURCES / "clickstream" / "interactions_export.csv"
    df = read_csv(src, dtype=str)                       # raw: keep everything as text
    hwm = state.get("clickstream_event_id_hwm")
    if mode == "incremental" and hwm:
        df = df[df["event_id"] > hwm]                   # CDC-style: only new log entries
    if df.empty:
        log.info("clickstream: no new events since high-water mark %s", hwm)
        m = {"batch_id": batch_id, "source": "clickstream", "entity": "interactions",
             "mode": mode, "row_count": 0, "file": None}
        _audit(m, "NO_NEW_DATA", time.time() - t0)
        return m
    out = _partition("clickstream", "interactions", batch_id) / f"batch_{batch_id}.csv"
    write_csv(df, out)
    new_hwm = df["event_id"].max()
    m = _manifest(out, "clickstream", "interactions", batch_id, len(df), df.columns, mode,
                  {"hwm_before": hwm, "hwm_after": new_hwm})
    state["clickstream_event_id_hwm"] = new_hwm
    _audit(m, "SUCCESS", time.time() - t0)
    log.info("clickstream: ingested %d rows (%s), hwm %s -> %s", len(df), mode, hwm, new_hwm)
    return m


# ------------------------------------------------------------------ source B
def ingest_users(batch_id: str) -> dict:
    t0 = time.time()
    df = read_csv(SOURCES / "crm" / "users_export.csv", dtype=str)
    out = _partition("crm", "users", batch_id) / f"batch_{batch_id}.csv"
    write_csv(df, out)
    m = _manifest(out, "crm", "users", batch_id, len(df), df.columns, "full_snapshot")
    _audit(m, "SUCCESS", time.time() - t0)
    log.info("crm users: ingested %d rows (full snapshot)", len(df))
    return m


# ------------------------------------------------------------------ source C
def _get_json(url: str, retries: int = 4, backoff: float = 0.5) -> dict:
    req = urllib.request.Request(url, headers={"X-API-Key": API_KEY,
                                               "User-Agent": f"ingestor/{ROLL_NO}"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = backoff * 2 ** (attempt - 1)
                log.warning("API %s returned %s, retry %d/%d in %.1fs", url, e.code,
                            attempt, retries - 1, wait)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(backoff * 2 ** (attempt - 1))
                continue
            raise
    raise RuntimeError("unreachable")


def ingest_products(batch_id: str, page_size: int = 100) -> dict:
    t0 = time.time()
    server = start_server(PORT)                 # local stand-in for the catalog service
    try:
        base = f"http://{HOST}:{PORT}/api/v1/products"
        page, records, pages = 1, [], 0
        while page:
            body = _get_json(f"{base}?page={page}&page_size={page_size}")
            records.extend(body["data"])
            pages += 1
            page = body["next_page"]
    finally:
        server.shutdown()
    out = _partition("catalog_api", "products", batch_id) / f"batch_{batch_id}.json"
    with open(out, "w", encoding="utf-8") as f:     # preserve source JSON as-is
        json.dump({"_watermark": WATERMARK, "endpoint": "/api/v1/products",
                   "pages": pages, "records": records}, f, indent=1)
    cols = sorted({k for r in records for k in r})
    m = _manifest(out, "catalog_api", "products", batch_id, len(records), cols, "full_pull",
                  {"pages": pages})
    _audit(m, "SUCCESS", time.time() - t0)
    log.info("catalog api: ingested %d records over %d pages", len(records), pages)
    return m


def main(mode: str = "incremental") -> dict:
    batch_id = utc_now().strftime("%Y%m%d%H%M%S")
    state = _load_state()
    results = {"batch_id": batch_id}
    try:
        results["clickstream"] = ingest_clickstream(batch_id, mode, state)
        results["users"] = ingest_users(batch_id)
        results["products"] = ingest_products(batch_id)
    except Exception as exc:
        _audit({"batch_id": batch_id, "source": "pipeline", "mode": mode}, f"FAILED: {exc}", 0)
        log.exception("Ingestion failed")
        raise
    state["last_batch_id"] = batch_id
    _save_state(state)
    write_json(results, REPORTS / "ingestion_last_run.json")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    main(ap.parse_args().mode)
