# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Shared paths, logging and watermark helpers used by every pipeline stage."""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROLL_NO = "2025AG05312"
WATERMARK = f"Roll No: {ROLL_NO}"
COURSE = "AIML ZG529 Data Management for Machine Learning"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = DATA / "sources"          # simulated upstream source systems (not part of the lake)
RAW = DATA / "raw"                  # landing zone / bronze, immutable, partitioned
VALIDATED = DATA / "validated"      # silver-1: rows that passed validation
QUARANTINE = DATA / "errors"        # error dataset: rows that failed critical checks
PROCESSED = DATA / "processed"      # silver-2: cleaned and conformed
WAREHOUSE = DATA / "warehouse"      # gold: SQLite star schema
FEATURE_STORE = DATA / "features"   # feature store (registry, offline, online)
REPORTS = ROOT / "reports"
SQL_DIR = ROOT / "src" / "transformation" / "sql"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"

MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS = (ROOT / "mlartifacts").as_posix()
REGISTERED_MODEL_NAME = f"recsys_{ROLL_NO}"
EXPERIMENT_NAME = f"recsys_{ROLL_NO}"

# Business constants shared across stages
EVENT_WEIGHTS = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}
TEST_DAYS = 14          # time-based holdout window (last N days)
TOP_K = 10


def get_logger(name: str) -> logging.Logger:
    """Logger that writes to console and to logs/pipeline.log."""
    LOGS.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(f"%(asctime)s | {ROLL_NO} | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOGS / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- watermarked I/O
def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """CSV with a '#' comment watermark line. Read back with read_csv()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {WATERMARK} | {COURSE}\n")
        df.to_csv(f, index=False)
    return path


def read_csv(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", **kw)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Parquet with the roll number stored in the file's key-value metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"watermark"] = WATERMARK.encode()
    pq.write_table(table.replace_schema_metadata(meta), path)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, dict):
        obj = {"_watermark": WATERMARK, **obj}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def watermark_figure(fig, text: str = WATERMARK) -> None:
    """Stamp a matplotlib figure with a diagonal and a footer watermark."""
    fig.text(0.5, 0.5, text, fontsize=34, color="grey", alpha=0.13,
             ha="center", va="center", rotation=30, zorder=0)
    fig.text(0.99, 0.01, text, fontsize=8, color="#555555", ha="right", va="bottom")


def stamp_sqlite(conn) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS _watermark (roll_no TEXT, course TEXT, created_at TEXT)")
    if conn.execute("SELECT COUNT(*) FROM _watermark").fetchone()[0] == 0:   # write once: no-op on reads
        conn.execute("INSERT INTO _watermark VALUES (?,?,?)", (ROLL_NO, COURSE, utc_now().isoformat()))
        conn.commit()


# ---------------------------------------------------------------- chart style (validated palette)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def new_fig(w: float = 7.5, h: float = 4.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                         "axes.titleweight": "bold", "axes.titlesize": 11,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    fig, ax = plt.subplots(figsize=(w, h), dpi=130)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def save_fig(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    watermark_figure(fig)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, metadata={"Description": WATERMARK, "Author": ROLL_NO})
    plt.close(fig)
    return path
