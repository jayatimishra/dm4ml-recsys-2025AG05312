# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Run every pipeline stage from an empty data folder, capture its real terminal
output, and render each capture as a terminal-style PNG for the execution PDF.

python tools/capture_execution.py          -> docs/execution/*.png + captures.json
Needs Playwright + Chromium (pip install playwright && playwright install chromium).
"""
import html
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "execution"
ROLL = "2025AG05312"
PROMPT = f"[{ROLL}] dm4ml-recsys $"
MAX_LINES = 40
ENV = {**os.environ, "MLFLOW_DISABLE_TELEMETRY": "true", "DO_NOT_TRACK": "1", "MLFLOW_DISABLE_AGENT_HINT": "1",
       "PYTHONWARNINGS": "ignore::UserWarning,ignore::FutureWarning"}

STEPS = [  # (id, title, shell command)
    ("00_setup", "Environment", "python --version && pip show pandas mlflow prefect dvc pandera 2>/dev/null | grep -E '^(Name|Version)' | paste - - && git log --oneline --decorate -3"),
    ("01_simulate_sources", "Source systems (synthetic data)", "python -m src.ingestion.generate_sources --end-date 2026-11-01"),
    ("02_ingest", "Stage 1: Data ingestion", "python -m src.ingestion.ingest --mode incremental && find data/raw -type f | sort"),
    ("03_validate", "Stage 2: Data validation", "python -m src.validation.validate && echo && head -5 data/errors/interactions_rejected.csv | cut -c1-150"),
    ("04_prepare", "Stage 3: Data preparation", "python -m src.preparation.prepare && ls reports/eda"),
    ("05_transform", "Stage 4: Data transformation", "python -m src.transformation.transform && python -c \"import sqlite3; c=sqlite3.connect('data/warehouse/recsys_warehouse.db'); [print(f'{t:22s}', c.execute(f'select count(*) from {t}').fetchone()[0]) for t in ['dim_user','dim_product','dim_date','fact_interactions','feat_user','feat_item','agg_user_item']]\" && ls data/processed/splits data/processed/derived"),
    ("06_feature_store", "Stage 5: Feature management", "python -m src.features.feature_store && python -c \"import sys; sys.path.insert(0,'.'); from src.features.feature_store import FeatureStore; fs=FeatureStore(); print(fs.list_features().groupby('feature_view').feature_name.count()); print(fs.get_online_features('user_profile_behaviour', {'user_id': 42})[['user_id','u_n_events','u_n_purchases','u_avg_rating_given','u_favourite_category','u_days_since_last_event']].to_string())\""),
    ("07_pre_training_checks", "Stage 6: Pre-training validation", "python -m src.validation.pre_training_checks"),
    ("08_train", "Stage 7: Model training (MLflow)", "python -m src.training.train"),
    ("09_evaluate", "Stage 8: Model evaluation", "python -m src.evaluation.evaluate && tail -n +3 reports/model/results_table.md"),
    ("10_serve", "Stage 9: Prediction service", "python -m src.serving.serve --batch && { python -m src.serving.serve --serve > /tmp/serve_capture.log 2>&1 & SRV=$!; } ; for i in $(seq 1 40); do sleep 1; curl -s 'http://127.0.0.1:8600/health' >/dev/null && break; done; curl -s 'http://127.0.0.1:8600/recommend?user_id=42&k=5' | python -m json.tool | head -40; kill $SRV"),
    ("11_orchestration", "Orchestration: full DAG run", "python dags/run_pipeline.py 2>&1 | grep orchestrator"),
    ("12_dvc", "Versioning: DVC pipeline", "dvc dag && sed -n '/dvc params diff/,/dvc metrics diff/p' reports/versioning/dvc_evidence.txt | head -30"),
]

CSS = """body{margin:0;background:#f4f4f2;font-family:Menlo,'DejaVu Sans Mono',monospace}
.win{margin:14px;border-radius:9px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.25);background:#1d1f21;width:1180px}
.bar{background:#2d2f31;color:#c5c8c6;font:12px sans-serif;padding:7px 12px;display:flex;gap:8px;align-items:center}
.dot{width:11px;height:11px;border-radius:50%;display:inline-block}
pre{margin:0;padding:12px 16px;color:#d6d6d6;font-size:12.5px;line-height:1.38;white-space:pre-wrap;word-break:break-all}
.p{color:#8fd16b}.c{color:#e6e6e6;font-weight:bold}.ok{color:#8fd16b}.warn{color:#f0c674}.err{color:#ff7b72}
.wm{position:fixed;top:45%;left:22%;font:bold 60px sans-serif;color:rgba(255,255,255,.07);transform:rotate(-18deg)}
.foot{color:#8a8d8f;font-size:11px;padding:0 16px 10px}"""


def trim(text: str) -> str:
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in text.rstrip().splitlines()]
    lines = [re.sub(r" \| " + ROLL + r" \| ", " | ", l) for l in lines]
    if len(lines) > MAX_LINES:
        lines = lines[:22] + [f"... ({len(lines) - 36} lines omitted, full output in docs/execution/captures.json) ..."] + lines[-14:]
    return "\n".join(lines)


def colour(line: str) -> str:
    e = html.escape(line)
    if re.search(r"FAIL|ERROR|Traceback", line):
        return f"<span class='err'>{e}</span>"
    if re.search(r"WARNING|retry", line):
        return f"<span class='warn'>{e}</span>"
    if re.search(r"SUCCESS|PASS|PASSED|Completed", line):
        return f"<span class='ok'>{e}</span>"
    return e


def render(page, sid, title, cmd, output, stamp):
    body = "\n".join(colour(l) for l in output.splitlines())
    doc = f"""<html><head><style>{CSS}</style></head><body><div class='wm'>Roll No: {ROLL}</div>
<div class='win'><div class='bar'><span class='dot' style='background:#ff5f56'></span><span class='dot' style='background:#ffbd2e'></span>
<span class='dot' style='background:#27c93f'></span>&nbsp; {html.escape(title)}  |  Roll No: {ROLL}  |  {stamp}</div>
<pre><span class='p'>{html.escape(PROMPT)}</span> <span class='c'>{html.escape(cmd if len(cmd) < 180 else cmd[:177] + '...')}</span>
{body}</pre><div class='foot'>Roll No: {ROLL} | captured {stamp}</div></div></body></html>"""
    page.set_content(doc)
    page.locator(".win").screenshot(path=str(OUT / f"{sid}.png"))


def main():
    from playwright.sync_api import sync_playwright

    for d in ["data", "models", "mlartifacts", "logs"]:
        shutil.rmtree(ROOT / d, ignore_errors=True)
    (ROOT / "mlflow.db").unlink(missing_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    caps = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1220, "height": 800})
        for sid, title, cmd in STEPS:
            t0 = time.time()
            r = subprocess.run(cmd, shell=True, cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=900)
            out = (r.stdout + r.stderr)
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            caps.append({"id": sid, "title": title, "command": cmd, "exit_code": r.returncode,
                         "seconds": round(time.time() - t0, 1), "captured_at": stamp, "output": out})
            render(page, sid, title, cmd, trim(out) + f"\n[exit code {r.returncode}]", stamp)
            print(f"{sid:24s} exit={r.returncode} {time.time() - t0:.1f}s")
            if r.returncode != 0:
                print(out[-3000:])
                sys.exit(1)
        browser.close()
    (OUT / "captures.json").write_text(json.dumps({"_watermark": f"Roll No: {ROLL}", "steps": caps}, indent=1))


if __name__ == "__main__":
    main()
