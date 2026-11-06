# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Dependency-free orchestrator (fallback when Prefect/Airflow are not installed).

python dags/run_pipeline.py      (parameters come from configs/params.yaml)
Runs the DAG in topological order with per-task retries, timing and a run report
written to reports/orchestration/run_<timestamp>.json. Stops on first failure.
"""
from __future__ import annotations

import time
import traceback

from pipeline_dag import DAG, run_task, topological_order

from src.common import REPORTS, ROLL_NO, get_logger, utc_now, write_json

log = get_logger("orchestrator")


def main() -> dict:
    run_id = utc_now().strftime("%Y%m%d%H%M%S")
    report = {"run_id": run_id, "roll_no": ROLL_NO, "tasks": []}
    status = "SUCCESS"
    for t in topological_order():
        retries = DAG[t][2]
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                run_task(t)
                report["tasks"].append({"task": t, "attempt": attempt + 1, "status": "SUCCESS",
                                        "seconds": round(time.time() - t0, 2)})
                log.info("task %-14s SUCCESS in %.1fs", t, time.time() - t0)
                break
            except Exception as exc:
                report["tasks"].append({"task": t, "attempt": attempt + 1, "status": "FAILED",
                                        "error": repr(exc), "seconds": round(time.time() - t0, 2)})
                log.error("task %s attempt %d failed: %s", t, attempt + 1, exc)
                if attempt == retries:
                    status = f"FAILED at {t}"
                    traceback.print_exc()
        if status != "SUCCESS":
            break
    report["status"] = status
    write_json(report, REPORTS / "orchestration" / f"run_{run_id}.json")
    log.info("Pipeline run %s: %s", run_id, status)
    return report


if __name__ == "__main__":
    main()
