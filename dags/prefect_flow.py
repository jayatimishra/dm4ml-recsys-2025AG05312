# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Prefect 3 orchestration of the recommendation data pipeline.

Run once:        python dags/prefect_flow.py
Schedule daily:  python dags/prefect_flow.py --serve   (cron 02:30 IST, needs a Prefect server)
UI:              prefect server start   -> http://127.0.0.1:4200

Each stage is a Prefect task with retries, a timeout and logging. Task results
are passed downstream so Prefect records the dependency graph; the validation
task has no retries because a failed quality gate should stop the run, not repeat it.
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta

from prefect import flow, get_run_logger, task

from pipeline_dag import DAG, run_task, topological_order

ROLL_NO = "2025AG05312"


def make_task(task_id: str):
    retries = DAG[task_id][2]

    @task(name=task_id, retries=retries, retry_delay_seconds=10, timeout_seconds=1800,
          tags=[ROLL_NO, "recsys"])
    def _t(*upstream):
        logger = get_run_logger()
        logger.info("[%s] starting %s", ROLL_NO, task_id)
        result = run_task(task_id)
        logger.info("[%s] finished %s", ROLL_NO, task_id)
        return json.loads(json.dumps(result, default=str)) if isinstance(result, dict) else str(type(result))
    return _t


@flow(name=f"recsys-data-pipeline-{ROLL_NO}", log_prints=True,
      description="simulate sources -> ingest -> validate -> prepare -> transform -> feature store "
                  "-> pre-training checks -> train -> evaluate -> serve")
def recsys_pipeline():
    tasks = {t: make_task(t) for t in DAG}
    results = {}
    for t in topological_order():                      # upstream results passed in => explicit edges
        results[t] = tasks[t](*[results[u] for u in DAG[t][1]])
    return {"winner": results["train"].get("winner"), "registered_version": results["train"].get("version")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="register a daily deployment")
    a = ap.parse_args()
    if a.serve:
        recsys_pipeline.serve(name=f"daily-{ROLL_NO}", cron="30 2 * * *")
    else:
        print(recsys_pipeline())
