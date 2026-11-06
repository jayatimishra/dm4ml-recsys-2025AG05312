# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Apache Airflow DAG for the same pipeline (alternative to the Prefect flow).

Deploy: copy this file into $AIRFLOW_HOME/dags,
set RECSYS_HOME to the project root, then `airflow dags trigger recsys_pipeline_2025AG05312`.
Stages run as BashOperators so each is an isolated process (same commands DVC uses),
with retries, an SLA, and a failure callback that writes to the pipeline log.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

ROLL_NO = "2025AG05312"
HOME = os.environ.get("RECSYS_HOME", "/opt/recsys")
STAGES = [  # (task_id, module + args, retries) -- same order as dags/pipeline_dag.py
    ("simulate_sources", "src.ingestion.generate_sources --end-date 2026-11-01", 1),
    ("ingest", "src.ingestion.ingest --mode incremental", 3),
    ("validate", "src.validation.validate", 0),
    ("prepare", "src.preparation.prepare", 1),
    ("transform", "src.transformation.transform", 1),
    ("feature_store", "src.features.feature_store", 1),
    ("pre_training_checks", "src.validation.pre_training_checks", 0),
    ("train", "src.training.train", 1),
    ("evaluate", "src.evaluation.evaluate", 1),
    ("serve_batch", "src.serving.serve --batch", 1),
]


def _on_failure(context):
    with open(os.path.join(HOME, "logs", "airflow_failures.log"), "a") as f:
        f.write(f"{ROLL_NO} | {context['task_instance'].task_id} failed at {datetime.utcnow()}\n")


default_args = {
    "owner": f"ml-platform-{ROLL_NO}",
    "retry_delay": timedelta(minutes=5),
    "on_failure_callback": _on_failure,
    "execution_timeout": timedelta(minutes=30),
}

with DAG(
    dag_id=f"recsys_pipeline_{ROLL_NO}",
    description="E-commerce recommendation data pipeline (AIML ZG529 assignment)",
    start_date=datetime(2026, 11, 1),
    schedule="30 21 * * *",          # 21:00 UTC = 02:30 IST daily
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=[ROLL_NO, "recsys", "dmml"],
) as dag:
    prev = None
    for task_id, script, retries in STAGES:
        t = BashOperator(task_id=task_id, retries=retries,
                         bash_command=f"cd {HOME} && python -m {script}")
        if prev:
            prev >> t
        prev = t
