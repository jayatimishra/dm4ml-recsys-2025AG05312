<!-- Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment 2 -->
# End-to-End Data Management Pipeline for a Recommendation System

**Roll No: 2025AG05312** · AIML ZG529 Data Management for Machine Learning · Assignment 2

An e-commerce product recommender built as a reproducible data pipeline:

```
simulate sources -> ingest -> validate -> prepare -> transform -> feature store
                 -> pre-training checks -> train -> evaluate -> serve
```

Final report: [`docs/Final_Report_2025AG05312.pdf`](docs/Final_Report_2025AG05312.pdf) ·
Architecture: [`docs/architecture.png`](docs/architecture.png) ·
Results: [`reports/model/results_table.md`](reports/model/results_table.md)

## Results at a glance (test window, K = 10, 921 users)

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Diversity |
|---|---|---|---|---|---|
| Popularity (baseline) | 0.0312 | 0.1312 | 0.0851 | 0.068 | 0.896 |
| **Item-KNN CF (selected)** | **0.0451** | **0.1850** | **0.1244** | 0.455 | 0.719 |
| SVD matrix factorisation | 0.0367 | 0.1504 | 0.0990 | 0.382 | 0.653 |
| Hybrid (SVD + content + popularity) | 0.0359 | 0.1505 | 0.1007 | 0.710 | 0.341 |

Rating prediction (supplementary): global mean RMSE 0.9688 / MAE 0.7923, bias model RMSE 0.9201 / MAE 0.7587.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run the pipeline

Pick **one** of the following. Each runs the same DAG (defined once in `dags/pipeline_dag.py`) and
takes about a minute.

```bash
# A. Prefect flow (orchestrator used for the submitted run)
python dags/prefect_flow.py
#    optional UI:  prefect server start   ->  http://127.0.0.1:4200
#    daily schedule (02:30): python dags/prefect_flow.py --serve

# B. DVC (reproducible, versioned; only re-runs stages whose inputs changed)
dvc repro
dvc metrics show

# C. Plain Python runner (no orchestrator dependency)
python dags/run_pipeline.py
```

Any single stage can be run on its own, e.g. `python -m src.validation.validate`.

Inspect results:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db        # http://127.0.0.1:5000 (after your own run)
mlflow ui --backend-store-uri sqlite:///docs/evidence/mlflow_submitted_runs.db   # runs/metrics of the submitted run
python -m src.serving.serve --serve                       # http://127.0.0.1:8600/recommend?user_id=42&k=10
open reports/validation/validation_report.html
```

### Reproduce the two data versions

```bash
git checkout data-v1 && dvc checkout     # history up to 17 Oct 2026 (needs the DVC cache/remote,
                                         # which is not inside the ZIP: regenerate as below)
git checkout main && dvc checkout        # full history up to 31 Oct 2026
dvc diff data-v1 data-v2                 # what changed between versions
```

To recreate both versions from scratch: set `source.end_date: "2026-10-18"` in
`configs/params.yaml`, run `dvc repro`, commit; then set `"2026-11-01"` and run `dvc repro` again.
The second run ingests only the 14,798 new events (incremental high-water mark).
Point DVC at your own remote with `dvc remote modify localstore url <path-or-s3-uri>`.

## Project structure

```
configs/params.yaml            pipeline parameters (tracked by DVC)
dags/                          pipeline_dag.py (single DAG definition), prefect_flow.py,
                               airflow_dag.py, run_pipeline.py
src/common.py                  paths, logging, watermark-aware I/O helpers
src/ingestion/                 generate_sources.py (documented synthetic sources),
                               mock_product_api.py (REST source), ingest.py
src/validation/                validate.py (pre-ingestion contracts + quality gate),
                               pre_training_checks.py (Pandera, leakage, PSI drift)
src/preparation/prepare.py     cleaning, imputation, standardisation, EDA
src/transformation/            transform.py + sql/schema.sql (star schema) + sql/features.sql
                               (point-in-time feature SQL); splits and derived datasets
src/features/feature_store.py  registry, offline store, online store, catalog generator
src/training/                  recommenders.py (popularity, item-KNN, SVD, hybrid),
                               rating_model.py, train.py (tuning, test, MLflow registry)
src/evaluation/                metrics.py (P/R/NDCG/MAP/HitRate@K, coverage, diversity,
                               novelty, RMSE/MAE), evaluate.py (comparison table, charts)
src/serving/serve.py           prediction service (batch + HTTP) on the champion model
tools/                         report, diagram and watermark-check scripts
dvc.yaml, dvc.lock             versioned pipeline definition + pinned hashes
data/  (generated)             sources/ raw/ validated/ errors/ processed/{clean,splits,derived}/
                               warehouse/ features/
reports/                       validation, EDA, analytics, feature catalog, model results,
                               orchestration and versioning evidence
docs/                          final report, architecture diagrams, screenshots
```

## Deliverables map

| Brief deliverable | Where |
|---|---|
| Source code in modules | `src/`, `dags/` |
| README, requirements.txt | this file, `requirements.txt` |
| Architecture diagram | `docs/architecture.png` (full), `docs/architecture_compact.png` |
| Validation report / logs | `reports/validation/validation_report.html`, `validation_checks.csv`, `quality_gate.csv`, `pre_training_report.json`; error dataset `data/errors/` |
| Feature catalog | `reports/feature_catalog.md`, `reports/feature_registry.csv` |
| Model evaluation comparing models | `reports/model/results_table.md`, `ranking_metrics.png`, `beyond_accuracy.png`, `rating_results.csv` |
| Orchestration DAG + screenshot | `dags/prefect_flow.py`, `dags/airflow_dag.py`; `docs/screenshots/prefect_flow_run.png` |
| Experiment tracking | `docs/evidence/mlflow_submitted_runs.db` (params + metrics of every submitted run); `docs/screenshots/mlflow_experiment_runs.png`, `mlflow_model_registry.png` |
| Data versioning | `dvc.yaml`, `dvc.lock`, Git tags `data-v1`/`data-v2`, `reports/versioning/dvc_evidence.txt` |
| Final report (PDF) | `docs/Final_Report_2025AG05312.pdf` |

## Data

Synthetic, generated deterministically (seed 529) by `src/ingestion/generate_sources.py`: 1,500 users,
600 products, 77,217 raw events (view, add_to_cart, purchase, optional 1-5 rating) from 5 May to
31 Oct 2026. Behaviour follows a latent-factor model so collaborative filtering has real signal.
Data-quality defects are injected on purpose (duplicates, null and orphan keys, invalid ratings,
malformed timestamps, casing drift, price-as-text, negative prices) and are caught by validation.

## Watermark

Every file carries the roll number **2025AG05312**: a header comment in code/config/SQL/Markdown,
a `#` comment line in CSVs (read with `comment="#"`), a `_watermark` key in JSON, file metadata in
Parquet, a `_watermark` table in SQLite databases, and a visible stamp on every PNG and PDF page.
Check with `python tools/check_watermarks.py`.
