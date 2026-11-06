<!-- Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment 2 -->
# Run every stage and screenshot it (Roll No: 2025AG05312)

Run these in **Terminal on your own machine**, from the repo folder. After each command,
take a screenshot of the Terminal window (macOS: `Cmd + Shift + 4`, then `Space`, then click
the window) and save it into `docs/execution_mac/` with the file name shown. Keep your
username in the prompt visible: that is the proof you ran it.

## 0. Setup (once)

```bash
cd ~/Downloads/dm4ml-recsys-2025AG05312          # wherever you unzipped it
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
mkdir -p docs/execution_mac
rm -rf data/sources data/raw data/validated data/processed/clean data/processed/splits \
       data/warehouse data/features models/*.pkl mlflow.db mlartifacts logs    # start from empty
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS="ignore::UserWarning,ignore::FutureWarning"
```

Screenshot `00_setup.png` after:

```bash
whoami && python --version && git log --oneline --decorate -3
```

## 1-9. One stage at a time

| Save as | Command |
|---|---|
| `01_simulate_sources.png` | `python -m src.ingestion.generate_sources --end-date 2026-11-01` |
| `02_ingest.png` | `python -m src.ingestion.ingest --mode incremental && find data/raw -type f \| sort` |
| `03_validate.png` | `python -m src.validation.validate && head -5 data/errors/interactions_rejected.csv` |
| `04_prepare.png` | `python -m src.preparation.prepare && ls reports/eda` |
| `05_transform.png` | `python -m src.transformation.transform && ls data/processed/splits data/processed/derived` |
| `06_feature_store.png` | `python -m src.features.feature_store && head -20 reports/feature_catalog.md` |
| `07_pre_training_checks.png` | `python -m src.validation.pre_training_checks` |
| `08_train.png` | `python -m src.training.train 2>&1 \| tail -25` |
| `09_evaluate.png` | `python -m src.evaluation.evaluate && cat reports/model/results_table.md` |
| `10_serve.png` | `python -m src.serving.serve --batch && head -12 reports/sample_recommendations.csv` |

Optional live API check (second Terminal tab):

```bash
python -m src.serving.serve --serve                    # tab 1, leave running
curl -s "http://127.0.0.1:8600/recommend?user_id=42&k=5" | python -m json.tool   # tab 2
```

## 10. Orchestration

`11_orchestration.png`:

```bash
python dags/run_pipeline.py 2>&1 | grep orchestrator
```

Prefect run with its UI (optional, stronger evidence):

```bash
prefect server start                                   # tab 1, leave running
export PREFECT_API_URL=http://127.0.0.1:4200/api        # tab 2
python dags/prefect_flow.py
```

Open http://127.0.0.1:4200, click the latest flow run, screenshot the browser and save it as
`docs/screenshots/prefect_flow_run.png` (replaces the existing one).

MLflow UI (optional): `mlflow ui --backend-store-uri sqlite:///mlflow.db`, open
http://127.0.0.1:5000, screenshot the Training runs and Models pages into `docs/screenshots/`
as `mlflow_experiment_runs.png` and `mlflow_model_registry.png`.

## 11. Versioning

`12_dvc.png`:

```bash
dvc dag && dvc status && git tag
```

## Build the PDF from your screenshots

```bash
python tools/watermark_assets.py        # stamps Roll No 2025AG05312 on your PNGs
python tools/build_execution_pdf.py     # uses docs/execution_mac/*.png wherever present
open docs/Execution_Screenshots_2025AG05312.pdf
python tools/check_watermarks.py        # should report 0 missing
```

Missing screenshots fall back to the reference captures in `docs/execution/`, so you can
replace them one at a time. Commit and push when done:

```bash
git add -A && git commit -m "Execution screenshots from my machine (2025AG05312)" && git push
```
