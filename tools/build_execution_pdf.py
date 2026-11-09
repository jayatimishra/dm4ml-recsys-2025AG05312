# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Build docs/Execution_Screenshots_2025AG05312.pdf: GitHub link + execution
screenshots for every pipeline stage (from tools/capture_execution.py) and the
UI screenshots (Prefect, MLflow, validation report). Watermarked on every page."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image as PILImage  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,  # noqa: E402
                                Spacer, Table, TableStyle)

from tools.build_report import watermark  # noqa: E402
from src.common import ROLL_NO, ROOT  # noqa: E402

GITHUB = "https://github.com/jayatimishra/dm4ml-recsys-2025AG05312"
OUT = ROOT / "docs" / f"Execution_Screenshots_{ROLL_NO}.pdf"
EX, SS, RM = ROOT / "docs" / "execution", ROOT / "docs" / "screenshots", ROOT / "reports"
MAC = ROOT / "docs" / "execution_mac"   # your own terminal screenshots (see docs/RUN_AND_SCREENSHOT.md)

# (caption, image path, note) grouped by brief stage
SECTIONS = [
    ("Setup", [("Environment and Git history (tags data-v1, data-v2)", EX / "00_setup.png", "")]),
    ("1. Data ingestion", [
        ("Synthetic source systems generated (documented, seed 529)", EX / "01_simulate_sources.png", ""),
        ("Ingestion of CSV exports + paginated REST API into the partitioned raw zone", EX / "02_ingest.png",
         "The WARNING line is the deliberate HTTP 503 from the mock API, recovered by retry with backoff.")]),
    ("2. Data validation", [
        ("Validation run, quality gate and the error dataset (data/errors)", EX / "03_validate.png", ""),
        ("Validation report (reports/validation/validation_report.html)", SS / "validation_report_html.png",
         "freshness<=7d shows FAIL because the synthetic event log ends on 31 Oct and this run was on 9 Nov "
         "(8 days stale). Freshness is a warning-level gate check by design, so the overall gate still PASSED; "
         "in production this would alert that the upstream export has stopped arriving.")]),
    ("3. Data preparation", [("Cleaning, standardisation and EDA", EX / "04_prepare.png", ""),
                             ("EDA: item long tail", RM / "eda" / "03_item_long_tail.png", "")]),
    ("4. Data transformation", [("Star schema load, point-in-time features, time-based splits, derived datasets",
                                 EX / "05_transform.png", "")]),
    ("5. Feature management", [("Feature registry, offline/online materialisation and an online lookup",
                                EX / "06_feature_store.png", "")]),
    ("6. Pre-training validation", [("Pandera schemas, temporal leakage check, PSI drift",
                                     EX / "07_pre_training_checks.png", "")]),
    ("7. Model training and experiment tracking", [
        ("Tuning on validation window, one-shot test, release + MLflow registry", EX / "08_train.png", ""),
        ("MLflow: experiment runs (tuning, test and release runs for each pipeline execution)", SS / "mlflow_experiment_runs.png", ""),
        ("MLflow model registry: recsys_2025AG05312 versions 1-3, latest carries alias 'champion'", SS / "mlflow_model_registry.png", "")]),
    ("8. Model evaluation", [("Results table comparing all models", EX / "09_evaluate.png", ""),
                             ("Accuracy metrics", RM / "model" / "ranking_metrics.png", ""),
                             ("Beyond-accuracy metrics", RM / "model" / "beyond_accuracy.png", "")]),
    ("9. Prediction service", [("Batch recommendations and HTTP /recommend endpoint", EX / "10_serve.png", "")]),
    ("10. Orchestration and reproducibility", [
        ("Prefect flow run noble-porcupine: Completed, 10 task runs in 29 s, dependency graph and logs", SS / "prefect_flow_run.png", ""),
        ("Prefect: run history", SS / "prefect_runs_list.png", ""),
        ("Full DAG run with the dependency-free runner", EX / "11_orchestration.png", ""),
        ("DVC pipeline DAG and data-version diff", EX / "12_dvc.png", "")]),
]


COMMANDS = [
    ("Setup", "whoami && python --version && git log --oneline --decorate -3"),
    ("Source systems", "python -m src.ingestion.generate_sources --end-date 2026-11-01"),
    ("1 Ingestion", "python -m src.ingestion.ingest --mode incremental"),
    ("2 Validation", "python -m src.validation.validate"),
    ("3 Preparation", "python -m src.preparation.prepare"),
    ("4 Transformation", "python -m src.transformation.transform"),
    ("5 Feature store", "python -m src.features.feature_store"),
    ("6 Pre-training checks", "python -m src.validation.pre_training_checks"),
    ("7 Training", "python -m src.training.train"),
    ("8 Evaluation", "python -m src.evaluation.evaluate"),
    ("9 Serving", "python -m src.serving.serve --batch"),
    ("10 Orchestration", "python dags/run_pipeline.py  |  python dags/prefect_flow.py"),
    ("11 Versioning", "dvc dag && dvc status && git tag"),
]


def pick(path: Path) -> Path:
    """Prefer a real terminal screenshot docs/execution_mac/<same name> when present."""
    own = MAC / path.name
    return own if path.parent == EX and own.exists() else path


TMP = Path(tempfile.mkdtemp())


def img(path: Path, max_w=180 * mm, max_h=200 * mm, max_px=1650):
    """Embed a downscaled copy (about 230 dpi at print size) so the PDF stays small."""
    path = pick(path)
    im = PILImage.open(path)
    w, h = im.size
    if w > max_px:
        im = im.convert("RGB").resize((max_px, round(h * max_px / w)), PILImage.LANCZOS)
        im = im.quantize(colors=256, method=PILImage.Quantize.FASTOCTREE, dither=PILImage.Dither.NONE)
        path = TMP / f"{path.parent.name}_{path.name}"
        im.save(path, optimize=True)
        w, h = im.size
    s = min(max_w / w, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def main():
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1c5cab"),
                        spaceBefore=4, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    cap = ParagraphStyle("c", parent=body, fontSize=8.2, textColor=colors.HexColor("#52514e"), spaceAfter=2)
    own = any(MAC.glob("*.png"))
    story = [Paragraph("Execution Screenshots: End-to-End Recommendation Data Pipeline", h1),
             Paragraph(f"<b>Roll No:</b> {ROLL_NO} &nbsp;|&nbsp; AIML ZG529 Data Management for Machine Learning, "
                       "Assignment 2", body), Spacer(1, 6),
             Paragraph(f"<b>GitHub repository:</b> <link href='{GITHUB}' color='#1c5cab'><u>{GITHUB}</u></link>", body),
             Spacer(1, 6),
             Paragraph("All screenshots were taken on my own machine: terminal output from running each stage in order "
                       "from an empty data folder (commands in docs/RUN_AND_SCREENSHOT.md), then the Prefect UI, the "
                       "MLflow UI and the HTML validation report from the same run." if own else
                       "Terminal panels are captured output of each command (tools/capture_execution.py), run in "
                       "order from an empty data folder.", body), Spacer(1, 4)]
    rows = [["#", "Stage", "Command"]] + [
        [str(i), title, Paragraph(f"<font face='Courier' size='6.8'>{cmd}</font>", body)]
        for i, (title, cmd) in enumerate(COMMANDS)]
    t = Table(rows, colWidths=[8 * mm, 52 * mm, 120 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    story += [t, Spacer(1, 4), Paragraph("Every stage completed successfully; the full-DAG run in section 10 reports "
                                         "SUCCESS for all 10 tasks.", cap), PageBreak()]

    for title, items in SECTIONS:
        first = True
        for caption, path, note in items:
            block = ([Paragraph(title, h2)] if first else []) + [Paragraph(caption, cap), img(path)]
            if note:
                block.append(Paragraph(note, cap))
            story += [KeepTogether(block), Spacer(1, 7)]
            first = False

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                            bottomMargin=16 * mm, title=f"Execution screenshots - {ROLL_NO}", author=f"Roll No {ROLL_NO}")
    doc.build(story, onFirstPage=watermark, onLaterPages=watermark)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
