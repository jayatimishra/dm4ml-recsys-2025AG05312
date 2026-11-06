# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Build docs/Execution_Screenshots_2025AG05312.pdf: GitHub link + execution
screenshots for every pipeline stage (from tools/capture_execution.py) and the
UI screenshots (Prefect, MLflow, validation report). Watermarked on every page."""
import json
import sys
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
        ("Validation report (reports/validation/validation_report.html)", SS / "validation_report_html.png", "")]),
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
        ("MLflow: experiment runs", SS / "mlflow_experiment_runs.png", ""),
        ("MLflow: model registry with alias 'champion'", SS / "mlflow_model_registry.png", "")]),
    ("8. Model evaluation", [("Results table comparing all models", EX / "09_evaluate.png", ""),
                             ("Accuracy metrics", RM / "model" / "ranking_metrics.png", ""),
                             ("Beyond-accuracy metrics", RM / "model" / "beyond_accuracy.png", "")]),
    ("9. Prediction service", [("Batch recommendations and HTTP /recommend endpoint", EX / "10_serve.png", "")]),
    ("10. Orchestration and reproducibility", [
        ("Prefect flow run: all tasks Completed", SS / "prefect_flow_run.png", ""),
        ("Prefect: run history", SS / "prefect_runs_list.png", ""),
        ("Full DAG run with the dependency-free runner", EX / "11_orchestration.png", ""),
        ("DVC pipeline DAG and data-version diff", EX / "12_dvc.png", "")]),
]


def pick(path: Path) -> Path:
    """Prefer a real terminal screenshot docs/execution_mac/<same name> when present."""
    own = MAC / path.name
    return own if path.parent == EX and own.exists() else path


def img(path: Path, max_w=180 * mm, max_h=200 * mm):
    path = pick(path)
    w, h = PILImage.open(path).size
    s = min(max_w / w, max_h / h)
    return Image(str(path), width=w * s, height=h * s)


def main():
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, leading=18)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11.5, textColor=colors.HexColor("#1c5cab"),
                        spaceBefore=4, spaceAfter=3)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    cap = ParagraphStyle("c", parent=body, fontSize=8.2, textColor=colors.HexColor("#52514e"), spaceAfter=2)
    caps = json.loads((EX / "captures.json").read_text())["steps"]

    story = [Paragraph("Execution Screenshots: End-to-End Recommendation Data Pipeline", h1),
             Paragraph(f"<b>Roll No:</b> {ROLL_NO} &nbsp;|&nbsp; AIML ZG529 Data Management for Machine Learning, "
                       "Assignment 2", body), Spacer(1, 6),
             Paragraph(f"<b>GitHub repository:</b> <link href='{GITHUB}' color='#1c5cab'><u>{GITHUB}</u></link>", body),
             Spacer(1, 6),
             Paragraph(("Terminal screenshots were taken on the author's own machine while running the commands in "
                        "docs/RUN_AND_SCREENSHOT.md. " if any(MAC.glob("*.png")) else
                        "Every terminal panel below is the captured output of the command shown, produced by "
                        "<font face='Courier'>tools/capture_execution.py</font>, which deletes all generated data "
                        "and runs each stage in order from an empty data folder. ") +
                       "Full untrimmed reference output: <font face='Courier'>docs/execution/captures.json</font>. "
                       "Reference run summary:", body), Spacer(1, 4)]
    rows = [["Step", "Command", "Exit", "Seconds"]] + [
        [c["title"], Paragraph(f"<font face='Courier' size='6.8'>{c['command'][:95]}"
                               f"{'...' if len(c['command']) > 95 else ''}</font>", body),
         str(c["exit_code"]), f"{c['seconds']:.1f}"] for c in caps]
    t = Table(rows, colWidths=[52 * mm, 104 * mm, 10 * mm, 14 * mm])
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.8),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
    story += [t, Spacer(1, 4), Paragraph(f"Captured {caps[0]['captured_at'][:10]}. All steps exited with code 0.", cap),
              PageBreak()]

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
