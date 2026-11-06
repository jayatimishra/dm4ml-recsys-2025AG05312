# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Compact two-row architecture diagram for the report (docs/architecture_compact.png).
The full diagram is docs/architecture.mmd -> docs/architecture.png (Mermaid)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from src.common import INK, INK2, ROOT, SERIES, new_fig, save_fig  # noqa: E402

STORE, GATE, STEP = "#eef4fc", "#fdf1ea", "#f3f3f1"
ROW1 = [("Sources", "Clickstream CSV\nCRM users CSV\nCatalog REST API", STEP),
        ("1 Ingestion", "CDC high-water mark\nsnapshot, paged API\nretry + manifests", STEP),
        ("data/raw", "immutable\npartitioned\nappend-only", STORE),
        ("2 Validation gate", "29 contract checks\nquality gate\nerrors -> data/errors", GATE),
        ("3 Preparation", "repair, impute\nstandardise\nEDA", STEP)]
ROW2 = [("4 Transformation", "SQLite star schema\npoint-in-time SQL\nsplits + UI matrix", STORE),
        ("5 Feature store", "registry (37 feat.)\noffline Parquet\nonline key-value", STORE),
        ("6 Pre-training gate", "Pandera schemas\nleakage check\nPSI drift", GATE),
        ("7-8 Train + evaluate", "4 recommenders\n+ rating models\nMLflow + registry", STEP),
        ("9 Serving", "champion model\nbatch + HTTP\nonline features", STEP)]


def main():
    fig, ax = new_fig(10, 3.9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")
    w, h, gap = 16.5, 11.5, 3.6

    def row(items, y, reverse=False):
        xs = [2 + i * (w + gap) for i in range(len(items))]
        if reverse:
            xs = xs[::-1]
        centers = []
        for (title, body, color), x in zip(items, xs):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                        fc=color, ec=SERIES[1] if color == GATE else (SERIES[0] if color == STORE else "#b9b8b2"),
                                        lw=1.1))
            ax.text(x + w / 2, y + h - 2.1, title, ha="center", va="center", fontsize=8.6,
                    weight="bold", color=INK)
            ax.text(x + w / 2, y + 4.3, body, ha="center", va="center", fontsize=7.2, color=INK2,
                    linespacing=1.35)
            centers.append((x, y))
        return centers

    top = row(ROW1, 25)
    bot = row(ROW2, 8, reverse=False)
    arrow = dict(arrowstyle="-|>", mutation_scale=10, color="#6b6a65", lw=1.1)
    for (x1, y1), (x2, _) in zip(top, top[1:]):
        ax.add_patch(FancyArrowPatch((x1 + w + 0.4, y1 + h / 2), (x2 - 0.4, y1 + h / 2), **arrow))
    for (x1, y1), (x2, _) in zip(bot, bot[1:]):
        ax.add_patch(FancyArrowPatch((x1 + w + 0.4, y1 + h / 2), (x2 - 0.4, y1 + h / 2), **arrow))
    lx, ly = top[-1]
    fx, fy = bot[0]
    ax.add_patch(FancyArrowPatch((lx + w / 2, ly - 0.4), (fx + w / 2, fy + h + 0.4),
                                 connectionstyle="arc3,rad=-0.08", **arrow))
    ax.text(50, 3.2, "Orchestration: Prefect flow (executed) / Airflow DAG / Python runner, one shared DAG    |    "
            "Versioning: Git + DVC (dvc.yaml, dvc.lock, tags data-v1, data-v2)",
            ha="center", va="center", fontsize=7.6, color=INK2,
            bbox=dict(boxstyle="round,pad=0.45", fc="#fcfcfb", ec="#d6d5cf", ls="--"))
    ax.set_title("Pipeline architecture", loc="left")
    save_fig(fig, ROOT / "docs" / "architecture_compact.png")


if __name__ == "__main__":
    main()
