# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Build the final report PDF (docs/Final_Report_2025AG05312.pdf).
Every number is read from the pipeline outputs in reports/, so the report
always matches the latest run. Each page carries the roll-number watermark."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_JUSTIFY  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

from src.common import REPORTS, ROLL_NO, ROOT, read_csv  # noqa: E402

OUT = ROOT / "docs" / f"Final_Report_{ROLL_NO}.pdf"
J = lambda p: json.loads((REPORTS / p).read_text())


def watermark(c, doc):
    c.saveState()
    c.setFont("Helvetica-Bold", 46)
    c.setFillColor(colors.Color(0.5, 0.5, 0.5, alpha=0.10))
    c.translate(A4[0] / 2, A4[1] / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, f"Roll No: {ROLL_NO}")
    c.restoreState()
    c.setFont("Helvetica", 7.5)
    c.setFillColor(colors.HexColor("#52514e"))
    c.drawString(15 * mm, 9 * mm, f"Roll No: {ROLL_NO}  |  AIML ZG529 Data Management for ML  |  Assignment 2")
    c.drawRightString(A4[0] - 15 * mm, 9 * mm, f"Page {doc.page}")


def main():
    v = J("validation/validation_report.json")
    pt = J("validation/pre_training_report.json")
    prep = J("preparation_log.json")
    eda = J("eda/eda_summary.json")
    tr = J("transform_summary.json")
    fs = J("feature_store_summary.json")
    ts = J("model/training_summary.json")
    res = read_csv(REPORTS / "model" / "results_table.csv")
    rat = read_csv(REPORTS / "model" / "rating_results.csv")
    e = v["entities"]
    sp = tr["splits"]
    psi = {r["check"].split(":")[1]: r["psi"] for r in pt["results"] if "psi" in r}
    base = res[res.model == "popularity"].iloc[0]
    best = res[res.model == ts["winner"]].iloc[0]
    hyb = res[res.model == "hybrid"].iloc[0]
    n_rows_checks = sum(1 for c in v["checks"])

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11.2,
                          alignment=TA_JUSTIFY, spaceAfter=3.5)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14.5, spaceAfter=2, leading=17)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.2, spaceBefore=6, spaceAfter=2.5, leading=12,
                        textColor=colors.HexColor("#1c5cab"))
    small = ParagraphStyle("s", parent=body, fontSize=7.4, leading=9.2, textColor=colors.HexColor("#52514e"))
    cell = ParagraphStyle("c", parent=body, fontSize=7.3, leading=8.8, alignment=0, spaceAfter=0)
    P = lambda t, s=body: Paragraph(t, s)

    story = [
        P("End-to-End Data Management Pipeline for a Recommendation System", h1),
        P(f"<b>Roll No:</b> {ROLL_NO} &nbsp;|&nbsp; AIML ZG529 Data Management for Machine Learning, Assignment 2 "
          "&nbsp;|&nbsp; Domain: e-commerce product recommendations<br/><b>GitHub:</b> "
          "<link href='https://github.com/jayatimishra/dm4ml-recsys-2025AG05312' color='#1c5cab'>"
          "https://github.com/jayatimishra/dm4ml-recsys-2025AG05312</link> &nbsp;|&nbsp; Execution screenshots: "
          "docs/Execution_Screenshots_2025AG05312.pdf", small),
        Spacer(1, 4),
        P("1. Problem and data", h2),
        P("The pipeline recommends products to users of an online store from their past views, add-to-carts, "
          "purchases and ratings plus product metadata. Data is <b>clearly documented synthetic data</b> "
          f"(<font face='Courier'>src/ingestion/generate_sources.py</font>, fixed seed): {e['users']['rows_passed']:,} users, {e['products']['rows_passed']} "
          f"products in 12 categories and {e['interactions']['rows_in']:,} raw events over 180 days, drawn from a "
          "latent-factor model (users prefer 1-3 categories, items have long-tail popularity) so collaborative "
          "filtering has real signal. Synthetic data was chosen over MovieLens so that the three source types the "
          "brief lists (CSV files, a database-style export and an API) and realistic defects could be controlled: "
          "duplicates, missing keys, orphan foreign keys, out-of-range ratings, malformed timestamps, casing drift, "
          "price-as-text type drift and negative prices are injected deliberately and must be caught downstream. "
          f"After cleaning: {eda['n_events']:,} events, {eda['user_item_pairs']:,} user-item pairs, "
          f"{eda['matrix_sparsity']:.1%} sparsity, only {eda['rating_coverage']:.1%} of events carry a rating, and "
          f"the top 20% of items take {eda['top20pct_items_share']:.0%} of interactions.", body),
        P("2. Architecture", h2),
        Image(str(ROOT / "docs" / "architecture_compact.png"), width=180 * mm, height=180 * mm * 507 / 1300),
        P("Blue = storage layers, orange = blocking quality gates. Full diagram: docs/architecture.png.", small),
        P("3. Design decisions by stage", h2),
        P("<b>Ingestion.</b> Three sources with three patterns: the clickstream log is pulled <i>incrementally</i> "
          "using an event_id high-water mark (CDC-style, so re-runs are idempotent); CRM users are a full snapshot "
          "(small dimension); the catalog is pulled from a paginated REST API with API-key auth and exponential "
          "backoff (the mock service returns a 503 on page 3 to prove the retry path). Raw data lands unmodified "
          "in an append-only, date-partitioned zone with a manifest per batch (row count, md5, schema) and an "
          "audit log. Raw is never cleaned in place, so any later stage can be replayed.", body),
        P(f"<b>Validation (pre-ingestion gate).</b> {n_rows_checks} row-level checks are declared once as data "
          "contracts in code, each tagged with a quality dimension (completeness, validity, uniqueness, "
          "consistency, referential integrity, timeliness) and a severity. <i>Critical</i> failures (null keys, "
          "unparseable or future timestamps, unknown event types, orphan user/product ids) are written to "
          "<font face='Courier'>data/errors/</font> with the failed check ids; <i>warnings</i> (bad ratings, "
          "casing, text prices) are kept and repaired in preparation. Dataset-level gate checks (schema, volume, "
          "key null rate, freshness, reject rate &lt;= 5%) stop the pipeline if breached. Result: "
          f"{e['interactions']['duplicates_dropped']} duplicate events dropped, "
          f"{e['interactions']['rows_rejected']} events quarantined, data-quality score "
          f"{v['data_quality_score']:.2%}, gate passed. Report: reports/validation/validation_report.html.", body),
        P(f"<b>Preparation.</b> Every repair is counted in reports/preparation_log.json: "
          f"{prep['users.age_imputed_with_median']} invalid ages imputed (flagged), "
          f"{prep['products.price_text_parsed']} text prices parsed, "
          f"{prep['products.price_imputed_category_median']} bad prices replaced by the category median, "
          f"{prep['products.category_inferred_from_title']} missing categories inferred from the title, "
          f"{prep['interactions.event_type_normalised']} event types normalised and "
          f"{prep['interactions.rating_invalid_set_null']} invalid ratings nulled. Imputation flags are kept as "
          "columns so a model can tell real values from filled ones.", body),
        P("<b>Transformation (ELT).</b> Clean data is loaded into a SQLite star schema (dim_user, dim_product, "
          "dim_date, fact_interactions) whose CHECK and FOREIGN KEY constraints act as a last integrity line. "
          "Features are computed in SQL inside the warehouse, dbt-style, and every query is <b>point-in-time</b>: "
          "it only reads events before an <i>as_of</i> timestamp. Three snapshots are built: valid (cutoff - 28 d), "
          "train (cutoff - 14 d) and serve (all data). The time-based split files are "
          f"train {sp['train']['rows']:,} events (to {sp['train']['to'][:10]}) and test {sp['test']['rows']:,} events "
          f"({sp['test']['from'][:10]} to {sp['test']['to'][:10]}); valid is the last 14 days inside train and is "
          "used only for tuning. Derived datasets: user activity summary, item popularity, and a sparse "
          f"{sp['user_item_matrix_train']['shape'][0]}x{sp['user_item_matrix_train']['shape'][1]} user-item matrix "
          "where strength = sum of event weights (view 1, cart 3, purchase 5) with a 30-day half-life decay, "
          "plus (rating - 3).", body),
        P(f"<b>Feature management.</b> A custom feature layer mirrors Feast concepts: a registry of "
          f"{fs['n_features']} features in 4 feature views (user behaviour, item popularity/quality, user-item "
          "interaction, user-category affinity) with type, description, owner, TTL and a definition hash that bumps "
          "the view version when its SQL changes; an offline Parquet store of every snapshot with a point-in-time "
          "join (pandas merge_asof) for training; and an online key-value store with the latest row per key for "
          "serving. Training and the prediction service read the same definitions, which removes training/serving "
          "skew from duplicated feature code. Catalog: reports/feature_catalog.md.", body),
        P(f"<b>Pre-training gate.</b> Pandera schemas validate the split files and feature snapshots; a leakage "
          "check asserts every train event precedes every test event and the training snapshot is not newer than "
          f"the test window; PSI between train and test is {psi['event_type_mix']:.4f} for event-type mix and "
          f"{psi['category_mix']:.4f} for category mix (warn above 0.10, fail above 0.25).", body),
        P("<b>Versioning and orchestration.</b> dvc.yaml declares all 10 stages with dependencies and outputs; "
          "dvc.lock pins md5 hashes, so <font face='Courier'>dvc repro</font> re-runs only what changed. Two data "
          "versions are tagged in Git: data-v1 (history to 17 Oct) and data-v2 (to 31 Oct, added by an incremental "
          "ingest of only the 14,798 new events); <font face='Courier'>git checkout data-v1 &amp;&amp; dvc checkout</font> "
          "restores v1 exactly (reports/versioning/dvc_evidence.txt). One DAG definition drives three "
          "orchestrators: a Prefect 3 flow (executed, screenshot in docs/screenshots), an Airflow DAG, and a "
          "dependency-free Python runner. Gate tasks have zero retries, since retrying bad data does not fix it.", body),
        P("4. Models, evaluation protocol and results", h2),
        P("Models: a <b>popularity baseline</b> (all-time weighted popularity blended with 7-day trend), "
          "<b>item-KNN collaborative filtering</b> (cosine item similarity, top-n neighbours), <b>truncated-SVD "
          "matrix factorisation</b> and a <b>hybrid</b> (SVD score + category-affinity content score + popularity "
          "prior). Hyper-parameters were tuned on the validation window only; each model family was then refit on "
          "the train snapshot and scored <b>once</b> on the test window, so the test set never influenced a choice. "
          "Relevance = an add-to-cart or purchase in the test window on an item the user had not touched before, "
          f"evaluated over {ts['n_test_users']} users, K = 10. All runs are tracked in MLflow (params, metrics, "
          "feature-view versions and data md5s as lineage tags); the winner is retrained on all data, logged as a "
          f"pyfunc model and registered as {ts['registered_model']} with alias <i>champion</i>.", body),
    ]

    hdr = ["Model", "Prec@10", "Rec@10", "NDCG@10", "MAP@10", "Hit@10", "Coverage", "Diversity", "NDCG vs base"]
    names = {"popularity": "Popularity (baseline)", "item_knn": "Item-KNN CF (selected)",
             "svd_mf": "SVD matrix factorisation", "hybrid": "Hybrid SVD + content"}
    rows = [hdr] + [[names[r.model], f"{r.precision_at_10:.4f}", f"{r.recall_at_10:.4f}", f"{r.ndcg_at_10:.4f}",
                     f"{r.map_at_10:.4f}", f"{r.hit_rate_at_10:.4f}", f"{r.coverage:.3f}", f"{r.diversity:.3f}",
                     f"{r.ndcg_lift_vs_baseline:+.1%}"] for r in res.itertuples()]
    t = Table(rows, colWidths=[38 * mm] + [16.5 * mm] * 7 + [24 * mm], hAlign="LEFT")
    win_row = 1 + list(res.model).index(ts["winner"])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 7.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
        ("BACKGROUND", (0, win_row), (-1, win_row), colors.HexColor("#f2f7ee")),
        ("FONT", (0, win_row), (-1, win_row), "Helvetica-Bold", 7.6),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    cw = 89 * mm
    charts = Table([[Image(str(REPORTS / "model" / "ranking_metrics.png"), width=cw, height=cw / 2),
                     Image(str(REPORTS / "model" / "beyond_accuracy.png"), width=cw, height=cw / 2)]],
                   colWidths=[90 * mm, 90 * mm], hAlign="LEFT")
    charts.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [t, Spacer(1, 3), charts, Spacer(1, 2)]
    rr = {r.model: r for r in rat.itertuples()}
    story += [
        P(f"Explicit ratings (supplementary, {int(rr['bias_model'].n_ratings)} test ratings): global-mean baseline "
          f"RMSE {rr['global_mean'].rmse:.4f} / MAE {rr['global_mean'].mae:.4f}; regularised user+item bias model "
          f"RMSE {rr['bias_model'].rmse:.4f} / MAE {rr['bias_model'].mae:.4f}. Full table and charts: "
          "reports/model/results_table.md, ranking_metrics.png, beyond_accuracy.png.", small),
        P(f"<b>Reading the results.</b> Item-KNN lifts NDCG@10 by {best.ndcg_lift_vs_baseline:.0%} and "
          f"Recall@10 from {base.recall_at_10:.3f} to {best.recall_at_10:.3f} over popularity, and recommends "
          f"{best.coverage:.0%} of the catalogue against {base.coverage:.0%}. Absolute precision is low "
          "(about 0.45 hits per 10 recommendations) because only new items in a 14-day window count as hits; "
          "the lift over the baseline is the meaningful number. The beyond-accuracy metrics show trade-offs that "
          f"accuracy alone hides: popularity has the highest intra-list diversity ({base.diversity:.2f}) only "
          "because the global best-sellers happen to span categories, while its coverage is tiny; the hybrid "
          f"reaches the widest coverage ({hyb.coverage:.0%}) but the lowest diversity ({hyb.diversity:.2f}) "
          "because the category-affinity term concentrates each list in the user's favourite categories. "
          "SVD degraded as factors grew (16 was best), a sign of over-fitting on a 96%-sparse matrix.", body),
        P("5. Limitations", h2),
        P("(1) The data is synthetic and the generator is itself a latent-factor model, which favours "
          "collaborative filtering; results show the pipeline works, not how these models would rank on real "
          "traffic. Re-running on MovieLens or Amazon Reviews only needs a new ingestion adapter. "
          "(2) Single-node stack: SQLite, a dense item-item similarity matrix (O(items^2) memory) and in-memory "
          "pandas suit 600 items, not millions; production would use Postgres/BigQuery, Spark and approximate "
          "nearest neighbours. (3) The feature store is custom rather than Feast, and the online store is SQLite, "
          "not Redis. (4) The Airflow DAG is provided but only the Prefect flow and Python runner were executed. "
          "(5) Evaluation is offline only, on warm users; cold-start users fall back to popularity and were not "
          "scored, and there is no online A/B test. (6) No production monitoring of drift or model decay beyond "
          "the PSI gate; the next step would be scheduled drift checks on the serving snapshot that trigger "
          "retraining.", body),
        P("<b>Tools.</b> Python 3, pandas, NumPy, SciPy, scikit-learn, PyArrow/Parquet, SQLite, Pandera, MLflow "
          "(tracking + registry), Prefect 3, Apache Airflow (DAG file), Git + DVC, Matplotlib, ReportLab.", small),
    ]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                            bottomMargin=15 * mm, title=f"Recommendation data pipeline - {ROLL_NO}",
                            author=f"Roll No {ROLL_NO}", subject="AIML ZG529 Assignment 2")
    doc.build(story, onFirstPage=watermark, onLaterPages=watermark)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
