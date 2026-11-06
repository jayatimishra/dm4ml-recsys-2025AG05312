# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 7: PREDICTION SERVICE.

Loads the model registered as models:/recsys_2025AG05312@champion from the MLflow
registry and enriches each recommendation with product features read from the
ONLINE feature store (same definitions used in training).

  python -m src.serving.serve --batch                -> reports/sample_recommendations.csv
  python -m src.serving.serve --serve                -> http://127.0.0.1:8600/recommend?user_id=42&k=10
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import mlflow
import pandas as pd

from src.common import (MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, REPORTS, ROLL_NO, get_logger,
                    write_csv, write_json)
from src.features.feature_store import FeatureStore

log = get_logger("serving")
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@champion"


class RecommendationService:
    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = mlflow.pyfunc.load_model(MODEL_URI)
        self.fs = FeatureStore()
        self.version = mlflow.MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion").version

    def recommend(self, user_id: int, k: int = 10) -> pd.DataFrame:
        recs = self.model.predict(pd.DataFrame({"user_id": [float(user_id)], "k": [float(k)]}))
        items = pd.concat([self.fs.get_online_features("item_popularity_quality", {"product_id": int(p)})
                           for p in recs.product_id], ignore_index=True)
        out = recs.merge(items[["product_id", "category", "brand", "price", "i_avg_rating", "in_stock"]],
                         on="product_id", how="left")
        out["model_version"] = self.version
        return out


def batch(users=(7, 42, 314, 1001, 99999)) -> pd.DataFrame:
    svc = RecommendationService()
    df = pd.concat([svc.recommend(u, 10) for u in users], ignore_index=True)
    prof = pd.concat([svc.fs.get_online_features("user_profile_behaviour", {"user_id": u}) for u in users])
    write_csv(df, REPORTS / "sample_recommendations.csv")
    write_json({"model_uri": MODEL_URI, "model_version": svc.version,
                "user_profiles": prof[["user_id", "u_favourite_category", "u_n_events",
                                       "u_n_purchases"]].astype(str).to_dict("records")},
               REPORTS / "serving_context.json")
    log.info("Batch recommendations for %s written (model v%s); user 99999 is a cold-start test",
             list(users), svc.version)
    return df


def serve(port: int = 8600):
    svc = RecommendationService()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path == "/health":
                body, code = {"status": "ok", "model": MODEL_URI, "version": svc.version}, 200
            elif url.path == "/recommend" and "user_id" in q:
                recs = svc.recommend(int(q["user_id"][0]), int(q.get("k", ["10"])[0]))
                body, code = {"served_by": ROLL_NO, "recommendations": recs.to_dict("records")}, 200
            else:
                body, code = {"error": "use /recommend?user_id=<id>&k=<n>"}, 400
            payload = json.dumps(body, default=str).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

    log.info("Serving %s on http://127.0.0.1:%d/recommend?user_id=42", MODEL_URI, port)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--batch", action="store_true")
    a = ap.parse_args()
    serve() if a.serve else batch()
