# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 0: simulate the upstream SOURCE SYSTEMS of an e-commerce company.

Not part of the pipeline proper. It stands in for systems the data team does
not own:
  1. Clickstream export (CSV)   -> data/sources/clickstream/interactions_export.csv
  2. CRM user export (CSV)      -> data/sources/crm/users_export.csv
  3. Product catalog service    -> data/sources/catalog/products.json (served by
                                   mock_product_api.py as a paginated REST API)

Behaviour is generated from a latent-factor model (users prefer categories,
items have long-tail popularity) so collaborative filtering has real signal.
Realistic data-quality defects are injected on purpose so the validation stage
has something to catch. Generation is deterministic (fixed seed); --end-date
truncates the event log so incremental ingestion and data versioning can be
demonstrated (v1 = partial history, v2 = full history).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common import ROLL_NO, SOURCES, WATERMARK, get_logger, write_csv

log = get_logger("ingestion.generate")

SEED = 529
N_USERS = 1500
N_ITEMS = 600
N_FACTORS = 8
START = pd.Timestamp("2026-05-05")
END = pd.Timestamp("2026-11-01")          # exclusive
CATEGORIES = ["Electronics", "Books", "Fashion", "Home & Kitchen", "Beauty", "Sports",
              "Toys", "Grocery", "Automotive", "Music", "Office", "Garden"]
BRANDS = [f"Brand{c}" for c in "ABCDEFGHIJKLMNOPQRST"]
CITIES = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Hyderabad", "Pune", "Kolkata",
          "Ahmedabad", "Jaipur", "Kochi"]
DEVICES = ["mobile", "desktop", "tablet"]


def build_world(rng: np.random.Generator):
    n_cat = len(CATEGORIES)
    cat_centers = rng.normal(0, 1, (n_cat, N_FACTORS))
    item_cat = rng.integers(0, n_cat, N_ITEMS)
    item_vec = cat_centers[item_cat] + rng.normal(0, 0.45, (N_ITEMS, N_FACTORS))
    # long-tail popularity bias (Zipf-like)
    pop_rank = rng.permutation(N_ITEMS) + 1
    item_bias = -0.9 * np.log(pop_rank)
    item_bias = (item_bias - item_bias.mean()) / item_bias.std() * 0.7

    # each user mixes 1-3 favourite categories
    user_vec = np.zeros((N_USERS, N_FACTORS))
    for u in range(N_USERS):
        k = rng.integers(1, 4)
        favs = rng.choice(n_cat, k, replace=False)
        w = rng.dirichlet(np.ones(k))
        user_vec[u] = w @ cat_centers[favs] + rng.normal(0, 0.3, N_FACTORS)
    return item_cat, item_vec, item_bias, user_vec


def make_products(rng, item_cat) -> list[dict]:
    created = START - pd.to_timedelta(rng.integers(0, 400, N_ITEMS), unit="D")
    base_price = {c: p for c, p in zip(CATEGORIES, [15000, 450, 1200, 2200, 650, 1800, 900,
                                                    300, 2500, 700, 500, 1100])}
    rows = []
    for i in range(N_ITEMS):
        cat = CATEGORIES[item_cat[i]]
        price = round(float(base_price[cat] * rng.lognormal(0, 0.5)), 2)
        rows.append({
            "product_id": 1001 + i,
            "title": f"{cat} item {1001 + i}",
            "category": cat,
            "brand": BRANDS[rng.integers(0, len(BRANDS))],
            "price": price,
            "currency": "INR",
            "created_at": created[i].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "in_stock": bool(rng.random() > 0.08),
        })
    # ---- injected catalog defects
    idx = rng.choice(N_ITEMS, 40, replace=False)
    for j in idx[:12]:
        rows[j]["price"] = None                          # missing price
    for j in idx[12:15]:
        rows[j]["price"] = -abs(rows[j]["price"])         # negative price
    for j in idx[15:21]:
        rows[j]["price"] = f"INR {rows[j]['price']}"      # type drift: string price
    for j in idx[21:33]:
        rows[j]["category"] = None                       # missing category
    for j in idx[33:36]:
        rows[j]["category"] = rows[j]["category"].upper()  # inconsistent casing
    rows.extend([dict(rows[j]) for j in idx[36:40]])     # duplicate records
    return rows


def make_users(rng) -> pd.DataFrame:
    signup = START - pd.to_timedelta(rng.integers(0, 700, N_USERS), unit="D")
    df = pd.DataFrame({
        "user_id": np.arange(1, N_USERS + 1),
        "signup_date": signup.strftime("%Y-%m-%d"),
        "age": rng.integers(18, 66, N_USERS).astype(float),
        "gender": rng.choice(["F", "M", "O"], N_USERS, p=[0.48, 0.48, 0.04]),
        "city": rng.choice(CITIES, N_USERS),
        "is_prime_member": rng.random(N_USERS) < 0.3,
    })
    # ---- injected CRM defects
    bad = rng.choice(N_USERS, 90, replace=False)
    df.loc[bad[:15], "age"] = rng.choice([-1, 0, 150, 999], 15)   # out of range
    df.loc[bad[15:35], "age"] = np.nan                            # missing
    df.loc[bad[35:80], "city"] = None                             # missing
    df.loc[bad[80:85], "gender"] = "unknown"                      # invalid domain
    df = pd.concat([df, df.iloc[bad[85:90]]], ignore_index=True)  # duplicates
    return df


def make_events(rng, item_cat, item_vec, item_bias, user_vec) -> pd.DataFrame:
    n_days = (END - START).days
    activity = np.clip(rng.lognormal(3.2, 0.8, N_USERS), 4, 400).astype(int)
    first_day = rng.integers(0, n_days - 20, N_USERS)
    recs = []
    for u in range(N_USERS):
        logits = user_vec[u] @ item_vec.T / 3.0 + item_bias
        p = np.exp(logits - logits.max())
        p /= p.sum()
        items = rng.choice(N_ITEMS, activity[u], p=p)
        aff = (user_vec[u] @ item_vec[items].T) / 3.0
        aff_n = 1 / (1 + np.exp(-(aff - 0.5)))
        days = rng.integers(first_day[u], n_days, activity[u])
        secs = rng.integers(0, 86400, activity[u])
        device = DEVICES[rng.choice(3, p=[0.62, 0.30, 0.08])]
        for k, it in enumerate(items):
            ts = START + pd.Timedelta(days=int(days[k]), seconds=int(secs[k]))
            sess = f"s{u + 1}_{days[k]}"
            recs.append((u + 1, 1001 + it, "view", np.nan, ts, sess, device))
            if rng.random() < 0.10 + 0.40 * aff_n[k]:
                ts2 = ts + pd.Timedelta(seconds=int(rng.integers(20, 900)))
                recs.append((u + 1, 1001 + it, "add_to_cart", np.nan, ts2, sess, device))
                if rng.random() < 0.25 + 0.45 * aff_n[k]:
                    ts3 = ts2 + pd.Timedelta(seconds=int(rng.integers(30, 1800)))
                    rating = np.nan
                    if rng.random() < 0.55:
                        rating = float(np.clip(np.round(1.5 + 3.5 * aff_n[k] + rng.normal(0, 0.7)), 1, 5))
                    recs.append((u + 1, 1001 + it, "purchase", rating, ts3, sess, device))
    df = pd.DataFrame(recs, columns=["user_id", "product_id", "event_type", "rating",
                                     "event_timestamp", "session_id", "device"])
    df = df[df.event_timestamp < END].sort_values("event_timestamp").reset_index(drop=True)
    df.insert(0, "event_id", [f"e{n:07d}" for n in range(1, len(df) + 1)])
    df["event_timestamp"] = df["event_timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["user_id"] = df["user_id"].astype("Int64")

    # ---- injected clickstream defects
    n = len(df)
    pick = lambda k: rng.choice(n, k, replace=False)
    df.loc[pick(int(n * 0.002)), "user_id"] = pd.NA                     # missing key
    k = int(n * 0.002)
    df.loc[pick(k), "product_id"] = rng.integers(9001, 9020, k)  # orphan product
    pur = df.index[df.event_type == "purchase"]
    df.loc[rng.choice(pur, 40, replace=False), "rating"] = rng.choice([0.0, 6.0, 9.0], 40)
    k = int(n * 0.004)
    df.loc[pick(k), "event_type"] = rng.choice(["VIEW", "Purchase ", "add-to-cart"], k)
    df.loc[pick(int(n * 0.0005)), "event_type"] = "unknown_evt"        # invalid domain
    df.loc[pick(int(n * 0.0005)), "event_timestamp"] = "not-a-timestamp"
    dups = df.iloc[pick(int(n * 0.012))]
    df = pd.concat([df, dups], ignore_index=True)                       # duplicate events
    return df


def main(end_date: str | None) -> None:
    rng = np.random.default_rng(SEED)
    item_cat, item_vec, item_bias, user_vec = build_world(rng)
    products = make_products(rng, item_cat)
    users = make_users(rng)
    events = make_events(rng, item_cat, item_vec, item_bias, user_vec)

    if end_date:  # deterministic truncation for the v1 snapshot
        # cut on the log position (event_id) so malformed-timestamp rows are kept only if
        # they were already written to the log at extraction time
        ts = pd.to_datetime(events.event_timestamp, errors="coerce")
        cutoff_id = events.loc[ts < pd.Timestamp(end_date), "event_id"].max()
        events = events[events.event_id <= cutoff_id]

    write_csv(events, SOURCES / "clickstream" / "interactions_export.csv")
    write_csv(users, SOURCES / "crm" / "users_export.csv")
    (SOURCES / "catalog").mkdir(parents=True, exist_ok=True)
    with open(SOURCES / "catalog" / "products.json", "w", encoding="utf-8") as f:
        json.dump({"_watermark": WATERMARK, "products": products}, f, indent=1)
    log.info("Generated sources for %s: events=%d users=%d products=%d (end_date=%s)",
             ROLL_NO, len(events), len(users), len(products), end_date or END.date())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-date", default=None, help="truncate events before this date (YYYY-MM-DD)")
    main(ap.parse_args().end_date)
