# =====================================================================
# Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
# =====================================================================
"""Stage 3: DATA PREPARATION (cleaning, standardisation, EDA).

Repairs the WARNING-level defects flagged by Stage 2 (critical rows were
already quarantined) and writes conformed, typed data to data/processed/.
Every repair is counted in reports/preparation_log.json so the cleaning is
auditable. EDA plots and summary statistics go to reports/eda/.

Repairs
  users        invalid/missing age -> median (flag age_imputed); invalid gender -> 'U';
               missing city -> 'Unknown'
  products     price text "INR 123" -> 123.0; price <= 0 or missing -> category median
               (flag price_imputed); casing drift in category -> controlled vocabulary;
               missing category -> inferred from title; price outliers winsorised at p99.5
  interactions event_type normalised (VIEW/Purchase /add-to-cart -> canonical);
               out-of-range ratings -> NaN; rating on non-purchase -> NaN; typed columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import (EVENT_WEIGHTS, PROCESSED, REPORTS, SERIES, VALIDATED, get_logger,
                    new_fig, read_parquet, save_fig, write_json, write_parquet)
from src.validation.validate import CATEGORIES, GENDERS, norm_event_type, to_price

log = get_logger("preparation")
EDA = REPORTS / "eda"


def prep_users(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df.user_id.astype(int)
    df["signup_date"] = pd.to_datetime(df.signup_date)
    age = pd.to_numeric(df.age, errors="coerce")
    bad = age.isna() | ~age.between(13, 100)
    med = float(age[~bad].median())
    df["age_imputed"] = bad
    df["age"] = age.where(~bad, med).astype(int)
    fixes["users.age_imputed_with_median"] = int(bad.sum())
    g_bad = ~df.gender.isin(GENDERS)
    df.loc[g_bad, "gender"] = "U"
    fixes["users.gender_set_unknown"] = int(g_bad.sum())
    fixes["users.city_filled_unknown"] = int(df.city.isna().sum())
    df["city"] = df.city.fillna("Unknown")
    df["is_prime_member"] = df.is_prime_member.astype(str).str.lower().eq("true")
    return df[["user_id", "signup_date", "age", "age_imputed", "gender", "city", "is_prime_member"]]


def prep_products(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    df = df.copy()
    df["product_id"] = df.product_id.astype(int)
    vocab = {c.lower(): c for c in CATEGORIES}
    cat = df.category.astype("string")
    casing = cat.notna() & ~cat.isin(CATEGORIES) & cat.str.lower().isin(vocab)
    fixes["products.category_casing_fixed"] = int(casing.sum())
    cat = cat.str.lower().map(vocab)
    from_title = df.title.str.extract(r"^(.*) item \d+$")[0].str.lower().map(vocab)
    miss = cat.isna()
    fixes["products.category_inferred_from_title"] = int((miss & from_title.notna()).sum())
    df["category"] = cat.fillna(from_title).fillna("Unknown")

    raw_price = df.price.astype("string")
    texty = raw_price.str.contains(r"[A-Za-z]", na=False)
    fixes["products.price_text_parsed"] = int(texty.sum())
    price = to_price(df.price)
    bad = price.isna() | (price <= 0)
    fixes["products.price_imputed_category_median"] = int(bad.sum())
    med = price[~bad].groupby(df.category[~bad]).median()
    df["price_imputed"] = bad
    df["price"] = price.where(~bad, df.category.map(med))
    cap = df.price.quantile(0.995)
    fixes["products.price_winsorised_p99_5"] = int((df.price > cap).sum())
    df["price"] = df.price.clip(upper=cap).round(2)
    df["created_at"] = pd.to_datetime(df.created_at, utc=True).dt.tz_localize(None)
    df["in_stock"] = df.in_stock.astype(str).str.lower().eq("true")
    return df[["product_id", "title", "category", "brand", "price", "price_imputed",
               "currency", "created_at", "in_stock"]]


def prep_interactions(df: pd.DataFrame, fixes: dict) -> pd.DataFrame:
    df = df.copy()
    et = norm_event_type(df.event_type)
    fixes["interactions.event_type_normalised"] = int((et != df.event_type).sum())
    df["event_type"] = et
    df["user_id"] = pd.to_numeric(df.user_id).astype(int)
    df["product_id"] = pd.to_numeric(df.product_id).astype(int)
    df["event_timestamp"] = pd.to_datetime(df.event_timestamp, format="%Y-%m-%d %H:%M:%S")
    r = pd.to_numeric(df.rating, errors="coerce")
    bad = r.notna() & (~r.between(1, 5) | (df.event_type != "purchase"))
    fixes["interactions.rating_invalid_set_null"] = int(bad.sum())
    df["rating"] = r.where(~bad)
    df["event_weight"] = df.event_type.map(EVENT_WEIGHTS)
    df["event_date"] = df.event_timestamp.dt.normalize()
    return (df[["event_id", "user_id", "product_id", "event_type", "event_weight", "rating",
                "event_timestamp", "event_date", "session_id", "device"]]
            .sort_values("event_timestamp").reset_index(drop=True))


# ------------------------------------------------------------------ EDA
def eda(users, products, inter) -> dict:
    types = ["view", "add_to_cart", "purchase"]
    # 1. daily events by type
    fig, ax = new_fig()
    daily = inter.groupby(["event_date", "event_type"]).size().unstack(fill_value=0)
    for i, t in enumerate(types):
        ax.plot(daily.index, daily[t], color=SERIES[i], lw=2, label=t)
        ax.annotate(t, (daily.index[-1], daily[t].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color="#52514e", fontsize=8, va="center")
    ax.set_title("Daily interactions by event type")
    ax.set_ylabel("events per day")
    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, EDA / "01_daily_events.png")

    # 2. conversion funnel
    fig, ax = new_fig(6, 3.6)
    counts = inter.event_type.value_counts().reindex(types)
    ax.bar(types, counts.values, color=SERIES[0], width=0.55)
    for x, v in enumerate(counts.values):
        ax.text(x, v, f"{v:,}\n({v / counts.iloc[0]:.1%} of views)", ha="center", va="bottom",
                fontsize=8, color="#52514e")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_title("Engagement funnel")
    ax.set_ylabel("events")
    save_fig(fig, EDA / "02_funnel.png")

    # 3. long-tail item popularity
    fig, ax = new_fig()
    pop = inter.groupby("product_id").size().sort_values(ascending=False).values
    ax.loglog(np.arange(1, len(pop) + 1), pop, color=SERIES[0], lw=2)
    top20 = pop[: int(len(pop) * 0.2)].sum() / pop.sum()
    ax.set_title(f"Item popularity is long-tailed: top 20% of items get {top20:.0%} of interactions")
    ax.set_xlabel("item rank (log)")
    ax.set_ylabel("interactions (log)")
    save_fig(fig, EDA / "03_item_long_tail.png")

    # 4. user activity
    fig, ax = new_fig()
    act = inter.groupby("user_id").size()
    ax.hist(act, bins=np.logspace(np.log10(act.min()), np.log10(act.max()), 30),
            color=SERIES[0], edgecolor="#fcfcfb", linewidth=1)
    ax.set_xscale("log")
    ax.set_title(f"Interactions per user (median {act.median():.0f})")
    ax.set_xlabel("interactions per user (log)")
    ax.set_ylabel("users")
    save_fig(fig, EDA / "04_user_activity.png")

    # 5. rating distribution
    fig, ax = new_fig(6, 3.6)
    rc = inter.rating.dropna().astype(int).value_counts().sort_index()
    ax.bar(rc.index.astype(str), rc.values, color=SERIES[0], width=0.55)
    ax.set_title(f"Explicit ratings (only {inter.rating.notna().mean():.1%} of events carry one)")
    ax.set_xlabel("rating")
    ax.set_ylabel("count")
    save_fig(fig, EDA / "05_ratings.png")

    # 6. category share of purchases vs views
    fig, ax = new_fig(7.5, 4.4)
    m = inter.merge(products[["product_id", "category"]], on="product_id")
    share = (m[m.event_type.isin(["view", "purchase"])].groupby(["category", "event_type"]).size()
             .unstack().fillna(0))
    share = (share / share.sum()).sort_values("view")
    y = np.arange(len(share))
    ax.barh(y - 0.2, share["view"], 0.38, color=SERIES[0], label="views")
    ax.barh(y + 0.2, share["purchase"], 0.38, color=SERIES[1], label="purchases")
    ax.set_yticks(y, share.index)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e4e3df")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("Category share of views vs purchases")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, EDA / "06_category_share.png")

    # 7. hour-of-day pattern
    fig, ax = new_fig(7.5, 3.4)
    hr = inter.event_timestamp.dt.hour.value_counts().sort_index()
    ax.bar(hr.index, hr.values, color=SERIES[0], width=0.8)
    ax.set_title("Interactions by hour of day")
    ax.set_xlabel("hour")
    ax.set_ylabel("events")
    save_fig(fig, EDA / "07_hour_of_day.png")

    n_u, n_i = inter.user_id.nunique(), inter.product_id.nunique()
    pairs = inter[["user_id", "product_id"]].drop_duplicates().shape[0]
    summary = {
        "n_events": int(len(inter)), "n_users_active": int(n_u), "n_items_interacted": int(n_i),
        "n_users_total": int(len(users)), "n_items_total": int(len(products)),
        "user_item_pairs": int(pairs), "matrix_sparsity": round(1 - pairs / (n_u * n_i), 5),
        "date_range": [str(inter.event_timestamp.min()), str(inter.event_timestamp.max())],
        "event_mix": inter.event_type.value_counts().to_dict(),
        "view_to_cart": round(counts["add_to_cart"] / counts["view"], 4),
        "cart_to_purchase": round(counts["purchase"] / counts["add_to_cart"], 4),
        "rating_coverage": round(float(inter.rating.notna().mean()), 4),
        "mean_rating": round(float(inter.rating.mean()), 3),
        "top20pct_items_share": round(float(top20), 4),
        "median_events_per_user": float(act.median()),
        "price_by_category_median": products.groupby("category").price.median().round(0).to_dict(),
    }
    return summary


def main() -> dict:
    fixes: dict = {}
    users = prep_users(read_parquet(VALIDATED / "users.parquet"), fixes)
    products = prep_products(read_parquet(VALIDATED / "products.parquet"), fixes)
    inter = prep_interactions(read_parquet(VALIDATED / "interactions.parquet"), fixes)
    for name, df in [("users", users), ("products", products), ("interactions", inter)]:
        write_parquet(df, PROCESSED / "clean" / f"{name}.parquet")
    write_json(fixes, REPORTS / "preparation_log.json")
    summary = eda(users, products, inter)
    write_json(summary, EDA / "eda_summary.json")
    log.info("Prepared users=%d products=%d interactions=%d; repairs=%s", len(users),
             len(products), len(inter), fixes)
    return summary


if __name__ == "__main__":
    main()
