-- =====================================================================
-- Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
-- =====================================================================
-- Gold layer: star schema in SQLite (data/warehouse/recsys_warehouse.db)
-- Grain of fact_interactions: one row per user event.
-- Grain of agg_user_item:    one row per (user, product) as of a snapshot.

-- Drop dependants first so re-runs are idempotent with foreign_keys=ON
DROP TABLE IF EXISTS fact_interactions;
DROP TABLE IF EXISTS feat_user;
DROP TABLE IF EXISTS feat_item;
DROP TABLE IF EXISTS agg_user_item;
DROP TABLE IF EXISTS feat_user_category;

DROP TABLE IF EXISTS dim_user;
CREATE TABLE dim_user (
    user_id          INTEGER PRIMARY KEY,
    signup_date      TEXT    NOT NULL,
    age              INTEGER NOT NULL CHECK (age BETWEEN 13 AND 100),
    age_imputed      INTEGER NOT NULL,
    gender           TEXT    NOT NULL CHECK (gender IN ('F','M','O','U')),
    city             TEXT    NOT NULL,
    is_prime_member  INTEGER NOT NULL
);

DROP TABLE IF EXISTS dim_product;
CREATE TABLE dim_product (
    product_id     INTEGER PRIMARY KEY,
    title          TEXT NOT NULL,
    category       TEXT NOT NULL,
    brand          TEXT,
    price          REAL NOT NULL CHECK (price > 0),
    price_imputed  INTEGER NOT NULL,
    currency       TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    in_stock       INTEGER NOT NULL
);

DROP TABLE IF EXISTS dim_date;
CREATE TABLE dim_date (
    date_key     TEXT PRIMARY KEY,      -- YYYY-MM-DD
    year         INTEGER, month INTEGER, day INTEGER,
    day_of_week  INTEGER, is_weekend INTEGER, week_of_year INTEGER
);

CREATE TABLE fact_interactions (
    event_id         TEXT PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES dim_user(user_id),
    product_id       INTEGER NOT NULL REFERENCES dim_product(product_id),
    date_key         TEXT    NOT NULL REFERENCES dim_date(date_key),
    event_type       TEXT    NOT NULL CHECK (event_type IN ('view','add_to_cart','purchase')),
    event_weight     REAL    NOT NULL,
    rating           REAL    CHECK (rating IS NULL OR rating BETWEEN 1 AND 5),
    event_timestamp  TEXT    NOT NULL,
    session_id       TEXT,
    device           TEXT
);
CREATE INDEX ix_fact_user ON fact_interactions(user_id, event_timestamp);
CREATE INDEX ix_fact_item ON fact_interactions(product_id, event_timestamp);
CREATE INDEX ix_fact_ts   ON fact_interactions(event_timestamp);

-- Feature snapshot tables (feat_user, feat_item, agg_user_item, feat_user_category)
-- are created by sql/features.sql, keyed by as_of (point-in-time).
