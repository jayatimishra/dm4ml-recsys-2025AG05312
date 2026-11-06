-- =====================================================================
-- Roll No: 2025AG05312 | AIML ZG529 Data Management for ML | Assignment
-- =====================================================================
-- Feature transformations (ELT: run inside the warehouse, dbt-style).
-- Every query is POINT-IN-TIME: only events with event_timestamp < :as_of
-- are used, so a snapshot taken at the training cutoff cannot see the
-- test window (no temporal leakage).
-- Each '-- name: X' block is a SELECT whose result is appended to table X
-- (rows for the same as_of are replaced first, so re-runs are idempotent).
-- Parameters: :as_of  (TEXT 'YYYY-MM-DD HH:MM:SS'),  :half_life_days (REAL)

-- name: feat_user
WITH ev AS (
    SELECT f.*, p.price, p.category FROM fact_interactions f
    JOIN dim_product p USING (product_id)
    WHERE f.event_timestamp < :as_of
),
fav AS (
    SELECT user_id, category FROM (
        SELECT user_id, category,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY SUM(event_weight) DESC, category) AS rk
        FROM ev GROUP BY user_id, category)
    WHERE rk = 1
)
SELECT :as_of                                                   AS as_of,
       u.user_id,
       COUNT(ev.event_id)                                       AS u_n_events,
       SUM(ev.event_type = 'view')                              AS u_n_views,
       SUM(ev.event_type = 'add_to_cart')                       AS u_n_carts,
       SUM(ev.event_type = 'purchase')                          AS u_n_purchases,
       ROUND(1.0 * SUM(ev.event_type = 'purchase') / NULLIF(SUM(ev.event_type = 'view'), 0), 4)
                                                                AS u_conversion_rate,
       ROUND(AVG(ev.rating), 3)                                 AS u_avg_rating_given,
       COUNT(DISTINCT ev.product_id)                            AS u_n_distinct_items,
       COUNT(DISTINCT ev.date_key)                              AS u_active_days,
       ROUND(julianday(:as_of) - julianday(MAX(ev.event_timestamp)), 2) AS u_days_since_last_event,
       ROUND(julianday(:as_of) - julianday(u.signup_date), 0)   AS u_tenure_days,
       ROUND(AVG(CASE WHEN ev.event_type = 'purchase' THEN ev.price END), 2) AS u_avg_purchase_price,
       fav.category                                             AS u_favourite_category,
       u.age, u.gender, u.city, u.is_prime_member
FROM dim_user u
LEFT JOIN ev  ON ev.user_id = u.user_id
LEFT JOIN fav ON fav.user_id = u.user_id
GROUP BY u.user_id;

-- name: feat_item
WITH ev AS (SELECT * FROM fact_interactions WHERE event_timestamp < :as_of)
SELECT :as_of                                                   AS as_of,
       p.product_id,
       COUNT(ev.event_id)                                       AS i_n_events,
       SUM(ev.event_type = 'view')                              AS i_n_views,
       SUM(ev.event_type = 'add_to_cart')                       AS i_n_carts,
       SUM(ev.event_type = 'purchase')                          AS i_n_purchases,
       ROUND(1.0 * SUM(ev.event_type = 'purchase') / NULLIF(SUM(ev.event_type = 'view'), 0), 4)
                                                                AS i_conversion_rate,
       ROUND(AVG(ev.rating), 3)                                 AS i_avg_rating,
       COUNT(ev.rating)                                         AS i_n_ratings,
       COUNT(DISTINCT ev.user_id)                               AS i_n_unique_users,
       SUM(ev.event_timestamp >= datetime(:as_of, '-7 days'))   AS i_events_last_7d,
       ROUND(julianday(:as_of) - julianday(MAX(ev.event_timestamp)), 2) AS i_days_since_last_event,
       ROUND(julianday(:as_of) - julianday(p.created_at), 0)    AS i_age_days,
       p.price, p.category, p.brand, p.in_stock
FROM dim_product p
LEFT JOIN ev ON ev.product_id = p.product_id
GROUP BY p.product_id;

-- name: agg_user_item
-- Implicit-feedback strength with exponential time decay:
--   strength = SUM( event_weight * 0.5 ^ (age_days / half_life) ), + bonus for explicit rating
SELECT :as_of AS as_of, user_id, product_id,
       COUNT(*)                                                  AS ui_n_events,
       MAX(event_type = 'purchase')                              AS ui_purchased,
       MAX(rating)                                               AS ui_max_rating,
       ROUND(SUM(event_weight * power(0.5, (julianday(:as_of) - julianday(event_timestamp)) / :half_life_days))
             + COALESCE(MAX(rating) - 3, 0), 4)                  AS ui_strength,
       MAX(event_timestamp)                                      AS ui_last_event_ts
FROM fact_interactions
WHERE event_timestamp < :as_of
GROUP BY user_id, product_id;

-- name: feat_user_category
-- Normalised share of a user's weighted engagement per category (content signal for the hybrid model)
WITH w AS (
    SELECT f.user_id, p.category, SUM(f.event_weight) AS w
    FROM fact_interactions f JOIN dim_product p USING (product_id)
    WHERE f.event_timestamp < :as_of
    GROUP BY f.user_id, p.category
)
SELECT :as_of AS as_of, user_id, category,
       ROUND(w / SUM(w) OVER (PARTITION BY user_id), 5) AS uc_affinity
FROM w;
