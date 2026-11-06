<!-- Roll No: 2025AG05312 -->
# Feature catalog

Generated from the feature registry (`data/features/registry.db`). All features are computed point-in-time in the warehouse (`src/transformation/sql/features.sql`) and served from the offline store for training and the online store for prediction.

## user_profile_behaviour (v1)

User demographics and engagement behaviour aggregated up to as_of. Entity key: `user_id`. Source table: `feat_user`. TTL: 30 days. Owner: ml-platform (2025AG05312). Definition hash: `ff4af9216c56`.

| Feature | Type | Description | Used by |
|---|---|---|---|
| `u_n_events` | int | Total events by the user | serving enrichment / analysis |
| `u_n_views` | int | Product views | serving enrichment / analysis |
| `u_n_carts` | int | Add-to-cart events | serving enrichment / analysis |
| `u_n_purchases` | int | Purchases | serving enrichment / analysis |
| `u_conversion_rate` | float | Purchases / views | serving enrichment / analysis |
| `u_avg_rating_given` | float | Mean explicit rating the user gave | serving enrichment / analysis |
| `u_n_distinct_items` | int | Distinct products interacted with | serving enrichment / analysis |
| `u_active_days` | int | Distinct active days | serving enrichment / analysis |
| `u_days_since_last_event` | float | Recency in days at as_of | serving enrichment / analysis |
| `u_tenure_days` | float | Days since signup | serving enrichment / analysis |
| `u_avg_purchase_price` | float | Mean price of purchased items (INR) | serving enrichment / analysis |
| `u_favourite_category` | str | Category with highest weighted engagement | serving context |
| `age` | int | Age (imputed when invalid) | serving enrichment / analysis |
| `gender` | str | F/M/O/U | serving enrichment / analysis |
| `city` | str | City | serving enrichment / analysis |
| `is_prime_member` | int | Prime membership flag | serving enrichment / analysis |

## item_popularity_quality (v1)

Product popularity, conversion and quality signals up to as_of. Entity key: `product_id`. Source table: `feat_item`. TTL: 7 days. Owner: ml-platform (2025AG05312). Definition hash: `a0875da21a2f`.

| Feature | Type | Description | Used by |
|---|---|---|---|
| `i_n_events` | int | Total events on the product | serving enrichment / analysis |
| `i_n_views` | int | Views | popularity baseline |
| `i_n_carts` | int | Add-to-cart events | popularity baseline |
| `i_n_purchases` | int | Purchases | popularity baseline |
| `i_conversion_rate` | float | Purchases / views | serving enrichment / analysis |
| `i_avg_rating` | float | Mean explicit rating | serving enrichment |
| `i_n_ratings` | int | Number of ratings | serving enrichment / analysis |
| `i_n_unique_users` | int | Distinct users who interacted | serving enrichment / analysis |
| `i_events_last_7d` | int | Trending signal: events in the 7 days before as_of | popularity baseline (trending) |
| `i_days_since_last_event` | float | Days since last interaction | serving enrichment / analysis |
| `i_age_days` | float | Days since product listing | serving enrichment / analysis |
| `price` | float | Price in INR (cleaned) | serving enrichment |
| `category` | str | Category | hybrid, diversity metric |
| `brand` | str | Brand | serving enrichment / analysis |
| `in_stock` | int | Stock flag | serving enrichment |

## user_item_interaction (v1)

Time-decayed implicit feedback strength per user-item pair. Entity key: `user_id, product_id`. Source table: `agg_user_item`. TTL: 30 days. Owner: ml-platform (2025AG05312). Definition hash: `0e29a99a4468`.

| Feature | Type | Description | Used by |
|---|---|---|---|
| `ui_n_events` | int | Events on this pair | serving enrichment / analysis |
| `ui_purchased` | int | 1 if purchased | serving enrichment / analysis |
| `ui_max_rating` | float | Highest explicit rating on the pair | serving enrichment / analysis |
| `ui_strength` | float | SUM(weight * 0.5^(age/30d)) + (rating - 3) | all CF models (implicit matrix) |
| `ui_last_event_ts` | str | Last event timestamp | serving enrichment / analysis |

## user_category_affinity (v1)

Share of the user's weighted engagement per category. Entity key: `user_id, category`. Source table: `feat_user_category`. TTL: 30 days. Owner: ml-platform (2025AG05312). Definition hash: `d18c1bbc6b51`.

| Feature | Type | Description | Used by |
|---|---|---|---|
| `uc_affinity` | float | Normalised category affinity in [0,1] | hybrid (content signal) |
