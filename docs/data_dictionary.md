# Data Dictionary

## Raw Tables

### users.csv

| Column | Description |
|---|---|
| user_id | Unique user identifier. |
| signup_date | User signup date. |
| acquisition_channel | Acquisition source such as Organic, Paid Search, Social, Referral, or Email. |
| region | User region. |
| device | Primary device. |
| persona | Behavioral persona label. |
| engagement_score | Latent engagement score between 0 and 1. |
| price_sensitivity | Latent price sensitivity score between 0 and 1. |
| tenure_days | Days between signup and experiment start. |

### experiment_assignments.csv

| Column | Description |
|---|---|
| user_id | User identifier. |
| experiment_name | Name of randomized experiment. |
| variant | Control or Treatment. |
| assigned_at | Experiment assignment timestamp. |

### campaign_exposures.csv

| Column | Description |
|---|---|
| user_id | User identifier. |
| campaign_name | Observational campaign name. |
| exposed | Whether the user received the non-randomized campaign. |
| propensity_true | Simulated true exposure propensity. |
| exposed_at | Campaign exposure timestamp. |

### daily_activity.csv

| Column | Description |
|---|---|
| user_id | User identifier. |
| event_date | Activity date. |
| period | Pre or post experiment period. |
| active | Whether user was active that day. |
| sessions | Number of sessions. |
| page_views | Number of page views. |
| add_to_cart | Whether user added to cart. |
| purchase | Whether user purchased. |

### transactions.csv

| Column | Description |
|---|---|
| transaction_id | Transaction identifier. |
| user_id | User identifier. |
| order_date | Transaction date. |
| revenue | Transaction revenue. |
| gross_margin | Estimated gross margin. |

## Optional Olist Input Tables

If real Olist marketplace data is placed under `data/olist/`, the marketplace layer uses those files instead of the built-in fallback.

| File | Description |
|---|---|
| olist_orders_dataset.csv | Order status and purchase, delivery, and estimated delivery timestamps. |
| olist_order_items_dataset.csv | Order-item revenue, freight, seller, and product identifiers. |
| olist_order_reviews_dataset.csv | Customer review scores and review timestamps. |
| olist_customers_dataset.csv | Customer location and unique customer identifiers. |
| olist_products_dataset.csv | Product category metadata. |
| olist_order_payments_dataset.csv | Payment type and payment value. |

## Processed Outputs

| File | Description |
|---|---|
| user_features.csv | User-level features and outcomes. |
| experiment_summary.csv | Global A/B test results. |
| segment_lift.csv | Segment-level experiment lift. |
| psm_summary.csv | Propensity score matching diagnostics and ATT. |
| did_summary.csv | Difference-in-differences estimate. |
| uplift_scores.csv | User-level predicted uplift scores. |
| uplift_deciles.csv | Uplift decile summary table. |
| model_metrics.csv | Predictive model comparison. |
| feature_importance.csv | Feature importance for selected model. |
| model_explanations.csv | Directional model explanation table using permutation importance and feature summaries. |
| calibration_curve.csv | Predicted probability calibration bins. |
| threshold_analysis.csv | Business-value analysis across model score thresholds. |
| experiment_quality_checks.csv | SRM, pre-period balance, and guardrail checks for experiment trustworthiness. |
| marketplace_orders.csv | Olist-compatible marketplace order layer for revenue, delivery, reviews, and product/category diagnostics. |
| metric_tree.csv | North-star, primary, guardrail, and diagnostic metrics used in the product decision frame. |
| root_cause_drivers.csv | Driver-level bad-review and late-delivery diagnostics by payment type, product category, and customer state. |
| monitoring_metrics.csv | Post-launch monitoring metrics, deltas, thresholds, and alert status. |
| monitoring_drift.csv | Feature drift checks using population stability index. |
| marketplace_source.json | Metadata describing whether real Olist data or fallback marketplace data was used. |
| sql_*.csv | DuckDB-generated reporting tables. |
