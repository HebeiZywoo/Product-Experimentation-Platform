CREATE OR REPLACE TABLE user_funnel AS
SELECT
    variant,
    COUNT(*) AS users,
    AVG(pre_sessions) AS avg_pre_sessions,
    AVG(post_sessions) AS avg_post_sessions,
    AVG(converted_post) AS post_conversion_rate,
    AVG(retained_post) AS post_retention_rate,
    AVG(post_revenue) AS avg_post_revenue
FROM user_features
GROUP BY variant
ORDER BY variant;

CREATE OR REPLACE TABLE segment_experiment AS
SELECT
    segment,
    variant,
    COUNT(*) AS users,
    AVG(converted_post) AS conversion_rate,
    AVG(retained_post) AS retention_rate,
    AVG(post_revenue) AS avg_revenue,
    AVG(post_sessions) AS avg_sessions
FROM user_features
GROUP BY segment, variant
ORDER BY segment, variant;

CREATE OR REPLACE TABLE channel_cohorts AS
SELECT
    acquisition_channel,
    variant,
    COUNT(*) AS users,
    AVG(pre_session_rate) AS avg_pre_session_rate,
    AVG(post_session_rate) AS avg_post_session_rate,
    AVG(converted_post) AS conversion_rate,
    AVG(post_revenue) AS avg_post_revenue
FROM user_features
GROUP BY acquisition_channel, variant
ORDER BY acquisition_channel, variant;

CREATE OR REPLACE TABLE campaign_observational_summary AS
SELECT
    exposed,
    COUNT(*) AS users,
    AVG(propensity_score) AS avg_propensity_score,
    AVG(pre_revenue) AS avg_pre_revenue,
    AVG(post_revenue) AS avg_post_revenue,
    AVG(revenue_delta) AS avg_revenue_delta,
    AVG(converted_post) AS conversion_rate
FROM uplift_scores
GROUP BY exposed
ORDER BY exposed;

CREATE OR REPLACE TABLE uplift_priority AS
SELECT
    uplift_decile,
    COUNT(*) AS users,
    AVG(predicted_uplift) AS avg_predicted_uplift,
    AVG(p_if_treated) AS avg_p_if_treated,
    AVG(p_if_control) AS avg_p_if_control,
    AVG(post_revenue) AS avg_post_revenue
FROM uplift_scores
GROUP BY uplift_decile
ORDER BY uplift_decile;

