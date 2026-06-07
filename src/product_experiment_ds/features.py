from __future__ import annotations

import pandas as pd


def build_user_features(
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    campaign: pd.DataFrame,
    activity: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    pre = activity[activity["period"] == "pre"]
    post = activity[activity["period"] == "post"]

    pre_features = (
        pre.groupby("user_id")
        .agg(
            pre_active_days=("active", "sum"),
            pre_sessions=("sessions", "sum"),
            pre_page_views=("page_views", "sum"),
            pre_cart_adds=("add_to_cart", "sum"),
            pre_purchases=("purchase", "sum"),
        )
        .reset_index()
    )
    post_outcomes = (
        post.groupby("user_id")
        .agg(
            post_active_days=("active", "sum"),
            post_sessions=("sessions", "sum"),
            post_page_views=("page_views", "sum"),
            post_cart_adds=("add_to_cart", "sum"),
            post_purchases=("purchase", "sum"),
        )
        .reset_index()
    )
    tx_pre = transactions[transactions["order_date"] < "2026-03-01"]
    tx_post = transactions[transactions["order_date"] >= "2026-03-01"]

    pre_revenue = (
        tx_pre.groupby("user_id")
        .agg(pre_revenue=("revenue", "sum"), pre_margin=("gross_margin", "sum"))
        .reset_index()
    )
    post_revenue = (
        tx_post.groupby("user_id")
        .agg(post_revenue=("revenue", "sum"), post_margin=("gross_margin", "sum"))
        .reset_index()
    )

    features = users.merge(assignments[["user_id", "variant"]], on="user_id", how="left")
    features = features.merge(campaign[["user_id", "exposed"]], on="user_id", how="left")
    for frame in [pre_features, post_outcomes, pre_revenue, post_revenue]:
        features = features.merge(frame, on="user_id", how="left")
    features = features.fillna(0)

    features["treatment"] = (features["variant"] == "Treatment").astype(int)
    features["converted_post"] = (features["post_purchases"] > 0).astype(int)
    features["retained_post"] = (features["post_active_days"] >= 14).astype(int)
    features["pre_conversion_rate"] = features["pre_purchases"] / 56
    features["pre_session_rate"] = features["pre_sessions"] / 56
    features["pre_cart_rate"] = features["pre_cart_adds"] / 56
    features["post_conversion_rate"] = features["post_purchases"] / 42
    features["post_session_rate"] = features["post_sessions"] / 42
    features["revenue_delta"] = features["post_revenue"] - features["pre_revenue"] * (42 / 56)
    features["high_value_user"] = (
        (features["pre_revenue"] > features["pre_revenue"].quantile(0.70))
        | (features["pre_sessions"] > features["pre_sessions"].quantile(0.70))
    ).astype(int)
    return features


def segment_users(features: pd.DataFrame) -> pd.DataFrame:
    segmented = features.copy()
    conditions = [
        (segmented["pre_sessions"] >= segmented["pre_sessions"].quantile(0.75))
        & (segmented["pre_revenue"] >= segmented["pre_revenue"].quantile(0.65)),
        (segmented["pre_sessions"] <= segmented["pre_sessions"].quantile(0.30))
        & (segmented["pre_revenue"] <= segmented["pre_revenue"].quantile(0.35)),
        (segmented["price_sensitivity"] >= segmented["price_sensitivity"].quantile(0.70)),
        (segmented["engagement_score"] >= segmented["engagement_score"].quantile(0.70)),
    ]
    labels = ["Power customers", "Dormant low spend", "Price sensitive", "Engaged browsers"]
    segmented["segment"] = "Core users"
    for condition, label in zip(conditions, labels):
        segmented.loc[condition, "segment"] = label
    return segmented
