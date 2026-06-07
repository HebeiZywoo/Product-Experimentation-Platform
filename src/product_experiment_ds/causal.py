from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "engagement_score",
    "price_sensitivity",
    "tenure_days",
    "pre_active_days",
    "pre_sessions",
    "pre_cart_adds",
    "pre_purchases",
    "pre_revenue",
]


def estimate_propensity_scores(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(max_iter=1000)),
        ]
    )
    X = features[FEATURE_COLUMNS]
    y = features["exposed"].astype(int)
    model.fit(X, y)
    scored = features.copy()
    scored["propensity_score"] = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, scored["propensity_score"])
    return scored, {"propensity_auc": float(auc), "treated_share": float(y.mean())}


def psm_att(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored, diagnostics = estimate_propensity_scores(features)
    treated = scored[scored["exposed"] == 1].copy()
    control = scored[scored["exposed"] == 0].copy()

    matcher = NearestNeighbors(n_neighbors=1, metric="euclidean")
    matcher.fit(control[["propensity_score"]])
    distances, indices = matcher.kneighbors(treated[["propensity_score"]])
    matched_control = control.iloc[indices.flatten()].copy().reset_index(drop=True)
    matched_treated = treated.reset_index(drop=True)

    matched = pd.DataFrame(
        {
            "treated_user_id": matched_treated["user_id"],
            "control_user_id": matched_control["user_id"],
            "treated_outcome": matched_treated["converted_post"],
            "control_outcome": matched_control["converted_post"],
            "treated_revenue_delta": matched_treated["revenue_delta"],
            "control_revenue_delta": matched_control["revenue_delta"],
            "propensity_distance": distances.flatten(),
        }
    )
    att_conversion = matched["treated_outcome"].mean() - matched["control_outcome"].mean()
    att_revenue = matched["treated_revenue_delta"].mean() - matched["control_revenue_delta"].mean()
    diagnostics.update(
        {
            "matched_pairs": int(len(matched)),
            "mean_propensity_distance": float(matched["propensity_distance"].mean()),
            "att_conversion": float(att_conversion),
            "att_revenue_delta": float(att_revenue),
        }
    )
    return pd.DataFrame([diagnostics]), matched


def did_estimate(features: pd.DataFrame) -> pd.DataFrame:
    panel = []
    for row in features.itertuples(index=False):
        panel.append(
            {
                "user_id": row.user_id,
                "period": "pre",
                "post": 0,
                "exposed": int(row.exposed),
                "revenue": row.pre_revenue,
                "converted": int(row.pre_purchases > 0),
            }
        )
        panel.append(
            {
                "user_id": row.user_id,
                "period": "post",
                "post": 1,
                "exposed": int(row.exposed),
                "revenue": row.post_revenue,
                "converted": int(row.converted_post),
            }
        )
    panel_df = pd.DataFrame(panel)
    model = smf.ols("revenue ~ exposed + post + exposed:post", data=panel_df).fit(cov_type="HC1")
    term = "exposed:post"
    return pd.DataFrame(
        [
            {
                "estimand": "Difference-in-differences revenue impact",
                "estimate": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(model.conf_int().loc[term, 0]),
                "ci_high": float(model.conf_int().loc[term, 1]),
            }
        ]
    )


def train_uplift_t_learner(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_features = FEATURE_COLUMNS + ["pre_conversion_rate", "pre_session_rate", "high_value_user"]
    treated = features[features["treatment"] == 1]
    control = features[features["treatment"] == 0]

    treatment_model = RandomForestClassifier(n_estimators=160, max_depth=7, random_state=17)
    control_model = RandomForestClassifier(n_estimators=160, max_depth=7, random_state=23)
    treatment_model.fit(treated[model_features], treated["converted_post"])
    control_model.fit(control[model_features], control["converted_post"])

    scored = features.copy()
    scored["p_if_treated"] = treatment_model.predict_proba(scored[model_features])[:, 1]
    scored["p_if_control"] = control_model.predict_proba(scored[model_features])[:, 1]
    scored["predicted_uplift"] = scored["p_if_treated"] - scored["p_if_control"]
    scored["uplift_decile"] = pd.qcut(
        scored["predicted_uplift"].rank(method="first"), 10, labels=False
    )
    scored["uplift_decile"] = 10 - scored["uplift_decile"]

    deciles = (
        scored.groupby("uplift_decile")
        .agg(
            users=("user_id", "count"),
            avg_predicted_uplift=("predicted_uplift", "mean"),
            avg_p_if_treated=("p_if_treated", "mean"),
            avg_p_if_control=("p_if_control", "mean"),
            observed_treatment_rate=("treatment", "mean"),
            observed_conversion=("converted_post", "mean"),
            avg_post_revenue=("post_revenue", "mean"),
        )
        .reset_index()
        .sort_values("uplift_decile")
    )
    return scored, deciles
