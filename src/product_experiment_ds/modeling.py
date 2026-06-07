from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = [
    "engagement_score",
    "price_sensitivity",
    "tenure_days",
    "pre_active_days",
    "pre_sessions",
    "pre_page_views",
    "pre_cart_adds",
    "pre_purchases",
    "pre_revenue",
    "pre_conversion_rate",
    "pre_session_rate",
    "high_value_user",
]


FEATURE_LABELS = {
    "engagement_score": "Engagement score",
    "price_sensitivity": "Price sensitivity",
    "tenure_days": "Tenure",
    "pre_active_days": "Pre-period active days",
    "pre_sessions": "Pre-period sessions",
    "pre_page_views": "Pre-period page views",
    "pre_cart_adds": "Pre-period cart adds",
    "pre_purchases": "Pre-period purchases",
    "pre_revenue": "Pre-period revenue",
    "pre_conversion_rate": "Pre-period conversion rate",
    "pre_session_rate": "Pre-period session rate",
    "high_value_user": "High-value user flag",
}


def feature_story(feature: str, direction: str) -> str:
    label = FEATURE_LABELS.get(feature, feature)
    if direction == "positive":
        return f"Higher {label.lower()} is associated with higher predicted conversion."
    if direction == "negative":
        return f"Higher {label.lower()} is associated with lower predicted conversion."
    return f"{label} changes model ranking, but direction is model-dependent."


def build_threshold_analysis(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    rows = []
    value_per_conversion = 35
    contact_cost = 18
    for threshold in np.arange(0.10, 0.91, 0.05):
        selected = probabilities >= threshold
        selected_users = int(selected.sum())
        if selected_users == 0:
            precision = 0.0
            recall = 0.0
            true_positives = 0
        else:
            true_positives = int(y_true[selected].sum())
            precision = true_positives / selected_users
            recall = true_positives / int(y_true.sum()) if y_true.sum() else 0.0
        net_value = true_positives * value_per_conversion - selected_users * contact_cost
        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "selected_users": selected_users,
                "precision": precision,
                "recall": recall,
                "true_positives": true_positives,
                "expected_net_value": net_value,
            }
        )
    return pd.DataFrame(rows).sort_values("expected_net_value", ascending=False)


def build_calibration_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    return pd.DataFrame(
        {
            "bin": range(1, len(observed) + 1),
            "mean_predicted_probability": predicted,
            "observed_conversion_rate": observed,
            "calibration_error": observed - predicted,
        }
    )


def train_conversion_models(
    features: pd.DataFrame, model_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_dir.mkdir(parents=True, exist_ok=True)
    X = features[MODEL_FEATURES]
    y = features["converted_post"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.28, random_state=18, stratify=y)

    candidates = {
        "Logistic Regression": Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]
        ),
        "Random Forest": RandomForestClassifier(n_estimators=180, max_depth=8, random_state=31),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=3, random_state=42),
    }
    rows = []
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y_test, prob),
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best_name = metrics.iloc[0]["model"]
    best_model = fitted[best_name]
    best_prob = best_model.predict_proba(X_test)[:, 1]
    joblib.dump(best_model, model_dir / "conversion_model.joblib")
    (model_dir / "model_features.json").write_text(json.dumps(MODEL_FEATURES, indent=2))

    feature_importance = pd.DataFrame({"feature": MODEL_FEATURES})
    signed_effect = pd.Series(0.0, index=MODEL_FEATURES)
    if hasattr(best_model, "feature_importances_"):
        feature_importance["importance"] = best_model.feature_importances_
    elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps["model"], "coef_"):
        coefficients = best_model.named_steps["model"].coef_[0]
        feature_importance["importance"] = abs(coefficients)
        signed_effect = pd.Series(coefficients, index=MODEL_FEATURES)
    else:
        feature_importance["importance"] = 0
    feature_importance = feature_importance.sort_values("importance", ascending=False)

    permutation = permutation_importance(
        best_model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=44,
    )
    model_explanations = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "feature_label": [FEATURE_LABELS.get(feature, feature) for feature in MODEL_FEATURES],
            "permutation_importance": permutation.importances_mean,
            "model_importance": feature_importance.set_index("feature").reindex(MODEL_FEATURES)["importance"].fillna(0).to_numpy(),
            "signed_effect": signed_effect.reindex(MODEL_FEATURES).fillna(0).to_numpy(),
        }
    )
    model_explanations["direction"] = np.where(
        model_explanations["signed_effect"] > 0,
        "positive",
        np.where(model_explanations["signed_effect"] < 0, "negative", "model-dependent"),
    )
    model_explanations["interpretation"] = [
        feature_story(feature, direction)
        for feature, direction in zip(model_explanations["feature"], model_explanations["direction"])
    ]
    model_explanations = model_explanations.sort_values("permutation_importance", ascending=False)

    calibration = build_calibration_table(y_test, best_prob)
    threshold_analysis = build_threshold_analysis(y_test.reset_index(drop=True), best_prob)
    return metrics, feature_importance, model_explanations, calibration, threshold_analysis
