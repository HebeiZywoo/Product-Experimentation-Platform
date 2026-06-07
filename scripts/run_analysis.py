from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product_experiment_ds.causal import did_estimate, estimate_propensity_scores, psm_att, train_uplift_t_learner
from src.product_experiment_ds.experiments import experiment_quality_checks, experiment_summary, sample_size_for_lift
from src.product_experiment_ds.features import build_user_features, segment_users
from src.product_experiment_ds.marketplace import save_marketplace_layer
from src.product_experiment_ds.modeling import train_conversion_models


RAW_DIR = ROOT / "data" / "raw"
OLIST_DIR = ROOT / "data" / "olist"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"


def read_raw() -> dict[str, pd.DataFrame]:
    return {
        "users": pd.read_csv(RAW_DIR / "users.csv"),
        "assignments": pd.read_csv(RAW_DIR / "experiment_assignments.csv"),
        "campaign": pd.read_csv(RAW_DIR / "campaign_exposures.csv"),
        "activity": pd.read_csv(RAW_DIR / "daily_activity.csv"),
        "transactions": pd.read_csv(RAW_DIR / "transactions.csv"),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw()
    features = build_user_features(**raw)
    features = segment_users(features)

    exp_summary, segment_lift = experiment_summary(features)
    base_rate = float(exp_summary[exp_summary["metric"] == "Post-period conversion"]["control_rate"].iloc[0])
    sample_size = sample_size_for_lift(base_rate=base_rate, min_detectable_lift=0.03)
    psm_summary, psm_matches = psm_att(features)
    did_summary = did_estimate(features)
    propensity_scored, _ = estimate_propensity_scores(features)
    uplift_scores, uplift_deciles = train_uplift_t_learner(features)
    uplift_scores = uplift_scores.merge(propensity_scored[["user_id", "propensity_score"]], on="user_id", how="left")
    quality_checks = experiment_quality_checks(features, exp_summary)
    model_metrics, feature_importance, model_explanations, calibration, threshold_analysis = train_conversion_models(
        features, MODEL_DIR
    )

    features.to_csv(PROCESSED_DIR / "user_features.csv", index=False)
    exp_summary.to_csv(PROCESSED_DIR / "experiment_summary.csv", index=False)
    segment_lift.to_csv(PROCESSED_DIR / "segment_lift.csv", index=False)
    psm_summary.to_csv(PROCESSED_DIR / "psm_summary.csv", index=False)
    psm_matches.to_csv(PROCESSED_DIR / "psm_matches.csv", index=False)
    did_summary.to_csv(PROCESSED_DIR / "did_summary.csv", index=False)
    uplift_scores.to_csv(PROCESSED_DIR / "uplift_scores.csv", index=False)
    uplift_deciles.to_csv(PROCESSED_DIR / "uplift_deciles.csv", index=False)
    model_metrics.to_csv(PROCESSED_DIR / "model_metrics.csv", index=False)
    feature_importance.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)
    model_explanations.to_csv(PROCESSED_DIR / "model_explanations.csv", index=False)
    calibration.to_csv(PROCESSED_DIR / "calibration_curve.csv", index=False)
    threshold_analysis.to_csv(PROCESSED_DIR / "threshold_analysis.csv", index=False)
    quality_checks.to_csv(PROCESSED_DIR / "experiment_quality_checks.csv", index=False)
    save_marketplace_layer(PROCESSED_DIR, OLIST_DIR, features)
    (PROCESSED_DIR / "power_analysis.json").write_text(json.dumps({"users_per_group_for_3pp_lift": sample_size}, indent=2))

    print("Analysis complete")
    print(exp_summary.to_string(index=False))
    print(psm_summary.to_string(index=False))
    print(did_summary.to_string(index=False))
    print(model_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
