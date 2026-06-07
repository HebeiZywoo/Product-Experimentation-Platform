from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def proportion_test(control: pd.Series, treatment: pd.Series) -> dict[str, float]:
    n_control = len(control)
    n_treatment = len(treatment)
    p_control = control.mean()
    p_treatment = treatment.mean()
    lift = p_treatment - p_control
    pooled = (control.sum() + treatment.sum()) / (n_control + n_treatment)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_control + 1 / n_treatment))
    z = lift / se if se else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    ci_se = math.sqrt(
        p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment
    )
    return {
        "control_rate": float(p_control),
        "treatment_rate": float(p_treatment),
        "absolute_lift": float(lift),
        "relative_lift": float(lift / p_control) if p_control else 0.0,
        "ci_low": float(lift - 1.96 * ci_se),
        "ci_high": float(lift + 1.96 * ci_se),
        "z_score": float(z),
        "p_value": float(p_value),
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
    }


def continuous_ttest(control: pd.Series, treatment: pd.Series, metric_name: str) -> dict[str, float | str]:
    result = stats.ttest_ind(treatment, control, equal_var=False)
    return {
        "metric": metric_name,
        "control_mean": float(control.mean()),
        "treatment_mean": float(treatment.mean()),
        "absolute_lift": float(treatment.mean() - control.mean()),
        "p_value": float(result.pvalue),
    }


def sample_size_for_lift(base_rate: float, min_detectable_lift: float, alpha: float = 0.05, power: float = 0.80) -> int:
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p1 = base_rate
    p2 = base_rate + min_detectable_lift
    pooled = (p1 + p2) / 2
    numerator = z_alpha * math.sqrt(2 * pooled * (1 - pooled)) + z_beta * math.sqrt(
        p1 * (1 - p1) + p2 * (1 - p2)
    )
    return int(math.ceil((numerator / min_detectable_lift) ** 2))


def experiment_summary(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = features[features["variant"] == "Control"]
    treatment = features[features["variant"] == "Treatment"]
    conversion = proportion_test(control["converted_post"], treatment["converted_post"])
    retention = proportion_test(control["retained_post"], treatment["retained_post"])
    revenue = continuous_ttest(control["post_revenue"], treatment["post_revenue"], "post_revenue")

    conversion["metric"] = "Post-period conversion"
    retention["metric"] = "Post-period retention"
    rows = [conversion, retention, revenue]
    summary = pd.DataFrame(rows)

    segment_rows = []
    for segment, segment_df in features.groupby("segment"):
        if segment_df["variant"].nunique() < 2:
            continue
        segment_control = segment_df[segment_df["variant"] == "Control"]
        segment_treatment = segment_df[segment_df["variant"] == "Treatment"]
        test = proportion_test(segment_control["converted_post"], segment_treatment["converted_post"])
        test["segment"] = segment
        segment_rows.append(test)
    segment_lift = pd.DataFrame(segment_rows).sort_values("absolute_lift", ascending=False)
    return summary, segment_lift


def standardized_mean_difference(control: pd.Series, treatment: pd.Series) -> float:
    pooled = math.sqrt((control.var(ddof=1) + treatment.var(ddof=1)) / 2)
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((treatment.mean() - control.mean()) / pooled)


def experiment_quality_checks(features: pd.DataFrame, exp_summary: pd.DataFrame) -> pd.DataFrame:
    control = features[features["variant"] == "Control"]
    treatment = features[features["variant"] == "Treatment"]
    total = len(features)
    expected = total / 2
    srm_stat = ((len(control) - expected) ** 2 / expected) + ((len(treatment) - expected) ** 2 / expected)
    srm_p = 1 - stats.chi2.cdf(srm_stat, df=1)

    rows = [
        {
            "check": "Sample ratio mismatch",
            "metric": "assignment balance",
            "value": srm_p,
            "threshold": 0.05,
            "status": "Pass" if srm_p >= 0.05 else "Review",
            "interpretation": "Variant counts are consistent with the intended 50/50 split.",
        }
    ]
    balance_features = [
        "engagement_score",
        "price_sensitivity",
        "tenure_days",
        "pre_sessions",
        "pre_revenue",
        "pre_purchases",
    ]
    for feature in balance_features:
        smd = standardized_mean_difference(control[feature], treatment[feature])
        rows.append(
            {
                "check": "Pre-period balance",
                "metric": feature,
                "value": abs(smd),
                "threshold": 0.10,
                "status": "Pass" if abs(smd) < 0.10 else "Review",
                "interpretation": "Absolute standardized mean difference below 0.10 indicates good balance.",
            }
        )

    retention = exp_summary[exp_summary["metric"] == "Post-period retention"].iloc[0]
    rows.append(
        {
            "check": "Guardrail metric",
            "metric": "retention lift",
            "value": retention["absolute_lift"],
            "threshold": 0,
            "status": "Pass" if retention["absolute_lift"] >= 0 else "Review",
            "interpretation": "Retention did not decline after treatment.",
        }
    )
    return pd.DataFrame(rows)
