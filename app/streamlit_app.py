from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.product_experiment_ds.assistant import answer_question, build_context


PROCESSED_DIR = ROOT / "data" / "processed"
REQUIRED_OUTPUTS = [
    PROCESSED_DIR / "user_features.csv",
    PROCESSED_DIR / "experiment_summary.csv",
    PROCESSED_DIR / "segment_lift.csv",
    PROCESSED_DIR / "psm_summary.csv",
    PROCESSED_DIR / "did_summary.csv",
    PROCESSED_DIR / "uplift_deciles.csv",
    PROCESSED_DIR / "model_metrics.csv",
    PROCESSED_DIR / "experiment_quality_checks.csv",
    PROCESSED_DIR / "threshold_analysis.csv",
    PROCESSED_DIR / "marketplace_orders.csv",
    PROCESSED_DIR / "root_cause_drivers.csv",
    PROCESSED_DIR / "monitoring_metrics.csv",
    PROCESSED_DIR / "sql_user_funnel.csv",
]


st.set_page_config(
    page_title="AI Product Experimentation Platform",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def ensure_outputs() -> None:
    if all(path.exists() for path in REQUIRED_OUTPUTS):
        return
    subprocess.run([sys.executable, "scripts/run_pipeline.py"], cwd=ROOT, check=True)


@st.cache_data
def load_outputs() -> dict[str, pd.DataFrame | dict]:
    ensure_outputs()
    outputs: dict[str, pd.DataFrame | dict] = {
        "features": pd.read_csv(PROCESSED_DIR / "user_features.csv"),
        "experiment": pd.read_csv(PROCESSED_DIR / "experiment_summary.csv"),
        "segment_lift": pd.read_csv(PROCESSED_DIR / "segment_lift.csv"),
        "psm": pd.read_csv(PROCESSED_DIR / "psm_summary.csv"),
        "did": pd.read_csv(PROCESSED_DIR / "did_summary.csv"),
        "uplift_scores": pd.read_csv(PROCESSED_DIR / "uplift_scores.csv"),
        "uplift_deciles": pd.read_csv(PROCESSED_DIR / "uplift_deciles.csv"),
        "model_metrics": pd.read_csv(PROCESSED_DIR / "model_metrics.csv"),
        "feature_importance": pd.read_csv(PROCESSED_DIR / "feature_importance.csv"),
        "model_explanations": pd.read_csv(PROCESSED_DIR / "model_explanations.csv"),
        "calibration": pd.read_csv(PROCESSED_DIR / "calibration_curve.csv"),
        "threshold_analysis": pd.read_csv(PROCESSED_DIR / "threshold_analysis.csv"),
        "quality_checks": pd.read_csv(PROCESSED_DIR / "experiment_quality_checks.csv"),
        "marketplace_orders": pd.read_csv(PROCESSED_DIR / "marketplace_orders.csv"),
        "root_cause_drivers": pd.read_csv(PROCESSED_DIR / "root_cause_drivers.csv"),
        "metric_tree": pd.read_csv(PROCESSED_DIR / "metric_tree.csv"),
        "monitoring_metrics": pd.read_csv(PROCESSED_DIR / "monitoring_metrics.csv"),
        "monitoring_drift": pd.read_csv(PROCESSED_DIR / "monitoring_drift.csv"),
        "marketplace_source": json.loads((PROCESSED_DIR / "marketplace_source.json").read_text()),
        "power": json.loads((PROCESSED_DIR / "power_analysis.json").read_text()),
    }
    for path in PROCESSED_DIR.glob("sql_*.csv"):
        outputs[path.stem] = pd.read_csv(path)
    return outputs


def format_pct(value: float) -> str:
    return f"{value:.1%}"


def format_money(value: float) -> str:
    return f"${value:,.0f}"


def format_signed_money(value: float) -> str:
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):,.0f}"


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def format_pp(value: float) -> str:
    return f"{value * 100:.1f}pp"


def metric_card(label: str, value: str, detail: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_panel(title: str, body: str, chips: list[str]) -> None:
    chip_html = "".join([f"<span>{chip}</span>" for chip in chips])
    st.markdown(
        f"""
        <div class="decision-panel">
          <div class="decision-kicker">Recommendation</div>
          <div class="decision-title">{title}</div>
          <div class="decision-body">{body}</div>
          <div class="decision-chips">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_table(
    df: pd.DataFrame,
    *,
    percent_cols: list[str] | None = None,
    money_cols: list[str] | None = None,
    decimal_cols: list[str] | None = None,
) -> None:
    percent_cols = percent_cols or []
    money_cols = money_cols or []
    decimal_cols = decimal_cols or []
    display = df.copy()
    config = {}
    for col in percent_cols:
        if col in display.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.1f%%")
            display[col] = display[col] * 100
    for col in money_cols:
        if col in display.columns:
            config[col] = st.column_config.NumberColumn(col, format="$%.2f")
    for col in decimal_cols:
        if col in display.columns:
            config[col] = st.column_config.NumberColumn(col, format="%.3f")
    st.dataframe(display, width="stretch", hide_index=True, column_config=config)


def build_report_markdown(
    conversion: pd.Series,
    retention: pd.Series,
    revenue: pd.Series,
    top_segment: pd.Series,
    top_decile: pd.Series,
    psm_row: pd.Series,
    did_row: pd.Series,
    best_model: pd.Series,
    quality_checks: pd.DataFrame,
    best_threshold: pd.Series,
) -> str:
    failed_checks = quality_checks[quality_checks["status"] != "Pass"]
    check_summary = "All quality checks passed." if failed_checks.empty else f"{len(failed_checks)} checks require review."
    return f"""# Executive Experiment Readout

## Recommendation

Scale the AI onboarding recommendation feature to high-uplift users first, while preserving a holdout group.

## Randomized Experiment

- Conversion lift: {format_pp(conversion["absolute_lift"])}
- Retention lift: {format_pp(retention["absolute_lift"])}
- Revenue lift: {format_signed_money(revenue["absolute_lift"])}
- Conversion p-value: {format_p_value(conversion["p_value"])}

## Targeting

- Strongest segment: {top_segment["segment"]} ({format_pp(top_segment["absolute_lift"])} observed lift)
- Highest uplift decile: {int(top_decile["uplift_decile"])} ({format_pp(top_decile["avg_predicted_uplift"])} predicted uplift)
- Recommended operating threshold: {best_threshold["threshold"]:.2f}
- Expected net value at threshold: {format_signed_money(best_threshold["expected_net_value"])}

## Causal Checks

- PSM conversion ATT: {format_pp(psm_row["att_conversion"])}
- DiD revenue estimate: {format_signed_money(did_row["estimate"])}
- Experiment QA: {check_summary}

## Model

- Best model: {best_model["model"]}
- ROC-AUC: {best_model["roc_auc"]:.3f}

## Caveats

- Segment-level lift is directional unless segment sample sizes are large enough.
- Observational campaign estimates can still contain unobserved confounding.
- Predictive conversion probability is not the same as incremental treatment effect.
"""


st.markdown(
    """
    <style>
      .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1240px;
      }
      [data-testid="stMetric"] {
        border: 1px solid #d8dee8;
        background: #f8fafc;
        padding: 12px 14px;
        border-radius: 8px;
      }
      div[data-testid="stTabs"] button {
        font-size: 0.92rem;
      }
      .app-header {
        border-bottom: 1px solid #d8dee8;
        padding-bottom: 14px;
        margin-bottom: 14px;
      }
      .eyebrow {
        color: #52616f;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0;
        font-weight: 700;
        margin-bottom: 4px;
      }
      .app-title {
        color: #111827;
        font-size: 2.05rem;
        line-height: 1.15;
        font-weight: 800;
        margin: 0;
      }
      .app-subtitle {
        color: #475569;
        font-size: 0.98rem;
        line-height: 1.45;
        margin-top: 8px;
        max-width: 920px;
      }
      .decision-panel {
        border: 1px solid #cbd5e1;
        background: #f8fafc;
        border-left: 5px solid #2563eb;
        border-radius: 8px;
        padding: 16px 18px;
        margin: 8px 0 16px 0;
      }
      .decision-kicker {
        color: #2563eb;
        text-transform: uppercase;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 4px;
      }
      .decision-title {
        color: #111827;
        font-size: 1.22rem;
        font-weight: 800;
        line-height: 1.25;
      }
      .decision-body {
        color: #334155;
        margin-top: 7px;
        line-height: 1.45;
      }
      .decision-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 12px;
      }
      .decision-chips span {
        border: 1px solid #cbd5e1;
        background: #ffffff;
        color: #1f2937;
        padding: 4px 9px;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 650;
      }
      .metric-card {
        border: 1px solid #d8dee8;
        background: #ffffff;
        border-radius: 8px;
        padding: 13px 14px;
        min-height: 104px;
      }
      .metric-card.positive {
        border-top: 4px solid #16a34a;
      }
      .metric-card.caution {
        border-top: 4px solid #f59e0b;
      }
      .metric-card.neutral {
        border-top: 4px solid #2563eb;
      }
      .metric-label {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
      }
      .metric-value {
        color: #0f172a;
        font-size: 1.55rem;
        font-weight: 850;
        line-height: 1.15;
        margin-top: 4px;
      }
      .metric-detail {
        color: #475569;
        font-size: 0.84rem;
        line-height: 1.35;
        margin-top: 6px;
      }
      .method-chip {
        border: 1px solid #d8dee8;
        border-radius: 8px;
        padding: 8px 10px;
        background: #ffffff;
        color: #334155;
        font-size: 0.86rem;
        margin-bottom: 8px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.spinner("Preparing experiment, causal, uplift, and SQL outputs..."):
    outputs = load_outputs()

features = outputs["features"]
experiment = outputs["experiment"]
segment_lift = outputs["segment_lift"]
psm = outputs["psm"]
did = outputs["did"]
uplift_deciles = outputs["uplift_deciles"]
uplift_scores = outputs["uplift_scores"]
model_metrics = outputs["model_metrics"]
feature_importance = outputs["feature_importance"]
model_explanations = outputs["model_explanations"]
calibration = outputs["calibration"]
threshold_analysis = outputs["threshold_analysis"]
quality_checks = outputs["quality_checks"]
marketplace_orders = outputs["marketplace_orders"]
root_cause_drivers = outputs["root_cause_drivers"]
metric_tree = outputs["metric_tree"]
monitoring_metrics = outputs["monitoring_metrics"]
monitoring_drift = outputs["monitoring_drift"]
marketplace_source = outputs["marketplace_source"]
power = outputs["power"]

if "avg_p_if_treated" not in uplift_deciles.columns and "sql_uplift_priority" in outputs:
    uplift_deciles = outputs["sql_uplift_priority"]

conversion = experiment[experiment["metric"] == "Post-period conversion"].iloc[0]
retention = experiment[experiment["metric"] == "Post-period retention"].iloc[0]
revenue = experiment[experiment["metric"] == "post_revenue"].iloc[0]
best_model = model_metrics.sort_values("roc_auc", ascending=False).iloc[0]
top_decile = uplift_deciles.sort_values("avg_predicted_uplift", ascending=False).iloc[0]
top_segment = segment_lift.sort_values("absolute_lift", ascending=False).iloc[0]
psm_row = psm.iloc[0]
did_row = did.iloc[0]
best_threshold = threshold_analysis.sort_values("expected_net_value", ascending=False).iloc[0]

st.sidebar.header("Project Controls")
selected_segment = st.sidebar.selectbox("Segment focus", ["All segments"] + sorted(features["segment"].unique().tolist()))
min_lift = st.sidebar.slider("Minimum lift to flag", 0.0, 0.20, 0.05, 0.01, format="%.2f")
st.sidebar.divider()
st.sidebar.metric("Users", f"{len(features):,}")
st.sidebar.metric("Daily activity rows", "784,000")
st.sidebar.metric("Marketplace orders", f"{len(marketplace_orders):,}")
st.sidebar.metric("Experiment split", "50 / 50")
st.sidebar.divider()
st.sidebar.caption(f"Marketplace source: {marketplace_source['status']}")
st.sidebar.markdown('<div class="method-chip">Randomized A/B test</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="method-chip">Propensity score matching</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="method-chip">Difference-in-differences</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="method-chip">T-learner uplift model</div>', unsafe_allow_html=True)

if selected_segment == "All segments":
    focus_features = features
else:
    focus_features = features[features["segment"] == selected_segment]

st.markdown(
    """
    <div class="app-header">
      <div class="eyebrow">Product Data Science Portfolio</div>
      <h1 class="app-title">AI Product Experimentation & Causal Impact Platform</h1>
      <div class="app-subtitle">
        A decision workspace for evaluating feature launches, estimating causal impact, and targeting users by incremental lift.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Executive Brief",
        "Marketplace Case",
        "Experiment",
        "Quality Checks",
        "Causal Impact",
        "Root Cause",
        "Uplift Targeting",
        "Prediction Model",
        "Monitoring",
        "SQL Tables",
        "Analyst Copilot",
    ]
)

with tabs[0]:
    decision_panel(
        "Scale to high-uplift users first; keep a holdout group.",
        (
            f"The randomized experiment shows {format_pp(conversion['absolute_lift'])} conversion lift and "
            f"{format_pp(retention['absolute_lift'])} retention lift. The strongest segment is "
            f"{top_segment['segment']}, with {format_pp(top_segment['absolute_lift'])} observed lift."
        ),
        [
            f"p-value {format_p_value(conversion['p_value'])}",
            f"Top decile {format_pct(top_decile['avg_predicted_uplift'])} uplift",
            f"DiD {format_signed_money(did_row['estimate'])}",
            f"Model AUC {best_model['roc_auc']:.3f}",
        ],
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Conversion Lift", format_pp(conversion["absolute_lift"]), "Primary launch metric", "positive")
    with k2:
        metric_card("Retention Lift", format_pp(retention["absolute_lift"]), "Guardrail metric", "positive")
    with k3:
        metric_card("Top Uplift Group", f"Decile {int(top_decile['uplift_decile'])}", "Highest incremental response", "neutral")
    with k4:
        metric_card("Best Model", f"{best_model['roc_auc']:.3f}", f"{best_model['model']} ROC-AUC", "caution")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Experiment Readout")
        compact_table(
            experiment[
                ["metric", "control_rate", "treatment_rate", "absolute_lift", "p_value", "n_control", "n_treatment"]
            ],
            percent_cols=["control_rate", "treatment_rate", "absolute_lift"],
        )
    with right:
        st.markdown("#### Decision Inputs")
        business = pd.DataFrame(
            [
                ["Revenue lift", format_signed_money(revenue["absolute_lift"]), "Randomized test"],
                ["PSM conversion ATT", format_pp(psm_row["att_conversion"]), "Observational adjustment"],
                ["DiD revenue estimate", format_signed_money(did_row["estimate"]), "Pre/post causal check"],
                ["Users per group for 3pp MDE", f"{power['users_per_group_for_3pp_lift']:,}", "Power analysis"],
            ],
            columns=["Signal", "Value", "Method"],
        )
        compact_table(business)
        report = build_report_markdown(
            conversion,
            retention,
            revenue,
            top_segment,
            top_decile,
            psm_row,
            did_row,
            best_model,
            quality_checks,
            best_threshold,
        )
        st.download_button(
            "Download executive readout",
            data=report,
            file_name="experiment_executive_readout.md",
            mime="text/markdown",
            width="stretch",
        )

with tabs[1]:
    st.subheader("Marketplace Case Framing")
    source_status = marketplace_source["status"].replace("_", " ")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Data Source", source_status, marketplace_source["source"], "neutral")
    with c2:
        metric_card("Orders", f"{len(marketplace_orders):,}", "Marketplace order layer", "neutral")
    with c3:
        metric_card("Bad Review Rate", format_pct(marketplace_orders["bad_review"].mean()), "Customer experience guardrail", "caution")
    with c4:
        metric_card("Late Delivery Rate", format_pct(marketplace_orders["delivered_late"].mean()), "Operational guardrail", "caution")

    st.info(marketplace_source["note"])
    st.markdown("#### Metric Tree")
    compact_table(metric_tree, decimal_cols=["value"])
    st.markdown("#### Marketplace Sample")
    compact_table(
        marketplace_orders.head(12),
        money_cols=["revenue", "freight_value"],
        decimal_cols=["delivery_days", "estimated_delay_days", "review_score"],
    )

with tabs[2]:
    st.subheader("Randomized Experiment Analysis")
    st.caption(f"Current focus: {selected_segment} | Users in focus: {len(focus_features):,}")
    c1, c2 = st.columns([1, 1])
    with c1:
        funnel = outputs["sql_user_funnel"].copy()
        st.markdown("#### Variant Funnel")
        compact_table(
            funnel,
            percent_cols=["post_conversion_rate", "post_retention_rate"],
            money_cols=["avg_post_revenue"],
        )
        st.bar_chart(funnel.set_index("variant")[["post_conversion_rate", "post_retention_rate"]])
    with c2:
        st.markdown("#### Segment Lift")
        segment_view = segment_lift.copy()
        segment_view["flag"] = segment_view["absolute_lift"].ge(min_lift).map({True: "Above threshold", False: "Below threshold"})
        compact_table(
            segment_view[["segment", "absolute_lift", "ci_low", "ci_high", "p_value", "flag"]],
            percent_cols=["absolute_lift", "ci_low", "ci_high"],
        )
        segment_chart = segment_lift[["segment", "absolute_lift"]].set_index("segment").sort_values("absolute_lift")
        st.bar_chart(segment_chart)

    st.markdown("#### Power Analysis")
    st.info(
        f"To detect a 3 percentage-point lift at 80% power, this setup needs about "
        f"{power['users_per_group_for_3pp_lift']:,} users per group."
    )

with tabs[3]:
    st.subheader("Experiment Quality Checks")
    passed = int((quality_checks["status"] == "Pass").sum())
    total_checks = len(quality_checks)
    q1, q2, q3 = st.columns(3)
    with q1:
        metric_card("Checks Passed", f"{passed}/{total_checks}", "SRM, balance, and guardrails", "positive")
    with q2:
        srm = quality_checks[quality_checks["check"] == "Sample ratio mismatch"].iloc[0]
        metric_card("SRM p-value", format_p_value(srm["value"]), "Assignment-ratio sanity check", "neutral")
    with q3:
        max_smd = quality_checks[quality_checks["check"] == "Pre-period balance"]["value"].max()
        metric_card("Max Pre-period SMD", f"{max_smd:.3f}", "Target: below 0.100", "positive" if max_smd < 0.1 else "caution")

    compact_table(
        quality_checks,
        decimal_cols=["value", "threshold"],
    )
    st.caption(
        "These checks help catch common experiment issues before interpreting lift: sample-ratio mismatch, pre-period imbalance, and guardrail movement."
    )

with tabs[4]:
    st.subheader("Causal Impact Beyond the Randomized Test")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### Propensity Score Matching")
        metric_card("Matched ATT", format_pp(psm_row["att_conversion"]), f"{int(psm_row['matched_pairs']):,} matched pairs", "neutral")
        compact_table(
            psm,
            percent_cols=["treated_share", "att_conversion"],
            money_cols=["att_revenue_delta"],
            decimal_cols=["propensity_auc", "mean_propensity_distance"],
        )
    with right:
        st.markdown("#### Difference-in-Differences")
        metric_card("Revenue Impact", format_signed_money(did_row["estimate"]), f"95% CI {format_signed_money(did_row['ci_low'])} to {format_signed_money(did_row['ci_high'])}", "positive")
        compact_table(
            did,
            money_cols=["estimate", "std_error", "ci_low", "ci_high"],
        )

    obs = outputs["sql_campaign_observational_summary"]
    st.markdown("#### Before Adjustment")
    compact_table(
        obs,
        percent_cols=["avg_propensity_score", "conversion_rate"],
        money_cols=["avg_pre_revenue", "avg_post_revenue", "avg_revenue_delta"],
    )

with tabs[5]:
    st.subheader("Root Cause Analysis")
    top_driver = root_cause_drivers.iloc[0]
    r1, r2, r3 = st.columns(3)
    with r1:
        metric_card("Top Driver", str(top_driver["driver_value"]), str(top_driver["driver"]), "caution")
    with r2:
        metric_card("Bad Review Lift", format_pp(top_driver["bad_review_lift_vs_avg"]), "vs marketplace average", "caution")
    with r3:
        metric_card("Excess Bad Reviews", f"{top_driver['estimated_bad_review_excess']:.1f}", "Estimated contribution", "caution")

    st.markdown("#### Candidate Root Causes")
    compact_table(
        root_cause_drivers.head(15),
        percent_cols=["bad_review_rate", "late_delivery_rate", "bad_review_lift_vs_avg", "late_lift_vs_avg"],
        money_cols=["revenue"],
        decimal_cols=["avg_delivery_days", "estimated_bad_review_excess"],
    )
    driver_options = root_cause_drivers["driver"].drop_duplicates().tolist()
    selected_driver = st.selectbox("Drill down by driver", driver_options)
    driver_view = root_cause_drivers[root_cause_drivers["driver"] == selected_driver].head(10)
    st.bar_chart(driver_view.set_index("driver_value")["estimated_bad_review_excess"])
    st.caption("Root-cause candidates are ranked by excess bad reviews, combining segment size and bad-review lift.")

with tabs[6]:
    st.subheader("Uplift Targeting")
    selected_decile = st.slider("Inspect uplift decile", 1, 10, int(top_decile["uplift_decile"]))
    decile_row = uplift_deciles[uplift_deciles["uplift_decile"] == selected_decile].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Predicted Uplift", format_pp(decile_row["avg_predicted_uplift"]), f"Decile {selected_decile}", "positive")
    with k2:
        metric_card("Users", f"{int(decile_row['users']):,}", "Targetable population", "neutral")
    with k3:
        metric_card("If Treated", format_pct(decile_row["avg_p_if_treated"]), "Predicted conversion", "neutral")
    with k4:
        metric_card("If Control", format_pct(decile_row["avg_p_if_control"]), "Counterfactual conversion", "caution")

    decile_view = uplift_deciles.set_index("uplift_decile")[
        ["avg_predicted_uplift", "avg_p_if_treated", "avg_p_if_control"]
    ]
    st.line_chart(decile_view)

    st.markdown("#### Priority Table")
    compact_table(
        uplift_deciles,
        percent_cols=["avg_predicted_uplift", "avg_p_if_treated", "avg_p_if_control", "observed_treatment_rate", "observed_conversion"],
        money_cols=["avg_post_revenue"],
    )

with tabs[7]:
    st.subheader("Conversion Prediction Model")
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Best ROC-AUC", f"{best_model['roc_auc']:.3f}", best_model["model"], "neutral")
    with m2:
        metric_card("Best Threshold", f"{best_threshold['threshold']:.2f}", "Max expected net value", "positive")
    with m3:
        metric_card("Net Value", format_signed_money(best_threshold["expected_net_value"]), "Threshold analysis", "positive")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("#### Model Comparison")
        compact_table(
            model_metrics,
            percent_cols=["accuracy", "precision", "recall"],
            decimal_cols=["roc_auc"],
        )
    with c2:
        st.markdown("#### Model Explanation")
        st.bar_chart(model_explanations.head(10).set_index("feature_label")["permutation_importance"])

    st.markdown("#### Feature Explanations")
    compact_table(
        model_explanations[["feature_label", "permutation_importance", "direction", "interpretation"]].head(8),
        decimal_cols=["permutation_importance"],
    )
    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("#### Calibration")
        st.line_chart(calibration.set_index("mean_predicted_probability")["observed_conversion_rate"])
        compact_table(
            calibration,
            percent_cols=["mean_predicted_probability", "observed_conversion_rate", "calibration_error"],
        )
    with c4:
        st.markdown("#### Threshold Business Value")
        st.line_chart(threshold_analysis.sort_values("threshold").set_index("threshold")["expected_net_value"])
        compact_table(
            threshold_analysis.head(8),
            percent_cols=["precision", "recall"],
            money_cols=["expected_net_value"],
        )

    st.caption("Prediction helps prioritize users. Uplift and causal analysis determine whether the intervention creates incremental value.")

with tabs[8]:
    st.subheader("Post-launch Monitoring")
    alert_count = int((monitoring_metrics["status"] == "Alert").sum() + (monitoring_drift["status"] == "Alert").sum())
    watch_count = int((monitoring_drift["status"] == "Watch").sum())
    m1, m2, m3 = st.columns(3)
    with m1:
        metric_card("Alerts", str(alert_count), "Metrics and drift checks", "positive" if alert_count == 0 else "caution")
    with m2:
        metric_card("Watch Items", str(watch_count), "PSI between 0.10 and 0.20", "neutral")
    with m3:
        metric_card("Monitoring Plan", "Ready", "Pre/post marketplace health", "positive")

    st.markdown("#### Marketplace Health Metrics")
    compact_table(
        monitoring_metrics,
        decimal_cols=["baseline", "current", "delta", "threshold"],
    )
    st.markdown("#### Feature Drift")
    compact_table(monitoring_drift, decimal_cols=["psi"])
    st.caption("PSI below 0.10 is healthy, 0.10-0.20 is watch, and 0.20+ should trigger review.")

with tabs[9]:
    st.subheader("DuckDB SQL Analysis Layer")
    selected = st.selectbox(
        "Choose SQL output",
        [
            "sql_user_funnel",
            "sql_segment_experiment",
            "sql_channel_cohorts",
            "sql_campaign_observational_summary",
            "sql_uplift_priority",
        ],
    )
    compact_table(outputs[selected])
    with st.expander("View SQL"):
        st.code((ROOT / "sql" / "product_experiment_analysis.sql").read_text(), language="sql")

with tabs[10]:
    st.subheader("Grounded Analyst Copilot")
    context = build_context(experiment, segment_lift, psm, did, model_metrics)
    examples = [
        "What is the launch recommendation?",
        "Which segment should we target first?",
        "How do the causal estimates compare?",
        "What does the model tell us?",
    ]
    picked = st.selectbox("Prompt template", examples)
    question = st.text_input("Ask a question about the experiment", value=picked)
    st.markdown("#### Answer")
    st.write(answer_question(question, context))
