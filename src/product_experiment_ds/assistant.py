from __future__ import annotations

import pandas as pd


def build_context(
    experiment_summary: pd.DataFrame,
    segment_lift: pd.DataFrame,
    psm_summary: pd.DataFrame,
    did_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> dict[str, str]:
    conversion = experiment_summary[experiment_summary["metric"] == "Post-period conversion"].iloc[0]
    top_segment = segment_lift.sort_values("absolute_lift", ascending=False).iloc[0]
    psm = psm_summary.iloc[0]
    did = did_summary.iloc[0]
    best_model = model_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    return {
        "conversion_lift": f"{conversion['absolute_lift']:.2%}",
        "conversion_p": f"{conversion['p_value']:.4f}",
        "top_segment": str(top_segment["segment"]),
        "top_segment_lift": f"{top_segment['absolute_lift']:.2%}",
        "psm_att": f"{psm['att_conversion']:.2%}",
        "did_revenue": f"${did['estimate']:.2f}",
        "best_model": str(best_model["model"]),
        "best_auc": f"{best_model['roc_auc']:.3f}",
    }


def answer_question(question: str, context: dict[str, str]) -> str:
    q = question.lower()
    if "segment" in q or "who" in q or "target" in q:
        return (
            f"The strongest observed segment is {context['top_segment']}, with an experiment lift of "
            f"{context['top_segment_lift']}. I would target this group first, then keep a holdout group "
            "so the next launch still has a clean read on incremental impact."
        )
    if "causal" in q or "psm" in q or "did" in q or "bias" in q:
        return (
            f"The randomized experiment estimates {context['conversion_lift']} lift. For the non-random "
            f"discount campaign, propensity score matching estimates {context['psm_att']} conversion ATT, "
            f"while difference-in-differences estimates {context['did_revenue']} revenue impact. I would "
            "treat the observational result as directional because unobserved confounding can remain."
        )
    if "model" in q or "auc" in q or "predict" in q:
        return (
            f"The best conversion model is {context['best_model']} with ROC-AUC {context['best_auc']}. "
            "The model is useful for prioritization, but the causal and experiment layers should drive "
            "the launch decision because predictive lift is not the same as treatment effect."
        )
    if "recommend" in q or "decision" in q:
        return (
            f"My recommendation is to scale the feature to high-uplift users first. The A/B test shows "
            f"{context['conversion_lift']} conversion lift (p={context['conversion_p']}), and the uplift "
            "model helps avoid spending incentives on users who would convert anyway."
        )
    return (
        "This project combines product analytics, randomized testing, causal inference, uplift modeling, "
        "and a deployed Streamlit dashboard. Ask about segments, causal validity, model performance, or "
        "the launch recommendation for a more specific answer."
    )

