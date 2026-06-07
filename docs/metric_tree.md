# Metric Tree and Product Framing

## Product Scenario

The project is framed as a marketplace growth decision: should the team scale an AI onboarding recommendation or seller/customer intervention, and which users or marketplace segments should receive it first?

The dashboard can run with:

- Olist CSV files placed under `data/olist/`.
- A built-in Olist-compatible fallback generated from the product experiment data.

## Metric Tree

| Level | Metric | Why it matters |
|---|---|---|
| North Star | Marketplace repeat purchase readiness | Captures whether the marketplace experience is likely to bring users back. |
| Primary | Conversion / repeat purchase | Measures whether the intervention improves customer behavior. |
| Guardrail | Late delivery rate | Prevents growth tactics from hiding operational quality problems. |
| Guardrail | Bad review rate | Captures customer experience deterioration. |
| Diagnostic | Delivery days | Helps explain review and retention changes. |
| Diagnostic | Revenue per order | Translates product changes into business impact. |

## Decision Rule

Scale the intervention only if:

- Randomized lift is positive and statistically credible.
- Experiment QA checks pass.
- Guardrail metrics do not deteriorate.
- Root-cause analysis does not reveal a major operational bottleneck in the target segment.
- Monitoring thresholds are ready before full rollout.

