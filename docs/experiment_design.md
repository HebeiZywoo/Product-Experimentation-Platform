# Experiment Design Notes

## Experiment

Feature: AI onboarding recommendations  
Primary metric: post-period conversion  
Guardrail metric: post-period retention  
Secondary metric: post-period revenue  
Unit of randomization: user  
Split: 50/50 treatment vs. control  
Post-period window: 42 days  

## Hypothesis

Users who receive AI onboarding recommendations will be more likely to convert and remain active during the post-period than users in the control group.

## Metrics

- Conversion: user has at least one post-period purchase.
- Retention: user has at least 14 active days in the post-period.
- Revenue: total post-period transaction revenue.
- Lift: treatment metric minus control metric.

## Statistical Tests

- Two-proportion z-test for conversion and retention.
- Welch t-test for revenue.
- 95% confidence interval for absolute lift.
- Power analysis for a 3 percentage-point minimum detectable effect.

## Guardrails and Caveats

- Segment-level lift is directional unless each segment has sufficient sample size.
- Revenue lift can be skewed by high-spend users, so conversion and retention remain primary decision metrics.
- The randomized experiment is stronger evidence than observational campaign analysis.
- Future versions should add novelty checks, pre-registered stopping rules, and ramp-up monitoring.

## Quality Checks Added

The dashboard includes an experiment QA layer before the causal interpretation:

- Sample ratio mismatch check using a chi-square test against the intended 50/50 split.
- Pre-period balance checks using standardized mean differences for engagement, price sensitivity, tenure, sessions, revenue, and purchases.
- Guardrail check to ensure retention does not decline after treatment.

In the current run, all 8 checks pass. The maximum pre-period standardized mean difference is below 0.10, which supports interpreting the randomized lift as credible.
