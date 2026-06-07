# Case Study: Product Experimentation & Causal Impact Platform

## Business Question

A product team launched an AI onboarding recommendation feature. The team needs to know:

- Did the feature improve conversion and retention?
- Which users benefited most?
- Can the next campaign be targeted more efficiently?
- How should non-randomized campaign results be interpreted?
- Are there marketplace quality or delivery problems that could make the rollout risky?

## Approach

I built a realistic product data science workflow with seven layers:

1. Data generation and feature engineering for user behavior, experiments, campaigns, and transactions.
2. DuckDB SQL reporting tables for funnel, segment, channel, campaign, and uplift analysis.
3. Randomized experiment analysis with lift, confidence intervals, p-values, and power analysis.
4. Causal inference for observational campaign analysis using propensity score matching and difference-in-differences.
5. Predictive and uplift modeling to prioritize users for the next rollout.
6. Olist-compatible marketplace analysis for order quality, reviews, delivery delay, and revenue.
7. Root-cause and post-launch monitoring outputs to connect model recommendations with operational readiness.

## Results

The randomized experiment showed a positive launch signal:

- Conversion lift: 11.8 percentage points.
- Retention lift: 13.2 percentage points.
- Average post-period revenue lift: $22.71.

The observational discount campaign also looked positive after adjustment:

- PSM conversion ATT: 12.1 percentage points.
- DiD revenue impact: $19.10 per exposed user.

Because the campaign was not randomized, I treat the observational result as directional. The randomized feature test is the cleaner decision input.

The marketplace layer adds a second question: even if the feature improves conversion, where could customer experience break? The root-cause view identifies payment types, categories, and states with elevated bad-review or late-delivery rates. The monitoring view turns those risks into thresholds that can be watched after rollout.

## Recommendation

Scale the AI onboarding recommendation feature to high-uplift users first while preserving a holdout group. Use the uplift decile table to prioritize users, but exclude or review segments with elevated bad-review and late-delivery risk before full rollout. Monitor conversion, retention, revenue, review quality, delivery quality, and feature drift after launch.

## What Makes This a Data Scientist Project

This project goes beyond a standard prediction model. It connects product decision-making with:

- Experimental design.
- Causal inference.
- SQL analytics.
- ML model comparison.
- Uplift targeting.
- Root-cause analysis.
- Launch monitoring.
- Deployment and communication.

The key lesson is that predictive models answer "who is likely to convert," while experiments and causal methods answer "what caused conversion to improve."
