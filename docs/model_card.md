# Model Card

## Models

This project trains two types of models:

1. Conversion prediction models.
2. Uplift targeting models.

## Prediction Target

`converted_post`: whether a user made at least one purchase in the 42-day post-period.

## Features

Features include engagement score, price sensitivity, tenure, pre-period active days, sessions, page views, cart adds, purchases, revenue, and engineered rates.

## Candidate Models

- Logistic Regression.
- Random Forest.
- Gradient Boosting.

## Selected Model

The selected model is the candidate with the highest ROC-AUC on the holdout set. In the current run, Logistic Regression performs best with ROC-AUC around 0.660.

## Explainability

The dashboard includes a model explanation view with:

- Permutation importance on the holdout set.
- Model coefficient direction when available.
- Plain-English feature interpretations for the strongest drivers.

This avoids treating the model as a black box and helps connect model behavior to product reasoning.

## Calibration and Thresholding

The dashboard includes:

- A calibration table comparing predicted conversion probability with observed conversion rate.
- A threshold business-value table using conversion value and contact cost assumptions.
- A recommended operating threshold that maximizes expected net value.

In the current run, the recommended threshold is 0.50 under the configured business-value assumptions.

## Uplift Model

The uplift layer uses a T-learner:

- One Random Forest model estimates conversion probability if treated.
- One Random Forest model estimates conversion probability if control.
- Predicted uplift is `p_if_treated - p_if_control`.

## Intended Use

- Prioritize users for feature rollout or campaign targeting.
- Support product experimentation and business planning.
- Compare predictive risk with incremental treatment effect.

## Limitations

- Synthetic data is realistic but not production data.
- Uplift estimates depend on model quality and randomized assignment quality.
- Observational campaign analysis can still have unobserved confounding after matching.
- The model should not be used as the sole basis for business decisions without monitoring and validation.
