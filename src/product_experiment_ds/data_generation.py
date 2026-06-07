from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 7


@dataclass(frozen=True)
class DataConfig:
    n_users: int = 8000
    pre_days: int = 56
    post_days: int = 42
    experiment_start: str = "2026-03-01"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def make_users(rng: np.random.Generator, config: DataConfig) -> pd.DataFrame:
    channels = np.array(["Organic", "Paid Search", "Social", "Referral", "Email"])
    regions = np.array(["West", "Northeast", "South", "Midwest"])
    devices = np.array(["iOS", "Android", "Web"])
    personas = np.array(["Value seeker", "Power user", "New explorer", "At-risk", "Premium"])

    channel = rng.choice(channels, config.n_users, p=[0.30, 0.22, 0.18, 0.16, 0.14])
    region = rng.choice(regions, config.n_users, p=[0.32, 0.22, 0.29, 0.17])
    device = rng.choice(devices, config.n_users, p=[0.38, 0.34, 0.28])
    persona = rng.choice(personas, config.n_users, p=[0.26, 0.20, 0.24, 0.18, 0.12])

    channel_score = pd.Series(channel).map(
        {"Organic": 0.08, "Paid Search": -0.05, "Social": -0.02, "Referral": 0.16, "Email": 0.12}
    ).to_numpy()
    persona_score = pd.Series(persona).map(
        {"Value seeker": -0.02, "Power user": 0.25, "New explorer": 0.02, "At-risk": -0.22, "Premium": 0.30}
    ).to_numpy()

    signup_start = pd.Timestamp("2025-08-01").value // 10**9
    signup_end = pd.Timestamp("2026-02-15").value // 10**9
    signup_ts = rng.integers(signup_start, signup_end, config.n_users)

    engagement = np.clip(rng.beta(2.1, 2.7, config.n_users) + channel_score + persona_score, 0.02, 0.98)
    price_sensitivity = np.clip(rng.beta(2.7, 2.2, config.n_users) - persona_score * 0.3, 0.02, 0.98)
    tenure_days = (pd.Timestamp(config.experiment_start) - pd.to_datetime(signup_ts, unit="s")).days

    users = pd.DataFrame(
        {
            "user_id": [f"U{i:05d}" for i in range(1, config.n_users + 1)],
            "signup_date": pd.to_datetime(signup_ts, unit="s").normalize(),
            "acquisition_channel": channel,
            "region": region,
            "device": device,
            "persona": persona,
            "engagement_score": engagement.round(4),
            "price_sensitivity": price_sensitivity.round(4),
            "tenure_days": tenure_days,
        }
    )
    return users


def make_experiment_assignments(rng: np.random.Generator, users: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    treatment = rng.choice(["Control", "Treatment"], len(users), p=[0.50, 0.50])
    return pd.DataFrame(
        {
            "user_id": users["user_id"],
            "experiment_name": "AI onboarding recommendations",
            "variant": treatment,
            "assigned_at": pd.Timestamp(config.experiment_start),
        }
    )


def make_observational_campaign(rng: np.random.Generator, users: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    score = (
        -1.2
        + 2.3 * users["engagement_score"].to_numpy()
        + 1.1 * users["price_sensitivity"].to_numpy()
        + (users["persona"].eq("Value seeker").to_numpy() * 0.7)
        + (users["acquisition_channel"].eq("Email").to_numpy() * 0.55)
    )
    propensity = sigmoid(score)
    exposed = rng.binomial(1, propensity)
    return pd.DataFrame(
        {
            "user_id": users["user_id"],
            "campaign_name": "Personalized discount rescue",
            "exposed": exposed,
            "propensity_true": propensity.round(4),
            "exposed_at": pd.Timestamp(config.experiment_start) + pd.Timedelta(days=5),
        }
    )


def make_daily_activity(
    rng: np.random.Generator,
    users: pd.DataFrame,
    assignments: pd.DataFrame,
    campaign: pd.DataFrame,
    config: DataConfig,
) -> pd.DataFrame:
    start = pd.Timestamp(config.experiment_start)
    periods = [("pre", -config.pre_days, -1), ("post", 0, config.post_days - 1)]
    assignment_map = assignments.set_index("user_id")["variant"]
    exposure_map = campaign.set_index("user_id")["exposed"]

    rows = []
    for period, begin, end in periods:
        dates = pd.date_range(start + pd.Timedelta(days=begin), start + pd.Timedelta(days=end), freq="D")
        for user in users.itertuples(index=False):
            variant = assignment_map.loc[user.user_id]
            exposed = int(exposure_map.loc[user.user_id])
            base = -1.75 + 2.4 * user.engagement_score - 0.55 * user.price_sensitivity
            base += 0.20 if user.device == "iOS" else 0
            base += 0.18 if user.acquisition_channel == "Referral" else 0

            treatment_effect = 0.0
            if period == "post" and variant == "Treatment":
                treatment_effect += 0.16 + 0.42 * user.engagement_score
                treatment_effect += 0.20 if user.persona in ("New explorer", "Power user") else 0
            if period == "post" and exposed:
                treatment_effect += 0.08 + 0.36 * user.price_sensitivity

            active_prob = sigmoid(base + treatment_effect)
            daily_active = rng.binomial(1, np.clip(active_prob, 0.02, 0.90), len(dates))
            sessions = daily_active * (1 + rng.poisson(0.95 + 1.4 * user.engagement_score, len(dates)))
            page_views = sessions * rng.poisson(3.0 + 4.0 * user.engagement_score, len(dates))
            add_to_cart = rng.binomial(1, np.clip(0.06 + 0.18 * user.engagement_score + 0.035 * sessions, 0, 0.70))
            purchase_prob = np.clip(0.003 + 0.020 * add_to_cart + 0.003 * sessions + 0.012 * user.engagement_score, 0, 0.32)
            if period == "post" and variant == "Treatment":
                purchase_prob = np.clip(purchase_prob + 0.004 + 0.010 * user.engagement_score, 0, 0.36)
            if period == "post" and exposed:
                purchase_prob = np.clip(purchase_prob + 0.002 + 0.012 * user.price_sensitivity, 0, 0.36)
            purchases = rng.binomial(1, purchase_prob)

            for date, active, sess, views, cart, purchase in zip(
                dates, daily_active, sessions, page_views, add_to_cart, purchases
            ):
                rows.append(
                    {
                        "user_id": user.user_id,
                        "event_date": date,
                        "period": period,
                        "active": int(active),
                        "sessions": int(sess),
                        "page_views": int(views),
                        "add_to_cart": int(cart),
                        "purchase": int(purchase),
                    }
                )
    return pd.DataFrame(rows)


def make_transactions(rng: np.random.Generator, activity: pd.DataFrame, users: pd.DataFrame) -> pd.DataFrame:
    purchasers = activity[activity["purchase"] == 1].merge(users[["user_id", "persona", "price_sensitivity"]], on="user_id")
    if purchasers.empty:
        return pd.DataFrame(columns=["transaction_id", "user_id", "order_date", "revenue", "gross_margin"])

    persona_aov = {
        "Value seeker": 42,
        "Power user": 72,
        "New explorer": 48,
        "At-risk": 35,
        "Premium": 105,
    }
    base_aov = purchasers["persona"].map(persona_aov).to_numpy()
    discount_drag = 1 - 0.18 * purchasers["price_sensitivity"].to_numpy()
    revenue = np.maximum(8, rng.normal(base_aov * discount_drag, base_aov * 0.22))
    margin_rate = np.clip(rng.normal(0.48, 0.06, len(purchasers)), 0.30, 0.66)

    return pd.DataFrame(
        {
            "transaction_id": [f"T{i:07d}" for i in range(1, len(purchasers) + 1)],
            "user_id": purchasers["user_id"].to_numpy(),
            "order_date": purchasers["event_date"].to_numpy(),
            "revenue": revenue.round(2),
            "gross_margin": (revenue * margin_rate).round(2),
        }
    )


def generate_all(output_dir: Path, config: DataConfig = DataConfig()) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    users = make_users(rng, config)
    assignments = make_experiment_assignments(rng, users, config)
    campaign = make_observational_campaign(rng, users, config)
    activity = make_daily_activity(rng, users, assignments, campaign, config)
    transactions = make_transactions(rng, activity, users)

    tables = {
        "users": users,
        "experiment_assignments": assignments,
        "campaign_exposures": campaign,
        "daily_activity": activity,
        "transactions": transactions,
    }
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    return tables
