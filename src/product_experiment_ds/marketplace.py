from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


OLIST_FILES = {
    "orders": "olist_orders_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "products": "olist_products_dataset.csv",
}


def olist_available(olist_dir: Path) -> bool:
    return all((olist_dir / filename).exists() for filename in OLIST_FILES.values())


def build_marketplace_orders(olist_dir: Path, fallback_features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if olist_available(olist_dir):
        return build_from_olist(olist_dir)
    return build_fallback_marketplace(fallback_features)


def build_from_olist(olist_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    orders = pd.read_csv(olist_dir / OLIST_FILES["orders"])
    customers = pd.read_csv(olist_dir / OLIST_FILES["customers"])
    items = pd.read_csv(olist_dir / OLIST_FILES["items"])
    reviews = pd.read_csv(olist_dir / OLIST_FILES["reviews"])
    payments = pd.read_csv(olist_dir / OLIST_FILES["payments"])
    products = pd.read_csv(olist_dir / OLIST_FILES["products"])

    item_agg = (
        items.groupby("order_id")
        .agg(
            revenue=("price", "sum"),
            freight_value=("freight_value", "sum"),
            seller_count=("seller_id", "nunique"),
            product_id=("product_id", "first"),
        )
        .reset_index()
    )
    payment_agg = (
        payments.groupby("order_id")
        .agg(payment_value=("payment_value", "sum"), payment_type=("payment_type", "first"))
        .reset_index()
    )
    review_agg = reviews.groupby("order_id").agg(review_score=("review_score", "mean")).reset_index()
    product_lookup = products[["product_id", "product_category_name"]].drop_duplicates("product_id")

    df = (
        orders.merge(customers, on="customer_id", how="left")
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
        .merge(product_lookup, on="product_id", how="left")
    )
    date_cols = [
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["estimated_delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.days
    df["delivered_late"] = (df["estimated_delay_days"] > 0).astype(int)
    df["bad_review"] = (df["review_score"].fillna(5) <= 2).astype(int)
    df["order_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)
    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df["payment_type"] = df["payment_type"].fillna("unknown")
    df["customer_state"] = df["customer_state"].fillna("unknown")
    df["revenue"] = df["revenue"].fillna(df["payment_value"]).fillna(0)
    df = df[
        [
            "order_id",
            "customer_unique_id",
            "customer_state",
            "order_purchase_timestamp",
            "order_month",
            "product_category_name",
            "payment_type",
            "revenue",
            "freight_value",
            "seller_count",
            "delivery_days",
            "estimated_delay_days",
            "delivered_late",
            "review_score",
            "bad_review",
        ]
    ].rename(columns={"customer_unique_id": "user_id", "order_purchase_timestamp": "order_date"})
    summary = {
        "source": "Olist Brazilian E-Commerce Public Dataset",
        "status": "real_olist_csv_loaded",
        "orders": f"{len(df):,}",
        "note": "Treatment is simulated on top of real marketplace orders for product experimentation framing.",
    }
    return df, summary


def build_fallback_marketplace(features: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    rng = np.random.default_rng(91)
    categories = np.array(["electronics", "home", "beauty", "sports", "fashion", "books", "grocery"])
    payment_types = np.array(["credit_card", "voucher", "boleto", "debit_card"])
    state_map = {"West": "SP", "Northeast": "BA", "South": "RS", "Midwest": "GO"}
    rows = []
    for row in features.itertuples(index=False):
        n_orders = int(max(0, row.pre_purchases + row.post_purchases))
        if n_orders == 0 and rng.random() < 0.18:
            n_orders = 1
        for idx in range(n_orders):
            post = idx >= row.pre_purchases
            order_date = pd.Timestamp("2026-03-01") + pd.Timedelta(days=int(rng.integers(0, 42))) if post else pd.Timestamp("2026-02-01") + pd.Timedelta(days=int(rng.integers(0, 28)))
            category = rng.choice(categories)
            payment = rng.choice(payment_types, p=[0.68, 0.08, 0.18, 0.06])
            revenue = max(8, rng.normal(58 + row.engagement_score * 38 + row.high_value_user * 45, 22))
            delivery_days = max(1, int(rng.normal(8 + row.price_sensitivity * 4 + (payment == "boleto") * 2, 4)))
            late_prob = min(0.72, 0.08 + 0.025 * delivery_days + (category == "electronics") * 0.06)
            delivered_late = int(rng.random() < late_prob)
            bad_review_prob = min(0.65, 0.04 + delivered_late * 0.22 + (delivery_days > 14) * 0.12)
            bad_review = int(rng.random() < bad_review_prob)
            review_score = int(rng.choice([1, 2], p=[0.45, 0.55])) if bad_review else int(rng.choice([3, 4, 5], p=[0.18, 0.36, 0.46]))
            rows.append(
                {
                    "order_id": f"M{len(rows) + 1:07d}",
                    "user_id": row.user_id,
                    "customer_state": state_map.get(row.region, "SP"),
                    "order_date": order_date,
                    "order_month": order_date.to_period("M").strftime("%Y-%m"),
                    "product_category_name": category,
                    "payment_type": payment,
                    "revenue": round(revenue, 2),
                    "freight_value": round(max(5, rng.normal(14, 4)), 2),
                    "seller_count": int(rng.choice([1, 1, 1, 2])),
                    "delivery_days": delivery_days,
                    "estimated_delay_days": max(-6, delivery_days - int(rng.normal(9, 3))),
                    "delivered_late": delivered_late,
                    "review_score": review_score,
                    "bad_review": bad_review,
                }
            )
    df = pd.DataFrame(rows)
    summary = {
        "source": "Olist-compatible fallback generated from product experiment users",
        "status": "fallback_used",
        "orders": f"{len(df):,}",
        "note": "Place Olist CSVs under data/olist/ to switch this layer to real marketplace orders.",
    }
    return df, summary


def build_root_cause_outputs(marketplace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_bad = marketplace["bad_review"].mean()
    overall_late = marketplace["delivered_late"].mean()
    rows = []
    for dimension in ["customer_state", "product_category_name", "payment_type"]:
        grouped = (
            marketplace.groupby(dimension)
            .agg(
                orders=("order_id", "count"),
                revenue=("revenue", "sum"),
                bad_review_rate=("bad_review", "mean"),
                late_delivery_rate=("delivered_late", "mean"),
                avg_delivery_days=("delivery_days", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "driver_value"})
        )
        grouped["driver"] = dimension
        grouped["bad_review_lift_vs_avg"] = grouped["bad_review_rate"] - overall_bad
        grouped["late_lift_vs_avg"] = grouped["late_delivery_rate"] - overall_late
        grouped["estimated_bad_review_excess"] = grouped["bad_review_lift_vs_avg"] * grouped["orders"]
        rows.append(grouped)
    drivers = pd.concat(rows, ignore_index=True)
    drivers = drivers[drivers["orders"] >= max(20, int(len(marketplace) * 0.01))]
    drivers = drivers.sort_values("estimated_bad_review_excess", ascending=False)

    metric_tree = pd.DataFrame(
        [
            ["North Star", "Marketplace repeat purchase readiness", 1 - overall_bad, "Higher is better"],
            ["Primary", "Bad review rate", overall_bad, "Lower is better"],
            ["Guardrail", "Late delivery rate", overall_late, "Lower is better"],
            ["Diagnostic", "Average delivery days", marketplace["delivery_days"].mean(), "Lower is better"],
            ["Diagnostic", "Average order revenue", marketplace["revenue"].mean(), "Higher is better"],
        ],
        columns=["level", "metric", "value", "direction"],
    )
    return drivers, metric_tree


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 2:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=quantiles)
    actual_counts, _ = np.histogram(actual, bins=quantiles)
    expected_pct = np.clip(expected_counts / expected_counts.sum(), 0.001, None)
    actual_pct = np.clip(actual_counts / actual_counts.sum(), 0.001, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def build_monitoring_outputs(marketplace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = marketplace.sort_values("order_date").copy()
    midpoint = ordered["order_date"].quantile(0.50)
    baseline = ordered[ordered["order_date"] <= midpoint]
    current = ordered[ordered["order_date"] > midpoint]

    metric_specs = [
        ("bad_review_rate", baseline["bad_review"].mean(), current["bad_review"].mean(), 0.03, "lower"),
        ("late_delivery_rate", baseline["delivered_late"].mean(), current["delivered_late"].mean(), 0.03, "lower"),
        ("avg_delivery_days", baseline["delivery_days"].mean(), current["delivery_days"].mean(), 1.5, "lower"),
        ("avg_revenue", baseline["revenue"].mean(), current["revenue"].mean(), 8.0, "higher"),
    ]
    metric_rows = []
    for metric, base, curr, threshold, direction in metric_specs:
        delta = curr - base
        if direction == "lower":
            status = "Alert" if delta > threshold else "Healthy"
        else:
            status = "Alert" if delta < -threshold else "Healthy"
        metric_rows.append(
            {
                "monitor": metric,
                "baseline": base,
                "current": curr,
                "delta": delta,
                "threshold": threshold,
                "status": status,
            }
        )

    drift_rows = []
    for feature in ["revenue", "freight_value", "delivery_days", "review_score"]:
        psi = population_stability_index(baseline[feature], current[feature])
        drift_rows.append(
            {
                "feature": feature,
                "psi": psi,
                "status": "Alert" if psi >= 0.20 else "Watch" if psi >= 0.10 else "Healthy",
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(drift_rows)


def save_marketplace_layer(
    processed_dir: Path,
    olist_dir: Path,
    features: pd.DataFrame,
) -> None:
    marketplace, summary = build_marketplace_orders(olist_dir, features)
    drivers, metric_tree = build_root_cause_outputs(marketplace)
    monitoring, drift = build_monitoring_outputs(marketplace)
    marketplace.to_csv(processed_dir / "marketplace_orders.csv", index=False)
    drivers.to_csv(processed_dir / "root_cause_drivers.csv", index=False)
    metric_tree.to_csv(processed_dir / "metric_tree.csv", index=False)
    monitoring.to_csv(processed_dir / "monitoring_metrics.csv", index=False)
    drift.to_csv(processed_dir / "monitoring_drift.csv", index=False)
    (processed_dir / "marketplace_source.json").write_text(json.dumps(summary, indent=2))
