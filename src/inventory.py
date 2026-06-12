from __future__ import annotations

import math

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .config import GROUP_COLUMNS


def classify_trend(recent_mean: float, previous_mean: float, tolerance: float = 0.05) -> str:
    if previous_mean <= 0 and recent_mean > 0:
        return "rastući trend"
    if previous_mean <= 0:
        return "stabilan trend"

    ratio = (recent_mean - previous_mean) / previous_mean
    if ratio > tolerance:
        return "rastući trend"
    if ratio < -tolerance:
        return "opadajući trend"
    return "stabilan trend"


def build_inventory_analysis(
    historical_df: pd.DataFrame,
    featured_test_df: pd.DataFrame,
    predicted_sales: pd.Series,
    forecast_horizon: int = 30,
    lead_time_days: int = 7,
    safety_factor: float = 1.65,
) -> pd.DataFrame:
    analysis = featured_test_df[GROUP_COLUMNS].copy()
    analysis["predicted_sales"] = predicted_sales.to_numpy()

    summary = (
        analysis.groupby(GROUP_COLUMNS, as_index=False)["predicted_sales"]
        .mean()
        .rename(columns={"predicted_sales": "predicted_daily_sales"})
    )

    hist = historical_df.copy()
    hist["recent_30"] = hist.groupby(GROUP_COLUMNS)["sales"].transform(lambda s: s.rolling(30, min_periods=7).mean())
    hist["previous_30"] = hist.groupby(GROUP_COLUMNS)["sales"].transform(
        lambda s: s.shift(30).rolling(30, min_periods=7).mean()
    )

    latest = (
        hist.sort_values("date")
        .groupby(GROUP_COLUMNS, as_index=False)
        .tail(1)[GROUP_COLUMNS + ["sales", "recent_30", "previous_30"]]
        .rename(columns={"sales": "current_stock_proxy"})
    )

    daily_std = (
        hist.groupby(GROUP_COLUMNS, as_index=False)["sales"]
        .std()
        .rename(columns={"sales": "daily_sales_std"})
    )

    merged = summary.merge(latest, on=GROUP_COLUMNS, how="left").merge(daily_std, on=GROUP_COLUMNS, how="left")
    merged["daily_sales_std"] = merged["daily_sales_std"].fillna(0.0)

    merged["trend"] = merged.apply(
        lambda row: classify_trend(
            float(row.get("recent_30", 0) or 0),
            float(row.get("previous_30", 0) or 0),
        ),
        axis=1,
    )
    merged["safety_stock"] = (safety_factor * merged["daily_sales_std"] * math.sqrt(lead_time_days)).round(2)
    merged["reorder_point"] = (
        merged["predicted_daily_sales"] * lead_time_days + merged["safety_stock"]
    ).round(2)
    merged["days_of_cover"] = (
        merged["current_stock_proxy"] / merged["predicted_daily_sales"].clip(lower=0.1)
    ).round(2)
    merged["recommended_order_qty"] = (
        merged["predicted_daily_sales"] * forecast_horizon + merged["safety_stock"] - merged["current_stock_proxy"]
    ).clip(lower=0).round().astype(int)

    clustering_features = merged[["predicted_daily_sales", "daily_sales_std", "days_of_cover", "recommended_order_qty"]].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(clustering_features)
    n_clusters = min(3, len(merged))
    if n_clusters >= 2:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        merged["inventory_cluster"] = kmeans.fit_predict(scaled)
    else:
        merged["inventory_cluster"] = 0

    return merged.sort_values(["store", "item"]).reset_index(drop=True)
