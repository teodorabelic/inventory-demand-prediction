from __future__ import annotations

import pandas as pd

from .config import DATE_COLUMN, TARGET_COLUMN

# eda analizira podatke pre treniranja modela

# vraca dict sa osnovnim informacijama
def build_eda_summary(df: pd.DataFrame) -> dict:
    daily_sales = df.groupby(DATE_COLUMN, as_index=False)[TARGET_COLUMN].sum()
    daily_sales[DATE_COLUMN] = daily_sales[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    monthly_sales = (
        df.assign(month=df[DATE_COLUMN].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)[TARGET_COLUMN]
        .sum()
    )
    top_stores = (
        df.groupby("store", as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(TARGET_COLUMN, ascending=False)
        .head(5)
    )
    top_items = (
        df.groupby("item", as_index=False)[TARGET_COLUMN]
        .sum()
        .sort_values(TARGET_COLUMN, ascending=False)
        .head(5)
    )

    return {
        "rows": int(len(df)),
        "date_min": df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "date_max": df[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "stores": int(df["store"].nunique()),
        "items": int(df["item"].nunique()),
        "sales_mean": float(df[TARGET_COLUMN].mean()),
        "sales_std": float(df[TARGET_COLUMN].std()),
        "sales_min": float(df[TARGET_COLUMN].min()),
        "sales_max": float(df[TARGET_COLUMN].max()),
        "top_stores": top_stores.to_dict(orient="records"),
        "top_items": top_items.to_dict(orient="records"),
        "monthly_sales_head": monthly_sales.head(12).to_dict(orient="records"),
        "daily_sales_head": daily_sales.head(10).to_dict(orient="records"),
    }


# pravi konkretne tabele
def build_eda_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary_by_store = df.groupby("store", as_index=False).agg(
        average_sales=(TARGET_COLUMN, "mean"),
        total_sales=(TARGET_COLUMN, "sum"),
        sales_std=(TARGET_COLUMN, "std"),
    )

    summary_by_item = df.groupby("item", as_index=False).agg(
        average_sales=(TARGET_COLUMN, "mean"),
        total_sales=(TARGET_COLUMN, "sum"),
        sales_std=(TARGET_COLUMN, "std"),
    )

    correlation_features = df.copy()
    correlation_features["day_of_week"] = correlation_features[DATE_COLUMN].dt.dayofweek
    correlation_features["month"] = correlation_features[DATE_COLUMN].dt.month
    correlation_table = correlation_features[["store", "item", "day_of_week", "month", TARGET_COLUMN]].corr(
        numeric_only=True
    )

    return {
        "eda_store_summary": summary_by_store,
        "eda_item_summary": summary_by_item,
        "eda_correlation": correlation_table.reset_index(),
    }
