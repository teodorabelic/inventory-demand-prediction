from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, GROUP_COLUMNS, TARGET_COLUMN


@dataclass
class DatasetBundle:
    raw: pd.DataFrame
    prepared: pd.DataFrame


def load_sales_data(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {DATE_COLUMN, *GROUP_COLUMNS, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN]).sort_values(GROUP_COLUMNS + [DATE_COLUMN])
    return df.reset_index(drop=True)


def clean_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean = clean.drop_duplicates()
    clean = clean[clean[TARGET_COLUMN] >= 0].copy()

    # detekcija anomalija preko IQR pravila
    q1 = clean[TARGET_COLUMN].quantile(0.25)
    q3 = clean[TARGET_COLUMN].quantile(0.75)
    # srednjih 50% podataka
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    # anomalije se samo naznace, ne brisu se
    clean["is_anomaly"] = clean[TARGET_COLUMN] > upper
    return clean


def generate_demo_data(
    start_date: str = "2018-01-01",
    periods: int = 730,
    n_stores: int = 5,
    n_items: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=periods, freq="D")

    rows: list[dict[str, float | int | str]] = []
    for store in range(1, n_stores + 1):
        store_bias = rng.uniform(2, 10)
        for item in range(1, n_items + 1):
            item_bias = rng.uniform(8, 30)
            trend = rng.uniform(-0.01, 0.05)
            weekly_amp = rng.uniform(1.5, 5.0)
            yearly_amp = rng.uniform(2.0, 8.0)

            for day_idx, date in enumerate(dates):
                weekly = weekly_amp * np.sin(2 * np.pi * date.dayofweek / 7)
                yearly = yearly_amp * np.sin(2 * np.pi * date.dayofyear / 365.25)
                noise = rng.normal(0, 2.0)
                base = store_bias + item_bias + trend * day_idx + weekly + yearly + noise
                sales = max(0, int(round(base)))
                rows.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store": store,
                        "item": item,
                        "sales": sales,
                    }
                )

    demo = pd.DataFrame(rows)
    demo["date"] = pd.to_datetime(demo["date"])
    return demo.sort_values(["store", "item", "date"]).reset_index(drop=True)
