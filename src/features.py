from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATE_COLUMN, GROUP_COLUMNS, TARGET_COLUMN

# vremenska serija, pa dajem modelu informacije o proslom
# ponasanju prodaje

# kalendarski atributi
def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["day_of_week"] = out[DATE_COLUMN].dt.dayofweek
    out["day_of_month"] = out[DATE_COLUMN].dt.day
    out["week_of_year"] = out[DATE_COLUMN].dt.isocalendar().week.astype(int)
    out["month"] = out[DATE_COLUMN].dt.month
    out["quarter"] = out[DATE_COLUMN].dt.quarter
    out["year"] = out[DATE_COLUMN].dt.year
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["season"] = ((out["month"] % 12 + 3) // 3).astype(int)
    return out


# lag
def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(GROUP_COLUMNS)[TARGET_COLUMN]
    for lag in (1, 7, 14, 30):
        out[f"lag_{lag}"] = grouped.shift(lag)
    for window in (7, 14, 30):
        out[f"rolling_mean_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
        )
        out[f"rolling_std_{window}"] = grouped.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).std()
        )
    return out


# trend
def _add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(GROUP_COLUMNS)[TARGET_COLUMN]
    out["sales_diff_1"] = grouped.transform(lambda s: s.shift(1).diff(1))
    out["sales_pct_change_1"] = grouped.transform(lambda s: s.shift(1).pct_change(1)).replace(
        [np.inf, -np.inf], np.nan
    )
    out["sales_diff_7"] = grouped.transform(lambda s: s.shift(1).diff(7))
    out["sales_pct_change_7"] = grouped.transform(lambda s: s.shift(1).pct_change(7)).replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def build_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    featured = _add_calendar_features(df)
    featured = _add_lag_features(featured)
    featured = _add_trend_features(featured)

    featured = featured.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)
    feature_cols = [c for c in featured.columns if c not in {TARGET_COLUMN, DATE_COLUMN}]
    featured[feature_cols] = featured[feature_cols].replace([np.inf, -np.inf], np.nan)

    if dropna:
        featured = featured.dropna().reset_index(drop=True)
    return featured


def split_time_series_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique_dates = np.array(sorted(df[DATE_COLUMN].unique()))
    if len(unique_dates) < 10:
        raise ValueError("Dataset is too small for time-based split.")

    split_idx = int(len(unique_dates) * (1 - test_size))
    split_date = unique_dates[split_idx]
    train_df = df[df[DATE_COLUMN] < split_date].copy()
    test_df = df[df[DATE_COLUMN] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time split produced empty train or test data.")

    return train_df, test_df


def get_model_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_columns = [c for c in df.columns if c not in {DATE_COLUMN, TARGET_COLUMN, "is_anomaly"}]
    x = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()
    return x, y
