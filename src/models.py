from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    mae: float
    rmse: float
    mape: float
    r2: float


def mean_absolute_percentage_error_safe(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, 1, y_true)
    return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error_safe(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_preprocessor(numeric_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_columns)],
        remainder="drop",
    )


def model_candidates(random_state: int = 42) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=180,
            max_depth=16,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7, weights="distance"),
        "MLP Regressor": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            learning_rate_init=0.001,
            max_iter=250,
            random_state=random_state,
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost Regressor"] = XGBRegressor(
            n_estimators=250,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    return candidates


def train_and_compare_models(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42,
) -> tuple[list[ModelResult], ModelResult, pd.DataFrame]:
    results: list[ModelResult] = []
    predictions_rows: list[pd.DataFrame] = []
    numeric_columns = list(x_train.columns)

    for model_name, estimator in model_candidates(random_state=random_state).items():
        preprocessor = build_preprocessor(numeric_columns)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        metrics = evaluate_predictions(y_test, preds)
        results.append(
            ModelResult(
                name=model_name,
                pipeline=pipeline,
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                mape=metrics["mape"],
                r2=metrics["r2"],
            )
        )
        predictions_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "actual_sales": y_test.to_numpy(),
                    "predicted_sales": preds,
                }
            )
        )

    results = sorted(results, key=lambda result: result.rmse)
    best = results[0]
    predictions = pd.concat(predictions_rows, ignore_index=True)
    return results, best, predictions


def extract_feature_importance(best_model: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    estimator = best_model.named_steps["model"]
    importances = None

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_).reshape(-1)
        importances = np.abs(coef)

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
