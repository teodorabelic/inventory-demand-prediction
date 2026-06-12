from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import ARTIFACTS_DIR


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR


def save_training_outputs(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    classification_predictions_df: pd.DataFrame,
    best_model,
    summary: dict,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> None:
    artifacts_dir = ensure_artifacts_dir()
    metrics_df.to_csv(artifacts_dir / "metrics.csv", index=False)
    predictions_df.to_csv(artifacts_dir / "predictions.csv", index=False)
    feature_importance_df.to_csv(artifacts_dir / "feature_importance.csv", index=False)
    inventory_df.to_csv(artifacts_dir / "inventory_recommendations.csv", index=False)
    classification_predictions_df.to_csv(artifacts_dir / "demand_class_predictions.csv", index=False)
    joblib.dump(best_model, artifacts_dir / "best_model.joblib")
    for name, table in (extra_tables or {}).items():
        table.to_csv(artifacts_dir / f"{name}.csv", index=False)
    (artifacts_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
