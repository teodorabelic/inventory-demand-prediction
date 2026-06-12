from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.config import DATE_COLUMN, DEFAULT_DATA_PATH, DEMO_DATA_PATH
from src.classification import add_demand_class_labels, train_gaussian_nb_classifier
from src.data import clean_sales_data, generate_demo_data, load_sales_data
from src.eda import build_eda_summary, build_eda_tables
from src.features import build_features, get_model_matrices, split_time_series_data
from src.inventory import build_inventory_analysis
from src.models import extract_feature_importance, train_and_compare_models
from src.recommendations import generate_recommendations
from src.reporting import save_training_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train demand forecasting and inventory optimization pipeline.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to input CSV data.")
    parser.add_argument("--forecast-horizon", type=int, default=30, help="Future planning horizon in days.")
    parser.add_argument("--use-demo-data", action="store_true", help="Use generated demo data instead of file input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ucitavanje podataka
    if args.use_demo_data:
        raw_df = generate_demo_data()
        data_source = "generated demo data"
    else:
        data_path = Path(args.data)
        if not data_path.exists() and Path(DEMO_DATA_PATH).exists():
            data_path = Path(DEMO_DATA_PATH)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {args.data}. Add Kaggle train.csv to data/train.csv or use --use-demo-data."
            )
        raw_df = load_sales_data(data_path)
        data_source = str(data_path)

    # ciscenje podataka
    clean_df = clean_sales_data(raw_df)
    # pravljenje feature-a
    featured_df = build_features(clean_df, dropna=True)

    # podela na train i test - po datumima, ne random
    train_df, test_df = split_time_series_data(featured_df, test_size=0.2)
    # y - sales, x - date, sales, is_anomaly
    x_train, y_train = get_model_matrices(train_df)
    x_test, y_test = get_model_matrices(test_df)

    # trenira vise modela i poredi ih
    model_results, best_result, predictions_df = train_and_compare_models(x_train, y_train, x_test, y_test)
    # dodatna klasifikacija nivoa potraznje
    train_class_df, test_class_df = add_demand_class_labels(train_df, test_df)
    x_train_class = x_train.copy()
    x_test_class = x_test.copy()
    y_train_class = train_class_df["demand_class"].copy()
    y_test_class = test_class_df["demand_class"].copy()
    _, classification_predictions_df, classification_metrics = train_gaussian_nb_classifier(
        x_train_class,
        y_train_class,
        x_test_class,
        y_test_class,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "model": result.name,
                "mae": result.mae,
                "rmse": result.rmse,
                "mape": result.mape,
                "r2": result.r2,
            }
            for result in model_results
        ]
    )
    metrics_df["accuracy"] = pd.NA
    metrics_df["f1_weighted"] = pd.NA
    metrics_df.loc[len(metrics_df)] = {
        "model": "GaussianNB Demand Classifier",
        "mae": pd.NA,
        "rmse": pd.NA,
        "mape": pd.NA,
        "r2": pd.NA,
        "accuracy": classification_metrics["accuracy"],
        "f1_weighted": classification_metrics["f1_weighted"],
    }

    # predikcija najboljeg modela
    best_predictions = best_result.pipeline.predict(x_test)
    output_predictions_df = test_df[[DATE_COLUMN, "store", "item", "sales"]].copy()
    output_predictions_df["predicted_sales"] = best_predictions
    output_predictions_df["residual"] = output_predictions_df["sales"] - output_predictions_df["predicted_sales"]

    # izvlaci znacaj atributa
    feature_importance_df = extract_feature_importance(best_result.pipeline, list(x_train.columns))
    # analiza zaliha
    inventory_df = build_inventory_analysis(
        historical_df=clean_df,
        featured_test_df=test_df,
        predicted_sales=pd.Series(best_predictions, index=test_df.index),
        forecast_horizon=args.forecast_horizon,
    )
    # prevodi analizu u preporuke
    inventory_df = generate_recommendations(inventory_df)
    # spremaju statistike za Streamlit i notebook
    eda_summary = build_eda_summary(clean_df)
    eda_tables = build_eda_tables(clean_df)

    summary = {
        "data_source": data_source,
        "rows_raw": int(len(raw_df)),
        "rows_clean": int(len(clean_df)),
        "rows_featured": int(len(featured_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "best_model": best_result.name,
        "forecast_horizon": int(args.forecast_horizon),
        "eda": eda_summary,
        "classification_metrics": classification_metrics,
        "metrics": {
            "mae": best_result.mae,
            "rmse": best_result.rmse,
            "mape": best_result.mape,
            "r2": best_result.r2,
        },
    }

    save_training_outputs(
        metrics_df=metrics_df,
        predictions_df=output_predictions_df,
        feature_importance_df=feature_importance_df,
        inventory_df=inventory_df,
        classification_predictions_df=classification_predictions_df,
        best_model=best_result.pipeline,
        summary=summary,
        extra_tables=eda_tables,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
