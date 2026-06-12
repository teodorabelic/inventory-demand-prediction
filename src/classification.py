from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB


# podela sales
DEMAND_LABELS = ["low", "medium", "high"]


def add_demand_class_labels(train_df: pd.DataFrame, test_df: pd.DataFrame, target_column: str = "sales") -> tuple[pd.DataFrame, pd.DataFrame]:
    train_labeled = train_df.copy()
    test_labeled = test_df.copy()

    low_threshold = float(train_labeled[target_column].quantile(0.33))
    high_threshold = float(train_labeled[target_column].quantile(0.66))

    def classify(value: float) -> str:
        if value <= low_threshold:
            return "low"
        if value <= high_threshold:
            return "medium"
        return "high"

    train_labeled["demand_class"] = train_labeled[target_column].apply(classify)
    test_labeled["demand_class"] = test_labeled[target_column].apply(classify)
    return train_labeled, test_labeled


# trenira GaussianNB
def train_gaussian_nb_classifier(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[GaussianNB, pd.DataFrame, dict[str, float]]:
    model = GaussianNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    prediction_df = pd.DataFrame(
        {
            "actual_demand_class": y_test.to_numpy(),
            "predicted_demand_class": predictions,
        }
    )
    return model, prediction_df, metrics
