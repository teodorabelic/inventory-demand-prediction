from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR

# vizuelni deo projekta
# ucitava podatke iz artifacts/ i pokazuje ih u Streamlit interfejsu

st.set_page_config(page_title="Predikcija potraznje i optimizacija zaliha", layout="wide")

def read_csv_artifact(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# cita training_summary.json - sazetak rezultata treninga
# nastaje nakon pokretanja python scripts/train_pipeline.py
# da app.py brzo procita informacije bez ponovnog treniranja
def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.title("Inteligentni sistem za predikciju potraznje i optimizaciju zaliha")

    # ucitavanje rezultata iz artifacts/
    summary = read_summary(ARTIFACTS_DIR / "training_summary.json")
    metrics = read_csv_artifact(ARTIFACTS_DIR / "metrics.csv")
    predictions = read_csv_artifact(ARTIFACTS_DIR / "predictions.csv")
    inventory = read_csv_artifact(ARTIFACTS_DIR / "inventory_recommendations.csv")
    feature_importance = read_csv_artifact(ARTIFACTS_DIR / "feature_importance.csv")
    demand_class_predictions = read_csv_artifact(ARTIFACTS_DIR / "demand_class_predictions.csv")
    eda_store_summary = read_csv_artifact(ARTIFACTS_DIR / "eda_store_summary.csv")
    eda_item_summary = read_csv_artifact(ARTIFACTS_DIR / "eda_item_summary.csv")

    # ukoliko model nije istreniran
    if not summary:
        st.warning("Nema istreniranih artefakata. Pokreni `python scripts/train_pipeline.py --use-demo-data` ili koristi Kaggle train.csv.")
        st.stop()

    # glavne metrike
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Najbolji model", summary.get("best_model", "-"))
    col2.metric("RMSE", f"{summary.get('metrics', {}).get('rmse', 0):.2f}")
    col3.metric("MAE", f"{summary.get('metrics', {}).get('mae', 0):.2f}")
    col4.metric("MAPE", f"{summary.get('metrics', {}).get('mape', 0):.2f}%")

    # tabela performansi svih modela
    st.subheader("Poredjenje modela")
    st.dataframe(metrics, use_container_width=True)

    eda = summary.get("eda", {})
    if eda:
        st.subheader("EDA pregled")
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Broj redova", eda.get("rows", 0))
        e2.metric("Prodavnice", eda.get("stores", 0))
        e3.metric("Proizvodi", eda.get("items", 0))
        e4.metric("Prosecna prodaja", f"{eda.get('sales_mean', 0):.2f}")
        # 2 tabele
        left_eda, right_eda = st.columns(2)
        with left_eda:
            # pregled po prodavnici
            st.dataframe(eda_store_summary.head(10), use_container_width=True)
        with right_eda:
            # pregled po proizvodu
            st.dataframe(eda_item_summary.head(10), use_container_width=True)

    # graf
    if not predictions.empty:
        st.subheader("Stvarna i predvidjena prodaja")
        sample = predictions.copy()
        sample["date"] = pd.to_datetime(sample["date"])
        selected_store = st.selectbox("Prodavnica", sorted(sample["store"].unique()))
        filtered_items = sorted(sample.loc[sample["store"] == selected_store, "item"].unique())
        selected_item = st.selectbox("Proizvod", filtered_items)

        subset = sample[(sample["store"] == selected_store) & (sample["item"] == selected_item)].sort_values("date")
        chart_df = subset.set_index("date")[["sales", "predicted_sales"]]
        st.line_chart(chart_df, use_container_width=True)
        st.dataframe(subset.tail(30), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Preporuke za zalihe")
        st.dataframe(inventory.head(100), use_container_width=True)
    with right:
        st.subheader("Znacaj atributa")
        if feature_importance.empty:
            st.info("Odabrani model ne izbacuje direktan znacaj atributa.")
        else:
            top_features = feature_importance.head(15).set_index("feature")
            st.bar_chart(top_features["importance"], use_container_width=True)
            st.dataframe(feature_importance.head(25), use_container_width=True)

    if not inventory.empty:
        st.subheader("Distribucija preporuka")
        recommendation_counts = inventory["recommendation"].value_counts()
        st.bar_chart(recommendation_counts, use_container_width=True)
        if "inventory_cluster" in inventory.columns:
            st.subheader("KMeans segmentacija artikala")
            cluster_counts = inventory["inventory_cluster"].value_counts().sort_index()
            st.bar_chart(cluster_counts, use_container_width=True)

    if not demand_class_predictions.empty:
        st.subheader("GaussianNB klasifikacija nivoa traznje")
        st.dataframe(demand_class_predictions.head(50), use_container_width=True)


if __name__ == "__main__":
    main()
