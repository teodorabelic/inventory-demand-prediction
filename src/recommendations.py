from __future__ import annotations

import pandas as pd


def generate_recommendations(inventory_df: pd.DataFrame) -> pd.DataFrame:
    df = inventory_df.copy()
    labels: list[str] = []
    risks: list[str] = []

    for _, row in df.iterrows():
        days_of_cover = float(row["days_of_cover"])
        current_stock = float(row["current_stock_proxy"])
        reorder_point = float(row["reorder_point"])
        predicted_daily_sales = float(row["predicted_daily_sales"])
        trend = row["trend"]

        if current_stock < reorder_point * 0.7 or days_of_cover < 5:
            labels.append("potrebno je hitno narucivanje")
            risks.append("visok rizik od nestasice")
        elif current_stock < reorder_point or days_of_cover < 10:
            labels.append("preporucuje se dopuna zaliha")
            risks.append("umeren rizik od nestasice")
        elif days_of_cover > 45 and predicted_daily_sales < 5:
            labels.append("postoji rizik od viska zaliha")
            risks.append("visok rizik od viska")
        elif trend == "opadajući trend" and days_of_cover > 30:
            labels.append("preporucuje se smanjenje narednih narudzbina")
            risks.append("rizik od akumulacije zaliha")
        else:
            labels.append("nema potrebe za dodatnim narucivanjem")
            risks.append("nizak rizik")

    df["recommendation"] = labels
    df["risk_level"] = risks
    return df
