# Demand Forecasting and Inventory Optimization

A machine-learning system for retail demand forecasting, demand-level classification, inventory segmentation, and replenishment recommendations. The project includes a reproducible training pipeline and a Streamlit dashboard for exploring model results.

## Highlights

- Exploratory data analysis for store and item sales
- Time-series feature engineering and chronological train/test splitting
- Comparison of multiple regression models
- Demand-level classification with Gaussian Naive Bayes
- Inventory segmentation with KMeans clustering
- Safety-stock and replenishment recommendations
- Streamlit dashboard for metrics, predictions, feature importance, and inventory insights

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Matplotlib, Seaborn, and joblib.

## Project Structure

```text
app.py                         Streamlit dashboard
src/                           Data, features, models, inventory, and reporting modules
scripts/generate_demo_data.py  Generates a synthetic dataset
scripts/train_pipeline.py     Runs training and evaluation
data/demo_train.csv            Small synthetic dataset for reproducible demos
artifacts/                     Locally generated models, metrics, and predictions
odbrana_projekta.ipynb         Project analysis notebook
```

## Machine-Learning Methods

The regression pipeline compares Linear Regression, Ridge Regression, KNN Regressor, Random Forest Regressor, Gradient Boosting Regressor, MLP Regressor, and XGBoost Regressor when XGBoost is available. The best model is selected using regression performance on a chronological holdout set.

Gaussian Naive Bayes classifies demand as low, medium, or high. KMeans groups inventory items using predicted demand, demand variability, days of cover, and recommended order quantity.

## Feature Engineering

The pipeline derives calendar features from the date, including day of week, month, and year. It also creates per-store/item lag features and rolling/trend statistics. Rows with unavailable historical features are removed, and the data is split by time rather than randomly to avoid using future observations during training.

## Evaluation

The included demo run selected XGBoost Regressor as the best model:

| Metric | Result |
| --- | ---: |
| MAE | 1.82 |
| RMSE | 2.30 |
| MAPE | 6.21% |
| R² | 0.974 |

The Gaussian Naive Bayes classifier achieved 90.8% accuracy and a weighted F1-score of 0.910 on the same demo evaluation workflow. Results can vary when a different dataset or configuration is used.

## Data

The pipeline accepts the Kaggle *Store Item Demand Forecasting Challenge* format with these columns:

```text
date, store, item, sales
```

Place the dataset at `data/train.csv`. The real dataset is intentionally not included. The repository includes `data/demo_train.csv`, a synthetic dataset that can be used without external downloads.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Generate Data and Train

Run the complete demo workflow:

```powershell
python scripts/generate_demo_data.py
python scripts/train_pipeline.py --use-demo-data
```

To train on the Kaggle-format dataset:

```powershell
python scripts/train_pipeline.py --data data/train.csv --forecast-horizon 30
```

Training outputs are written to `artifacts/`, including metrics, predictions, feature importance, inventory recommendations, and the serialized best model.

## Run the Dashboard

```powershell
streamlit run app.py
```

The dashboard reads the generated files from `artifacts/`. Run the training pipeline first if those files are not available.

## Notes

- XGBoost is optional at runtime; the remaining models can still be evaluated if it is unavailable.
- The project is intended for educational and portfolio demonstration purposes.
