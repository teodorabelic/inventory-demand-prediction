# Inteligentni sistem za predikciju potraznje i optimizaciju zaliha

Ovaj projekat implementira specifikaciju iz PDF-a i ukljucuje vise pristupa koji se naslanjaju na materijale sa vezbi:

- EDA nad istorijom prodaje
- feature engineering za vremenske serije
- treniranje i poredjenje vise regresionih modela
- KNN regressor kao instance-based pristup
- Gaussian Naive Bayes klasifikaciju nivoa traznje
- KMeans segmentaciju artikala za analizu zaliha
- izbor najboljeg modela na osnovu evaluacije
- analiza trendova prodaje
- optimizacija zaliha i poslovne preporuke
- Streamlit interfejs za pregled rezultata

## Struktura

- `app.py` - Streamlit aplikacija
- `scripts/generate_demo_data.py` - generise demo skup podataka
- `scripts/train_pipeline.py` - trenira modele i cuva rezultate
- `src/` - jezgro sistema
- `data/` - ulazni podaci
- `artifacts/` - sacuvani modeli, metrike i izlazi

## Ocekivani podaci

Primarni ulaz je Kaggle skup `train.csv` iz:

`Store Item Demand Forecasting Challenge`

Ulazni CSV treba da sadrzi kolone:

- `date`
- `store`
- `item`
- `sales`

Kopiraj ga u `data/train.csv`.

Ako skup jos nije dodat, mozes pokrenuti demo varijantu:

```powershell
python scripts/generate_demo_data.py
```

## Instalacija

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Trening

```powershell
python scripts/train_pipeline.py
```

Opcije:

```powershell
python scripts/train_pipeline.py --data data/train.csv --forecast-horizon 30
python scripts/train_pipeline.py --use-demo-data
```

## Pokretanje aplikacije

```powershell
streamlit run app.py
```

## Sta se cuva u `artifacts/`

- `metrics.csv` - rezultati svih modela
- `predictions.csv` - test predikcije najboljeg modela
- `inventory_recommendations.csv` - preporuke za zalihe
- `feature_importance.csv` - znacaj atributa
- `best_model.joblib` - najbolji model
- `training_summary.json` - osnovni metadata izlaz

## Implementirani modeli i metode

- Linear Regression
- Ridge Regression
- KNN Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- MLP Regressor
- XGBoost Regressor, ako je paket dostupan
- Gaussian Naive Bayes za klasifikaciju nivoa traznje
- KMeans za segmentaciju inventara

## Napomena

Ako `xgboost` ili `shap` nisu instalirani, ostatak sistema i dalje radi. Te funkcionalnosti su implementirane kao opcione.
