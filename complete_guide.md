# AgroVision-ML — Complete Technical Guide

**Version:** 2.0 | **Last Updated:** September 2026 | **Dataset:** Agmarknet Tea Mandi Data

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Full Project Structure](#2-full-project-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Data: Download & Format](#4-data-download--format)
5. [Member 1 — Yield/Arrival Model](#5-member-1--yieldarrival-model)
6. [Member 2 — Price & Volatility Model](#6-member-2--price--volatility-model)
7. [Integration: How Both Members Connect](#7-integration-how-both-members-connect)
8. [Member 3 — FastAPI Engineering Guide](#8-member-3--fastapi-engineering-guide)
9. [Bugs Found & Fixed](#9-bugs-found--fixed)
10. [Known Limitations](#10-known-limitations)
11. [Adding a New Commodity / Dataset](#11-adding-a-new-commodity--dataset)
12. [Learning Roadmap](#12-learning-roadmap)
13. [Quick Reference Commands](#13-quick-reference-commands)

---

## 1. Project Overview

AgroVision-ML is a **multi-model agricultural intelligence platform** built for Indian mandi (agricultural marketplace) data. It uses machine learning to answer three key questions about tea market behavior:

| Question | Model Type | Member |
|----------|-----------|--------|
| How much tea will arrive at the mandi? | XGBoost Regression | Member 1 |
| What will the price be? | XGBoost Regression | Member 2 |
| How volatile will the price be? | XGBoost Classification | Member 2 |

**Data Source:** [Agmarknet](https://agmarknet.gov.in/) via [Data.gov.in API](https://data.gov.in/)

**Commodity:** Tea (default) — extensible to onion, potato, and any other mandi commodity

---

## 2. Full Project Structure

```
AgroVision-ML/                           ← Project root
│
├── run_all.py                           ← Unified entry point (runs both pipelines)
├── requirements.txt                     ← All dependencies
├── README.md                            ← Quick start reference
├── complete_guide.md                    ← This document
├── .gitignore                           ← Excludes .pkl, raw CSVs, outputs
│
├── scripts/
│   └── download_data.py                 ← Download data from Data.gov.in API
│
├── shared/
│   ├── __init__.py
│   └── predict_bridge.py                ← Member 3 integration API
│
├── member1-yield-model/                 ═══ MEMBER 1 ═══
│   ├── main.py                          ← Entry point: full pipeline
│   ├── inspect_data.py                  ← Utility: quick CSV inspection
│   ├── requirements.txt                 ← Member 1 specific (subset of root)
│   │
│   ├── data/
│   │   ├── raw/                         ← PUT tea_cleaned.csv HERE
│   │   ├── processed/                   ← Auto: cleaned_dataset.csv
│   │   └── features/                    ← Auto: feature_matrix.csv
│   │
│   ├── models/
│   │   └── yield_model.pkl              ← Auto: saved best model
│   │
│   ├── outputs/                         ← Auto: plots & CSVs
│   │   ├── actual_vs_predicted.png
│   │   ├── residual_analysis.png
│   │   ├── feature_importance.png
│   │   ├── evaluation_report.csv
│   │   └── model_comparison.csv
│   │
│   └── src/                             ← Pipeline source code
│       ├── __init__.py
│       ├── config.py                    ← ⚙ Paths, column names, hyperparams
│       ├── data_loader.py               ← Step 1: Load & validate CSV
│       ├── data_cleaner.py              ← Step 2: Clean data, extract dates
│       ├── feature_engineer.py          ← Step 3: Lag, rolling, encode features
│       ├── model_trainer.py             ← Step 4: Train 4 models, select best
│       ├── model_evaluator.py           ← Step 5: Evaluate, plot, save report
│       └── predict.py                   ← Inference API (for Member 3)
│
└── member2-price-volatility/            ═══ MEMBER 2 ═══
    ├── run.py                           ← Entry point: CLI
    ├── setup.py                         ← Package metadata
    ├── 1.py                             ← Legacy (use scripts/download_data.py)
    │
    ├── data/
    │   ├── raw/                         ← PUT tea.csv HERE
    │   └── processed/                   ← Auto: all intermediate CSVs
    │
    ├── artifacts/                       ← Auto: saved models & configs
    │   ├── tea_price_model.pkl
    │   ├── tea_volatility_model.pkl
    │   ├── tea_feature_config.json
    │   └── tea_volatility_config.json
    │
    ├── tests/
    │   ├── test_leakage.py              ← Tests for data leakage
    │   └── test_pipeline.py             ← Integration tests
    │
    └── src/                             ← Pipeline source code
        ├── __init__.py
        ├── config.py                    ← ⚙ All settings (multi-commodity)
        ├── pipeline.py                  ← Orchestrates full pipeline
        │
        ├── data/                        ← Data processing steps
        │   ├── __init__.py
        │   ├── check_raw_data.py        ← Inspect raw CSV structure
        │   ├── clean_data.py            ← Step 1: Clean & validate
        │   ├── inspect_data.py          ← EDA utilities
        │   └── prepare_timeseries.py    ← Step 2: Create future price target
        │
        ├── features/                    ← Feature engineering steps
        │   ├── __init__.py
        │   ├── create_lag_features.py         ← Step 3a: Price lags (1/7/14/30)
        │   ├── create_rolling_features.py     ← Step 3b: Rolling mean/std
        │   ├── create_date_features.py        ← Step 3c: Year/month/weekday
        │   ├── create_arrival_features.py     ← Step 3d: Arrival lags
        │   ├── create_pct_change_features.py  ← Step 3e: % change features
        │   ├── create_volatility_target.py    ← Step 4: Label LOW/MEDIUM/HIGH
        │   ├── create_volatility_dataset.py   ← Step 5: Build volatility train set
        │   ├── create_volatility_test.py      ← Step 6: Build volatility test set
        │   └── review_features.py             ← Utility: display feature stats
        │
        ├── models/                      ← Model training scripts
        │   ├── __init__.py
        │   ├── baseline.py                     ← Naive baseline
        │   ├── random_forest_price.py          ← Random Forest price model
        │   ├── xgboost_price.py                ← XGBoost price model (primary)
        │   ├── xgboost_volatility.py           ← XGBoost volatility trainer
        │   ├── volatility_classifier.py        ← Volatility classification
        │   ├── save_final_models.py             ← Export production .pkl files
        │   ├── predict.py                       ← Inference API (for Member 3)
        │   └── test_prediction.py              ← Smoke test predictions
        │
        ├── evaluation/                  ← Evaluation & validation
        │   ├── __init__.py
        │   ├── time_split.py            ← Chronological train/test split
        │   ├── error_analysis.py        ← Residual & error analysis
        │   ├── check_price_anomalies.py ← Detect anomalous prices
        │   └── final_validation.py      ← Leakage & data integrity checks
        │
        └── utils/                       ← Shared utilities
            └── __init__.py              ← Metrics: RMSE, MAE, R², MAPE
```

---

## 3. Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip (comes with Python)

### Step 1: Clone / Open Project
```bash
# If using git
git clone <your-repo-url>
cd AgroVision-ML

# Or just navigate to the folder
cd D:\MCA\SEM3\AgroVision-ML
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

> [!TIP]
> If you only want to run one member's code, you can use their local `requirements.txt`:
> ```bash
> pip install -r member1-yield-model/requirements.txt
> ```

### Step 4: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, xgboost; print('All packages OK')"
```

---

## 4. Data: Download & Format

### Option A: Download Automatically (Recommended)
```bash
python scripts/download_data.py --commodity tea
```

This fetches from **Data.gov.in Agmarknet API** and saves to:
- `member1-yield-model/data/raw/tea_cleaned.csv`
- `member2-price-volatility/data/raw/tea.csv`

```bash
# Download other commodities
python scripts/download_data.py --commodity onion
python scripts/download_data.py --commodity potato --limit 100000
```

### Option B: Manual CSV Placement

If you already have a CSV, place it as:
```
member1-yield-model/data/raw/tea_cleaned.csv
member2-price-volatility/data/raw/tea.csv
```

### Expected CSV Column Format (Agmarknet)

| Column | Example Values | Required By |
|--------|---------------|------------|
| `State Name` | `West Bengal`, `Tamil Nadu` | Both |
| `District Name` | `Darjeeling` | Both |
| `Market Name` | `Siliguri` | Both |
| `Variety` | `CTC Dust`, `Orthodox` | Both |
| `Group` | `Beverages` | Member 1 |
| `Arrivals (Tonnes)` | `50.5` | Both |
| `Min Price (Rs./Quintal)` | `180` | Both |
| `Max Price (Rs./Quintal)` | `320` | Both |
| `Modal Price (Rs./Quintal)` | `250` | Both |
| `Reported Date` | `15 Jan 2024` | Both |

> [!IMPORTANT]
> Column names must match exactly (including spaces and capitalization), OR you must update `config.py` in both member folders. See [Section 11](#11-adding-a-new-commodity--dataset) for column mapping instructions.

---

## 5. Member 1 — Yield/Arrival Model

### What It Does
Predicts **how many tonnes of tea will arrive** at Indian mandis on a future date, using historical price and arrival patterns.

### Pipeline Overview
```
CSV → Load → Clean → Feature Engineering → Train 4 Models → Select Best → Evaluate → Save .pkl
```

### Step-by-Step Pipeline

| Step | File | Function | Description |
|------|------|----------|-------------|
| 1 | `src/data_loader.py` | `load_all_data()` | Load & validate CSV |
| 2 | `src/data_cleaner.py` | `clean_and_save()` | Parse dates, remove outliers, extract temporal columns |
| 3 | `src/feature_engineer.py` | `engineer_features()` | Create lag, rolling, price, temporal, encoded features |
| 4 | `src/model_trainer.py` | `train_models()` | Train LR, RF, GB, XGBoost; select best by R² |
| 5 | `src/model_evaluator.py` | `evaluate_model()` | Compute RMSE/MAE/R²/MAPE; generate plots |

### How to Run
```bash
cd member1-yield-model

python main.py                  # Full pipeline (XGBoost tuning: ~5-10 min)
python main.py --no-tune        # Fast mode (~1-2 min, fixed hyperparams)
python main.py --predict-only   # Load saved model without training
```

### Configuration (`src/config.py`)

```python
# Change data file
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "tea_cleaned.csv")

# Change target variable
TARGET_COLUMN = "Arrivals (Tonnes)"

# Change column mapping (if your CSV uses different names)
COLUMNS = {
    "state":       "State Name",
    "district":    "District Name",
    "market":      "Market Name",
    "variety":     "Variety",
    "group":       "Group",
    "arrivals":    "Arrivals (Tonnes)",
    "min_price":   "Min Price (Rs./Quintal)",
    "max_price":   "Max Price (Rs./Quintal)",
    "modal_price": "Modal Price (Rs./Quintal)",
    "date":        "Reported Date",
}
```

### Features Engineered

| Category | Feature Names | Count |
|----------|--------------|-------|
| **Lag (Arrivals)** | `Arrival_Lag_1`, `Arrival_Lag_3`, `Arrival_Lag_7` | 3 |
| **Lag (Price)** | `Price_Lag_1`, `Price_Lag_3`, `Price_Lag_7` | 3 |
| **Rolling Mean** | `Arrival_RollMean_7/14/30`, `Price_RollMean_7/14/30` | 6 |
| **Rolling Std** | `Price_Volatility_14`, `Arrival_Volatility_14` | 2 |
| **Price Derived** | `Price_Spread`, `Price_Ratio`, `Log_Modal_Price`, `Price_Change`, `Price_Pct_Change` | 5 |
| **Temporal** | `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear`, `Quarter`, `Month_Sin/Cos`, `DayOfWeek_Sin/Cos`, `Year_Trend`, `Is_Weekend` | 11 |
| **Encoded** | `State_Name_Encoded`, `District_Name_Encoded`, `Market_Name_Encoded`, `Variety_Encoded` | 4 |

### Models Trained (Auto-select Best)

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline |
| Random Forest | `n_estimators=200`, `max_depth=8`, regularized |
| Gradient Boosting | `n_estimators=200`, `max_depth=4`, `lr=0.1` |
| XGBoost | GridSearchCV tuning (configurable) |

### Key Design Decisions
- **Log-transform target:** `y_train = np.log1p(arrivals)` — handles right-skewed distribution
- **Inverse transform predictions:** `prediction = np.expm1(raw_pred)` — correct output scale
- **Time-based split:** Last 20% chronologically (NOT random) — realistic for time-series
- **L1/L2 regularization** on XGBoost to prevent overfitting

### Output Files

| File | Description |
|------|-------------|
| `models/yield_model.pkl` | Dict: `{model, feature_names, use_log_target, model_name}` |
| `outputs/actual_vs_predicted.png` | Scatter: actual vs predicted arrivals |
| `outputs/residual_analysis.png` | Residuals vs predicted + distribution |
| `outputs/feature_importance.png` | Top 15 feature importances |
| `outputs/evaluation_report.csv` | RMSE, MAE, R², MAPE for best model |
| `outputs/model_comparison.csv` | All 4 models compared |

---

## 6. Member 2 — Price & Volatility Model

### What It Does
1. **Price Forecasting** — Predicts the future modal price (₹/Quintal) using XGBoost regression
2. **Volatility Classification** — Labels price volatility as `LOW`, `MEDIUM`, or `HIGH` using XGBoost classifier

### Pipeline Overview
```
CSV → Clean → Time-Series Prep → Feature Engineering (5 steps) → Train/Test Split
    → Price Model → Volatility Labels → Volatility Model → Save .pkl Artifacts
```

### How to Run
```bash
cd member2-price-volatility

# Full pipeline (tea, default)
python run.py

# Specific commodity
python run.py --commodity tea
python run.py --commodity onion

# All commodities in data/raw/ at once
python run.py --commodity all

# Price model only (skip volatility)
python run.py --commodity tea --skip-volatility
```

### Multi-Commodity Support

Member 2 supports **any commodity** without code changes:
```bash
# Place data:  member2-price-volatility/data/raw/onion.csv
python run.py --commodity onion
# Saves:       artifacts/onion_price_model.pkl
#              artifacts/onion_volatility_model.pkl
```

You can also set commodity via environment variable:
```bash
set AGROVISION_COMMODITY=onion  # Windows
export AGROVISION_COMMODITY=onion  # Mac/Linux
python run.py
```

### Pipeline Steps in Detail

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/data/clean_data.py` | Parse dates (auto-detect format), remove negatives/zeros, fill NaN |
| 2 | `src/data/prepare_timeseries.py` | Create `future_modal_price` target via `shift(-1)` |
| 3a | `src/features/create_lag_features.py` | Price lags: 1, 7, 14, 30 days |
| 3b | `src/features/create_rolling_features.py` | Rolling mean/std: 7, 14, 30 days |
| 3c | `src/features/create_date_features.py` | Year, month, day, day-of-week, week-of-year |
| 3d | `src/features/create_arrival_features.py` | Arrival lag/rolling features |
| 3e | `src/features/create_pct_change_features.py` | Price & arrival % change |
| 4 | `src/evaluation/time_split.py` | Chronological 80/20 train/test split |
| 5 | `src/models/xgboost_price.py` | Train & evaluate price XGBoost |
| 6 | `src/features/create_volatility_target.py` | Label rows: LOW / MEDIUM / HIGH |
| 7 | `src/features/create_volatility_dataset.py` | Build volatility train dataset |
| 8 | `src/features/create_volatility_test.py` | Build volatility test dataset |
| 9 | `src/models/save_final_models.py` | Train final models on all data; save .pkl |

### Configuration (`src/config.py`)

```python
# Multi-commodity: change at runtime
set_commodity("onion")          # All paths auto-update

# Column names (edit if your CSV differs)
DATE_COLUMN = "Reported Date"
PRICE_COLUMN = "Modal Price (Rs./Quintal)"
ARRIVAL_COLUMN = "Arrivals (Tonnes)"
GROUP_COLUMNS = ["Market Name", "Variety"]

# Forecast horizon
FORECAST_HORIZON_MAX_DAYS = 7   # Drop rows where next record > 7 days away
```

### Leakage-Free Feature Design

> [!IMPORTANT]
> Member 2 explicitly fixes **data leakage** in feature selection. The following are **intentionally excluded** from the price model:

| Excluded Feature | Why Excluded |
|-----------------|-------------|
| `Modal Price (Rs./Quintal)` | This IS the current price — using it to predict future price is near-direct leakage |
| `Min Price (Rs./Quintal)` | Same row current value |
| `Max Price (Rs./Quintal)` | Same row current value |
| `arrival_change` | Uses current-row arrivals |
| `price_pct_change` | Excluded from volatility model (derived from same signal as target) |

### Price Model Features

| Feature | Safe? | Description |
|---------|-------|-------------|
| `lag_1`, `lag_7`, `lag_14`, `lag_30` | ✅ | Historical price N records ago |
| `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30` | ✅ | Rolling average (shifted) |
| `rolling_std_7`, `rolling_std_14` | ✅ | Rolling std deviation (shifted) |
| `year`, `month`, `day`, `day_of_week`, `week_of_year` | ✅ | Date components (known at prediction time) |
| `arrival_lag_1`, `arrival_lag_7` | ✅ | Historical arrivals |
| `arrival_rolling_mean_7`, `arrival_rolling_mean_14` | ✅ | Rolling arrival mean (shifted) |
| `price_pct_change` | ✅ (price model) | Previous row price change |
| `arrival_pct_change` | ✅ | Previous row arrival change |

### Volatility Classification

Volatility is defined as the **standard deviation of price % changes** over a rolling window per market+variety group:

```
LOW    : std(price_pct_change) < 33rd percentile
MEDIUM : 33rd percentile ≤ std ≤ 66th percentile
HIGH   : std(price_pct_change) > 66th percentile
```

Encoded as: `LOW=0`, `MEDIUM=1`, `HIGH=2`

### Output Artifacts

| File | Description |
|------|-------------|
| `artifacts/tea_price_model.pkl` | XGBoost regressor for price |
| `artifacts/tea_volatility_model.pkl` | XGBoost classifier for volatility |
| `artifacts/tea_feature_config.json` | Feature list + model metadata (JSON) |
| `artifacts/tea_volatility_config.json` | Volatility thresholds |

### Running Tests
```bash
cd member2-price-volatility
pytest tests/test_leakage.py     # Verify no data leakage
pytest tests/test_pipeline.py    # Integration tests
```

---

## 7. Integration: How Both Members Connect

### Architecture Diagram

```
                        ┌─────────────────────────────────────┐
                        │   AgroVision-ML Project Root        │
                        ├──────────────┬──────────────────────┤
                        │              │                      │
              ┌─────────▼──────┐  ┌───▼────────────────────┐ │
              │  Member 1      │  │  Member 2               │ │
              │  Yield Model   │  │  Price + Volatility     │ │
              │                │  │                         │ │
              │  yield_model   │  │  tea_price_model.pkl    │ │
              │  .pkl          │  │  tea_volatility_model   │ │
              │                │  │  .pkl                   │ │
              └────────┬───────┘  └──────────────┬──────────┘ │
                       │                          │            │
                       └──────────────┬───────────┘            │
                                      │                        │
                             ┌────────▼────────┐              │
                             │  shared/         │              │
                             │  predict_bridge  │              │
                             │  .py             │              │
                             └────────┬────────┘              │
                                      │                        │
                             ┌────────▼────────┐              │
                             │  Member 3        │              │
                             │  FastAPI Service │              │
                             │  (separate repo) │              │
                             └─────────────────┘              │
                        └─────────────────────────────────────┘
```

### Data Flow

```
Same CSV data →  Member 1: Predicts ARRIVALS (tonnes)
              →  Member 2: Predicts PRICE (₹/Quintal) + VOLATILITY (LOW/MEDIUM/HIGH)
```

Both outputs feed Member 3's risk engine and dashboard.

### Shared Prediction Bridge

The `shared/predict_bridge.py` module wraps both predict APIs:

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

predictor = AgroVisionPredictor()

# Prepare feature data for Member 2
data = pd.DataFrame([{
    "lag_1": 240.0,
    "lag_7": 235.0,
    "lag_14": 238.0,
    "lag_30": 232.0,
    "rolling_mean_7": 241.5,
    "rolling_mean_14": 239.0,
    "rolling_mean_30": 236.5,
    "rolling_std_7": 5.2,
    "rolling_std_14": 6.1,
    "year": 2024,
    "month": 3,
    "day": 15,
    "day_of_week": 4,
    "week_of_year": 11,
    "arrival_lag_1": 52.0,
    "arrival_lag_7": 48.0,
    "arrival_rolling_mean_7": 50.5,
    "arrival_rolling_mean_14": 49.2,
    "price_pct_change": 0.02,
    "arrival_pct_change": 0.08,
}])

# ─── Full prediction (both models) ───────────────────────────
result = predictor.predict_all(data, commodity="tea")
print(result)
# {
#   "yield_arrivals_tonnes":     52.3,
#   "predicted_price_rs_quintal": 248.50,
#   "volatility":                 "LOW",
#   "commodity":                  "tea"
# }

# ─── Member 1 only (yield) ────────────────────────────────────
yield_input = data.iloc[0].to_dict()
yield_result = predictor.predict_yield(yield_input)
print(yield_result)  # {"yield_arrivals_tonnes": 52.3}

# ─── Member 2 only (price + volatility) ───────────────────────
price_result = predictor.predict_price(data, commodity="tea")
print(price_result)
# {"predicted_price_rs_quintal": 248.50, "volatility": "LOW", "commodity": "tea"}

# ─── Get required feature lists ───────────────────────────────
features = predictor.get_required_features()
print(features["price_features"])    # List of 20 price model features
print(features["yield_features"])    # List of yield model features
```

---

## 8. Member 3 — FastAPI Engineering Guide

### What You Need

1. **Both ML pipelines trained** — run `python run_all.py` from project root
2. **Models exist:**
   - `member1-yield-model/models/yield_model.pkl`
   - `member2-price-volatility/artifacts/tea_price_model.pkl`
   - `member2-price-volatility/artifacts/tea_volatility_model.pkl`

### Minimal FastAPI Integration

```python
# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.predict_bridge import AgroVisionPredictor

app = FastAPI(title="AgroVision API", version="2.0")
predictor = AgroVisionPredictor()  # Loads models once at startup


class PredictionRequest(BaseModel):
    commodity: str = "tea"
    lag_1: float
    lag_7: float
    lag_14: float
    lag_30: float
    rolling_mean_7: float
    rolling_mean_14: float
    rolling_mean_30: float
    rolling_std_7: float
    rolling_std_14: float
    year: int
    month: int
    day: int
    day_of_week: int
    week_of_year: int
    arrival_lag_1: float
    arrival_lag_7: float
    arrival_rolling_mean_7: float
    arrival_rolling_mean_14: float
    price_pct_change: float = 0.0
    arrival_pct_change: float = 0.0


@app.post("/predict")
def predict(request: PredictionRequest):
    """Get price, volatility, and yield predictions."""
    try:
        data = pd.DataFrame([request.dict(exclude={"commodity"})])
        result = predictor.predict_all(data, commodity=request.commodity)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not found: {e}")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Feature error: {e}")


@app.get("/features")
def get_features():
    """Return the list of required input features."""
    return predictor.get_required_features()


@app.get("/health")
def health():
    return {"status": "ok", "commodities_supported": ["tea", "onion"]}
```

### Running the API
```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload --port 8000
```

### API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Full prediction: yield + price + volatility |
| `/features` | GET | List required input features |
| `/health` | GET | Health check |

### Recommended Request Example (curl)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "commodity": "tea",
    "lag_1": 240, "lag_7": 235, "lag_14": 238, "lag_30": 232,
    "rolling_mean_7": 241, "rolling_mean_14": 239, "rolling_mean_30": 236,
    "rolling_std_7": 5.2, "rolling_std_14": 6.1,
    "year": 2024, "month": 3, "day": 15, "day_of_week": 4, "week_of_year": 11,
    "arrival_lag_1": 52, "arrival_lag_7": 48,
    "arrival_rolling_mean_7": 50, "arrival_rolling_mean_14": 49,
    "price_pct_change": 0.02, "arrival_pct_change": 0.08
  }'
```

Expected response:
```json
{
  "yield_arrivals_tonnes": 52.3,
  "predicted_price_rs_quintal": 248.50,
  "volatility": "LOW",
  "commodity": "tea"
}
```

---

## 9. Bugs Found & Fixed

### 🔴 Critical (Already Fixed in Current Codebase)

#### Bug 1 — Model Prediction Crash (`member1/src/predict.py`)
**Problem:** `model_trainer.py` saves model as a dict `{"model": estimator, "feature_names": [...]}`, but `predict.py` called `.predict()` on the dict directly → `AttributeError`.

**Fix:** Extract `model_info["model"]` before calling `.predict()`. Backward compat for raw model objects.

---

#### Bug 2 — Missing Log Inverse Transform (`member1/src/predict.py`)
**Problem:** Model trained on `log1p(y)` but predictions returned raw log values (~2.5) instead of actual tonnes (~50).

**Fix:** Added `np.expm1(prediction)` when `use_log_target=True`.

---

#### Bug 3 — Evaluation on Log Scale (`member1/src/model_evaluator.py`)
**Problem:** RMSE/MAE/R² computed on log values → falsely optimistic metrics (RMSE=0.3 vs actual RMSE=15.0).

**Fix:** Inverse transform before computing all metrics and generating all plots.

---

#### Bug 4 — Infinity Drops Valid Data (`member2/src/features/create_pct_change_features.py`)
**Problem:** Division by zero → `inf` → replaced with `pd.NA` → `dropna()` removed valid rows silently.

**Fix:** Replace `inf` with `0.0` (zero % change is the correct interpretation).

---

#### Bug 5 — Hardcoded Data Path (`member1/inspect_data.py`)
**Problem:** Hardcoded `'data/raw/tea_cleaned.csv'` instead of using config.

**Fix:** Import `RAW_DATA_FILE` from config.

---

#### Bug 6 — Date Format Lock-in (`member2/src/data/clean_data.py`)
**Problem:** Hardcoded `format="%d %b %Y"` → any CSV with `YYYY-MM-DD` format would cause 100% NaT.

**Fix:** Fallback auto-detection if >50% of dates fail explicit parsing.

---

#### Bug 7 — README Mismatch (`member1/README.md`)
**Problem:** Old README instructed placing `crop_production.csv`, `weather_data.csv`, etc. — code only uses tea mandi data.

**Fix:** Rewrote README to accurately describe the tea mandi pipeline.

---

## 10. Known Limitations

These are design trade-offs / limitations (not bugs). Be aware of them:

| # | Issue | Module | Impact | Mitigation |
|---|-------|--------|--------|-----------|
| 1 | **Forecast horizon mixing** | `member2/src/data/prepare_timeseries.py` | `shift(-1)` predicts "next record" regardless of gap (1 day or 3 months). The model sees mixed horizons. | Add `FORECAST_HORIZON_MAX_DAYS` filter (config exists but verify it's applied consistently) |
| 2 | **Aggressive NaN dropping (Member 1)** | `member1/src/model_trainer.py` | Drops all rows with ANY NaN in features → loses first 30 records per group. XGBoost handles NaN natively. | Could pass `min_periods=1` to rolling or let XGBoost handle NaN |
| 3 | **Hardcoded anomaly threshold** | `member2/src/evaluation/check_price_anomalies.py` | `threshold=100000` is arbitrary. Onion prices are very different from tea prices. | Make threshold commodity-specific in config |
| 4 | **Inconsistent groupby** | Multiple `member2/src/` files | Some use `config.GROUP_COLUMNS`, others hardcode `["Market Name", "Variety"]` | Standardize all to `config.GROUP_COLUMNS` |
| 5 | **Volatility labels bypass config** | `member2/src/features/create_volatility_target.py` | Hardcodes `"LOW"`, `"MEDIUM"`, `"HIGH"` strings instead of using `config.VOLATILITY_CLASSES` | Minor refactor to use `VOLATILITY_CLASSES` |
| 6 | **Single-row std edge case** | `member1/src/data_cleaner.py` | Groups with exactly 1 row produce `NaN` std — caught, but risky if many such groups exist | Already handled, but verify on new datasets |
| 7 | **Model cache not thread-safe** | `member2/src/models/predict.py` | `_model_cache` is a module-level dict — not safe in multi-threaded FastAPI (multiple workers) | Use `threading.Lock()` or per-request loading for production |

---

## 11. Adding a New Commodity / Dataset

> [!IMPORTANT]
> Both members expect **Agmarknet-format** CSV data. If your new CSV has the same columns, just drop it in. If columns differ, follow the steps below.

### Step 1: Check Your CSV Structure
```python
import pandas as pd
df = pd.read_csv("your_new_data.csv")
print(df.columns.tolist())
print(df.dtypes)
print(df.head())
```

### Step 2: Map Your Columns

Your CSV needs these types of data (column names can differ):

| Required Data | Default Column Name |
|---------------|---------------------|
| Date | `Reported Date` |
| Modal Price | `Modal Price (Rs./Quintal)` |
| Min Price | `Min Price (Rs./Quintal)` |
| Max Price | `Max Price (Rs./Quintal)` |
| Arrivals/Volume | `Arrivals (Tonnes)` |
| Market name | `Market Name` |
| Variety/Category | `Variety` |

### Step 3: Update Configs

**For Member 1** — Edit `member1-yield-model/src/config.py`:
```python
# 1. Change the raw data filename
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "your_commodity_data.csv")

# 2. Map YOUR column names
COLUMNS = {
    "state":       "Your State Column",
    "district":    "Your District Column",
    "market":      "Your Market Column",
    "variety":     "Your Variety Column",
    "group":       "Your Group Column",    # Remove if not present
    "arrivals":    "Your Arrivals Column",
    "min_price":   "Your Min Price Column",
    "max_price":   "Your Max Price Column",
    "modal_price": "Your Modal Price Column",
    "date":        "Your Date Column",
}

# 3. Set prediction target
TARGET_COLUMN = "Your Arrivals Column"
```

**For Member 2** — Edit `member2-price-volatility/src/config.py`:
```python
# Column names (change to match your CSV)
DATE_COLUMN    = "Your Date Column"
PRICE_COLUMN   = "Your Modal Price Column"
MIN_PRICE_COLUMN = "Your Min Price Column"
MAX_PRICE_COLUMN = "Your Max Price Column"
ARRIVAL_COLUMN = "Your Arrivals Column"
GROUP_COLUMNS  = ["Your Market Column", "Your Variety Column"]
```

### Step 4: Place Your Data
```
# For Member 1:
member1-yield-model/data/raw/your_commodity.csv

# For Member 2 (filename = commodity name):
member2-price-volatility/data/raw/your_commodity.csv
```

### Step 5: Run Pipelines
```bash
# Member 1 (update config first)
cd member1-yield-model
python main.py --no-tune        # Fast first run to verify

# Member 2 (commodity flag auto-sets paths)
cd member2-price-volatility
python run.py --commodity your_commodity
```

> [!WARNING]
> Always run `python inspect_data.py` (Member 1) or `python src/data/check_raw_data.py` (Member 2) **before** training to verify your CSV columns match the config. Mismatched column names cause silent NaN failures.

---

## 12. Learning Roadmap

If you're building or understanding this project from scratch:

### Phase 1 — Python Foundations (1–2 weeks)
- [ ] Variables, loops, functions, classes, file I/O
- [ ] Virtual environments (`pip install`, `requirements.txt`)
- [ ] Git basics (`commit`, `push`, `branch`, `merge`)
- [ ] Terminal / command line

**Resources:** Python.org tutorial, freeCodeCamp Python, Git official docs

---

### Phase 2 — Data Science Libraries (2–3 weeks)
- [ ] **NumPy:** Arrays, `np.log1p()`, `np.expm1()`, broadcasting
- [ ] **Pandas:** `read_csv`, `groupby`, `shift()`, `rolling()`, `merge`, datetime parsing
- [ ] **Matplotlib/Seaborn:** Scatter, histogram, bar chart, saving figures
- [ ] Data cleaning: NaN, outliers, duplicates, type conversion

**Resources:** Kaggle's Pandas course, "Python for Data Analysis" (Wes McKinney)

---

### Phase 3 — Machine Learning (3–4 weeks)
- [ ] **Scikit-learn:** `fit()`, `predict()`, `score()`, train/test split
- [ ] **Regression:** Linear Regression, Random Forest, Gradient Boosting
- [ ] **Classification:** Logistic Regression, Random Forest Classifier
- [ ] **Metrics:** RMSE, MAE, R², MAPE, Accuracy, F1, Confusion Matrix
- [ ] **Cross-validation:** K-Fold, GridSearchCV for hyperparameter tuning
- [ ] **XGBoost:** `XGBRegressor`, `XGBClassifier`, parameter tuning
- [ ] **Time-series:** Why not random split, look-ahead bias, chronological splitting

**Resources:** Scikit-learn docs, Andrew Ng's ML course, Kaggle ML intro

---

### Phase 4 — ML Pipeline Design (1–2 weeks)
- [ ] Project structure: `src/`, `data/`, `models/`, `config.py`
- [ ] Pipeline pattern: Load → Clean → Features → Train → Evaluate
- [ ] Configuration management: centralizing paths, column names, hyperparameters
- [ ] Model serialization: `joblib.dump()` / `joblib.load()`
- [ ] Target transformations: log1p for skewed data, remembering to expm1 predictions
- [ ] Feature importance analysis
- [ ] Error analysis and debugging ML models

**Resources:** Scikit-learn Pipeline docs, "Hands-On ML" (Aurélien Géron)

---

### Phase 5 — Domain Knowledge (1 week)
- [ ] **Agmarknet data:** What mandi price data looks like, what arrivals mean
- [ ] Agricultural market concepts: arrivals, modal/min/max prices, seasonality
- [ ] **Volatility classification:** Price stability as LOW/MEDIUM/HIGH and why it matters
- [ ] Time-series forecasting for agricultural commodities
- [ ] Indian agricultural market dynamics, crop seasonality

**Resources:** Agmarknet website, ICAR publications, RBI agricultural reports

---

### Phase 6 — Engineering & Deployment (2–3 weeks)
- [ ] **FastAPI:** REST APIs, request/response models, middleware, dependencies
- [ ] **Model serving:** Loading `.pkl` files, prediction endpoints, caching
- [ ] **Docker:** Containerizing the ML service, environment reproducibility
- [ ] **Frontend:** Streamlit or React dashboard for visualizations
- [ ] **CI/CD:** Automated testing and deployment pipelines

**Resources:** FastAPI docs, Streamlit docs, Docker getting started

---

### Key Concepts This Project Teaches

| Concept | Where | Why It Matters |
|---------|-------|----------------|
| Log transformation | `member1/src/model_trainer.py` | Handles skewed target distributions |
| Inverse transform | `member1/src/predict.py`, `model_evaluator.py` | Returns predictions in original scale |
| Time-based split | Both members | Prevents future data leaking into training |
| Lag features | Both members | Captures temporal dependencies |
| Rolling statistics | Both members | Captures trends and volatility signal |
| Cyclical encoding | `member1/src/feature_engineer.py` | Makes month/day-of-week features circular |
| Label encoding | `member1/src/feature_engineer.py` | Converts categorical strings to numbers |
| GridSearchCV | `member1/src/model_trainer.py` | Automated hyperparameter tuning |
| Model serialization | Both members | Saving models for production use |
| Feature importance | Both members | Understanding what drives predictions |
| Data leakage prevention | `member2/src/config.py` | Ensures honest model evaluation |
| Multi-commodity support | `member2/src/config.py` | Making pipelines commodity-agnostic |
| Model caching | `member2/src/models/predict.py` | Efficient production inference |

---

## 13. Quick Reference Commands

### One-Time Setup
```bash
# Install all dependencies (from project root)
pip install -r requirements.txt

# Download tea data
python scripts/download_data.py --commodity tea
```

### Run Pipelines
```bash
# Both pipelines together
python run_all.py
python run_all.py --no-tune          # Fast mode
python run_all.py --member 1         # Only Member 1
python run_all.py --member 2         # Only Member 2
python run_all.py --commodity onion  # Member 2 with onion

# Member 1 individually
cd member1-yield-model
python main.py                       # Full (with tuning ~5-10 min)
python main.py --no-tune             # Fast (~1-2 min)
python main.py --predict-only        # Just load saved model

# Member 2 individually
cd member2-price-volatility
python run.py --commodity tea
python run.py --commodity all        # All CSVs in data/raw/
python run.py --skip-volatility      # Price model only
```

### Verify Outputs
```bash
# Check Member 1 outputs (from project root)
dir member1-yield-model\models\yield_model.pkl
dir member1-yield-model\outputs\

# Check Member 2 artifacts
dir member2-price-volatility\artifacts\

# Run Member 2 tests
cd member2-price-volatility
pytest tests/

# Test Member 2 predictions manually
python src/models/test_prediction.py
```

### Inspect Data Before Running
```bash
# Member 1 data inspection
cd member1-yield-model
python inspect_data.py

# Member 2 data inspection
cd member2-price-volatility
python src/data/check_raw_data.py
python src/data/inspect_data.py
```

### Troubleshooting
```bash
# If imports fail — always run from correct directory:
cd member1-yield-model && python main.py       # Member 1
cd member2-price-volatility && python run.py   # Member 2
cd AgroVision-ML && python run_all.py          # Root orchestrator

# Check Python and package versions
python --version
pip show pandas numpy scikit-learn xgboost

# Re-install dependencies
pip install -r requirements.txt --upgrade
```

---

> [!TIP]
> **When adding new data:** Always run `inspect_data.py` or `check_raw_data.py` first to verify your CSV columns match what the config expects. Fix column mappings in `config.py` BEFORE running the pipeline.

> [!CAUTION]
> **Never commit `.pkl` model files or raw CSV data to Git.** The `.gitignore` excludes these, but double-check before `git push`. Model files are 1–3 MB each and data CSVs can be 10–100 MB.

> [!NOTE]
> **Model cache thread safety:** In production FastAPI with multiple workers, the `_model_cache` in `member2/src/models/predict.py` is not thread-safe. For production, either use `threading.Lock()` around cache access or load models once at startup using FastAPI's `lifespan` context.
