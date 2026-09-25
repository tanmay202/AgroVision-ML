# AgroVision-ML

Multi-commodity agricultural mandi price forecasting and volatility classification pipeline.

## Features

- **Price Forecasting**: XGBoost regressor predicts next-observation market price
- **Volatility Classification**: XGBoost classifier labels market volatility (LOW/MEDIUM/HIGH)
- **Multi-Commodity**: Run for any commodity — `python run.py --commodity tea`
- **Single Command**: Complete pipeline in one step (replaces 13 manual scripts)
- **No Data Leakage**: All features use only past observations

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data
Place your CSV file in `data/raw/` named `{commodity}.csv` (e.g., `tea.csv`).

**Expected columns** (from data.gov.in / Agmarknet):
- `State Name`, `District Name`, `Market Name`, `Variety`, `Group`
- `Arrivals (Tonnes)`
- `Min Price (Rs./Quintal)`, `Max Price (Rs./Quintal)`, `Modal Price (Rs./Quintal)`
- `Reported Date` (format: `dd Mon YYYY`, e.g., `15 Jan 2024`)

### 3. Run the pipeline
```bash
python run.py --commodity tea          # Single commodity
python run.py --commodity onion        # Different commodity
python run.py --commodity all          # All CSVs in data/raw/
python run.py --commodity tea --skip-volatility  # Price model only
```

### 4. Use predictions (for Member 3 / FastAPI)
```python
from src.models.predict import predict
import pandas as pd

# Load a sample row with required features
result = predict(sample_df, commodity="tea")
# {"predicted_price": 8500.0, "volatility": "LOW", "commodity": "tea"}
```

## Project Structure
```
AgroVision-ML/
├── run.py                         ← CLI entry point
├── setup.py                       ← Package setup
├── requirements.txt
├── data/
│   ├── raw/                       ← Your downloaded CSVs
│   └── processed/                 ← Auto-generated per commodity
├── artifacts/                     ← Saved models & configs per commodity
├── src/
│   ├── config.py                  ← Central config (features, paths, columns)
│   ├── pipeline.py                ← Unified pipeline runner
│   ├── data/
│   │   ├── clean_data.py          ← Step 1: Data cleaning
│   │   └── prepare_timeseries.py  ← Step 2: Target creation
│   ├── features/
│   │   ├── create_lag_features.py         ← Step 3a: Price lags
│   │   ├── create_rolling_features.py     ← Step 3b: Rolling stats
│   │   ├── create_date_features.py        ← Step 3c: Calendar features
│   │   ├── create_arrival_features.py     ← Step 3d: Arrival features
│   │   ├── create_pct_change_features.py  ← Step 3e: % changes
│   │   ├── create_volatility_target.py    ← Step 5: Volatility labels
│   │   ├── create_volatility_dataset.py   ← Step 6: Vol train data
│   │   └── create_volatility_test.py      ← Step 7: Vol test data
│   ├── evaluation/
│   │   └── time_split.py          ← Step 4: Per-group time split
│   ├── models/
│   │   ├── xgboost_price.py       ← XGBoost price model
│   │   ├── random_forest_price.py ← RF price model
│   │   ├── baseline.py            ← Naive baseline
│   │   ├── save_final_models.py   ← Model serialization
│   │   └── predict.py             ← Prediction API
│   └── utils/
│       └── __init__.py            ← Shared metrics & utilities
├── tests/
│   ├── test_leakage.py            ← Leakage validation tests
│   └── test_pipeline.py           ← End-to-end pipeline tests
└── memeber1/                      ← Member 1: Yield/Arrival forecasting
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `clean_data.py` | Parse dates, validate prices, remove impossible values |
| 2 | `prepare_timeseries.py` | Create `future_modal_price` target, filter by forecast horizon |
| 3a | `create_lag_features.py` | Price lags (1, 7, 14, 30 observations back) |
| 3b | `create_rolling_features.py` | Rolling mean/std (7, 14, 30 day windows) |
| 3c | `create_date_features.py` | Year, month, day, day_of_week, week_of_year |
| 3d | `create_arrival_features.py` | Arrival lags and rolling means |
| 3e | `create_pct_change_features.py` | Price and arrival % changes |
| 4 | `time_split.py` | Per-group chronological 80/20 split |
| 5 | `create_volatility_target.py` | Data-driven volatility labels |
| 6-7 | `create_volatility_dataset.py/test.py` | Volatility model datasets |
| 8 | `save_final_models.py` | Train final models on all training data |

## Team
- **Member 1**: Yield ML — Tea arrival forecasting (in `memeber1/`)
- **Member 2**: Price ML — Mandi price forecasting & volatility classification (in `src/`)
- **Member 3**: Engineering — Risk engine, API, dashboard
