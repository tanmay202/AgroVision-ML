# AgroVision — Member 1: Yield ML

## Overview
Tea Mandi arrival forecasting pipeline using Agmarknet mandi price/arrival data and XGBoost regression.
Predicts future tea arrivals (tonnes) based on historical price and arrival patterns.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data
Place your tea mandi CSV file in `data/raw/` as `tea_cleaned.csv`.

**Expected columns** (from Agmarknet):
- `State Name`, `District Name`, `Market Name`, `Variety`, `Group`
- `Arrivals (Tonnes)`
- `Min Price (Rs./Quintal)`, `Max Price (Rs./Quintal)`, `Modal Price (Rs./Quintal)`
- `Reported Date` (format: `dd Mon YYYY`, e.g., `15 Jan 2024`)

> **Using different data?** Edit `src/config.py` → `COLUMNS` dict and `RAW_DATA_FILE` path.

### 3. Run the pipeline
```bash
python main.py                  # Full pipeline with XGBoost tuning
python main.py --no-tune        # Skip hyperparameter tuning (faster)
python main.py --predict-only   # Load saved model and predict
```

## Project Structure
```
memeber1/
├── data/raw/           ← Your downloaded CSV (tea_cleaned.csv)
├── data/processed/     ← Cleaned dataset (auto-generated)
├── data/features/      ← Feature matrix (auto-generated)
├── src/
│   ├── config.py           ← Paths, column names & settings
│   ├── data_loader.py      ← Load & validate CSV
│   ├── data_cleaner.py     ← Clean data, remove outliers
│   ├── feature_engineer.py ← Create ML features (lags, rolling, encoding)
│   ├── model_trainer.py    ← Train LR, RF, GB, XGBoost models
│   ├── model_evaluator.py  ← Evaluate & visualize results
│   └── predict.py          ← Predict on new data (for FastAPI)
├── models/             ← Saved trained models (.pkl)
├── outputs/            ← Plots & evaluation reports
├── main.py             ← End-to-end pipeline runner
└── requirements.txt
```

## Connecting to Different Data
If your CSV column names differ from the defaults:
1. Edit `src/config.py` → `COLUMNS` dictionary to match your column names
2. Edit `src/config.py` → `TARGET_COLUMN` to your prediction target
3. Edit `src/config.py` → `RAW_DATA_FILE` to your filename

## Team
- **Member 1 (You)**: Yield ML — Tea arrival forecasting pipeline
- **Member 2**: Price ML — Mandi price forecasting & volatility classification
- **Member 3**: Engineering — Risk engine, API, dashboard
