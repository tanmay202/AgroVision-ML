# AgroVision — Member 1: Yield ML

## Overview
Crop-yield forecasting pipeline for Indian agriculture using multi-source data (yield, weather, NDVI, soil) and XGBoost regression.

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data
Place your downloaded CSV files in `data/raw/`:
- `crop_production.csv` — Crop yield data
- `weather_data.csv` — Weather data
- `ndvi_data.csv` — NDVI satellite data
- `soil_data.csv` — Soil data (optional)

### 3. Run the pipeline
```bash
python main.py
```

## Project Structure
```
AgroVision/
├── data/raw/           ← Your downloaded CSVs
├── data/processed/     ← Cleaned & merged data (auto-generated)
├── data/features/      ← Feature matrix (auto-generated)
├── src/
│   ├── config.py           ← Paths & settings
│   ├── data_loader.py      ← Load & validate CSVs
│   ├── data_cleaner.py     ← Clean & merge data
│   ├── feature_engineer.py ← Create ML features
│   ├── model_trainer.py    ← Train XGBoost model
│   ├── model_evaluator.py  ← Evaluate & visualize
│   └── predict.py          ← Predict on new data
├── models/             ← Saved trained models
├── outputs/            ← Plots & reports
├── main.py             ← End-to-end pipeline
└── requirements.txt
```

## Connecting Your Data
If your CSV column names differ from the defaults, edit `src/data_loader.py` column mappings.

## Team
- **Member 1 (You)**: Yield ML — This pipeline
- **Member 2**: Price ML — Mandi price forecasting
- **Member 3**: Engineering — Risk engine, API, dashboard
