# AgroVision-ML — Member 2: Multi-Commodity Intelligence Guide

This directory documents the three agricultural commodities supported by **Member 2 (Price Forecasting & Market Volatility Engine)** in the AgroVision-ML platform.

---

## 1. Supported Commodities Overview

Member 2 provides point-in-time predictive modeling across wholesale agricultural markets (APMC Mandis in West Bengal) with clean architectural separation for datasets, feature engineering, model weights, and benchmark scorecards.

| Commodity | Category | Primary Market Scope | Historical Dataset | Mandis | Varieties | Production Price Model | Volatility Classifier | Release Status |
|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|
| [🌾 **Rice**](rice/README.md) | Cereals | West Bengal Rice Mandis | 315,079 rows (2001–2024) | 87 | 34 | Tuned XGBoost Regressor (48 features)<br>+ Direct 14D & 30D Regressors | `v1` Persistence baseline<br>`v2` Normalized XGBoost (35 features)<br>`v3` Shadow 14D (38 features) | **Production Champion** |
| [🌾 **Wheat**](wheat/README.md) | Cereals | West Bengal Wheat Mandis | 64,364 rows (2001–2024) | 38 | 16 | Tuned XGBoost Regressor (20 features) | Class-Balanced XGBoost Classifier (19 features) | **Production Ready** |
| [🥔 **Potato**](potato/README.md) | Vegetables | West Bengal Potato Mandis | 265,047 rows (2002–2024) | 88 | 13 | Tuned XGBoost Regressor (20 features) | Class-Balanced XGBoost Classifier (19 features) | **Production Ready** |

---

## 2. Directory & Storage Separation Structure

All commodity artifacts and datasets are organized under dedicated directories:

```text
member2-price-volatility/
├── commodities/                       ← Commodity documentation guides
│   ├── README.md                      ← Master multi-commodity guide (this file)
│   ├── rice/README.md                 ← Rice architecture, metrics & operational guide
│   ├── wheat/README.md                ← Wheat architecture, metrics & operational guide
│   └── potato/README.md               ← Potato architecture, metrics & operational guide
│
├── artifacts/                         ← Model artifacts & feature configurations
│   ├── rice/                          ← Separated Rice models, configs, residuals & mappings
│   │   ├── rice_price_model.pkl
│   │   ├── rice_price_final_config.json
│   │   ├── rice_price_model_14d.pkl
│   │   ├── rice_price_model_30d.pkl
│   │   ├── rice_volatility_normalized_model.pkl
│   │   ├── rice_group_mappings.json
│   │   └── rice_holdout_residuals*.json
│   ├── wheat/                         ← Separated Wheat models, configs & mappings
│   │   ├── wheat_price_model.pkl
│   │   ├── wheat_volatility_model.pkl
│   │   ├── wheat_feature_config.json
│   │   ├── wheat_volatility_config.json
│   │   └── wheat_group_mappings.json
│   └── potato/                        ← Separated Potato models, configs & mappings
│       ├── potato_price_model.pkl
│       ├── potato_volatility_model.pkl
│       ├── potato_feature_config.json
│       ├── potato_volatility_config.json
│       └── potato_group_mappings.json
│
└── data/
    ├── raw/                           ← Source raw Agmarknet CSVs (rice/, wheat/, potato/)
    └── processed/                     ← Cleaned time-series tables, features & splits
```

---

## 3. How to Train Any Commodity

The unified training entry point (`run.py` or `src/pipeline.py`) supports individual or multi-commodity training:

```bash
# Train Rice pipeline (Price + Volatility)
python member2-price-volatility/run.py --commodity rice

# Train Wheat pipeline
python member2-price-volatility/run.py --commodity wheat

# Train Potato pipeline
python member2-price-volatility/run.py --commodity potato

# Train All Three Commodities
python member2-price-volatility/run.py --commodity all
```

---

## 4. Multi-Commodity Prediction APIs

### Python API (`member2_price_volatility.src.models.predict`)

```python
import pandas as pd
from member2_price_volatility.src.models.predict import predict

# Predict Wheat price and volatility
wheat_history = pd.read_csv("member2-price-volatility/data/processed/wheat/wheat_test.csv").iloc[:35]
res_w = predict(wheat_history, commodity="wheat")
print(f"Wheat Modal Price: Rs. {res_w['predicted_price']}/qtl | Regime: {res_w['volatility']}")

# Predict Potato price and volatility
potato_history = pd.read_csv("member2-price-volatility/data/processed/potato/potato_test.csv").iloc[:35]
res_p = predict(potato_history, commodity="potato")
print(f"Potato Modal Price: Rs. {res_p['predicted_price']}/qtl | Regime: {res_p['volatility']}")

# Predict Rice price and volatility
rice_history = pd.read_csv("member2-price-volatility/data/processed/rice/rice_test.csv").iloc[:35]
res_r = predict(rice_history, commodity="rice")
print(f"Rice Modal Price: Rs. {res_r['predicted_price']}/qtl | Volatility: {res_r['volatility']}")
```

---

## 5. Comparative Commodity Scorecard

| Metric | Rice (48 Features) | Wheat (20 Features) | Potato (20 Features) |
|:---|:---:|:---:|:---:|
| **Training Observations** | 247,198 | 51,491 | 212,037 |
| **Validation / Test Rows** | 20,465 holdout | 12,873 test | 53,010 test |
| **Price Forecasting $R^2$** | **0.9805** | **0.9520** | **0.9240** |
| **Price MAE (Rs./Quintal)** | **Rs. 18.98** | **Rs. 24.50** | **Rs. 32.80** |
| **Price Accuracy Metric ($100 - \text{MAPE}$)** | **99.46%** | **98.70%** | **97.60%** |
| **Volatility Accuracy** | 68.25% (V2 norm) | **63.49%** | 50.88% |
| **Volatility Balanced Accuracy** | 63.20% | **60.37%** | 44.42% |
| **Volatility Macro F1** | 0.6247 | **0.6101** | 0.4610 |
| **Direct 14D/30D Calendar Horizons** | Supported (Production) | Architecture Compatible | Architecture Compatible |
| **Prospective Shadow Engine** | Supported (V3 14D) | Architecture Compatible | Architecture Compatible |

---

## 6. Commodity-Specific Documentation Links

- [🌾 **Rice Deep-Dive Guide**](rice/README.md)
- [🌾 **Wheat Deep-Dive Guide**](wheat/README.md)
- [🥔 **Potato Deep-Dive Guide**](potato/README.md)
