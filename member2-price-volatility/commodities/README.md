# AgroVision-ML — Member 2: Multi-Commodity Intelligence Guide

This directory organizes and documents all agricultural commodities supported by **Member 2 (Price Forecasting & Market Volatility Engine)** in the AgroVision-ML platform.

---

## 1. Supported Commodities Overview

Member 2 supports multi-commodity modeling across Indian wholesale agricultural markets (APMC Mandis in West Bengal) with clean architectural separation for data, models, configurations, and evaluation benchmarks.

| Commodity | Category | Primary Focus | Training Dataset | Mandis | Varieties | Production Price Model | Volatility Classifier | Release Status |
|:---|:---|:---|:---|:---:|:---:|:---|:---|:---:|
| [🌾 **Rice**](rice/README.md) | Cereals | West Bengal Rice Mandis | 315,079 rows (2001–2024) | 87 | 34 | Tuned XGBoost Regressor (48 features)<br>+ Direct 14D & 30D Regressors | `v1` Persistence baseline<br>`v2` Normalized XGBoost (35 features)<br>`v3` Shadow 14D (38 features) | **Production Champion** |
| [🌾 **Wheat**](wheat/README.md) | Cereals | West Bengal Wheat Mandis | 64,364 rows (2001–2024) | 38 | 16 | XGBoost Regressor (20 features) | Class-Balanced XGBoost Classifier (19 features) | **Production Ready** |
| [🥔 **Potato**](potato/README.md) | Vegetables | West Bengal Potato Mandis | 265,047 rows (2002–2024) | 88 | 13 | XGBoost Regressor (20 features) | Class-Balanced XGBoost Classifier (19 features) | **Production Ready** |
| [🍵 **Tea**](tea/README.md) | Plantation | North Bengal Tea Auctions | 6,432 rows (2001–2023) | 13 | 4 | XGBoost Regressor (20 features) | Class-Balanced XGBoost Classifier (19 features) | **Reference Baseline** |

---

## 2. Directory & Storage Separation Structure

All commodity artifacts and datasets are structured with per-commodity separation:

```text
member2-price-volatility/
├── commodities/                       ← Commodity documentation guides
│   ├── README.md                      ← Master multi-commodity guide (this file)
│   ├── rice/                          ← Rice architecture, metrics & operational guide
│   │   └── README.md
│   ├── wheat/                         ← Wheat architecture, metrics & operational guide
│   │   └── README.md
│   ├── potato/                        ← Potato architecture, metrics & operational guide
│   │   └── README.md
│   └── tea/                           ← Tea architecture & reference baseline guide
│       └── README.md
│
├── artifacts/                         ← Model artifacts & feature configurations
│   ├── README.md                      ← Artifact catalog & provenance overview
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
│   ├── potato/                        ← Separated Potato models, configs & mappings
│   │   ├── potato_price_model.pkl
│   │   ├── potato_volatility_model.pkl
│   │   ├── potato_feature_config.json
│   │   ├── potato_volatility_config.json
│   │   └── potato_group_mappings.json
│   └── tea/                           ← Separated Tea baseline models & mappings
│       ├── tea_price_model.pkl
│       ├── tea_volatility_model.pkl
│       ├── tea_feature_config.json
│       ├── tea_volatility_config.json
│       └── tea_group_mappings.json
│
├── data/
│   ├── raw/                           ← Source raw Agmarknet CSVs
│   │   ├── README.md                  ← Raw data catalog & schema specification
│   │   ├── rice/                      ← Raw Rice records (rice.csv)
│   │   ├── wheat/                     ← Raw Wheat records (wheat.csv)
│   │   ├── potato/                    ← Raw Potato records (potato.csv)
│   │   └── tea/                       ← Raw Tea records (tea.csv)
│   │
│   └── processed/                     ← Cleaned, time-series, features & train/test splits
│       ├── README.md                  ← Processed dataset dictionary
│       ├── rice/                      ← Processed Rice feature tables & splits
│       ├── wheat/                     ← Processed Wheat feature tables & splits
│       ├── potato/                    ← Processed Potato feature tables & splits
│       └── tea/                       ← Processed Tea feature tables & splits
```

---

## 3. How to Train Any Commodity

The pipeline provides a single unified entry point (`run.py` or `src/pipeline.py`) capable of training and evaluating models for any commodity:

### 3.1 Train a Single Commodity

```bash
# Train Rice pipeline (Price + Volatility)
python member2-price-volatility/run.py --commodity rice

# Train Wheat pipeline
python member2-price-volatility/run.py --commodity wheat

# Train Potato pipeline
python member2-price-volatility/run.py --commodity potato

# Train Tea baseline
python member2-price-volatility/run.py --commodity tea
```

### 3.2 Train All Commodities in One Command

```bash
python member2-price-volatility/run.py --commodity all
```
This automatically discovers all raw commodity CSVs, verifies data completeness, trains price and volatility models for each commodity, evaluates out-of-sample metrics, and updates the respective commodity artifact folders.

### 3.3 Fast Mode (Price Model Only)

```bash
python member2-price-volatility/run.py --commodity wheat --skip-volatility
python member2-price-volatility/run.py --commodity potato --skip-volatility
```

---

## 4. How to Run Multi-Commodity Predictions

### 4.1 Python API (`models.predict`)

```python
import pandas as pd
from member2_price_volatility.src.models.predict import predict

# Predict Wheat price and volatility from 30+ days of historical observations
wheat_history = pd.read_csv("member2-price-volatility/data/processed/wheat/wheat_test.csv").iloc[:35]
result = predict(wheat_history, commodity="wheat")
print(f"Predicted Wheat Modal Price: ₹{result['predicted_price']}/qtl")
print(f"Predicted Volatility Regime: {result['volatility']}")

# Predict Potato price and volatility
potato_history = pd.read_csv("member2-price-volatility/data/processed/potato/potato_test.csv").iloc[:35]
result = predict(potato_history, commodity="potato")
print(f"Predicted Potato Modal Price: ₹{result['predicted_price']}/qtl")
print(f"Predicted Volatility Regime: {result['volatility']}")

# Predict Rice price and volatility (uses production RiceInferencePipeline)
rice_history = pd.read_csv("member2-price-volatility/data/processed/rice/rice_test.csv").iloc[:35]
result = predict(rice_history, commodity="rice")
print(f"Predicted Rice Modal Price: ₹{result['predicted_price']}/qtl")
print(f"Predicted Volatility: {result['volatility']}")
```

### 4.2 Shared Bridge (`shared.predict_bridge`)

For Member 3 (FastAPI service and full application integration):

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

bridge = AgroVisionPredictor()

# 1. Wheat
wheat_data = pd.read_csv("member2-price-volatility/data/processed/wheat/wheat_test.csv").iloc[:35]
res_wheat = bridge.predict_price(wheat_data, commodity="wheat")
print("Wheat forecast:", res_wheat)

# 2. Potato
potato_data = pd.read_csv("member2-price-volatility/data/processed/potato/potato_test.csv").iloc[:35]
res_potato = bridge.predict_price(potato_data, commodity="potato")
print("Potato forecast:", res_potato)

# 3. Rice
rice_data = pd.read_csv("member2-price-volatility/data/processed/rice/rice_test.csv").iloc[:35]
res_rice = bridge.predict_price(rice_data, commodity="rice")
print("Rice forecast:", res_rice)
```

---

## 5. Comparative Commodity Scorecard

| Metric | Rice (48 Feat) | Wheat (20 Feat) | Potato (20 Feat) | Tea (20 Feat) |
|:---|:---:|:---:|:---:|:---:|
| **Training Observations** | 247,198 | 51,491 | 212,037 | 5,145 |
| **Validation / Test Rows** | 20,465 holdout | 12,873 test | 53,010 test | 1,287 test |
| **Price Forecasting $R^2$** | **0.9805** | **0.9520** | **0.9240** | 0.8910 |
| **Price MAE (₹/Quintal)** | **₹18.98** | **₹24.50** | **₹32.80** | ₹45.20 |
| **Price Accuracy ($100 - \text{MAPE}$)** | **99.46%** | **98.70%** | **97.60%** | 96.20% |
| **Volatility Accuracy** | 68.25% (V2 norm) | **63.49%** | 50.88% | 58.40% |
| **Volatility Balanced Accuracy** | 63.20% | **60.37%** | 44.42% | 54.10% |
| **Volatility Macro F1** | 0.6247 | **0.6101** | 0.4610 | 0.5350 |
| **Direct 14D/30D Calendar Horizons** | Supported (Production) | Planned | Planned | N/A |
| **Prospective Shadow Engine** | Supported (V3 14D) | Architecture Compatible | Architecture Compatible | N/A |

---

## 6. Commodity-Specific Documentation Links

- [🌾 **Rice Deep-Dive Guide**](rice/README.md)
- [🌾 **Wheat Deep-Dive Guide**](wheat/README.md)
- [🥔 **Potato Deep-Dive Guide**](potato/README.md)
- [🍵 **Tea Deep-Dive Guide**](tea/README.md)
