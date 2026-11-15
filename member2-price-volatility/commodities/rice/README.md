# AgroVision-ML — Member 2: Rice Commodity Pipeline

Complete engineering and machine learning specification for **Rice mandi modal price forecasting, direct calendar horizons, and market volatility estimation** in West Bengal wholesale agricultural markets.

---

## 1. Overview & Market Scope

Rice (*Oryza sativa*) is the primary staple food crop of West Bengal, representing the flagship commodity of the AgroVision-ML intelligence ecosystem. This module provides highly validated, point-in-time forecasting across wholesale mandis under strict leakage-prevention guarantees.

- **Market Focus:** 87 West Bengal APMC wholesale mandis across 18 agricultural districts.
- **Varieties Handled:** 34 commercial and traditional varieties, including *Common*, *Fine*, *Super Fine*, *Minikit*, *Swarna*, *Ratna*, *Gobindobhog*, *IR-36*, *Dudheswar*, *Pankaj*, *Masuri*, *I.R. 8*, and *Other*.
- **Temporal Horizon:** 2001-03-29 to 2024-02-01 (315,079 chronological market observations).
- **History Requirement:** Minimum 30 earlier observations per Market + Variety group required for feature engineering and inference.

---

## 2. Model Architecture & Benchmark Scorecard

### 2.1 Short-Term Price Forecasting (Frozen 48-Feature XGBoost Regressor)
Evaluated on the independent retrospective holdout period (2023-02-01 to 2024-02-01, $N = 20,465$ observations):

| Evaluation Horizon | Sample Size ($N$) | Percentage Accuracy ($100 - \text{MAPE}$) | Within $\pm ₹25$/qtl | MAE (₹/qtl) | RMSE (₹/qtl) | $R^2$ Variance |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Ahead** | 17,419 (85.1%) | **99.54%** | **85.76%** | **₹15.75** | 69.14 | **0.9869** |
| **2-Day Ahead** | 1,739 (8.5%) | **99.20%** | **78.21%** | **₹26.01** | 118.68 | **0.9634** |
| **3-Day Ahead** | 721 (3.5%) | **98.45%** | **67.68%** | **₹48.41** | 151.84 | **0.9412** |
| **4–7 Days Ahead** | 586 (2.9%) | **98.15%** | **63.80%** | **₹51.20** | 148.90 | **0.9250** |
| **Overall Holdout** | **20,465 (100%)** | **99.46%** | **84.09%** | **₹18.98** | **85.45** | **0.9805** |

### 2.2 Direct Calendar-Horizon Price Models
- **14-Day Calendar Horizon (`rice_price_model_14d.pkl`)**:
  - Sample: $N=18,921$ holdout rows.
  - **MAE: ₹59.87/qtl** (outperforms persistence ₹60.16)
  - **RMSE: 141.01** (outperforms persistence 146.29)
  - **$R^2$: 0.9472** (outperforms persistence 0.9432)
  - Status: **Opt-in Production**.
- **30-Day Calendar Horizon (`rice_price_model_30d.pkl`)**:
  - Sample: $N=17,860$ holdout rows.
  - **RMSE: 200.25** (outperforms persistence 204.52)
  - **$R^2$: 0.8940** (outperforms persistence 0.8895)
  - Status: **Experimental Opt-in**.

### 2.3 Volatility Classification Models
- **`v1` Persistence Baseline (Production Default)**: Evaluates historical 5-observation range ($0.5$, $50.0$ Rs/qtl thresholds). Accuracy: 67.08%, Balanced Acc: 60.20%.
- **`v2_normalized` ML Champion (Opt-in)**: 35-feature price-normalized XGBoost classifier ($0.014286\%$, $1.960784\%$ thresholds). Accuracy: **68.25%**, Balanced Acc: **63.20%**, Macro F1: **0.6247**, HIGH Recall: **55.62%** ($p = 1.60\times 10^{-5}$).
- **`v3` 14D Prospective Shadow Pipeline**: 38-feature XGBoost with isotonic calibration running in an isolated background SQLite store with zero farmer-facing exposure.

---

## 3. Commodity Artifacts & File Structure

All Rice assets are organized in dedicated commodity directories:

```text
member2-price-volatility/
├── artifacts/
│   └── rice/
│       ├── rice_price_model.pkl                  ← Frozen 48-feature short-term price model
│       ├── rice_price_final_config.json          ← Short-term price model configuration
│       ├── rice_price_model_14d.pkl              ← Direct 14-day calendar horizon model
│       ├── rice_price_model_14d_config.json      ← 14-day horizon configuration
│       ├── rice_price_model_30d.pkl              ← Direct 30-day calendar horizon model
│       ├── rice_price_model_30d_config.json      ← 30-day horizon configuration
│       ├── rice_volatility_normalized_model.pkl  ← Champion XGBoost normalized model (v2)
│       ├── rice_volatility_normalized_config.json← Normalized volatility config
│       ├── rice_feature_config.json              ← Full feature definitions
│       ├── rice_group_mappings.json              ← Canonical strings for 87 mandis & 34 varieties
│       ├── rice_holdout_residuals.json           ← Short-term empirical residual distribution
│       ├── rice_holdout_residuals_7d.json        ← 7-day empirical residual distribution
│       ├── rice_holdout_residuals_14d.json       ← 14-day empirical residual distribution
│       └── rice_holdout_residuals_30d.json       ← 30-day empirical residual distribution
│
├── data/
│   ├── raw/
│   │   └── rice/
│   │       └── rice.csv                          ← Source Agmarknet CSV (315,079 rows)
│   │
│   └── processed/
│       └── rice/
│           ├── rice_cleaned.csv                  ← Deduplicated and cleaned time series
│           ├── rice_timeseries.csv               ← Time-series group records
│           ├── rice_features_final.csv           ← 48-feature model-ready matrix
│           ├── rice_train.csv                    ← Training partition (2002–2021)
│           ├── rice_test.csv                     ← Holdout test partition (2023–2024)
│           ├── rice_volatility_train.csv
│           ├── rice_volatility_test.csv
│           ├── rice_horizon_predictions.csv
│           └── rice_horizon_14d_30d_metrics.csv
```

---

## 4. How to Train the Rice Pipeline

```bash
# Run the complete Rice ML pipeline
python member2-price-volatility/run.py --commodity rice

# Run without tuning volatility (price model only)
python member2-price-volatility/run.py --commodity rice --skip-volatility
```

---

## 5. How to Run Rice Predictions

### 5.1 Using Production `RiceInferencePipeline`
```python
import pandas as pd
from src.inference import RiceInferencePipeline

pipeline = RiceInferencePipeline()

# Load historical mandi observations (30+ rows)
history = pd.read_csv("member2-price-volatility/data/processed/rice/rice_test.csv").iloc[:35]

# Default prediction (v1 persistence volatility)
res_v1 = pipeline.predict(history, market="Burdwan", variety="Common")
print(f"Rice Modal Price: ₹{res_v1['predicted_modal_price']}/qtl, Volatility: {res_v1['predicted_volatility_class']}")

# Opt-in normalized prediction (v2 ML classifier)
res_v2 = pipeline.predict(history, market="Burdwan", variety="Common", volatility_version="v2_normalized")
print(f"Normalized Volatility Probabilities: {res_v2['normalized_volatility_probabilities']}")

# Direct calendar horizon predictions (14-day or 30-day)
res_14d = pipeline.predict_horizon(history, horizon_days=14)
print(f"14-Day Calendar Price: ₹{res_14d['predicted_modal_price']}/qtl")
```

### 5.2 Using Shared Bridge (`shared.predict_bridge`)
```python
from shared.predict_bridge import AgroVisionPredictor

bridge = AgroVisionPredictor()
res = bridge.predict_price(history, commodity="rice")
print("Bridge Price:", res["predicted_price_rs_quintal"])
```
