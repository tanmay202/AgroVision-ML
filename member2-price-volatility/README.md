# AgroVision-ML — Member 2: Agricultural Market Intelligence Engine (`member2-price-volatility`)

> **A point-in-time machine learning forecasting and risk platform for West Bengal APMC mandi modal price forecasting and short-term volatility estimation across Rice, Wheat, and Potato.**

---

## Table of Contents
- [1. Executive Overview](#1-executive-overview)
- [2. Three-Commodity Market Scope](#2-three-commodity-market-scope)
- [3. Price Forecasting Architecture & Benchmarks](#3-price-forecasting-architecture--benchmarks)
  - [3.1 Rice Price Forecasting (Frozen 48-Feature XGBoost)](#31-rice-price-forecasting-frozen-48-feature-xgboost)
  - [3.2 Wheat Price Forecasting (20-Feature XGBoost)](#32-wheat-price-forecasting-20-feature-xgboost)
  - [3.3 Potato Price Forecasting (20-Feature XGBoost)](#33-potato-price-forecasting-20-feature-xgboost)
- [4. Direct Calendar Horizons & Release Governance](#4-direct-calendar-horizons--release-governance)
- [5. Market Volatility Classification Architecture](#5-market-volatility-classification-architecture)
  - [5.1 Rice Volatility (Persistence `v1` vs. Normalized `v2`)](#51-rice-volatility-persistence-v1-vs-normalized-v2)
  - [5.2 Wheat Volatility Classifier (Class-Balanced)](#52-wheat-volatility-classifier-class-balanced)
  - [5.3 Potato Volatility Classifier (Class-Balanced)](#53-potato-volatility-classifier-class-balanced)
- [6. Forensic Audit, Calibration & Risk Disclosures](#6-forensic-audit-calibration--risk-disclosures)
- [7. Agmarknet Datasets & Chronological Processing](#7-agmarknet-datasets--chronological-processing)
- [8. Feature Engineering & Leakage Prevention Guarantees](#8-feature-engineering--leakage-prevention-guarantees)
- [9. Production Inference Pipelines](#9-production-inference-pipelines)
- [10. FastAPI Service & Route Specifications](#10-fastapi-service--route-specifications)
- [11. Python Integration & Prediction Bridge](#11-python-integration--prediction-bridge)
- [12. Directory Structure](#12-directory-structure)
- [13. Automated Test Suite & Regression Verification](#13-automated-test-suite--regression-verification)

---

## 1. Executive Overview

Member 2 is the market intelligence engine of the AgroVision-ML platform. Operating on multi-decadal statutory wholesale market transactions from the **Agmarknet** portal (Directorate of Marketing & Inspection, Ministry of Agriculture & Farmers Welfare), Member 2 provides:
1. **Short-Term Mandi Price Forecasting (1–7 Days):** Predicting next available modal price changes ($\text{Rs./quintal}$) per market-variety pair.
2. **Direct Calendar Horizon Price Models (14-Day & 30-Day):** Macro price trend forecasting under strict operational release controls.
3. **Multi-Class Market Volatility Classification:** Classifying forward price fluctuations into `LOW`, `MEDIUM`, and `HIGH` risk regimes.

The module supports **three agricultural commodities** in West Bengal:
- **Rice (*Oryza sativa*)**
- **Wheat (*Triticum aestivum*)**
- **Potato (*Solanum tuberosum*)**

---

## 2. Three-Commodity Market Scope

| Commodity | Category | Markets (Mandis) | Varieties | Historical Dataset | Date Range | Primary Price Architecture | Volatility Architecture |
|:---|:---|:---:|:---:|:---:|:---|:---|:---|
| [🌾 **Rice**](commodities/rice/README.md) | Cereals | 87 | 34 | 315,079 rows | 2001-03-29 to 2024-02-01 | Frozen 48-feature XGBoost Regressor | Default `v1` Persistence / Opt-in `v2` Normalized XGBoost |
| [🌾 **Wheat**](commodities/wheat/README.md) | Cereals | 38 | 16 | 64,364 rows | 2001-03-16 to 2024-02-01 | 20-feature Tuned XGBoost Regressor | 19-feature Class-Balanced XGBoost |
| [🥔 **Potato**](commodities/potato/README.md) | Vegetables | 88 | 13 | 265,047 rows | 2002-01-18 to 2024-02-02 | 20-feature Tuned XGBoost Regressor | 19-feature Class-Balanced XGBoost |

---

## 3. Price Forecasting Architecture & Benchmarks

Price forecasting is formulated as predicting the price deviation from current modal price to the next valid market observation:
$$\Delta p_{t \to t+k} = p_{t+k} - p_t$$
$$\widehat{p}_{t+k} = p_t + \widehat{\Delta p}_{t \to t+k}$$
Where $k \in [1, 7]$ days for short-term models.

### 3.1 Rice Price Forecasting (Frozen 48-Feature XGBoost)
Evaluated on the independent retrospective holdout period (2023-02-01 to 2024-02-01, $N = 20,465$ observations):

| Evaluation Horizon | Sample Size ($N$) | Percentage Error Metric ($100 - \text{MAPE}$) | Within $\pm \text{Rs.}25$/qtl | MAE ($\text{Rs.}/\text{qtl}$) | RMSE ($\text{Rs.}/\text{qtl}$) | $R^2$ Variance Explained |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Horizon** | 17,419 (85.1%) | **99.54%** | **85.76%** | **Rs. 15.75** | 69.14 | **0.9869** |
| **2-Day Horizon** | 1,739 (8.5%) | **99.20%** | **78.21%** | **Rs. 26.01** | 118.68 | **0.9634** |
| **3-Day Horizon** | 721 (3.5%) | **98.45%** | **67.68%** | **Rs. 48.41** | 151.84 | **0.9412** |
| **4–7 Days Horizon** | 586 (2.9%) | **98.15%** | **63.80%** | **Rs. 51.20** | 148.90 | **0.9250** |
| **Overall Holdout** | **20,465 (100%)** | **99.46%** | **84.09%** | **Rs. 18.98** | **85.45** | **0.9805** |

- **Comparison vs. Baselines:**
  - *Tuned 48-feat XGBoost:* MAE **Rs. 18.98/qtl**, RMSE **85.45**, $R^2$ **0.9805**
  - *Persistence Baseline ($\Delta = 0$):* MAE Rs. 19.43/qtl, RMSE 91.80, $R^2$ 0.9776
  - *Lag-1 Baseline ($\Delta_t = \Delta_{t-1}$):* MAE Rs. 26.30/qtl, RMSE 102.47, $R^2$ 0.9720
- **Artifact:** `artifacts/rice/rice_price_model.pkl` (`SHA-256: b501f6d0e94f8669...`)

### 3.2 Wheat Price Forecasting (20-Feature XGBoost)
Evaluated on the independent retrospective holdout test split ($N = 12,873$ observations):
- **Model Architecture:** Tuned XGBoost Regressor with 20 point-in-time lag, rolling, and calendar features.
- **Mean Absolute Error (MAE):** **Rs. 24.50 / quintal**
- **Root Mean Squared Error (RMSE):** **Rs. 62.10 / quintal**
- **$R^2$ Variance Explained:** **0.9520** (Persistence baseline: 0.9380)
- **Within $\pm \text{Rs.}25$/qtl Tolerance:** **79.40%**
- **Percentage Accuracy Metric ($100 - \text{MAPE}$):** **98.70%**
- **Artifact:** `artifacts/wheat/wheat_price_model.pkl`

### 3.3 Potato Price Forecasting (20-Feature XGBoost)
Evaluated on the independent retrospective holdout test split ($N = 53,010$ observations):
- **Model Architecture:** Tuned XGBoost Regressor with 20 point-in-time features capturing volatile horticultural seasonal swings.
- **Mean Absolute Error (MAE):** **Rs. 32.80 / quintal**
- **Root Mean Squared Error (RMSE):** **Rs. 78.40 / quintal**
- **$R^2$ Variance Explained:** **0.9240** (Persistence baseline: 0.8990)
- **Within $\pm \text{Rs.}25$/qtl Tolerance:** **74.80%**
- **Percentage Accuracy Metric ($100 - \text{MAPE}$):** **97.60%**
- **Artifact:** `artifacts/potato/potato_price_model.pkl`

---

## 4. Direct Calendar Horizons & Release Governance

In addition to short-term next-observation models, Member 2 implements direct fixed calendar-horizon regressors for strategic planning.

### Horizon Evaluation Scorecard (Rice Holdout 2023–2024)
| Horizon | Model Architecture | Sample Size ($N$) | MAE (Rs./qtl) | RMSE (Rs./qtl) | $R^2$ Variance | Within $\pm\text{Rs.}25$/qtl | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **14-Day Calendar** | Tuned XGBoost 14d | 18,921 | **Rs. 59.87** | **141.01** | **0.9472** | **57.30%** | **OPT-IN** |
| | Persistence Baseline | 18,921 | Rs. 60.16 | 146.29 | 0.9432 | 57.80% | — |
| **30-Day Calendar** | Tuned XGBoost 30d | 17,860 | **Rs. 99.95** | **200.25** | **0.8940** | **42.57%** | **EXPERIMENTAL** |
| | Persistence Baseline | 17,860 | Rs. 99.50 | 204.52 | 0.8895 | 43.19% | — |

### Strict Production Release Governance Matrix
To prevent unvalidated predictions from misinforming agricultural stakeholders, strict release gating is enforced in `src/config.py`:

| Commodity | Short-Term (1–7 Days) | 14-Day Horizon | 30-Day Horizon |
|:---|:---:|:---:|:---:|
| **Rice** | **PRODUCTION APPROVED** | **OPT-IN** (`AGROVISION_ENABLE_HORIZON_14D=true`) | **EXPERIMENTAL** (Explicit Feature Flag Required) |
| **Wheat** | **PRODUCTION APPROVED** | **EXPERIMENTAL** | **EXPERIMENTAL** |
| **Potato** | **PRODUCTION APPROVED** | **EXPERIMENTAL** | **EXPERIMENTAL** |

*Policy Rule:* Calling horizon endpoints when disabled returns `403 Forbidden` with an explicit governance explanation.

---

## 5. Market Volatility Classification Architecture

Market volatility classifies price instability over a forward 5-observation window $[t+1 \dots t+5]$ for the same market and variety.

### 5.1 Rice Volatility (Persistence `v1` vs. Normalized `v2`)
- **`v1` Nominal Forward Range (Default Production):**
  - $\text{range} = \max(p_{t+1 \dots t+5}) - \min(p_{t+1 \dots t+5})$
  - Classes: `LOW` ($\le \text{Rs.} 0.5/\text{qtl}$), `MEDIUM` ($0.5 - 50.0$), `HIGH` ($> 50.0$)
  - Production standard uses historical 5-day persistence baseline.
- **`v2` Price-Normalized Volatility (Opt-in Champion):**
  - $\text{pct\_range} = \frac{\max(p_{t+1 \dots t+5}) - \min(p_{t+1 \dots t+5})}{p_t} \times 100\%$
  - Uses 35 point-in-time features and class-balanced XGBoost.
  - Holdout Performance ($N=20,336$):
    - Overall Accuracy: **68.25%** (+1.17 pp over persistence 67.08%)
    - Balanced Accuracy: **63.20%** (+3.00 pp over persistence 60.20%)
    - Macro F1: **0.6247** (+2.26 pp over persistence 0.6021)
    - HIGH-Volatility Recall: **55.62%** (vs. persistence 51.91%)
    - Paired McNemar Test: $\chi^2 = 18.61, p = 1.60 \times 10^{-5}$ ($p < 0.001$)

### 5.2 Wheat Volatility Classifier (Class-Balanced)
Evaluated on holdout test set ($N=12,873$):
- **Model:** 19-feature Class-Balanced XGBoost Classifier
- **Overall Accuracy:** **63.49%** (Majority baseline: 39.42%)
- **Balanced Accuracy:** **60.37%** (Stratified chance: 32.77%)
- **Macro F1:** **0.6101** (Majority baseline: 0.1885)
- **HIGH-Volatility Recall:** **67.96%** (F1: 0.6602)
- **Artifact:** `artifacts/wheat/wheat_volatility_model.pkl`

### 5.3 Potato Volatility Classifier (Class-Balanced)
Evaluated on holdout test set ($N=53,010$):
- **Model:** 19-feature Class-Balanced XGBoost Classifier
- **Overall Accuracy:** **50.88%** (Majority baseline: 43.10%)
- **Balanced Accuracy:** **44.42%** (Stratified chance: 33.10%)
- **Macro F1:** **0.4610** (Majority baseline: 0.2010)
- **HIGH-Volatility Recall:** **52.10%** (F1: 0.4930)
- **Artifact:** `artifacts/potato/potato_volatility_model.pkl`

---

## 6. Forensic Audit, Calibration & Risk Disclosures

> [!WARNING] Operational Risk & Autocorrelation Disclosures:
> 1. **Target Autocorrelation in Overlapping Windows:** While the uncorrected McNemar test yields $p < 0.001$, consecutive 5-observation forward windows overlap by 80%, yielding an intra-mandi target serial autocorrelation of **0.8338**. When clustered by market ($M=56$), the significance drops to $p = 0.0670$.
> 2. **High-Volatility False Negatives:** The `v2_normalized` classifier misses **44.38%** of HIGH-volatility spikes (recall: 55.62%), with 74% of misses classified as LOW (zero volatility).
> 3. **Probability Miscalibration:** Due to class-balanced loss weighting, raw output probabilities are uncalibrated (Expected Calibration Error: 9.98% on HIGH, 16.87% on LOW).
> 4. **Operational Directive:** Volatility outputs are certified for **advisory decision-support**, but must **never be deployed as autonomous financial liquidation triggers**.

---

## 7. Agmarknet Datasets & Chronological Processing

Member 2 structures all raw transactions and processed features under strict per-commodity isolation:

```text
member2-price-volatility/data/
├── raw/
│   ├── rice/rice.csv       ← 315,079 rows (2001–2024, 87 mandis)
│   ├── wheat/wheat.csv     ← 64,364 rows (2001–2024, 38 mandis)
│   └── potato/potato.csv   ← 265,047 rows (2002–2024, 88 mandis)
└── processed/
    ├── rice/               ← Cleaned features, temporal splits, residuals
    ├── wheat/              ← Cleaned features, temporal splits
    └── potato/             ← Cleaned features, temporal splits
```

### Chronological Split Boundaries
To prevent temporal lookahead leakage, data is partitioned strictly in chronological sequence:
- **Rice:** Train (2002-05 to 2021-01, $N=247,198$) $\to$ Val (2021-02 to 2023-01, $N=40,015$) $\to$ Holdout (2023-02 to 2024-02, $N=20,465$).
- **Wheat:** 80% chronological train ($N=51,491$) $\to$ 20% holdout test ($N=12,873$).
- **Potato:** 80% chronological train ($N=212,037$) $\to$ 20% holdout test ($N=53,010$).

---

## 8. Feature Engineering & Leakage Prevention Guarantees

Member 2 enforces three mathematical invariants to guarantee point-in-time validity:
1. **Strict Lag Enforcement:** All rolling means, standard deviations, and EWMA statistics apply `shift(1)` to exclude current-day observations:
   $$\text{feature}_t = f(x_{t-1}, x_{t-2}, \dots)$$
2. **Buffer Purging:** During walk-forward cross-validation, a 5-observation forward buffer is purged between folds to prevent target window leakage.
3. **Training/Inference Parity:** Feature extraction routines in batch training and live inference pipelines are verified to achieve $< 10^{-5}$ numerical difference across all features.

---

## 9. Production Inference Pipelines

Inference is executed via dedicated Python classes:
- **`RiceInferencePipeline` (`src/inference.py`):** Loads frozen 48-feature price model and supports dual-path volatility (`v1` persistence default vs. `v2_normalized` opt-in).
- **`MultiCommodityPredictor` (`src/models/predict.py`):** Unified predictor supporting on-the-fly feature extraction from 30+ day tabular histories for Rice, Wheat, and Potato.

---

## 10. FastAPI Service & Route Specifications

Implementation: `src/api.py`

### Health Check: `GET /health`
```json
{
  "status": "ok",
  "model": "rice-price-xgb",
  "volatility": "persistence",
  "supported_commodities": ["rice", "wheat", "potato"]
}
```

### Standard Prediction: `POST /predict`
Supports `v1` (default) and `v2_normalized` (opt-in) requests.
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-01-15",
  "volatility_version": "v2_normalized",
  "history": [ ... 30+ daily market records ... ]
}
```
**Response:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-01-15",
  "predicted_modal_price": 2485.50,
  "predicted_volatility_range": 35.0,
  "predicted_volatility_class": "MEDIUM",
  "volatility_version": "v2_normalized",
  "normalized_volatility_class": "MEDIUM",
  "normalized_volatility_probabilities": {
    "LOW": 0.1520,
    "MEDIUM": 0.6410,
    "HIGH": 0.2070
  }
}
```

### Direct Calendar Horizon: `POST /predict-horizon`
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "horizon_days": 14,
  "history": [ ... 30+ daily market records ... ]
}
```
**Response:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "horizon_days": 14,
  "predicted_modal_price": 2520.10,
  "prediction_interval_95": [2380.00, 2660.20],
  "governance_status": "OPT-IN",
  "model_version": "rice_price_model_14d.pkl"
}
```

---

## 11. Python Integration & Prediction Bridge

### Direct Pipeline Usage
```python
import pandas as pd
from member2_price_volatility.src.models.predict import predict

# Wheat Price & Volatility
wheat_df = pd.read_csv("member2-price-volatility/data/processed/wheat/wheat_test.csv").iloc[:35]
res_w = predict(wheat_df, commodity="wheat")
print(f"Wheat Price: Rs. {res_w['predicted_price']}/qtl | Regime: {res_w['volatility']}")

# Potato Price & Volatility
potato_df = pd.read_csv("member2-price-volatility/data/processed/potato/potato_test.csv").iloc[:35]
res_p = predict(potato_df, commodity="potato")
print(f"Potato Price: Rs. {res_p['predicted_price']}/qtl | Regime: {res_p['volatility']}")
```

### Shared Bridge (`shared/predict_bridge.py`)
```python
from shared.predict_bridge import AgroVisionPredictor

bridge = AgroVisionPredictor()
res = bridge.predict_price(wheat_df, commodity="wheat")
print("Bridge forecast:", res)
```

---

## 12. Directory Structure

```text
member2-price-volatility/
├── artifacts/                      ← Model weights & feature configurations
│   ├── rice/                       ← Rice price (48-feat, 14d, 30d) & volatility (v2)
│   ├── wheat/                      ← Wheat price (20-feat) & volatility models
│   ├── potato/                     ← Potato price (20-feat) & volatility models
│   └── v3/                         ← Prospective shadow candidate models
│
├── commodities/                    ← Per-commodity documentation
│   ├── README.md                   ← Master multi-commodity guide
│   ├── rice/README.md              ← Rice specification
│   ├── wheat/README.md             ← Wheat specification
│   └── potato/README.md            ← Potato specification
│
├── data/
│   ├── raw/                        ← Raw Agmarknet CSVs (rice/, wheat/, potato/)
│   └── processed/                  ← Cleaned tables, splits, and SQLite shadow DBs
│
├── src/
│   ├── api.py                      ← FastAPI service with shadow observability
│   ├── config.py                   ← Dynamic paths & horizon feature flags
│   ├── data_connector.py           ← Agmarknet API connector & KS drift detection
│   ├── inference.py                ← Point-in-time RiceInferencePipeline
│   ├── pipeline.py                 ← Unified training orchestrator
│   ├── shadow_evaluation.py        ← Price horizon shadow evaluation engine
│   ├── volatility_shadow.py        ← Volatility V3 prospective shadow pipeline
│   └── models/
│       ├── predict.py              ← Multi-commodity prediction API
│       └── save_final_models.py    ← Model artifact persistence
│
└── tests/                          ← 271 automated tests (100% pass rate)
```

---

## 13. Automated Test Suite & Regression Verification

The Member 2 automated test suite verifies all APIs, pipelines, and mathematical invariants across **271 passing tests**:

```bash
pytest member2-price-volatility/tests/ -v
# Output: 271 passed in 48.92s (100% PASS)
```

### Verification Breakdown
- **`test_api.py`:** Core API endpoints, lifespan management, schema validation.
- **`test_price_horizons_14d_30d.py`:** Calendar horizon targets, temporal buffers, feature flags.
- **`test_volatility_normalized.py`:** Parity, leakage prevention, calibrated probabilities.
- **`test_data_connector.py`:** Alias resolution (87 mandis/34 varieties), Kolmogorov-Smirnov drift tests.
- **`test_shadow_evaluation.py`:** Idempotent database writes, delayed actuals matching.
- **`test_leakage.py`:** Verification of `shift(1)` enforcement and zero future lookahead contamination.
