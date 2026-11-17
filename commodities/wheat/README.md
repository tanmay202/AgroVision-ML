# Wheat (*Triticum aestivum*) — Unified Agricultural Intelligence Guide

> **End-to-end biological crop yield forecasting, wholesale mandi price prediction, and market volatility intelligence for West Bengal Wheat.**

---

## Table of Contents
- [1. Agricultural Context & Commodity Profile](#1-agricultural-context--commodity-profile)
- [2. Pre-Harvest Biological Yield Modeling (Member 1)](#2-pre-harvest-biological-yield-modeling-member-1)
  - [2.1 Growing Season & Agrometeorological Cutoff](#21-growing-season--agrometeorological-cutoff)
  - [2.2 18-Feature Specification](#22-18-feature-specification)
  - [2.3 Model Architecture & Production Artifact](#23-model-architecture--production-artifact)
  - [2.4 Statutory DES APY Ground-Truth Dataset](#24-statutory-des-apy-ground-truth-dataset)
  - [2.5 Validation Performance vs. Baselines](#25-validation-performance-vs-baselines)
- [3. Post-Harvest Wholesale Mandi Market Intelligence (Member 2)](#3-post-harvest-wholesale-mandi-market-intelligence-member-2)
  - [3.1 Market Scope & Variety Coverage](#31-market-scope--variety-coverage)
  - [3.2 20-Feature Price Regressor Performance](#32-20-feature-price-regressor-performance)
  - [3.3 19-Feature Class-Balanced Volatility Classifier](#33-19-feature-class-balanced-volatility-classifier)
  - [3.4 Horizon Release Governance](#34-horizon-release-governance)
- [4. Semantic Contract & Domain Data Governance](#4-semantic-contract--domain-data-governance)
- [5. Unified Prediction Code Examples](#5-unified-prediction-code-examples)
- [6. Complete Output Metadata Schemas](#6-complete-output-metadata-schemas)
- [7. Operational Status & Known Limitations](#7-operational-status--known-limitations)

---

## 1. Agricultural Context & Commodity Profile

Wheat (*Triticum aestivum*) is the principal winter cereal food grain grown in the Gangetic alluvial and northern sub-Himalayan plains of West Bengal (notably Murshidabad, Nadia, Malda, Birbhum, and Uttar Dinajpur). 

Cultivated exclusively as a **Rabi** crop, wheat follows the monsoon Aman rice harvest. The crop has a tightly constrained biological window due to the short, mild eastern Indian winter:
- **Sowing Window:** November 15 – December 15.
- **Crown Root Initiation & Tillering:** December.
- **Heading & Anthesis:** January – early February.
- **Grain Filling & Maturity:** February – March. Vulnerable to premature terminal heat stress if daily maximum temperatures rise above $30^\circ\text{C}$ in late February.
- **Harvest Window:** March 15 – April 15.

---

## 2. Pre-Harvest Biological Yield Modeling (Member 1)

### 2.1 Growing Season & Agrometeorological Cutoff
To deliver timely operational forecasts well in advance of harvest without data leakage, meteorological observations are aggregated up to a strict mid-season cutoff:
- **Operational Prediction Cutoff:** **January 31** (~45–60 days ahead of peak harvest).
- Captures vegetative health, tillering adequacy, and early reproductive temperature stresses from NASA POWER daily reanalysis.

### 2.2 18-Feature Specification
The certified production model (`m1_wheat_delta_ridge.pkl`) operates on 18 features:
1. `roll3_mean`: 3-year historical moving average yield (kg/ha)
2. `yield_lag1`: Crop yield 1 year prior (kg/ha)
3. `yield_lag2`: Crop yield 2 years prior (kg/ha)
4. `yield_lag3`: Crop yield 3 years prior (kg/ha)
5. `area_lag1`: Planted area in prior season (ha)
6. `InSeason_Precip_Cum`: Cumulative precipitation from October 1 to January 31 (mm)
7. `InSeason_Precip_Mean`: Daily mean precipitation (mm/day)
8. `InSeason_Precip_Max`: Maximum single-day precipitation (mm/day)
9. `InSeason_T2M_Mean`: Mean daily 2m temperature (°C)
10. `InSeason_T2M_Min`: Minimum daily temperature (°C)
11. `InSeason_T2M_Max`: Maximum daily temperature (°C)
12. `InSeason_T2M_Range`: Diurnal temperature range (`T2M_Max` - `T2M_Min`, °C)
13. `InSeason_RH2M_Mean`: Mean relative humidity (%)
14. `InSeason_Solar_Mean`: Mean all-sky solar insolation (MJ/m²/day)
15. `InSeason_Wind_Mean`: Mean daily 2m wind speed (m/s)
16. `heat_stress_days`: Count of days exceeding critical vegetative heat thresholds
17. `district_encoded_baseline`: Target-encoded historical district baseline productivity
18. `area_trend_ratio`: Ratio of current area proxy to historical moving area

### 2.3 Model Architecture & Production Artifact
- **Architecture:** Regularized Ridge Regression ($\alpha = 20.0$) on Delta from 3-Year Rolling Mean:
  $$\widehat{y}_{d, t} = \bar{y}_{d, t}^{\text{roll3}} + \widehat{\Delta}_{\text{weather}}$$
- **Production Artifact:** `member1-yield-model/models/m1_wheat_delta_ridge.pkl`
- **Certified SHA-256:** `a24ee4a3e90f58186272c5e79ae55cee78a350df1af57e8024145c18607a55c0`
- **Release Status:** **PRODUCTION READY**

### 2.4 Statutory DES APY Ground-Truth Dataset
- **Source:** Directorate of Economics and Statistics (DES), Ministry of Agriculture (`data.desagri.gov.in`).
- **Coverage:** 486 district-years across 22 West Bengal administrative districts (1997–2022).
- **Parity Verification:** 100% mathematical yield identity parity ($\text{Yield} = \text{Production} \times 1000 / \text{Area}$) verified on all 486 rows.

### 2.5 Validation Performance vs. Baselines
1. **Independent Post-2020 Out-of-Time Holdout (2020–2022, $N=61$):**
   - **Model MAE:** **370.91 kg/ha**
   - **Persistence Baseline MAE:** 377.14 kg/ha (M6 Ridge Baseline: 443.18 kg/ha)
   - **Net Improvement:** **+6.23 kg/ha**
   - **Mean Bias:** $-47.38\text{ kg/ha}$
2. **8-Year Pooled Rolling-Origin Evaluation (2015–2022, $N=159$):**
   - **Model MAE:** **428.46 kg/ha**
   - **Persistence Baseline MAE:** 496.80 kg/ha
   - **Net Statistically Significant Gain:** **+68.35 kg/ha**
   - **95% Bootstrap Confidence Interval:** `[+31.84, +107.90] kg/ha`
   - **Paired $t$-test:** $t = 3.86, p = 0.00016$ ($p < 0.001$)
   - **Win Rate vs. Persistence:** **61.6%** (98 of 159 district-years)

---

## 3. Post-Harvest Wholesale Mandi Market Intelligence (Member 2)

### 3.1 Market Scope & Variety Coverage
- **Wholesale Markets:** 38 APMC mandis across West Bengal wheat belts.
- **Varieties Tracked:** 16 commercial and traditional varieties (Kalyansona, Sonalika, PBW-343, HD-2967, Deshi, Other).
- **Historical Dataset:** 64,364 chronological market observations from Agmarknet (2001–2024).

### 3.2 20-Feature Price Regressor Performance
Evaluated on the independent retrospective holdout test split ($N = 12,873$):
- **Model Architecture:** Tuned XGBoost Regressor (20 features: lags, rolling means, arrival volume ratios, calendar encodings).
- **Mean Absolute Error (MAE):** **Rs. 24.50 / quintal**
- **Root Mean Squared Error (RMSE):** **Rs. 62.10 / quintal**
- **$R^2$ Variance Explained:** **0.9520** (vs. Persistence baseline: 0.9380)
- **Within $\pm \text{Rs.}25$/qtl Practical Tolerance:** **79.40%**
- **Percentage Error Metric ($100 - \text{MAPE}$):** **98.70%**
- **Artifact:** `member2-price-volatility/artifacts/wheat/wheat_price_model.pkl`

### 3.3 19-Feature Class-Balanced Volatility Classifier
Evaluated on holdout test set ($N = 12,873$):
- **Model Architecture:** 19-feature Class-Balanced XGBoost Classifier.
- **Overall Accuracy:** **63.49%** (Majority baseline: 39.42%, **+24.07 pp**)
- **Balanced Accuracy:** **60.37%** (Stratified chance: 32.77%, **+27.60 pp**)
- **Macro F1 Score:** **0.6101** (Majority baseline: 0.1885, **+42.16 pp**)
- **HIGH-Volatility Recall:** **67.96%** (F1: 0.6602)
- **Artifact:** `member2-price-volatility/artifacts/wheat/wheat_volatility_model.pkl`

### 3.4 Horizon Release Governance
- **Short-Term (1–7 Days):** **PRODUCTION APPROVED**
- **14-Day Calendar Horizon:** **EXPERIMENTAL**
- **30-Day Calendar Horizon:** **EXPERIMENTAL**

---

## 4. Semantic Contract & Domain Data Governance

AgroVision strictly isolates three distinct agricultural metrics:
1. **Biological Crop Yield (`predicted_yield_kg_per_ha`, `predicted_yield_tonnes_per_ha`):**
   Productivity intensity predicted by Member 1 (typical West Bengal wheat: 1,800–3,500 kg/ha).
2. **Total Regional Production (`estimated_production_tonnes`):**
   Computed **only** when cultivated area in hectares is explicitly supplied:
   $$\text{Production (tonnes)} = \text{Yield (tonnes/ha)} \times \text{Area (ha)}$$
3. **Mandi Market Arrivals:**
   Quantity of produce physically traded at wholesale APMC yards (tonnes). Consumed by Member 2 price features; **never** predicted by Member 1.

---

## 5. Unified Prediction Code Examples

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

bridge = AgroVisionPredictor()

# 1. Predict Wheat Yield (Rabi Season)
wheat_yield_query = {
    "District": "Murshidabad",
    "Year": 2023,
    "Yield_lag1": 2250.0,
    "Area_ha": 65.0  # Optional: enables total harvest estimation
}
yield_output = bridge.predict_wheat_yield(wheat_yield_query)
print(f"Predicted Yield: {yield_output['predicted_yield_kg_per_ha']} kg/ha")
print(f"Total Production: {yield_output['estimated_production_tonnes']} tonnes")

# 2. Predict Wheat Mandi Price & Volatility
wheat_history = pd.read_csv("member2-price-volatility/data/processed/wheat/wheat_test.csv").iloc[:35]
price_output = bridge.predict_price(wheat_history, commodity="wheat")
print(f"Predicted Modal Price: Rs. {price_output['predicted_price']}/qtl")
print(f"Volatility Regime: {price_output['volatility']}")
```

---

## 6. Complete Output Metadata Schemas

```json
{
  "crop": "wheat",
  "district": "Murshidabad",
  "season": "Rabi",
  "year": 2023,
  "predicted_yield_kg_per_ha": 2341.52,
  "predicted_yield_tonnes_per_ha": 2.3415,
  "baseline_yield_kg_ha": 2210.00,
  "predicted_delta_kg_ha": 131.52,
  "estimated_production_tonnes": 152.20,
  "prediction_interval_95": [2045.10, 2637.94],
  "prediction_method": "m1_wheat_delta_ridge",
  "weather_feature_status": "complete",
  "weather_features_used": true,
  "model_name": "m1_wheat_delta_ridge.pkl",
  "release_status": "PRODUCTION READY",
  "warnings": []
}
```

---

## 7. Operational Status & Known Limitations

- **Yield Production Release:** Certified **`PRODUCTION READY`** (`m1_wheat_delta_ridge.pkl`).
- **Short-Term Price Release:** Certified **`PRODUCTION APPROVED`** (`wheat_price_model.pkl`).
- **Horizon Policy:** 14-day and 30-day wheat price horizons are **EXPERIMENTAL**.
- **Agronomic Constraints:** Late terminal heat anomalies occurring after the January 31 cutoff cannot be incorporated into pre-harvest point estimates.
