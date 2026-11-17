# Potato (*Solanum tuberosum*) — Unified Agricultural Intelligence Guide

> **End-to-end biological crop yield forecasting, wholesale mandi price prediction, and market volatility intelligence for West Bengal Potato.**

---

## Table of Contents
- [1. Agricultural Context & Commodity Profile](#1-agricultural-context--commodity-profile)
- [2. Pre-Harvest Biological Yield Modeling (Member 1)](#2-pre-harvest-biological-yield-modeling-member-1)
  - [2.1 Growing Season & Agrometeorological Cutoff](#21-growing-season--agrometeorological-cutoff)
  - [2.2 19-Feature Specification](#22-19-feature-specification)
  - [2.3 Model Architecture & Production Artifact](#23-model-architecture--production-artifact)
  - [2.4 The Tree-Saturation Audit & Huber Robustness](#24-the-tree-saturation-audit--huber-robustness)
  - [2.5 Statutory DES APY Ground-Truth Dataset](#25-statutory-des-apy-ground-truth-dataset)
  - [2.6 Validation Performance vs. Baselines](#26-validation-performance-vs-baselines)
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

Potato (*Solanum tuberosum*) is the highest-volume horticultural commercial cash crop in West Bengal. Producing between 25% and 30% of India's total potato harvest, the state's production is intensively clustered in the lower Gangetic basin: Hooghly, Purba Bardhaman, Paschim Medinipur, Bankura, and Jalpaiguri.

Unlike grain crops, potato is characterized by:
- **Massive Productivity Intensity:** Typical yields range between **22,000 and 42,000 kg/ha** (22–42 tonnes/ha).
- **Extreme Capital & Input Density:** Heavy outlays for certified seed tubers (e.g., Kufri Jyoti), intensive fertilizer, and prophylactic fungicides.
- **High Market Volatility:** Massive seasonal harvesting gluts in February/March followed by structured releases from cold-storage facilities throughout summer and autumn create sharp price swings.

The crop is grown exclusively in the **Rabi** season:
- **Planting Window:** October 25 – November 30.
- **Vegetative Emergence:** November – December.
- **Tuber Initiation & Rapid Bulking:** December 15 – January 31. Highly sensitive to night temperatures ($T_{\min} > 18^\circ\text{C}$ drastically halts tuberization) and late blight epidemics triggered by persistent fog and relative humidity $> 85\%$.
- **Harvest Window:** February 10 – March 20.

---

## 2. Pre-Harvest Biological Yield Modeling (Member 1)

### 2.1 Growing Season & Agrometeorological Cutoff
- **Operational Prediction Cutoff:** **January 31** (~30–45 days ahead of peak harvest).
- Captures cumulative precipitation, minimum temperature profiles, diurnal ranges, and humidity-induced blight risk indices during peak tuber development.

### 2.2 19-Feature Specification
The certified production model (`m1_potato_delta_huber.pkl`) operates on 19 features:
1. `roll3_mean`: 3-year historical moving average yield (kg/ha)
2. `yield_lag1`: Crop yield 1 year prior (kg/ha)
3. `yield_lag2`: Crop yield 2 years prior (kg/ha)
4. `yield_lag3`: Crop yield 3 years prior (kg/ha)
5. `area_lag1`: Planted area in prior season (ha)
6. `InSeason_Precip_Cum`: Cumulative rainfall from November 1 to January 31 (mm)
7. `InSeason_Precip_Mean`: Daily mean precipitation (mm/day)
8. `InSeason_Precip_Max`: Maximum single-day precipitation (mm/day)
9. `InSeason_T2M_Mean`: Mean daily 2m air temperature (°C)
10. `InSeason_T2M_Min`: Minimum daily temperature observed (°C)
11. `InSeason_T2M_Max`: Maximum daily temperature observed (°C)
12. `InSeason_T2M_Range`: Diurnal temperature range (`T2M_Max` - `T2M_Min`, °C)
13. `InSeason_RH2M_Mean`: Mean relative humidity (%)
14. `InSeason_Solar_Mean`: Mean all-sky solar insolation (MJ/m²/day)
15. `InSeason_Wind_Mean`: Mean daily 2m wind speed (m/s)
16. `high_humidity_days`: Count of days with $\text{RH} \ge 85\%$ (late blight proxy)
17. `warm_night_days`: Count of days with $T_{\min} > 18^\circ\text{C}$ (tuber inhibition index)
18. `temp_range_x_humidity`: Interaction index between diurnal range and relative humidity
19. `district_encoded_baseline`: Target-encoded historical district baseline productivity

### 2.3 Model Architecture & Production Artifact
- **Architecture:** Robust M-estimation via `HuberRegressor` ($\alpha = 15.0, \epsilon = 1.35$) on Delta from 3-Year Rolling Mean:
  $$\widehat{y}_{d, t} = \bar{y}_{d, t}^{\text{roll3}} + \widehat{\Delta}_{\text{weather}}$$
- **Production Artifact:** `member1-yield-model/models/m1_potato_delta_huber.pkl`
- **Certified SHA-256:** `44102855ffeaacd88c303070c75317b08abceaa4d426e9d56636e86cf59c1e3a`
- **Release Status:** **PRODUCTION READY**

### 2.4 The Tree-Saturation Audit & Huber Robustness
During the Milestone 7 audit, candidate tree-based models (CatBoost, Random Forest) failed on post-2020 validation data:
- **The Failure:** As high-yielding certified seeds expanded West Bengal potato yields to $35,000 - 42,000\text{ kg/ha}$, tree-based models flatlined at historical training ceilings ($\approx 28,000\text{ kg/ha}$), resulting in a bias of $-2,984.91\text{ kg/ha}$ and high MAE ($4,810.78\text{ kg/ha}$).
- **The Huber Advantage:** The Huber loss function treats small residuals quadratically and large deviations linearly ($\le 1.35\sigma$), protecting coefficients from being destabilized by occasional severe blight crop failures while extrapolating upward during technology-driven yield surges.

### 2.5 Statutory DES APY Ground-Truth Dataset
- **Source:** Directorate of Economics and Statistics (DES), Ministry of Agriculture (`data.desagri.gov.in`).
- **Coverage:** 496 plains district-years across 22 West Bengal administrative districts (1997–2022).
- **Parity Verification:** 100% mathematical yield identity parity verified on all 496 rows.

### 2.6 Validation Performance vs. Baselines
1. **Independent Post-2020 Out-of-Time Holdout (2020–2022, $N=66$):**
   - **Model MAE:** **2,981.26 kg/ha**
   - **Persistence Baseline MAE:** 3,934.92 kg/ha (M6 CatBoost Baseline: 4,810.78 kg/ha)
   - **Net Statistically Significant Gain:** **+953.66 kg/ha (+24.24% error reduction)**
   - **95% Bootstrap Confidence Interval:** `[+184.12, +1,850.70] kg/ha`
   - **Paired $t$-test:** $t = 2.62, p = 0.0110$ ($p < 0.05$)
   - **Win Rate vs. Persistence:** **63.6%** (42 of 66 district-years)
2. **8-Year Pooled Rolling-Origin Evaluation (2015–2022, $N=168$):**
   - **Model MAE:** **4,560.00 kg/ha**
   - **Persistence Baseline MAE:** 5,648.12 kg/ha
   - **Net Statistically Significant Gain:** **+1,088.13 kg/ha (+19.27% error reduction)**
   - **95% Bootstrap Confidence Interval:** `[+471.11, +1,720.92] kg/ha`
   - **Paired $t$-test:** $t = 4.09, p = 0.00007$ ($p < 0.0001$)
   - **Win Rate vs. Persistence:** **61.9%** (104 of 168 district-years)

*Note on Error Magnitude:* In absolute terms, an MAE of 2,981 kg/ha represents an average relative percentage error of **~8.5%** across a 35,000 kg/ha baseline.

---

## 3. Post-Harvest Wholesale Mandi Market Intelligence (Member 2)

### 3.1 Market Scope & Variety Coverage
- **Wholesale Markets:** 88 APMC mandis across West Bengal potato hubs.
- **Varieties Tracked:** 13 commercial varieties (Jyoti, Chandramukhi, Kufri Jyoti, Pokhraj, Nainital, Other).
- **Historical Dataset:** 265,047 chronological market observations from Agmarknet (2002–2024).

### 3.2 20-Feature Price Regressor Performance
Evaluated on the independent retrospective holdout test split ($N = 53,010$):
- **Model Architecture:** Tuned XGBoost Regressor (20 features: historical lags, rolling statistics, arrival volume indicators, calendar seasonality).
- **Mean Absolute Error (MAE):** **Rs. 32.80 / quintal**
- **Root Mean Squared Error (RMSE):** **Rs. 78.40 / quintal**
- **$R^2$ Variance Explained:** **0.9240** (vs. Persistence baseline: 0.8990)
- **Within $\pm \text{Rs.}25$/qtl Practical Tolerance:** **74.80%**
- **Percentage Error Metric ($100 - \text{MAPE}$):** **97.60%**
- **Artifact:** `member2-price-volatility/artifacts/potato/potato_price_model.pkl`

### 3.3 19-Feature Class-Balanced Volatility Classifier
Evaluated on holdout test set ($N = 53,010$):
- **Model Architecture:** 19-feature Class-Balanced XGBoost Classifier.
- **Overall Accuracy:** **50.88%** (Majority baseline: 43.10%, **+7.78 pp**)
- **Balanced Accuracy:** **44.42%** (Stratified chance: 33.10%, **+11.32 pp**)
- **Macro F1 Score:** **0.4610** (Majority baseline: 0.2010, **+26.00 pp**)
- **HIGH-Volatility Recall:** **52.10%** (F1: 0.4930)
- **Artifact:** `member2-price-volatility/artifacts/potato/potato_volatility_model.pkl`

### 3.4 Horizon Release Governance
- **Short-Term (1–7 Days):** **PRODUCTION APPROVED**
- **14-Day Calendar Horizon:** **EXPERIMENTAL**
- **30-Day Calendar Horizon:** **EXPERIMENTAL**

---

## 4. Semantic Contract & Domain Data Governance

AgroVision strictly isolates three distinct agricultural metrics:
1. **Biological Crop Yield (`predicted_yield_kg_per_ha`, `predicted_yield_tonnes_per_ha`):**
   Productivity intensity predicted by Member 1 (typical West Bengal potato: 20,000–42,000 kg/ha).
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

# 1. Predict Potato Yield (Rabi Season)
potato_yield_query = {
    "District": "Hooghly",
    "Year": 2023,
    "Yield_lag1": 31000.0,
    "Area_ha": 35.0  # Optional: enables total harvest estimation
}
yield_output = bridge.predict_potato_yield(potato_yield_query)
print(f"Predicted Yield: {yield_output['predicted_yield_tonnes_per_ha']} t/ha")
print(f"Total Production: {yield_output['estimated_production_tonnes']} tonnes")

# 2. Predict Potato Mandi Price & Volatility
potato_history = pd.read_csv("member2-price-volatility/data/processed/potato/potato_test.csv").iloc[:35]
price_output = bridge.predict_price(potato_history, commodity="potato")
print(f"Predicted Modal Price: Rs. {price_output['predicted_price']}/qtl")
print(f"Volatility Regime: {price_output['volatility']}")
```

---

## 6. Complete Output Metadata Schemas

```json
{
  "crop": "potato",
  "district": "Hooghly",
  "season": "Rabi",
  "year": 2023,
  "predicted_yield_kg_per_ha": 32840.50,
  "predicted_yield_tonnes_per_ha": 32.8405,
  "baseline_yield_kg_ha": 30500.00,
  "predicted_delta_kg_ha": 2340.50,
  "estimated_production_tonnes": 1149.42,
  "prediction_interval_95": [29120.00, 36561.00],
  "prediction_method": "m1_potato_delta_huber",
  "weather_feature_status": "complete",
  "weather_features_used": true,
  "model_name": "m1_potato_delta_huber.pkl",
  "release_status": "PRODUCTION READY",
  "warnings": []
}
```

---

## 7. Operational Status & Known Limitations

- **Yield Production Release:** Certified **`PRODUCTION READY`** (`m1_potato_delta_huber.pkl`).
- **Short-Term Price Release:** Certified **`PRODUCTION APPROVED`** (`potato_price_model.pkl`).
- **Horizon Policy:** 14-day and 30-day potato price horizons are **EXPERIMENTAL**.
- **Cold Storage Distortion:** Post-harvest cold-storage release strategies are determined by trader cartel behavior and credit cycles, introducing non-meteorological variance in late-season market prices.
