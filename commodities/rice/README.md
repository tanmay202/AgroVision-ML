# Rice (*Oryza sativa*) — Unified Agricultural Intelligence Guide

> **End-to-end biological crop yield forecasting, wholesale mandi price prediction, and market volatility intelligence for West Bengal Rice.**

---

## Table of Contents
- [1. Agricultural Context & Commodity Profile](#1-agricultural-context--commodity-profile)
- [2. Pre-Harvest Biological Yield Modeling (Member 1)](#2-pre-harvest-biological-yield-modeling-member-1)
  - [2.1 Growing Seasons & Cutoff Windows](#21-growing-seasons--cutoff-windows)
  - [2.2 20-Feature Specification](#22-20-feature-specification)
  - [2.3 Weather Preprocessing & NASA POWER Integration](#23-weather-preprocessing--nasa-power-integration)
  - [2.4 Model Architecture & Production Artifact](#24-model-architecture--production-artifact)
  - [2.5 Validation Performance vs. Baselines](#25-validation-performance-vs-baselines)
  - [2.6 Provisional Summer (Boro) Routing](#26-provisional-summer-boro-routing)
- [3. Post-Harvest Wholesale Mandi Market Intelligence (Member 2)](#3-post-harvest-wholesale-mandi-market-intelligence-member-2)
  - [3.1 Market Scope & Variety Coverage](#31-market-scope--variety-coverage)
  - [3.2 48-Feature Point-in-Time Architecture](#32-48-feature-point-in-time-architecture)
  - [3.3 Short-Term & Calendar Horizon Price Performance](#33-short-term--calendar-horizon-price-performance)
  - [3.4 Market Volatility Classification (`v1` vs. `v2_normalized`)](#34-market-volatility-classification-v1-vs-v2_normalized)
- [4. Semantic Contract & Domain Data Governance](#4-semantic-contract--domain-data-governance)
- [5. Unified Prediction Code Examples](#5-unified-prediction-code-examples)
- [6. Complete Output Metadata Schemas](#6-complete-output-metadata-schemas)
- [7. Operational Status & Known Limitations](#7-operational-status--known-limitations)

---

## 1. Agricultural Context & Commodity Profile

Rice (*Oryza sativa*) is the foundational staple food crop of West Bengal, occupying over 70% of the state's gross cropped area and anchoring regional food security. Unlike northern grain-producing belts characterized by a single monoculture season, West Bengal cultivates rice year-round across three distinct seasonal agro-climatic regimes:
- **Autumn (Aus):** Pre-monsoon upland crop (sown April–May, harvested August–September). High vulnerability to early summer convective dryness.
- **Winter (Aman):** Main monsoon kharif crop (sown June–July, harvested November–December). Represents ~70% of total annual production; heavily reliant on southwest monsoon continuity.
- **Summer (Boro):** Irrigated winter/summer dry crop (transplanted November–December, harvested March–May). Highly productive due to high solar radiation, but vulnerable to heat waves during grain filling.

---

## 2. Pre-Harvest Biological Yield Modeling (Member 1)

Member 1 predicts district-level agricultural productivity intensity (kg/ha and tonnes/ha) across all 22 administrative districts.

### 2.1 Growing Seasons & Cutoff Windows
To provide actionable forecasts ahead of harvest without lookahead data leakage, observations are aggregated up to strict seasonal cutoffs:

| Season | Local Term | Sowing / Planting | Operational Cutoff | Harvest Window | Predictive Horizon Ahead of Harvest |
|:---|:---|:---|:---|:---|:---:|
| **Autumn** | Aus | May 1 | **July 31** | August – September | ~30–45 Days |
| **Winter** | Aman | June 1 | **September 30** | November – December | ~45–60 Days |
| **Summer** | Boro | November 1 (prior year) | **February 28** | March – May | ~30–60 Days |

### 2.2 20-Feature Specification
The certified production model (`m1_improved_delta_ridge.pkl`) accepts 20 features in strict order:
1. `Yield_lag1`: Previous year's crop yield (kg/ha) — baseline anchor
2. `Yield_lag2`: Crop yield 2 years prior (kg/ha)
3. `Area_lag1`: Previous year's planted area (ha)
4. `Area_lag2`: Planted area 2 years prior (ha)
5. `Precip_lag1`: Previous year's annual precipitation (mm)
6. `T2M_lag1`: Previous year's mean temperature (°C)
7. `RH2M_lag1`: Previous year's mean relative humidity (%)
8. `NDVI_lag1`: Previous year's mean vegetation index [-1, 1]
9. `InSeason_Precip_Cum`: Cumulative rainfall from season start to cutoff (mm)
10. `InSeason_Precip_Mean`: Daily mean rainfall across cutoff window (mm/day)
11. `InSeason_Precip_Max`: Maximum single-day rainfall across cutoff window (mm/day)
12. `InSeason_T2M_Mean`: Mean daily 2m air temperature (°C)
13. `InSeason_T2M_Min`: Minimum daily temperature observed (°C)
14. `InSeason_T2M_Max`: Maximum daily temperature observed (°C)
15. `InSeason_T2M_Range`: Diurnal temperature range (`T2M_Max` - `T2M_Min`, °C)
16. `InSeason_RH2M_Mean`: Mean daily relative humidity (%)
17. `InSeason_Solar_Mean`: Mean daily solar insolation (`ALLSKY_SFC_SW_DWN`, MJ/m²/day)
18. `InSeason_Wind_Mean`: Mean daily 2m wind speed (`WS2M`, m/s)
19. `District`: Canonical district string (e.g. 'Hooghly', 'Purba Bardhaman')
20. `Season`: Seasonal identifier ('Autumn', 'Winter', 'Summer')

### 2.3 Weather Preprocessing & NASA POWER Integration
- **Automated Weather Service:** `src/weather_service.py` connects to NASA POWER API (`power.larc.nasa.gov`), caching daily observations in SQLite (`weather_cache.db`).
- **Persistence Fallback:** If weather data is unavailable, the pipeline falls back to historical yield persistence ($\text{Yield}_{\text{lag1}}$) with method `"persistence"`.

### 2.4 Model Architecture & Production Artifact
- **Formulation:** Residual Delta on Historical Lag-1:
  $$\text{Predicted Yield (kg/ha)} = \text{Yield}_{\text{lag1}} + \widehat{\Delta}_{\text{weather}}$$
- **Model:** Regularized Ridge Regression with standard scaling.
- **Artifact:** `member1-yield-model/models/m1_improved_delta_ridge.pkl`
- **Certified SHA-256:** `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56`
- **Release Status:** **PRODUCTION READY**

### 2.5 Validation Performance vs. Baselines
Evaluated on the independent historical out-of-time holdout (2020–21 through 2022–23, $N = 192$ district-seasons):

| Metric | Delta Ridge Champion | Persistence Baseline | Net Advantage |
|:---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | **261.84 kg/ha** | 284.17 kg/ha | **+22.33 kg/ha** |
| **Root Mean Squared Error (RMSE)** | **377.73 kg/ha** | 412.30 kg/ha | **+34.57 kg/ha** |
| **$R^2$ Variance Explained** | **0.5566** | 0.4720 | **+0.0846** |

*Note: $R^2$ measures proportion of variance explained by the model, not percentage accuracy.*

### 2.6 Provisional Summer (Boro) Routing
In offline retrospective evaluations (2017–2019 test set), persistence baseline outperformed candidate ML models on Summer (Boro) by 13.41 kg/ha MAE. As an operational safeguard, the production bridge automatically routes Boro predictions to persistence (`enable_seasonal_routing=True`, `prediction_method="provisional_summer_persistence"`).

---

## 3. Post-Harvest Wholesale Mandi Market Intelligence (Member 2)

### 3.1 Market Scope & Variety Coverage
- **Wholesale Markets:** 87 APMC mandis across West Bengal (e.g., Burdwan, Sainthia, Medinipur, Kalna).
- **Varieties Tracked:** 34 commercial varieties (Common, Fine, Super Fine, Swarna, Minikit, Gobindobhog, Ratna, IR-36, etc.).
- **Historical Dataset:** 315,079 chronological records from Agmarknet (2001–2024).

### 3.2 48-Feature Point-in-Time Architecture
Price forecasting operates on **48 point-in-time features**:
- **Base Features (20):** Lags (`lag_1`, `lag_7`, `lag_14`, `lag_30`), rolling statistics (7, 14, 30 days), calendar variables, arrival volume lags.
- **Momentum Features (5):** Price velocity and acceleration ratios.
- **Trend Features (13):** Linear regression slopes, EWMA indicators, rolling medians, Z-scores.
- **History Features (10):** Directional streaks, non-zero movement counters.

### 3.3 Short-Term & Calendar Horizon Price Performance
Evaluated on independent retrospective holdout (2023-02-01 to 2024-02-01, $N=20,465$):

| Horizon | Model Architecture | Sample Size ($N$) | MAE (Rs./qtl) | RMSE (Rs./qtl) | $R^2$ Variance | Within $\pm\text{Rs.}25$/qtl | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1-Day Ahead** | Frozen 48-feat XGBoost | 17,419 | **Rs. 15.75** | 69.14 | **0.9869** | **85.76%** | **PRODUCTION** |
| **Overall 1–7D** | Frozen 48-feat XGBoost | 20,465 | **Rs. 18.98** | 85.45 | **0.9805** | **84.09%** | **PRODUCTION** |
| **14-Day Calendar**| Tuned XGBoost 14d | 18,921 | **Rs. 59.87** | 141.01 | **0.9472** | **57.30%** | **OPT-IN** |
| **30-Day Calendar**| Tuned XGBoost 30d | 17,860 | **Rs. 99.95** | 200.25 | **0.8940** | **42.57%** | **EXPERIMENTAL** |

### 3.4 Market Volatility Classification (`v1` vs. `v2_normalized`)
- **Default `v1` (Persistence):** Classifies based on historical 5-observation nominal price range.
- **Opt-in `v2_normalized` (XGBoost):** Evaluates forward price-normalized percentage range over 5 observations ($N=20,336$ holdout):
  - Overall Accuracy: **68.25%** (+1.17 pp over baseline 67.08%)
  - Balanced Accuracy: **63.20%** (+3.00 pp over baseline 60.20%)
  - Macro F1: **0.6247** (+2.26 pp over baseline 0.6021)
  - HIGH-Volatility Recall: **55.62%** (Paired McNemar $\chi^2 = 18.61, p < 0.001$)

---

## 4. Semantic Contract & Domain Data Governance

AgroVision strictly isolates three distinct agricultural metrics:
1. **Biological Crop Yield (`predicted_yield_kg_per_ha`, `predicted_yield_tonnes_per_ha`):**
   Productivity intensity predicted by Member 1 (typical West Bengal rice: 1.5–4.5 t/ha).
2. **Total Regional Production (`estimated_production_tonnes`):**
   Computed **only** when cultivated area in hectares is explicitly supplied:
   $$\text{Production (tonnes)} = \text{Yield (tonnes/ha)} \times \text{Area (ha)}$$
3. **Mandi Market Arrivals:**
   Quantity of produce physically traded at APMC wholesale yards (tonnes). Consumed by Member 2 price features; **never** predicted by Member 1.

---

## 5. Unified Prediction Code Examples

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

bridge = AgroVisionPredictor()

# 1. Predict Rice Yield (Winter Aman Season)
yield_query = {
    "District": "Hooghly",
    "Season": "Winter",
    "Yield_lag1": 2850.0,
    "Area_ha": 150.0  # Optional: enables total harvest estimation
}
yield_output = bridge.predict_yield(yield_query)
print(f"Predicted Yield: {yield_output['predicted_yield_tonnes_per_ha']} t/ha")
print(f"Total Production: {yield_output['estimated_production_tonnes']} tonnes")

# 2. Predict Mandi Price & Volatility (Short-Term 1-7D)
mandi_history = pd.DataFrame([...])  # 35+ daily Agmarknet records
price_output = bridge.predict_price(mandi_history, commodity="rice")
print(f"Predicted Price: Rs. {price_output['predicted_price_rs_quintal']}/qtl")
print(f"Volatility Regime: {price_output['volatility']}")

# 3. Predict All Combined
combined = bridge.predict_all(mandi_history, commodity="rice", yield_input=yield_query)
```

---

## 6. Complete Output Metadata Schemas

```json
{
  "commodity": "rice",
  "crop_yield": {
    "predicted_yield_kg_per_ha": 2985.40,
    "predicted_yield_tonnes_per_ha": 2.9854,
    "baseline_yield_kg_ha": 2850.00,
    "predicted_delta_kg_ha": 135.40,
    "estimated_production_tonnes": 447.81,
    "prediction_method": "m1_improved_delta_ridge",
    "weather_feature_status": "complete",
    "weather_features_used": true,
    "release_status": "PRODUCTION READY",
    "model_name": "m1_improved_delta_ridge.pkl"
  },
  "market_price": {
    "market": "Burdwan",
    "variety": "Common",
    "predicted_price_rs_quintal": 2485.50,
    "volatility": "MEDIUM",
    "volatility_probabilities": {"LOW": 0.15, "MEDIUM": 0.65, "HIGH": 0.20},
    "governance_status": "PRODUCTION APPROVED"
  }
}
```

---

## 7. Operational Status & Known Limitations

- **Yield Production Release:** Certified **`PRODUCTION READY`** (`m1_improved_delta_ridge.pkl`).
- **Short-Term Price Release:** Certified **`PRODUCTION APPROVED`** (`rice_price_model.pkl`).
- **Horizon Policy:** 14-day direct calendar horizon is **OPT-IN**; 30-day horizon is **EXPERIMENTAL**.
- **Provisional Boro Routing:** Summer persistence routing is an interim operational safeguard pending multi-cycle prospective harvest validation.
