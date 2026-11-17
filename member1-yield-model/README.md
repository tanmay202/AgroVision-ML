# Member 1: Multi-Crop Yield Intelligence Engine (`member1-yield-model`)

> **Production-grade agrometeorological machine learning inference pipelines for seasonal crop yield prediction (kg/ha and tonnes/ha) in West Bengal, India across Rice, Wheat, and Potato.**

---

## Table of Contents
- [1. Executive Overview](#1-executive-overview)
- [2. Production Model Registry](#2-production-model-registry)
- [3. Residual Target Architecture & Formulation](#3-residual-target-architecture--formulation)
- [4. Crop Pipeline 1: Rice (*Oryza sativa*)](#4-crop-pipeline-1-rice-oryza-sativa)
- [5. Crop Pipeline 2: Wheat (*Triticum aestivum*)](#5-crop-pipeline-2-wheat-triticum-aestivum)
- [6. Crop Pipeline 3: Potato (*Solanum tuberosum*)](#6-crop-pipeline-3-potato-solanum-tuberosum)
- [7. Evaluation Methodology & Validation Terminology](#7-evaluation-methodology--validation-terminology)
- [8. Automated NASA POWER Agrometeorological Service](#8-automated-nasa-power-agrometeorological-service)
- [9. Semantic Contract: Yield vs. Production vs. Arrivals](#9-semantic-contract-yield-vs-production-vs-arrivals)
- [10. Python API & Integration Examples](#10-python-api--integration-examples)
- [11. Complete Prediction Metadata Schemas](#11-complete-prediction-metadata-schemas)
- [12. Directory Structure](#12-directory-structure)
- [13. Verification & Automated Test Suite](#13-verification--automated-test-suite)
- [14. Known Limitations & Scientific Disclosures](#14-known-limitations--scientific-disclosures)

---

## 1. Executive Overview

Member 1 delivers automated, leak-free, district-level seasonal biological crop yield forecasting across the 22 agricultural districts of West Bengal, India. By combining multi-decadal statutory agricultural statistics from the **Directorate of Economics and Statistics (DES)** with satellite-derived agrometeorological reanalysis from **NASA POWER**, Member 1 isolates weather-driven yield anomalies from multi-year technological baselines.

The module supports **three production-ready crops**:
1. **Rice (*Oryza sativa*)**: Autumn (Aus), Winter (Aman), and Summer (Boro) seasons.
2. **Wheat (*Triticum aestivum*)**: Rabi season.
3. **Potato (*Solanum tuberosum*)**: Rabi season.

All three crop models operate under strict zero-leakage temporal cutoffs, robust missing-weather fallbacks, and reproducible cryptographic artifact protection.

---

## 2. Production Model Registry

All model artifacts are frozen, verified with cryptographic SHA-256 digests, and deployed under strict versioning control:

| Crop | Scientific Name | Architecture | Features | Season Coverage | Certified Artifact | SHA-256 Digest | Release Status |
|:---|:---|:---|:---:|:---|:---|:---|:---:|
| **Rice** | *Oryza sativa* | Delta Ridge on Historical Lag-1 | 20 | Autumn (Aus)<br>Winter (Aman)<br>Summer (Boro) | `m1_improved_delta_ridge.pkl` | `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56` | **PRODUCTION READY** |
| **Wheat** | *Triticum aestivum* | Delta Ridge on 3-Yr Rolling Mean | 18 | Rabi | `m1_wheat_delta_ridge.pkl` | `a24ee4a3e90f58186272c5e79ae55cee78a350df1af57e8024145c18607a55c0` | **PRODUCTION READY** |
| **Potato** | *Solanum tuberosum* | Delta Huber on 3-Yr Rolling Mean | 19 | Rabi | `m1_potato_delta_huber.pkl` | `44102855ffeaacd88c303070c75317b08abceaa4d426e9d56636e86cf59c1e3a` | **PRODUCTION READY** |

### Preserved Baseline Artifacts (Archival Verification)
To ensure absolute backward auditability and reproduction parity, baseline models are retained bit-for-bit:
- **Wheat Baseline:** `m1_wheat_ridge.pkl` (`SHA-256: 37755a521ef80b46e2884dc39159a350b4aab9f513136882684383a815a46376`)
- **Potato Baseline:** `m1_potato_catboost.pkl` (`SHA-256: e21a28f270350ae4dd1527546e9c7722bf94eb02ae4248a1af70ae0174974b6f`)

---

## 3. Residual Target Architecture & Formulation

Rather than predicting raw biological yield directly—which suffers from severe trend non-stationarity and regional baseline disparities—Member 1 employs **residual delta forecasting**:

### A. Rice: Delta on Lag-1 Baseline
For West Bengal rice, where year-over-year yield autocorrelation is exceptionally strong across 3 seasonal cycles:
$$\text{Predicted Yield}_{d, s, t} = \text{Yield}_{d, s, t-1} + \widehat{\Delta}_{\text{weather}}$$
$$\text{Predicted Yield (tonnes/ha)} = \frac{\text{Predicted Yield (kg/ha)}}{1000.0}$$

### B. Wheat & Potato: Delta on 3-Year Rolling Mean Baseline
For single-season winter crops (Wheat and Potato), rapid seed adoption, mechanization, and irrigation improvements create multi-year upward drift. Evaluating against a single lag ($\text{lag}_1$) introduces high noise if the prior year experienced an idiosyncratic shock. Member 1 uses a **3-year rolling mean baseline**:
$$\bar{y}_{d, t}^{\text{roll3}} = \frac{1}{3} \sum_{k=1}^{3} y_{d, t-k}$$
$$\widehat{y}_{d, t} = \bar{y}_{d, t}^{\text{roll3}} + \widehat{\Delta}_{\text{weather}}$$

#### The Tree-Saturation Breakthrough
During the Milestone 7 audit of post-2020 Potato data, tree-based gradient boosted models (such as CatBoost and Random Forest) exhibited **severe tree saturation**:
- As certified seed varieties and cold storage expansion surged West Bengal potato yields to $35,000 - 42,000\text{ kg/ha}$ in 2020–2022, tree-based models flatlined at their historical training maximums ($\approx 28,000\text{ kg/ha}$), incurring massive negative bias ($-2,984.91\text{ kg/ha}$) and high error (MAE $4,810.78\text{ kg/ha}$).
- Formulating the target as a Delta on 3-Year Rolling Mean enables linear and Huber M-estimators to extrapolate dynamically beyond historical maximums while isolating weather shocks.

---

## 4. Crop Pipeline 1: Rice (*Oryza sativa*)

### 4.1 Agronomic Context & Seasons
Rice is cultivated across three distinct seasonal agro-climatic regimes in West Bengal:
- **Autumn (Aus):** Pre-kharif upland rice; sown April–May, harvested August–September. High risk of moisture stress during early growth.
- **Winter (Aman):** Main monsoon kharif crop; sown June–July, harvested November–December. Accounts for ~70% of state production. Susceptible to monsoon withdrawal anomalies and terminal floods.
- **Summer (Boro):** Irrigated dry-season lowland rice; transplanted November–December, harvested March–May. High temperature stress during reproductive and ripening stages.

### 4.2 Exact 20-Feature Order Specification (`m1_improved_delta_ridge.pkl`)
The frozen model strictly requires 20 features in the exact sequence:
1. `Yield_lag1`: Prior year seasonal crop yield (kg/ha)
2. `Yield_lag2`: Crop yield 2 years prior (kg/ha)
3. `Area_lag1`: Prior year planted area (ha)
4. `Area_lag2`: Planted area 2 years prior (ha)
5. `Precip_lag1`: Prior year total annual precipitation (mm)
6. `T2M_lag1`: Prior year mean annual temperature (°C)
7. `RH2M_lag1`: Prior year mean relative humidity (%)
8. `NDVI_lag1`: Prior year mean vegetation index [-1, 1]
9. `InSeason_Precip_Cum`: Cumulative precipitation up to cutoff (mm)
10. `InSeason_Precip_Mean`: Daily mean precipitation up to cutoff (mm/day)
11. `InSeason_Precip_Max`: Maximum single-day precipitation up to cutoff (mm/day)
12. `InSeason_T2M_Mean`: Mean daily 2m temperature up to cutoff (°C)
13. `InSeason_T2M_Min`: Minimum daily temperature up to cutoff (°C)
14. `InSeason_T2M_Max`: Maximum daily temperature up to cutoff (°C)
15. `InSeason_T2M_Range`: Diurnal temperature range (`T2M_Max` - `T2M_Min`, °C)
16. `InSeason_RH2M_Mean`: Mean relative humidity up to cutoff (%)
17. `InSeason_Solar_Mean`: Mean daily all-sky solar insolation (`ALLSKY_SFC_SW_DWN`, MJ/m²/day)
18. `InSeason_Wind_Mean`: Mean daily 2m wind speed (`WS2M`, m/s)
19. `District`: Canonical district string (one-hot encoded internally)
20. `Season`: Seasonal identifier (`Autumn`, `Winter`, `Summer`)

### 4.3 Seasonal Weather Cutoff Windows
To guarantee operational forecasting utility prior to harvest, weather features are aggregated strictly up to mid-season cutoff dates:
- **Autumn (Aus):** Window May 1 – **July 31 Cutoff** (~92 days)
- **Winter (Aman):** Window June 1 – **September 30 Cutoff** (~122 days)
- **Summer (Boro):** Window November 1 (prior year) – **February 28 Cutoff** (~120 days)

### 4.4 Weather Missing Policy & Seasonal Routing
- **`weather_missing_policy="persistence"` (Default):** If in-season weather features are incomplete (< 10 valid), the system bypasses the ML model, returning $\text{Yield}_{\text{lag1}}$ as point forecast ($\Delta = 0.0\text{ kg/ha}$) with method `"persistence"`.
- **`weather_missing_policy="model_imputation"`:** Opt-in mode replacing missing weather features with historical training medians; logged with explicit uncertainty warnings.
- **Provisional Summer Persistence Routing (`enable_seasonal_routing=True`):** In historical validation across 2017–2019, persistence outperformed candidate ML models on Summer (Boro) by 13.41 kg/ha MAE. The production bridge automatically routes Summer requests to persistence (`prediction_method="provisional_summer_persistence"`).

### 4.5 Independent Holdout Validation (2020–2022)
Evaluated on $N=192$ independent district-season holdout observations:
- **Model MAE:** **261.84 kg/ha**
- **Persistence Baseline MAE:** 284.17 kg/ha
- **Reported Improvement:** **+22.33 kg/ha**
- **Root Mean Squared Error (RMSE):** 377.73 kg/ha
- **Coefficient of Determination ($R^2$):** **0.5566** *(Note: $R^2$ measures variance explained, not percentage accuracy)*

---

## 5. Crop Pipeline 2: Wheat (*Triticum aestivum*)

### 5.1 Agronomic Context & Growing Calendar
Wheat in West Bengal is exclusively a winter **Rabi** crop, cultivated predominantly in central and northern alluvial tracts (Murshidabad, Nadia, Malda, Birbhum, Uttar Dinajpur):
- **Sowing Window:** November 15 – December 15.
- **Vegetative & Crown Root Stage:** December.
- **Heading, Anthesis & Grain Filling:** January – February. Highly sensitive to terminal heat stress ($T_{\max} > 30^\circ\text{C}$).
- **Harvest Window:** March 15 – April 15.
- **Operational Prediction Cutoff:** **January 31** (captures vegetative development, tillering, and early grain formation ~60 days ahead of harvest).

### 5.2 Model Architecture & Feature Engineering
- **Production Artifact:** `m1_wheat_delta_ridge.pkl` (`SHA-256: a24ee4a3...`)
- **Estimator:** Ridge Regression with $L_2$ regularization parameter $\alpha = 20.0$, preceded by `StandardScaler`.
- **Feature Dimension:** **18 features**:
  - `roll3_mean`: 3-year rolling mean yield (kg/ha)
  - `yield_lag1`, `yield_lag2`, `yield_lag3`: Historical yield lags
  - `area_lag1`: Planted area in prior season (ha)
  - 10 in-season agrometeorological aggregates (October 1 to January 31 cutoff): cumulative rainfall, mean rainfall, maximum daily rainfall, mean temperature, min temperature, max temperature, diurnal range, relative humidity, solar radiation, wind speed.
  - Frequency-encoded historical district yield anchors.

### 5.3 Statutory Ground-Truth Dataset (DES APY)
Extracted directly from the Government of India DES portal (`des_west_bengal_wheat_official_1997_2022.csv`):
- **Records:** 486 district-years across 22 administrative districts (1997–2022).
- **Yield Identity Verification:** 100% mathematical parity ($\text{Yield} = \text{Production} \times 1000 / \text{Area}$) verified on all 486 rows.

### 5.4 Dual-Horizon Validation Performance
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

## 6. Crop Pipeline 3: Potato (*Solanum tuberosum*)

### 6.1 Agronomic Context & Growing Calendar
West Bengal is India's second-largest potato producer (concentrated in Hooghly, Purba Bardhaman, Paschim Medinipur, Bankura, and Jalpaiguri). Potato is a fast-growing, highly capital-intensive Rabi tuber:
- **Planting Window:** October 25 – November 30.
- **Tuber Initiation & Bulking:** December 15 – January 31. Highly sensitive to soil moisture deficits, high night temperatures ($T_{\min} > 18^\circ\text{C}$ inhibits tuberization), and late blight outbreaks triggered by high humidity ($\text{RH} > 85\%$).
- **Harvest Window:** February 10 – March 20.
- **Operational Prediction Cutoff:** **January 31** (evaluated during peak tuber bulking ~30–45 days ahead of harvest).

### 6.2 Model Architecture & Feature Engineering
- **Production Artifact:** `m1_potato_delta_huber.pkl` (`SHA-256: 44102855...`)
- **Estimator:** Robust M-estimation via `HuberRegressor` ($\alpha = 15.0, \epsilon = 1.35$), preceded by `StandardScaler`.
  - *Why Huber over Ridge?* Potato yields exhibit high non-Gaussian variance across failure years (late blight epidemics, unseasonal heavy rains) and boom years ($> 38,000\text{ kg/ha}$). The Huber loss uses squared loss for small residuals and linear loss for large deviations ($\le 1.35\sigma$), preventing rare extreme outliers from distorting regional coefficients.
- **Feature Dimension:** **19 features**:
  - `roll3_mean`: 3-year rolling mean yield (kg/ha)
  - `yield_lag1`, `yield_lag2`, `yield_lag3`: Historical yield lags
  - `area_lag1`: Planted area in prior season (ha)
  - 10 in-season agrometeorological aggregates (November 1 to January 31 cutoff).
  - Agronomic interaction indices (`temp_range_x_humidity`, `heat_stress_days`).
  - Target-encoded regional baseline yield anchors.

### 6.3 Statutory Ground-Truth Dataset (DES APY)
Extracted directly from the Government of India DES portal (`des_west_bengal_potato_official_1997_2022.csv`):
- **Records:** 496 plains district-years across 22 administrative districts (1997–2022).
- **Yield Identity Verification:** 100% mathematical parity verified on all 496 rows.

### 6.4 Dual-Horizon Validation Performance
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

*Note on Potato Error Magnitude:* Typical West Bengal potato yields range between 22,000 and 42,000 kg/ha (22–42 t/ha). An MAE of 2,981 kg/ha represents an average relative error of **~8.5%**, well within agronomically actionable planning standards.

---

## 7. Evaluation Methodology & Validation Terminology

To preserve scientific rigor, AgroVision strictly distinguishes between different evaluation protocols:

1. **Independent Historical Out-of-Time Holdout (2020–2022):**
   - Models are trained on data strictly prior to 2020.
   - The 2020–21 through 2022–23 observations were completely withheld during all training, feature selection, and regularization tuning.
   - *Limitation:* While out-of-sample in time, these are historical retrospective evaluations, **not** genuine live prospective operational forecasts.
2. **8-Year Pooled Rolling-Origin Evaluation (2015–2022):**
   - Evaluates multi-year generalization across expanding temporal training windows (e.g., train 1997–2014 $\to$ test 2015; train 1997–2015 $\to$ test 2016 ... train 1997–2021 $\to$ test 2022).
   - Metrics are pooled across all $N$ evaluated test folds.
   - **Rule:** Never combine or average holdout (2020–2022) and rolling-origin (2015–2022) populations.
3. **Live Prospective Shadow Evaluation:**
   - In-season live inference generated prior to official CCE harvest publication, logged to persistent audit database `data/monitoring/shadow_forecasts.db`.
   - Compares real-time forecasts against subsequent statutory DES publications as they become available.

---

## 8. Automated NASA POWER Agrometeorological Service

Implementation: `member1-yield-model/src/weather_service.py`

### 8.1 Meteorological Variables Ingested
- **`PRECTOTCORR`:** Daily total precipitation corrected (mm/day)
- **`T2M`:** Daily mean 2-meter air temperature (°C)
- **`T2M_MIN` / `T2M_MAX`:** Daily minimum and maximum 2m temperatures (°C)
- **`RH2M`:** Daily mean relative humidity at 2 meters (%)
- **`ALLSKY_SFC_SW_DWN`:** All-sky surface shortwave downward irradiance (MJ/m²/day)
- **`WS2M`:** Daily mean wind speed at 2 meters (m/s)

### 8.2 Architectural Safeguards
1. **Centroid Coordinate Mapping:** Hardcoded, verified latitude/longitude coordinates for all 22 West Bengal administrative districts (e.g., Hooghly: `22.90°N, 88.39°E`; Murshidabad: `24.18°N, 88.27°E`).
2. **Persistent SQLite Caching:** Local database at `data/cache/weather_cache.db` caches every queried daily observation. Subsequent inference runs execute with 0 ms network overhead.
3. **Temporal Cutoff Enforcement:** Daily observations dated after seasonal cutoffs (July 31, September 30, February 28, January 31) are automatically filtered out.
4. **Offline Archive Fallback:** Seamless fallback to pre-compiled historical archives (`data/raw/weather/`) if the external NASA API experiences rate limits or outages.

---

## 9. Semantic Contract: Yield vs. Production vs. Arrivals

AgroVision enforces strict domain separation:

1. **Crop Yield (`predicted_yield_kg_per_ha`, `predicted_yield_tonnes_per_ha`):**
   Biological land productivity predicted by Member 1.
   - Rice range: 1,500 – 4,500 kg/ha (1.5 – 4.5 t/ha)
   - Wheat range: 1,800 – 3,500 kg/ha (1.8 – 3.5 t/ha)
   - Potato range: 20,000 – 42,000 kg/ha (20.0 – 42.0 t/ha)
2. **Total Crop Production (`estimated_production_tonnes`):**
   Estimated total regional harvest volume. Calculated **only** when the caller explicitly supplies cultivated area in hectares:
   $$\text{Production (tonnes)} = \text{Yield (tonnes/ha)} \times \text{Area (ha)}$$
3. **Mandi Market Arrivals:**
   Quantity of produce physically traded at wholesale APMC markets (tonnes). This is a dynamic market variable handled exclusively by Member 2, **never** output by Member 1.

> **Deprecated Key Warning:** Accessing `yield_arrivals_tonnes` emits a `DeprecationWarning`. This legacy alias will be permanently removed in v3.0.

---

## 10. Python API & Integration Examples

### 10.1 Unified Shared Bridge (`shared/predict_bridge.py`)

```python
from shared.predict_bridge import AgroVisionPredictor

bridge = AgroVisionPredictor()

# 1. Predict Rice Yield (Winter Aman)
rice_res = bridge.predict_yield({
    "District": "Hooghly",
    "Season": "Winter",
    "Yield_lag1": 2850.0,
    "Area_ha": 120.0  # Optional: computes estimated_production_tonnes
})
print(f"Rice Yield: {rice_res['predicted_yield_tonnes_per_ha']:.2f} t/ha")
print(f"Total Production: {rice_res['estimated_production_tonnes']:.1f} tonnes")

# 2. Predict Wheat Yield (Rabi)
wheat_res = bridge.predict_wheat_yield({
    "District": "Murshidabad",
    "Year": 2023,
    "Yield_lag1": 2250.0,
    "Area_ha": 50.0
})
print(f"Wheat Yield: {wheat_res['predicted_yield_kg_per_ha']:.1f} kg/ha")

# 3. Predict Potato Yield (Rabi)
potato_res = bridge.predict_potato_yield({
    "District": "Hooghly",
    "Year": 2023,
    "Yield_lag1": 29500.0,
    "Area_ha": 25.0
})
print(f"Potato Yield: {potato_res['predicted_yield_tonnes_per_ha']:.2f} t/ha")
```

### 10.2 Dedicated Crop Predictors

```python
# Wheat Predictor
from member1_yield_model.src.wheat_predict import WheatYieldPredictor

wp = WheatYieldPredictor()
res_w = wp.predict(district="Murshidabad", year=2023, yield_lag1=2300.0)
print(res_w["predicted_yield_kg_per_ha"], res_w["prediction_interval_95"])

# Potato Predictor
from member1_yield_model.src.potato_predict import PotatoYieldPredictor

pp = PotatoYieldPredictor()
res_p = pp.predict(district="Hooghly", year=2023, yield_lag1=31000.0)
print(res_p["predicted_yield_kg_per_ha"], res_p["prediction_interval_95"])
```

### 10.3 Command-Line Interface (CLI)

```bash
# Predict Rice
python member1-yield-model/scripts/predict_rice_yield.py \
  --district Hooghly --season Winter --yield-lag1 2850.0

# Batch Inference via JSON
python member1-yield-model/scripts/predict_rice_yield.py \
  --input data/batch_input.json --output data/predictions.json
```

---

## 11. Complete Prediction Metadata Schemas

### Example 1: Weather-Complete Wheat Prediction

```json
{
  "crop": "wheat",
  "district": "Murshidabad",
  "season": "Rabi",
  "year": 2023,
  "predicted_yield_kg_per_ha": 2341.52,
  "predicted_yield_tonnes_per_ha": 2.3415,
  "baseline_yield_kg_ha": 2210.0,
  "predicted_delta_kg_ha": 131.52,
  "estimated_production_tonnes": 117.08,
  "prediction_interval_95": [2045.10, 2637.94],
  "prediction_method": "m1_wheat_delta_ridge",
  "weather_feature_status": "complete",
  "weather_features_used": true,
  "model_name": "m1_wheat_delta_ridge.pkl",
  "release_status": "PRODUCTION READY",
  "warnings": []
}
```

### Example 2: Potato Prediction with Prediction Interval

```json
{
  "crop": "potato",
  "district": "Hooghly",
  "season": "Rabi",
  "year": 2023,
  "predicted_yield_kg_per_ha": 31450.80,
  "predicted_yield_tonnes_per_ha": 31.4508,
  "baseline_yield_kg_ha": 29800.0,
  "predicted_delta_kg_ha": 1650.80,
  "estimated_production_tonnes": 786.27,
  "prediction_interval_95": [27820.00, 35081.60],
  "prediction_method": "m1_potato_delta_huber",
  "weather_feature_status": "complete",
  "weather_features_used": true,
  "model_name": "m1_potato_delta_huber.pkl",
  "release_status": "PRODUCTION READY",
  "warnings": []
}
```

---

## 12. Directory Structure

```text
member1-yield-model/
├── data/
│   ├── cache/                      ← Local SQLite weather cache (weather_cache.db)
│   ├── processed/                  ← Cleaned DES APY & NASA POWER features
│   │   ├── des_west_bengal_wheat_official_1997_2022.csv
│   │   ├── des_west_bengal_potato_official_1997_2022.csv
│   │   ├── des_wheat_full_1997_2022.csv
│   │   └── des_potato_full_1997_2022.csv
│   └── raw/                        ← Raw statutory DES CSV & HTML archives
│
├── models/                         ← Versioned and certified model artifacts
│   ├── m1_improved_delta_ridge.pkl ← Certified Rice production model
│   ├── m1_wheat_delta_ridge.pkl    ← Certified Wheat production model
│   ├── m1_potato_delta_huber.pkl   ← Certified Potato production model
│   ├── m1_wheat_ridge.pkl          ← Preserved M6 Wheat baseline
│   ├── m1_potato_catboost.pkl      ← Preserved M6 Potato baseline
│   └── promotion_manifest.json     ← Checksum & promotion registry
│
├── scripts/                        ← Training, validation & CLI inference scripts
│   ├── predict_rice_yield.py       ← Rice inference CLI
│   ├── run_blind_post2020_validation.py
│   └── run_rolling_origin_validation.py
│
├── src/                            ← Production inference & engineering modules
│   ├── __init__.py
│   ├── predict.py                  ← Rice prediction engine
│   ├── wheat_predict.py            ← Wheat prediction interface (WheatYieldPredictor)
│   ├── potato_predict.py           ← Potato prediction interface (PotatoYieldPredictor)
│   ├── shadow_monitoring.py        ← Prospective forecast audit logging
│   ├── weather_preprocessor.py     ← Leak-free weather feature engineering
│   └── weather_service.py          ← Automated NASA POWER client & SQLite cache
│
└── tests/                          ← Comprehensive automated test suite
    ├── test_m1_regression.py
    ├── test_weather_service.py
    └── ... (151 unit and regression tests)
```

---

## 13. Verification & Automated Test Suite

Member 1 maintains a **100% pass rate** across its entire regression and unit test suite:

```bash
pytest member1-yield-model/tests/ -v
# Output: 151 passed in 42.18s (100% PASS)
```

### Key Test Coverage
- **`test_model_checksums`:** Verifies SHA-256 digests of all production and baseline models.
- **`test_weather_cutoffs`:** Confirms zero observations past seasonal cutoffs contaminate features.
- **`test_prediction_contracts`:** Enforces canonical field names, unit scaling, and deprecation warnings.
- **`test_missing_weather_fallback`:** Validates exact persistence baseline output when weather features are absent.
- **`test_provisional_summer_routing`:** Verifies policy routing for Summer (Boro) rice.

---

## 14. Known Limitations & Scientific Disclosures

1. **Historical Out-of-Time vs. Prospective Operational Evaluation:** All holdout results (2020–2022) are historical simulations. Genuine live prospective validation (predicting before official harvest and evaluating after CCE publication) is actively logged in shadow mode.
2. **Provisional Summer (Boro) Routing:** Boro rice persistence routing was motivated by the 2017–2019 test set. While operationally sound, it is provisional until confirmed across multiple future harvest cycles.
3. **NASA POWER Reanalysis Latency:** NASA POWER daily data typically has a 2–4 day latency from real-time satellite telemetry. Near real-time inference within 48 hours of cutoff relies on cache persistence.
4. **Potato Absolute Error Scale:** Potato yields are large ($20,000 - 42,000\text{ kg/ha}$). Stakeholders must evaluate performance on relative percentage error (~8.5% MAPE) rather than absolute kilogram quantities.
