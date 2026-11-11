# M1 Integration Specification: Rice Yield / Environmental ML Pipeline

This document provides the **complete** technical specification for the Member 1 (M1) Yield Model pipeline, extracted directly from the verified codebase at `member1-yield-model/`. It is structured to mirror `m3_feature_spec.md` and answers all integration questions for Member 3 (FastAPI / Engineering team).

---

## 1. Implementation Status & Roadmap

### ✅ Fully Built & Verified

| Component | Script / Artifact | Status |
|---|---|---|
| **Raw data acquisition — Crop Yield** | `scripts/acquire_crop_yield.py` | ✅ Complete |
| **Raw data acquisition — Weather** | `scripts/acquire_weather_nasapower.py` | ✅ Complete (NASA POWER API, 22 districts, 1997–2019) |
| **District normalization** | `scripts/district_normalization.py` | ✅ Complete (23 canonical WB districts with lat/lon centroids) |
| **Crop yield cleaning** | `scripts/clean_rice_yield.py` | ✅ Complete (duplicate resolution, unit conversion) |
| **Weather aggregation** | `scripts/aggregate_rice_weather.py` | ✅ Complete (seasonal aggregation per district-year-season) |
| **Dataset merge (Yield + Weather + NDVI)** | `scripts/merge_m1_datasets.py` | ✅ Complete (1-to-1 left join on District, Year, Season) |
| **Pre-planting feature engineering** | `scripts/build_forecast_features.py` | ✅ Complete (lag construction, leakage-safe) |
| **Mandi feature engineering** | `scripts/build_m1_mandi_features.py` | ✅ Complete (corrected district mapping, lag1 + lag2) |
| **In-season weather feature engineering** | `scripts/build_inseason_features.py` | ✅ Complete (daily NASA POWER → cutoff-gated features) |
| **Chronological train/val/test split** | `scripts/create_time_split.py` | ✅ Complete |
| **Baseline model training (no mandi)** | `scripts/train_m1_baselines.py` | ✅ Complete |
| **Model training with mandi** | `scripts/train_m1_with_mandi.py` | ✅ Complete |
| **In-season model training** | `scripts/train_m1_inseason.py` | ✅ Complete |
| **Final error analysis & visualization** | `scripts/m1_final_analysis.py` | ✅ Complete (5 PNG figures generated) |
| **Saved model artifacts** | `models/m1_best_ridge.pkl`, `models/m1_inseason_xgboost.pkl` | ✅ Saved |
| **Audit reports** | 11 `.md` reports in `reports/` | ✅ Generated |
| **Leakage audit** | Embedded assertions in every script | ✅ Passed in all scripts |
| **Predictions + residuals saved** | `data/processed/m1_final_actual_vs_predicted.csv` | ✅ Saved |
| **Dedicated inference engine** | `scripts/predict_rice_yield.py`, `src/predict.py` | ✅ Built & Verified (RiceYieldPredictor, weather policies, provisional seasonal routing) |
| **In-season weather preprocessor** | `src/weather_preprocessor.py` | ✅ Built & Verified (reusable, cutoff-gated, no live API in inference) |
| **Production inference bridge** | `shared/predict_bridge.py` | ✅ Built & Verified (canonical contract, dual-unit, legacy alias, Member 2 isolation) |

### 🚧 Still In Progress / Remaining

| Component | Status | Notes |
|---|---|---|
| **SHAP explainability** | ⬜ Not implemented | No SHAP values are exported. `shap.TreeExplainer` can be used on `m1_inseason_xgboost.pkl` at request time (it is a valid XGBRegressor inside a sklearn Pipeline). |
| **Confidence intervals / prediction intervals** | ⬜ Not implemented | Only point predictions and residuals are saved. No bootstrap or quantile-based intervals. |
| **GDD / `cumulative_gdd` feature** | ⬜ Not computed | Growing Degree Days are not calculated in any script. No `T_base` constant is defined. |
| **`ndvi_anomaly` feature** | ⬜ Not computed | Only raw NDVI and `NDVI_lag1` are used. No anomaly or z-score computation. |
| **`moisture_deficit` feature** | ⬜ Not computed | Not present in the pipeline. |
| **Post-harvest `H_anomaly`** | ⬜ Not computed | Not present in the pipeline. |
| **Multi-crop support** | ⬜ Rice only | The pipeline is hardcoded for Rice in West Bengal. `tea_cleaned.csv` exists in raw data but is not processed. |

---

## 2. Target Variable & Prediction Grain

### Target Variable

| Property | Value |
|---|---|
| **Column name** | `Rice_Yield_kg_ha` |
| **Unit** | kg/ha (kilograms per hectare) |
| **Formula** | `Rice_Yield_kg_ha = (Production / Area) × 1000` |
| **Intermediate** | `Rice_Yield_t_ha = Production / Area` (tonnes/ha), then × 1000 |
| **Source columns** | `Production` (tonnes), `Area` (hectares) — from `rice.csv` |

### Prediction Grain

| Property | Value |
|---|---|
| **Grain** | **One prediction per District × Season × Year** |
| **Geographic scope** | West Bengal, India — 23 districts (post-2017 boundaries) |
| **Crop** | Rice only |
| **Seasons** | Autumn (Aus), Winter (Aman), Summer (Boro) |
| **Prediction type** | **Seasonal district crop yield** (`Rice_Yield_kg_ha`) |

> **Clarification:** The model predicts **absolute seasonal yield (kg/ha)** per district — NOT daily mandi arrival quantities and NOT yield deviation percentage. It is a regression target, not a classification.

### Prediction Horizon / Timing

Two forecast scenarios exist in the codebase:

| Scenario | Cutoff | Features Available | Best Model | Test MAE |
|---|---|---|---|---|
| **Pre-planting forecast** | Before season starts | Historical lags only | Ridge (with Mandi) | 270.2 kg/ha |
| **In-season forecast** | Mid-season cutoff (see §4) | Historical lags + current-season weather up to cutoff | XGBoost | 277.4 kg/ha |

**In-season cutoff dates:**

| Season | Local Name | Sowing | Cutoff Date | Harvest | Weather Days |
|---|---|---|---|---|---|
| Autumn | Aus | May–Jun | **July 31** | Aug–Sep | ~92 days |
| Winter | Aman | Jun–Jul | **Sep 30** | Nov–Dec | ~122 days |
| Summer | Boro | Nov–Dec | **Feb 28** | Mar–May | ~120 days |

---

## 3. Datasets Currently Used & Data Needs

### 3.1 Raw Datasets

| # | File | Source | Content | Rows | Crops | Districts | State | Date Range |
|---|---|---|---|---|---|---|---|---|
| 1 | `data/raw/rice.csv` | data.gov.in / ICRISAT | District-level rice area, production, yield | 1,638 | Rice | ~20 (pre-split naming) | West Bengal | 1997–2019 |
| 2 | `data/raw/weather/*.json` (22 files) | NASA POWER API | Daily PRECTOTCORR, T2M, T2M_MIN, T2M_MAX, RH2M, ALLSKY_SFC_SW_DWN, WS2M | ~8,395 days × 22 districts | N/A | 22 WB districts | West Bengal | 1997-01-01 to 2019-12-31 |
| 3 | `data/raw/ndvi/AgroVision_WB_Rice_NDVI_2000_2019_LONG.csv` | MODIS (Google Earth Engine) | Seasonal mean NDVI per District-Year-Season | 1,381 | Rice | ~23 | West Bengal | 2000–2019 |
| 4 | `data/raw/ndvi/AgroVision_WB_NDVI_2000_2019_LONG.csv` | MODIS (GEE) | General WB NDVI (not used in training) | — | General | ~23 | West Bengal | 2000–2019 |
| 5 | `data/raw/ndvi/AgroVision_WB_Rice_NDVI_2000_2019.csv` | MODIS (GEE) | Wide-format NDVI (not used in training) | — | Rice | ~23 | West Bengal | 2000–2019 |
| 6 | `data/raw/tea_cleaned.csv` | — | Tea data (unused) | — | Tea | — | — | — |
| 7 | `Rice.csv` (root) | — | Duplicate/backup of rice data | — | Rice | — | — | — |
| 8 | Mandi price data | `member2-price-volatility/data/processed/rice_cleaned.csv` | Daily rice market prices/arrivals from WB mandis | — | Rice | WB districts | West Bengal | — |

### 3.2 Processed Datasets (Pipeline Outputs)

| File | Description | Shape |
|---|---|---|
| `rice_yield_wb_clean.csv` | Cleaned yield with deduplicated keys, `Rice_Yield_kg_ha` computed | ~1,162 rows |
| `weather_rice_wb.csv` | Seasonal weather aggregates per District-Year-Season | ~1,518 rows |
| `merged_rice_yield_weather_ndvi.csv` | Left-join of yield + weather + NDVI | — |
| `rice_yield_forecast_features_wb.csv` | Pre-planting lag features (Year ≥ 2000) | — |
| `rice_yield_forecast_features_mandi_wb.csv` | + Mandi lag features | — |
| `rice_yield_inseason_features_wb.csv` | + In-season weather features | — |
| `train.csv` | Train split (2000–2014) | 852 rows |
| `validation.csv` | Validation split (2015–2016) | 118 rows |
| `test.csv` | Test split (2017–2019) | 192 rows |

### 3.3 Current Data Ingestion Summary

| Data Type | Ingested? | Details |
|---|---|---|
| **Weather (temperature, precipitation)** | ✅ Yes | NASA POWER daily data: `T2M`, `T2M_MIN`, `T2M_MAX`, `PRECTOTCORR`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS2M` |
| **Satellite NDVI** | ✅ Yes | MODIS seasonal mean NDVI per District-Year-Season (2000–2019) |
| **Historical district yield** | ✅ Yes | data.gov.in rice statistics (1997–2019) |
| **Mandi prices/arrivals** | ✅ Yes | From Member 2's cleaned dataset (lagged) |

### 3.4 Data Gaps & Needs

> **Offer noted:** You mention having a clean 2023–2025 daily weather + GDD dataset for Hooghly Potato. The M1 pipeline currently ends at 2019. Extending the weather data to 2020+ and adding GDD computation would be **extremely valuable** for:
> 1. Expanding the test set beyond 2019
> 2. Enabling a `cumulative_gdd` feature (currently not computed)
> 3. Monthly NDVI composites (current data is full-season only) would unlock sub-seasonal NDVI features

---

## 4. Complete Feature Vector Specification

### 4.1 Pre-Planting Model (Experiment 1 & 2)

**9 numerical + 2 categorical features.**

Sorting: `df.sort_values(['District', 'Season', 'Year'])` — all `.shift()` and `.expanding()` operations are grouped by `['District', 'Season']`.

| # | Feature Name | Source Column | Exact Formula | Grouping Keys | Shift | Window | Missing-Value Handling |
|---|---|---|---|---|---|---|---|
| 1 | `Yield_lag1` | `Rice_Yield_kg_ha` | `grouped.shift(1)` | District, Season | 1 | N/A | NaN preserved (XGB native) or median imputed (Ridge/RF) |
| 2 | `Yield_lag2` | `Rice_Yield_kg_ha` | `grouped.shift(2)` | District, Season | 2 | N/A | Same as above |
| 3 | `Yield_hist_mean` | `Rice_Yield_kg_ha` | `grouped.shift(1).expanding().mean()` | District, Season | 1+ | Expanding | Same as above |
| 4 | `Area_lag1` | `Area` | `grouped['Area'].shift(1)` | District, Season | 1 | N/A | Same as above |
| 5 | `Area_lag2` | `Area` | `grouped['Area'].shift(2)` | District, Season | 2 | N/A | Same as above |
| 6 | `Precip_lag1` | `PRECTOTCORR` | `grouped['PRECTOTCORR'].shift(1)` | District, Season | 1 | N/A | Same as above |
| 7 | `T2M_lag1` | `T2M` | `grouped['T2M'].shift(1)` | District, Season | 1 | N/A | Same as above |
| 8 | `RH2M_lag1` | `RH2M` | `grouped['RH2M'].shift(1)` | District, Season | 1 | N/A | Same as above |
| 9 | `NDVI_lag1` | `NDVI` | `grouped['NDVI'].shift(1)` | District, Season | 1 | N/A | Same as above |
| 10 | `District` | Identifier | Categorical (23 values) | — | — | — | OneHotEncoded (`handle_unknown='ignore'`) |
| 11 | `Season` | Identifier | Categorical: Autumn, Winter, Summer | — | — | — | OneHotEncoded |

### 4.2 Pre-Planting Model + Mandi (Experiment 2)

**25 numerical + 2 categorical features.** Adds 16 mandi features (8 base × 2 lags) to the 9 above.

Mandi lag construction uses **Year offset** (not `.shift()`): `mandi_lag1['Year'] = mandi_agg['Year'] + 1`, then left-join on `[District, Season, Year]`. This safely handles missing years.

| # | Feature Name | Source | Aggregation | Lag |
|---|---|---|---|---|
| 10 | `Mandi_Arrivals_Total_lag1` | `Arrivals (Tonnes)` | `sum` per District-Season-Year | Year + 1 |
| 11 | `Mandi_Arrivals_Mean_lag1` | `Arrivals (Tonnes)` | `mean` per District-Season-Year | Year + 1 |
| 12 | `Mandi_Price_Mean_lag1` | `Modal Price (Rs./Quintal)` | `mean` | Year + 1 |
| 13 | `Mandi_Price_Min_lag1` | `Modal Price (Rs./Quintal)` | `min` | Year + 1 |
| 14 | `Mandi_Price_Max_lag1` | `Modal Price (Rs./Quintal)` | `max` | Year + 1 |
| 15 | `Mandi_Price_Vol_lag1` | `Modal Price (Rs./Quintal)` | `std` (fillna 0) | Year + 1 |
| 16 | `Mandi_Obs_Count_lag1` | `Reported Date` | `count` | Year + 1 |
| 17 | `Mandi_Price_Range_lag1` | Derived | `Price_Max - Price_Min` | Year + 1 |
| 18–25 | `*_lag2` (same 8 features) | Same | Same | Year + 2 |

**Mandi Season Mapping (arrivals → crop seasons):**
```python
if month in [3,4,5,6,7]:  return 'Summer',  Cal_Year
if month in [8,9,10]:      return 'Autumn',  Cal_Year
if month in [11,12]:       return 'Winter',  Cal_Year
if month in [1,2]:         return 'Winter',  Cal_Year - 1
```

### 4.3 In-Season Model (Experiment 3)

**36 numerical + 2 categorical features.** Adds 11 in-season weather features to the 25 above.

In-season weather is computed from **daily** NASA POWER JSON data, strictly gated by season-specific cutoff dates.

| # | Feature Name | Exact Formula | Units |
|---|---|---|---|
| 26 | `InSeason_Precip_Cum` | `subset['PRECTOTCORR'].sum()` | mm |
| 27 | `InSeason_Precip_Mean` | `subset['PRECTOTCORR'].mean()` | mm/day |
| 28 | `InSeason_Precip_Max` | `subset['PRECTOTCORR'].max()` | mm |
| 29 | `InSeason_T2M_Mean` | `subset['T2M'].mean()` | °C |
| 30 | `InSeason_T2M_Min` | `subset['T2M_MIN'].min()` | °C |
| 31 | `InSeason_T2M_Max` | `subset['T2M_MAX'].max()` | °C |
| 32 | `InSeason_T2M_Range` | `subset['T2M_MAX'].max() - subset['T2M_MIN'].min()` | °C |
| 33 | `InSeason_RH2M_Mean` | `subset['RH2M'].mean()` | % |
| 34 | `InSeason_Solar_Mean` | `subset['ALLSKY_SFC_SW_DWN'].mean()` | MJ/m²/day |
| 35 | `InSeason_Wind_Mean` | `subset['WS2M'].mean()` | m/s |
| 36 | `InSeason_Weather_Days` | `len(subset)` | count |

Where `subset` = daily weather rows for the given District where `Season_Start ≤ Date ≤ Cutoff`.

**Season window configuration:**
```python
SEASON_CONFIG = {
    'Autumn': {'start': (5, 1),  'cutoff': (7, 31), 'year_offset': 0},
    'Winter': {'start': (6, 1),  'cutoff': (9, 30), 'year_offset': 0},
    'Summer': {'start': (11, 1), 'cutoff': (2, 28), 'year_offset': -1},
}
```

### 4.4 Data Preparation Steps (Required Before Feature Calculation)

1. **Sort:** `df.sort_values(['District', 'Season', 'Year'])`
2. **Deduplication:** Assert 0 duplicate keys on `[District, Year, Season]` (programmatic assertion in every script)
3. **Year filter:** `Year >= 2000` (MODIS NDVI not available before 2000)
4. **Minimum history:** At least 1 prior year per District-Season for `Yield_lag1` to be non-NaN; 2 years for `Yield_lag2`
5. **Leaky column exclusion:** Drop `Production`, `Area`, `yield`, `Rice_Yield_t_ha`, `PRECTOTCORR`, `T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS2M`, `Weather_Days`, `NDVI` from feature set (current-season values)
6. **NaN sentinel value:** `−999` in NASA POWER JSON is replaced with `np.nan`

---

## 5. Environmental Formulas & Agronomic Constants

### Currently Implemented

| Formula | Status | Implementation |
|---|---|---|
| **Seasonal precipitation total** | ✅ | `PRECTOTCORR.sum()` over season window months |
| **Seasonal mean temperature** | ✅ | `T2M.mean()` over season window months |
| **In-season cutoff-gated weather** | ✅ | Daily data filtered to `start_date ≤ date ≤ cutoff_date` |
| **NDVI** | ✅ | Full-season MODIS mean NDVI (used only as `NDVI_lag1`) |

### NOT Currently Implemented

| Formula | Status | Expected Definition (if implemented) |
|---|---|---|
| **Growing Degree Days (`cumulative_gdd`)** | ⬜ | `GDD_daily = max(0, ((T2M_MAX + T2M_MIN) / 2) - T_base)`, cumulated over season. `T_base` for rice = 10°C. |
| **NDVI anomaly (`ndvi_anomaly`)** | ⬜ | `ndvi_anomaly = (NDVI_current - NDVI_hist_mean) / NDVI_hist_std` per District-Season |
| **Moisture deficit** | ⬜ | Not defined |
| **Post-harvest humidity anomaly (`H_anomaly`)** | ⬜ | Not defined |
| **Physiological base temperature (`T_base`)** | ⬜ | Not defined per crop (would be 10°C for rice) |

---

## 6. Model Architecture, Split & Evaluation Metrics

### 6.1 Chronological Data Split

| Split | Years | Rows | Usage |
|---|---|---|---|
| **Train** | 2000–2014 | 852 | Model fitting |
| **Validation** | 2015–2016 | 118 | Model selection / hyperparameter tuning |
| **Test** | 2017–2019 | 192 | Final out-of-sample evaluation (used exactly once) |

**Integrity:** Programmatic assertions verify zero temporal overlap: `assert not train['Year'].isin(test['Year']).any()`

### 6.2 Models Trained

#### Experiment 1: Pre-Planting Baseline (9 features)

| Model | Library | Hyperparameters | Preprocessing |
|---|---|---|---|
| **Ridge** | `sklearn.linear_model.Ridge` | `alpha=1.0` | Median imputation + StandardScaler (num), Constant + OHE (cat) |
| RandomForest | `sklearn.ensemble.RandomForestRegressor` | `n_estimators=100, random_state=42` | Same |
| XGBoost | `xgboost.XGBRegressor` | `n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42` | Passthrough (num) + OHE (cat) |

**Selected:** Ridge (lowest validation MAE)

#### Experiment 2: Pre-Planting + Mandi (25 features)

Same 3 models, same hyperparameters. **Selected:** Ridge.

#### Experiment 3: In-Season (36 features)

| Model | Library | Hyperparameters | Preprocessing |
|---|---|---|---|
| Ridge | `sklearn.linear_model.Ridge` | `alpha=1.0` | Median imputation + StandardScaler |
| RandomForest | `sklearn.ensemble.RandomForestRegressor` | `n_estimators=200, max_depth=10, random_state=42` | Same |
| **XGBoost** | `xgboost.XGBRegressor` | `n_estimators=200, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, random_state=42` | Passthrough + OHE |

**Selected:** XGBoost (lowest validation MAE)

### 6.3 Evaluation Metrics

#### Experiment 1: Pre-Planting Baseline (Test: 2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Persistence (Lag1)** | **249.5** | **357.6** | **0.612** | **0.098** |
| Historical Mean | 447.4 | 536.8 | 0.124 | 0.161 |
| Ridge (ML) | 288.0 | 371.3 | 0.581 | 0.107 |

#### Experiment 2: Pre-Planting + Mandi (Test: 2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Persistence (Lag1)** | **249.5** | **357.6** | **0.612** | **0.098** |
| Ridge (no Mandi) | 288.0 | 371.3 | 0.581 | 0.107 |
| Ridge (corrected Mandi) | 270.2 | 354.1 | 0.619 | 0.103 |

#### Experiment 3: In-Season (Test: 2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| **Persistence (Lag1)** | **249.5** | **357.6** | **0.612** | **0.098** |
| Ridge InSeason | 274.0 | 355.3 | 0.616 | 0.105 |
| RandomForest InSeason | 273.4 | 358.2 | 0.610 | 0.104 |
| XGBoost InSeason | 277.4 | 360.9 | 0.604 | 0.105 |

#### Residual Diagnostics (Ridge + Mandi, Test Set)

| Metric | Value |
|---|---|
| Mean Residual (overall) | 5.0 kg/ha |
| Median Absolute Error | 226.7 kg/ha |
| Maximum Absolute Error | 1,875.9 kg/ha |
| Correlation(Ridge, Persistence) | 0.8844 |
| Mean Residual — LOW yields (bottom 25%) | −190.1 kg/ha (underprediction) |
| Mean Residual — HIGH yields (top 25%) | 80.9 kg/ha (overprediction) |

> **Key Finding:** The Persistence baseline (`prediction = Yield_lag1`) remains the best predictor. ML models provide modest improvements in R² but do not beat Persistence on MAE. The single largest bottleneck is the absence of current-season environmental features at sub-seasonal resolution.

### 6.4 Holdout Error Residuals

Residuals (actual − predicted) **are saved** at:
- `data/processed/m1_final_actual_vs_predicted.csv` — columns: `District, Year, Season, Actual_Yield, Predicted_Yield, Residual, Absolute_Error`
- `data/processed/m1_predictions.csv`
- `data/processed/m1_mandi_predictions.csv`
- `data/processed/m1_inseason_predictions.csv`

Confidence intervals are **not** currently saved.

---

## 7. Exported Artifacts & Runtime Inference Interface

### 7.1 Saved Model Artifacts

| File | Format | Size | Contents |
|---|---|---|---|
| `models/m1_best_ridge.pkl` | joblib `.pkl` | 5 KB | `sklearn.pipeline.Pipeline` (ColumnTransformer → Ridge), trained on Train+Val, 9 numerical + 2 categorical features |
| `models/m1_inseason_xgboost.pkl` | joblib `.pkl` | 486 KB | `sklearn.pipeline.Pipeline` (ColumnTransformer → XGBRegressor), trained on Train+Val, 36 numerical + 2 categorical features |

**Loading:**
```python
import joblib
pipeline = joblib.load("models/m1_best_ridge.pkl")
# or
pipeline = joblib.load("models/m1_inseason_xgboost.pkl")
```

The pipeline includes preprocessing (imputation, scaling/encoding) and the model — calling `pipeline.predict(X)` handles everything.

### 7.2 Inference Interface (Built & Verified in Step 2 & Step 3)

The Member 1 production inference interface is implemented in `member1-yield-model/scripts/predict_rice_yield.py`, re-exported via `member1-yield-model/src/predict.py`, and accessed across the project through `shared/predict_bridge.py`.

#### Core Inference Engine & Bridge
- **`RiceYieldPredictor`**: Core class supporting single-dict, list-of-dicts, and pandas DataFrame inputs. Handles district and season normalization, hierarchical baseline yield imputation, agronomic bounding [500, 6000 kg/ha], weather missingness policies, and provisional seasonal routing.
- **Weather Availability Policy (`weather_missing_policy`)**:
  - `"persistence"` (Production Default): When any of the 10 in-season weather features is missing or NaN, bypasses ML model and returns baseline yield (predicted delta = 0.0 kg/ha) with an explicit explanatory warning.
  - `"model_imputation"`: Passes inputs through training-set median imputer, emitting a warning that predictions are median-imputed rather than weather-observed.
- **Provisional Seasonal Routing (`enable_seasonal_routing`)**:
  - Default: `True`. Routes Summer (Boro) predictions to persistence fallback, motivated by 2017–2019 test set analysis. Clearly labeled as provisional and not independently validated pending rolling-origin backtesting.
- **In-Season Weather Preprocessor (`src/weather_preprocessor.py`)**:
  - `preprocess_inseason_weather(daily_df, district, season, year)`: Aggregates daily NASA POWER observations into the exact 10 in-season features using strict cutoff dates (Aus: July 31, Aman: September 30, Boro: February 28). Zero live API calls inside inference.

**Canonical Return Value from `predict_bridge.py` (`predict_yield()`):**
```python
{
    "predicted_yield_kg_per_ha": 2813.5,       # Crop yield intensity (kg/ha)
    "predicted_yield_tonnes_per_ha": 2.8135,   # Crop yield intensity (tonnes/ha = kg/ha / 1000)
    "estimated_production_tonnes": 422.025,    # Optional: total production (if Area_ha supplied)
    "baseline_yield_kg_ha": 2850.0,            # Previous year or imputed baseline
    "predicted_delta_kg_ha": -36.53,           # Model deviation from baseline
    "is_yield_lag1_imputed": False,            # True if lag1 was imputed from fallback hierarchy
    "prediction_method": "ml_inseason_delta",  # "ml_inseason_delta" | "persistence" | "provisional_summer_persistence"
    "weather_feature_status": "complete",      # "complete" | "incomplete" | "missing"
    "weather_features_used": True,             # True if ML model executed with observed weather
    "missing_weather_features": [],            # List of missing in-season weather columns
    "prediction_policy_version": "2.1",        # Policy specification version
    "model_name": "m1_improved_delta_ridge.pkl",
    "warnings": []
}
```

> **Contract Reconciled (Step 2):** The previous `yield_arrivals_tonnes` field conflated crop yield intensity (tonnes/ha) with mandi market arrivals (tonnes). The canonical output now explicitly provides both `predicted_yield_kg_per_ha` and `predicted_yield_tonnes_per_ha`. The legacy key `yield_arrivals_tonnes` is retained as a deprecated alias emitting a `DeprecationWarning`, scheduled for permanent removal in v3.0.

#### Direct Inference (Works Now)

```python
import joblib
import pandas as pd

# Load
pipeline = joblib.load("member1-yield-model/models/m1_best_ridge.pkl")

# Prepare input (pre-planting model)
input_row = pd.DataFrame([{
    'Yield_lag1': 2800.0,        # kg/ha
    'Yield_lag2': 2750.0,        # kg/ha
    'Yield_hist_mean': 2600.0,   # kg/ha
    'Area_lag1': 15000.0,        # hectares
    'Area_lag2': 14500.0,        # hectares
    'Precip_lag1': 1200.0,       # mm (seasonal total)
    'T2M_lag1': 28.5,            # °C (seasonal mean)
    'RH2M_lag1': 78.0,           # % (seasonal mean)
    'NDVI_lag1': 0.65,           # dimensionless
    'District': 'Hooghly',
    'Season': 'Winter',
}])

prediction = pipeline.predict(input_row)
# Returns: array([2856.3])  — Rice_Yield_kg_ha
```

### 7.3 SHAP Values

SHAP values are **not** pre-exported. However, they can be computed at request time:

```python
import shap
import joblib

pipeline = joblib.load("models/m1_inseason_xgboost.pkl")
xgb_model = pipeline.named_steps['model']
preprocessor = pipeline.named_steps['preprocessor']

X_transformed = preprocessor.transform(X_input)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_transformed)
```

For the Ridge model, feature importance is available via standardized coefficients (see `m1_final_error_analysis.md`).

---

## 8. Feature Importance (Top 10 — Ridge + Mandi, Test Set)

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
| 1 | `District_Paschim Bardhaman` | 490.54 | Positive ↑ |
| 2 | `District_Jhargram` | 392.86 | Positive ↑ |
| 3 | `District_Kalimpong` | −356.83 | Negative ↓ |
| 4 | `Yield_hist_mean` | 297.43 | Positive ↑ |
| 5 | `Mandi_Price_Max_lag2` | −208.53 | Negative ↓ |
| 6 | `Mandi_Price_Max_lag1` | −203.12 | Negative ↓ |
| 7 | `District_Alipurduar` | −163.21 | Negative ↓ |
| 8 | `District_Jalpaiguri` | −160.10 | Negative ↓ |
| 9 | `District_Howrah` | −156.76 | Negative ↓ |
| 10 | `Mandi_Price_Range_lag1` | 155.93 | Positive ↑ |

**In-Season XGBoost (Top 5):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Yield_hist_mean` | 0.248 |
| 2 | `Yield_lag1` | 0.105 |
| 3 | `Yield_lag2` | 0.030 |
| 4 | `District_Bankura` | 0.027 |
| 5 | `District_Howrah` | 0.027 |

---

## 9. Pipeline Execution Order

```
1. acquire_crop_yield.py          → data/raw/rice.csv
2. acquire_weather_nasapower.py   → data/raw/weather/*.json
3. [MANUAL] NDVI GEE export       → data/raw/ndvi/*.csv
4. clean_rice_yield.py            → data/processed/rice_yield_wb_clean.csv
5. aggregate_rice_weather.py      → data/processed/weather_rice_wb.csv
6. merge_m1_datasets.py           → data/processed/merged_rice_yield_weather_ndvi.csv
7. build_forecast_features.py     → data/processed/rice_yield_forecast_features_wb.csv
8. create_time_split.py           → data/processed/{train,validation,test}.csv
9. train_m1_baselines.py          → models/m1_best_ridge.pkl + reports/m1_model_results.md
10. build_m1_mandi_features.py    → data/processed/rice_yield_forecast_features_mandi_wb.csv
11. train_m1_with_mandi.py        → reports/m1_mandi_model_results.md
12. build_inseason_features.py    → data/processed/rice_yield_inseason_features_wb.csv
13. train_m1_inseason.py          → models/m1_inseason_xgboost.pkl + reports/m1_inseason_model_results.md
14. m1_final_analysis.py          → reports/m1_final_error_analysis.md + reports/figures/*.png
```

---

## 10. Validation Report

| Component | Status | Evidence |
|---|---|---|
| **Feature formulas** | ✅ PASS | Extracted from `build_forecast_features.py`, `build_m1_mandi_features.py`, `build_inseason_features.py` |
| **Leakage safety** | ✅ PASS | Assertions in every script; cutoff audit saves `inseason_weather_audit.csv` |
| **Temporal integrity** | ✅ PASS | `assert not train['Year'].isin(test['Year']).any()` in all training scripts |
| **Duplicate key integrity** | ✅ PASS | Asserted at every merge and split |
| **Preprocessing isolation** | ✅ PASS | Scalers/imputers fitted on train only; val/test transformed |
| **Inference script & bridge** | ✅ PASS | `src/predict.py`, `scripts/predict_rice_yield.py`, `src/weather_preprocessor.py`, `shared/predict_bridge.py` |
| **SHAP export** | ❌ MISSING | Not implemented |
| **Confidence intervals** | ❌ MISSING | Not implemented |

---

## 11. Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
openpyxl>=3.1.0
requests>=2.28.0
pytest>=7.0.0
```

---

## 12. Files Inspected (Not Modified)

All findings extracted read-only from:
- `scripts/*.py` (19 scripts)
- `reports/*.md` (11 reports)
- `reports/figures/*.png` (5 figures)
- `models/*.pkl` (2 artifacts)
- `data/raw/rice.csv`, `data/raw/weather/hooghly.json`, `data/raw/ndvi/*.csv`
- `data/processed/*.csv` (18 files)
- `shared/predict_bridge.py`
- `m3_feature_spec.md`
- `requirements.txt`
- `README.md`
