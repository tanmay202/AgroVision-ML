# AgroVision Model 1 (Rice Yield Prediction): Deployment-Readiness Audit Report

**Auditor:** Senior Machine Learning Engineer  
**Date:** November 2026  
**Target:** AgroVision Model 1 — Seasonal Rice Yield Prediction in West Bengal  
**Candidate Models Audited:**
1. `m1_improved_delta_ridge.pkl` (Primary Candidate)
2. `m1_improved_delta_xgboost_conservative.pkl` (Secondary Candidate)
3. `m1_improved_delta_xgboost.pkl` (Alternative Candidate)

---

## Executive Summary & Deployment Verdict

| Evaluation Dimension | Status | Summary Assessment |
|---|---|---|
| **Codebase & Artifact Integrity** | **PASS** | All candidate artifacts load cleanly; 1-to-1 data alignment verified; zero leakage. |
| **Benchmark Reproducibility** | **PASS** | Exact test set metrics reproduced: R² = 0.6608, MAE = 246.36 kg/ha, RMSE = 334.11 kg/ha, MAPE = 9.48%. |
| **Statistical Significance** | **FAIL / CAUTION** | Observed MAE gain is +3.09 kg/ha; **95% Bootstrap CI is [-8.87, +15.19] kg/ha (p=0.309)**. Gain over Persistence is not statistically significant. |
| **Seasonal Robustness** | **MIXED** | Strong in **Autumn (+24.11 kg/ha)**; worse than Persistence in **Summer (-13.41 kg/ha)** and **Winter (-1.21 kg/ha)**. |
| **Pipeline & Interface** | **PASS** | Reusable prediction script `predict_rice_yield.py` and bridge `src/predict.py` implemented; 25/25 automated pytest tests pass. |
| **Overall Deployment Verdict** | **CONDITIONAL** | **NOT READY FOR BLIND AUTOMATED PRODUCTION**. Ready for selective seasonal deployment (Autumn) and decision-support shadowing with Persistence fallback. |

---

## 1. Project Structure Inspection

The project repository was inspected at `member1-yield-model/` within the `AgroVision-ML` monorepo:

```
member1-yield-model/
├── data/
│   ├── raw/
│   │   ├── rice.csv                                  # Official historical district rice yield (1997-2019)
│   │   ├── weather/*.json                            # Daily NASA POWER meteorology (22 districts, 1997-2019)
│   │   └── ndvi/                                     # MODIS seasonal mean NDVI (2000-2019)
│   └── processed/
│       ├── train.csv                                 # Chronological Train split (2000-2014, 852 rows)
│       ├── validation.csv                            # Validation split (2015-2016, 118 rows)
│       ├── test.csv                                  # Untouched Test split (2017-2019, 192 rows)
│       ├── rice_yield_inseason_features_wb.csv       # Merged in-season feature table (1,162 rows)
│       └── m1_improved_predictions.csv               # Candidate test predictions & residuals
├── models/
│   ├── m1_best_ridge.pkl                             # Original Experiment 1 model (Exp1 Ridge)
│   ├── m1_inseason_xgboost.pkl                       # Original Experiment 3 model (Exp3 XGBoost)
│   ├── m1_improved_delta_ridge.pkl                   # Improved Delta Ridge (alpha=300) [PRIMARY]
│   ├── m1_improved_delta_xgboost_conservative.pkl    # Improved Delta XGBoost (depth=2) [SECONDARY]
│   └── m1_improved_delta_xgboost.pkl                 # Improved Delta XGBoost (depth=3)
├── reports/
│   ├── m1_improved_model_results.md                  # Diagnostics and tuning evaluation report
│   ├── m1_deployment_readiness_audit.md              # THIS REPORT
│   └── figures/                                      # Diagnostic residual and importance plots
├── scripts/
│   ├── train_and_evaluate_improved.py                # Pipeline trainer and evaluator
│   ├── predict_rice_yield.py                         # Standalone / CLI reusable inference pipeline
│   └── [data acquisition and feature scripts]
├── src/
│   ├── __init__.py                                   # Package initialization
│   └── predict.py                                    # Module bridge for shared/predict_bridge.py
└── tests/
    ├── test_m1_improved_models.py                    # Regression and bounds test suite (4 tests)
    └── test_predict_rice_yield.py                    # Inference pipeline test suite (12 tests)
```

**Key Findings:**
1. Clean separation between raw data, processed artifacts, scripts, and tests.
2. The original benchmark models (`m1_best_ridge.pkl` and `m1_inseason_xgboost.pkl`) have remained untouched and intact.
3. Newly created `src/predict.py` closes the integration gap with `shared/predict_bridge.py`.

---

## 2. Candidate Model Loading Verification

All three improved candidate models were inspected and unpickled using `joblib`:

1. **`m1_improved_delta_ridge.pkl`**:
   - Class: `sklearn.pipeline.Pipeline`
   - Preprocessor: `ColumnTransformer` with `SimpleImputer(strategy='median')` + `StandardScaler()` on numerical columns; `SimpleImputer(strategy='constant', fill_value='missing')` + `OneHotEncoder(handle_unknown='ignore')` on categorical columns (`District`, `Season`).
   - Estimator: `sklearn.linear_model.Ridge(alpha=300.0, random_state=42)`.
   - Loading Status: **SUCCESS**.

2. **`m1_improved_delta_xgboost_conservative.pkl`**:
   - Class: `sklearn.pipeline.Pipeline`
   - Preprocessor: `ColumnTransformer` with passthrough numerical transformer; `OneHotEncoder(handle_unknown='ignore')` on categorical columns.
   - Estimator: `xgboost.XGBRegressor(max_depth=2, learning_rate=0.03, n_estimators=100, subsample=0.8, min_child_weight=5, reg_lambda=1.0)`.
   - Loading Status: **SUCCESS**.

3. **`m1_improved_delta_xgboost.pkl`**:
   - Class: `sklearn.pipeline.Pipeline`
   - Estimator: `xgboost.XGBRegressor(max_depth=3, learning_rate=0.05, n_estimators=130, subsample=0.7, min_child_weight=5, reg_lambda=1.0)`.
   - Loading Status: **SUCCESS**.

---

## 3. Evaluation Reproduction on Test Set (2017–2019, N=192)

Model inference was executed independently against the untouched test split (192 records).

### Complete Benchmark Comparison Table

| Model Architecture | R² | MAE (kg/ha) | MAE (t/ha) | RMSE (kg/ha) | RMSE (t/ha) | MAPE (%) | Beats Persistence? |
|---|---|---|---|---|---|---|---|
| **Persistence Baseline (`Yield_lag1`)** | 0.6115 | 249.46 | 0.2495 | 357.57 | 0.3576 | 9.78% | Baseline |
| **Historical District Mean** | 0.1244 | 447.44 | 0.4474 | 536.80 | 0.5368 | 16.08% | No (-197.98) |
| **Original Baseline Ridge (Exp 1)** | 0.5812 | 288.00 | 0.2880 | 371.26 | 0.3713 | 10.67% | No (-38.54) |
| **Original In-Season XGBoost (Exp 2)** | 0.6043 | 277.37 | 0.2774 | 360.88 | 0.3609 | 10.49% | No (-27.91) |
| **Improved Delta Ridge (alpha=300)** | **0.6608** | **246.36** | **0.2464** | **334.11** | **0.3341** | **9.48%** | **YES (+3.09)** |
| **Improved Delta XGBoost (Cons d=2)** | 0.6352 | 247.10 | 0.2471 | 346.50 | 0.3465 | 9.59% | **YES (+2.36)** |
| **Improved Delta XGBoost (Tuned d=3)** | 0.6087 | 271.82 | 0.2718 | 358.84 | 0.3588 | 10.34% | No (-22.36) |
| **Improved Ensemble (50% Ridge + 50% XGB Cons)** | 0.6528 | 245.46 | 0.2455 | 338.04 | 0.3380 | 9.50% | **YES (+4.00)** |

**Verification Confirmation:** The reported metrics for Improved Delta Ridge match the test evaluation results to within four decimal places.

---

## 4. Input Features, Types, Order, and Preprocessing Requirements

The candidate models expect exactly 20 features in the following strict column order:

```python
NUMERICAL_FEATURES = [
    'Yield_lag1',             # float64: Previous year yield in kg/ha
    'Yield_lag2',             # float64: Two years prior yield in kg/ha
    'Area_lag1',              # float64: Previous year cultivated area in hectares
    'Area_lag2',              # float64: Two years prior cultivated area in hectares
    'Precip_lag1',            # float64: Previous year annual/seasonal precipitation (mm)
    'T2M_lag1',               # float64: Previous year mean temperature (°C)
    'RH2M_lag1',              # float64: Previous year relative humidity (%)
    'NDVI_lag1',              # float64: Previous year seasonal mean NDVI [0.0 - 1.0]
    'InSeason_Precip_Cum',    # float64: Cumulative rainfall up to cutoff date (mm)
    'InSeason_Precip_Mean',   # float64: Mean daily rainfall up to cutoff date (mm/day)
    'InSeason_Precip_Max',    # float64: Maximum 24-hr rainfall up to cutoff date (mm)
    'InSeason_T2M_Mean',      # float64: Mean 2-meter air temperature up to cutoff (°C)
    'InSeason_T2M_Min',       # float64: Minimum daily temperature up to cutoff (°C)
    'InSeason_T2M_Max',       # float64: Maximum daily temperature up to cutoff (°C)
    'InSeason_T2M_Range',     # float64: Temperature range (Max - Min) up to cutoff (°C)
    'InSeason_RH2M_Mean',     # float64: Mean relative humidity up to cutoff (%)
    'InSeason_Solar_Mean',    # float64: Mean all-sky solar insolation (MJ/m²/day)
    'InSeason_Wind_Mean'      # float64: Mean 2-meter wind speed (m/s)
]

CATEGORICAL_FEATURES = [
    'District',               # object/string: One of 23 West Bengal canonical districts
    'Season'                  # object/string: One of 'Autumn', 'Summer', 'Winter'
]
```

### Preprocessing Pipelines
- **For Ridge:** Numerical columns undergo `SimpleImputer(strategy='median')` followed by `StandardScaler()`. Categoricals undergo `SimpleImputer(fill_value='missing')` followed by `OneHotEncoder(handle_unknown='ignore')`.
- **For XGBoost:** Numerical columns pass through directly (allowing native missingness handling), while categoricals undergo `OneHotEncoder(handle_unknown='ignore')`.

---

## 5. Consistency of Feature Processing (NDVI, Weather, Yield)

1. **Temporal Grouping:** All historical lag features (`Yield_lag1`, `Yield_lag2`, `Area_lag1`, etc.) are computed grouped strictly by `['District', 'Season']` and sorted chronologically by `Year`.
2. **In-Season Weather Aggregations:** Daily NASA POWER weather data is filtered to exact seasonal windows:
   - **Autumn (Aus):** May 1 to July 31 (cutoff July 31, ~92 days).
   - **Winter (Aman):** June 1 to September 30 (cutoff September 30, ~122 days).
   - **Summer (Boro):** November 1 to February 28 (cutoff February 28, ~120 days).
3. **NDVI Lag Construction:** MODIS NDVI 16-day composites are averaged over the crop season window and strictly lagged by 1 year (`NDVI_lag1`), eliminating any future satellite leakage.
4. **Leakage Verification:** No current-season production, area, or post-cutoff environmental data exists in the feature vector.

---

## 6. Edge Cases, Missing Data, and Robustness Auditing

| Scenario | Risk Level | Model / Pipeline Behavior | Mitigation Implemented |
|---|---|---|---|
| **Missing In-Season Weather** | Low | Ridge imputes with training median; XGBoost routes down default branches. | Preprocessor handles gracefully. |
| **Missing / String-Null `Yield_lag1`** | **Critical** | Residual model formula $\hat{y} = \text{Yield\_lag1} + \hat{\Delta}$ cannot evaluate if lag1 is NaN. String nulls (`'nan'`, `'NaN'`, `'null'`, `'None'`, `''`) previously propagated NaNs silently. | Safe float parser `_parse_float()` normalizes all null representations. Hierarchical fallback in `predict_rice_yield.py`: `Yield_lag2` $\to$ `Yield_hist_mean` $\to$ District-Season historical mean $\to$ District mean $\to$ State mean (2533.37 kg/ha). Flags `is_yield_lag1_imputed = True` and logs warning. |
| **Infinite Physical Inputs (`inf`, `-inf`)** | **High** | Passing $\pm\infty$ caused fatal `ValueError: Input X contains infinity` in `sklearn`. | Sanitizer converts $\pm\infty$ to `np.nan`, logs explicit warning, and allows imputer to supply valid replacement. |
| **Non-Positive Historical Yields ($\le 0$)** | Medium | Biological absurdity ($0\text{ kg/ha}$ or negative) previously accepted as baseline yield. | Rejects non-positive baselines with warning and cascades to secondary historical baselines. |
| **Casing & Regional Aliases** | Medium | Variations like `'hooghly'`, `'burdwan'`, `'boro'`, `'aman'` previously zero-encoded coefficients via OHE `handle_unknown='ignore'`. | Canonical normalizer maps case variations, historical spellings (`Burdwan` $\to$ `Bardhaman`), and agronomic season names (`Boro` $\to$ `Summer`, `Aman` $\to$ `Winter`, `Aus` $\to$ `Autumn`). |
| **Unseen District / Season** | Medium | OneHotEncoder outputs all zeros (`handle_unknown='ignore'`). | Pipeline runs without error; intercept + numerical effects drive prediction. Emits warning. |
| **Explicit Non-existent Model Path** | Medium | Previously silently swallowed missing paths and loaded default Ridge. | Now raises explicit `FileNotFoundError`, while preserving legacy `yield_model.pkl` bridge alias. |
| **Out-of-Bounds Physical Inputs** | Medium | Extreme inputs (e.g., negative rainfall, NDVI > 1.0) could corrupt linear combinations. | Input validator flags physical anomalies and warns the user. |
| **Implausible Predicted Yield** | High | Severe negative delta could result in negative or sub-subsistence yield. | Agronomic clamping enforces $500 \le \hat{y} \le 6000$ kg/ha and logs warning if clamped. |


---

## 7. Operational Availability of `Yield_lag1` at Forecast Cutoff

A realistic deployment audit requires verifying whether `Yield_lag1` is actually in hand when the model must make a forecast:

| Season | Cutoff Date (Forecast Issuance) | Previous Year Crop Harvested | Theoretical Elapsed Time | Government DES Publication Reality |
|---|---|---|---|---|
| **Autumn (Aus)** | **July 31** | Aug–Sep of Year-1 | ~10–11 months | **High Availability.** Official final statistics for Year-1 Aus are usually released by April/May. |
| **Winter (Aman)** | **September 30** | Nov–Dec of Year-1 | ~9–10 months | **Moderate Availability.** Directorate of Agriculture preliminary estimates are available; final Gazette estimates can be delayed. |
| **Summer (Boro)** | **February 28** | Mar–May of Year-1 | ~9–10 months | **Moderate-to-Low Availability.** Previous summer harvest numbers may still be undergoing district consolidation in February. |

> [!WARNING] **Operational Risk:** If government statistical agencies experience reporting backlogs exceeding 10 months, `Yield_lag1` will not be available in real time. In such cases, the deployment pipeline must automatically fall back to `Yield_lag2` or provisional DES crop estimates.

---

## 8. Unit Conversion Accuracy

- **Formula:** $\text{Yield}_{\text{t/ha}} = \frac{\text{Yield}_{\text{kg/ha}}}{1000.0}$
- **Validation:** 
  - $246.36 \text{ kg/ha} = 0.24636 \text{ tonnes/ha} \approx 0.2464 \text{ t/ha}$.
  - $2850.0 \text{ kg/ha} \to 2.8500 \text{ t/ha}$.
- **Verification:** Unit tests confirm exact mathematical equivalence within machine precision across all inputs.

---

## 9. Discrepancy Resolution: Row Counts & Feature Sets

### Row Count Discrepancy (850 vs 970)
- **Train split (`data/processed/train.csv`):** 852 rows (years 2000–2014). Early documentation shorthand referred to this as "~850 rows".
- **Validation split (`data/processed/validation.csv`):** 118 rows (years 2015–2016).
- **Test split (`data/processed/test.csv`):** 192 rows (years 2017–2019).
- **Total Historical Data:** $852 + 118 + 192 = 1,162$ rows.
- **Candidate Retraining:** In `train_and_evaluate_improved.py`, candidate models were trained on **Train + Validation combined** ($852 + 118 = 970$ rows) to capture recent structural shifts prior to testing on the 192 out-of-sample test rows. This is standard ML best practice and explains the exact 970 row count.

### Feature Count Discrepancy (9 vs 25 vs 36 vs 18)
- **Baseline Model (Exp 1):** 9 numerical features + 2 categorical = 11 features.
- **Mandi Model (Exp 2):** 9 base + 16 Mandi features (8 lag1 + 8 lag2) + 2 categorical = 27 features.
- **In-Season Model (Exp 3):** 25 + 11 in-season weather features + 2 categorical = 38 features (36 numerical).
- **Improved Delta Models:** **18 numerical features + 2 categorical = 20 features.**
  - Pruned 16 noisy Mandi features (which had high test-period missingness and degraded performance).
  - Pruned `Yield_hist_mean` (which induced severe downward bias of ~364 kg/ha due to historical yield growth).
  - Pruned redundant daily weather counts (`InSeason_Weather_Days`).
  - Retained 8 clean historical lags + 10 in-season weather summary statistics.

---

## 10. Seasonal Underperformance Analysis: Summer & Winter

Analysis of test set performance broken down by season reveals striking disparities:

| Season | N | Baseline MAE (kg/ha) | Improved Ridge MAE (kg/ha) | Net Improvement | Performance Verdict |
|---|---|---|---|---|---|
| **Autumn (Aus)** | 63 | 277.46 | **253.35** | **+24.11 kg/ha** (+8.7%) | **Decisive Outperformance** |
| **Summer (Boro)** | 63 | **209.93** | 223.34 | **-13.41 kg/ha** (-6.4%) | **Underperformed Baseline** |
| **Winter (Aman)** | 66 | **260.46** | 261.67 | **-1.21 kg/ha** (-0.5%) | **Equal / Slight Underperformance** |

### Agronomic & Meteorological Root Cause Analysis

1. **Why Autumn (Aus) Succeeds:**
   - Aus rice is sown in May/June and grown during early monsoon. It is predominantly rainfed or partially irrigated.
   - Cumulative rainfall and temperature between May 1 and July 31 strongly determine seedling survival, tillering, and biomass.
   - The in-season weather features provide genuine, high-signal explanatory power.

2. **Why Summer (Boro) Fails:**
   - Boro rice is cultivated in the dry winter/spring season (Nov–May) and is **nearly 100% reliant on controlled irrigation** (groundwater shallow tube-wells, canal lift systems).
   - In-season rainfall between Nov 1 and Feb 28 in West Bengal is almost non-existent (<50 mm total). Variations in winter precipitation do not correlate with rice yields.
   - The critical yield-determining phase (flowering and panicle development) occurs in **March and April** under high solar radiation and extreme heat stress. Because the cutoff date is **February 28**, the model is completely blind to March–April heatwaves.
   - Feeding essentially dry-season weather up to Feb 28 introduces noise, making the model worse than simply repeating the prior year's yield.

3. **Why Winter (Aman) Fails:**
   - Aman is the primary monsoon crop (harvested in Nov–Dec). The cutoff date is **September 30**.
   - In coastal and Gangetic West Bengal, October and November bring post-monsoon cyclonic storms from the Bay of Bengal, causing severe lodging and localized flooding of mature standing paddy.
   - Terminal moisture stress (if the monsoon withdraws early in mid-October) is another major risk factor.
   - Because the cutoff date is September 30, the model misses all October–November extreme events.

---

## 11. Statistical Significance: Paired Bootstrap Confidence Interval

To test whether the +3.09 kg/ha improvement of Improved Delta Ridge over the Persistence Baseline is statistically meaningful or merely a sample artifact, a paired bootstrap test was conducted:

- **Resamples:** 10,000 paired bootstrap iterations on the test set ($N=192$).
- **Observed Mean Difference:** $\Delta\text{MAE} = \text{MAE}_{\text{Persistence}} - \text{MAE}_{\text{Ridge}} = 249.46 - 246.36 = +3.09 \text{ kg/ha}$.
- **95% Bootstrap Confidence Interval:** **[-8.87, +15.19] kg/ha**
- **Empirical p-value:** **$p = 0.3091$**

```
Bootstrap Difference Distribution (MAE_persistence - MAE_ridge):
Mean:  +3.11 kg/ha
2.5%:  -8.87 kg/ha  <---- Below zero!
97.5%: +15.19 kg/ha
p-value: 30.91% of resamples show Persistence beating Ridge.
```

### Statistical Conclusion
Because the 95% confidence interval spans zero (from -8.87 to +15.19 kg/ha) and the two-tailed p-value is 0.309, **the null hypothesis cannot be rejected**. The claimed improvement over the naive persistence baseline is **NOT statistically significant at the $\alpha = 0.05$ level**.

---

## 12. Prediction Pipeline Implementation

The reproducible, reusable prediction pipeline was implemented in:
- `member1-yield-model/scripts/predict_rice_yield.py`
- Package bridge: `member1-yield-model/src/predict.py`

### Key Capabilities
1. **Multi-Format Input:** Accepts single `dict`, list of dicts, or `pandas.DataFrame`.
2. **Dual-Unit Output:** Returns predictions in both `kg/ha` and `tonnes/ha`.
3. **Structured Fallback for Missing Baseline:** If `Yield_lag1` is absent, non-positive, or string-null, automatically traverses fallback chain (`Yield_lag2` $\to$ `Yield_hist_mean` $\to$ `(District, Season)` mean $\to$ District mean $\to$ Season mean $\to$ state mean) and sets `is_yield_lag1_imputed = True`.
4. **Infinite Value Sanitization:** Replaces $\pm\infty$ with `np.nan` and emits warnings, preventing `ValueError` crashes in `sklearn`.
5. **Canonical Normalization:** Normalizes casing and aliases for 23 West Bengal districts (e.g. `burdwan` $\to$ `Bardhaman`, `24 paraganas north` $\to$ `North 24 Parganas`) and seasons (`boro` $\to$ `Summer`, `aman` $\to$ `Winter`, `aus` $\to$ `Autumn`).
6. **Seasonal Policy Routing:** Implements `--enable-seasonal-routing` / `enable_seasonal_routing=True` to automatically route Summer (Boro) predictions to Persistence Baseline (`Yield_lag1`), resolving the seasonal inversion identified in Section 10.
7. **Agronomic Clamping:** Bounds predictions within $[500, 6000]$ kg/ha (can be disabled via `--no-clamp` or `clamp_yield=False`).
8. **CLI Support:** Comprehensive command-line interface with `--district`, `--season`, `--yield-lag1`, `--data` (direct JSON string), `--input`, `--output`, `--enable-seasonal-routing`, and feature parameter flags.
9. **Bridge Compatibility:** Implements `load_model()` and `predict_yield()` adhering to `shared/predict_bridge.py`, with automatic alias resolution for legacy `yield_model.pkl`.

---

## 13. Automated Test Suite Verification

A dedicated test suite was built in `member1-yield-model/tests/test_predict_rice_yield.py` and `member1-yield-model/tests/test_m1_improved_models.py`.

Execution results:
```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1, pluggy-1.6.0
collected 25 items

test_m1_improved_models.py::test_data_integrity                               PASSED [  4%]
test_m1_improved_models.py::test_original_models_preservation                 PASSED [  8%]
test_m1_improved_models.py::test_improved_models_performance                  PASSED [ 12%]
test_m1_improved_models.py::test_predictions_agronomic_bounds                 PASSED [ 16%]
test_predict_rice_yield.py::test_candidate_models_loading                     PASSED [ 20%]
test_predict_rice_yield.py::test_predict_single_dict                          PASSED [ 24%]
test_predict_rice_yield.py::test_predict_dataframe_batch                      PASSED [ 28%]
test_predict_rice_yield.py::test_predict_list_of_dicts                        PASSED [ 32%]
test_predict_rice_yield.py::test_missing_numerical_features_imputation         PASSED [ 36%]
test_predict_rice_yield.py::test_missing_yield_lag1_fallback_hierarchy         PASSED [ 40%]
test_predict_rice_yield.py::test_string_null_and_nan_imputation               PASSED [ 44%]
test_predict_rice_yield.py::test_infinite_float_sanitization                  PASSED [ 48%]
test_predict_rice_yield.py::test_non_positive_baseline_rejection              PASSED [ 52%]
test_predict_rice_yield.py::test_district_and_season_normalization            PASSED [ 56%]
test_predict_rice_yield.py::test_unseen_district_and_season                   PASSED [ 60%]
test_predict_rice_yield.py::test_agronomic_bounds_clamping                    PASSED [ 64%]
test_predict_rice_yield.py::test_invalid_input_ranges_generate_warnings       PASSED [ 68%]
test_predict_rice_yield.py::test_reproduction_accuracy                        PASSED [ 72%]
test_predict_rice_yield.py::test_nonexistent_model_path_raises_error          PASSED [ 76%]
test_predict_rice_yield.py::test_legacy_bridge_model_alias                    PASSED [ 80%]
test_predict_rice_yield.py::test_seasonal_policy_routing                      PASSED [ 84%]
test_predict_rice_yield.py::test_shared_bridge_api                            PASSED [ 88%]
test_predict_rice_yield.py::test_agrovision_predictor_bridge_integration     PASSED [ 92%]
test_predict_rice_yield.py::test_cli_execution_with_flags                     PASSED [ 96%]
test_predict_rice_yield.py::test_cli_execution_with_data_json                 PASSED [100%]

======================= 25 passed, 7 warnings in 6.11s ========================
```

All 25 tests pass without error.

---

## 14. Senior MLE Recommendations for Production Deployment

### 1. Do Not Deploy for Summer (Boro)
In Summer, the model is **13.41 kg/ha worse than simple Persistence**. 
**Action:** For `Season == 'Summer'`, route directly to the Persistence Baseline (`Yield_lag1`), or extend the cutoff date to April 15 to capture heatwave and solar radiation dynamics during grain filling.

### 2. Implement a Hybrid Routing Strategy
- **Autumn (Aus):** Deploy `m1_improved_delta_ridge.pkl` (delivers solid +24.11 kg/ha improvement).
- **Winter (Aman):** Deploy `m1_improved_delta_ridge.pkl` or `m1_improved_delta_xgboost_conservative.pkl` in ensemble with Persistence (50/50 blend).
- **Summer (Boro):** Use pure Persistence Baseline (`Yield_lag1`).

### 3. Handle Government Data Delays via Policy
Do not allow silent failure if `Yield_lag1` is delayed. The API must explicitly flag when a prediction was made using imputed historical averages rather than verified prior-year yield.

### 4. Transition to Satellite-Driven Yield Proxies
To overcome the statistical fragility demonstrated by the bootstrap test (95% CI [-8.87, +15.19]), Model 1 should incorporate in-season Sentinel-2 or MODIS NDVI vegetation condition indices (VCI) and evapotranspiration anomalies rather than relying exclusively on sparse district-level rainfall and temperature averages.
