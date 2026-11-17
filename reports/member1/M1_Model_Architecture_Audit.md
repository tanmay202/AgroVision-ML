# AgroVision Member 1: Model Architecture & Artifact Integrity Audit

**Report Identifier:** `M1-ARCH-AUDIT-2026-11`  
**Target Component:** `member1-yield-model` (Rice Crop Yield Prediction, West Bengal)  
**Lead Auditor:** Senior Agricultural Machine Learning Researcher & MLOps Engineer  
**Audit Date:** November 16, 2026  
**Status:** Certified Independent Technical Audit  

---

## 1. Executive Summary

This independent technical audit inspects the machine learning model architecture, serialized artifacts, preprocessing pipelines, and inference behavior of AgroVision Member 1 (`member1-yield-model`). Member 1 is designed to forecast district-level rice crop productivity intensity across West Bengal's three distinct agro-climatic growing seasons:
- **Autumn (*Aus*)**
- **Winter (*Aman*)**
- **Summer (*Boro*)**

The approved production model artifact under review is **`m1_improved_delta_ridge.pkl`**, managed through `models/promotion_manifest.json` and served via `member1-yield-model/scripts/predict_rice_yield.py` and `shared/predict_bridge.py`.

### Key Audit Findings
1. **Artifact Bit-Level Integrity:** The production artifact `m1_improved_delta_ridge.pkl` exactly matches its certified SHA-256 checksum (`3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56`) across `models/`, `models/staging/`, and `promotion_manifest.json`. No unauthorized tampering, overwriting, or silent alteration has occurred.
2. **Target Formulation:** Member 1 operates on a **residual delta target formulation** ($\Delta = y - \text{Yield\_lag1}$), rather than direct yield prediction ($y$). This architectural choice resolves the historical benchmark failure where direct models underperformed a naive persistence baseline.
3. **Feature Ordering Rigidity:** The model expects an exact sequence of **20 input features** (18 numerical features and 2 categorical features). Feature order and column names are enforced deterministically by the feature extractor prior to sklearn pipeline ingestion.
4. **Transform Pipeline Mechanics:** The pipeline employs a scikit-learn `ColumnTransformer` expanding the 20 raw features into **44 transformed features** (18 standardized numerical + 23 one-hot district indicators + 3 one-hot season indicators), followed by an $L_2$-regularized Ridge regressor ($\alpha = 300.0$).
5. **Inference Guarantees:** Predictions are reconstructed as $\hat{y} = \text{Yield\_lag1} + \hat{\Delta}$ and bounded by strict agronomic clamping ($[500.0, 6000.0]$ kg/ha). A hierarchical imputation protocol handles missing or non-positive baseline yields.

---

## 2. Artifact Registry & Checksum Verification

All model artifacts in `member1-yield-model/models/` were verified against cryptographic hashes recorded in `models/promotion_manifest.json` and accompanying `.metadata.json` sidecars.

| Artifact Filename | Role | Serialized Size | SHA-256 Checksum | Promotion Status | Integrity Match |
|---|---|---|---|---|---|
| `m1_improved_delta_ridge.pkl` | **Production Primary** | 6,018 bytes | `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56` | Approved Production | **VERIFIED (Exact)** |
| `m1_improved_delta_xgboost_conservative.pkl` | Candidate Alt (Depth 2) | 105,862 bytes | `9875a356d20b26e8776acd8f804b9b605397f94742ebf7253269e180f92368b7` | Promoted Alternate | **VERIFIED (Exact)** |
| `m1_improved_delta_xgboost.pkl` | Candidate Alt (Tuned Depth 3) | 150,094 bytes | `f0aefb368615fda9e094617cc914e45757601ec60e36a30d24e45760d09d2cc6` | Promoted Alternate | **VERIFIED (Exact)** |
| `m1_best_ridge.pkl` | Legacy Benchmark (Direct $y$) | 5,082 bytes | `bfa899da7edea7f1f6564b8baa5e9a5f4bd54aa672a576107677960485dc78f5` | Historical Baseline | **VERIFIED (Exact)** |
| `m1_inseason_xgboost.pkl` | Legacy Benchmark (Direct $y$) | 489,165 bytes | `6d735a1b4e8cf27ec514fb5f4e9f4887a19508f69223d932fdeefca688a9c5f5` | Historical Baseline | **VERIFIED (Exact)** |

### Certified Runtime Environment
The production artifacts were serialized under the following certified environment specifications:
- **Python:** `3.13.3`
- **scikit-learn:** `1.7.2`
- **xgboost:** `3.4.1`
- **joblib:** `1.5.2`
- **pandas:** `2.2.3`
- **numpy:** `2.2.6`
- **scipy:** `1.16.3`

Deserialization via `joblib.load()` occurs cleanly without deprecation warnings, pickling protocol errors, or C-extension incompatibilities.

---

## 3. Mathematical Architecture & Target Formulation

### 3.1 Failure of Direct Yield Formulations
Early iterations of Member 1 (Experiments 1 and 2, represented by `m1_best_ridge.pkl` and `m1_inseason_xgboost.pkl`) trained models to predict total yield $y$ directly from environmental features, Mandi price proxies, and historical rolling averages (`Yield_hist_mean`). 
On the untouched 2017–2019 test set ($N=192$), direct models exhibited significant structural errors:
- **Persistence Baseline ($\hat{y} = \text{Yield\_lag1}$):** MAE = $249.46$ kg/ha, $R^2 = 0.6115$.
- **Direct Ridge (`m1_best_ridge`):** MAE = $288.00$ kg/ha ($+38.54$ kg/ha degradation vs baseline), $R^2 = 0.5812$.
- **Direct XGBoost (`m1_inseason_xgboost`):** MAE = $277.37$ kg/ha ($+27.91$ kg/ha degradation vs baseline), $R^2 = 0.6043$.

Direct regression failed because district-level crop yields exhibit high temporal autocorrelation and strong localized spatial intercepts. When direct models attempt to fit absolute yields, the estimator wastes capacity learning static spatial level differences rather than dynamic annual anomalies driven by weather shocks. Furthermore, including expanding historical means (`Yield_hist_mean`) introduced a chronic downwards bias in an era of technological yield trend growth.

### 3.2 Residual Delta Formulation
To eliminate spatial level confounding, `m1_improved_delta_ridge.pkl` reformulates the regression objective to target the year-over-year yield delta ($\Delta$):

$$\Delta_t = y_t - \text{Yield\_lag1}_{t}$$

Where:
- $y_t$ is the actual district rice yield in kilograms per hectare (kg/ha) for year $t$ and season $s$.
- $\text{Yield\_lag1}_{t}$ is the observed rice yield from the immediately preceding year ($t-1$) for the identical district and season.

At inference time, the final predicted yield ($\hat{y}_t$) is reconstructed additively:

$$\hat{y}_t = \text{Yield\_lag1}_{t} + \hat{\Delta}_t$$

### 3.3 Agronomic Clamping
To guarantee physical validity, predictions undergo closed-interval boundary enforcement:

$$\hat{y}_{\text{final}} = \max\left(500.0, \, \min\left(6000.0, \, \hat{y}_t\right)\right)$$

- Lower bound ($500.0$ kg/ha): Below this yield, rainfed or irrigated rice in West Bengal is considered a complete crop failure / unharvested parcel.
- Upper bound ($6000.0$ kg/ha): Plausible upper physiological yield limit for semi-dwarf indica rice varieties under optimal management in the lower Gangetic plain.

---

## 4. Feature Specification & Ordering Contract

The model architecture strictly mandates **20 features** in the exact sequence specified below. Reordering, dropping, or injecting columns prior to the ColumnTransformer raises an assertion or shape mismatch error.

```python
ALL_FEATURES = [
    # --- 18 Numerical Features ---
    'Yield_lag1',           # Rice yield 1 year prior (kg/ha)
    'Yield_lag2',           # Rice yield 2 years prior (kg/ha)
    'Area_lag1',            # Harvested area 1 year prior (ha)
    'Area_lag2',            # Harvested area 2 years prior (ha)
    'Precip_lag1',          # Historical annual rainfall lag (mm)
    'T2M_lag1',             # Historical mean temperature lag (°C)
    'RH2M_lag1',            # Historical relative humidity lag (%)
    'NDVI_lag1',            # Historical satellite vegetation index lag [-1, 1]
    'InSeason_Precip_Cum',  # In-season cumulative rainfall to cutoff (mm)
    'InSeason_Precip_Mean', # In-season daily mean rainfall to cutoff (mm/day)
    'InSeason_Precip_Max',  # In-season single-day maximum rainfall (mm/day)
    'InSeason_T2M_Mean',    # In-season mean daily temperature (°C)
    'InSeason_T2M_Min',     # In-season minimum daily temperature (°C)
    'InSeason_T2M_Max',     # In-season maximum daily temperature (°C)
    'InSeason_T2M_Range',   # In-season diurnal temperature range (T2M_Max - T2M_Min) (°C)
    'InSeason_RH2M_Mean',   # In-season mean relative humidity (%)
    'InSeason_Solar_Mean',  # In-season mean solar radiation (MJ/m²/day or kW-hr/m²/day)
    'InSeason_Wind_Mean',   # In-season mean wind speed at 2m (m/s)
    # --- 2 Categorical Features ---
    'District',             # Canonical district name (23 recognized in WB)
    'Season'                # Canonical season ('Autumn', 'Summer', 'Winter')
]
```

### Excluded Feature Types (Ablation Audit)
During the improvement cycle, two major categories of noisy features were ablated:
1. **Mandi Market Features:** 16 lagged Agmarknet features (`Mandi_Arrivals_*`, `Mandi_Price_*`, `Mandi_Price_Vol_*`) were eliminated because mandi market arrivals reflect post-harvest marketing velocity and inter-district trade rather than agronomic biological field productivity.
2. **`Yield_hist_mean`:** Excluded because the expanding historical average from 2000 severely penalized districts experiencing modern yield improvements (HYV seeds, mechanization), dragging predictions downwards.

---

## 5. Pipeline Preprocessing Mechanics

The serialized pipeline `m1_improved_delta_ridge.pkl` is an instance of `sklearn.pipeline.Pipeline` with two top-level steps:
1. `'preprocessor'` (`sklearn.compose.ColumnTransformer`)
2. `'model'` (`sklearn.linear_model.Ridge`)

### Preprocessing DAG
```
Input Raw DataFrame (20 columns)
 ├── Numerical Columns (18 cols)
 │    ├── SimpleImputer(strategy='median')
 │    └── StandardScaler(with_mean=True, with_std=True)
 └── Categorical Columns (2 cols: District, Season)
      ├── SimpleImputer(strategy='constant', fill_value='missing')
      └── OneHotEncoder(handle_unknown='ignore', sparse_output=False)
 └── Feature Concatenation -> 44 Transformed Features
      └── Ridge(alpha=300.0, random_state=42) -> Predicted Delta (kg/ha)
```

### Transformed Feature Dimensionality
The `ColumnTransformer` outputs **44 continuous features**:
- **Indices 0–17 (18 features):** Scaled numerical features (`num__Yield_lag1` through `num__InSeason_Wind_Mean`).
- **Indices 18–40 (23 features):** One-hot encoded indicators for 23 West Bengal districts (`cat__District_Alipurduar` through `cat__District_Uttar Dinajpur`).
- **Indices 41–43 (3 features):** One-hot encoded indicators for the 3 rice seasons (`cat__Season_Autumn`, `cat__Season_Summer`, `cat__Season_Winter`).

Because `OneHotEncoder(handle_unknown='ignore')` is specified, any unseen district or season encountered at inference time evaluates to a zero-vector across the one-hot columns, gracefully falling back to the population intercept and standardized numerical features without crashing.

---

## 6. Model Parameters & Weight Interpretation

Inspection of the fitted Ridge estimator reveals an intercept of **$+37.47$ kg/ha** and the following coefficient vector across the 44 transformed features:

### Top Feature Weights in `m1_improved_delta_ridge.pkl`
| Feature Name | Transformed Index | Coefficient ($\beta$) | Agronomic / Statistical Interpretation |
|---|---|---|---|
| `num__Yield_lag1` | 0 | **-147.37** | **Mean Reversion:** High prior-year yield creates negative delta expectations; bumper crops rarely repeat back-to-back. |
| `num__Yield_lag2` | 1 | **+50.39** | Positive momentum / base productivity over a 2-year rolling horizon. |
| `num__InSeason_Precip_Max` | 10 | **-29.85** | High single-day deluge events cause waterlogging/lodging, depressing yield delta. |
| `num__InSeason_RH2M_Mean` | 15 | **+20.23** | High ambient relative humidity supports vegetative canopy growth in rice. |
| `num__Area_lag2` | 3 | **+18.37** | Medium-term area stability signal. |
| `num__InSeason_Solar_Mean` | 16 | **-18.08** | Excessive insolation associated with cloudless drought stress during vegetative phase. |
| `num__InSeason_T2M_Range` | 14 | **+17.84** | Wider diurnal temperature variation is beneficial for photosynthetic assimilation and night-time respiration rest. |
| `num__InSeason_T2M_Max` | 13 | **+13.56** | Warmer daytime temperatures up to vegetative threshold accelerate phenology. |
| `num__InSeason_Precip_Mean` | 9 | **-12.18** | High daily average rainfall often correlates with persistent overcast skies and reduced solar radiation. |
| `num__T2M_lag1` | 5 | **+11.51** | Prior year thermal regime baseline. |
| `cat__District_Howrah` | 26 | **-11.08** | District fixed effect: Urbanizing low-yield district. |
| `cat__Season_Autumn` | 41 | **-11.01** | Seasonal fixed effect: Aus season exhibits structurally lower delta yields due to early monsoon variability. |
| `cat__District_Purulia` | 38 | **-9.30** | District fixed effect: Drought-prone western red/laterite zone. |
| `cat__District_Maldah` | 30 | **+8.07** | District fixed effect: Productive alluvial plain. |
| `num__InSeason_Precip_Cum` | 8 | **-6.92** | Excess total monsoon precipitation slightly dampens yield deltas in flood-prone basins. |
| `cat__Season_Summer` | 42 | **+6.69** | Seasonal fixed effect: High-input Boro rice exhibits positive delta potential under controlled irrigation. |
| `cat__District_Bardhaman` | 20 | **+6.68** | Historic "Rice Bowl" district positive fixed effect. |
| `cat__District_Paschim Bardhaman` | 34 | **+6.39** | Industrial/agricultural split district positive fixed effect. |
| `cat__District_Birbhum` | 21 | **+6.25** | High-productivity alluvial zone. |
| `cat__Season_Winter` | 43 | **+4.32** | Main Aman crop season positive fixed effect. |
| `cat__District_Purba Bardhaman` | 36 | **+4.10** | Core irrigated rice tract positive fixed effect. |

The heavy negative coefficient on `Yield_lag1` ($\beta = -147.37$) is mathematically expected in residual delta regression: when a district experiences an exceptional bumper harvest in year $t-1$, yield in year $t$ typically regresses toward the long-term agronomic mean, producing a negative delta. Strong regularization ($\alpha = 300.0$) prevents coefficient explosion on collinear weather features.

---

## 7. Baseline Imputation Protocol & Missing Data Handling

In production inference, `RiceYieldPredictor._impute_yield_lag1()` implements a deterministic 7-tier hierarchical fallback when `Yield_lag1` is missing, null, non-numeric, or non-positive ($val \le 0$):

```
Hierarchy for Baseline Yield (Yield_lag1):
 ├── Tier 1: Input 'Yield_lag1' (must be finite float > 0)
 ├── Tier 2: Input 'Yield_lag2' (must be finite float > 0)
 ├── Tier 3: Input 'Yield_hist_mean' (if provided by caller and > 0)
 ├── Tier 4: Precomputed Historical District x Season Mean (DS_MEANS table)
 ├── Tier 5: Precomputed Historical District Mean (D_MEANS table)
 ├── Tier 6: Precomputed Historical Season Mean (S_MEANS table)
 └── Tier 7: Overall State Mean (2533.37 kg/ha)
```

### Table of Historical Baselines (Train+Val 2000–2016)
- **State Mean:** $2533.37$ kg/ha ($2.533$ t/ha)
- **Season Means:** Autumn: $2177.46$ kg/ha; Summer: $2998.44$ kg/ha; Winter: $2424.56$ kg/ha
- **District Means (Sample):** Bardhaman: $3010.86$ kg/ha; Purba Bardhaman: $3023.06$ kg/ha; Birbhum: $2931.41$ kg/ha; Bankura: $2763.53$ kg/ha; Purulia: $2073.31$ kg/ha; Cooch Behar: $2102.52$ kg/ha.

When an imputation tier is activated, the predictor emits an explicit warning in the return payload (`warnings` array) and sets the provenance flag `is_yield_lag1_imputed = True`.

---

## 8. Architectural Integrity Verdict

```
================================================================================
ARCHITECTURAL AUDIT VERDICT: PASS
================================================================================
Artifact SHA-256 Checksum: MATCH (100% verified across staging and production)
Dependency Compatibility:  PASS (Python 3.13.3, scikit-learn 1.7.2, joblib 1.5.2)
Target Mathematical Logic: SOUND (Additive residual delta + agronomic clamping)
Feature Contract:          STRICT (20 input features -> 44 transformed features)
Missing-Data Resiliency:   ROBUST (7-tier hierarchical imputation with audit logging)
================================================================================
```
The architecture is sound, mathematically verified, and fully stabilized for independent validation.
