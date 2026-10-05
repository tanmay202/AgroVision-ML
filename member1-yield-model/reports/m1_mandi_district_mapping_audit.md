# Mandi District Mapping Audit

## 1. The Duplication / Explosion Problem
During the initial Mandi integration, historical districts that were later partitioned were mapped (exploded) to all their modern equivalents. For example, `Burdwan` was mapped to `Bardhaman`, `Purba Bardhaman`, and `Paschim Bardhaman`.

This created a severe methodological flaw: a single market's arrival data (e.g., 50 tonnes of rice arriving at Katwa market in Burdwan) was fully duplicated across three distinct geographic districts. This artificially inflated total market arrivals and applied generalized prices to entirely different agronomic zones without market-level geospatial mapping.

## 2. Affected "Exploded" Districts
The audit script (`scripts/audit_mandi_district_mapping.py`) revealed massive data inflation:

| Source (Mandi) | Targets (Yield) | Original Obs | Inflated/Duplicated | Years Affected |
|---|---|---|---|---|
| **Burdwan** | Bardhaman, Purba Bardhaman, Paschim Bardhaman | 42,933 | **+ 85,866** | 2002 - 2024 |
| **Medinipur(W)** | Paschim Medinipur, Jhargram | 18,601 | **+ 18,601** | 2003 - 2024 |
| **Jalpaiguri** | Jalpaiguri, Alipurduar | 22,777 | **+ 22,777** | 2003 - 2024 |
| **Darjeeling** | Darjeeling, Kalimpong | 8,914 | **+ 8,914** | 2003 - 2024 |

**Total Impact:** The original 319,092 rows were inflated to 455,250 rows—creating 136,158 physically impossible, duplicate market observations.

## 3. The Methodological Correction
To resolve this, the mapping was corrected in `scripts/build_m1_mandi_features.py` to be strictly 1-to-1 historically:
- `Burdwan` -> `Bardhaman`
- `Medinipur(W)` -> `Paschim Medinipur`
- `Jalpaiguri` -> `Jalpaiguri`
- `Darjeeling` -> `Darjeeling`

**Handling Modern Districts:** Without exact geospatial tracking for 87 individual markets over 23 years, we cannot definitively allocate historical `Burdwan` arrivals between modern `Purba Bardhaman` and `Paschim Bardhaman`. Therefore, the modern fragmented districts (which only appear post-2015/2017) now legitimately receive `NaN` for their Mandi features. This `NaN` is safely and natively bypassed by XGBoost or imputed via median for Ridge/RF, ensuring zero data duplication or spatial leakage.

## 4. Effect on Model Metrics

**Validation Set (2015-2016):**
- **Old (Exploded) Ridge MAE:** 235.4 kg/ha
- **Corrected Ridge MAE:** 240.0 kg/ha

**Test Set (2017-2019):**
- **Old (Exploded) Ridge MAE:** 277.7 kg/ha
- **Corrected Ridge MAE:** 270.2 kg/ha
- **Persistence (Lag1) Baseline:** 249.5 kg/ha

## 5. Conclusion
**Is the old mapping valid?** No. Exploding records mathematically duplicated 136,158 market observations, creating synthetic and spatially erroneous arrival sums. 

**Did metrics change?** Yes. Correcting the mapping actually *improved* the model's out-of-sample Test MAE (from 277.7 down to 270.2 kg/ha), as the model was no longer confused by conflicting identical market data spanning multiple distinct districts. 

**Does the ML model beat the baseline?** No. Despite the cleaner data, the Ridge model (270.2 kg/ha) still fails to outperform the pure Persistence baseline (249.5 kg/ha). Pre-planting forecasts remain overwhelmingly dominated by year-over-year yield autocorrelation, and historical market volatility provides insufficient forward-looking predictive power.
