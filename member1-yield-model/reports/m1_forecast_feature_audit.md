# Member 1 Forecast Feature Engineering Audit

## 1. Forecast Scenario & Cutoff
**Scenario:** Pre-Planting / Start of Season Yield Forecast
**Cutoff Date:** Day 1 of the crop season (prior to planting).

**Rationale:** The merged dataset (`merged_rice_yield_weather_ndvi.csv`) provides Weather (NASA POWER) and NDVI (MODIS) as full-season aggregates. These aggregates span the entire growing season up to harvest. Using them to forecast yield at any point *during* the season would introduce future data leakage, as the aggregate inherently contains measurements from after the forecast cutoff. 

To ensure a scientifically defensible and strictly leakage-safe model using the provided data, we must forecast at the **start of the season**, meaning no current-season data is available. All model inputs must be exclusively derived from historical (previous years') data.

## 2. Feature Definitions
All features were engineered using strict historical grouping by `District` and `Season`, sorted chronologically by `Year`, and shifted backward to ensure no overlap with the target year.

| Feature Name | Definition |
|---|---|
| `Yield_lag1` / `Yield_lag2` | The Rice Yield (kg/ha) for the exact same district and season from 1 and 2 years prior. |
| `Area_lag1` / `Area_lag2` | The harvested Area for the exact same district and season from 1 and 2 years prior. |
| `Precip_lag1` | Total seasonal precipitation (PRECTOTCORR) from the previous year. |
| `T2M_lag1` | Mean seasonal temperature (T2M) from the previous year. |
| `RH2M_lag1` | Mean seasonal humidity (RH2M) from the previous year. |
| `NDVI_lag1` | Mean seasonal NDVI from the previous year. |
| `Yield_hist_mean` | The expanding average yield for that district and season, calculated *up to but not including* the current year. |

## 3. Features Excluded (Leakage)
The following columns were explicitly dropped to prevent leakage:
- **`Production`, `yield`, `Rice_Yield_t_ha`:** Alternate target formats.
- **`Area`:** Final harvested area is not fully known on Day 1 of the season.
- **`PRECTOTCORR`, `T2M`, `T2M_MIN`, `T2M_MAX`, `RH2M`, `ALLSKY_SFC_SW_DWN`, `WS2M`, `Weather_Days`:** Current-season weather aggregates (contains future weather).
- **`NDVI`:** Current-season vegetation index (contains future crop health data).

## 4. Missing-Data Policy
- **NDVI Availability:** MODIS NDVI data collection began in 2000. Therefore, the dataset was filtered to `Year >= 2000` to establish a stable baseline where recent lags (e.g. `NDVI_lag1` for the year 2001) could be populated.
- **Undivided Bardhaman:** As established in the initial merge audit, undivided `Bardhaman` physically lacks modern environmental shapefile coverage in this dataset. Its historical weather and NDVI lags correctly remain `NaN`. 
- **Handling NaNs:** We preserved legitimate `NaN` values rather than synthesizing or interpolating data. Modern gradient boosting frameworks (like XGBoost or LightGBM) natively handle `NaN` values during training through default-direction splitting.

## 5. Leakage Validation
- **Temporal Check:** By sorting chronologically and using `shift(1)` within specific `District` + `Season` groups, it is mathematically impossible for current-year information to bleed into the feature set.
- **Explicit Dropping:** Verified via code assertion that `Production`, `Area`, `PRECTOTCORR`, and `NDVI` do not exist in the final dataframe.
- **Duplication Check:** Re-verified that the number of duplicate keys (`District` + `Year` + `Season`) equals 0. The row count did not multiply.

## 6. Final Dataset Specs
- **Final Row Count:** 1,162 rows (filtered to Year >= 2000)
- **Target Variable:** `Rice_Yield_kg_ha`
- **Final Feature List (15 columns):**
  - `State_Name` (Identifier)
  - `District` (Identifier)
  - `Year` (Identifier/Time)
  - `Season` (Identifier)
  - `Crop` (Identifier)
  - `Rice_Yield_kg_ha` (Target)
  - `Yield_lag1`
  - `Yield_lag2`
  - `Area_lag1`
  - `Area_lag2`
  - `Precip_lag1`
  - `T2M_lag1`
  - `RH2M_lag1`
  - `NDVI_lag1`
  - `Yield_hist_mean`
