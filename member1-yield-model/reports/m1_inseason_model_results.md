# M1 In-Season Model Results

## 1. Forecast Cutoff Definitions

| Season | Sowing | Cutoff Date | Harvest | Weather Days Available |
|---|---|---|---|---|
| **Autumn** (Aus) | May–Jun | **July 31** | Aug–Sep | ~92 days |
| **Winter** (Aman) | Jun–Jul | **Sep 30** | Nov–Dec | ~122 days |
| **Summer** (Boro) | Nov–Dec | **Feb 28** | Mar–May | ~120 days |

## 2. Dataset
- **Train:** 852 rows (2000–2014)
- **Validation:** 118 rows (2015–2016)
- **Test:** 192 rows (2017–2019)

## 3. Feature List

### Historical Features (from M1 Baseline)
`Yield_lag1`, `Yield_lag2`, `Yield_hist_mean`, `Area_lag1`, `Area_lag2`,
`Precip_lag1`, `T2M_lag1`, `RH2M_lag1`, `NDVI_lag1`

### Historical Mandi Features (corrected mapping)
`Mandi_Arrivals_Total_lag1`, `Mandi_Arrivals_Mean_lag1`, `Mandi_Price_Mean_lag1`, `Mandi_Price_Min_lag1`, `Mandi_Price_Max_lag1`, `Mandi_Price_Vol_lag1`, `Mandi_Obs_Count_lag1`, `Mandi_Price_Range_lag1`, `Mandi_Arrivals_Total_lag2`, `Mandi_Arrivals_Mean_lag2`, `Mandi_Price_Mean_lag2`, `Mandi_Price_Min_lag2`, `Mandi_Price_Max_lag2`, `Mandi_Price_Vol_lag2`, `Mandi_Obs_Count_lag2`, `Mandi_Price_Range_lag2`

### In-Season Weather Features (up to cutoff only)
`InSeason_Precip_Cum`, `InSeason_Precip_Mean`, `InSeason_Precip_Max`, `InSeason_T2M_Mean`, `InSeason_T2M_Min`, `InSeason_T2M_Max`, `InSeason_T2M_Range`, `InSeason_RH2M_Mean`, `InSeason_Solar_Mean`, `InSeason_Wind_Mean`, `InSeason_Weather_Days`

### Categorical
`District`, `Season`

**Total numerical features:** 36
**Total features incl. encoded categoricals:** 36 + encoded districts/seasons

### NDVI Note
The MODIS NDVI dataset contains only full-season aggregates. Since full-season NDVI includes
observations *after* the forecast cutoff, using it would constitute future leakage.
Only `NDVI_lag1` (previous year's NDVI) is used.

## 4. Leakage Audit
- All in-season weather observations verified to have `max_date <= cutoff_date`
- No current-year target columns in features
- All mandi features are lagged (contain `lag`)
- Train/Val/Test have zero temporal overlap
- Preprocessing fitted exclusively on training data

## 5. Validation Results (2015–2016)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence | 217.8 | 369.9 | 0.584 | 0.105 |
| Historical Mean | 441.8 | 510.7 | 0.207 | 0.177 |
| Ridge InSeason | 240.7 | 364.7 | 0.593 | 0.110 |
| RandomForest InSeason | 247.9 | 358.8 | 0.606 | 0.110 |
| XGBoost InSeason | 237.3 | 369.6 | 0.582 | 0.111 |

**Selected Model:** XGBoost

## 6. Final Test Results (2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence [Test] | 249.5 | 357.6 | 0.612 | 0.098 |
| Historical Mean [Test] | 447.4 | 536.8 | 0.124 | 0.161 |
| Ridge InSeason | 274.0 | 355.3 | 0.616 | 0.105 |
| RandomForest InSeason | 273.4 | 358.2 | 0.610 | 0.104 |
| **XGBoost InSeason** | **277.4** | **360.9** | **0.604** | **0.105** |

## 7. Cross-Experiment Comparison

| Model | Test MAE (kg/ha) | vs Persistence |
|---|---|---|
| Persistence Baseline | 249.5 | — |
| Pre-planting Ridge + Mandi | 270.2 | +20.7 worse |
| In-Season XGBoost | 277.4 | +27.9 worse |

> ⚠️ The in-season model did not beat the Persistence baseline.
> In-season weather to cutoff provides insufficient signal to overcome yield autocorrelation.

## 8. Feature Importance (Top 15)
```text
                 Feature  Importance
         Yield_hist_mean    0.248238
              Yield_lag1    0.104569
              Yield_lag2    0.030348
        District_Bankura    0.027181
         District_Howrah    0.027173
District_Purba Medinipur    0.021010
    Mandi_Price_Max_lag1    0.019302
    Mandi_Price_Max_lag2    0.018798
      InSeason_Wind_Mean    0.018371
   Mandi_Price_Mean_lag1    0.018164
        InSeason_T2M_Max    0.018070
               Area_lag2    0.017298
   Mandi_Price_Mean_lag2    0.016390
 District_Uttar Dinajpur    0.015500
         District_Maldah    0.015039
```

## 9. Limitations
- **NDVI unavailable at sub-seasonal resolution:** The MODIS dataset only has full-season
  aggregates, preventing its use as an in-season feature. Monthly NDVI composites would
  significantly improve the model.
- **Small dataset:** ~850 training rows limits complex model capacity.
- **Uniform cutoffs:** A single cutoff per season is a simplification. In practice,
  forecasts could be updated as more weather observations arrive.
