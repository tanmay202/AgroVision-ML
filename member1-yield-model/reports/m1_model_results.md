# Member 1 Rice Yield Model Results

## 1. Dataset Sizes
- **Train (2000-2014):** 852 rows
- **Validation (2015-2016):** 118 rows
- **Test (2017-2019):** 192 rows

## 2. Feature List
- **Categorical:** ['District', 'Season']
- **Numerical (Lags):** ['Yield_lag1', 'Yield_lag2', 'Yield_hist_mean', 'Area_lag1', 'Area_lag2', 'Precip_lag1', 'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1']
- **Target:** Rice_Yield_kg_ha
*(Strictly pre-planting historical features. No current-season environmental features were used).*

## 3. Validation Metrics (Hyperparameter / Model Selection)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | 217.8 | 369.9 | 0.584 | 0.105 |
| Historical Mean | 441.8 | 510.7 | 0.207 | 0.177 |
| Ridge | 273.2 | 373.1 | 0.574 | 0.119 |
| Random Forest | 292.4 | 398.6 | 0.514 | 0.124 |
| XGBoost | 304.3 | 447.6 | 0.387 | 0.133 |

**Selected Model:** Ridge

## 4. Final Test Metrics (Out-of-Sample: 2017-2019)
The selected Ridge model was retrained on Train + Validation, and evaluated once on the strictly untouched Test set.

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | 249.5 | 357.6 | 0.612 | 0.098 |
| Historical Mean | 447.4 | 536.8 | 0.124 | 0.161 |
| **Ridge (ML)** | **288.0** | **371.3** | **0.581** | **0.107** |

> ⚠️ **Warning:** The ML model failed to beat the best baseline (overfitting or insufficient signal).

## 5. Diagnostics

### Mean Absolute Error by Season (Test Set)
```text
Season
Summer    304.881393
Autumn    286.610113
Winter    273.221517
```

### Mean Absolute Error by District (Top 5 Worst in Test Set)
```text
District
Paschim Bardhaman    467.159120
Purba Bardhaman      449.897800
Jhargram             438.048899
Howrah               431.278812
Purba Medinipur      424.480475
```

## 6. Leakage Audit
- **Temporal Integrity:** Verified programmatically. Train (<=2014), Val (2015-2016), and Test (2017-2019) have zero temporal overlap.
- **Target Leakage:** Current-year `Production`, `Area`, `PRECTOTCORR`, and `NDVI` were strictly excluded.
- **Preprocessing:** Standard scalers, median imputers, and OneHotEncoders were fitted *exclusively* on the training sets before transforming validation/test data.

## 7. Limitations
- **Data Gap:** Undivided Bardhaman's missing environmental data is imputed to the median (Ridge/RF) or natively bypassed (XGBoost).
- **Lag Dependency:** Pre-planting forecasts rely heavily on historical autoregression. If a massive shock occurs in the *current* season (e.g., mid-season flood), the model will miss it because current-season weather is correctly withheld to prevent data leakage.
