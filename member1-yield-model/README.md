# Member 1: Rice Crop Yield Model (`member1-yield-model`)

> **Production ML inference pipeline for seasonal rice crop yield prediction (kg/ha and tonnes/ha) in West Bengal, India.**

Member 1 predicts district-level agricultural productivity intensity for West Bengal rice across three agro-climatic seasons:
- **Autumn (Aus)**
- **Winter (Aman)**
- **Summer (Boro)**

The production pipeline utilizes residual-target formulation:
$$\text{Predicted Yield (kg/ha)} = \text{Yield}_{\text{lag1}} + \Delta_{\text{predicted}}$$
$$\text{Predicted Yield (tonnes/ha)} = \frac{\text{Predicted Yield (kg/ha)}}{1000.0}$$

---

## 1. Feature Specifications

The production Delta ML model (`m1_improved_delta_ridge.pkl`) operates on **20 features** strictly ordered as follows:

### Historical Numerical Lags (8 features)
1. `Yield_lag1`: Previous year's crop yield (kg/ha) — baseline anchor
2. `Yield_lag2`: Crop yield 2 years prior (kg/ha)
3. `Area_lag1`: Previous year's harvested/planted area (ha)
4. `Area_lag2`: Harvested/planted area 2 years prior (ha)
5. `Precip_lag1`: Previous year's total precipitation (mm)
6. `T2M_lag1`: Previous year's mean temperature (°C)
7. `RH2M_lag1`: Previous year's mean relative humidity (%)
8. `NDVI_lag1`: Previous year's mean vegetation index [-1, 1]

### In-Season Weather Aggregates Up to Cutoff (10 features)
9. `InSeason_Precip_Cum`: Cumulative rainfall from season start to cutoff (mm)
10. `InSeason_Precip_Mean`: Daily mean rainfall across the cutoff window (mm/day)
11. `InSeason_Precip_Max`: Maximum single-day rainfall across the cutoff window (mm/day)
12. `InSeason_T2M_Mean`: Mean daily 2m air temperature (°C)
13. `InSeason_T2M_Min`: Minimum daily temperature observed up to cutoff (°C)
14. `InSeason_T2M_Max`: Maximum daily temperature observed up to cutoff (°C)
15. `InSeason_T2M_Range`: Diurnal temperature range (`InSeason_T2M_Max` - `InSeason_T2M_Min`, °C)
16. `InSeason_RH2M_Mean`: Mean daily relative humidity (%)
17. `InSeason_Solar_Mean`: Mean daily solar insolation (`ALLSKY_SFC_SW_DWN`, MJ/m²/day)
18. `InSeason_Wind_Mean`: Mean daily 2m wind speed (`WS2M`, m/s)

### Categorical Coordinates (2 features)
19. `District`: Canonical West Bengal district (e.g., 'Hooghly', 'Purba Bardhaman')
20. `Season`: Crop season ('Autumn', 'Summer', 'Winter')

---

## 2. In-Season Weather Preprocessing Interface

AgroVision strictly isolates daily meteorological observation ingestion from model inference. The reusable preprocessor is accessible at:

```python
from member1_yield_model.src.weather_preprocessor import preprocess_inseason_weather
# Or via shared bridge:
from shared.predict_bridge import preprocess_inseason_weather
```

### Season Cutoffs & Gating Window
Observations are gated strictly up to mid-season cutoffs to ensure genuine forecasting without lookahead leakage:

| Season | Local Name | Window Start | Cutoff Date | Harvest Window | Expected Days |
|---|---|---|---|---|---|
| **Autumn** | Aus | May 1 | **July 31** | August – September | ~92 days |
| **Winter** | Aman | June 1 | **September 30** | November – December | ~122 days |
| **Summer** | Boro | November 1 (prior year) | **February 28** | March – May | ~120 days |

*Zero Future-Date Leakage Guarantee:* Observations dated after the cutoff are strictly excluded from aggregation. If callers supply dates after the cutoff date, `preprocess_inseason_weather` safely drops them and emits a leakage-protection audit log.

### Preprocessing Function Signature

```python
result = preprocess_inseason_weather(
    daily_weather=df_or_records,
    district="Hooghly",
    year=2019,
    season="Winter",
    max_allowed_date=None  # Optional temporal barrier
)

features = result["features"]    # Exact 10 InSeason_* feature dict
metadata = result["metadata"]    # Completeness audit, observed days, exclusions
is_complete = result["is_complete"]  # True only if all 10 features are valid non-NaN
warnings = result["warnings"]    # Any anomaly or date exclusions
```

---

## 3. Weather Availability Policy (`weather_missing_policy`)

Production callers often submit early inference requests where in-season meteorological data is completely unavailable or partially missing:

```json
{
  "District": "Hooghly",
  "Season": "Winter",
  "Yield_lag1": 2850.0
}
```

### The Problem of Silent Imputation
Trained scikit-learn models natively fill missing features with training-set medians. Although technically executable, silent median imputation produces synthetic deviations from baseline that do **not** reflect observed weather. Treating median-imputed predictions as "weather-informed" misleads agricultural stakeholders.

### Supported Policy Modes

| Policy Mode | Trigger Condition | Behavior | Prediction Method |
|---|---|---|---|
| `"persistence"` *(Default Production)* | Missing or incomplete weather features (< 10 valid) | ML weather model bypassed; returns $\text{Yield}_{\text{lag1}}$ as yield; $\Delta = 0.0$ | `"persistence"` |
| `"persistence"` | Complete weather (all 10 features valid) | Approved Delta ML pipeline executes with observed weather features | `"improved_delta_ridge"` |
| `"model_imputation"` *(Explicit Opt-in)* | Missing or incomplete weather features | Model executes using training-set median imputation; clear warning added | `"model_imputation"` |

### Difference Between Observed Weather & Median-Imputed Weather
- **Observed Weather Inference (`weather_features_used: true`):** The prediction captures actual agronomic conditions (e.g. monsoon delay, heatwaves, drought anomalies) directly from measured NASA POWER ground data up to the cutoff date.
- **Median-Imputed Inference (`weather_features_used: false`):** The model substitutes a 17-year historical average (2000–2016) for every missing meteorological variable. The resulting prediction does **not** reflect current weather conditions and must never be cited as an active seasonal climate forecast.

---

## 4. Provisional Seasonal Policy Routing (`enable_seasonal_routing`)

In offline evaluations across the 2017–2019 test set (192 records), the Summer (Boro) crop exhibited distinct dynamics where persistence baseline ($\text{Yield}_{\text{lag1}}$) outperformed candidate ML delta models by **13.41 kg/ha** MAE.

### Policy Behavior
- When `enable_seasonal_routing=True` (production default):
  - Summer (Boro) requests are routed directly to the persistence baseline ($\text{Yield}_{\text{lag1}}$).
  - Predicted delta is set to `0.0 kg/ha`.
  - Provenance is explicitly recorded as `prediction_method="provisional_summer_persistence"`.
  - An explanatory warning is included in the response.
- Autumn (Aus) and Winter (Aman) predictions are **never** intercepted by seasonal routing and continue to use the ML delta model when weather features are supplied.
- When `enable_seasonal_routing=False`:
  - Summer predictions execute through the candidate ML model if complete weather features are present.

### Crucial Scientific Qualification
> **IMPORTANT:** The provisional Summer-to-persistence routing was motivated by observations on the 2017–2019 test set. Because this dataset influenced the architectural decision, Summer persistence routing **is not independently validated** and must not be described as statistically proven. It serves as an interim production safeguard pending rolling-origin evaluation in Step 5.

---

## 5. Prediction Metadata Schema & Examples

### Example 1: Weather-Complete Prediction (ML Model Used)

```json
{
  "predicted_yield_kg_per_ha": 3042.85,
  "predicted_yield_tonnes_per_ha": 3.0429,
  "baseline_yield_kg_ha": 2850.0,
  "predicted_delta_kg_ha": 192.85,
  "is_yield_lag1_imputed": false,
  "prediction_method": "m1_improved_delta_ridge",
  "weather_feature_status": "complete",
  "weather_features_used": true,
  "missing_weather_features": [],
  "prediction_policy_version": "2.1",
  "seasonal_routing_applied": false,
  "seasonal_routing_policy": "provisional_summer_persistence",
  "model_name": "m1_improved_delta_ridge.pkl",
  "warnings": []
}
```

### Example 2: Weather-Missing Prediction (Persistence Fallback Applied)

```json
{
  "predicted_yield_kg_per_ha": 2850.0,
  "predicted_yield_tonnes_per_ha": 2.85,
  "baseline_yield_kg_ha": 2850.0,
  "predicted_delta_kg_ha": 0.0,
  "is_yield_lag1_imputed": false,
  "prediction_method": "persistence",
  "weather_feature_status": "missing",
  "weather_features_used": false,
  "missing_weather_features": [
    "InSeason_Precip_Cum", "InSeason_Precip_Mean", "InSeason_Precip_Max",
    "InSeason_T2M_Mean", "InSeason_T2M_Min", "InSeason_T2M_Max",
    "InSeason_T2M_Range", "InSeason_RH2M_Mean", "InSeason_Solar_Mean",
    "InSeason_Wind_Mean"
  ],
  "prediction_policy_version": "2.1",
  "seasonal_routing_applied": false,
  "seasonal_routing_policy": "provisional_summer_persistence",
  "model_name": "m1_improved_delta_ridge.pkl",
  "warnings": [
    "In-season weather features unavailable; persistence fallback applied."
  ]
}
```

### Example 3: Provisional Summer Routing Applied

```json
{
  "predicted_yield_kg_per_ha": 3200.0,
  "predicted_yield_tonnes_per_ha": 3.2,
  "baseline_yield_kg_ha": 3200.0,
  "predicted_delta_kg_ha": 0.0,
  "is_yield_lag1_imputed": false,
  "prediction_method": "provisional_summer_persistence",
  "weather_feature_status": "complete",
  "weather_features_used": false,
  "missing_weather_features": [],
  "prediction_policy_version": "2.1",
  "seasonal_routing_applied": true,
  "seasonal_routing_policy": "provisional_summer_persistence",
  "model_name": "m1_improved_delta_ridge.pkl",
  "warnings": [
    "Seasonal routing active: Provisional seasonal routing policy applied — Summer (Boro) yield routed to persistence baseline. Scientific qualification: This policy is provisional and was motivated by historical 2017-2019 test set evaluations; it is not independently validated and is subject to revision after Step 5 rolling-origin evaluation.",
    "In-season weather features were supplied but ML model bypassed by provisional seasonal routing."
  ]
}
```

---

## 6. Limitations of Predictions Without In-Season Weather

When operating without in-season meteorological features under the persistence fallback policy:
1. **Climate Shocks Are Uncaptured:** Extreme weather occurrences during the growing season (e.g., severe dry spells, flood events, cyclone damage) cannot alter the point forecast.
2. **Lag-Dominance Assumption:** The prediction relies entirely on historical yield momentum ($\text{Yield}_{\text{lag1}}$). If agricultural policy, seed variety, or planting area changed significantly in the forecast year, persistence will fail to capture the deviation.
3. **Imputation Uncertainty:** When using `"model_imputation"`, median replacement masks meteorological variance and reduces forecast sharpness, tending toward unconditional historical delta norms.

---

## 7. Python API & CLI Quick Reference

### Python API (Shared Bridge)

```python
from shared.predict_bridge import AgroVisionPredictor

# Initialize production bridge (defaults: persistence fallback, provisional seasonal routing enabled)
predictor = AgroVisionPredictor(
    enable_seasonal_routing=True,
    weather_missing_policy="persistence"
)

# Single prediction
result = predictor.predict_yield({
    "District": "Hooghly",
    "Season": "Winter",
    "Yield_lag1": 2850.0,
    "Area_ha": 150.0  # Optional: computes estimated_production_tonnes
})

print(f"Yield: {result['predicted_yield_tonnes_per_ha']} t/ha")
print(f"Method: {result['prediction_method']}")
print(f"Weather Used: {result['weather_features_used']}")
```

### CLI Application

```bash
# Production inference with persistence fallback
python member1-yield-model/scripts/predict_rice_yield.py \
  --district Hooghly \
  --season Winter \
  --yield-lag1 2850.0 \
  --weather-missing-policy persistence

# Batch file inference with JSON output
python member1-yield-model/scripts/predict_rice_yield.py \
  --input data/input_batch.json \
  --output data/predictions.json \
  --weather-missing-policy persistence
```
