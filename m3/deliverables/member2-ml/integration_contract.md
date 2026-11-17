# AgroVision-ML — Member 2 to Member 3 ML Integration Contract

## 1. Scope & Ownership Boundaries

- **Member 2 (Price Prediction & Volatility ML):** Owns model training, feature engineering, frozen artifact integrity, model evaluation, and the core Python inference library (`RicePricePredictor`).
- **Member 3 (Backend & Application Integration):** Owns application architecture, FastAPI / Express / Django routing, database persistence, user authentication, and the frontend web/mobile interface.

Member 2 does **not** dictate frontend framework, web server routing, or application layout. Member 2 delivers a self-contained, high-performance, pure-Python library that Member 3 can import and run anywhere.

---

## 2. Artifact Registry & Bit-for-Bit Checksums

All models are pre-trained and serialized. Member 3 must place these files in an accessible directory (default: `deliverables/member2-ml/artifacts` or configured via `AGROVISION_ARTIFACTS_DIR`):

| Artifact Filename | Type | Size | SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- | :--- |
| `rice_price_model.pkl` | Model (XGBoost) | 2.15 MB | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **Production (Frozen)** |
| `rice_price_final_config.json` | Config | 1.74 KB | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | **Production (Frozen)** |
| `rice_price_model_14d.pkl` | Model (XGBoost) | 2.15 MB | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | **Opt-In** |
| `rice_price_model_14d_config.json` | Config | 1.77 KB | `86c74c27c55bf7ff423cad5c3bced60e30a4b8d88e66bfd358fd42dcdda16da7` | **Opt-In** |
| `rice_price_model_30d.pkl` | Model (XGBoost) | 2.15 MB | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | **Experimental** |
| `rice_price_model_30d_config.json` | Config | 1.77 KB | `5650b870ab767840f0953b919efd26df6e41cc4c271c7f55dfa1911a053f280d` | **Experimental** |
| `rice_volatility_normalized_model.pkl`| Classifier (XGBoost) | 719 KB | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | **Production** |
| `rice_volatility_normalized_config.json` | Config | 2.24 KB | `c57651ff4a54f7cdde0e3199c63750ac72d09d205530643d1ff670bce1396f61` | **Production** |
| `rice_volatility_model.pkl` | Baseline | 507 B | `f1acfa394e18b59cb2b25a318bbef7fe941626e8867ecccd80fc719622979253` | Baseline |
| `rice_volatility_config.json` | Baseline Config | 323 B | `d6f321dec263ef2d2a7aa27cd41dc01c19a311af01ce3855c110b01c6e132b75` | Baseline |
| `rice_holdout_residuals_7d.json` | Holdout Residuals (T+7) | 236 KB | `e6513bee3637fcffef17d00ad0477d123b7d65699691335c4b545d585768c5dc` | **Deliverable** |
| `rice_holdout_residuals_14d.json` | Holdout Residuals (T+14) | 228 KB | `03a56701f8d201500ed5bcdaae528dfafb568cfc7c1d1d93dc6f9a78d1cc58ff` | **Deliverable** |
| `rice_holdout_residuals_30d.json` | Holdout Residuals (T+30) | 217 KB | `d28ffc61d6e2b08b9a9e970c8d98572a4aa828c8bd57737fab6b643bc0d94bd3` | **Deliverable** |
| `rice_holdout_residuals.json` | Holdout Residuals (1-7d) | 237 KB | `0c44332f80f8674b7b7f2b23857a22032760655bdb57b8a7434d410a59dee3d2` | **Deliverable** |
| `rice_group_mappings.json` | Crop Group Mapping (Rice) | 57 KB | `02432f7e77f7b017a8b4ff709952ba7e1e98165448e071557f782717eae47291` | **Deliverable** |
| `tea_group_mappings.json` | Crop Group Mapping (Tea) | 4.7 KB | `c5634ab4dd8f2dc84ee96b4d35ddc442841da4c52e2151f96a7f866722d03e61` | **Deliverable** |

---

## 3. Python API Interface

Member 3 interfaces with Member 2 deliverables through `RicePricePredictor`:

```python
from src.predictor import RicePricePredictor, InsufficientHistoryError

predictor = RicePricePredictor(artifacts_dir="path/to/artifacts")  # auto-discovers if omitted
```

### 3.1 Input History Contract

The predictor accepts historical observations either as a `pandas.DataFrame` or as a `List[Dict[str, Any]]`.

**Data Requirements:**
1. **Minimum Observations:** At least **30 valid daily observations** strictly at or before the forecast origin date $t$.
2. **Column Names / Aliases Supported:**
   - **Date:** `Reported Date`, `reported_date`, `date` (format: `YYYY-MM-DD`)
   - **Modal Price:** `Modal Price (Rs./Quintal)`, `modal_price`, `price` (numerical, in Rs./Quintal)
   - **Arrivals:** `Arrivals (Tonnes)`, `arrivals` (numerical, in Tonnes)
   - **Market:** `Market Name`, `market`, `market_name` (string, e.g. `"Burdwan"`)
   - **Variety:** `Variety`, `variety` (string, e.g. `"Common"`)

**Input Example (List of Dicts):**
```python
history = [
    {"date": "2024-01-01", "modal_price": 2500.0, "arrivals": 120.0, "market": "Burdwan", "variety": "Common"},
    {"date": "2024-01-02", "modal_price": 2510.0, "arrivals": 115.0, "market": "Burdwan", "variety": "Common"},
    # ... at least 30 observations total ...
    {"date": "2024-02-04", "modal_price": 2670.0, "arrivals": 130.0, "market": "Burdwan", "variety": "Common"}
]
```

---

## 4. Forecasting Horizons & Output Specifications

Member 2 provides three distinct price forecasting horizons:

### 4.1 Horizon 1: Short-Term Forecast (1–7 Days)

- **Method:** `predictor.predict_short_term(history, market=None, variety=None, prediction_date=None, include_volatility=True)`
- **Target:** The price of the next valid mandi observation within 1 to 7 calendar days.
- **Model:** Frozen 48-feature XGBoost regressor (`rice_price_model.pkl`).
- **Status:** `production` (default system forecast).

**Output Schema:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-02-04",
  "forecast_origin_date": "2024-02-04",
  "horizon_days": 1,
  "requested_target_date": "2024-02-05",
  "current_modal_price": 2670.0,
  "predicted_modal_price": 2692.50,
  "predicted_price_change": 22.50,
  "target_date_convention": "Next valid observation within 1–7 calendar days",
  "model_name": "rice-price-xgb",
  "model_version": "1.0.0",
  "release_status": "production",
  "limitations": "Production baseline model for short-term 1-7 day rice price forecasting in West Bengal mandis.",
  "volatility_version": "v2_normalized",
  "predicted_volatility_range": 45.0,
  "predicted_volatility_class": "MEDIUM",
  "normalized_volatility_class": "MEDIUM",
  "normalized_volatility_probabilities": {
    "LOW": 0.1250,
    "MEDIUM": 0.7420,
    "HIGH": 0.1330
  },
  "historical_pct_range_5": 1.68,
  "thresholds": {
    "low_medium_threshold": 0.014286,
    "medium_high_threshold": 1.960784
  }
}
```

---

### 4.2 Horizon 2: 14-Day Calendar Horizon Forecast

- **Method:** `predictor.predict_horizon(history, horizon_days=14, ...)` or `predictor.predict(history, horizon=14)`
- **Target:** First eligible mandi observation falling within $[t + 14, t + 17]$ calendar days.
- **Model:** 48-feature XGBoost regressor (`rice_price_model_14d.pkl`).
- **Status:** `opt-in` (verified holdout benchmark: MAE ₹59.87 vs ₹60.16 persistence baseline).

**Output Schema:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-02-04",
  "forecast_origin_date": "2024-02-04",
  "horizon_days": 14,
  "requested_target_date": "2024-02-18",
  "current_modal_price": 2670.0,
  "predicted_modal_price": 2696.28,
  "predicted_price_change": 26.28,
  "target_date_convention": "First observation in [t + 14, t + 17] calendar days within Market + Variety",
  "model_name": "rice-price-xgb-14d",
  "model_version": "1.0.0",
  "release_status": "opt-in",
  "limitations": "Opt-in release. Evaluated on 18,921 holdout observations (MAE: Rs.59.87 vs Rs.60.16 persistence, RMSE: Rs.141.01 vs Rs.146.29). Performance is validated across major rice varieties (Fine, Common, Super Fine); accuracy may vary in low-liquidity mandis.",
  "horizon_metadata": {
    "max_tolerance_days": 3,
    "train_period": ["2002-05-28", "2021-01-31"],
    "holdout_benchmark": {
      "holdout_rows": 18921,
      "model_mae": 59.87,
      "persistence_mae": 60.16,
      "model_rmse": 141.01,
      "persistence_rmse": 146.29
    }
  }
}
```

---

### 4.3 Horizon 3: 30-Day Calendar Horizon Forecast

- **Method:** `predictor.predict_horizon(history, horizon_days=30, ...)` or `predictor.predict(history, horizon=30)`
- **Target:** First eligible mandi observation falling within $[t + 30, t + 33]$ calendar days.
- **Model:** 48-feature XGBoost regressor (`rice_price_model_30d.pkl`).
- **Status:** `experimental` (outperforms persistence on moving markets: MAE ₹161.45 vs ₹164.91, RMSE ₹257.07 vs ₹263.30; parity on flat markets).

**Output Schema:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-02-04",
  "forecast_origin_date": "2024-02-04",
  "horizon_days": 30,
  "requested_target_date": "2024-03-05",
  "current_modal_price": 2670.0,
  "predicted_modal_price": 2712.10,
  "predicted_price_change": 42.10,
  "target_date_convention": "First observation in [t + 30, t + 33] calendar days within Market + Variety",
  "model_name": "rice-price-xgb-30d",
  "model_version": "1.0.0",
  "release_status": "experimental",
  "limitations": "Experimental release. Evaluated on 17,860 holdout observations. Demonstrates variance reduction on moving markets (movement MAE: Rs.161.45 vs Rs.164.91 persistence, RMSE: Rs.257.07 vs Rs.263.30), but exhibits parity on flat markets (overall MAE: Rs.99.95 vs Rs.99.50). Interpret with caution.",
  "horizon_metadata": {
    "max_tolerance_days": 3,
    "train_period": ["2002-05-28", "2021-01-31"],
    "holdout_benchmark": {
      "holdout_rows": 17860,
      "model_mae": 99.95,
      "persistence_mae": 99.50,
      "model_rmse": 204.38,
      "persistence_rmse": 208.57,
      "movement_mae": 161.45,
      "movement_persistence_mae": 164.91
    }
  }
}
```

---

## 5. Error Handling & HTTP Status Code Mapping for Member 3

When Member 3 wraps `RicePricePredictor` in an API service (e.g. FastAPI / Express), map Python exceptions to HTTP status codes as follows:

| Python Exception | Cause | Recommended HTTP Status | Recommended Client Message |
| :--- | :--- | :--- | :--- |
| `InsufficientHistoryError` | History contains fewer than 30 observations for the specified market & variety. | **400 Bad Request** | `"Insufficient historical records. At least 30 observations required to compute rolling window features."` |
| `ValueError` | Unsupported horizon (e.g. 7 or 45), missing columns, or invalid dates. | **400 Bad Request** | Clear explanation of the invalid argument (e.g. `"Unsupported horizon_days: 45. Supported: 14, 30."`). |
| `FileNotFoundError` | Missing `.pkl` model artifact or `.json` config file. | **503 Service Unavailable** | `"ML model artifact unavailable on the server. Please check artifact deployment."` |
| `TypeError` | History input is not a DataFrame or list of dicts. | **400 Bad Request** | `"Invalid payload format. Expected array of objects or table."` |

---

## 6. Guidance for Member 3 UI Integration

1. **Default Horizon:** Keep the 1–7 day short-term forecast as the default active selection in the UI.
2. **Horizon Selector:** If implementing a horizon switcher (e.g., tabs or segmented buttons):
   - Tab 1: **Short-Term (1–7 Days)** — Default.
   - Tab 2: **14-Day Forecast** — Labeled `Opt-In`.
   - Tab 3: **30-Day Forecast** — Explicitly labeled with an amber `[EXPERIMENTAL]` badge.
3. **Target Date Visualization:** Always display `requested_target_date` and the `target_date_convention` to avoid misleading farmers into believing intermediate daily prices were forecasted.
4. **Authentic History:** Never fabricate historical prices in the client. Supply genuine historical observations to ensure realistic feature construction.

---

## 7. Prediction Interval Construction Using Holdout Residuals

Member 2 has exported final holdout error residuals computed strictly on out-of-sample test data (`2023-02-01` to `2024-02-01`):

- `rice_holdout_residuals_7d.json` (for 7-day forecasts)
- `rice_holdout_residuals_14d.json` (for 14-day forecasts)
- `rice_holdout_residuals_30d.json` (for 30-day forecasts)
- `rice_holdout_residuals.json` (for legacy/short-term 1–7 day forecasts)

### 7.1 Schema of Residual JSON Artifacts
Each file contains full model provenance, target definitions, summary statistics, empirical quantiles, and the complete residual list:
```json
{
  "crop": "rice",
  "horizon_days": 14,
  "model_version": "1.0.0-production-14d",
  "model_artifact": "rice_price_model_14d.pkl",
  "model_artifact_sha256": "1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab",
  "target_definition": "First observation in forward window [T+14, T+17] calendar days within Market + Variety",
  "holdout_period": {
    "start": "2023-02-01",
    "end": "2024-02-01"
  },
  "n_holdout_samples": 18921,
  "metrics": {
    "mae": 59.87,
    "rmse": 141.01,
    "r2": 0.9472,
    "mape": 1.67,
    "directional_accuracy": 28.59,
    "directional_accuracy_moving_actuals": 61.72,
    "naive_baseline_mae": 60.16
  },
  "error_quantiles": {
    "p05": -104.31,
    "p10": -98.12,
    "p25": -2.73,
    "p50": 0.02,
    "p75": 47.58,
    "p90": 100.22,
    "p95": 199.32
  }
}
```

### 7.2 Backend Implementation Example for Member 3
To construct empirical 90% prediction intervals downstream in your FastAPI/Flask route handler:

```python
import json
from pathlib import Path

# Load once at application startup
ARTIFACTS_DIR = Path("deliverables/member2-ml/artifacts")
RESIDUAL_MAP = {
    7: json.loads((ARTIFACTS_DIR / "rice_holdout_residuals_7d.json").read_text(encoding="utf-8")),
    14: json.loads((ARTIFACTS_DIR / "rice_holdout_residuals_14d.json").read_text(encoding="utf-8")),
    30: json.loads((ARTIFACTS_DIR / "rice_holdout_residuals_30d.json").read_text(encoding="utf-8")),
    "short": json.loads((ARTIFACTS_DIR / "rice_holdout_residuals.json").read_text(encoding="utf-8")),
}

def add_prediction_intervals(prediction_dict: dict, horizon: int = 14) -> dict:
    quantiles = RESIDUAL_MAP[horizon]["error_quantiles"]
    pred = prediction_dict["predicted_modal_price"]

    # 90% Empirical Prediction Interval: [Predicted + p05, Predicted + p95]
    prediction_dict["predicted_price_lower"] = round(float(pred + quantiles["p05"]), 2)
    prediction_dict["predicted_price_upper"] = round(float(pred + quantiles["p95"]), 2)
    prediction_dict["interval_confidence"] = "90%"
    return prediction_dict
```

### 7.3 Statistical Nature & Calibration Disclaimer

> [!IMPORTANT]
> **Empirical Uncertainty Intervals vs. Theoretical Coverage:**
> The prediction intervals generated by `[predicted_modal_price + p05, predicted_modal_price + p95]` are **empirical historical uncertainty intervals** computed strictly from unrounded residuals over the audited out-of-sample holdout period (`2023-02-01` to `2024-02-01`).
>
> Agricultural mandi price movements exhibit conditional heteroskedasticity (volatility clustering), non-Gaussian heavy tails, and liquidity differences across seasons and mandis. Consequently:
> 1. These bounds reflect the empirical distribution of errors on the audited holdout dataset.
> 2. They are **not guaranteed to achieve an exact 90% coverage rate** on future out-of-sample observations without online conformal calibration or market-specific quantile adjustments.
> 3. Downstream UI displays must clearly indicate that these intervals are statistical uncertainty guidelines rather than contractual or financial price guarantees.

---

## 8. Crop and Group Mapping JSON Files

To prevent unseen-category errors in production, Member 2 has exported the complete universe of categorical strings from historical training data:

- `rice_group_mappings.json` (West Bengal Rice Mandis: 87 markets, 34 varieties, 297 observed combinations across 257,872 training rows)
- `tea_group_mappings.json` (Tea Mandis: 13 markets, 5 varieties, 19 observed combinations across 5,181 training rows)

### 8.1 Schema of Crop Group Mapping Artifacts
```json
{
  "crop": "rice",
  "train_date_max": "2021-01-31",
  "total_training_rows": 257872,
  "columns": {
    "State Name": ["West Bengal"],
    "District Name": ["Bankura", "Birbhum", "Burdwan", ...],
    "Market Name": ["Alipurduar", "Bankura Sadar", "Burdwan", ...],
    "Variety": ["Aman", "Common", "Fine", "Miniket", "Super Fine", ...],
    "Group": ["Cereals"]
  },
  "observed_combinations": [
    ["West Bengal", "Bankura", "Bankura Sadar", "Common", "Cereals"],
    ["West Bengal", "Burdwan", "Burdwan", "Common", "Cereals"],
    ...
  ]
}
```

### 8.2 Guidance for Member 3 Scrapers & Input Validation
1. **Dropdown Population:** Use `columns["Market Name"]` and `columns["Variety"]` directly in your UI dropdowns to guarantee 100% exact character-for-character matching.
2. **Combination Verification:** Use `observed_combinations` to validate user queries. If a requested `(Market, Variety)` pair is not in `observed_combinations`, return an informative HTTP 400 response explaining that no historical market data exists for that pair in West Bengal.
