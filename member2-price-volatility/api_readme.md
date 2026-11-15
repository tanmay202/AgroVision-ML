# Rice Price & Volatility API

A robust, production-ready FastAPI web service providing point-in-time predictions for Rice modal price and volatility in West Bengal. The underlying pipeline uses a heavily validated XGBoost 48-feature price model and a deterministic persistence baseline for volatility classification.

## Setup & Running

Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```

## Endpoints

### `GET /health`
Verifies that the API is running and that the model has successfully loaded.

**Response:**
```json
{
  "status": "ok",
  "model": "rice-price-xgb",
  "volatility": "persistence"
}
```

### `POST /predict`
Returns the expected future modal price and 5-day volatility classification.

**Request Schema:**
- `market`: (string) The market name (e.g. `"Burdwan"`)
- `variety`: (string) The variety name (e.g. `"Common"`)
- `prediction_date`: (string) ISO date (`YYYY-MM-DD`)
- `volatility_version`: (string, optional) `"v1"` (default persistence baseline) or `"v2_normalized"` (opt-in 35-feature XGBoost classifier)
- `history`: (array) A chronological list of `HistoricalObservation` objects covering at least 30 valid prior dates.
  - `reported_date` (string)
  - `modal_price` (float)
  - `arrivals` (float)

**Example Request (Default `v1` Persistence):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "history": [
    {
      "reported_date": "2023-09-01",
      "modal_price": 2000.0,
      "arrivals": 100.0
    },
    ... (at least 30 records total required) ...
    {
      "reported_date": "2023-10-31",
      "modal_price": 2400.0,
      "arrivals": 150.0
    }
  ]
}'
```

**Example Response (`v1` Default):**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "predicted_modal_price": 2445.67,
  "predicted_volatility_range": 40.0,
  "predicted_volatility_class": "MEDIUM",
  "volatility_version": "v1"
}
```

**Example Request (`v2_normalized` Opt-in ML Classifier):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "volatility_version": "v2_normalized",
  "history": [ ... 30+ records ... ]
}'
```

**Example Response (`v2_normalized`):**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "predicted_modal_price": 2445.67,
  "predicted_volatility_range": 40.0,
  "predicted_volatility_class": "MEDIUM",
  "volatility_version": "v2_normalized",
  "normalized_volatility_class": "MEDIUM",
  "normalized_volatility_probabilities": {
    "LOW": 0.1245,
    "MEDIUM": 0.6512,
    "HIGH": 0.2243
  },
  "historical_pct_range_5": 1.67,
  "thresholds": {
    "low_medium_threshold": 0.014286,
    "medium_high_threshold": 1.960784
  }
}
```

## Error Handling
- `400 Bad Request`: When `history` contains fewer than 30 valid past observations, ensuring the 30-day historical window features do not collapse, or when an unsupported `horizon_days` is requested.
- `422 Unprocessable Entity`: When required JSON properties are missing or malformed.
- `503 Service Unavailable`: When model artifacts are unavailable or when horizon forecasting is disabled via configuration switches.

---

### `POST /predict-horizon`
Direct 14-day or 30-day calendar horizon modal price forecasting.

**Request Schema (`PredictHorizonRequest`):**
- `market`: (string) Market name (e.g. `"Burdwan"`)
- `variety`: (string) Variety name (e.g. `"Common"`)
- `prediction_date`: (string) Forecast origin date (`YYYY-MM-DD`)
- `horizon_days`: (integer) `14` or `30`
- `history`: (array) Chronological list of `HistoricalObservation` objects ($\ge 30$ records)

**Example Request (14-Day Horizon):**
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict-horizon' \
  -H 'Content-Type: application/json' \
  -d '{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-02-04",
  "horizon_days": 14,
  "history": [
    {
      "reported_date": "2024-01-01",
      "modal_price": 2500.0,
      "arrivals": 120.0
    },
    ... (at least 30 records total) ...
    {
      "reported_date": "2024-02-04",
      "modal_price": 2670.0,
      "arrivals": 130.0
    }
  ]
}'
```

**Example Response:**
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
  "limitations": "Opt-in release. Evaluated on 18,921 holdout observations...",
  "horizon_metadata": {
    "max_tolerance_days": 3,
    "train_period": ["2002-05-28", "2021-01-31"]
  }
}
```

---

### Observability & Shadow Monitoring Endpoints

#### `GET /health/horizons`
Reports artifact existence, switch statuses, and readiness for 14d and 30d price models.

#### `GET /metrics/horizons`
Reports request counts, success rates, average latencies (ms), and error categories for horizon forecasting.

#### `GET /shadow/status`
Reports operational status of the direct price horizon shadow pipeline, active database path, and counts of pending, evaluated, and unavailable forecasts.

#### `GET /shadow/metrics?horizon_days=14&cohort=PROSPECTIVE`
Returns comparative accuracy benchmarks (MAE, RMSE, $R^2$, MAPE, tolerance bounds) between direct horizon models and the persistence baseline.

#### `POST /shadow/evaluate`
Triggers delayed outcome resolution for closed price horizon forecasting windows.

#### `GET /shadow/volatility/status?cohort=PROSPECTIVE`
Reports operational health, active database path, and forecast status counts (`PENDING`, `EVALUATED`, `WINDOW_EMPTY`) for the isolated Volatility V3 prospective shadow pipeline.

#### `GET /shadow/volatility/metrics?cohort=PROSPECTIVE&margin=0.01`
Computes comparative classification performance (Macro F1, Balanced Accuracy, HIGH recall/precision, ECE, paired cluster non-inferiority test) between Volatility V3 and 14D Calendar Persistence.

#### `POST /shadow/volatility/evaluate`
Triggers delayed outcome resolution for all pending Volatility V3 forecasts where the full 14-calendar-day window has definitively closed.

---

## Operational Deployment Switches
By default, 14-day and 30-day endpoints and shadow monitoring are safely disabled in fresh installations to enforce strict opt-in semantics and protect client systems. To configure:

```bash
# ─── 1. Price Horizon Forecasting Switches ───
export AGROVISION_ENABLE_HORIZON_FORECASTING=true  # Master switch
export AGROVISION_ENABLE_HORIZON_14D=true          # 14-day opt-in model
export AGROVISION_ENABLE_HORIZON_30D=true          # 30-day experimental model

# ─── 2. Shadow Evaluation Switches ───
export AGROVISION_ENABLE_SHADOW_EVALUATION=true    # Price horizon shadow engine
export AGROVISION_ENABLE_VOLATILITY_SHADOW=true    # Volatility V3 shadow engine
export AGROVISION_SHADOW_DB_PATH="data/processed/shadow_evaluation.db"  # Persistent SQLite store
```
