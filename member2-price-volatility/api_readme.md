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
- `market`: (string) The market name
- `variety`: (string) The variety name
- `prediction_date`: (string) ISO date (`YYYY-MM-DD`)
- `history`: (array) A chronological list of `HistoricalObservation` objects covering at least 30 valid prior dates.
  - `reported_date` (string)
  - `modal_price` (float)
  - `arrivals` (float)

**Example Request:**
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

**Example Response:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "predicted_modal_price": 2445.67,
  "predicted_volatility_range": 40.0,
  "predicted_volatility_class": "MEDIUM"
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

### Observability Endpoints

#### `GET /health/horizons`
Reports artifact existence, switch statuses, and readiness for 14d and 30d models.

#### `GET /metrics/horizons`
Reports request counts, success rates, average latencies (ms), and error categories for horizon forecasting.

---

## Operational Deployment Switches
By default, 14-day and 30-day endpoints are disabled in fresh installations to enforce safe opt-in semantics. To enable them:
```bash
# Master switch
export AGROVISION_ENABLE_HORIZON_FORECASTING=true

# Horizon switches
export AGROVISION_ENABLE_HORIZON_14D=true
export AGROVISION_ENABLE_HORIZON_30D=true  # Experimental
```
