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
- `400 Bad Request`: When `history` contains fewer than 30 valid past observations, ensuring the 30-day historical window features do not collapse.
- `422 Unprocessable Entity`: When required JSON properties are missing or malformed.
