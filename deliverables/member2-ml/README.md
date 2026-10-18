# AgroVision-ML — Member 2 Standalone Machine Learning Deliverable Package

**Package Name:** `member2-ml`  
**Author:** Member 2 (Price Prediction and Volatility ML)  
**Target Consumer:** Member 3 (Backend & Application Integration)  
**Status:** Certified & Ready for Production Integration  

---

## 1. Package Overview

This deliverable package provides the standalone, pure-Python machine learning module for West Bengal rice modal price and volatility forecasting.

It requires **no web frameworks, no FastAPI, and no uvicorn**. Member 3 can import and run this module directly within existing application code, background workers, or API route handlers.

### Key Capabilities
- **Short-Term Price Forecasting (1–7 Days):** Verified production 48-feature XGBoost regressor.
- **14-Day Calendar Horizon Forecasting:** Opt-in 48-feature XGBoost regressor forecasting the first eligible observation in $[t + 14, t + 17]$ calendar days.
- **30-Day Calendar Horizon Forecasting:** Experimental 48-feature XGBoost regressor forecasting the first eligible observation in $[t + 30, t + 33]$ calendar days.
- **Price-Normalized Volatility Classification (`v2_normalized`):** 35-feature XGBoost classifier predicting 5-day volatility categories (`LOW`, `MEDIUM`, `HIGH`).
- **Flexible Data Ingestion:** Accepts either pandas DataFrames or lists of Python dictionaries, with automatic column alias mapping.

---

## 2. Directory Structure

```text
deliverables/member2-ml/
├── README.md                           # This guide for Member 3
├── integration_contract.md             # Formal integration specifications & schemas
├── requirements.txt                    # Core ML dependencies (pure Python/ML)
├── artifacts/                          # Bit-for-bit serialized models and configurations
│   ├── rice_price_model.pkl            # Frozen short-term production model
│   ├── rice_price_final_config.json    # Frozen short-term configuration (48 features)
│   ├── rice_price_model_14d.pkl        # 14-day calendar horizon model
│   ├── rice_price_model_14d_config.json
│   ├── rice_price_model_30d.pkl        # 30-day calendar horizon model (experimental)
│   ├── rice_price_model_30d_config.json
│   ├── rice_volatility_normalized_model.pkl
│   ├── rice_volatility_normalized_config.json
│   ├── rice_volatility_model.pkl
│   ├── rice_volatility_config.json
│   ├── rice_holdout_residuals_7d.json  # Holdout residuals for 7-day prediction intervals
│   ├── rice_holdout_residuals_14d.json # Holdout residuals for 14-day prediction intervals
│   ├── rice_holdout_residuals_30d.json # Holdout residuals for 30-day prediction intervals
│   ├── rice_holdout_residuals.json     # Holdout residuals for short-term prediction intervals
│   ├── rice_group_mappings.json        # Exact market/variety/district strings for Rice (87 mandis)
│   ├── tea_group_mappings.json         # Exact market/variety/district strings for Tea (13 mandis)
│   └── model_handoff_manifest.json     # Comprehensive machine-readable handoff manifest
├── src/
│   ├── __init__.py                     # Package export declarations
│   ├── predictor.py                    # Standalone RicePricePredictor class
│   ├── feature_engineering.py          # 48 price features + 35 normalized vol features
│   └── config.py                       # Constants, column mappings, and metadata
├── reports/
│   ├── model_evaluation_summary.md     # Holdout benchmarks & evaluation report
│   └── direct_horizon_evaluation.md    # Direct T+7, T+14, T+30 evaluation & residual audit
├── examples/
│   └── quickstart.py                   # Self-contained runnable usage script
└── tests/
    ├── __init__.py
    └── test_predictor.py               # Standalone test suite (11/11 passing)
```

---

## 3. Quick Start (Under 10 Lines of Code)

```python
from src.predictor import RicePricePredictor

# 1. Instantiate predictor (automatically finds artifacts/ directory)
predictor = RicePricePredictor()

# 2. Supply historical observations (at least 30 records)
history = [
    {"date": "2024-01-01", "modal_price": 2500.0, "arrivals": 120.0, "market": "Burdwan", "variety": "Common"},
    # ... >= 30 daily observations ...
    {"date": "2024-02-04", "modal_price": 2670.0, "arrivals": 130.0, "market": "Burdwan", "variety": "Common"},
]

# 3. Generate forecasts for any horizon:
forecast_1d = predictor.predict(history, horizon="short")  # 1-7 day production
forecast_14d = predictor.predict(history, horizon=14)      # 14-day opt-in
forecast_30d = predictor.predict(history, horizon=30)      # 30-day experimental

print(f"Current Price: Rs. {forecast_14d['current_modal_price']}")
print(f"Predicted 14d Price: Rs. {forecast_14d['predicted_modal_price']}")
print(f"Target Date: {forecast_14d['requested_target_date']}")
```

---

## 4. API Reference: `RicePricePredictor`

### Initialization
```python
predictor = RicePricePredictor(artifacts_dir=None)
```
- `artifacts_dir` *(optional)*: Path to directory containing `.pkl` and `.json` artifacts. If omitted, automatically discovers `deliverables/member2-ml/artifacts` or checks `AGROVISION_ARTIFACTS_DIR`.

### Methods

#### `predict(history, horizon="short", market=None, variety=None, prediction_date=None, **kwargs) -> dict`
Unified dispatch method supporting:
- `horizon="short"` (or `1`, `"1-7"`): Short-term 1–7 day production forecast.
- `horizon=14`: 14-day calendar horizon forecast.
- `horizon=30`: 30-day calendar horizon forecast.

#### `predict_short_term(history, market=None, variety=None, prediction_date=None, include_volatility=True, volatility_version="v2_normalized") -> dict`
Generates short-term forecast and volatility classification.

#### `predict_horizon(history, horizon_days, market=None, variety=None, prediction_date=None) -> dict`
Generates 14-day or 30-day horizon forecast with target dates and confidence limitations.

#### `get_required_features() -> List[str]`
Returns the exact list of 48 feature names expected by the price models.

#### `get_model_info() -> dict`
Returns status, release tiers, and metadata for all available models.

---

## 5. Running the Tests

To verify package installation and numerical parity:

```bash
pytest deliverables/member2-ml/tests/ -v
```

All 10 tests should pass, covering initialization, short-term prediction, 14d/30d horizons, schema tolerance, error handling, leakage prevention, and mathematical parity.

---

## 6. Integration Contract & Evaluation

- See [`integration_contract.md`](file:///d:/MCA/SEM3/AgroVision-ML/deliverables/member2-ml/integration_contract.md) for full request/response schemas, exception mapping, and UI display guidelines.
- See [`reports/model_evaluation_summary.md`](file:///d:/MCA/SEM3/AgroVision-ML/deliverables/member2-ml/reports/model_evaluation_summary.md) for holdout benchmark tables and evaluation metrics.
