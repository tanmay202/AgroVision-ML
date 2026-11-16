# AgroVision-ML — Member 3 API Integration Contract
## Multi-Commodity Unified Inference & Serving Interface

**Document Version:** 2.1.0  
**Effective Date:** 2026-11-16  
**Audience:** Member 3 (FastAPI / Backend Engineering Team), Model Governance, QA  
**Single Point of Import:** `shared.predict_bridge`  

---

## 1. Architectural Overview & Single Point of Entry

This contract specifies the programmatic and REST API interfaces between the Machine Learning models (Member 1: Yield; Member 2: Price & Volatility) and the Member 3 FastAPI backend service.

### 1.1 Python Import Convention
Backend services must interact with the ML subsystem strictly via the shared prediction bridge:

```python
from shared.predict_bridge import (
    AgroVisionPredictor,
    predict,
    predict_price,
    predict_price_horizon,
    predict_yield,
)
```

The bridge handles dynamic module resolution, lazy model loading, cross-commodity schema normalization, empirical residual lookup, and non-parametric prediction interval calculation.

---

## 2. Universal Price & Volatility Prediction API (`predict`)

### 2.1 Method Signature
```python
def predict(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    commodity: str = "rice",
    horizon: Union[int, str] = "short",
    market: Optional[str] = None,
    variety: Optional[str] = None,
    prediction_date: Optional[Union[str, pd.Timestamp]] = None,
    **kwargs
) -> Dict[str, Any]:
```

### 2.2 Input Arguments Specification

| Parameter | Type | Required | Default | Allowed Values / Format | Description |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `history` | `DataFrame` or `List[dict]` | **Yes** | None | At least 30 chronological days | Historical mandi time-series observations. |
| `commodity` | `str` | No | `"rice"` | `"rice"`, `"wheat"`, `"potato"`, `"tea"` | Target commodity. Case-insensitive. |
| `horizon` | `str` or `int` | No | `"short"` | `"short"` (1–7d), `14`, `30`, `"14d"`, `"30d"` | Forecast horizon window. |
| `market` | `str` | No | Inferred | e.g., `"Burdwan"`, `"Khanna"`, `"Agra"` | Target APMC mandi name. Inferred from `history` if omitted. |
| `variety` | `str` | No | Inferred | e.g., `"Common"`, `"Dara"`, `"Jyoti"` | Crop variety name. Inferred from `history` if omitted. |
| `prediction_date` | `str` | No | Inferred | `"YYYY-MM-DD"` | Forecast origin date $t$. Inferred as `max(date)` if omitted. |

### 2.3 Input DataFrame Schema Requirements

The input `history` must provide a minimum of **30 consecutive observations** (calendar days with forward-filled or raw observations) for feature engineering:

| Column Name | Alternative Aliases Accepted | Type | Mandatory? | Notes |
| :--- | :--- | :---: | :---: | :--- |
| `Arrival_Date` | `date`, `Date`, `timestamp` | `datetime` or `str` | **Yes** | ISO-8601 format (`YYYY-MM-DD`). |
| `Modal_Price` | `modal_price`, `price`, `Price` | `float` | **Yes** | Price in ₹/quintal (1 quintal = 100 kg). Must be > 0. |
| `Market_Name` | `market`, `Market Name`, `mandi` | `str` | No | Recommended for multi-mandi series. |
| `Variety` | `variety`, `Variety Name` | `str` | No | Recommended for variety-specific modeling. |
| `Arrivals_Tonnes` | `arrivals`, `Arrivals`, `arrivals_in_qtl` | `float` | No | Daily physical market arrival volume (tonnes). |

---

## 3. Standardized Output Schema (Canonical Keys & Backward Compatibility)

All prediction calls return a Python dictionary with guaranteed canonical keys, standardized type signatures, and backward-compatible aliases.

### 3.1 Field Dictionary Specification

| Field Name | Type | Key Status | Description |
| :--- | :---: | :---: | :--- |
| `commodity` | `str` | **Canonical** | Standardized commodity name (`"rice"`, `"wheat"`, `"potato"`, `"tea"`). |
| `predicted_price_rs_quintal` | `float` | **Canonical** | Forecasted modal price in ₹/quintal. Rounded to 2 decimal places. |
| `predicted_modal_price` | `float` | **Canonical / Alias** | Synchronous alias for `predicted_price_rs_quintal`. |
| `predicted_price` | `float` | **Alias** | Backward-compatible alias for legacy consumers. |
| `current_modal_price` | `float` | **Canonical** | Most recent observed modal price $P_t$ in ₹/quintal. |
| `predicted_price_change` | `float` | **Canonical** | Delta $\Delta P = \hat{P}_{t+h} - P_t$ in ₹/quintal. |
| `horizon_days` | `int` | **Canonical** | Forecast horizon integer: `1` (short-term), `14`, or `30`. |
| `target_date_convention` | `str` | **Canonical** | Horizon target interval description (e.g., `"First observation in [t+14, t+17] calendar days"`). |
| `volatility` | `str` | **Canonical** | Categorical volatility class: `"LOW"`, `"MEDIUM"`, `"HIGH"`. |
| `predicted_volatility_class`| `str` | **Canonical / Alias** | Explicit alias for `volatility`. |
| `volatility_probabilities` | `dict` | **Canonical** | Calibrated posterior probabilities: `{"LOW": float, "MEDIUM": float, "HIGH": float}`. Sums to 1.0. |
| `volatility_confidence` | `float` | **Canonical** | $\max(P(\text{class})) \in [0.0, 1.0]$. |
| `model_name` | `str` | **Canonical** | Exact model identifier (e.g., `"wheat-price-catboost-v2"`, `"rice-price-xgb-14d"`). |
| `model_version` | `str` | **Canonical** | Semantic version string (e.g., `"2.0.0"`, `"1.0.0"`). |
| `release_status` | `str` | **Canonical** | Operational release tier: `"production"`, `"opt-in"`, `"experimental"`. |
| `is_experimental` | `bool` | **Canonical** | Boolean flag indicating whether the prediction is experimental/advisory (`True`) or production (`False`). |
| `limitations` | `str` | **Canonical** | Agronomic, liquidity, and statistical caveat statement. |
| `residual_distribution` | `str` or `dict` | **Canonical** | Source empirical holdout residual distribution descriptor. |
| `prediction_interval` | `dict` | **Canonical** | Empirical uncertainty bounds: `{"p10": float, "p25": float, "p50": float, "p75": float, "p90": float, "symmetric_80_ci": [float, float]}`. |

---

## 4. Multi-Horizon Release Status & Governance Matrix

AgroVision enforces release tier tagging across commodities and horizons. Member 3 must use these flags to display appropriate UI badges or require opt-in switches:

| Commodity | Horizon | `release_status` | `is_experimental` | Member 3 Serving Recommendation |
| :--- | :---: | :---: | :---: | :--- |
| **Rice** | Short (1–7d) | `"production"` | `False` | Default public display. Production SLA. |
| **Rice** | 14-day | `"opt-in"` | `False` | Validated tier. Display with performance metadata. |
| **Rice** | 30-day | `"experimental"` | `True` | Advisory only. UI must display experimental disclaimer. |
| **Wheat** | Short (1–7d) | `"production"` | `False` | Default public display. Production SLA (CatBoost MAE V2). |
| **Wheat** | 14-day | `"experimental"` | `True` | Advisory only. Holdout showed deficit vs persistence. |
| **Wheat** | 30-day | `"experimental"` | `True` | Advisory only. Holdout showed deficit vs persistence. |
| **Potato** | Short (1–7d) | `"production"` | `False` | Default public display. Production SLA (XGB Pseudo-Huber V2). |
| **Potato** | 14-day | `"experimental"` | `True` | Advisory only. Holdout showed deficit vs persistence. |
| **Potato** | 30-day | `"experimental"` | `True` | Advisory only. Holdout showed deficit vs persistence. |
| **Tea** | Short (1–7d) | `"production"` | `False` | Default public display. Production SLA. |

---

## 5. Sample JSON Payloads

### 5.1 Wheat Short-Term Production Response
```json
{
  "commodity": "wheat",
  "predicted_price_rs_quintal": 2045.50,
  "predicted_modal_price": 2045.50,
  "predicted_price": 2045.50,
  "current_modal_price": 2040.00,
  "predicted_price_change": 5.50,
  "horizon_days": 1,
  "target_date_convention": "Next available trading day (t + 1 to t + 7)",
  "volatility": "LOW",
  "predicted_volatility_class": "LOW",
  "volatility_probabilities": {
    "LOW": 0.8124,
    "MEDIUM": 0.1531,
    "HIGH": 0.0345
  },
  "volatility_confidence": 0.8124,
  "model_name": "wheat-price-catboost-v2",
  "model_version": "2.0.0",
  "release_status": "production",
  "is_experimental": false,
  "limitations": "Production V2 release (CatBoost MAE with deadband eps=3.0). Validated on holdout data.",
  "residual_distribution": "wheat_residual_distributions.json:short_term",
  "prediction_interval": {
    "p10": 2029.70,
    "p25": 2040.30,
    "p50": 2045.50,
    "p75": 2050.60,
    "p90": 2061.70,
    "symmetric_80_ci": [2029.70, 2061.70]
  }
}
```

### 5.2 Potato 14-Day Experimental Horizon Response
```json
{
  "commodity": "potato",
  "predicted_price_rs_quintal": 985.20,
  "predicted_modal_price": 985.20,
  "predicted_price": 985.20,
  "current_modal_price": 950.00,
  "predicted_price_change": 35.20,
  "horizon_days": 14,
  "target_date_convention": "First observation in [t + 14, t + 17] calendar days",
  "volatility": "MEDIUM",
  "predicted_volatility_class": "MEDIUM",
  "volatility_probabilities": {
    "LOW": 0.2215,
    "MEDIUM": 0.6540,
    "HIGH": 0.1245
  },
  "volatility_confidence": 0.6540,
  "model_name": "potato-price-catboost-14d",
  "model_version": "2.0.0",
  "release_status": "experimental",
  "is_experimental": true,
  "limitations": "Experimental V2 advisory forecast. Holdout evaluation showed deficit vs persistence (+Rs.23.33/q MAE deficit). Use for trend advisory purposes only.",
  "residual_distribution": "potato_residual_distributions.json:14d",
  "prediction_interval": {
    "p10": 916.90,
    "p25": 956.10,
    "p50": 988.70,
    "p75": 1021.00,
    "p90": 1062.60,
    "symmetric_80_ci": [916.90, 1062.60]
  }
}
```

---

## 6. Error Handling & Validation Contract

The integration layer throws explicit standard Python exceptions that Member 3 must translate into standard HTTP status codes:

| Exception Class | Trigger Condition | Recommended HTTP Status | Response Schema |
| :--- | :--- | :---: | :--- |
| `ValueError` | Unsupported commodity (e.g. `"mango"`), invalid horizon (e.g. `45`), history `< 30` observations, or NaN in critical columns. | `400 Bad Request` | `{"detail": str, "error_type": "ValidationError"}` |
| `FileNotFoundError` | Model artifact or configuration file missing for the requested commodity. | `500 Internal Server Error` or `503 Service Unavailable` | `{"detail": str, "error_type": "ModelArtifactMissing"}` |
| `RuntimeError` | Feature engineering pipeline failure or administrative kill-switch active. | `500 Internal Server Error` | `{"detail": str, "error_type": "InferenceRuntimeError"}` |

---

## 7. Member 3 FastAPI Implementation Example

```python
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from shared.predict_bridge import AgroVisionPredictor

app = FastAPI(title="AgroVision Multi-Commodity Prediction API", version="2.1.0")
predictor = AgroVisionPredictor()

class MandiObservation(BaseModel):
    Arrival_Date: str = Field(..., example="2026-10-15")
    Modal_Price: float = Field(..., gt=0, example=2050.0)
    Market_Name: Optional[str] = Field("Burdwan", example="Burdwan")
    Variety: Optional[str] = Field("Common", example="Common")
    Arrivals_Tonnes: Optional[float] = Field(150.0, example=150.0)

class PredictionRequest(BaseModel):
    commodity: str = Field("rice", description="Target commodity: rice, wheat, potato, tea")
    horizon: str = Field("short", description="'short' (1-7d), '14', or '30'")
    history: List[MandiObservation]

@app.post("/api/v1/predict/price")
def predict_commodity_price(payload: PredictionRequest) -> Dict[str, Any]:
    if len(payload.history) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"At least 30 observations required; received {len(payload.history)}."
        )
    
    df = pd.DataFrame([obs.dict() for obs in payload.history])
    try:
        result = predictor.predict(
            history=df,
            commodity=payload.commodity,
            horizon=payload.horizon
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model artifacts unavailable: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
```
