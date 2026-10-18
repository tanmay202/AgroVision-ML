import os
import time
from threading import Lock
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd
import sys
from pathlib import Path
import logging

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline, InsufficientHistoryError
from config import (
    is_horizon_forecasting_enabled,
    is_horizon_14d_enabled,
    is_horizon_30d_enabled,
    is_shadow_evaluation_enabled,
)
from shadow_evaluation import (
    get_shadow_status,
    compute_shadow_evaluation_metrics,
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the pipeline instance
pipeline_instance = None


class HorizonTelemetryTracker:
    """Thread-safe / process-local lightweight telemetry tracker for horizon forecasting."""
    def __init__(self):
        self._lock = Lock()
        self.metrics = {
            14: {"requests": 0, "success": 0, "errors": 0, "total_latency_ms": 0.0},
            30: {"requests": 0, "success": 0, "errors": 0, "total_latency_ms": 0.0},
        }
        self.error_counts = {}

    def record(self, horizon: int, success: bool, latency_ms: float, error_category: Optional[str] = None):
        with self._lock:
            if horizon in self.metrics:
                m = self.metrics[horizon]
                m["requests"] += 1
                if success:
                    m["success"] += 1
                else:
                    m["errors"] += 1
                m["total_latency_ms"] += latency_ms

            if error_category:
                self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1

    def get_summary(self):
        with self._lock:
            out = {}
            for h in (14, 30):
                m = self.metrics[h]
                reqs = m["requests"]
                avg_lat = round(m["total_latency_ms"] / reqs, 2) if reqs > 0 else 0.0
                err_rate = round((m["errors"] / reqs) * 100.0, 2) if reqs > 0 else 0.0
                out[f"{h}d"] = {
                    "requests": reqs,
                    "success": m["success"],
                    "errors": m["errors"],
                    "error_rate_pct": err_rate,
                    "avg_latency_ms": avg_lat,
                }
            out["error_categories"] = dict(self.error_counts)
            return out


horizon_telemetry = HorizonTelemetryTracker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model once
    global pipeline_instance
    logger.info("Loading RiceInferencePipeline...")
    try:
        pipeline_instance = RiceInferencePipeline()
        logger.info("RiceInferencePipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError("Failed to load ML model during startup.")

    yield
    # Shutdown logic
    logger.info("Shutting down API and releasing resources.")
    pipeline_instance = None


app = FastAPI(title="Rice Price and Volatility API", lifespan=lifespan)

# --- Pydantic Schemas ---

class HistoricalObservation(BaseModel):
    reported_date: str = Field(..., description="Date in YYYY-MM-DD format")
    modal_price: float = Field(..., description="Modal price in Rs./Quintal")
    arrivals: float = Field(..., description="Arrivals in Tonnes")

class PredictRequest(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Date in YYYY-MM-DD format")
    history: List[HistoricalObservation]
    volatility_version: Optional[str] = Field("v1", description="Volatility version: 'v1' (persistence baseline) or 'v2_normalized' (normalized ML model)")

class PredictResponse(BaseModel):
    market: str
    variety: str
    prediction_date: str
    predicted_modal_price: float
    predicted_volatility_range: float
    predicted_volatility_class: str
    volatility_version: Optional[str] = "v1"
    normalized_volatility_class: Optional[str] = None
    normalized_volatility_probabilities: Optional[dict] = None
    historical_pct_range_5: Optional[float] = None
    thresholds: Optional[dict] = None

class PredictHorizonRequest(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Forecast origin date in YYYY-MM-DD format")
    horizon_days: int = Field(..., description="Target calendar horizon in days (14 or 30)")
    history: List[HistoricalObservation]

class PredictHorizonResponse(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Forecast origin date in YYYY-MM-DD format")
    forecast_origin_date: Optional[str] = Field(None, description="Explicit synonym for forecast origin date")
    horizon_days: int = Field(..., description="Target calendar horizon in days (14 or 30)")
    requested_target_date: Optional[str] = Field(None, description="Target forecast date: origin date + horizon days")
    current_modal_price: float
    predicted_modal_price: float
    predicted_price_change: float
    target_date_convention: str
    model_name: str
    model_version: Optional[str] = "1.0.0"
    release_status: Optional[str] = None
    limitations: Optional[str] = None
    horizon_metadata: Optional[dict] = None

# --- Endpoints ---

@app.get("/health")
async def health_check():
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "ok",
        "model": "rice-price-xgb",
        "volatility": "persistence"
    }

@app.get("/health/horizons")
async def horizon_health_check():
    """Detailed health check for horizon forecasting artifacts and configuration switches."""
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Base pipeline not loaded.")

    artifacts_dir = pipeline_instance.artifacts_dir
    m14_exists = (artifacts_dir / "rice_price_model_14d.pkl").exists()
    c14_exists = (artifacts_dir / "rice_price_model_14d_config.json").exists()
    m30_exists = (artifacts_dir / "rice_price_model_30d.pkl").exists()
    c30_exists = (artifacts_dir / "rice_price_model_30d_config.json").exists()

    return {
        "status": "ok",
        "master_switch_enabled": is_horizon_forecasting_enabled(),
        "horizon_14d": {
            "enabled": is_horizon_14d_enabled(),
            "model_artifact_present": m14_exists,
            "config_artifact_present": c14_exists,
            "ready": is_horizon_14d_enabled() and m14_exists and c14_exists,
            "release_status": "opt-in",
        },
        "horizon_30d": {
            "enabled": is_horizon_30d_enabled(),
            "model_artifact_present": m30_exists,
            "config_artifact_present": c30_exists,
            "ready": is_horizon_30d_enabled() and m30_exists and c30_exists,
            "release_status": "experimental",
        },
    }

@app.get("/metrics/horizons")
async def horizon_metrics():
    """Observability metrics for horizon forecasting (requests, latencies, error categories)."""
    return horizon_telemetry.get_summary()

@app.get("/shadow/status")
async def shadow_status():
    """Operational status and counts for internal shadow evaluation."""
    return get_shadow_status()

@app.get("/shadow/metrics")
async def shadow_metrics(horizon_days: Optional[int] = None, cohort: Optional[str] = None):
    """Aggregate live performance benchmarks for shadow evaluation against observed outcomes."""
    if horizon_days is not None and horizon_days not in (14, 30):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported horizon_days: {horizon_days}. Expected 14 or 30."
        )
    if cohort is not None and cohort.upper() not in ("RETROSPECTIVE", "PROSPECTIVE", "ALL"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported cohort: {cohort}. Expected RETROSPECTIVE, PROSPECTIVE, or ALL."
        )
    cohort_filter = None if (cohort is None or cohort.upper() == "ALL") else cohort.upper()
    return compute_shadow_evaluation_metrics(horizon_days=horizon_days, cohort=cohort_filter)

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if pipeline_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
        
    if not request.history:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty history provided.")
        
    # Convert Pydantic request to pandas DataFrame expected by inference.py
    data = []
    for obs in request.history:
        data.append({
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": obs.reported_date,
            "Modal Price (Rs./Quintal)": obs.modal_price,
            "Arrivals (Tonnes)": obs.arrivals
        })
        
    df = pd.DataFrame(data)
    
    try:
        result = pipeline_instance.predict(
            df=df,
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date,
            volatility_version=request.volatility_version,
        )
        return PredictResponse(**result)
        
    except InsufficientHistoryError as e:
        logger.warning(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except ValueError as e:
        logger.warning(f"Value error during prediction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during prediction.")

@app.post("/predict-horizon", response_model=PredictHorizonResponse)
async def predict_horizon(request: PredictHorizonRequest):
    t_start = time.perf_counter()
    h = request.horizon_days

    if pipeline_instance is None:
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="model_not_loaded")
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if h not in (14, 30):
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="invalid_horizon")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported horizon_days: {h}. Supported horizons are 14 and 30 days."
        )

    # Operational release switches
    if not is_horizon_forecasting_enabled():
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="feature_disabled_globally")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Horizon forecasting is currently disabled globally by administrator configuration."
        )

    if h == 14 and not is_horizon_14d_enabled():
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="feature_disabled_14d")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="14-day horizon forecasting is currently disabled by administrator configuration."
        )

    if h == 30 and not is_horizon_30d_enabled():
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="feature_disabled_30d")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="30-day horizon forecasting is currently disabled by administrator configuration."
        )

    if not request.history:
        horizon_telemetry.record(h, success=False, latency_ms=0.0, error_category="empty_history")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty history provided.")

    data = []
    for obs in request.history:
        data.append({
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": obs.reported_date,
            "Modal Price (Rs./Quintal)": obs.modal_price,
            "Arrivals (Tonnes)": obs.arrivals
        })

    df = pd.DataFrame(data)

    try:
        result = pipeline_instance.predict_horizon(
            df=df,
            horizon_days=h,
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=True, latency_ms=latency_ms)
        logger.info(
            "HORIZON_PREDICTION_SUCCESS: horizon=%dd market='%s' variety='%s' origin='%s' latency_ms=%.2f",
            h, request.market, request.variety, request.prediction_date, latency_ms
        )
        return PredictHorizonResponse(**result)

    except InsufficientHistoryError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=False, latency_ms=latency_ms, error_category="insufficient_history")
        logger.warning("HORIZON_PREDICTION_ERROR: horizon=%dd category='insufficient_history' detail='%s'", h, str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FileNotFoundError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=False, latency_ms=latency_ms, error_category="missing_artifact")
        logger.error("HORIZON_PREDICTION_ERROR: horizon=%dd category='missing_artifact' detail='%s'", h, str(e))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Horizon model artifact unavailable: {str(e)}")

    except RuntimeError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=False, latency_ms=latency_ms, error_category="runtime_disabled")
        logger.warning("HORIZON_PREDICTION_ERROR: horizon=%dd category='runtime_disabled' detail='%s'", h, str(e))
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    except ValueError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=False, latency_ms=latency_ms, error_category="validation_error")
        logger.warning("HORIZON_PREDICTION_ERROR: horizon=%dd category='validation_error' detail='%s'", h, str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        horizon_telemetry.record(h, success=False, latency_ms=latency_ms, error_category="internal_error")
        logger.error("HORIZON_PREDICTION_ERROR: horizon=%dd category='internal_error' detail='%s'", h, str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during horizon prediction.")
