"""
AgroVision-ML — Universal Multi-Commodity Staging API (v2)
=========================================================
Unified FastAPI server supporting:
1. All 4 commodities: Rice, Wheat, Potato, Tea.
2. Short-term, 14-day, and 30-day forecasting horizons.
3. Observability telemetry (P50/P95 latencies, error tracking, drift monitoring).
4. Automated and on-demand mandi feed ingestion.
5. Prospective shadow evaluation and delayed outcome matching.
6. 100% backward compatibility with legacy Rice endpoints.
"""

import sys
import os
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = SRC_DIR.parent
PROJECT_ROOT = MEMBER2_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEMBER2_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from shared.predict_bridge import AgroVisionPredictor, predict as bridge_predict
from observability import get_observability_engine
from ingestion.multi_commodity_ingestion import (
    persist_commodity_observations,
    normalize_and_validate_commodity_records,
    audit_commodity_freshness_and_drift,
    get_ingestion_summary,
)
from multi_commodity_shadow import (
    get_commodity_shadow_status,
    compute_commodity_shadow_metrics,
    evaluate_pending_commodity_shadow_forecasts,
    is_commodity_shadow_enabled,
)
from inference import RiceInferencePipeline, InsufficientHistoryError
from config import (
    is_horizon_forecasting_enabled,
    is_horizon_14d_enabled,
    is_horizon_30d_enabled,
    is_shadow_evaluation_enabled,
    is_volatility_shadow_enabled,
)
from shadow_evaluation import (
    get_shadow_status as get_rice_shadow_status,
    compute_shadow_evaluation_metrics as compute_rice_shadow_metrics,
)
from volatility_shadow import (
    compute_volatility_shadow_metrics,
    VolatilityShadowEvaluator,
    get_volatility_db_connection,
)

logger = logging.getLogger("agrovision.staging_api")
logging.basicConfig(level=logging.INFO)

# Global unified predictor and observability engine
_predictor: Optional[AgroVisionPredictor] = None
_rice_pipeline: Optional[RiceInferencePipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor, _rice_pipeline
    logger.info("Starting AgroVision Staging Unified API...")
    try:
        _predictor = AgroVisionPredictor()
        logger.info("Unified AgroVisionPredictor initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize AgroVisionPredictor: {e}")
        raise RuntimeError(f"Startup failure: {e}")

    try:
        _rice_pipeline = RiceInferencePipeline()
        logger.info("Rice legacy inference pipeline initialized.")
    except Exception as e:
        logger.warning(f"RiceInferencePipeline legacy initialization warning: {e}")

    # Initialize observability engine
    obs = get_observability_engine()
    logger.info("ObservabilityEngine active.")

    yield

    logger.info("Shutting down AgroVision Staging API...")
    _predictor = None
    _rice_pipeline = None


app = FastAPI(
    title="AgroVision-ML Universal Staging API",
    version="2.0.0-staging",
    description="Multi-commodity agricultural price, yield, and volatility forecasting platform.",
    lifespan=lifespan,
)


# ============================================================
# PYDANTIC DATA CONTRACTS
# ============================================================

class HistoricalObservation(BaseModel):
    reported_date: str = Field(..., description="Date in YYYY-MM-DD format")
    modal_price: float = Field(..., description="Modal price in Rs./Quintal")
    arrivals: float = Field(0.0, description="Arrivals in Tonnes")


class MultiCommodityPredictRequest(BaseModel):
    commodity: str = Field("rice", description="Commodity: 'rice', 'wheat', 'potato', 'tea'")
    market: str = Field(..., description="APMC wholesale market name")
    variety: str = Field(..., description="Commodity variety")
    prediction_date: str = Field(..., description="Forecast origin date (YYYY-MM-DD)")
    horizon: Optional[Union[int, str]] = Field("short", description="Forecast horizon: 'short', 14, 30, '14d', '30d'")
    history: List[HistoricalObservation] = Field(..., description="Historical time-series observations")
    return_intervals: Optional[bool] = Field(True, description="Whether to include empirical residual prediction intervals")


class MultiCommodityPredictResponse(BaseModel):
    commodity: str
    market: str
    variety: str
    prediction_date: str
    current_modal_price: float
    predicted_modal_price: float
    predicted_price_rs_quintal: float
    predicted_price_change: float
    volatility: str
    predicted_volatility_class: str
    volatility_probabilities: Optional[Dict[str, float]] = None
    volatility_confidence: Optional[float] = None
    horizon_days: int
    target_date_convention: str
    model_name: str
    model_version: str
    release_status: str
    is_experimental: bool
    limitations: Optional[str] = None
    residual_distribution: Optional[Dict[str, Any]] = None
    prediction_interval: Optional[Dict[str, Any]] = None


# Legacy Rice Schemas
class LegacyPredictRequest(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Date in YYYY-MM-DD format")
    history: List[HistoricalObservation]
    volatility_version: Optional[str] = "v1"


class LegacyPredictResponse(BaseModel):
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


class LegacyPredictHorizonRequest(BaseModel):
    market: str
    variety: str
    prediction_date: str = Field(..., description="Forecast origin date in YYYY-MM-DD format")
    horizon_days: int = Field(..., description="Target calendar horizon in days (14 or 30)")
    history: List[HistoricalObservation]


class LegacyPredictHorizonResponse(BaseModel):
    market: str
    variety: str
    prediction_date: str
    forecast_origin_date: Optional[str] = None
    horizon_days: int
    requested_target_date: Optional[str] = None
    current_modal_price: float
    predicted_modal_price: float
    predicted_price_change: float
    target_date_convention: str
    model_name: str
    model_version: Optional[str] = "1.0.0"
    release_status: Optional[str] = None
    limitations: Optional[str] = None
    horizon_metadata: Optional[dict] = None


class IngestionBatchRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="Raw or normalized records to ingest")


# ============================================================
# UNIVERSAL V2 ENDPOINTS
# ============================================================

@app.get("/v2/health")
async def v2_health_all():
    """System-wide health check across all supported agricultural commodities."""
    obs = get_observability_engine()
    return obs.get_all_health()


@app.get("/v2/health/{commodity}")
async def v2_health_commodity(commodity: str):
    """Independent commodity health status, model metadata, and error rate."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Commodity '{comm}' is not supported. Supported: rice, wheat, potato, tea."
        )
    obs = get_observability_engine()
    return obs.get_commodity_health(comm)


@app.get("/v2/metrics")
async def v2_metrics_all():
    """Aggregate telemetry, P50/P95 latencies, and ingestion counters for all commodities."""
    obs = get_observability_engine()
    return obs.get_all_metrics()


@app.get("/v2/metrics/{commodity}")
async def v2_metrics_commodity(commodity: str):
    """Commodity-specific telemetry, latencies, and drift alerts."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Commodity '{comm}' is not supported."
        )
    obs = get_observability_engine()
    return obs.get_commodity_metrics(comm)


@app.post("/v2/predict", response_model=MultiCommodityPredictResponse)
async def v2_predict(request: MultiCommodityPredictRequest):
    """
    Universal prediction endpoint for Rice, Wheat, Potato, and Tea.
    Supports short, 14-day, and 30-day horizons according to commodity release governance.
    """
    global _predictor
    t_start = time.perf_counter()
    comm = request.commodity.lower().strip()
    obs_engine = get_observability_engine()

    if _predictor is None:
        obs_engine.record_prediction(comm, latency_ms=0.0, success=False, error_msg="Predictor not initialized")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model predictor not initialized.")

    if comm not in ("rice", "wheat", "potato", "tea"):
        obs_engine.record_prediction(comm, latency_ms=0.0, success=False, error_msg=f"Unsupported commodity: {comm}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported commodity: '{comm}'. Supported commodities are: rice, wheat, potato, tea."
        )

    # Validate history length
    if not request.history or len(request.history) < 30:
        obs_engine.record_prediction(comm, latency_ms=0.0, success=False, error_msg="Insufficient history (<30 rows)")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient history: minimum 30 observations required. Provided: {len(request.history) if request.history else 0}."
        )

    # Convert request history to DataFrame
    rows = []
    for h in request.history:
        rows.append({
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": h.reported_date,
            "Modal Price (Rs./Quintal)": h.modal_price,
            "Arrivals (Tonnes)": h.arrivals,
        })
    df_history = pd.DataFrame(rows)

    try:
        raw_res = _predictor.predict(
            history=df_history,
            commodity=comm,
            horizon=request.horizon,
            return_intervals=request.return_intervals,
            prediction_date=request.prediction_date,
        )
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        obs_engine.record_prediction(comm, latency_ms=latency_ms, success=True)

        return MultiCommodityPredictResponse(
            commodity=raw_res.get("commodity", comm),
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date,
            current_modal_price=float(raw_res["current_modal_price"]),
            predicted_modal_price=float(raw_res["predicted_modal_price"]),
            predicted_price_rs_quintal=float(raw_res.get("predicted_price_rs_quintal", raw_res["predicted_modal_price"])),
            predicted_price_change=float(raw_res.get("predicted_price_change", 0.0)),
            volatility=str(raw_res["volatility"]),
            predicted_volatility_class=str(raw_res.get("predicted_volatility_class", raw_res["volatility"])),
            volatility_probabilities=raw_res.get("volatility_probabilities"),
            volatility_confidence=raw_res.get("volatility_confidence"),
            horizon_days=int(raw_res["horizon_days"]),
            target_date_convention=str(raw_res["target_date_convention"]),
            model_name=str(raw_res["model_name"]),
            model_version=str(raw_res["model_version"]),
            release_status=str(raw_res["release_status"]),
            is_experimental=bool(raw_res.get("is_experimental", False)),
            limitations=raw_res.get("limitations"),
            residual_distribution=raw_res.get("residual_distribution"),
            prediction_interval=raw_res.get("prediction_interval"),
        )

    except InsufficientHistoryError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        obs_engine.record_prediction(comm, latency_ms=latency_ms, success=False, error_msg=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FileNotFoundError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        obs_engine.record_prediction(comm, latency_ms=latency_ms, success=False, error_msg="Missing artifact")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model artifact unavailable: {str(e)}"
        )

    except ValueError as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        obs_engine.record_prediction(comm, latency_ms=latency_ms, success=False, error_msg=str(e))
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except Exception as e:
        latency_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        obs_engine.record_prediction(comm, latency_ms=latency_ms, success=False, error_msg=str(e))
        logger.error(f"Prediction error for {comm}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error."
        )


@app.post("/v2/ingest/{commodity}")
async def v2_ingest_commodity(commodity: str, request: Optional[IngestionBatchRequest] = None):
    """
    Ingests daily mandi observations for the given commodity into isolated storage.
    Performs canonical alias mapping, unit checks, deduplication, and freshness auditing.
    """
    comm = commodity.lower().strip()
    obs_engine = get_observability_engine()

    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported commodity: '{comm}'."
        )

    if request is not None and request.records:
        df_raw = pd.DataFrame(request.records)
    else:
        # Default empty batch if no payload
        df_raw = pd.DataFrame()

    if df_raw.empty:
        return {
            "commodity": comm,
            "status": "noop",
            "message": "No records supplied for ingestion.",
            "inserted_count": 0,
        }

    clean_df, audit_info = normalize_and_validate_commodity_records(df_raw, commodity=comm)
    persist_res = persist_commodity_observations(clean_df, commodity=comm, audit_report=audit_info)
    inserted = persist_res.get("rows_inserted", 0)
    audit = audit_commodity_freshness_and_drift(clean_df, commodity=comm)

    obs_engine.record_ingestion(
        commodity=comm,
        success=True,
        rows=inserted,
        delayed=1 if audit.get("is_stale") else 0,
    )
    obs_engine.record_drift(
        commodity=comm,
        drift_detected=audit.get("distribution_drift_detected", False),
        details=audit,
    )

    return {
        "commodity": comm,
        "status": "success",
        "inserted_count": inserted,
        "audit_info": audit_info,
        "freshness_audit": audit,
    }


@app.get("/v2/ingestion/status/{commodity}")
async def v2_ingestion_status(commodity: str):
    """Operational status, row counts, and date bounds of ingested observations."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unsupported commodity: '{comm}'.")
    return get_ingestion_summary(comm)


@app.get("/v2/shadow/{commodity}/status")
async def v2_shadow_status(commodity: str):
    """Operational status and counts for commodity shadow forecasts."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unsupported commodity: '{comm}'.")
    return get_commodity_shadow_status(comm)


@app.get("/v2/shadow/{commodity}/metrics")
async def v2_shadow_metrics(commodity: str, horizon_days: Optional[int] = None, cohort: Optional[str] = "PROSPECTIVE"):
    """Evaluated performance metrics vs. persistence baseline for commodity shadow forecasts."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unsupported commodity: '{comm}'.")
    return compute_commodity_shadow_metrics(comm, horizon_days=horizon_days, cohort=cohort or "PROSPECTIVE")


@app.post("/v2/shadow/{commodity}/evaluate")
async def v2_shadow_evaluate(commodity: str, as_of_date: Optional[str] = None):
    """Triggers delayed outcome matching against subsequently realized APMC observations."""
    comm = commodity.lower().strip()
    if comm not in ("rice", "wheat", "potato", "tea"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unsupported commodity: '{comm}'.")
    return evaluate_pending_commodity_shadow_forecasts(comm, as_of_date=as_of_date)


# ============================================================
# LEGACY BACKWARD COMPATIBILITY ENDPOINTS (RICE)
# ============================================================

@app.get("/health")
async def legacy_health():
    """Rice legacy health check."""
    global _rice_pipeline
    if _rice_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "status": "ok",
        "model": "rice-price-xgb",
        "volatility": "persistence",
    }


@app.get("/health/horizons")
async def legacy_health_horizons():
    """Rice legacy horizon health check."""
    global _rice_pipeline
    if _rice_pipeline is None:
        raise HTTPException(status_code=503, detail="Base pipeline not loaded.")
    artifacts_dir = _rice_pipeline.artifacts_dir
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
async def legacy_metrics_horizons():
    """Legacy horizon telemetry metrics."""
    obs = get_observability_engine()
    rice_metrics = obs.get_commodity_metrics("rice")
    return {
        "14d": {"requests": rice_metrics["predictions"]["total_requests"], "errors": rice_metrics["predictions"]["total_errors"]},
        "30d": {"requests": 0, "errors": 0},
    }


@app.get("/shadow/status")
async def legacy_shadow_status():
    """Legacy Rice shadow status."""
    return get_rice_shadow_status()


@app.get("/shadow/metrics")
async def legacy_shadow_metrics(horizon_days: Optional[int] = None, cohort: Optional[str] = None):
    """Legacy Rice shadow metrics."""
    return compute_rice_shadow_metrics(horizon_days=horizon_days, cohort=cohort)


@app.get("/shadow/volatility/status")
async def legacy_shadow_volatility_status(cohort: Optional[str] = "PROSPECTIVE"):
    """Legacy Rice volatility shadow status."""
    c_upper = cohort.upper() if cohort else "ALL"
    cohort_filter = "" if c_upper == "ALL" else "WHERE cohort = ?"
    params = () if c_upper == "ALL" else (c_upper,)
    with get_volatility_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) as total FROM shadow_volatility_forecasts {cohort_filter};", params)
        total = cur.fetchone()["total"]
        cur.execute(f"SELECT COUNT(*) as pending FROM shadow_volatility_forecasts WHERE status = 'PENDING' {'AND cohort = ?' if params else ''};", params)
        pending = cur.fetchone()["pending"]
        cur.execute(f"SELECT COUNT(*) as evaluated FROM shadow_volatility_forecasts WHERE status = 'EVALUATED' {'AND cohort = ?' if params else ''};", params)
        evaluated = cur.fetchone()["evaluated"]
        cur.execute(f"SELECT COUNT(*) as empty FROM shadow_volatility_forecasts WHERE status = 'WINDOW_EMPTY' {'AND cohort = ?' if params else ''};", params)
        empty = cur.fetchone()["empty"]
        cur.execute(f"SELECT COUNT(DISTINCT market) as mandis FROM shadow_volatility_forecasts {cohort_filter};", params)
        mandis = cur.fetchone()["mandis"]

    return {
        "status": "ok",
        "shadow_volatility_enabled": is_volatility_shadow_enabled(),
        "model_version": "rice_volatility_v3_14d_v1",
        "production_default": "Calendar Persistence (14D)",
        "cohort": c_upper,
        "counts": {
            "total_forecasts": total,
            "pending_forecasts": pending,
            "evaluated_forecasts": evaluated,
            "window_empty_forecasts": empty,
            "participating_mandis": mandis,
        },
    }


@app.get("/shadow/volatility/metrics")
async def legacy_shadow_volatility_metrics(cohort: Optional[str] = "PROSPECTIVE", margin: Optional[float] = 0.01):
    cohort_filter = None if (cohort is None or cohort.upper() == "ALL") else cohort.upper()
    return compute_volatility_shadow_metrics(cohort=cohort_filter, non_inferiority_margin=margin or 0.01)


@app.post("/shadow/volatility/evaluate")
async def legacy_shadow_volatility_evaluate(as_of_date: Optional[str] = None):
    evaluator = VolatilityShadowEvaluator()
    res = evaluator.evaluate_pending_forecasts(as_of_date=as_of_date)
    return {"status": "success", "result": res}


@app.post("/predict", response_model=LegacyPredictResponse)
async def legacy_predict(request: LegacyPredictRequest):
    """Legacy Rice prediction endpoint."""
    global _rice_pipeline
    if _rice_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not request.history:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty history provided.")

    data = [
        {
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": obs.reported_date,
            "Modal Price (Rs./Quintal)": obs.modal_price,
            "Arrivals (Tonnes)": obs.arrivals,
        }
        for obs in request.history
    ]
    df = pd.DataFrame(data)

    try:
        result = _rice_pipeline.predict(
            df=df,
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date,
            volatility_version=request.volatility_version,
        )
        return LegacyPredictResponse(**result)
    except InsufficientHistoryError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in legacy predict: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")


@app.post("/predict-horizon", response_model=LegacyPredictHorizonResponse)
async def legacy_predict_horizon(request: LegacyPredictHorizonRequest):
    """Legacy Rice horizon prediction endpoint."""
    global _rice_pipeline
    if _rice_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    h = request.horizon_days
    if h not in (14, 30):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Supported horizons are 14 and 30 days.")

    if not is_horizon_forecasting_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Horizon forecasting disabled globally.")

    if h == 14 and not is_horizon_14d_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="14-day horizon disabled.")

    if h == 30 and not is_horizon_30d_enabled():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="30-day horizon disabled.")

    data = [
        {
            "Market Name": request.market,
            "Variety": request.variety,
            "Reported Date": obs.reported_date,
            "Modal Price (Rs./Quintal)": obs.modal_price,
            "Arrivals (Tonnes)": obs.arrivals,
        }
        for obs in request.history
    ]
    df = pd.DataFrame(data)

    try:
        result = _rice_pipeline.predict_horizon(
            df=df,
            horizon_days=h,
            market=request.market,
            variety=request.variety,
            prediction_date=request.prediction_date,
        )
        return LegacyPredictHorizonResponse(**result)
    except InsufficientHistoryError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Artifact unavailable: {e}")
    except Exception as e:
        logger.error(f"Error in legacy predict horizon: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error.")
