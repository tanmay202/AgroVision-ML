import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import asyncpg
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.db import get_db, lifespan
from app.schemas import (
    DriftReportResponse,
    ExplainResponse,
    ForecastResponse,
    HealthResponse,
    IngestionRunResponse,
    MarketSeriesItem,
    PriceForecastItem,
    RiskAssessmentItem,
    TelemetryProvenance,
    TradeHistoryRecord,
)
from live_ingestion.orchestrator import run_all_pipelines
from pipeline.drift_monitor import compute_drift_report
from pipeline.shap_explainer import explain_forecast
from pipeline.risk_engine import (
    compute_composite_risk,
    evaluate_asof_freshness,
    get_district_crop_yield,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

app = FastAPI(
    title="AgroVision Serving API",
    description="Asynchronous serving layer for AgroVision multi-horizon price, weather shock risk, satellite NDVI, and yield forecasts.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter()


@api_router.get("/health", response_model=HealthResponse)
async def health_check(conn: asyncpg.Connection = Depends(get_db)):
    """Verifies API liveness and PostgreSQL TCP connectivity."""
    val = await conn.fetchval("SELECT 1;")
    return {
        "status": "healthy" if val == 1 else "unhealthy",
        "database": "connected",
        "timestamp": datetime.now(UTC),
    }


@api_router.post("/ingest/run", response_model=IngestionRunResponse)
async def trigger_ingestion_cascade(
    district_id: str = Query("WB_HG_01"),
    force_weather_tier2: bool = Query(False, description="Force Tier 2 NASA POWER fallback test"),
    simulate_tier3_outage: bool = Query(False, description="Simulate Tier 3 ASOF circuit breaker trip"),
    include_market: bool = Query(False, description="Also poll live Agmarknet 2.0 XHR+OCR"),
):
    """Triggers the 3-tier resilient ingestion cascade on demand without blocking the ASGI event loop."""
    result = await asyncio.to_thread(
        run_all_pipelines,
        district_id=district_id,
        force_weather_tier2=force_weather_tier2,
        simulate_tier3_outage=simulate_tier3_outage,
        include_market=include_market,
    )
    return result


@api_router.get("/markets", response_model=list[MarketSeriesItem])
async def list_market_series(conn: asyncpg.Connection = Depends(get_db)):
    """
    Returns all registered mandi-crop combinations present in features.market_features
    along with observation counts and latest trade dates.
    """
    query = """
    SELECT
        m.mandi_id,
        md.mandi_name,
        m.crop_id,
        cd.canonical_name AS crop_name,
        m.district_id,
        ld.district_name,
        COUNT(*)::int AS total_observations,
        MAX((m.time AT TIME ZONE 'Asia/Kolkata')::date) AS latest_trade_date
    FROM features.market_features m
    JOIN features.mandi_dim md ON md.mandi_id = m.mandi_id
    JOIN features.crop_dim cd ON cd.crop_id = m.crop_id
    JOIN features.location_dim ld ON ld.district_id = m.district_id
    GROUP BY m.mandi_id, md.mandi_name, m.crop_id, cd.canonical_name, m.district_id, ld.district_name
    HAVING COUNT(*) >= 30
    ORDER BY ld.district_name, m.crop_id, total_observations DESC;
    """
    rows = await conn.fetch(query)
    return [dict(r) for r in rows]


@api_router.get("/history", response_model=list[TradeHistoryRecord])
async def get_market_history(
    mandi_id: str = Query(..., examples=["WB_MND_KALIPUR"]),
    crop_id: str = Query(..., examples=["RICE_FINE"]),
    limit: int = Query(35, ge=1, le=365),
    conn: asyncpg.Connection = Depends(get_db),
):
    """
    Extracts trailing authentic daily price and arrival observations for a given series.
    """
    query = """
    SELECT
        (time AT TIME ZONE 'Asia/Kolkata')::date AS trade_date,
        modal_price_inr,
        arrivals_mt
    FROM features.market_features
    WHERE mandi_id = $1 AND crop_id = $2
    ORDER BY time DESC
    LIMIT $3;
    """
    rows = await conn.fetch(query, mandi_id, crop_id, limit)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No trade history found for mandi '{mandi_id}' and crop '{crop_id}'.",
        )
    return [dict(r) for r in reversed(rows)]


@api_router.get("/forecast", response_model=ForecastResponse)
async def get_forecast(
    mandi_id: str = Query(..., examples=["WB_MND_KALIPUR"]),
    crop_id: str = Query(..., examples=["RICE_FINE"]),
    sim_rain_48h_mm: Optional[float] = Query(
        None,
        ge=0.0,
        le=300.0,
        description="Optional 48h forecasted rainfall override (mm) for live What-If weather shock simulation.",
    ),
    simulate_stale: bool = Query(
        False,
        description="Optional flag to test Tier 3 ASOF Circuit Breaker stale_data_warning behavior.",
    ),
    conn: asyncpg.Connection = Depends(get_db),
):
    """
    Pulls the latest multi-horizon forecast (short, 14d, 30d), calibrated volatility,
    multi-source weather+storage risk assessment, satellite NDVI telemetry,
    Tier 3 ASOF freshness status, and M1 crop yield prediction.
    """
    meta_query = """
    SELECT
        md.mandi_name,
        ld.district_id,
        ld.district_name,
        ld.storage_avail_index,
        cd.canonical_name
    FROM features.mandi_dim md
    JOIN features.location_dim ld ON ld.district_id = md.district_id
    CROSS JOIN features.crop_dim cd
    WHERE md.mandi_id = $1 AND cd.crop_id = $2;
    """
    meta = await conn.fetchrow(meta_query, mandi_id, crop_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Invalid mandi_id or crop_id.")

    # Separate ASOF lookups for Weather, Satellite NDVI, and Market freshness
    weather_asof = await conn.fetchrow(
        """
        SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS obs_date,
               COALESCE(weather_source, 'open-meteo') AS weather_source,
               forecast_rain_48h_mm,
               trailing_rain_3d_mm,
               rh_mean_pct
        FROM features.environmental_features
        WHERE district_id = $1
          AND forecast_rain_48h_mm IS NOT NULL
          AND (time AT TIME ZONE 'Asia/Kolkata')::date <= (NOW() AT TIME ZONE 'Asia/Kolkata')::date
        ORDER BY time DESC
        LIMIT 1;
        """,
        meta["district_id"],
    )

    ndvi_asof = await conn.fetchrow(
        """
        SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS obs_date,
               mean_ndvi,
               ndvi_anomaly
        FROM features.environmental_features
        WHERE district_id = $1
          AND mean_ndvi IS NOT NULL
        ORDER BY time DESC
        LIMIT 1;
        """,
        meta["district_id"],
    )

    market_latest_date = await conn.fetchval(
        """
        SELECT MAX((time AT TIME ZONE 'Asia/Kolkata')::date)
        FROM features.market_features
        WHERE mandi_id = $1 AND crop_id = $2;
        """,
        mandi_id,
        crop_id,
    )

    freshness = evaluate_asof_freshness(
        weather_obs_date=weather_asof["obs_date"] if weather_asof else None,
        weather_source=weather_asof["weather_source"] if weather_asof else "asof-empty",
        ndvi_obs_date=ndvi_asof["obs_date"] if ndvi_asof else None,
        market_latest_date=market_latest_date,
        force_stale_demo=simulate_stale,
    )

    telemetry = TelemetryProvenance(
        stale_data_warning=freshness["stale_data_warning"],
        stale_reasons=freshness["stale_reasons"],
        weather_source=weather_asof["weather_source"] if weather_asof else "asof-empty",
        weather_age_hours=freshness["weather_age_hours"],
        market_age_hours=freshness["market_age_hours"],
        latest_ndvi=float(ndvi_asof["mean_ndvi"]) if ndvi_asof else None,
        ndvi_anomaly=float(ndvi_asof["ndvi_anomaly"]) if ndvi_asof else None,
        ndvi_date=ndvi_asof["obs_date"] if ndvi_asof else None,
        ndvi_age_days=freshness["ndvi_age_days"],
    )

    forecast_query = """
    SELECT DISTINCT ON (pf.horizon_label)
        pf.horizon_label,
        pf.forecast_origin_date,
        pf.target_date,
        pf.current_modal_price,
        pf.predicted_modal_price,
        pf.predicted_price_change,
        pf.predicted_price_lower,
        pf.predicted_price_upper,
        pf.naive_baseline_price,
        pf.naive_baseline_mae,
        pf.volatility_class,
        pf.volatility_prob_low,
        pf.volatility_prob_medium,
        pf.volatility_prob_high,
        pf.model_version,
        pf.is_experimental,
        pr.run_id,
        pr.ran_at,
        ra.v_norm,
        ra.h_anomaly,
        ra.storage_vuln,
        ra.composite_risk_score,
        ra.risk_regime,
        ra.forecast_rain_48h_mm,
        ra.trailing_rain_3d_mm,
        ra.rh_mean_pct,
        ra.storage_avail_index AS ra_storage_avail,
        ra.advisory_action,
        ra.advisory_headline,
        ra.advisory_reason,
        ra.weather_adjusted_lower,
        ra.predicted_yield_kg_ha,
        ra.yield_method
    FROM serving.price_forecasts pf
    JOIN serving.pipeline_runs pr ON pr.run_id = pf.run_id
    LEFT JOIN serving.risk_assessments ra
      ON ra.run_id = pf.run_id
     AND ra.mandi_id = pf.mandi_id
     AND ra.crop_id = pf.crop_id
     AND ra.horizon_label = pf.horizon_label
    WHERE pf.mandi_id = $1 AND pf.crop_id = $2
    ORDER BY pf.horizon_label, pr.ran_at DESC;
    """
    rows = await conn.fetch(forecast_query, mandi_id, crop_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No forecasts generated yet for mandi '{mandi_id}' and crop '{crop_id}'.",
        )

    latest_run_id = str(rows[0]["run_id"])
    latest_ran_at = rows[0]["ran_at"]
    yield_info = get_district_crop_yield(meta["district_name"], crop_id)

    short_row = next((r for r in rows if r["horizon_label"] == "short"), rows[0])
    p_low = float(short_row["volatility_prob_low"] or 0.70)
    p_med = float(short_row["volatility_prob_medium"] or 0.20)
    p_high = float(short_row["volatility_prob_high"] or 0.10)

    live_rain_48h = float(weather_asof["forecast_rain_48h_mm"]) if weather_asof else None
    live_trail_3d = float(weather_asof["trailing_rain_3d_mm"]) if weather_asof else None
    live_rh = float(weather_asof["rh_mean_pct"]) if weather_asof and weather_asof["rh_mean_pct"] is not None else None

    forecast_items: list[PriceForecastItem] = []
    for r in rows:
        base_rain_48h = live_rain_48h if live_rain_48h is not None else float(r["forecast_rain_48h_mm"] or 0.0)
        base_trail_3d = live_trail_3d if live_trail_3d is not None else float(r["trailing_rain_3d_mm"] or 0.0)
        base_rh = live_rh if live_rh is not None else float(r["rh_mean_pct"] or 72.0)
        storage_idx = float(meta["storage_avail_index"])

        active_rain = float(sim_rain_48h_mm) if sim_rain_48h_mm is not None else base_rain_48h
        calc = compute_composite_risk(
            current_modal_price=float(r["current_modal_price"]),
            predicted_modal_price=float(r["predicted_modal_price"]),
            predicted_price_change=float(r["predicted_price_change"]),
            predicted_price_lower=float(r["predicted_price_lower"]),
            horizon_label=r["horizon_label"],
            vol_prob_low=p_low,
            vol_prob_medium=p_med,
            vol_prob_high=p_high,
            forecast_rain_48h_mm=active_rain,
            trailing_rain_3d_mm=base_trail_3d,
            rh_mean_pct=base_rh,
            storage_avail_index=storage_idx,
            district_name=meta["district_name"],
        )
        risk_item = RiskAssessmentItem(
            v_norm=calc["v_norm"],
            h_anomaly=calc["h_anomaly"],
            storage_vuln=calc["storage_vuln"],
            composite_risk_score=calc["composite_risk_score"],
            risk_regime=calc["risk_regime"],
            forecast_rain_48h_mm=calc["forecast_rain_48h_mm"],
            trailing_rain_3d_mm=calc["trailing_rain_3d_mm"],
            rh_mean_pct=calc["rh_mean_pct"],
            storage_avail_index=calc["storage_avail_index"],
            advisory_action=calc["advisory_action"],
            advisory_headline=calc["advisory_headline"],
            advisory_reason=calc["advisory_reason"],
            weather_adjusted_lower=calc["weather_adjusted_lower"],
            is_simulated=(sim_rain_48h_mm is not None),
        )

        forecast_items.append(
            PriceForecastItem(
                horizon_label=r["horizon_label"],
                forecast_origin_date=r["forecast_origin_date"],
                target_date=r["target_date"],
                current_modal_price=r["current_modal_price"],
                predicted_modal_price=r["predicted_modal_price"],
                predicted_price_change=r["predicted_price_change"],
                predicted_price_lower=r["predicted_price_lower"],
                predicted_price_upper=r["predicted_price_upper"],
                naive_baseline_price=r["naive_baseline_price"],
                naive_baseline_mae=r["naive_baseline_mae"],
                volatility_class=r["volatility_class"],
                volatility_prob_low=r["volatility_prob_low"],
                volatility_prob_medium=r["volatility_prob_medium"],
                volatility_prob_high=r["volatility_prob_high"],
                model_version=r["model_version"],
                is_experimental=r["is_experimental"],
                risk=risk_item,
            )
        )

    return ForecastResponse(
        mandi_id=mandi_id,
        mandi_name=meta["mandi_name"],
        district_id=meta["district_id"],
        district_name=meta["district_name"],
        storage_avail_index=float(meta["storage_avail_index"]),
        crop_id=crop_id,
        canonical_name=meta["canonical_name"],
        predicted_yield_kg_ha=yield_info["predicted_yield_kg_ha"],
        predicted_yield_q_acre=yield_info["predicted_yield_q_acre"],
        yield_method=yield_info["yield_method"],
        latest_run_id=latest_run_id,
        ran_at=latest_ran_at,
        stale_data_warning=telemetry.stale_data_warning,
        telemetry=telemetry,
        forecasts=forecast_items,
    )




@api_router.get("/drift", response_model=DriftReportResponse)
async def get_covariate_drift(
    mandi_id: Optional[str] = Query(None, description="Optional mandi filter (e.g. WB_MND_KALIPUR)"),
    crop_id: Optional[str] = Query(None, description="Optional crop filter (e.g. RICE_COMMON)"),
    window_days: int = Query(30, ge=7, le=180, description="Trailing window in days"),
):
    """Evaluates PSI and Wasserstein-1 covariate drift against frozen training distributions."""
    try:
        return await asyncio.to_thread(
            compute_drift_report,
            mandi_id=mandi_id,
            crop_id=crop_id,
            window_days=window_days,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc




@api_router.get("/explain", response_model=ExplainResponse)
async def get_forecast_explanation(
    mandi_id: str = Query(..., description="Mandi ID (e.g. WB_MND_KALIPUR)"),
    crop_id: str = Query(..., description="Crop ID (e.g. RICE_COMMON)"),
    horizon: str = Query("short", description="Forecast horizon: short, 14d, or 30d"),
    top_k: int = Query(8, ge=3, le=20, description="Number of top SHAP features to return"),
):
    """Computes request-time TreeSHAP feature attributions (in ₹/q) for a forecast horizon."""
    try:
        return await asyncio.to_thread(
            explain_forecast,
            mandi_id=mandi_id,
            crop_id=crop_id,
            horizon_label=horizon,
            top_k=top_k,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.include_router(api_router, prefix="/api")
app.include_router(api_router)

DIST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
