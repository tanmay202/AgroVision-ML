"""
AgroVision Unified 3-Tier Live Ingestion Orchestrator
Coordinates all three production data feeds with standardized resilience:
  1. Market (Agmarknet 2.0 XHR+OCR -> Tier 3 ASOF Circuit Breaker)
  2. Weather (Tier 1 Open-Meteo -> Tier 2 NASA POWER -> Tier 3 ASOF Circuit Breaker)
  3. Satellite NDVI (Tier 1 MODIS Terra -> Tier 2 MODIS Aqua -> Tier 3 ASOF Circuit Breaker)
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg

from live_ingestion.agmarknet_client import AgmarknetClient
from live_ingestion.loader import load_market_features
from live_ingestion.ndvi_client import ingest_district_ndvi
from live_ingestion.weather_client import ingest_district_weather

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("IngestionOrchestrator")

IST = timezone(timedelta(hours=5, minutes=30))

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}


def ingest_market_with_circuit_breaker(
    lookback_days: int = 14,
    simulate_outage: bool = False,
) -> dict[str, Any]:
    """Fetches recent Hooghly Rice trades from Agmarknet 2.0; trips Tier 3 ASOF on outage."""
    if simulate_outage:
        logger.critical("[TIER 3 CIRCUIT BREAKER] Simulated Agmarknet outage. Preserving ASOF market state.")
        return {
            "status": "CIRCUIT_OPEN_ASOF",
            "tier": 3,
            "source": "asof-cache",
            "rows_upserted": 0,
        }

    today_ist = datetime.now(IST).date()
    from_dt = (today_ist - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    to_dt = today_ist.strftime("%Y-%m-%d")

    try:
        logger.info("[Tier 2 XHR+OCR] Querying Agmarknet Hooghly Rice (%s to %s)...", from_dt, to_dt)
        with AgmarknetClient(timeout=15.0) as client:
            raw_records = client.fetch_hooghly_rice(from_date=from_dt, to_date=to_dt, limit=50)
        upserted = load_market_features(raw_records)
        return {
            "status": "OK",
            "tier": 2,
            "source": "agmarknet-v2-ocr",
            "rows_upserted": upserted,
        }
    except Exception as exc:
        logger.critical(
            "[TIER 3 CIRCUIT BREAKER] Agmarknet ingestion failed (%s). Preserving ASOF market state.",
            exc,
        )
        return {
            "status": "CIRCUIT_OPEN_ASOF",
            "tier": 3,
            "source": "asof-cache",
            "rows_upserted": 0,
            "error": str(exc),
        }


def run_all_pipelines(
    district_id: str = "WB_HG_01",
    crop_id: str = "RICE_COMMON",
    force_weather_tier2: bool = False,
    simulate_tier3_outage: bool = False,
    include_market: bool = True,
) -> dict[str, Any]:
    """Executes all 3 ingestion feeds and returns structured tier + health telemetry."""
    started_at = datetime.now(IST).isoformat()

    if simulate_tier3_outage:
        logger.critical("[TIER 3 CIRCUIT BREAKER] Simulating upstream outage across all 3 sources.")
        weather_res = {"status": "CIRCUIT_OPEN_ASOF", "tier": 3, "source": "asof-cache", "rows_upserted": 0}
        ndvi_res = {"status": "CIRCUIT_OPEN_ASOF", "tier": 3, "source": "asof-cache", "rows_upserted": 0}
        market_res = {"status": "CIRCUIT_OPEN_ASOF", "tier": 3, "source": "asof-cache", "rows_upserted": 0}
    else:
        weather_res = ingest_district_weather(
            district_id=district_id, crop_id=crop_id, force_tier2=force_weather_tier2
        )
        ndvi_res = ingest_district_ndvi(district_id=district_id, crop_id=crop_id)
        market_res = (
            ingest_market_with_circuit_breaker(lookback_days=14, simulate_outage=False)
            if include_market
            else {"status": "SKIPPED", "tier": 0, "source": "existing-db", "rows_upserted": 0}
        )

    statuses = [weather_res["status"], ndvi_res["status"]]
    if include_market:
        statuses.append(market_res["status"])

    overall = (
        "CIRCUIT_OPEN_ASOF"
        if "CIRCUIT_OPEN_ASOF" in statuses
        else ("DEGRADED_TIER2" if "DEGRADED_TIER2" in statuses else "HEALTHY")
    )

    return {
        "overall_status": overall,
        "ran_at_ist": started_at,
        "district_id": district_id,
        "pipelines": {
            "weather": weather_res,
            "ndvi": ndvi_res,
            "market": market_res,
        },
    }


if __name__ == "__main__":
    summary = run_all_pipelines(include_market=False)
    print("Unified 3-tier ingestion summary:", summary)
