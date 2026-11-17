"""
AgroVision 3-Tier Resilient Satellite NDVI Ingestion Client
  Tier 1: NASA ORNL DAAC MODIS Terra (MOD13Q1 250m 16-day zonal subset, 3x3 km window)
  Tier 2: NASA ORNL DAAC MODIS Aqua  (MYD13Q1 250m 16-day zonal subset, 8-day staggered)
  Tier 3: ASOF Circuit Breaker (fail-secure: write zero rows, preserve last-known-good NDVI)
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import psycopg
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("NDVIClient")

IST = timezone(timedelta(hours=5, minutes=30))
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)

DB_CONFIG = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "agrovision"),
    "user": os.getenv("PGUSER", "agrovision"),
    "password": os.getenv("PGPASSWORD", "postgres"),
}

DISTRICT_COORDS: dict[str, dict[str, Any]] = {
    "WB_HG_01": {"name": "Hooghly", "lat": 22.90, "lon": 88.39},
    "WB_BR_01": {"name": "Purba Bardhaman", "lat": 23.23, "lon": 87.86},
    "WB_BK_01": {"name": "Bankura", "lat": 23.07, "lon": 87.32},
    "WB_MU_01": {"name": "Murshidabad", "lat": 24.10, "lon": 88.25},
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=12),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
def fetch_latest_modis_composites(
    client: httpx.Client, product: str, lat: float, lon: float, n_latest: int = 3
) -> list[dict[str, Any]]:
    """Fetches the n_latest zonal mean NDVI composites (3x3 km = 81 pixels) from ORNL DAAC."""
    r_dates = client.get(
        f"https://modis.ornl.gov/rst/api/v1/{product}/dates",
        params={"latitude": lat, "longitude": lon},
    )
    r_dates.raise_for_status()
    dates = r_dates.json().get("dates", [])
    if not dates:
        raise ValueError(f"No composite dates returned for {product}")

    recent = dates[-n_latest:]
    r_sub = client.get(
        f"https://modis.ornl.gov/rst/api/v1/{product}/subset",
        params={
            "latitude": lat,
            "longitude": lon,
            "band": "250m_16_days_NDVI",
            "startDate": recent[0]["modis_date"],
            "endDate": recent[-1]["modis_date"],
            "kmAboveBelow": 1,
            "kmLeftRight": 1,
        },
    )
    r_sub.raise_for_status()

    results = []
    for entry in r_sub.json().get("subset", []):
        cal = date.fromisoformat(entry["calendar_date"])
        valid = [px * 0.0001 for px in entry["data"] if -2000 <= px <= 10000]
        if valid:
            results.append({
                "obs_date": cal,
                "mean_ndvi": round(sum(valid) / len(valid), 4),
                "valid_pixels": len(valid),
                "product": product,
            })
    return results


def ingest_district_ndvi(
    district_id: str = "WB_HG_01", crop_id: str = "RICE_COMMON"
) -> dict[str, Any]:
    """
    Fetches latest Terra (MOD13Q1) and Aqua (MYD13Q1) zonal NDVI composites,
    computes ndvi_anomaly against features.ndvi_baseline, and upserts ONLY
    vegetation columns into features.environmental_features.
    """
    meta = DISTRICT_COORDS[district_id]
    observations: dict[date, dict[str, Any]] = {}
    terra_ok = False
    aqua_ok = False

    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": "AgroVision-Pipeline/2.0"}) as client:
        try:
            logger.info("[Tier 1] Querying MODIS Terra (MOD13Q1) for %s...", meta["name"])
            for row in fetch_latest_modis_composites(client, "MOD13Q1", meta["lat"], meta["lon"], n_latest=3):
                observations[row["obs_date"]] = row
            terra_ok = True
        except Exception as exc1:
            logger.warning("[Tier 1 FAILED] MOD13Q1 Terra unavailable (%s). Falling back to Tier 2 Aqua...", exc1)

        try:
            logger.info("[Tier 2] Querying MODIS Aqua (MYD13Q1) for %s...", meta["name"])
            for row in fetch_latest_modis_composites(client, "MYD13Q1", meta["lat"], meta["lon"], n_latest=3):
                observations[row["obs_date"]] = row
            aqua_ok = True
        except Exception as exc2:
            logger.warning("[Tier 2 WARNING] MYD13Q1 Aqua unavailable (%s).", exc2)

    if not observations:
        logger.critical(
            "[TIER 3 CIRCUIT BREAKER] Both MODIS Terra and Aqua failed for %s. "
            "Preserving last-known-good ASOF NDVI state.",
            district_id,
        )
        return {
            "status": "CIRCUIT_OPEN_ASOF",
            "tier": 3,
            "source": "asof-cache",
            "rows_upserted": 0,
        }

    active_tier = 1 if terra_ok else 2
    active_source = (
        "modis-terra+aqua" if (terra_ok and aqua_ok)
        else ("modis-terra" if terra_ok else "modis-aqua")
    )

    upsert_sql = """
    INSERT INTO features.environmental_features (
        time, district_id, crop_id, season_start_date,
        mean_ndvi, ndvi_anomaly
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s
    )
    ON CONFLICT (time, district_id, crop_id) DO UPDATE SET
        mean_ndvi = EXCLUDED.mean_ndvi,
        ndvi_anomaly = EXCLUDED.ndvi_anomaly;
    """

    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT day_of_year, multi_year_mean_ndvi FROM features.ndvi_baseline WHERE district_id = %s;",
            (district_id,),
        )
        baseline_map = {int(doy): float(val) for doy, val in cur.fetchall()}

        rows_to_upsert = []
        for obs_dt in sorted(observations):
            rec = observations[obs_dt]
            doy = obs_dt.timetuple().tm_yday
            base_ndvi = baseline_map.get(doy, rec["mean_ndvi"])
            anomaly = round(rec["mean_ndvi"] - base_ndvi, 4)

            dt_ist = datetime(obs_dt.year, obs_dt.month, obs_dt.day, 0, 0, tzinfo=IST)
            season_start = date(obs_dt.year, 6, 1) if obs_dt.month >= 6 else date(obs_dt.year, 1, 1)
            rows_to_upsert.append((
                dt_ist,
                district_id,
                crop_id,
                season_start,
                rec["mean_ndvi"],
                anomaly,
            ))

        cur.executemany(upsert_sql, rows_to_upsert)

    logger.info(
        "[Tier %d | %s] Upserted %d satellite NDVI composites for %s.",
        active_tier,
        active_source,
        len(rows_to_upsert),
        district_id,
    )
    return {
        "status": "OK" if active_tier == 1 else "DEGRADED_TIER2",
        "tier": active_tier,
        "source": active_source,
        "rows_upserted": len(rows_to_upsert),
    }


if __name__ == "__main__":
    res = ingest_district_ndvi("WB_HG_01")
    print("NDVI cascade result:", res)
    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS dt, mean_ndvi, ndvi_anomaly
            FROM features.environmental_features
            WHERE district_id = 'WB_HG_01' AND mean_ndvi IS NOT NULL
            ORDER BY time DESC;
        """)
        print("Stored satellite NDVI composites:")
        for r in cur.fetchall():
            print(" ", r)
