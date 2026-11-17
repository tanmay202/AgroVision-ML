"""
AgroVision 3-Tier Resilient Weather Ingestion Client
  Tier 1: Open-Meteo Forecast API (trailing 14d actuals + 7d forward forecast)
  Tier 2: NASA POWER Daily Point API (MERRA-2 agro-meteorological reanalysis fallback)
  Tier 3: ASOF Circuit Breaker (fail-secure: write zero rows, preserve last-known-good state)
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
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
logger = logging.getLogger("WeatherClient")

IST = timezone(timedelta(hours=5, minutes=30))
HTTP_TIMEOUT = httpx.Timeout(connect=10.0, read=25.0, write=10.0, pool=10.0)

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


def _enrich_weather_features(df: pd.DataFrame, has_forward_forecast: bool) -> pd.DataFrame:
    df = df.sort_values("obs_date").reset_index(drop=True)
    df["trailing_rain_3d_mm"] = (
        df["precipitation_mm"].rolling(window=3, min_periods=1).sum().round(2)
    )
    if has_forward_forecast:
        df["forecast_rain_48h_mm"] = (
            df["precipitation_mm"].shift(-1).fillna(0.0)
            + df["precipitation_mm"].shift(-2).fillna(0.0)
        ).round(2)
    else:
        # Reanalysis fallback has no forward storm forecast; default to 0.0 (never NULL)
        df["forecast_rain_48h_mm"] = 0.0

    t_avg = (df["t_max_c"] + df["t_min_c"]) / 2.0
    daily_gdd = (t_avg - 10.0).clip(lower=0.0).fillna(0.0)
    df["cumulative_gdd"] = daily_gdd.cumsum().round(2)
    df["moisture_deficit"] = (4.2 - df["precipitation_mm"]).clip(lower=0.0).round(2)
    return df


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
def fetch_tier1_open_meteo(lat: float, lon: float, past_days: int = 14, forecast_days: int = 7) -> pd.DataFrame:
    """Tier 1: Open-Meteo trailing observations + 7-day forward forecast in Asia/Kolkata."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "Asia/Kolkata",
    }
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": "AgroVision-Pipeline/2.0"}) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

    daily = payload["daily"]
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(daily["time"]).date,
        "precipitation_mm": [float(x or 0.0) for x in daily["precipitation_sum"]],
        "t_max_c": [float(x) if x is not None else None for x in daily["temperature_2m_max"]],
        "t_min_c": [float(x) if x is not None else None for x in daily["temperature_2m_min"]],
        "rh_mean_pct": [float(x) if x is not None else None for x in daily["relative_humidity_2m_mean"]],
    })
    return _enrich_weather_features(df, has_forward_forecast=True)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPStatusError, httpx.TransportError)),
    reraise=True,
)
def fetch_tier2_nasa_power(lat: float, lon: float, past_days: int = 14) -> pd.DataFrame:
    """Tier 2 Fallback: NASA POWER Daily Point API (MERRA-2 agro-meteorological actuals)."""
    today_ist = datetime.now(IST).date()
    start_dt = today_ist - timedelta(days=past_days)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start_dt.strftime("%Y%m%d"),
        "end": today_ist.strftime("%Y%m%d"),
        "format": "JSON",
    }
    with httpx.Client(timeout=HTTP_TIMEOUT, headers={"User-Agent": "AgroVision-Pipeline/2.0"}) as client:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        props = resp.json()["properties"]["parameter"]

    precip_map = props["PRECTOTCORR"]
    tmax_map = props["T2M_MAX"]
    tmin_map = props["T2M_MIN"]
    rh_map = props["RH2M"]

    records = []
    for ymd, rain_val in sorted(precip_map.items()):
        # NASA POWER uses -999.0 for recent days still undergoing assimilation
        if rain_val is None or float(rain_val) <= -900.0:
            continue
        obs_dt = datetime.strptime(ymd, "%Y%m%d").date()
        tmax = float(tmax_map[ymd]) if tmax_map.get(ymd, -999) > -900 else None
        tmin = float(tmin_map[ymd]) if tmin_map.get(ymd, -999) > -900 else None
        rh = float(rh_map[ymd]) if rh_map.get(ymd, -999) > -900 else None
        records.append({
            "obs_date": obs_dt,
            "precipitation_mm": float(rain_val),
            "t_max_c": tmax,
            "t_min_c": tmin,
            "rh_mean_pct": rh,
        })

    if not records:
        raise ValueError("NASA POWER returned zero valid non-sentinel daily records.")

    return _enrich_weather_features(pd.DataFrame(records), has_forward_forecast=False)


def ingest_district_weather(
    district_id: str = "WB_HG_01",
    crop_id: str = "RICE_COMMON",
    force_tier2: bool = False,
) -> dict[str, Any]:
    """
    Executes the 3-tier weather cascade and upserts only weather columns into
    features.environmental_features (preserving mean_ndvi / ndvi_anomaly).
    """
    meta = DISTRICT_COORDS[district_id]
    df: pd.DataFrame | None = None
    active_source = "asof-cache"
    active_tier = 3

    if not force_tier2:
        try:
            logger.info("[Tier 1] Querying Open-Meteo for %s (%s)...", meta["name"], district_id)
            df = fetch_tier1_open_meteo(meta["lat"], meta["lon"])
            active_source = "open-meteo"
            active_tier = 1
        except Exception as exc1:
            logger.warning("[Tier 1 FAILED] Open-Meteo unavailable (%s). Failing over to Tier 2 NASA POWER...", exc1)

    if df is None:
        try:
            logger.info("[Tier 2] Querying NASA POWER fallback for %s (%s)...", meta["name"], district_id)
            df = fetch_tier2_nasa_power(meta["lat"], meta["lon"])
            active_source = "nasa-power"
            active_tier = 2
        except Exception as exc2:
            logger.critical(
                "[TIER 3 CIRCUIT BREAKER] Both Open-Meteo and NASA POWER failed for %s (%s). "
                "Aborting write to preserve last-known-good ASOF state.",
                district_id,
                exc2,
            )
            return {
                "status": "CIRCUIT_OPEN_ASOF",
                "tier": 3,
                "source": "asof-cache",
                "rows_upserted": 0,
            }

    upsert_sql = """
    INSERT INTO features.environmental_features (
        time, district_id, crop_id, season_start_date,
        precipitation_mm, t_min_c, t_max_c,
        cumulative_gdd, moisture_deficit,
        rh_mean_pct, forecast_rain_48h_mm, trailing_rain_3d_mm, weather_source
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s,
        %s, %s, %s, %s
    )
    ON CONFLICT (time, district_id, crop_id) DO UPDATE SET
        precipitation_mm = EXCLUDED.precipitation_mm,
        t_min_c = EXCLUDED.t_min_c,
        t_max_c = EXCLUDED.t_max_c,
        cumulative_gdd = EXCLUDED.cumulative_gdd,
        moisture_deficit = EXCLUDED.moisture_deficit,
        rh_mean_pct = EXCLUDED.rh_mean_pct,
        forecast_rain_48h_mm = EXCLUDED.forecast_rain_48h_mm,
        trailing_rain_3d_mm = EXCLUDED.trailing_rain_3d_mm,
        weather_source = EXCLUDED.weather_source;
    """

    rows_to_upsert = []
    for _, r in df.iterrows():
        dt_ist = datetime(r["obs_date"].year, r["obs_date"].month, r["obs_date"].day, 0, 0, tzinfo=IST)
        season_start = date(r["obs_date"].year, 6, 1) if r["obs_date"].month >= 6 else date(r["obs_date"].year, 1, 1)
        rows_to_upsert.append((
            dt_ist,
            district_id,
            crop_id,
            season_start,
            float(r["precipitation_mm"]),
            float(r["t_min_c"]) if pd.notna(r["t_min_c"]) else None,
            float(r["t_max_c"]) if pd.notna(r["t_max_c"]) else None,
            float(r["cumulative_gdd"]),
            float(r["moisture_deficit"]),
            float(r["rh_mean_pct"]) if pd.notna(r["rh_mean_pct"]) else None,
            float(r["forecast_rain_48h_mm"]),
            float(r["trailing_rain_3d_mm"]),
            active_source,
        ))

    with psycopg.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
        cur.executemany(upsert_sql, rows_to_upsert)

    logger.info(
        "[Tier %d | %s] Upserted %d weather rows for %s.",
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
    res = ingest_district_weather("WB_HG_01")
    print("Weather cascade result:", res)
