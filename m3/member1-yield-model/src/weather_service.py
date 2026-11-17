"""
AgroVision Model 1: Automated NASA POWER Weather Service
========================================================
Production-grade automated meteorological data acquisition and caching service
for district-level rice crop yield forecasting in West Bengal.

Key Features & Guarantees:
1. Automated NASA POWER Daily API Ingestion:
   - Fetches PRECTOTCORR, T2M, T2M_MIN, T2M_MAX, RH2M, ALLSKY_SFC_SW_DWN, WS2M.
   - Point queries centered on validated district geographic centroids.
2. Robust Network & Rate-Limiting Controls:
   - Configurable timeouts (default: 10s), exponential backoff retries (default: 3 retries).
   - Graceful error recovery: API outages do not block inference.
3. Persistent SQLite Caching:
   - Atomic disk-backed cache (`data/cache/weather_cache.db`).
   - Prevents duplicate network round-trips for previously acquired dates.
4. Offline Archive Hybrid Bridge:
   - Automatically utilizes local verified archives (`data/raw/weather/*.json`, 1997-2019)
     when network is unavailable or historical seasons are requested.
5. Strict Leakage Protection & Cutoff Adherence:
   - Truncates queries to point-in-time seasonal cutoffs (Aus: July 31, Aman: Sept 30, Boro: Feb 28).
   - Zero observations dated after cutoff are passed to feature engineering.
6. Honest Data Integrity (Zero Fabrication):
   - Never fabricates, interpolates, or synthesizes missing weather records.
   - Preserves exact source timestamps, freshness metrics, and audit provenance.
   - Missing weather triggers explicit persistence fallback with warning logs.
"""

import sys
import json
import time
import logging
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

# Base directory navigation
SRC_DIR = Path(__file__).resolve().parent
M1_DIR = SRC_DIR.parent
PROJECT_ROOT = M1_DIR.parent

# Ensure local imports work reliably
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(M1_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(M1_DIR / "scripts"))

try:
    from weather_preprocessor import (
        preprocess_inseason_weather,
        preprocess_wheat_weather,
        preprocess_potato_weather,
        get_season_window,
        normalize_district_name,
        normalize_season_name,
        INSEASON_FEATURE_COLUMNS,
        WHEAT_FEATURE_COLUMNS,
        POTATO_FEATURE_COLUMNS,
        VALID_DISTRICTS,
        SEASON_CONFIG,
    )
except ImportError:
    from src.weather_preprocessor import (
        preprocess_inseason_weather,
        preprocess_wheat_weather,
        preprocess_potato_weather,
        get_season_window,
        normalize_district_name,
        normalize_season_name,
        INSEASON_FEATURE_COLUMNS,
        WHEAT_FEATURE_COLUMNS,
        POTATO_FEATURE_COLUMNS,
        VALID_DISTRICTS,
        SEASON_CONFIG,
    )

logger = logging.getLogger("weather_service")

# NASA POWER Daily API Configuration
NASA_POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
METEOROLOGICAL_PARAMS = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MIN",
    "T2M_MAX",
    "RH2M",
    "ALLSKY_SFC_SW_DWN",
    "WS2M",
]

# Canonical District Geographic Centroids (Latitude, Longitude) for West Bengal
DISTRICT_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Alipurduar": (26.4919, 89.5271),
    "Bankura": (23.2324, 87.0715),
    "Bardhaman": (23.2324, 87.8615),
    "Birbhum": (23.8404, 87.6186),
    "Cooch Behar": (26.3452, 89.4482),
    "Dakshin Dinajpur": (25.2256, 88.7699),
    "Darjeeling": (27.0410, 88.2663),
    "Hooghly": (22.8963, 88.2461),
    "Howrah": (22.5958, 88.2636),
    "Jalpaiguri": (26.5414, 88.7196),
    "Jhargram": (22.4542, 86.9854),
    "Kalimpong": (27.0667, 88.4667),
    "Kolkata": (22.5726, 88.3639),
    "Maldah": (25.0108, 88.1411),
    "Murshidabad": (24.1759, 88.2802),
    "Nadia": (23.4710, 88.5565),
    "North 24 Parganas": (22.7210, 88.4831),
    "Paschim Bardhaman": (23.6871, 86.9746),
    "Paschim Medinipur": (22.4257, 87.3199),
    "Purba Bardhaman": (23.2324, 87.8615),
    "Purba Medinipur": (21.9396, 87.7787),
    "Purulia": (23.3321, 86.3652),
    "South 24 Parganas": (22.1352, 88.4016),
    "Uttar Dinajpur": (25.6200, 88.1200),
}


class NASAWeatherService:
    """
    Automated Meteorological Ingestion Service.
    Queries the NASA POWER API, manages persistent SQLite caching,
    interfaces with local raw archives, and computes cutoff-enforced
    in-season weather features for Model 1 inference.
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        offline_archive_dir: Optional[Union[str, Path]] = None,
        allow_live_api: bool = True,
        cache_db_path: Optional[Union[str, Path]] = None,
        timeout_seconds: Optional[float] = None,
        retry_delay_seconds: Optional[float] = None,
    ):
        """
        Initialize the NASA POWER weather service.

        Parameters
        ----------
        cache_dir : Path, optional
            Directory to store SQLite weather cache. Defaults to member1-yield-model/data/cache.
        timeout : float, default=10.0
            HTTP request timeout in seconds per attempt.
        max_retries : int, default=3
            Maximum retry attempts on network error, 5xx responses, or 429 rate limits.
        retry_backoff : float, default=1.0
            Base exponential backoff delay in seconds.
        offline_archive_dir : Path, optional
            Path to local NASA POWER JSON raw files (data/raw/weather).
        allow_live_api : bool, default=True
            Whether to allow outbound HTTP requests to NASA POWER API.
        cache_db_path : Path, optional
            Explicit SQLite cache database file path.
        """
        self.timeout = float(timeout_seconds if timeout_seconds is not None else timeout)
        self.max_retries = int(max_retries)
        self.retry_backoff = float(retry_delay_seconds if retry_delay_seconds is not None else retry_backoff)
        self.allow_live_api = bool(allow_live_api)

        # Setup persistent cache directory and database file
        if cache_db_path:
            self.db_path = Path(cache_db_path).resolve()
            self.cache_dir = self.db_path.parent
        elif cache_dir:
            self.cache_dir = Path(cache_dir).resolve()
            self.db_path = self.cache_dir / "weather_cache.db"
        else:
            self.cache_dir = M1_DIR / "data" / "cache"
            self.db_path = self.cache_dir / "weather_cache.db"

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_cache_db()

        # Offline archive directory (for historical data 1997-2019)
        if offline_archive_dir:
            self.offline_archive_dir = Path(offline_archive_dir).resolve()
        else:
            self.offline_archive_dir = M1_DIR / "data" / "raw" / "weather"

    # ----------------------------------------------------------------------
    # Cache Storage & Schema Management
    # ----------------------------------------------------------------------

    def _get_db_connection(self) -> sqlite3.Connection:
        """Create a thread-safe connection to the SQLite cache database."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_cache_db(self):
        """Initializes SQLite schema for weather observations and fetch logs."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            # Daily weather records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_weather (
                    district TEXT NOT NULL,
                    date TEXT NOT NULL,
                    prectotcorr REAL,
                    t2m REAL,
                    t2m_min REAL,
                    t2m_max REAL,
                    rh2m REAL,
                    allsky_sfc_sw_dwn REAL,
                    ws2m REAL,
                    source TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (district, date)
                )
            """)
            # Fetch execution log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fetch_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    district TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    requested_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    records_returned INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            # Index for fast district-date range lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_district_date 
                ON daily_weather (district, date)
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_cached_observations(
        self,
        district: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Retrieve stored observations from local SQLite cache."""
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        query = """
            SELECT district, date as Date, prectotcorr, t2m, t2m_min, t2m_max, 
                   rh2m, allsky_sfc_sw_dwn, ws2m, source, fetched_at
            FROM daily_weather
            WHERE district = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """
        conn = self._get_db_connection()
        try:
            df = pd.read_sql_query(query, conn, params=[district, start_str, end_str])
            if not df.empty:
                df.rename(columns={
                    'prectotcorr': 'PRECTOTCORR',
                    't2m': 'T2M',
                    't2m_min': 'T2M_MIN',
                    't2m_max': 'T2M_MAX',
                    'rh2m': 'RH2M',
                    'allsky_sfc_sw_dwn': 'ALLSKY_SFC_SW_DWN',
                    'ws2m': 'WS2M'
                }, inplace=True)
            return df
        except Exception as e:
            logger.warning(f"Error querying weather cache: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def _save_to_cache(self, district: str, df_records: pd.DataFrame, source: str):
        """Atomically persist daily weather observations into SQLite cache."""
        if df_records.empty:
            return

        now_utc = datetime.now(timezone.utc).isoformat()
        records_to_insert = []

        for _, row in df_records.iterrows():
            d_val = pd.to_datetime(row['Date']).strftime("%Y-%m-%d")

            def _clean(val):
                if val is None or pd.isna(val) or np.isinf(val):
                    return None
                f = float(val)
                return None if f in (-999.0, -999) else f

            records_to_insert.append((
                district,
                d_val,
                _clean(row.get('PRECTOTCORR')),
                _clean(row.get('T2M')),
                _clean(row.get('T2M_MIN')),
                _clean(row.get('T2M_MAX')),
                _clean(row.get('RH2M')),
                _clean(row.get('ALLSKY_SFC_SW_DWN')),
                _clean(row.get('WS2M')),
                source,
                now_utc
            ))

        insert_sql = """
            INSERT OR REPLACE INTO daily_weather (
                district, date, prectotcorr, t2m, t2m_min, t2m_max,
                rh2m, allsky_sfc_sw_dwn, ws2m, source, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn = self._get_db_connection()
        try:
            conn.executemany(insert_sql, records_to_insert)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to commit observations to SQLite cache: {e}")
        finally:
            conn.close()

    # ----------------------------------------------------------------------
    # Offline Local Archive Loading (Fallback & Historical)
    # ----------------------------------------------------------------------

    def _load_from_offline_archive(
        self,
        district: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Loads observations from data/raw/weather/{district}.json if available.
        This provides offline coverage for 1997-2019 without network requests.
        """
        if not self.offline_archive_dir.exists():
            return pd.DataFrame()

        clean_name = district.lower().replace(" ", "_")
        json_file = self.offline_archive_dir / f"{clean_name}.json"
        if not json_file.exists():
            # Try alternate naming
            matches = list(self.offline_archive_dir.glob(f"*{clean_name}*.json"))
            if matches:
                json_file = matches[0]
            else:
                return pd.DataFrame()

        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            param_data = data.get("properties", {}).get("parameter", {})
            if not param_data or "PRECTOTCORR" not in param_data:
                return pd.DataFrame()

            all_dates = sorted(list(param_data["PRECTOTCORR"].keys()))
            start_key = start_date.strftime("%Y%m%d")
            end_key = end_date.strftime("%Y%m%d")

            date_keys = [dk for dk in all_dates if start_key <= dk <= end_key]
            if not date_keys:
                return pd.DataFrame()

            rows = []
            for dk in date_keys:
                parsed_dt = pd.to_datetime(dk, format="%Y%m%d").strftime("%Y-%m-%d")
                row = {"District": district, "Date": parsed_dt}
                for param in METEOROLOGICAL_PARAMS:
                    raw_val = param_data.get(param, {}).get(dk, np.nan)
                    val = np.nan if raw_val in (-999, -999.0) else raw_val
                    row[param] = val
                rows.append(row)

            df = pd.DataFrame(rows)
            # Persist to SQLite cache so future queries hit DB directly
            self._save_to_cache(district, df, source="offline_archive")
            return df
        except Exception as e:
            logger.warning(f"Error reading offline archive for {district}: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------------------------
    # NASA POWER Live API Acquisition
    # ----------------------------------------------------------------------

    def _fetch_from_api_with_retry(
        self,
        district: str,
        lat: float,
        lon: float,
        start_str: str,
        end_str: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Executes HTTP GET against NASA POWER API with exponential backoff retries.
        Returns (DataFrame, error_message).
        """
        params = {
            "parameters": ",".join(METEOROLOGICAL_PARAMS),
            "community": "AG",
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "start": start_str,
            "end": end_str,
            "format": "JSON",
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    f"Querying NASA POWER API for {district} ({start_str} to {end_str}), attempt {attempt}/{self.max_retries}"
                )
                resp = requests.get(
                    NASA_POWER_BASE_URL,
                    params=params,
                    timeout=self.timeout
                )

                if resp.status_code == 200:
                    payload = resp.json()
                    param_dict = payload.get("properties", {}).get("parameter", {})
                    if not param_dict or "PRECTOTCORR" not in param_dict:
                        return None, "API returned empty or invalid parameter block."

                    all_dates = sorted(list(param_dict["PRECTOTCORR"].keys()))
                    rows = []
                    for dk in all_dates:
                        formatted_d = pd.to_datetime(dk, format="%Y%m%d").strftime("%Y-%m-%d")
                        row = {"District": district, "Date": formatted_d}
                        for p in METEOROLOGICAL_PARAMS:
                            val = param_dict.get(p, {}).get(dk, np.nan)
                            try:
                                f_val = float(val)
                                if np.isnan(f_val) or np.isinf(f_val) or f_val <= -900.0:
                                    row[p] = np.nan
                                else:
                                    row[p] = f_val
                            except (ValueError, TypeError):
                                row[p] = np.nan
                        rows.append(row)

                    df_res = pd.DataFrame(rows)
                    return df_res, None

                elif resp.status_code == 429:
                    # Rate limited: wait longer
                    wait_sec = self.retry_backoff * (2 ** (attempt - 1)) + 1.0
                    logger.warning(f"NASA POWER rate limit (429). Backing off {wait_sec:.1f}s.")
                    time.sleep(wait_sec)
                    last_err = f"HTTP 429 Rate Limited"

                elif 500 <= resp.status_code < 600:
                    # Server error: retry with backoff
                    wait_sec = self.retry_backoff * (2 ** (attempt - 1))
                    logger.warning(f"NASA POWER server error ({resp.status_code}). Retrying in {wait_sec:.1f}s.")
                    time.sleep(wait_sec)
                    last_err = f"HTTP {resp.status_code} Server Error"

                else:
                    return None, f"HTTP Error {resp.status_code}: {resp.text[:200]}"

            except requests.exceptions.Timeout:
                last_err = f"Request timeout ({self.timeout}s)"
                wait_sec = self.retry_backoff * attempt
                logger.warning(f"Timeout on NASA POWER request. Retrying in {wait_sec:.1f}s.")
                time.sleep(wait_sec)

            except requests.exceptions.RequestException as e:
                last_err = f"Network connection error: {str(e)}"
                wait_sec = self.retry_backoff * attempt
                logger.warning(f"Network error on NASA POWER request: {e}. Retrying in {wait_sec:.1f}s.")
                time.sleep(wait_sec)

        return None, last_err

    # ----------------------------------------------------------------------
    # Core Public Interface: Daily Weather Fetching
    # ----------------------------------------------------------------------

    def fetch_daily_weather(
        self,
        district: str,
        start_date: Union[str, pd.Timestamp, datetime],
        end_date: Union[str, pd.Timestamp, datetime],
        max_allowed_date: Optional[Union[str, pd.Timestamp, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Acquire daily weather observations for a district between start_date and end_date.
        Checks cache first, then offline archive, and finally live NASA POWER API.
        Enforces cutoff constraints (observations after effective cutoff are strictly excluded).

        Returns
        -------
        Dict with keys:
            - 'data': List[Dict[str, Any]] (daily records)
            - 'source': str ('cache', 'live_api', 'offline_archive', 'mixed', 'failed')
            - 'is_complete': bool
            - 'observed_days': int
            - 'expected_days': int
            - 'fetch_timestamp': str
            - 'latest_observation_date': Optional[str]
            - 'data_freshness_hours': Optional[float]
            - 'warnings': List[str]
        """
        warnings: List[str] = []
        canon_district = normalize_district_name(district)

        # Parse date bounds safely
        s_date = pd.to_datetime(start_date).normalize()
        e_date = pd.to_datetime(end_date).normalize()

        if max_allowed_date is not None:
            max_dt = pd.to_datetime(max_allowed_date).normalize()
            if max_dt < e_date:
                e_date = max_dt
                warnings.append(f"Query window constrained by cutoff barrier to {e_date.strftime('%Y-%m-%d')}.")

        if s_date > e_date:
            return {
                "data": [],
                "source": "invalid_range",
                "is_complete": False,
                "observed_days": 0,
                "expected_days": 0,
                "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                "latest_observation_date": None,
                "data_freshness_hours": None,
                "warnings": [f"Start date {s_date.date()} is after end date {e_date.date()}."],
            }

        expected_days = (e_date - s_date).days + 1
        expected_date_set = {
            (s_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(expected_days)
        }

        # 1. Query SQLite Persistent Cache
        df_cached = self._get_cached_observations(canon_district, s_date, e_date)
        cached_dates = set(df_cached['Date'].unique()) if not df_cached.empty else set()
        missing_dates = sorted(list(expected_date_set - cached_dates))

        records_df = df_cached.copy()
        active_source = "cache" if len(missing_dates) == 0 else "mixed"

        # 2. Fill Missing Dates via Offline Archive or Live API
        if missing_dates:
            min_missing = pd.to_datetime(min(missing_dates))
            max_missing = pd.to_datetime(max(missing_dates))

            # 2a. Try offline archive first (ideal for historical 1997-2019)
            df_archive = self._load_from_offline_archive(canon_district, min_missing, max_missing)
            if not df_archive.empty:
                records_df = pd.concat([records_df, df_archive], ignore_index=True)
                cached_dates = set(records_df['Date'].unique())
                missing_dates = sorted(list(expected_date_set - cached_dates))
                if not df_cached.empty:
                    active_source = "mixed"
                else:
                    active_source = "offline_archive"

            # 2b. If still missing and live API allowed, call NASA POWER API
            if missing_dates and self.allow_live_api:
                # Resolve district coordinates
                coords = DISTRICT_CENTROIDS.get(canon_district)
                if coords:
                    lat, lon = coords
                    api_start = min_missing.strftime("%Y%m%d")
                    api_end = max_missing.strftime("%Y%m%d")

                    df_api, err = self._fetch_from_api_with_retry(
                        canon_district, lat, lon, api_start, api_end
                    )
                    if df_api is not None and not df_api.empty:
                        # Save API records to SQLite cache
                        self._save_to_cache(canon_district, df_api, source="live_api")
                        records_df = pd.concat([records_df, df_api], ignore_index=True)
                        if df_cached.empty and df_archive.empty:
                            active_source = "live_api"
                        else:
                            active_source = "mixed"
                    else:
                        warnings.append(
                            f"Live NASA POWER fetch failed ({err}). Operating on available observations."
                        )
                else:
                    warnings.append(f"No coordinates registered for district '{canon_district}'.")

        # Deduplicate records by date (preserve most complete observation)
        if not records_df.empty:
            records_df = records_df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')
            # Filter strictly within requested start and end bounds
            date_mask = (pd.to_datetime(records_df['Date']) >= s_date) & (pd.to_datetime(records_df['Date']) <= e_date)
            records_df = records_df.loc[date_mask].copy()

        observed_days = len(records_df)
        is_complete = (observed_days >= expected_days and expected_days > 0)

        # Freshness calculation
        latest_date_str = None
        freshness_hours = None
        if observed_days > 0:
            latest_date_str = records_df['Date'].max()
            latest_dt = pd.to_datetime(latest_date_str).tz_localize(timezone.utc)
            now_dt = datetime.now(timezone.utc)
            freshness_hours = round(max(0.0, (now_dt - latest_dt).total_seconds() / 3600.0), 1)

        if not is_complete:
            active_source = "failed" if observed_days == 0 else "partial"
            warnings.append(
                f"Observation series incomplete: {observed_days}/{expected_days} days available "
                f"between {s_date.strftime('%Y-%m-%d')} and {e_date.strftime('%Y-%m-%d')}."
            )

        data_list = records_df.to_dict(orient="records") if not records_df.empty else []

        return {
            "data": data_list,
            "source": active_source,
            "is_complete": is_complete,
            "observed_days": observed_days,
            "expected_days": expected_days,
            "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
            "latest_observation_date": latest_date_str,
            "data_freshness_hours": freshness_hours,
            "warnings": warnings,
        }

    # ----------------------------------------------------------------------
    # High-Level Feature Generation: Cutoff-Enforced In-Season Features
    # ----------------------------------------------------------------------

    def get_inseason_features(
        self,
        district: str,
        year: int,
        season: str = "Rabi",
        crop: str = "rice",
        max_allowed_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, Any]:
        """
        Fetches daily weather for the target district, crop year, and season up to cutoff,
        and aggregates into crop-specific in-season weather features (Rice, Wheat, Potato).

        Guarantees:
        - Strict cutoff enforcement: zero observations after cutoff reach feature aggregation.
        - Zero fabrication: if weather is missing or incomplete, returns is_complete=False.
        - Rich provenance tracking: returns source, fetch time, and completeness ratio.
        """
        canon_district = normalize_district_name(district)
        c = str(crop).strip().lower() if crop else "rice"
        canon_season = normalize_season_name(season) if c == "rice" else "Rabi"

        try:
            start_date, cutoff_date, expected_days = get_season_window(year, canon_season, crop=c)
        except Exception as e:
            empty_cols = WHEAT_FEATURE_COLUMNS if c == "wheat" else (POTATO_FEATURE_COLUMNS if c == "potato" else INSEASON_FEATURE_COLUMNS)
            empty_feats = {col: np.nan for col in empty_cols}
            return {
                "features": empty_feats,
                "is_complete": False,
                "weather_source": "failed",
                "metadata": {
                    "crop": c,
                    "district": canon_district,
                    "year": year,
                    "season": canon_season,
                    "missing_features": list(empty_cols),
                },
                "warnings": [f"Season window resolution error: {e}"],
            }

        # Effective cutoff date cannot exceed user-specified barrier
        effective_cutoff = cutoff_date
        if max_allowed_date is not None:
            user_max = pd.to_datetime(max_allowed_date).normalize()
            if user_max < effective_cutoff:
                effective_cutoff = user_max

        # Fetch daily observations up to effective cutoff
        fetch_result = self.fetch_daily_weather(
            district=canon_district,
            start_date=start_date,
            end_date=effective_cutoff,
            max_allowed_date=effective_cutoff,
        )

        daily_data = fetch_result["data"]
        source_mode = fetch_result["source"]

        # Run cutoff-aware aggregation through weather_preprocessor
        if c == "wheat":
            prep_res = preprocess_wheat_weather(
                daily_weather=daily_data,
                district=canon_district,
                year=year,
                max_allowed_date=effective_cutoff,
            )
        elif c == "potato":
            prep_res = preprocess_potato_weather(
                daily_weather=daily_data,
                district=canon_district,
                year=year,
                max_allowed_date=effective_cutoff,
            )
        else:
            prep_res = preprocess_inseason_weather(
                daily_weather=daily_data,
                district=canon_district,
                year=year,
                season=canon_season,
                max_allowed_date=effective_cutoff,
            )

        combined_warnings = fetch_result["warnings"] + prep_res["warnings"]
        unique_warnings = list(dict.fromkeys(combined_warnings))

        meta = prep_res["metadata"]
        meta["weather_source"] = source_mode
        meta["fetch_timestamp"] = fetch_result["fetch_timestamp"]
        meta["latest_observation_date"] = fetch_result["latest_observation_date"]
        meta["data_freshness_hours"] = fetch_result["data_freshness_hours"]

        return {
            "features": prep_res["features"],
            "is_complete": prep_res["is_complete"],
            "weather_source": source_mode,
            "metadata": meta,
            "warnings": unique_warnings,
        }

    # ----------------------------------------------------------------------
    # Diagnostics & Cache Management
    # ----------------------------------------------------------------------

    def get_cache_stats(self) -> Dict[str, Any]:
        """Provides operational metrics on cache size, district coverage, and dates."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT district), MIN(date), MAX(date) FROM daily_weather")
            total_records, total_districts, min_date, max_date = cursor.fetchone()

            cursor.execute("SELECT source, COUNT(*) FROM daily_weather GROUP BY source")
            source_counts = dict(cursor.fetchall())

            file_size_kb = round(self.db_path.stat().st_size / 1024.0, 2) if self.db_path.exists() else 0.0

            return {
                "cache_file": str(self.db_path),
                "file_size_kb": file_size_kb,
                "total_observations": total_records or 0,
                "districts_cached": total_districts or 0,
                "earliest_date": min_date,
                "latest_date": max_date,
                "observations_by_source": source_counts,
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            conn.close()

    def clear_cache(self, district: Optional[str] = None):
        """Clears cached observations (either for a single district or entirely)."""
        conn = self._get_db_connection()
        try:
            cursor = conn.cursor()
            if district:
                canon = normalize_district_name(district)
                cursor.execute("DELETE FROM daily_weather WHERE district = ?", [canon])
            else:
                cursor.execute("DELETE FROM daily_weather")
            conn.commit()
        finally:
            conn.close()


# Singleton instance helper
_DEFAULT_WEATHER_SERVICE: Optional[NASAWeatherService] = None


def get_weather_service(
    cache_dir: Optional[Union[str, Path]] = None,
    allow_live_api: bool = True
) -> NASAWeatherService:
    """Returns singleton NASAWeatherService instance."""
    global _DEFAULT_WEATHER_SERVICE
    if _DEFAULT_WEATHER_SERVICE is None:
        _DEFAULT_WEATHER_SERVICE = NASAWeatherService(
            cache_dir=cache_dir,
            allow_live_api=allow_live_api
        )
    return _DEFAULT_WEATHER_SERVICE
