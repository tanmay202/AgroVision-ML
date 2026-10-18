"""
AgroVision-ML — Member 2: Prospective Live Shadow Data Connector

Provides isolated ingestion, normalization, freshness auditing, and drift
monitoring for West Bengal rice mandi price observations.

Architectural Guarantees:
1. Zero Production Impact: Does not alter frozen production model or endpoints.
2. Official Data Source Integration: Implements connector for the official Open
   Government Data Platform India (data.gov.in) AGMARKNET resource.
3. Secret Safety: Never hard-codes API keys; reads from DATA_GOV_IN_API_KEY.
4. Documented Fallback: Robust CSV batch connector for offline or keyless runs.
5. Strict Mapping: Explicit market and variety mappings; unmapped items are
   recorded and reported rather than silently misclassified.
6. Drift & Freshness Monitoring: Detects stale data feeds and price distribution shifts.
7. Concurrency Safety: Process-level file locking prevents overlapping daily runs.
"""

import sys
import os
import json
import time
import logging
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import urllib.request
import urllib.error
import urllib.parse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resource metadata for official Government of India Open Government Data Platform
OGD_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
OGD_BASE_URL = f"https://api.data.gov.in/resource/{OGD_RESOURCE_ID}"

# ============================================================
# 1. CANONICAL MARKET & VARIETY MAPPINGS
# ============================================================

# Known West Bengal rice mandis in AgroVision-ML
CANONICAL_WEST_BENGAL_MANDIS = {
    "Kalna", "Bishnupur(Bankura)", "Burdwan", "Alipurduar", "Jalpaiguri Sadar",
    "Egra/contai", "Samsi", "Kalyani", "Tamluk (Medinipur E)", "Champadanga",
    "Jhargram", "Memari", "Bolpur", "Barasat", "English Bazar", "Diamond Harbour(South 24-pgs)",
    "Baruipur(Canning)", "Balarampur", "Ghatal", "Rampurhat", "Katwa", "Suri",
    "Habra", "Basirhat", "Medinipur", "Coochbehar", "Siliguri", "Malda",
}

# Aliases for common spelling differences between Agmarknet and dataset
MARKET_ALIASES = {
    "bardhaman": "Burdwan",
    "burdwan": "Burdwan",
    "kalna": "Kalna",
    "kalyani": "Kalyani",
    "jhargram": "Jhargram",
    "alipurduar": "Alipurduar",
    "diamond harbour": "Diamond Harbour(South 24-pgs)",
    "diamond harbour(south 24-pgs)": "Diamond Harbour(South 24-pgs)",
    "baruipur": "Baruipur(Canning)",
    "canning": "Baruipur(Canning)",
    "baruipur(canning)": "Baruipur(Canning)",
    "bishnupur": "Bishnupur(Bankura)",
    "bishnupur(bankura)": "Bishnupur(Bankura)",
    "jalpaiguri": "Jalpaiguri Sadar",
    "jalpaiguri sadar": "Jalpaiguri Sadar",
    "tamluk": "Tamluk (Medinipur E)",
    "tamluk (medinipur e)": "Tamluk (Medinipur E)",
    "egra": "Egra/contai",
    "contai": "Egra/contai",
    "egra/contai": "Egra/contai",
}

# Canonical rice varieties
CANONICAL_VARIETIES = {
    "Common", "Fine", "Super Fine", "Other", "Ratnachudi (718 5-749)",
    "Sona Mansoori Non Basmati", "Masuri", "Coarse", "H.Y.V.", "Sona",
}

# Variety aliases from Agmarknet / OGD feeds
VARIETY_ALIASES = {
    "common": "Common",
    "paddy (dhan) (common)": "Common",
    "paddy(dhan)(common)": "Common",
    "rice (common)": "Common",
    "dhan(common)": "Common",
    "fine": "Fine",
    "paddy (dhan) (fine)": "Fine",
    "paddy(dhan)(fine)": "Fine",
    "rice (fine)": "Fine",
    "super fine": "Super Fine",
    "superfine": "Super Fine",
    "paddy (dhan) (super fine)": "Super Fine",
    "other": "Other",
    "swarna": "Other",
    "minikit": "Other",
    "ratnachudi": "Ratnachudi (718 5-749)",
    "ratnachudi (718 5-749)": "Ratnachudi (718 5-749)",
    "sona masoori": "Sona Mansoori Non Basmati",
    "sona mansoori": "Sona Mansoori Non Basmati",
    "sona mansoori non basmati": "Sona Mansoori Non Basmati",
    "masuri": "Masuri",
    "coarse": "Coarse",
}


def map_market_name(raw_name: str) -> Tuple[str, bool]:
    """
    Maps incoming mandi name to canonical AgroVision-ML market name.
    Returns (mapped_name, is_known).
    """
    clean = str(raw_name).strip()
    clean_low = clean.lower()
    if clean_low in MARKET_ALIASES:
        return MARKET_ALIASES[clean_low], True
    # Check exact match against canonical list (case-insensitive)
    for c in CANONICAL_WEST_BENGAL_MANDIS:
        if c.lower() == clean_low:
            return c, True
    return clean, False


def map_variety_name(raw_name: str) -> Tuple[str, bool]:
    """
    Maps incoming variety string to canonical AgroVision-ML variety name.
    Returns (mapped_name, is_known).
    """
    clean = str(raw_name).strip()
    clean_low = clean.lower()
    if clean_low in VARIETY_ALIASES:
        return VARIETY_ALIASES[clean_low], True
    for v in CANONICAL_VARIETIES:
        if v.lower() == clean_low:
            return v, True
    return clean, False


# ============================================================
# 2. CONCURRENCY CONTROL / PROCESS LOCKING
# ============================================================

class LockAcquisitionError(Exception):
    """Raised when another shadow execution process currently holds the lock."""
    pass


class ShadowExecutionLock:
    """
    File-based mutual exclusion lock to ensure single-instance daily shadow execution.
    Handles stale lock detection and clean release.
    """
    def __init__(self, lock_path: Optional[Union[str, Path]] = None, stale_timeout_sec: int = 3600):
        if lock_path is None:
            lock_path = Path("member2-price-volatility/data/processed/shadow_execution.lock")
        self.lock_path = Path(lock_path)
        self.stale_timeout_sec = stale_timeout_sec
        self.is_acquired = False

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.exists():
            try:
                content = self.lock_path.read_text(encoding="utf-8").strip()
                lock_data = json.loads(content)
                pid = lock_data.get("pid")
                ts = lock_data.get("timestamp", 0)
                age = time.time() - ts

                # Check if process is still alive (Windows & POSIX safe)
                pid_alive = False
                if pid:
                    try:
                        import psutil
                        pid_alive = psutil.pid_exists(pid)
                    except ImportError:
                        # Fallback without psutil: check os.kill(pid, 0) on posix or assume age check
                        if os.name == "nt":
                            pid_alive = age < self.stale_timeout_sec
                        else:
                            try:
                                os.kill(pid, 0)
                                pid_alive = True
                            except OSError:
                                pid_alive = False

                if pid_alive and age < self.stale_timeout_sec:
                    raise LockAcquisitionError(
                        f"Active lock held by PID {pid} since {lock_data.get('time_iso')}. Skipping overlapping execution."
                    )
                else:
                    logger.warning(f"Removing stale lock file from PID {pid} (age: {age:.0f}s).")
                    self.lock_path.unlink(missing_ok=True)
            except (json.JSONDecodeError, OSError):
                self.lock_path.unlink(missing_ok=True)

        payload = {
            "pid": os.getpid(),
            "timestamp": time.time(),
            "time_iso": datetime.now(timezone.utc).isoformat(),
        }
        try:
            self.lock_path.write_text(json.dumps(payload), encoding="utf-8")
            self.is_acquired = True
            return True
        except OSError as e:
            raise LockAcquisitionError(f"Failed to create lock file: {e}")

    def release(self):
        if self.is_acquired and self.lock_path.exists():
            try:
                self.lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            self.is_acquired = False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# ============================================================
# 3. OFFICIAL DATA.GOV.IN / AGMARKNET API CONNECTOR
# ============================================================

class DataConnectorAuthError(Exception):
    """Raised when data.gov.in rejects request due to missing or invalid API key."""
    pass


class DataConnectorNetworkError(Exception):
    """Raised on connection timeout or unexpected API server outage."""
    pass


class DataGovInConnector:
    """
    Official Open Government Data (data.gov.in) AGMARKNET Daily Prices Connector.
    Resource: Current Daily Price of Various Commodities from Various Markets (Mandi)
    Resource ID: 9ef84268-d588-465a-a308-a864a43d0070
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_sec: int = 15,
        max_retries: int = 2,
        state_filter: str = "West Bengal",
        commodity_filter: str = "Rice",
    ):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY") or os.getenv("AGMARKNET_API_KEY")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.state_filter = state_filter
        self.commodity_filter = commodity_filter

    def _sanitize_message(self, text: str) -> str:
        """Sanitizes error messages to ensure API keys are never exposed in logs or exceptions."""
        if not self.api_key or not text:
            return str(text)
        return str(text).replace(self.api_key, "[REDACTED_API_KEY]")

    def check_connection(self) -> Dict[str, Any]:
        """
        Tests access to the official data.gov.in endpoint.
        Returns connection diagnostics without leaking secrets.
        """
        if not self.api_key:
            return {
                "accessible": False,
                "auth_configured": False,
                "status_code": 400,
                "error_category": "missing_api_key",
                "message": (
                    "Authorization key missing. The official data.gov.in API requires a valid developer API key. "
                    "Set the DATA_GOV_IN_API_KEY environment variable. Free registration: https://data.gov.in"
                ),
                "resource_id": OGD_RESOURCE_ID,
            }

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": 1,
            "filters[state]": self.state_filter,
        }
        url = f"{OGD_BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgroVision-ML/1.0 (Agriculture Forecasting Pilot)"})
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "accessible": True,
                    "auth_configured": True,
                    "status_code": resp.status,
                    "records_total": data.get("total", 0),
                    "resource_id": OGD_RESOURCE_ID,
                    "message": "Connection to data.gov.in AGMARKNET API successful.",
                }
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            clean_body = self._sanitize_message(body[:200])
            return {
                "accessible": False,
                "auth_configured": True,
                "status_code": e.code,
                "error_category": f"http_{e.code}",
                "message": f"data.gov.in returned HTTP {e.code}: {clean_body}",
                "resource_id": OGD_RESOURCE_ID,
            }
        except Exception as e:
            return {
                "accessible": False,
                "auth_configured": True,
                "status_code": None,
                "error_category": "network_exception",
                "message": f"Network exception connecting to data.gov.in: {self._sanitize_message(str(e))}",
                "resource_id": OGD_RESOURCE_ID,
            }

    def fetch_records(
        self,
        limit: int = 1000,
        offset: int = 0,
        reported_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetches West Bengal rice mandi records from official data.gov.in API.
        Guarantees that credentials are never exposed in raised exceptions.
        """
        if not self.api_key:
            raise DataConnectorAuthError(
                "Cannot fetch records from data.gov.in: DATA_GOV_IN_API_KEY is not set."
            )

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "filters[state]": self.state_filter,
            "filters[commodity]": self.commodity_filter,
        }
        if reported_date:
            params["filters[arrival_date]"] = reported_date

        url = f"{OGD_BASE_URL}?{urllib.parse.urlencode(params)}"
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "AgroVision-ML/1.0"})
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw_data = json.loads(resp.read().decode("utf-8"))
                    records = raw_data.get("records", [])
                    return {
                        "records": records,
                        "total_in_source": raw_data.get("total", len(records)),
                        "count": len(records),
                        "offset": offset,
                        "limit": limit,
                    }
            except urllib.error.HTTPError as e:
                clean_reason = self._sanitize_message(e.reason)
                if e.code in (401, 403, 400):
                    raise DataConnectorAuthError(f"data.gov.in authentication failure (HTTP {e.code}): {clean_reason}")
                if attempt == self.max_retries:
                    raise DataConnectorNetworkError(f"data.gov.in server error (HTTP {e.code}): {clean_reason}")
                time.sleep(1.0 * (attempt + 1))
            except Exception as e:
                clean_err = self._sanitize_message(str(e))
                if attempt == self.max_retries:
                    raise DataConnectorNetworkError(f"Failed to fetch data from data.gov.in: {clean_err}")
                time.sleep(1.0 * (attempt + 1))

        return {"records": [], "total_in_source": 0, "count": 0, "offset": offset, "limit": limit}


# ============================================================
# 4. NORMALIZATION & INGESTION PIPELINE
# ============================================================

def normalize_and_map_records(
    df: pd.DataFrame,
    source_tag: str = "batch_csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes raw observations, applies explicit market and variety mappings,
    drops malformed/corrupted rows, and produces detailed auditing metadata.
    
    Returns:
    (clean_df, audit_report)
    """
    raw_count = len(df)
    col_map = {}
    for c in df.columns:
        c_low = str(c).strip().lower()
        if c_low in ("updated_date", "created_date", "last_updated", "metadata_updated"):
            continue  # Explicitly ignore envelope/metadata update timestamps
        elif c_low in ("market name", "market", "market_name"):
            col_map[c] = "market"
        elif c_low in ("variety", "commodity_variety"):
            col_map[c] = "variety"
        elif c_low in ("reported date", "reported_date", "date", "arrival_date"):
            col_map[c] = "reported_date"
        elif c_low in ("modal price (rs./quintal)", "modal_price", "modal_price_rs_quintal", "price", "modal price"):
            col_map[c] = "modal_price"
        elif c_low in ("arrivals (tonnes)", "arrivals", "arrivals_tonnes", "arrival_quantity"):
            col_map[c] = "arrivals"

    work_df = df.rename(columns=col_map).copy()
    required = ["market", "variety", "reported_date", "modal_price"]
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in observations DataFrame: {missing}")

    if "arrivals" not in work_df.columns:
        work_df["arrivals"] = 0.0

    # Parse and validate dates (format="mixed" cleanly parses both ISO YYYY-MM-DD and DD/MM/YYYY formats)
    work_df["raw_date"] = work_df["reported_date"]
    work_df["parsed_date"] = pd.to_datetime(work_df["reported_date"], errors="coerce", format="mixed")
    invalid_date_mask = work_df["parsed_date"].isna()
    work_df["reported_date"] = work_df["parsed_date"].dt.strftime("%Y-%m-%d")

    # Parse and validate prices
    work_df["modal_price"] = pd.to_numeric(work_df["modal_price"], errors="coerce")
    invalid_price_mask = work_df["modal_price"].isna() | (work_df["modal_price"] <= 0)

    # Parse arrivals
    work_df["arrivals"] = pd.to_numeric(work_df["arrivals"], errors="coerce").fillna(0.0)

    # Filter invalid records
    valid_mask = ~invalid_date_mask & ~invalid_price_mask
    valid_df = work_df[valid_mask].copy()

    # Map markets and varieties explicitly
    unmapped_mandis = set()
    unmapped_varieties = set()
    mapped_markets = []
    mapped_varieties = []

    for _, row in valid_df.iterrows():
        m_name, m_known = map_market_name(row["market"])
        v_name, v_known = map_variety_name(row["variety"])
        if not m_known:
            unmapped_mandis.add(str(row["market"]))
        if not v_known:
            unmapped_varieties.add(str(row["variety"]))
        mapped_markets.append(m_name)
        mapped_varieties.append(v_name)

    valid_df["market"] = mapped_markets
    valid_df["variety"] = mapped_varieties
    valid_df["source_id"] = source_tag
    valid_df["ingestion_timestamp"] = datetime.now(timezone.utc).isoformat()

    # Deduplicate within batch
    clean_df = valid_df.drop_duplicates(subset=["market", "variety", "reported_date"], keep="last")
    final_cols = ["market", "variety", "reported_date", "modal_price", "arrivals", "source_id", "ingestion_timestamp"]
    clean_df = clean_df[final_cols].reset_index(drop=True)

    audit = {
        "raw_input_rows": raw_count,
        "accepted_valid_rows": len(clean_df),
        "rejected_corrupt_rows": raw_count - len(valid_df),
        "rejected_invalid_dates": int(invalid_date_mask.sum()),
        "rejected_invalid_prices": int(invalid_price_mask.sum()),
        "batch_duplicates_resolved": len(valid_df) - len(clean_df),
        "unmapped_mandi_labels": list(unmapped_mandis),
        "unmapped_variety_labels": list(unmapped_varieties),
        "min_date": clean_df["reported_date"].min() if not clean_df.empty else None,
        "max_date": clean_df["reported_date"].max() if not clean_df.empty else None,
        "distinct_mandis": int(clean_df["market"].nunique()) if not clean_df.empty else 0,
        "distinct_varieties": int(clean_df["variety"].nunique()) if not clean_df.empty else 0,
    }

    return clean_df, audit


# ============================================================
# 5. DATA FRESHNESS & DISTRIBUTION DRIFT AUDIT
# ============================================================

# Historical reference baselines (derived strictly from 2002-2021 training corpus for Rice modal price)
HISTORICAL_TRAINING_BASELINE = {
    "price_mean": 2420.50,
    "price_std": 620.30,
    "price_p25": 1950.00,
    "price_median": 2380.00,
    "price_p75": 2850.00,
    "price_iqr": 900.00,
}

# Variety-specific training baselines (derived strictly from training set, N >= 1,000)
VARIETY_TRAINING_BASELINES = {
    "Common": {"median": 2500.00, "mean": 2445.29, "std": 309.24},
    "Fine": {"median": 2520.00, "mean": 2465.76, "std": 658.68},
    "Super Fine": {"median": 3100.00, "mean": 3077.69, "std": 757.35},
    "Other": {"median": 2300.00, "mean": 2262.18, "std": 701.38},
    "Coarse": {"median": 1700.00, "mean": 1753.96, "std": 480.94},
    "Masuri": {"median": 2250.00, "mean": 2214.46, "std": 525.71},
    "Ratnachudi (718 5-749)": {"median": 2312.50, "mean": 2283.77, "std": 707.08},
    "Sona Mansoori Non Basmati": {"median": 2200.00, "mean": 2254.83, "std": 458.28},
    "H.Y.V.": {"median": 2000.00, "mean": 2027.57, "std": 425.40},
    "Sona": {"median": 2400.00, "mean": 2185.19, "std": 349.26},
}

# Market-specific training baselines (derived strictly from training set, top West Bengal mandis)
MARKET_TRAINING_BASELINES = {
    "Burdwan": {"median": 1860.00, "mean": 1842.60},
    "Kalna": {"median": 1900.00, "mean": 1989.75},
    "Kalyani": {"median": 2450.00, "mean": 2596.37},
    "Jhargram": {"median": 2800.00, "mean": 2770.83},
    "Alipurduar": {"median": 2300.00, "mean": 2373.04},
    "Jalpaiguri Sadar": {"median": 2080.00, "mean": 2185.09},
    "Bishnupur(Bankura)": {"median": 2250.00, "mean": 2171.13},
    "Barasat": {"median": 1860.00, "mean": 1925.76},
    "Champadanga": {"median": 2300.00, "mean": 2226.31},
    "Bolpur": {"median": 2250.00, "mean": 2068.73},
    "Memari": {"median": 1980.00, "mean": 2002.60},
    "Tamluk (Medinipur E)": {"median": 2300.00, "mean": 2116.32},
    "English Bazar": {"median": 2150.00, "mean": 2282.33},
    "Egra/contai": {"median": 2300.00, "mean": 2179.73},
    "Samsi": {"median": 2400.00, "mean": 2392.99},
    "Diamond Harbour(South 24-pgs)": {"median": 2500.00, "mean": 2480.00},
}


def audit_data_freshness_and_drift(
    clean_df: pd.DataFrame,
    reference_date: Optional[Union[str, date]] = None,
    max_allowable_lag_days: int = 14,
) -> Dict[str, Any]:
    """
    Audits incoming data batch for reporting freshness and statistical distribution drift.
    
    Includes:
    - Calendar freshness lag auditing against max_allowable_lag_days.
    - Global price distribution checks (mean, std, median, IQR vs historical baseline).
    - Group-aware drift evaluation across individual varieties and markets (N >= 5).
    - Composition-shift disambiguation: avoids false alarms when batch simply contains
      higher-tier (e.g. Super Fine) or lower-tier (e.g. Coarse) varieties.
    """
    if clean_df.empty:
        return {
            "status": "empty_batch",
            "freshness": {"is_stale": True, "lag_days": None, "message": "Zero valid records in batch."},
            "drift": {
                "drift_detected": False,
                "composition_shift_only": False,
                "group_drift_anomalies": [],
                "message": "No data to compute distribution.",
            },
        }

    ref_dt = pd.to_datetime(reference_date).date() if reference_date else datetime.now(timezone.utc).date()
    max_reported_date = pd.to_datetime(clean_df["reported_date"].max()).date()
    lag_days = (ref_dt - max_reported_date).days

    is_stale = lag_days > max_allowable_lag_days
    stale_msg = (
        f"Data is FRESH (Reporting lag: {lag_days} calendar days)."
        if not is_stale
        else f"STALE DATA WARNING: Latest observation is {max_reported_date} ({lag_days} days prior to reference date {ref_dt})."
    )

    # Global Price statistics
    prices = clean_df["modal_price"].values.astype(float)
    p_mean = float(np.mean(prices))
    p_std = float(np.std(prices))
    p_med = float(np.median(prices))
    p_25, p_75 = np.percentile(prices, [25, 75])
    p_iqr = float(p_75 - p_25)

    base = HISTORICAL_TRAINING_BASELINE
    med_pct_diff = ((p_med - base["price_median"]) / base["price_median"]) * 100.0

    # Group-aware drift evaluation
    group_drift_anomalies = []
    group_evaluations = {"varieties": {}, "markets": {}}
    variety_shifts = []

    # Variety-level evaluations (N >= 5)
    for var, v_group in clean_df.groupby("variety"):
        if len(v_group) >= 5 and var in VARIETY_TRAINING_BASELINES:
            v_prices = v_group["modal_price"].values.astype(float)
            v_med = float(np.median(v_prices))
            v_base_med = VARIETY_TRAINING_BASELINES[var]["median"]
            v_diff_pct = ((v_med - v_base_med) / v_base_med) * 100.0
            variety_shifts.append(abs(v_diff_pct))
            is_var_drift = abs(v_diff_pct) > 35.0
            group_evaluations["varieties"][var] = {
                "count": len(v_group),
                "batch_median": round(v_med, 2),
                "baseline_median": round(v_base_med, 2),
                "pct_shift": round(v_diff_pct, 2),
                "drift_flag": is_var_drift,
            }
            if is_var_drift:
                group_drift_anomalies.append(
                    f"Variety '{var}' price shifted by {v_diff_pct:+.1f}% vs baseline (batch: {v_med:.1f}, baseline: {v_base_med:.1f})"
                )

    # Market-level evaluations (N >= 5)
    for mkt, m_group in clean_df.groupby("market"):
        if len(m_group) >= 5 and mkt in MARKET_TRAINING_BASELINES:
            m_prices = m_group["modal_price"].values.astype(float)
            m_med = float(np.median(m_prices))
            m_base_med = MARKET_TRAINING_BASELINES[mkt]["median"]
            m_diff_pct = ((m_med - m_base_med) / m_base_med) * 100.0
            is_mkt_drift = abs(m_diff_pct) > 35.0
            group_evaluations["markets"][mkt] = {
                "count": len(m_group),
                "batch_median": round(m_med, 2),
                "baseline_median": round(m_base_med, 2),
                "pct_shift": round(m_diff_pct, 2),
                "drift_flag": is_mkt_drift,
            }
            if is_mkt_drift:
                group_drift_anomalies.append(
                    f"Market '{mkt}' price shifted by {m_diff_pct:+.1f}% vs baseline (batch: {m_med:.1f}, baseline: {m_base_med:.1f})"
                )

    # Check whether global shift is primarily a variety composition effect:
    # If global median is shifted > 35%, but all evaluated varieties are within 25% of their own baselines,
    # the shift is driven by variety selection bias rather than true market price drift.
    global_raw_drift = abs(med_pct_diff) > 35.0 or (p_std / base["price_std"]) > 2.0
    composition_shift_only = False
    if global_raw_drift and variety_shifts and all(s <= 25.0 for s in variety_shifts) and not group_drift_anomalies:
        composition_shift_only = True
        drift_detected = False
        drift_msg = (
            f"COMPOSITION SHIFT OBSERVED: Global median differs by {med_pct_diff:+.1f}% from overall baseline "
            f"(Rs.{p_med:.2f} vs Rs.{base['price_median']:.2f}), but individual varieties match their specific "
            f"training baselines within 25%. Batch composition variation is nominal."
        )
    elif group_drift_anomalies:
        drift_detected = True
        drift_msg = (
            f"GROUP DISTRIBUTION DRIFT DETECTED: {len(group_drift_anomalies)} group anomaly(s) found: "
            + "; ".join(group_drift_anomalies[:3])
            + ". Documented review recommended prior to any model promotion."
        )
    elif global_raw_drift:
        drift_detected = True
        drift_msg = (
            f"DISTRIBUTION SHIFT DETECTED: Batch median price (Rs.{p_med:.2f}) differs by {med_pct_diff:+.1f}% "
            f"from training baseline (Rs.{base['price_median']:.2f}). Investigation required before model promotion."
        )
    else:
        drift_detected = False
        drift_msg = "Price distribution is consistent with historical baseline expectations (global and group-level)."

    return {
        "status": "audited",
        "freshness": {
            "source_latest_date": str(max_reported_date),
            "reference_date": str(ref_dt),
            "reporting_lag_days": lag_days,
            "max_allowable_lag_days": max_allowable_lag_days,
            "is_stale": is_stale,
            "message": stale_msg,
        },
        "drift": {
            "drift_detected": drift_detected,
            "composition_shift_only": composition_shift_only,
            "group_drift_anomalies": group_drift_anomalies,
            "group_evaluations": group_evaluations,
            "batch_stats": {
                "mean": round(p_mean, 2),
                "std": round(p_std, 2),
                "median": round(p_med, 2),
                "p25": round(float(p_25), 2),
                "p75": round(float(p_75), 2),
                "iqr": round(p_iqr, 2),
            },
            "baseline_stats": base,
            "median_pct_shift": round(med_pct_diff, 2),
            "message": drift_msg,
        },
        "coverage": {
            "distinct_markets": int(clean_df["market"].nunique()),
            "distinct_varieties": int(clean_df["variety"].nunique()),
            "total_observations": len(clean_df),
        },
    }
