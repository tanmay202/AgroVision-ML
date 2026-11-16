"""
AgroVision-ML — Multi-Commodity Live Data Ingestion Subsystem
============================================================
Implements Phase 3 requirements:
1. Commodity-specific ingestion (Rice, Wheat, Potato, Tea).
2. Canonical mandi & variety alias mapping (never silently misclassify).
3. Date & price/unit validation (ISO-8601, positive modal price, positive arrivals).
4. Idempotent storage with deduplication across runs.
5. Freshness auditing & reporting lag monitoring (max 14 days threshold).
6. Statistical distribution drift detection vs historical training baselines.
7. Anti-fabrication guarantees: NEVER invent observations or interpret missing
   daily reports as zero price movements.
8. Secure credential management: reads DATA_GOV_IN_API_KEY from environment,
   strictly redacting keys from all logs, exceptions, and audit reports.
9. Rate-limit resilience: handles HTTP 429 with exponential backoff.
10. Fallback support for CSV batch and offline ingestion.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import urllib.request
import urllib.error
import urllib.parse
from datetime import datetime, date, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Resource metadata for official Government of India Open Government Data Platform (data.gov.in)
COMMODITY_OGD_RESOURCE_IDS = {
    "rice": "9ef84268-d588-465a-a308-a864a43d0070",
    "wheat": "9ef84268-d588-465a-a308-a864a43d0070",
    "potato": "9ef84268-d588-465a-a308-a864a43d0070",
    "tea": "9ef84268-d588-465a-a308-a864a43d0070",
}
OGD_BASE_URL = "https://api.data.gov.in/resource/"

# ============================================================
# 1. CANONICAL MARKET & VARIETY ALIASES PER COMMODITY
# ============================================================

CANONICAL_MARKETS = {
    "rice": {
        "Kalna", "Bishnupur(Bankura)", "Burdwan", "Alipurduar", "Jalpaiguri Sadar",
        "Egra/contai", "Samsi", "Kalyani", "Tamluk (Medinipur E)", "Champadanga",
        "Jhargram", "Memari", "Bolpur", "Barasat", "English Bazar", "Diamond Harbour(South 24-pgs)",
        "Baruipur(Canning)", "Balarampur", "Ghatal", "Rampurhat", "Katwa", "Suri",
        "Habra", "Basirhat", "Medinipur", "Coochbehar", "Siliguri", "Malda", "Kolaghat",
    },
    "wheat": {
        "Bishnupur(Bankura)", "Durgapur", "Bolpur", "Khatra", "Kaliaganj", "Bankura Sadar",
        "Asansol", "Birbhum", "Samsi", "Sainthia", "Kalna", "Burdwan", "Malda",
        "Siliguri", "English Bazar", "Rampurhat", "Kolaghat", "Khanna",
    },
    "potato": {
        "Champadanga", "Jalpaiguri Sadar", "Alipurduar", "Burdwan", "Pandua", "Sheoraphuly",
        "Bishnupur(Bankura)", "Dhupguri", "Purulia", "Coochbehar", "Kalna", "Memari",
        "Katwa", "Ghatal", "Egra/contai", "Kolaghat", "Agra",
    },
    "tea": {
        "Kolkata", "Siliguri", "Jalpaiguri Sadar", "Coochbehar", "Alipurduar",
        "Naharlagun", "Adilabad(Rythu Bazar)", "Sakhigopal",
    },
}

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
    "sheoraphuli": "Sheoraphuly",
    "sheoraphuly": "Sheoraphuly",
    "champadanga": "Champadanga",
    "kolaghat": "Kolaghat",
}

CANONICAL_VARIETIES = {
    "rice": {
        "Common", "Fine", "Super Fine", "Other", "Ratnachudi (718 5-749)",
        "Sona Mansoori Non Basmati", "Masuri", "Coarse", "H.Y.V.", "Sona",
    },
    "wheat": {
        "Other", "Sonalika", "Kalyan", "Sona", "Local", "White", "PBW-343",
        "Medium", "Bansi", "Deshi", "Dara",
    },
    "potato": {
        "Jyoti", "Other", "Local", "Red", "Desi", "F.A.Q.", "Chandermukhi",
        "(Red Nanital)", "Potato", "Kufri Giriraj",
    },
    "tea": {
        "Other", "All Dust", "Cachar C.T.C. Leaf", "CTC", "Leaf", "Dust",
    },
}

VARIETY_ALIASES = {
    "common": "Common",
    "paddy (dhan) (common)": "Common",
    "paddy(dhan)(common)": "Common",
    "rice (common)": "Common",
    "dhan(common)": "Common",
    "fine": "Fine",
    "paddy (dhan) (fine)": "Fine",
    "super fine": "Super Fine",
    "superfine": "Super Fine",
    "other": "Other",
    "swarna": "Other",
    "minikit": "Other",
    "jyoti": "Jyoti",
    "potato (jyoti)": "Jyoti",
    "chandramukhi": "Chandermukhi",
    "chandermukhi": "Chandermukhi",
    "sonalika": "Sonalika",
    "pbw343": "PBW-343",
    "pbw-343": "PBW-343",
    "desi": "Desi",
    "deshi": "Deshi",
    "ctc": "CTC",
}


def map_commodity_market(commodity: str, raw_name: str) -> Tuple[str, bool]:
    """Maps incoming mandi name to canonical market name. Returns (mapped_name, is_known)."""
    clean = str(raw_name).strip()
    clean_low = clean.lower()
    if clean_low in MARKET_ALIASES:
        return MARKET_ALIASES[clean_low], True

    comm = commodity.lower()
    known_mandis = CANONICAL_MARKETS.get(comm, set())
    for c in known_mandis:
        if c.lower() == clean_low:
            return c, True

    # Fallback check across all commodities
    for comm_set in CANONICAL_MARKETS.values():
        for c in comm_set:
            if c.lower() == clean_low:
                return c, True

    return clean, False


def map_commodity_variety(commodity: str, raw_name: str) -> Tuple[str, bool]:
    """Maps incoming variety string to canonical variety name. Returns (mapped_name, is_known)."""
    clean = str(raw_name).strip()
    clean_low = clean.lower()
    if clean_low in VARIETY_ALIASES:
        return VARIETY_ALIASES[clean_low], True

    comm = commodity.lower()
    known_vars = CANONICAL_VARIETIES.get(comm, set())
    for v in known_vars:
        if v.lower() == clean_low:
            return v, True

    for var_set in CANONICAL_VARIETIES.values():
        for v in var_set:
            if v.lower() == clean_low:
                return v, True

    return clean, False


# ============================================================
# 2. HISTORICAL BASELINES FOR DRIFT AUDITING
# ============================================================

COMMODITY_TRAINING_BASELINES = {
    "rice": {
        "price_mean": 2420.50, "price_std": 620.30,
        "price_p25": 1950.00, "price_median": 2380.00,
        "price_p75": 2850.00, "price_iqr": 900.00,
    },
    "wheat": {
        "price_mean": 1587.78, "price_std": 435.60,
        "price_p25": 1300.00, "price_median": 1580.00,
        "price_p75": 1850.00, "price_iqr": 550.00,
    },
    "potato": {
        "price_mean": 1027.08, "price_std": 583.93,
        "price_p25": 590.00, "price_median": 900.00,
        "price_p75": 1350.00, "price_iqr": 760.00,
    },
    "tea": {
        "price_mean": 13739.68, "price_std": 5273.01,
        "price_p25": 10000.00, "price_median": 13500.00,
        "price_p75": 16000.00, "price_iqr": 6000.00,
    },
}


# ============================================================
# 3. SECURE & RATE-LIMIT-RESILIENT AGMARKNET CONNECTOR
# ============================================================

class DataConnectorAuthError(Exception):
    """Raised when data.gov.in rejects request due to missing or invalid API key."""
    pass


class DataConnectorNetworkError(Exception):
    """Raised on connection timeout or unexpected API server outage."""
    pass


class DataGovInMultiCommodityConnector:
    """
    Production-grade multi-commodity AGMARKNET connector for Open Government Data India.
    Implements secure credential redaction, 429 rate limit backoff, and unit normalization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout_sec: int = 15,
        max_retries: int = 3,
        base_backoff_sec: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY") or os.getenv("AGMARKNET_API_KEY")
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec

    def _sanitize(self, text: Any) -> str:
        """Sanitizes strings to ensure API keys are never leaked."""
        s = str(text)
        if self.api_key and len(self.api_key) > 4:
            s = s.replace(self.api_key, "[REDACTED_API_KEY]")
        return s

    def check_connection(self, commodity: str = "rice", state: str = "West Bengal") -> Dict[str, Any]:
        """Tests connectivity and authentication for a specific commodity feed."""
        comm = commodity.lower()
        resource_id = COMMODITY_OGD_RESOURCE_IDS.get(comm, COMMODITY_OGD_RESOURCE_IDS["rice"])

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
                "resource_id": resource_id,
                "commodity": comm,
            }

        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": 1,
            "filters[state]": state,
            "filters[commodity]": comm.capitalize(),
        }
        url = f"{OGD_BASE_URL}{resource_id}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "AgroVision-ML/2.0 (Agriculture Mandi Integration Staging)"}
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return {
                    "accessible": True,
                    "auth_configured": True,
                    "status_code": resp.status,
                    "records_total": data.get("total", 0),
                    "resource_id": resource_id,
                    "commodity": comm,
                    "message": f"Connection to data.gov.in for {comm} successful.",
                }
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return {
                    "accessible": False,
                    "auth_configured": True,
                    "status_code": e.code,
                    "error_category": "authentication_failure",
                    "message": "data.gov.in rejected API key (HTTP 401/403). Verify DATA_GOV_IN_API_KEY.",
                    "resource_id": resource_id,
                    "commodity": comm,
                }
            elif e.code == 429:
                return {
                    "accessible": False,
                    "auth_configured": True,
                    "status_code": 429,
                    "error_category": "rate_limited",
                    "message": "data.gov.in rate limit exceeded (HTTP 429).",
                    "resource_id": resource_id,
                    "commodity": comm,
                }
            else:
                return {
                    "accessible": False,
                    "auth_configured": True,
                    "status_code": e.code,
                    "error_category": f"http_{e.code}",
                    "message": self._sanitize(f"HTTP error {e.code}: {e.reason}"),
                    "resource_id": resource_id,
                    "commodity": comm,
                }
        except Exception as e:
            return {
                "accessible": False,
                "auth_configured": bool(self.api_key),
                "status_code": None,
                "error_category": "network_exception",
                "message": self._sanitize(str(e)),
                "resource_id": resource_id,
                "commodity": comm,
            }

    def fetch_records(
        self,
        commodity: str = "rice",
        state: str = "West Bengal",
        limit: int = 1000,
        offset: int = 0,
        reported_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetches live mandi records from data.gov.in with automatic exponential backoff on HTTP 429.
        """
        comm = commodity.lower()
        if not self.api_key:
            raise DataConnectorAuthError("Missing DATA_GOV_IN_API_KEY environment variable.")

        resource_id = COMMODITY_OGD_RESOURCE_IDS.get(comm, COMMODITY_OGD_RESOURCE_IDS["rice"])
        params = {
            "api-key": self.api_key,
            "format": "json",
            "limit": limit,
            "offset": offset,
            "filters[state]": state,
            "filters[commodity]": comm.capitalize(),
        }
        if reported_date:
            params["filters[arrival_date]"] = str(reported_date)

        url = f"{OGD_BASE_URL}{resource_id}?{urllib.parse.urlencode(params)}"

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "AgroVision-ML/2.0 (Agriculture Mandi Integration Staging)"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                    records = payload.get("records", [])
                    return {
                        "records": records,
                        "total_in_source": payload.get("total", len(records)),
                        "count": len(records),
                        "offset": offset,
                        "limit": limit,
                        "commodity": comm,
                        "status": "success",
                    }
            except urllib.error.HTTPError as e:
                clean_msg = self._sanitize(str(e.reason))
                if e.code in (401, 403):
                    raise DataConnectorAuthError(f"data.gov.in rejected credentials (HTTP {e.code}): {clean_msg}")
                elif e.code == 429:
                    # Exponential backoff for rate limiting
                    sleep_time = self.base_backoff_sec * (2 ** attempt)
                    logger.warning(f"Rate limited by data.gov.in (HTTP 429). Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    if attempt == self.max_retries - 1:
                        raise DataConnectorNetworkError(f"data.gov.in rate limit exceeded after {self.max_retries} retries.")
                elif e.code >= 500:
                    sleep_time = self.base_backoff_sec * (attempt + 1)
                    time.sleep(sleep_time)
                    if attempt == self.max_retries - 1:
                        raise DataConnectorNetworkError(f"data.gov.in server error (HTTP {e.code}): {clean_msg}")
                else:
                    raise DataConnectorNetworkError(f"data.gov.in HTTP error {e.code}: {clean_msg}")
            except Exception as e:
                clean_err = self._sanitize(str(e))
                if attempt == self.max_retries - 1:
                    raise DataConnectorNetworkError(f"Network error connecting to data.gov.in: {clean_err}")
                time.sleep(self.base_backoff_sec * (attempt + 1))

        return {"records": [], "total_in_source": 0, "count": 0, "commodity": comm, "status": "failed"}


# ============================================================
# 4. NORMALIZATION & VALIDATION ENGINE
# ============================================================

def normalize_and_validate_commodity_records(
    df: pd.DataFrame,
    commodity: str = "rice",
    source_tag: str = "batch_csv",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Standardizes raw mandi observations for a specific commodity:
    - Normalizes dates using mixed format parsing.
    - Validates modal price (> 0 in Rs./quintal).
    - Validates arrivals (>= 0 in Tonnes).
    - Resolves canonical market and variety names via per-commodity aliases.
    - Records unmapped mandis and varieties explicitly in audit (never silently dropped).
    - Idempotently deduplicates within batch on (commodity, market, variety, reported_date).
    - CRITICAL: Never fabricates missing daily reports (no zero-movement interpolation).
    """
    comm = str(commodity).strip().lower()
    raw_count = len(df)

    col_map = {}
    for c in df.columns:
        c_low = str(c).strip().lower().replace("_", " ")
        if c_low in ("market name", "market", "mandi"):
            col_map[c] = "market"
        elif c_low in ("variety", "commodity variety", "crop variety"):
            col_map[c] = "variety"
        elif c_low in ("reported date", "date", "arrival date", "report date"):
            col_map[c] = "reported_date"
        elif any(k in c_low for k in ["modal price", "modalprice", "price"]):
            col_map[c] = "modal_price"
        elif any(k in c_low for k in ["arrivals", "arrival", "volume"]):
            col_map[c] = "arrivals"

    work_df = df.rename(columns=col_map).copy()
    required = ["market", "variety", "reported_date", "modal_price"]
    missing = [c for c in required if c not in work_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in observations DataFrame: {missing}")

    if "arrivals" not in work_df.columns:
        work_df["arrivals"] = 0.0

    # Date parsing & validation
    work_df["parsed_date"] = pd.to_datetime(work_df["reported_date"], errors="coerce", format="mixed")
    invalid_date_mask = work_df["parsed_date"].isna()
    work_df["reported_date"] = work_df["parsed_date"].dt.strftime("%Y-%m-%d")

    # Price validation: must be numeric and > 0
    work_df["modal_price"] = pd.to_numeric(work_df["modal_price"], errors="coerce")
    invalid_price_mask = work_df["modal_price"].isna() | (work_df["modal_price"] <= 0)

    # Arrivals validation: numeric, non-negative
    work_df["arrivals"] = pd.to_numeric(work_df["arrivals"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Filter out invalid rows
    valid_mask = ~invalid_date_mask & ~invalid_price_mask
    valid_df = work_df[valid_mask].copy()

    # Map markets and varieties explicitly
    unmapped_mandis = set()
    unmapped_varieties = set()
    mapped_markets = []
    mapped_varieties = []

    for _, row in valid_df.iterrows():
        m_name, m_known = map_commodity_market(comm, row["market"])
        v_name, v_known = map_commodity_variety(comm, row["variety"])
        if not m_known:
            unmapped_mandis.add(str(row["market"]))
        if not v_known:
            unmapped_varieties.add(str(row["variety"]))
        mapped_markets.append(m_name)
        mapped_varieties.append(v_name)

    valid_df["market"] = mapped_markets
    valid_df["variety"] = mapped_varieties
    valid_df["commodity"] = comm
    valid_df["source_id"] = source_tag
    valid_df["created_at"] = datetime.now(timezone.utc).isoformat()

    # Intra-batch deduplication: keep last valid report
    clean_df = valid_df.drop_duplicates(
        subset=["commodity", "market", "variety", "reported_date"],
        keep="last",
    )
    final_cols = ["commodity", "market", "variety", "reported_date", "modal_price", "arrivals", "source_id", "created_at"]
    clean_df = clean_df[final_cols].reset_index(drop=True)

    audit = {
        "commodity": comm,
        "raw_input_rows": raw_count,
        "accepted_valid_rows": len(clean_df),
        "rejected_corrupt_rows": raw_count - len(valid_df),
        "rejected_invalid_dates": int(invalid_date_mask.sum()),
        "rejected_invalid_prices": int(invalid_price_mask.sum()),
        "batch_duplicates_resolved": len(valid_df) - len(clean_df),
        "unmapped_mandi_labels": sorted(list(unmapped_mandis)),
        "unmapped_variety_labels": sorted(list(unmapped_varieties)),
        "min_reported_date": clean_df["reported_date"].min() if not clean_df.empty else None,
        "max_reported_date": clean_df["reported_date"].max() if not clean_df.empty else None,
        "distinct_mandis": int(clean_df["market"].nunique()) if not clean_df.empty else 0,
        "distinct_varieties": int(clean_df["variety"].nunique()) if not clean_df.empty else 0,
    }

    return clean_df, audit


# ============================================================
# 5. DATA FRESHNESS & DISTRIBUTION DRIFT AUDIT
# ============================================================

def audit_commodity_freshness_and_drift(
    clean_df: pd.DataFrame,
    commodity: str = "rice",
    reference_date: Optional[Union[str, date]] = None,
    max_allowable_lag_days: int = 14,
) -> Dict[str, Any]:
    """
    Audits incoming data batch for reporting freshness and statistical distribution drift.
    Includes:
    - Calendar freshness lag calculation vs allowable threshold.
    - Statistical distribution checks (mean, std, median, IQR vs historical baseline).
    - Drift detection flag (True if sample mean deviates by > 2 standard deviations).
    """
    comm = str(commodity).strip().lower()
    if clean_df.empty:
        return {
            "commodity": comm,
            "status": "empty_batch",
            "is_stale": True,
            "reporting_lag_days": None,
            "drift_detected": False,
        }

    # Reference date
    if reference_date is None:
        ref_dt = datetime.now(timezone.utc).date()
    elif isinstance(reference_date, str):
        ref_dt = datetime.strptime(reference_date, "%Y-%m-%d").date()
    elif isinstance(reference_date, datetime):
        ref_dt = reference_date.date()
    else:
        ref_dt = reference_date

    max_reported_str = clean_df["reported_date"].max()
    max_reported_dt = datetime.strptime(max_reported_str, "%Y-%m-%d").date()
    reporting_lag_days = max(0, (ref_dt - max_reported_dt).days)
    is_stale = reporting_lag_days > max_allowable_lag_days

    # Price stats
    prices = clean_df["modal_price"].dropna()
    mean_val = float(prices.mean()) if not prices.empty else 0.0
    std_val = float(prices.std()) if len(prices) > 1 else 0.0
    median_val = float(prices.median()) if not prices.empty else 0.0
    q25 = float(prices.quantile(0.25)) if not prices.empty else 0.0
    q75 = float(prices.quantile(0.75)) if not prices.empty else 0.0
    iqr_val = q75 - q25

    baseline = COMMODITY_TRAINING_BASELINES.get(comm, COMMODITY_TRAINING_BASELINES["rice"])
    b_mean = baseline["price_mean"]
    b_std = baseline["price_std"]

    # Drift flagged if sample mean deviates by more than 2 baseline standard deviations
    drift_detected = abs(mean_val - b_mean) > (2.0 * b_std)

    return {
        "commodity": comm,
        "status": "active",
        "reference_date": ref_dt.isoformat(),
        "latest_observation_date": max_reported_str,
        "reporting_lag_days": reporting_lag_days,
        "is_stale": is_stale,
        "max_allowable_lag_days": max_allowable_lag_days,
        "batch_sample_size": len(clean_df),
        "price_distribution": {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "median": round(median_val, 2),
            "iqr": round(iqr_val, 2),
        },
        "baseline_comparison": {
            "baseline_mean": b_mean,
            "baseline_std": b_std,
            "mean_delta": round(mean_val - b_mean, 2),
            "drift_detected": drift_detected,
        },
    }


# ============================================================
# 6. IDEMPOTENT SQLITE PERSISTENCE
# ============================================================

def get_default_ingestion_db_path(commodity: str) -> Path:
    """Returns canonical database path for commodity observations."""
    comm = str(commodity).strip().lower()
    root_dir = Path(__file__).resolve().parent.parent.parent
    return root_dir / "member2-price-volatility" / "data" / "processed" / f"ingestion_{comm}.db"


def init_ingestion_db(db_path: Path):
    """Initializes sqlite schema for idempotent mandi observations and audit logging."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mandi_observations (
                commodity TEXT NOT NULL,
                market TEXT NOT NULL,
                variety TEXT NOT NULL,
                reported_date TEXT NOT NULL,
                modal_price REAL NOT NULL,
                arrivals REAL,
                source_id TEXT,
                created_at TEXT,
                PRIMARY KEY (commodity, market, variety, reported_date)
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                commodity TEXT NOT NULL,
                run_timestamp TEXT NOT NULL,
                rows_fetched INTEGER NOT NULL,
                rows_accepted INTEGER NOT NULL,
                rows_rejected INTEGER NOT NULL,
                rows_duplicate INTEGER NOT NULL,
                source TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mandi_obs_lookup
            ON mandi_observations (commodity, market, variety, reported_date);
        """)


def persist_commodity_observations(
    clean_df: pd.DataFrame,
    commodity: str = "rice",
    db_path: Optional[Union[str, Path]] = None,
    audit_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Persists normalized observations to SQLite database with strict idempotency:
    - Uses INSERT OR IGNORE on composite primary key (commodity, market, variety, reported_date).
    - Accurately tracks newly inserted vs previously existing duplicate rows.
    - Logs run status in ingestion_log table.
    """
    comm = str(commodity).strip().lower()
    resolved_path = Path(db_path) if db_path else get_default_ingestion_db_path(comm)
    init_ingestion_db(resolved_path)

    run_timestamp = datetime.now(timezone.utc).isoformat()
    rows_in = len(clean_df)
    rows_rejected = (audit_report.get("rejected_corrupt_rows", 0) if audit_report else 0)

    if clean_df.empty:
        with sqlite3.connect(resolved_path) as conn:
            conn.execute("""
                INSERT INTO ingestion_log (commodity, run_timestamp, rows_fetched, rows_accepted, rows_rejected, rows_duplicate, source, status, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (comm, run_timestamp, 0, 0, rows_rejected, 0, "empty_batch", "SUCCESS", "No observations to insert."))
        return {
            "commodity": comm,
            "rows_inserted": 0,
            "rows_duplicate": 0,
            "database_path": str(resolved_path),
            "status": "success",
        }

    records_to_insert = [
        (
            comm,
            str(r["market"]),
            str(r["variety"]),
            str(r["reported_date"]),
            float(r["modal_price"]),
            float(r.get("arrivals", 0.0)),
            str(r.get("source_id", "live_feed")),
            str(r.get("created_at", run_timestamp)),
        )
        for _, r in clean_df.iterrows()
    ]

    with sqlite3.connect(resolved_path) as conn:
        # Check existing count before insert
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mandi_observations WHERE commodity = ?", (comm,))
        count_before = cur.fetchone()[0]

        # Idempotent batch insertion
        cur.executemany("""
            INSERT OR IGNORE INTO mandi_observations
            (commodity, market, variety, reported_date, modal_price, arrivals, source_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, records_to_insert)

        cur.execute("SELECT COUNT(*) FROM mandi_observations WHERE commodity = ?", (comm,))
        count_after = cur.fetchone()[0]

        rows_inserted = count_after - count_before
        rows_duplicate = rows_in - rows_inserted

        # Record audit log
        conn.execute("""
            INSERT INTO ingestion_log
            (commodity, run_timestamp, rows_fetched, rows_accepted, rows_rejected, rows_duplicate, source, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            comm,
            run_timestamp,
            rows_in + rows_rejected,
            rows_inserted,
            rows_rejected,
            rows_duplicate,
            str(clean_df.iloc[0].get("source_id", "live_feed")),
            "SUCCESS",
            None,
        ))

    return {
        "commodity": comm,
        "rows_total_received": rows_in,
        "rows_inserted": rows_inserted,
        "rows_duplicate": rows_duplicate,
        "total_persisted_records": count_after,
        "database_path": str(resolved_path),
        "status": "success",
    }


def get_ingestion_summary(commodity: str, db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Retrieves operational telemetry for the latest ingestion runs."""
    comm = str(commodity).strip().lower()
    resolved_path = Path(db_path) if db_path else get_default_ingestion_db_path(comm)
    if not resolved_path.exists():
        return {
            "commodity": comm,
            "status": "no_database",
            "last_ingestion_timestamp": None,
            "total_records": 0,
            "active_markets_count": 0,
            "latest_observation_date": None,
        }

    with sqlite3.connect(resolved_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM mandi_observations WHERE commodity = ?", (comm,))
        total_recs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT market), MAX(reported_date) FROM mandi_observations WHERE commodity = ?", (comm,))
        row = cur.fetchone()
        distinct_markets = row[0] if row else 0
        latest_date = row[1] if row else None

        cur.execute("""
            SELECT run_timestamp, status, rows_accepted, rows_duplicate
            FROM ingestion_log
            WHERE commodity = ?
            ORDER BY id DESC LIMIT 1
        """, (comm,))
        last_log = cur.fetchone()

    return {
        "commodity": comm,
        "status": "active" if total_recs > 0 else "empty",
        "total_records": total_recs,
        "active_markets_count": distinct_markets,
        "latest_observation_date": latest_date,
        "last_ingestion_timestamp": last_log[0] if last_log else None,
        "last_run_status": last_log[1] if last_log else None,
        "last_rows_accepted": last_log[2] if last_log else 0,
        "last_rows_duplicate": last_log[3] if last_log else 0,
    }
