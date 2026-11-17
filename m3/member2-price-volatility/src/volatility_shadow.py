"""
AgroVision-ML — Member 2: Prospective Shadow Monitoring Pipeline for Volatility V3

Implements a real, isolated, prospective shadow monitoring pipeline for the V3 14-day
volatility classifier (rice_volatility_v3_14d_model.pkl).

Key Architectural Guarantees:
1. Production Isolation: Active production routing remains 100% committed to Calendar
   Persistence (14D) and Normalized V2 where requested. V3 predictions are never exposed
   to farmer dashboards, public APIs, or agricultural advisory outputs.
2. Frozen Model Artifacts: Uses frozen candidate rice_volatility_v3_14d_model.pkl and
   rice_volatility_v3_14d_config.json. Zero retraining, zero modifying production models.
3. Deterministic Idempotency: Forecast identity is SHA256(market + variety + origin_date + horizon + model_version).
   Repeated runs or duplicate observations are strictly no-ops.
4. Gated Delayed Evaluation: Forecasts remain in PENDING status for at least 14 complete calendar
   days. No forecast is evaluated prematurely or before its target window definitively closes.
5. Exact Target Parity: Realized target volatility is computed using the identical methodology
   as experimental V3: forward price excursion over [t, t+14] calendar days anchored by origin
   price P_t, classified using identical thresholds (LOW <= 0.014286%, HIGH > 2.666666%).
   Empty trading windows transition to WINDOW_EMPTY and are excluded from scoring.
6. Reliability & Data Integrity:
   - Safe ACID SQLite transactions with WAL mode.
   - History validation (>= 14 historical observations required; prevents tree NaN default).
   - Strict probability validation (sum = 1.0, 0 <= p <= 1).
   - Audit logging for operational tracking and failure recovery after restart.
7. Governance & Non-Inferiority:
   - Tracks 90-day duration, 2,000 completed forecasts, 35 mandis, and <= 5% ECE.
   - Predefines non-inferiority margin and clustered confidence intervals.
   - Rejects p > 0.10 alone as proof of non-inferiority.
"""

import sys
import os
import json
import sqlite3
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
import joblib

SRC_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = SRC_DIR.parent
V3_DIR = MEMBER2_DIR / "experiments" / "v3"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))
sys.path.insert(0, str(V3_DIR))

from config import (
    get_shadow_db_path,
    is_volatility_shadow_enabled,
    get_volatility_v3_14d_model_path,
    get_volatility_v3_14d_config_path,
)
from volatility_v3_features import extract_v3_volatility_features, V3_FEATURE_COLS
from volatility_v3_target_definitions import (
    VOLATILITY_CLASSES,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
    V3_14D_THRESHOLDS,
    ZERO_MOVE_CUTOFF,
)

logger = logging.getLogger(__name__)

# Canonical model version string
V3_14D_MODEL_VERSION = "rice_volatility_v3_14d_v1"

# Thresholds for 14-day calendar volatility
DEFAULT_14D_LOW_CUTOFF = V3_14D_THRESHOLDS["low_medium_threshold"]  # 0.014286%
DEFAULT_14D_HIGH_CUTOFF = V3_14D_THRESHOLDS["medium_high_threshold"]  # 2.666666%


class InsufficientHistoryError(ValueError):
    """Raised when incoming observation history has fewer than the required observations."""
    pass


class InvalidProbabilityError(ValueError):
    """Raised when model probability output violates distribution invariants."""
    pass


# ============================================================
# 1. DATABASE SCHEMA & INITIALIZATION
# ============================================================

def init_volatility_shadow_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Initializes the SQLite schema for volatility shadow monitoring.
    
    Creates:
    - mandi_observations: Real historical and newly ingested mandi records.
    - shadow_volatility_forecasts: Stored point-in-time volatility forecasts and realized outcomes.
    - shadow_volatility_audit_log: Persistent operational audit log.
    """
    resolved_path = Path(db_path) if db_path else get_shadow_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        # 1. Mandi observations table (shared with price shadow pipeline)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mandi_observations (
                market TEXT NOT NULL,
                variety TEXT NOT NULL,
                reported_date TEXT NOT NULL,
                modal_price REAL NOT NULL,
                arrivals REAL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (market, variety, reported_date)
            );
        """)

        # 2. Dedicated shadow volatility forecasts table (matching Readiness Report Section 8.2)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_volatility_forecasts (
                forecast_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                variety TEXT NOT NULL,
                origin_date TEXT NOT NULL,
                horizon_days INTEGER NOT NULL DEFAULT 14,
                origin_price REAL NOT NULL,
                persistence_pred_class TEXT NOT NULL,
                v3_pred_class TEXT NOT NULL,
                v3_prob_low REAL NOT NULL,
                v3_prob_medium REAL NOT NULL,
                v3_prob_high REAL NOT NULL,
                v3_calibrated_prob_low REAL NOT NULL,
                v3_calibrated_prob_medium REAL NOT NULL,
                v3_calibrated_prob_high REAL NOT NULL,
                model_version TEXT NOT NULL DEFAULT 'rice_volatility_v3_14d_v1',
                created_at TEXT NOT NULL,
                target_window_end TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('PENDING', 'EVALUATED', 'WINDOW_EMPTY')),
                matched_trades_count INTEGER,
                actual_min_price REAL,
                actual_max_price REAL,
                actual_pct_range REAL,
                actual_class TEXT CHECK(actual_class IN ('LOW', 'MEDIUM', 'HIGH', NULL)),
                evaluated_at TEXT,
                cohort TEXT NOT NULL DEFAULT 'PROSPECTIVE'
            );
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_vol_status 
            ON shadow_volatility_forecasts (status, target_window_end);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_vol_market 
            ON shadow_volatility_forecasts (market, cohort, status);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_vol_origin 
            ON shadow_volatility_forecasts (origin_date, horizon_days);
        """)

        # 3. Dedicated shadow volatility audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_volatility_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                cohort TEXT,
                forecasts_generated INTEGER,
                forecasts_skipped INTEGER,
                outcomes_evaluated INTEGER,
                windows_empty INTEGER,
                still_pending INTEGER,
                duration_seconds REAL,
                details_json TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_vol_audit_ts
            ON shadow_volatility_audit_log (timestamp, action);
        """)

    return resolved_path


def get_volatility_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Returns an active SQLite connection configured with Row factory and busy timeout."""
    resolved_path = init_volatility_shadow_db(db_path)
    conn = sqlite3.connect(resolved_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000;")
    return conn


# ============================================================
# 2. DETERMINISTIC IDENTIFIER & HELPERS
# ============================================================

def make_volatility_forecast_id(
    market: str,
    variety: str,
    origin_date: str,
    horizon_days: int = 14,
    model_version: str = V3_14D_MODEL_VERSION,
) -> str:
    """
    Generates a deterministic unique identifier for a shadow volatility forecast:
    SHA256(market + variety + origin_date + horizon + model_version)
    """
    norm_mkt = str(market).strip().lower()
    norm_var = str(variety).strip().lower()
    norm_date = str(origin_date).strip()
    norm_h = str(int(horizon_days))
    norm_ver = str(model_version).strip().lower()
    key = f"{norm_mkt}|{norm_var}|{norm_date}|{norm_h}|{norm_ver}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def classify_volatility_range(
    pct_range: Optional[float],
    low_thresh: float = DEFAULT_14D_LOW_CUTOFF,
    high_thresh: float = DEFAULT_14D_HIGH_CUTOFF,
) -> Optional[str]:
    """
    Classifies continuous percentage range into discrete classes:
    - LOW: pct_range <= low_thresh (0.014286)
    - MEDIUM: low_thresh < pct_range <= high_thresh (2.666666)
    - HIGH: pct_range > high_thresh
    """
    if pct_range is None or np.isnan(pct_range):
        return None
    val = float(pct_range)
    if val <= low_thresh:
        return "LOW"
    elif val <= high_thresh:
        return "MEDIUM"
    else:
        return "HIGH"


def calculate_calendar_persistence_prediction(
    past_prices: List[float],
    origin_price: float,
    low_thresh: float = DEFAULT_14D_LOW_CUTOFF,
    high_thresh: float = DEFAULT_14D_HIGH_CUTOFF,
) -> Tuple[str, float]:
    """
    Calculates the 14-day calendar persistence prediction over [t-14, t]:
    Symmetric past price excursion anchored by origin price P_t.
    """
    pt = float(origin_price)
    valid_past = [p for p in past_prices if p is not None and not np.isnan(p) and p > 0]
    all_prices = [pt] + valid_past
    p_max = max(all_prices)
    p_min = min(all_prices)
    past_pct_range = ((p_max - p_min) / (pt + 1e-4)) * 100.0
    pred_class = classify_volatility_range(past_pct_range, low_thresh, high_thresh)
    return pred_class, past_pct_range


def validate_probability_vector(probs: np.ndarray, atol: float = 1e-4) -> bool:
    """Validates that probabilities are bounded in [0, 1] and sum to 1.0 within tolerance."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or len(probs) != 3:
        raise InvalidProbabilityError(f"Expected 3-element probability vector, got shape {probs.shape}")
    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
        raise InvalidProbabilityError("Probabilities contain NaN or Inf values")
    if np.any(probs < -atol) or np.any(probs > 1.0 + atol):
        raise InvalidProbabilityError(f"Probabilities out of bounds [0, 1]: {probs}")
    total = float(np.sum(probs))
    if abs(total - 1.0) > atol:
        raise InvalidProbabilityError(f"Probabilities do not sum to 1.0: sum={total:.6f}")
    return True


# ============================================================
# 3. SHADOW PREDICTION SERVICE
# ============================================================

class VolatilityShadowPredictionService:
    """
    Isolated shadow prediction service for V3 14-day volatility classifier.
    
    Guarantees:
    - Never modifies production routing or public advisory outputs.
    - Idempotent: repeated calls for the same observation are no-ops.
    - Gated to point-in-time observations strictly <= origin_date.
    - Strictly validates >= 14 observations to avoid tree default false alarms.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.model_path = Path(model_path) if model_path else get_volatility_v3_14d_model_path()
        self.config_path = Path(config_path) if config_path else get_volatility_v3_14d_config_path()
        self.db_path = Path(db_path) if db_path else get_shadow_db_path()

        self._model = None
        self._calibrator = None
        self._config = None
        self._feature_names = None
        self._thresholds = None

    def _load_artifacts(self):
        """Loads and caches frozen model and configuration artifacts."""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Frozen V3 14D model artifact not found at {self.model_path}. "
                "Ensure candidate artifact is placed in artifacts/v3/."
            )
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Frozen V3 14D config artifact not found at {self.config_path}."
            )

        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        loaded_obj = joblib.load(self.model_path)
        if isinstance(loaded_obj, dict):
            self._model = loaded_obj["model"]
            self._calibrator = loaded_obj.get("calibrator_isotonic")
            self._feature_names = loaded_obj.get("feature_names", V3_FEATURE_COLS)
            self._thresholds = loaded_obj.get("thresholds", V3_14D_THRESHOLDS)
        else:
            self._model = loaded_obj
            self._calibrator = None
            self._feature_names = V3_FEATURE_COLS
            self._thresholds = V3_14D_THRESHOLDS

        logger.info(
            "V3 14D model artifacts loaded successfully from %s (calibrator=%s)",
            self.model_path,
            "present" if self._calibrator is not None else "none",
        )

    def generate_forecast_record(
        self,
        market: str,
        variety: str,
        origin_date: str,
        history_df: pd.DataFrame,
        cohort: str = "PROSPECTIVE",
    ) -> Dict[str, Any]:
        """
        Generates a point-in-time shadow forecast record for a single mandi observation.
        """
        self._load_artifacts()

        # Normalize column names in history_df
        clean_df = history_df.copy()
        col_map = {}
        for c in clean_df.columns:
            c_low = str(c).strip().lower()
            if c_low in ("market name", "market", "market_name"):
                col_map[c] = "Market Name"
            elif c_low in ("variety", "commodity_variety"):
                col_map[c] = "Variety"
            elif c_low in ("reported date", "reported_date", "date"):
                col_map[c] = "Reported Date"
            elif c_low in ("modal price (rs./quintal)", "modal_price", "modal_price_rs_quintal", "price"):
                col_map[c] = "Modal Price (Rs./Quintal)"
            elif c_low in ("arrivals (tonnes)", "arrivals", "arrivals_tonnes"):
                col_map[c] = "Arrivals (Tonnes)"

        clean_df = clean_df.rename(columns=col_map)
        if "Market Name" not in clean_df.columns:
            clean_df["Market Name"] = market
        if "Variety" not in clean_df.columns:
            clean_df["Variety"] = variety
        if "Arrivals (Tonnes)" not in clean_df.columns:
            clean_df["Arrivals (Tonnes)"] = 0.0

        clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"], errors="coerce")
        origin_dt = pd.to_datetime(origin_date)
        clean_df = clean_df[clean_df["Reported Date"] <= origin_dt].drop_duplicates(subset=["Reported Date"], keep="last").sort_values("Reported Date").reset_index(drop=True)

        # Operational safety mandate: Require >= 14 observations to prevent all-NaN tree default
        if len(clean_df) < 14:
            raise InsufficientHistoryError(
                f"Market '{market}' Variety '{variety}' has only {len(clean_df)} historical records at {origin_date}; "
                f"minimum 14 observations required to prevent false HIGH alarms."
            )

        # Ensure origin row matches origin_date
        origin_row = clean_df.iloc[-1]
        if origin_row["Reported Date"].strftime("%Y-%m-%d") != origin_date:
            raise ValueError(
                f"Latest observation in history ({origin_row['Reported Date'].strftime('%Y-%m-%d')}) "
                f"does not match requested origin_date {origin_date}."
            )

        origin_price = float(origin_row["Modal Price (Rs./Quintal)"])
        if origin_price <= 0:
            raise ValueError(f"Invalid non-positive origin price: {origin_price}")

        # 1. Calculate 14D Calendar Persistence Prediction
        start_14d = origin_dt - pd.Timedelta(days=14)
        past_14d_sub = clean_df[(clean_df["Reported Date"] >= start_14d) & (clean_df["Reported Date"] < origin_dt)]
        past_prices = past_14d_sub["Modal Price (Rs./Quintal)"].astype(float).tolist()
        persistence_class, _ = calculate_calendar_persistence_prediction(
            past_prices=past_prices,
            origin_price=origin_price,
            low_thresh=self._thresholds.get("low_medium_threshold", DEFAULT_14D_LOW_CUTOFF),
            high_thresh=self._thresholds.get("medium_high_threshold", DEFAULT_14D_HIGH_CUTOFF),
        )

        # 2. Extract 38 Point-in-Time Features for V3 Model
        feat_df = extract_v3_volatility_features(
            clean_df,
            group_cols=["Market Name", "Variety"],
            date_col="Reported Date",
            price_col="Modal Price (Rs./Quintal)",
            arrival_col="Arrivals (Tonnes)" if "Arrivals (Tonnes)" in clean_df.columns else "Arrivals",
        )
        X_origin = feat_df[self._feature_names].iloc[[-1]]

        # 3. Model Predictions (Raw & Calibrated)
        raw_proba = self._model.predict_proba(X_origin)[0]
        validate_probability_vector(raw_proba)

        if self._calibrator is not None:
            cal_proba = self._calibrator.predict_proba(X_origin)[0]
            validate_probability_vector(cal_proba)
        else:
            cal_proba = raw_proba

        # Invariant check: normalized probabilities
        raw_p_low, raw_p_med, raw_p_high = float(raw_proba[0]), float(raw_proba[1]), float(raw_proba[2])
        cal_p_low, cal_p_med, cal_p_high = float(cal_proba[0]), float(cal_proba[1]), float(cal_proba[2])

        # Predicted class from calibrated probabilities
        v3_pred_idx = int(np.argmax(cal_proba))
        v3_pred_class = VOLATILITY_CLASSES[v3_pred_idx]

        # 4. Target Window End Date: origin_date + 14 calendar days
        target_window_end = (origin_dt + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

        # 5. Deterministic Forecast ID
        forecast_id = make_volatility_forecast_id(
            market=market,
            variety=variety,
            origin_date=origin_date,
            horizon_days=14,
            model_version=V3_14D_MODEL_VERSION,
        )

        now_ts = datetime.now(timezone.utc).isoformat()

        return {
            "forecast_id": forecast_id,
            "market": market,
            "variety": variety,
            "origin_date": origin_date,
            "horizon_days": 14,
            "origin_price": origin_price,
            "persistence_pred_class": persistence_class,
            "v3_pred_class": v3_pred_class,
            "v3_prob_low": raw_p_low,
            "v3_prob_medium": raw_p_med,
            "v3_prob_high": raw_p_high,
            "v3_calibrated_prob_low": cal_p_low,
            "v3_calibrated_prob_medium": cal_p_med,
            "v3_calibrated_prob_high": cal_p_high,
            "model_version": V3_14D_MODEL_VERSION,
            "created_at": now_ts,
            "target_window_end": target_window_end,
            "status": "PENDING",
            "cohort": cohort,
        }

    def record_forecast(
        self,
        record: Dict[str, Any],
        db_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """
        Stores a shadow volatility forecast into the database idempotently.
        Returns True if inserted, False if skipped as duplicate.
        """
        with get_volatility_db_connection(db_path or self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM shadow_volatility_forecasts WHERE forecast_id = ?;", (record["forecast_id"],))
            if cursor.fetchone() is not None:
                return False

            cursor.execute("""
                INSERT OR IGNORE INTO shadow_volatility_forecasts (
                    forecast_id, market, variety, origin_date, horizon_days,
                    origin_price, persistence_pred_class, v3_pred_class,
                    v3_prob_low, v3_prob_medium, v3_prob_high,
                    v3_calibrated_prob_low, v3_calibrated_prob_medium, v3_calibrated_prob_high,
                    model_version, created_at, target_window_end, status, cohort
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record["forecast_id"],
                record["market"],
                record["variety"],
                record["origin_date"],
                record["horizon_days"],
                record["origin_price"],
                record["persistence_pred_class"],
                record["v3_pred_class"],
                record["v3_prob_low"],
                record["v3_prob_medium"],
                record["v3_prob_high"],
                record["v3_calibrated_prob_low"],
                record["v3_calibrated_prob_medium"],
                record["v3_calibrated_prob_high"],
                record["model_version"],
                record["created_at"],
                record["target_window_end"],
                record["status"],
                record["cohort"],
            ))
            conn.commit()
            return cursor.rowcount > 0


# ============================================================
# 4. BATCH INGESTION & SHADOW FORECAST GENERATION
# ============================================================

def process_volatility_shadow_ingestion(
    history_df: pd.DataFrame,
    origin_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    cohort: str = "PROSPECTIVE",
    force_enabled: bool = False,
) -> Dict[str, Any]:
    """
    Ingests observations and generates idempotent shadow forecasts for all reporting mandis.
    """
    if not force_enabled and not is_volatility_shadow_enabled():
        return {
            "status": "disabled",
            "message": "Volatility shadow monitoring is disabled. Set AGROVISION_ENABLE_VOLATILITY_SHADOW=true.",
            "generated_count": 0,
            "skipped_count": 0,
        }

    # Normalize column names
    clean_df = history_df.copy()
    col_map = {}
    for c in clean_df.columns:
        c_low = str(c).strip().lower()
        if c_low in ("market name", "market", "market_name"):
            col_map[c] = "Market Name"
        elif c_low in ("variety", "commodity_variety"):
            col_map[c] = "Variety"
        elif c_low in ("reported date", "reported_date", "date"):
            col_map[c] = "Reported Date"
        elif c_low in ("modal price (rs./quintal)", "modal_price", "modal_price_rs_quintal", "price"):
            col_map[c] = "Modal Price (Rs./Quintal)"
        elif c_low in ("arrivals (tonnes)", "arrivals", "arrivals_tonnes"):
            col_map[c] = "Arrivals (Tonnes)"

    clean_df = clean_df.rename(columns=col_map)
    clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"])
    clean_df["Modal Price (Rs./Quintal)"] = pd.to_numeric(clean_df["Modal Price (Rs./Quintal)"], errors="coerce")
    if "Arrivals (Tonnes)" not in clean_df.columns:
        clean_df["Arrivals (Tonnes)"] = 0.0
    clean_df["Arrivals (Tonnes)"] = pd.to_numeric(clean_df["Arrivals (Tonnes)"], errors="coerce").fillna(0.0)

    clean_df = clean_df.dropna(subset=["Market Name", "Variety", "Reported Date", "Modal Price (Rs./Quintal)"])
    clean_df = clean_df[clean_df["Modal Price (Rs./Quintal)"] > 0]

    # Ingest into mandi_observations table
    init_volatility_shadow_db(db_path)
    now_ts = datetime.now(timezone.utc).isoformat()
    obs_tuples = clean_df[["Market Name", "Variety", "Reported Date", "Modal Price (Rs./Quintal)", "Arrivals (Tonnes)"]].itertuples(index=False)
    obs_records = [
        (
            str(row[0]).strip(),
            str(row[1]).strip(),
            row[2].strftime("%Y-%m-%d") if hasattr(row[2], "strftime") else str(row[2])[:10],
            float(row[3]),
            float(row[4]),
            now_ts,
        )
        for row in obs_tuples
    ]
    with get_volatility_db_connection(db_path) as conn:
        conn.executemany("""
            INSERT INTO mandi_observations (market, variety, reported_date, modal_price, arrivals, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, variety, reported_date) DO UPDATE SET
                modal_price = excluded.modal_price,
                arrivals = excluded.arrivals;
        """, obs_records)
        conn.commit()

    if origin_date is None:
        origin_date = clean_df["Reported Date"].max().strftime("%Y-%m-%d")

    service = VolatilityShadowPredictionService(db_path=db_path)
    generated = 0
    skipped = 0
    errors = []

    for (mkt, var), group in clean_df.groupby(["Market Name", "Variety"]):
        # Check if group has observation at origin_date
        grp_origin = group[group["Reported Date"] == pd.to_datetime(origin_date)]
        if grp_origin.empty:
            continue

        # If incoming batch history has fewer than 14 records, retrieve historical observations
        # from database to assemble full historical context up to origin_date
        feed_history = group
        if len(feed_history[feed_history["Reported Date"] <= pd.to_datetime(origin_date)]) < 14:
            with get_volatility_db_connection(db_path) as conn:
                db_hist = pd.read_sql_query("""
                    SELECT market as "Market Name", variety as "Variety",
                           reported_date as "Reported Date", modal_price as "Modal Price (Rs./Quintal)",
                           arrivals as "Arrivals (Tonnes)"
                    FROM mandi_observations
                    WHERE market = ? AND variety = ? AND reported_date <= ?
                    ORDER BY reported_date ASC;
                """, conn, params=(mkt, var, origin_date))
            if not db_hist.empty:
                feed_history = db_hist

        try:
            record = service.generate_forecast_record(
                market=mkt,
                variety=var,
                origin_date=origin_date,
                history_df=feed_history,
                cohort=cohort,
            )
            inserted = service.record_forecast(record, db_path=db_path)
            if inserted:
                generated += 1
            else:
                skipped += 1
        except InsufficientHistoryError as e:
            # Safely log; do not allow tree model to predict on insufficient data
            logger.debug("Skipping %s/%s at %s: %s", mkt, var, origin_date, str(e))
            continue
        except Exception as e:
            logger.error("Failed shadow forecast for %s/%s at %s: %s", mkt, var, origin_date, str(e))
            errors.append(f"{mkt}/{var}: {str(e)}")

    return {
        "status": "success",
        "origin_date": origin_date,
        "cohort": cohort,
        "observations_ingested": len(obs_records),
        "generated_count": generated,
        "skipped_duplicate_count": skipped,
        "errors_count": len(errors),
        "errors": errors[:5],
    }


# ============================================================
# 5. DELAYED OUTCOME EVALUATION
# ============================================================

class VolatilityShadowEvaluator:
    """
    Evaluates completed 14-day shadow volatility forecasts against realized observations.
    
    Invariants:
    - Never evaluates before the complete 14-calendar-day prediction horizon has elapsed.
    - Uses exact V3 excursion methodology: max/min over [t, t+14] anchored by origin price P_t.
    - If zero trading occurred in (t, t+14], marks status as WINDOW_EMPTY.
    - Safe idempotent state machine: re-running does not corrupt already EVALUATED forecasts.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        self.db_path = Path(db_path) if db_path else get_shadow_db_path()

    def evaluate_pending_forecasts(
        self,
        as_of_date: Optional[str] = None,
        db_path: Optional[Union[str, Path]] = None,
        recheck_empty: bool = False,
    ) -> Dict[str, int]:
        """
        Evaluates pending forecasts whose 14-day calendar window has elapsed.
        """
        target_db = Path(db_path) if db_path else self.db_path
        conn = get_volatility_db_connection(target_db)
        cursor = conn.cursor()

        if as_of_date is None:
            # Determine as_of_date from latest reported observation
            cursor.execute("SELECT MAX(reported_date) as max_dt FROM mandi_observations;")
            row = cursor.fetchone()
            as_of_date = row["max_dt"] if row and row["max_dt"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")

        as_of_dt = pd.to_datetime(as_of_date)

        # STRICT HORIZON GATING:
        # Only select forecasts whose target_window_end <= as_of_date.
        # Forecasts whose 14-calendar-day window has NOT elapsed REMAIN PENDING.
        status_filter = "status IN ('PENDING', 'WINDOW_EMPTY')" if recheck_empty else "status = 'PENDING'"
        cursor.execute(f"""
            SELECT 
                forecast_id, market, variety, origin_date, horizon_days,
                origin_price, persistence_pred_class, v3_pred_class,
                target_window_end, status
            FROM shadow_volatility_forecasts
            WHERE {status_filter} AND target_window_end <= ?;
        """, (as_of_date,))

        candidates = cursor.fetchall()
        if not candidates:
            conn.close()
            return {
                "evaluated_count": 0,
                "window_empty_count": 0,
                "still_pending_count": 0,
                "as_of_date": as_of_date,
            }

        evaluated_updates = []
        empty_updates = []
        now_ts = datetime.now(timezone.utc).isoformat()

        for cand in candidates:
            fid = cand["forecast_id"]
            mkt = cand["market"]
            var = cand["variety"]
            orig_date = cand["origin_date"]
            orig_price = float(cand["origin_price"])
            win_end = cand["target_window_end"]

            # Retrieve realized observations in (t, t+14] calendar days
            cursor.execute("""
                SELECT reported_date, modal_price 
                FROM mandi_observations
                WHERE market = ? AND variety = ? AND reported_date > ? AND reported_date <= ?
                ORDER BY reported_date ASC;
            """, (mkt, var, orig_date, win_end))
            forward_trades = cursor.fetchall()

            if not forward_trades:
                # Zero trading days occurred during the entire 14-day calendar window
                empty_updates.append((now_ts, fid))
                continue

            # Calculate exact realized volatility target matching experimental V3:
            # Forward excursion includes origin price P_t and all realized trades in (t, t+14]
            future_prices = [float(tr["modal_price"]) for tr in forward_trades if tr["modal_price"] > 0]
            if not future_prices:
                empty_updates.append((now_ts, fid))
                continue

            all_forward = [orig_price] + future_prices
            p_max = float(max(all_forward))
            p_min = float(min(all_forward))
            actual_pct_range = float(((p_max - p_min) / (orig_price + 1e-4)) * 100.0)
            actual_class = classify_volatility_range(actual_pct_range)

            evaluated_updates.append((
                len(future_prices),
                round(p_min, 2),
                round(p_max, 2),
                round(actual_pct_range, 4),
                actual_class,
                now_ts,
                fid,
            ))

        # Batch atomic updates
        if evaluated_updates:
            cursor.executemany("""
                UPDATE shadow_volatility_forecasts SET
                    status = 'EVALUATED',
                    matched_trades_count = ?,
                    actual_min_price = ?,
                    actual_max_price = ?,
                    actual_pct_range = ?,
                    actual_class = ?,
                    evaluated_at = ?
                WHERE forecast_id = ?;
            """, evaluated_updates)

        if empty_updates:
            cursor.executemany("""
                UPDATE shadow_volatility_forecasts SET
                    status = 'WINDOW_EMPTY',
                    matched_trades_count = 0,
                    actual_min_price = NULL,
                    actual_max_price = NULL,
                    actual_pct_range = NULL,
                    actual_class = NULL,
                    evaluated_at = ?
                WHERE forecast_id = ?;
            """, empty_updates)

        conn.commit()

        # Count total remaining pending forecasts
        cursor.execute("SELECT COUNT(*) as cnt FROM shadow_volatility_forecasts WHERE status = 'PENDING';")
        still_pending = cursor.fetchone()["cnt"]

        # Audit log entry
        cursor.execute("""
            INSERT INTO shadow_volatility_audit_log (
                timestamp, action, status, cohort,
                forecasts_generated, forecasts_skipped,
                outcomes_evaluated, windows_empty, still_pending,
                duration_seconds, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            now_ts,
            "delayed_outcome_evaluation",
            "SUCCESS",
            "MIXED",
            0,
            0,
            len(evaluated_updates),
            len(empty_updates),
            still_pending,
            0.0,
            json.dumps({"as_of_date": as_of_date}),
        ))
        conn.commit()
        conn.close()

        return {
            "evaluated_count": len(evaluated_updates),
            "window_empty_count": len(empty_updates),
            "still_pending_count": still_pending,
            "as_of_date": as_of_date,
        }


# ============================================================
# 6. COMPREHENSIVE INTERNAL MONITORING DASHBOARD & METRICS
# ============================================================

def compute_multiclass_brier_and_ece(
    y_true_labels: pd.Series,
    predicted_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes multiclass Brier score and Expected Calibration Error (ECE).
    """
    y_true_num = np.array([VOLATILITY_LABEL_MAP[lbl] for lbl in y_true_labels])
    n_classes = len(VOLATILITY_CLASSES)
    n_samples = len(y_true_num)

    if n_samples == 0:
        return {"multiclass_brier_score": np.nan, "macro_ece": np.nan, "class_ece": {}}

    y_one_hot = np.zeros((n_samples, n_classes))
    for c in range(n_classes):
        y_one_hot[:, c] = (y_true_num == c).astype(float)

    brier_score = float(np.mean(np.sum((predicted_proba - y_one_hot) ** 2, axis=1)))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    class_ece = {}

    for c_idx, c_name in enumerate(VOLATILITY_CLASSES):
        p_c = predicted_proba[:, c_idx]
        y_c = y_one_hot[:, c_idx]
        bin_indices = np.digitize(p_c, bin_edges[1:-1])

        ece_c = 0.0
        for b in range(n_bins):
            in_bin = (bin_indices == b)
            bin_count = int(np.sum(in_bin))
            if bin_count > 0:
                bin_acc = float(np.mean(y_c[in_bin]))
                bin_conf = float(np.mean(p_c[in_bin]))
                ece_c += (bin_count / n_samples) * abs(bin_acc - bin_conf)

        class_ece[c_name] = round(float(ece_c), 4)

    macro_ece = round(float(np.mean(list(class_ece.values()))), 4)
    return {
        "multiclass_brier_score": round(brier_score, 4),
        "macro_ece": macro_ece,
        "class_ece": class_ece,
    }


def compute_volatility_shadow_metrics(
    db_path: Optional[Union[str, Path]] = None,
    cohort: Optional[str] = "PROSPECTIVE",
    min_mandi_obs: int = 5,
    non_inferiority_margin: float = 0.01,
) -> Dict[str, Any]:
    """
    Computes rigorous internal monitoring metrics comparing V3 vs Calendar Persistence.
    
    Includes:
    - Counts & class distribution
    - Balanced Accuracy, Macro F1, HIGH recall/precision
    - Multiclass Brier score & Macro/class ECE
    - Per-mandi disaggregated performance
    - Clustered statistical test & clustered bootstrap confidence interval
    - Promotion Governance decision check
    """
    conn = get_volatility_db_connection(db_path)
    query = "SELECT * FROM shadow_volatility_forecasts"
    params = []
    if cohort and cohort.upper() != "ALL":
        query += " WHERE cohort = ?"
        params.append(cohort.upper())

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return {
            "status": "empty",
            "cohort": cohort,
            "message": "No shadow volatility forecasts found in database.",
            "counts": {"total": 0, "evaluated": 0, "pending": 0, "window_empty": 0, "mandis": 0},
        }

    total_count = len(df)
    eval_count = int((df["status"] == "EVALUATED").sum())
    pending_count = int((df["status"] == "PENDING").sum())
    empty_count = int((df["status"] == "WINDOW_EMPTY").sum())
    participating_mandis = int(df["market"].nunique())

    counts_summary = {
        "total_forecasts": total_count,
        "evaluated_forecasts": eval_count,
        "pending_forecasts": pending_count,
        "window_empty_forecasts": empty_count,
        "participating_mandis": participating_mandis,
    }

    if eval_count == 0:
        return {
            "status": "pending_outcomes",
            "cohort": cohort,
            "message": "Forecasts are currently awaiting outcome dates or arrival observations.",
            "counts": counts_summary,
        }

    eval_df = df[df["status"] == "EVALUATED"].copy()
    y_true = eval_df["actual_class"]
    y_v3 = eval_df["v3_pred_class"]
    y_base = eval_df["persistence_pred_class"]

    # Class distributions
    dist_true = y_true.value_counts(normalize=True).to_dict()
    dist_v3 = y_v3.value_counts(normalize=True).to_dict()
    dist_base = y_base.value_counts(normalize=True).to_dict()

    # Classification metrics
    acc_v3 = float(accuracy_score(y_true, y_v3))
    acc_base = float(accuracy_score(y_true, y_base))

    bacc_v3 = float(balanced_accuracy_score(y_true, y_v3))
    bacc_base = float(balanced_accuracy_score(y_true, y_base))

    mf1_v3 = float(f1_score(y_true, y_v3, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
    mf1_base = float(f1_score(y_true, y_base, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))

    prec_v3, rec_v3, f1_v3, _ = precision_recall_fscore_support(
        y_true, y_v3, labels=VOLATILITY_CLASSES, zero_division=0
    )
    prec_base, rec_base, f1_base, _ = precision_recall_fscore_support(
        y_true, y_base, labels=VOLATILITY_CLASSES, zero_division=0
    )

    per_class_metrics = {}
    for i, c in enumerate(VOLATILITY_CLASSES):
        per_class_metrics[c] = {
            "v3": {
                "precision": round(float(prec_v3[i]), 4),
                "recall": round(float(rec_v3[i]), 4),
                "f1": round(float(f1_v3[i]), 4),
            },
            "persistence": {
                "precision": round(float(prec_base[i]), 4),
                "recall": round(float(rec_base[i]), 4),
                "f1": round(float(f1_base[i]), 4),
            },
            "delta_f1": round(float(f1_v3[i] - f1_base[i]), 4),
        }

    # Confusion matrix & Safety diagnostics
    cm_v3 = confusion_matrix(y_true, y_v3, labels=VOLATILITY_CLASSES)
    cm_base = confusion_matrix(y_true, y_base, labels=VOLATILITY_CLASSES)

    high_idx = VOLATILITY_LABEL_MAP["HIGH"]
    low_idx = VOLATILITY_LABEL_MAP["LOW"]

    h_to_l_v3 = int(cm_v3[high_idx, low_idx])
    h_to_l_base = int(cm_base[high_idx, low_idx])
    l_to_h_v3 = int(cm_v3[low_idx, high_idx])
    l_to_h_base = int(cm_base[low_idx, high_idx])

    safety_metrics = {
        "v3_high_recall": round(float(rec_v3[high_idx]), 4),
        "base_high_recall": round(float(rec_base[high_idx]), 4),
        "v3_high_precision": round(float(prec_v3[high_idx]), 4),
        "base_high_precision": round(float(prec_base[high_idx]), 4),
        "v3_high_f1": round(float(f1_v3[high_idx]), 4),
        "base_high_f1": round(float(f1_base[high_idx]), 4),
        "high_to_low_confusion": {
            "v3": h_to_l_v3,
            "persistence": h_to_l_base,
            "reduction_by_v3": h_to_l_base - h_to_l_v3,
        },
        "low_to_high_confusion": {
            "v3": l_to_h_v3,
            "persistence": l_to_h_base,
        },
    }

    # Probability calibration metrics
    cal_prob_cols = ["v3_calibrated_prob_low", "v3_calibrated_prob_medium", "v3_calibrated_prob_high"]
    raw_prob_cols = ["v3_prob_low", "v3_prob_medium", "v3_prob_high"]
    cal_probs = eval_df[cal_prob_cols].to_numpy()
    raw_probs = eval_df[raw_prob_cols].to_numpy()

    cal_eval = compute_multiclass_brier_and_ece(y_true, cal_probs)
    raw_eval = compute_multiclass_brier_and_ece(y_true, raw_probs)

    calibration_metrics = {
        "calibrated_multiclass_brier": cal_eval["multiclass_brier_score"],
        "uncalibrated_multiclass_brier": raw_eval["multiclass_brier_score"],
        "calibrated_macro_ece": cal_eval["macro_ece"],
        "uncalibrated_macro_ece": raw_eval["macro_ece"],
        "class_ece": cal_eval["class_ece"],
    }

    # Market-Level Disaggregation
    mandi_records = []
    for mkt, m_group in eval_df.groupby("market"):
        if len(m_group) < min_mandi_obs:
            continue
        yt_m = m_group["actual_class"]
        yv_m = m_group["v3_pred_class"]
        yb_m = m_group["persistence_pred_class"]

        f1_v_m = float(f1_score(yt_m, yv_m, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
        f1_b_m = float(f1_score(yt_m, yb_m, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))

        mandi_records.append({
            "market": mkt,
            "samples": len(m_group),
            "v3_macro_f1": round(f1_v_m, 4),
            "base_macro_f1": round(f1_b_m, 4),
            "delta_f1": round(f1_v_m - f1_b_m, 4),
            "v3_win": int(f1_v_m > f1_b_m),
            "base_win": int(f1_b_m > f1_v_m),
            "tie": int(f1_v_m == f1_b_m),
        })

    mdf = pd.DataFrame(mandi_records)
    clustered_stats = {}
    market_performance = {}

    if not mdf.empty:
        n_mandis_scored = len(mdf)
        v3_wins = int(mdf["v3_win"].sum())
        base_wins = int(mdf["base_win"].sum())
        ties = int(mdf["tie"].sum())
        v3_win_rate = round((v3_wins / n_mandis_scored) * 100.0, 1)

        diffs = mdf["delta_f1"].to_numpy()
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if n_mandis_scored > 1 else 0.0
        se_diff = float(std_diff / np.sqrt(n_mandis_scored)) if n_mandis_scored > 1 else 0.0

        if se_diff > 1e-9 and n_mandis_scored > 1:
            t_stat = mean_diff / se_diff
            p_val = float(2.0 * (1.0 - student_t.cdf(abs(t_stat), df=n_mandis_scored - 1)))
            ci_t = [
                round(mean_diff - student_t.ppf(0.975, df=n_mandis_scored - 1) * se_diff, 4),
                round(mean_diff + student_t.ppf(0.975, df=n_mandis_scored - 1) * se_diff, 4),
            ]
        else:
            t_stat = 0.0
            p_val = 1.0
            ci_t = [round(mean_diff, 4), round(mean_diff, 4)]

        # Clustered Bootstrap (B=2000)
        np.random.seed(42)
        b_means = []
        for _ in range(2000):
            resamp = np.random.choice(diffs, size=n_mandis_scored, replace=True)
            b_means.append(float(np.mean(resamp)))
        ci_boot = [round(float(np.percentile(b_means, 2.5)), 4), round(float(np.percentile(b_means, 97.5)), 4)]

        clustered_stats = {
            "mandis_evaluated": n_mandis_scored,
            "mean_per_mandi_delta": round(mean_diff, 4),
            "cluster_robust_se": round(se_diff, 4),
            "paired_cluster_t_stat": round(t_stat, 4),
            "cluster_t_test_p_value": round(p_val, 4),
            "parametric_95_ci": ci_t,
            "clustered_bootstrap_95_ci": ci_boot,
            "v3_win_rate_pct": v3_win_rate,
            "v3_wins": v3_wins,
            "persistence_wins": base_wins,
            "ties": ties,
        }

        # Volume Tier Analysis
        q25 = mdf["samples"].quantile(0.25)
        q75 = mdf["samples"].quantile(0.75)
        tiers = {
            "small": mdf[mdf["samples"] <= q25],
            "medium": mdf[(mdf["samples"] > q25) & (mdf["samples"] <= q75)],
            "large": mdf[mdf["samples"] > q75],
        }
        tier_summary = {}
        for t_name, t_sub in tiers.items():
            if len(t_sub) > 0:
                tier_summary[t_name] = {
                    "mandis_count": len(t_sub),
                    "mean_samples": round(float(t_sub["samples"].mean()), 1),
                    "v3_win_rate_pct": round(float(t_sub["v3_win"].mean() * 100.0), 1),
                    "mean_delta_f1": round(float(t_sub["delta_f1"].mean()), 4),
                }

        market_performance = {
            "mandi_records": mdf.to_dict(orient="records"),
            "clustered_statistics": clustered_stats,
            "volume_tiers": tier_summary,
        }

    # Promotion Governance Assessment
    min_date = pd.to_datetime(eval_df["origin_date"]).min()
    max_date = pd.to_datetime(eval_df["target_window_end"]).max()
    monitoring_days = int((max_date - min_date).days) if pd.notna(min_date) and pd.notna(max_date) else 0

    ci_lower = clustered_stats.get("clustered_bootstrap_95_ci", [-999.0, 999.0])[0]
    mean_delta = clustered_stats.get("mean_per_mandi_delta", -999.0)

    # Correct Statistical Test: Predefined margin non-inferiority
    non_inferiority_proven = (ci_lower > -non_inferiority_margin) and (mean_delta >= 0.0)

    governance = {
        "predefined_non_inferiority_margin": non_inferiority_margin,
        "criteria": {
            "monitoring_duration_days": {
                "observed": monitoring_days,
                "required": 90,
                "passed": monitoring_days >= 90,
            },
            "completed_prospective_forecasts": {
                "observed": eval_count,
                "required": 2000,
                "passed": eval_count >= 2000,
            },
            "mandis_coverage": {
                "observed": participating_mandis,
                "required": 35,
                "passed": participating_mandis >= 35,
            },
            "macro_ece": {
                "observed": calibration_metrics["calibrated_macro_ece"],
                "required": 0.05,
                "passed": calibration_metrics["calibrated_macro_ece"] <= 0.05,
            },
            "mandi_win_rate": {
                "observed": clustered_stats.get("v3_win_rate_pct", 0.0),
                "required": 50.0,
                "passed": clustered_stats.get("v3_win_rate_pct", 0.0) >= 50.0,
            },
            "non_inferiority_test": {
                "ci_lower_bound": ci_lower,
                "margin_threshold": -non_inferiority_margin,
                "mean_delta": mean_delta,
                "p_value_alone_rejected": True,
                "passed": bool(non_inferiority_proven),
            },
        },
        "all_criteria_met": False,
        "verdict": "CONTINUE_SHADOW_MONITORING",
        "action": "Maintain Calendar Persistence in production. V3 remains strictly in shadow observation.",
    }

    all_passed = all(c["passed"] for c in governance["criteria"].values())
    governance["all_criteria_met"] = all_passed
    if all_passed:
        governance["verdict"] = "READY_FOR_GOVERNANCE_REVIEW"
        governance["action"] = "Submit formal promotion proposal to ML Governance Board."

    return {
        "status": "evaluated",
        "cohort": cohort,
        "counts": counts_summary,
        "pooled_performance": {
            "v3": {
                "accuracy": round(acc_v3, 4),
                "balanced_accuracy": round(bacc_v3, 4),
                "macro_f1": round(mf1_v3, 4),
            },
            "persistence": {
                "accuracy": round(acc_base, 4),
                "balanced_accuracy": round(bacc_base, 4),
                "macro_f1": round(mf1_base, 4),
            },
            "delta_accuracy": round(acc_v3 - acc_base, 4),
            "delta_macro_f1": round(mf1_v3 - mf1_base, 4),
            "delta_balanced_accuracy": round(bacc_v3 - bacc_base, 4),
        },
        "per_class_metrics": per_class_metrics,
        "safety_metrics": safety_metrics,
        "calibration_metrics": calibration_metrics,
        "class_distributions": {
            "actual": {k: round(v, 4) for k, v in dist_true.items()},
            "v3_predicted": {k: round(v, 4) for k, v in dist_v3.items()},
            "persistence_predicted": {k: round(v, 4) for k, v in dist_base.items()},
        },
        "market_level": market_performance,
        "promotion_governance": governance,
    }


def generate_volatility_shadow_report(
    db_path: Optional[Union[str, Path]] = None,
    cohort: Optional[str] = "PROSPECTIVE",
    output_path: Optional[Union[str, Path]] = None,
    non_inferiority_margin: float = 0.01,
) -> str:
    """
    Renders a formatted internal monitoring report as markdown.
    """
    metrics = compute_volatility_shadow_metrics(
        db_path=db_path,
        cohort=cohort,
        non_inferiority_margin=non_inferiority_margin,
    )

    lines = []
    lines.append("# Internal Monitoring Report: Member 2 Volatility V3 Shadow Pipeline")
    lines.append(f"**Cohort:** `{cohort or 'ALL'}` | **Report Timestamp:** `{datetime.now(timezone.utc).isoformat()}`\n")

    if metrics.get("status") == "empty":
        lines.append("> [!NOTE] Database is empty. No shadow volatility records found.\n")
        lines.append("## Promotion Governance Gate Assessment")
        lines.append("- **Evaluation Status:** No prospective forecasts recorded yet.")
        lines.append("**Final Governance Verdict:** `CONTINUE_SHADOW_MONITORING`")
        lines.append("**Action:** Maintain Calendar Persistence in production. Insufficient prospective observation history.\n")
        content = "\n".join(lines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(content, encoding="utf-8")
        return content

    counts = metrics.get("counts", {})
    lines.append("## 1. Operational Counts")
    lines.append(f"- **Total Forecasts Ingested:** {counts.get('total_forecasts', 0):,}")
    lines.append(f"- **Evaluated Forecasts:** {counts.get('evaluated_forecasts', 0):,}")
    lines.append(f"- **Pending Forecasts (Gated in 14-day window):** {counts.get('pending_forecasts', 0):,}")
    lines.append(f"- **Empty Target Windows (Zero forward trading):** {counts.get('window_empty_forecasts', 0):,}")
    lines.append(f"- **Participating Mandis:** {counts.get('participating_mandis', 0)}\n")

    if metrics.get("status") == "pending_outcomes":
        lines.append("> [!NOTE] All forecasts are currently awaiting completion of their 14-calendar-day window.\n")
        lines.append("## Promotion Governance Gate Assessment")
        lines.append("- **Evaluation Status:** Forecasts are currently pending 14-calendar-day outcome realization.")
        lines.append("**Final Governance Verdict:** `CONTINUE_SHADOW_MONITORING`")
        lines.append("**Action:** Maintain Calendar Persistence in production. Gated observation window has not elapsed.\n")
        content = "\n".join(lines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(content, encoding="utf-8")
        return content

    pooled = metrics.get("pooled_performance", {})
    v3_p = pooled.get("v3", {})
    base_p = pooled.get("persistence", {})
    lines.append("## 2. Pooled Comparative Performance (14-Day Calendar Horizon)")
    lines.append("| Metric | V3 14D XGBoost | Calendar Persistence | Delta (V3 - Base) |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **Accuracy** | {v3_p.get('accuracy', 0.0):.4f} | {base_p.get('accuracy', 0.0):.4f} | {pooled.get('delta_accuracy', 0.0):+.4f} |")
    lines.append(f"| **Balanced Accuracy** | {v3_p.get('balanced_accuracy', 0.0):.4f} | {base_p.get('balanced_accuracy', 0.0):.4f} | {pooled.get('delta_balanced_accuracy', 0.0):+.4f} |")
    lines.append(f"| **Macro F1** | {v3_p.get('macro_f1', 0.0):.4f} | {base_p.get('macro_f1', 0.0):.4f} | {pooled.get('delta_macro_f1', 0.0):+.4f} |\n")

    safety = metrics.get("safety_metrics", {})
    lines.append("## 3. High-Volatility Advisory Safety")
    lines.append(f"- **V3 HIGH Recall:** {safety.get('v3_high_recall', 0.0) * 100:.2f}% (Persistence: {safety.get('base_high_recall', 0.0) * 100:.2f}%)")
    lines.append(f"- **V3 HIGH Precision:** {safety.get('v3_high_precision', 0.0) * 100:.2f}% (Persistence: {safety.get('base_high_precision', 0.0) * 100:.2f}%)")
    lines.append(f"- **V3 HIGH F1:** {safety.get('v3_high_f1', 0.0):.4f} (Persistence: {safety.get('base_high_f1', 0.0):.4f})")
    h2l = safety.get("high_to_low_confusion", {})
    lines.append(f"- **Critical HIGH-to-LOW Errors:** V3: {h2l.get('v3', 0)} vs Persistence: {h2l.get('persistence', 0)} (Reduction: {h2l.get('reduction_by_v3', 0)})\n")

    cal = metrics.get("calibration_metrics", {})
    lines.append("## 4. Probability Calibration")
    lines.append(f"- **Multiclass Brier Score:** Calibrated: {cal.get('calibrated_multiclass_brier', 0.0):.4f} | Raw: {cal.get('uncalibrated_multiclass_brier', 0.0):.4f}")
    lines.append(f"- **Macro ECE:** Calibrated: {cal.get('calibrated_macro_ece', 0.0) * 100:.2f}% | Raw: {cal.get('uncalibrated_macro_ece', 0.0) * 100:.2f}%")
    lines.append(f"- **Class-Wise ECE:** LOW: {cal.get('class_ece', {}).get('LOW', 0.0)*100:.2f}% | MEDIUM: {cal.get('class_ece', {}).get('MEDIUM', 0.0)*100:.2f}% | HIGH: {cal.get('class_ece', {}).get('HIGH', 0.0)*100:.2f}%\n")

    mkt = metrics.get("market_level", {})
    c_stats = mkt.get("clustered_statistics", {})
    lines.append("## 5. Market-Level Cluster Significance")
    if c_stats and c_stats.get("mandis_evaluated", 0) > 0:
        lines.append(f"- **Mandis Evaluated (>= 5 obs):** {c_stats.get('mandis_evaluated', 0)}")
        lines.append(f"- **ML Wins vs Persistence Wins:** {c_stats.get('v3_wins', 0)} vs {c_stats.get('persistence_wins', 0)} (Ties: {c_stats.get('ties', 0)})")
        lines.append(f"- **ML Win Rate:** {c_stats.get('v3_win_rate_pct', 0.0):.1f}%")
        lines.append(f"- **Mean Per-Mandi Delta:** {c_stats.get('mean_per_mandi_delta', 0.0):+.4f} (SE: {c_stats.get('cluster_robust_se', 0.0):.4f})")
        lines.append(f"- **Paired Cluster t-stat:** {c_stats.get('paired_cluster_t_stat', 0.0):.4f} (p = {c_stats.get('cluster_t_test_p_value', 1.0):.4f})")
        lines.append(f"- **Clustered Bootstrap 95% CI:** `[{c_stats.get('clustered_bootstrap_95_ci', [0,0])[0]:.4f}, {c_stats.get('clustered_bootstrap_95_ci', [0,0])[1]:.4f}]`\n")
    else:
        lines.append("> [!NOTE] Mandi-level cluster analysis requires at least 5 evaluated observations per market. Prospective observation windows are currently accumulating sample depth across individual reporting mandis.\n")

    gov = metrics.get("promotion_governance", {})
    lines.append("## 6. Promotion Governance Gate Assessment")
    crit = gov.get("criteria", {})
    lines.append("| Promotion Gate | Target Threshold | Observed Value | Status |")
    lines.append("|---|---|---|---|")
    d_g = crit.get("monitoring_duration_days", {})
    lines.append(f"| Minimum Duration | >= 90 calendar days | {d_g.get('observed', 0)} days | {'PASSED' if d_g.get('passed') else 'PENDING'} |")
    c_g = crit.get("completed_prospective_forecasts", {})
    lines.append(f"| Completed Forecasts | >= 2,000 observations | {c_g.get('observed', 0):,} | {'PASSED' if c_g.get('passed') else 'PENDING'} |")
    m_g = crit.get("mandis_coverage", {})
    lines.append(f"| Mandi Coverage | >= 35 reporting mandis | {m_g.get('observed', 0)} | {'PASSED' if m_g.get('passed') else 'PENDING'} |")
    e_g = crit.get("macro_ece", {})
    lines.append(f"| Calibration Error | Macro ECE <= 5.0% | {e_g.get('observed', 0.0)*100:.2f}% | {'PASSED' if e_g.get('passed') else 'FAIL'} |")
    w_g = crit.get("mandi_win_rate", {})
    lines.append(f"| Mandi Parity | Win rate >= 50.0% | {w_g.get('observed', 0.0):.1f}% | {'PASSED' if w_g.get('passed') else 'PENDING'} |")
    ni_g = crit.get("non_inferiority_test", {})
    ci_disp = f"CI Lower = {ni_g.get('ci_lower_bound', -1.0):.4f}" if ni_g.get("ci_lower_bound", -999.0) > -900 else "PENDING (Sample depth accumulating)"
    margin_disp = f"-{abs(non_inferiority_margin):.4f}"
    lines.append(f"| Non-Inferiority Test | CI Lower > {margin_disp} & mean delta >= 0 | {ci_disp} | {'PASSED' if ni_g.get('passed') else 'PENDING'} |\n")

    lines.append(f"**Final Governance Verdict:** `{gov.get('verdict', 'CONTINUE_SHADOW_MONITORING')}`")
    lines.append(f"**Action:** {gov.get('action', 'Maintain Calendar Persistence.')}\n")

    content = "\n".join(lines)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")
    return content
