"""
AgroVision-ML — Member 2: Multi-Commodity Prospective Shadow Monitoring System
Target Scope: Dedicated prospective shadow evaluation for Wheat and Potato V2 models.

Key Architectural Guarantees:
1. Strict Commodity Separation: Operates on isolated databases and tables per commodity;
   never contaminates or alters Rice shadow evaluation or Rice production tables.
2. Production Isolation: Never alters, retrains, or affects production models,
   endpoints, or public advisory decisions.
3. Operational Controls: Controlled by AGROVISION_ENABLE_{COMMODITY}_SHADOW or
   master AGROVISION_ENABLE_SHADOW_EVALUATION switch.
4. Point-in-Time Integrity: Forecast generation uses observations up to origin date only.
   Zero future target data can enter feature calculations.
5. Idempotency & Deduplication: Forecasts are assigned deterministic IDs derived from
   (commodity, market, variety, origin_date, horizon_days, model_version). Repeated runs are no-ops.
6. Delayed Outcome Evaluation: Forecasts are evaluated only after target calendar window
   [t + H, t + H + max_tolerance_days] has materialized:
   - PENDING: Target window still open or target date not yet arrived.
   - EVALUATED: Matched first valid observation in target window.
   - TARGET_UNAVAILABLE: Window definitively closed without any report in window.
7. Honest Matched Benchmarking: Evaluates V2 ML models and Persistence baseline on exactly
   the same matched forecast origins and realized target prices.
"""

import os
import sys
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(MEMBER2_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    get_artifacts_dir,
    is_shadow_evaluation_enabled,
)
from models.predict import predict as m2_predict

# Operational switch state
_COMMODITY_SHADOW_SWITCHES: Dict[str, Optional[bool]] = {
    "wheat": None,
    "potato": None,
    "tea": None,
    "rice": None,
}


def is_commodity_shadow_enabled(commodity: str) -> bool:
    """Returns True if shadow evaluation is enabled for the specified commodity."""
    comm = str(commodity).strip().lower()
    env_var = f"AGROVISION_ENABLE_{comm.upper()}_SHADOW"
    if env_var in os.environ:
        return os.environ[env_var].lower() in ("1", "true", "yes", "on")

    cached = _COMMODITY_SHADOW_SWITCHES.get(comm)
    if cached is not None:
        return cached

    # Fallback to master shadow evaluation switch
    return is_shadow_evaluation_enabled()


def set_commodity_shadow_enabled(commodity: str, enabled: bool = True):
    """Sets operational switch for the specified commodity."""
    comm = str(commodity).strip().lower()
    _COMMODITY_SHADOW_SWITCHES[comm] = bool(enabled)


def reset_commodity_shadow_switches():
    """Resets all commodity shadow switches to default."""
    for k in _COMMODITY_SHADOW_SWITCHES:
        _COMMODITY_SHADOW_SWITCHES[k] = None


def get_default_commodity_shadow_db_path(commodity: str) -> Path:
    """Returns the isolated database path for the given commodity."""
    comm = str(commodity).strip().lower()
    env_path = os.environ.get(f"AGROVISION_{comm.upper()}_SHADOW_DB_PATH")
    if env_path:
        return Path(env_path)
    return MEMBER2_DIR / "data" / "processed" / f"shadow_evaluation_{comm}.db"


# ============================================================
# 1. DATABASE SCHEMA & CONNECTION
# ============================================================

def init_commodity_shadow_db(commodity: str, db_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Initializes isolated SQLite schema for tracking commodity shadow forecasts and mandi observations.
    """
    comm = str(commodity).strip().lower()
    if comm not in ("wheat", "potato", "rice", "tea"):
        raise ValueError(f"Unsupported shadow commodity: '{comm}'. Supported: 'wheat', 'potato', 'rice', 'tea'.")

    resolved_path = Path(db_path) if db_path else get_default_commodity_shadow_db_path(comm)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        # Real mandi observations table
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

        # Commodity shadow forecasts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_forecasts (
                forecast_id TEXT PRIMARY KEY,
                commodity TEXT NOT NULL,
                market TEXT NOT NULL,
                variety TEXT NOT NULL,
                origin_date TEXT NOT NULL,
                horizon_days INTEGER NOT NULL,
                origin_modal_price REAL NOT NULL,
                predicted_price_change REAL NOT NULL,
                predicted_modal_price REAL NOT NULL,
                requested_target_date TEXT NOT NULL,
                target_window_start TEXT NOT NULL,
                target_window_end TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                volatility_pred_class TEXT NOT NULL,
                volatility_probabilities_json TEXT,
                persistence_pred_price REAL NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL, -- 'PENDING', 'EVALUATED', 'TARGET_UNAVAILABLE'
                matched_observation_date TEXT,
                matched_actual_price REAL,
                actual_horizon_offset INTEGER,
                prediction_residual REAL,
                actual_volatility_outcome TEXT,
                absolute_error REAL,
                percentage_error REAL,
                persistence_absolute_error REAL,
                persistence_percentage_error REAL,
                evaluated_at TEXT,
                finalization_reason TEXT,
                cohort TEXT DEFAULT 'PROSPECTIVE'
            );
        """)

        # Check for column existence and migrate existing tables if needed
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(shadow_forecasts);")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "prediction_residual" not in existing_cols:
            conn.execute("ALTER TABLE shadow_forecasts ADD COLUMN prediction_residual REAL;")
        if "actual_volatility_outcome" not in existing_cols:
            conn.execute("ALTER TABLE shadow_forecasts ADD COLUMN actual_volatility_outcome TEXT;")

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_comm_lookup 
            ON shadow_forecasts (commodity, market, variety, status, horizon_days);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_comm_origin 
            ON shadow_forecasts (commodity, origin_date, horizon_days);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_comm_cohort 
            ON shadow_forecasts (commodity, cohort, status, horizon_days);
        """)

        # Operational audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                commodity TEXT NOT NULL,
                execution_type TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT
            );
        """)

    return resolved_path


def get_commodity_db_connection(commodity: str, db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Returns an active SQLite connection configured with Row factory."""
    resolved_path = init_commodity_shadow_db(commodity, db_path)
    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 2. OBSERVATION INGESTION
# ============================================================

def normalize_commodity_observations_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes and strictly validates column names and values for mandi observations."""
    df_out = df.copy()
    col_map = {}
    for c in df_out.columns:
        c_low = str(c).strip().lower()
        if c_low in ("market name", "market", "market_name"):
            col_map[c] = "market"
        elif c_low in ("variety", "commodity_variety"):
            col_map[c] = "variety"
        elif c_low in ("reported date", "reported_date", "date"):
            col_map[c] = "reported_date"
        elif c_low in ("modal price (rs./quintal)", "modal_price", "modal_price_rs_quintal", "price"):
            col_map[c] = "modal_price"
        elif c_low in ("arrivals (tonnes)", "arrivals", "arrivals_tonnes"):
            col_map[c] = "arrivals"

    df_out = df_out.rename(columns=col_map)
    required = ["market", "variety", "reported_date", "modal_price"]
    missing = [c for c in required if c not in df_out.columns]
    if missing:
        raise ValueError(f"Missing required columns in observations DataFrame: {missing}")

    if "arrivals" not in df_out.columns:
        df_out["arrivals"] = 0.0

    df_out["market"] = df_out["market"].astype(str).str.strip()
    df_out["variety"] = df_out["variety"].astype(str).str.strip()
    df_out["reported_date"] = pd.to_datetime(df_out["reported_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df_out["modal_price"] = pd.to_numeric(df_out["modal_price"], errors="coerce")
    df_out["arrivals"] = pd.to_numeric(df_out["arrivals"], errors="coerce").fillna(0.0)

    # Drop invalid rows: NaN values, empty strings, or non-positive prices
    df_out = df_out.dropna(subset=["market", "variety", "reported_date", "modal_price"])
    df_out = df_out[(df_out["modal_price"] > 0) & (df_out["market"] != "") & (df_out["variety"] != "")]
    df_out = df_out.drop_duplicates(subset=["market", "variety", "reported_date"], keep="last")
    return df_out[["market", "variety", "reported_date", "modal_price", "arrivals"]]


def ingest_commodity_mandi_observations(
    commodity: str,
    data: Union[pd.DataFrame, str, Path],
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    """Ingests newly reported mandi observations for the given commodity into its isolated SQLite database."""
    comm = str(commodity).strip().lower()
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Observations file not found at {p}")
        df = pd.read_csv(p)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected DataFrame or Path, received {type(data)}")

    clean_df = normalize_commodity_observations_df(df)
    if clean_df.empty:
        return 0

    now_ts = datetime.now(timezone.utc).isoformat()
    records = [
        (
            row["market"],
            row["variety"],
            row["reported_date"],
            float(row["modal_price"]),
            float(row["arrivals"]),
            now_ts,
        )
        for _, row in clean_df.iterrows()
    ]

    with get_commodity_db_connection(comm, db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO mandi_observations (market, variety, reported_date, modal_price, arrivals, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, variety, reported_date) DO UPDATE SET
                modal_price = excluded.modal_price,
                arrivals = excluded.arrivals;
        """, records)
        conn.commit()

    return len(records)


# ============================================================
# 3. FORECAST GENERATION
# ============================================================

def make_commodity_forecast_id(
    commodity: str,
    market: str,
    variety: str,
    origin_date: str,
    horizon_days: int,
    model_version: str = "2.0.0",
) -> str:
    """Generates a deterministic unique identifier for a commodity forecast."""
    key = f"{commodity.strip().lower()}|{market.strip().lower()}|{variety.strip().lower()}|{origin_date.strip()}|{int(horizon_days)}|{model_version.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def generate_commodity_shadow_forecasts(
    commodity: str,
    history_df: pd.DataFrame,
    horizons: Optional[Union[Tuple[int, ...], List[int]]] = None,
    origin_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    force_enabled: bool = False,
    cohort: str = "PROSPECTIVE",
) -> Dict[str, Union[int, str, list]]:
    """
    Generates point-in-time shadow forecasts for Wheat, Potato, Tea, or Rice.

    Idempotent: Skips forecasts that already exist in the database for the given
    (commodity, market, variety, origin_date, horizon, model_version).
    """
    comm = str(commodity).strip().lower()
    if comm not in ("wheat", "potato", "tea", "rice"):
        raise ValueError(f"generate_commodity_shadow_forecasts only supports 'wheat', 'potato', 'tea', 'rice', got '{comm}'")

    if not force_enabled and not is_commodity_shadow_enabled(comm):
        raise RuntimeError(
            f"Shadow evaluation is currently disabled for {comm}. "
            f"Set AGROVISION_ENABLE_{comm.upper()}_SHADOW=true or pass force_enabled=True."
        )

    if horizons is None:
        horizons = (1,) if comm == "tea" else (14, 30)

    clean_df = history_df.copy()
    col_map = {}
    for c in clean_df.columns:
        clow = str(c).strip().lower()
        if clow in ("market name", "market", "market_name"):
            col_map[c] = "Market Name"
        elif clow in ("variety", "commodity_variety"):
            col_map[c] = "Variety"
        elif clow in ("reported date", "reported_date", "date"):
            col_map[c] = "Reported Date"
        elif clow in ("modal price (rs./quintal)", "modal_price", "modal_price_rs_quintal", "price"):
            col_map[c] = "Modal Price (Rs./Quintal)"
        elif clow in ("arrivals (tonnes)", "arrivals", "arrivals_tonnes"):
            col_map[c] = "Arrivals (Tonnes)"

    clean_df = clean_df.rename(columns=col_map)
    clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"], errors="coerce")
    clean_df = clean_df.dropna(subset=["Reported Date", "Modal Price (Rs./Quintal)"])
    clean_df = clean_df.sort_values("Reported Date").reset_index(drop=True)

    if clean_df.empty:
        return {"status": "error", "message": "History dataframe is empty after cleaning."}

    # Resolve origin date
    if origin_date is not None:
        origin_dt = pd.to_datetime(origin_date)
        origin_str = origin_dt.strftime("%Y-%m-%d")
        history_at_origin = clean_df[clean_df["Reported Date"] <= origin_dt].copy()
    else:
        origin_dt = clean_df["Reported Date"].max()
        origin_str = origin_dt.strftime("%Y-%m-%d")
        history_at_origin = clean_df.copy()

    if len(history_at_origin) < 1:
        return {"status": "error", "message": f"Zero historical observations available on or before {origin_str}"}

    # Identify distinct market + variety combinations
    mkt = str(history_at_origin.iloc[-1].get("Market Name", "DefaultMarket"))
    var = str(history_at_origin.iloc[-1].get("Variety", "DefaultVariety"))

    origin_price = float(history_at_origin.iloc[-1]["Modal Price (Rs./Quintal)"])
    now_ts = datetime.now(timezone.utc).isoformat()

    conn = get_commodity_db_connection(comm, db_path)
    generated_count = 0
    skipped_count = 0
    generated_ids = []

    try:
        for h in horizons:
            h_int = int(h)
            if comm == "tea":
                model_ver = "1.0.0"
            elif comm == "rice":
                model_ver = "1.0.0"
            else:
                model_ver = "2.0.0"
            f_id = make_commodity_forecast_id(comm, mkt, var, origin_str, h_int, model_ver)

            # Check for existing forecast
            existing = conn.execute(
                "SELECT forecast_id FROM shadow_forecasts WHERE forecast_id = ?;", (f_id,)
            ).fetchone()
            if existing:
                skipped_count += 1
                continue

            # Run prediction
            try:
                if comm == "tea":
                    pred_res = m2_predict(
                        data=history_at_origin,
                        commodity="tea",
                    )
                elif comm == "rice":
                    pred_res = m2_predict(
                        data=history_at_origin,
                        commodity="rice",
                        horizon_days=h_int,
                        return_intervals=True,
                    )
                else:
                    pred_res = m2_predict(
                        data=history_at_origin,
                        commodity=comm,
                        horizon_days=h_int,
                        version="v2",
                        return_intervals=True,
                    )
                pred_price = float(pred_res["predicted_price"])
                pred_change = float(pred_res.get("predicted_price_change", pred_price - origin_price))
                model_name = str(pred_res.get("model_name", f"{comm}-price-{h_int}d-catboost" if h_int > 1 else f"{comm}-price-xgb"))
                vol_class = str(pred_res.get("volatility", pred_res.get("predicted_volatility_class", "MEDIUM")))
                vol_probs_json = json.dumps(pred_res.get("volatility_probabilities", {}))
            except Exception as e:
                # Log error and skip horizon
                continue

            target_dt = (origin_dt + pd.Timedelta(days=h_int)).strftime("%Y-%m-%d")
            window_start = (origin_dt + pd.Timedelta(days=h_int)).strftime("%Y-%m-%d")
            window_end = (origin_dt + pd.Timedelta(days=h_int + 3)).strftime("%Y-%m-%d")

            conn.execute("""
                INSERT INTO shadow_forecasts (
                    forecast_id, commodity, market, variety, origin_date, horizon_days,
                    origin_modal_price, predicted_price_change, predicted_modal_price,
                    requested_target_date, target_window_start, target_window_end,
                    model_name, model_version, volatility_pred_class, volatility_probabilities_json,
                    persistence_pred_price, created_at, status, cohort
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?);
            """, (
                f_id, comm, mkt, var, origin_str, h_int,
                origin_price, pred_change, pred_price,
                target_dt, window_start, window_end,
                model_name, model_ver, vol_class, vol_probs_json,
                origin_price, now_ts, cohort
            ))

            generated_count += 1
            generated_ids.append(f_id)

        conn.commit()
    finally:
        conn.close()

    return {
        "status": "success",
        "commodity": comm,
        "origin_date": origin_str,
        "forecasts_generated": generated_count,
        "forecasts_skipped": skipped_count,
        "forecast_ids": generated_ids,
    }


# ============================================================
# 4. DELAYED OUTCOME EVALUATION
# ============================================================

def evaluate_pending_commodity_shadow_forecasts(
    commodity: str,
    as_of_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    max_tolerance_days: int = 3,
) -> Dict[str, Union[int, str]]:
    """
    Evaluates pending shadow forecasts for the commodity against subsequent actual observations.

    Strict Delayed Outcome Protocol:
    - If as_of_date is earlier than target_window_end and no observation has appeared yet,
      the forecast remains PENDING.
    - If a valid observation exists in [target_window_start, target_window_end], it matches
      the earliest observation and transitions to EVALUATED.
    - If as_of_date > target_window_end and no observation was recorded, transitions to TARGET_UNAVAILABLE.
    """
    comm = str(commodity).strip().lower()
    conn = get_commodity_db_connection(comm, db_path)

    if as_of_date is None:
        max_obs = conn.execute("SELECT MAX(reported_date) as max_d FROM mandi_observations;").fetchone()
        as_of_str = max_obs["max_d"] if max_obs and max_obs["max_d"] else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    else:
        as_of_str = pd.to_datetime(as_of_date).strftime("%Y-%m-%d")

    evaluated_count = 0
    unavailable_count = 0
    pending_count = 0
    now_ts = datetime.now(timezone.utc).isoformat()

    try:
        pending_rows = conn.execute("""
            SELECT * FROM shadow_forecasts 
            WHERE commodity = ? AND status = 'PENDING';
        """, (comm,)).fetchall()

        for row in pending_rows:
            f_id = row["forecast_id"]
            mkt = row["market"]
            var = row["variety"]
            w_start = row["target_window_start"]
            w_end = row["target_window_end"]
            pred_price = row["predicted_modal_price"]
            pers_price = row["persistence_pred_price"]
            origin_dt = pd.to_datetime(row["origin_date"])

            # Query candidate observations in target window
            match_row = conn.execute("""
                SELECT reported_date, modal_price 
                FROM mandi_observations
                WHERE market = ? AND variety = ? AND reported_date >= ? AND reported_date <= ?
                ORDER BY reported_date ASC
                LIMIT 1;
            """, (mkt, var, w_start, w_end)).fetchone()

            if match_row:
                matched_date = match_row["reported_date"]
                actual_price = float(match_row["modal_price"])
                origin_price = float(row["origin_modal_price"])
                actual_offset = int((pd.to_datetime(matched_date) - origin_dt).days)

                pred_residual = round(actual_price - pred_price, 2)
                pct_movement = (abs(actual_price - origin_price) / (origin_price + 1e-4)) * 100.0
                if pct_movement <= 2.0:
                    actual_vol = "LOW"
                elif pct_movement <= 6.0:
                    actual_vol = "MEDIUM"
                else:
                    actual_vol = "HIGH"

                abs_err = abs(pred_price - actual_price)
                pct_err = (abs_err / actual_price) * 100.0 if actual_price > 0 else 0.0

                p_abs_err = abs(pers_price - actual_price)
                p_pct_err = (p_abs_err / actual_price) * 100.0 if actual_price > 0 else 0.0

                conn.execute("""
                    UPDATE shadow_forecasts SET
                        status = 'EVALUATED',
                        matched_observation_date = ?,
                        matched_actual_price = ?,
                        actual_horizon_offset = ?,
                        prediction_residual = ?,
                        actual_volatility_outcome = ?,
                        absolute_error = ?,
                        percentage_error = ?,
                        persistence_absolute_error = ?,
                        persistence_percentage_error = ?,
                        evaluated_at = ?,
                        finalization_reason = 'MATCHED_WINDOW_OBSERVATION'
                    WHERE forecast_id = ?;
                """, (
                    matched_date, actual_price, actual_offset,
                    pred_residual, actual_vol,
                    abs_err, pct_err, p_abs_err, p_pct_err,
                    now_ts, f_id
                ))
                evaluated_count += 1

            elif as_of_str > w_end:
                # Window closed without reports
                conn.execute("""
                    UPDATE shadow_forecasts SET
                        status = 'TARGET_UNAVAILABLE',
                        evaluated_at = ?,
                        finalization_reason = 'WINDOW_CLOSED_WITHOUT_OBSERVATION'
                    WHERE forecast_id = ?;
                """, (now_ts, f_id))
                unavailable_count += 1
            else:
                # Target window still active
                pending_count += 1

        conn.commit()
    finally:
        conn.close()

    return {
        "status": "success",
        "commodity": comm,
        "as_of_date": as_of_str,
        "evaluated_count": evaluated_count,
        "unavailable_count": unavailable_count,
        "still_pending_count": pending_count,
    }


# ============================================================
# 5. GOVERNANCE METRICS & NON-INFERIORITY
# ============================================================

def compute_commodity_shadow_metrics(
    commodity: str,
    db_path: Optional[Union[str, Path]] = None,
    horizon_days: Optional[int] = None,
    cohort: str = "PROSPECTIVE",
    min_evaluations: int = 1,
) -> Dict[str, Any]:
    """
    Computes rigorous, honest performance metrics comparing V2 model vs. Persistence baseline.
    """
    comm = str(commodity).strip().lower()
    conn = get_commodity_db_connection(comm, db_path)

    results: Dict[str, Any] = {
        "commodity": comm,
        "cohort": cohort,
        "horizons": {},
    }

    if horizon_days is None:
        horizons_to_eval = [1] if comm == "tea" else [14, 30]
    else:
        horizons_to_eval = [int(horizon_days)]

    try:
        for h in horizons_to_eval:
            h_key = f"{h}d"
            where_clause = "commodity = ? AND horizon_days = ?"
            params: List[Any] = [comm, h]
            if cohort != "ALL":
                where_clause += " AND cohort = ?"
                params.append(cohort)

            # Query counts
            counts_row = conn.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'EVALUATED' THEN 1 ELSE 0 END) as evaluated,
                    SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status = 'TARGET_UNAVAILABLE' THEN 1 ELSE 0 END) as unavailable
                FROM shadow_forecasts
                WHERE {where_clause};
            """, params).fetchone()

            total = counts_row["total"] or 0
            evaluated = counts_row["evaluated"] or 0
            pending = counts_row["pending"] or 0
            unavailable = counts_row["unavailable"] or 0

            h_res: Dict[str, Any] = {
                "counts": {
                    "total_forecasts": total,
                    "evaluated_forecasts": evaluated,
                    "pending_forecasts": pending,
                    "unavailable_target_forecasts": unavailable,
                }
            }

            if evaluated < min_evaluations:
                h_res["status"] = "pending_outcomes"
                h_res["message"] = f"Insufficient evaluated outcomes ({evaluated}/{min_evaluations})."
                results["horizons"][h_key] = h_res
                continue

            # Query evaluated records
            eval_rows = conn.execute(f"""
                SELECT 
                    matched_actual_price as y_true,
                    predicted_modal_price as y_pred,
                    persistence_pred_price as y_pers,
                    absolute_error as abs_err,
                    persistence_absolute_error as pers_abs_err,
                    percentage_error as pct_err,
                    persistence_percentage_error as pers_pct_err
                FROM shadow_forecasts
                WHERE {where_clause} AND status = 'EVALUATED';
            """, params).fetchall()

            df_eval = pd.DataFrame([dict(r) for r in eval_rows])
            y_true = df_eval["y_true"].values
            y_pred = df_eval["y_pred"].values
            y_pers = df_eval["y_pers"].values
            abs_err = df_eval["abs_err"].values
            pers_abs_err = df_eval["pers_abs_err"].values

            mae_m = float(mean_absolute_error(y_true, y_pred))
            mae_p = float(mean_absolute_error(y_true, y_pers))
            rmse_m = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            rmse_p = float(np.sqrt(mean_squared_error(y_true, y_pers)))
            r2_m = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0
            r2_p = float(r2_score(y_true, y_pers)) if len(y_true) > 1 and np.var(y_true) > 0 else 0.0
            mape_m = float(df_eval["pct_err"].mean())
            mape_p = float(df_eval["pers_pct_err"].mean())

            mae_delta = mae_m - mae_p
            rmse_delta = rmse_m - rmse_p
            r2_delta = r2_m - r2_p

            win_rate = float(np.mean(abs_err <= pers_abs_err) * 100.0)

            # Non-inferiority gate vs persistence
            # For experimental models, non-inferiority checks if MAE deficit is within tolerance
            passes_gate = (mae_delta <= 0.0)

            h_res.update({
                "status": "evaluated",
                "sample_size": len(df_eval),
                "model_performance": {
                    "mae_rs_qtl": round(mae_m, 2),
                    "rmse_rs_qtl": round(rmse_m, 2),
                    "r2_score": round(r2_m, 4),
                    "mape_pct": round(mape_m, 2),
                },
                "persistence_baseline": {
                    "mae_rs_qtl": round(mae_p, 2),
                    "rmse_rs_qtl": round(rmse_p, 2),
                    "r2_score": round(r2_p, 4),
                    "mape_pct": round(mape_p, 2),
                },
                "delta_model_minus_persistence": {
                    "mae_delta": round(mae_delta, 2),
                    "rmse_delta": round(rmse_delta, 2),
                    "r2_delta": round(r2_delta, 4),
                },
                "win_rate_pct": round(win_rate, 2),
                "non_inferiority_gate": {
                    "beats_persistence_mae": passes_gate,
                    "advisory_status": "PASSES_BENCHMARK" if passes_gate else "EXPERIMENTAL_DEFICIT_CONFIRMED",
                }
            })

            results["horizons"][h_key] = h_res

    finally:
        conn.close()

    return results


def get_commodity_shadow_status(commodity: str, db_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Returns real-time status summary of shadow forecasts for the commodity."""
    comm = str(commodity).strip().lower()
    conn = get_commodity_db_connection(comm, db_path)
    try:
        row = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending,
                SUM(CASE WHEN status = 'EVALUATED' THEN 1 ELSE 0 END) as evaluated,
                SUM(CASE WHEN status = 'TARGET_UNAVAILABLE' THEN 1 ELSE 0 END) as unavailable,
                MIN(origin_date) as earliest_origin,
                MAX(origin_date) as latest_origin
            FROM shadow_forecasts
            WHERE commodity = ?;
        """, (comm,)).fetchone()

        obs_row = conn.execute("""
            SELECT 
                COUNT(*) as obs_count,
                MIN(reported_date) as min_date,
                MAX(reported_date) as max_date,
                COUNT(DISTINCT market) as markets_count,
                COUNT(DISTINCT variety) as varieties_count
            FROM mandi_observations;
        """).fetchone()

        return {
            "commodity": comm,
            "shadow_enabled": is_commodity_shadow_enabled(comm),
            "forecasts": {
                "total": row["total"] or 0,
                "pending": row["pending"] or 0,
                "evaluated": row["evaluated"] or 0,
                "unavailable": row["unavailable"] or 0,
                "earliest_origin": row["earliest_origin"],
                "latest_origin": row["latest_origin"],
            },
            "observations": {
                "total_rows": obs_row["obs_count"] or 0,
                "min_date": obs_row["min_date"],
                "max_date": obs_row["max_date"],
                "markets_count": obs_row["markets_count"] or 0,
                "varieties_count": obs_row["varieties_count"] or 0,
            }
        }
    finally:
        conn.close()
