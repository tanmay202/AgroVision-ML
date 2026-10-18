"""
AgroVision-ML — Member 2: Live Shadow-Mode Evaluation for Rice Price Forecasting

Provides controlled, opt-in shadow-mode evaluation of 14-day and 30-day rice price
forecasting models against subsequent actual mandi observations.

Key Architectural Guarantees:
1. Production Isolation: Never alters, retrains, or affects frozen 1-7 day production model
   or legacy prediction endpoints (POST /predict).
2. Operational Controls: Controlled by AGROVISION_ENABLE_SHADOW_EVALUATION switch.
   Can evaluate internally while public client horizon endpoints remain disabled.
3. Point-in-Time Integrity: Forecast generation uses observations up to origin date only.
   Zero future target data can enter feature calculations.
4. Idempotency & Deduplication: Forecasts are assigned deterministic IDs derived from
   (market, variety, origin_date, horizon_days, model_version). Repeated runs are no-ops.
5. Delayed Outcome Evaluation: Forecasts are evaluated only after target calendar window
   [t + H, t + H + 3] has materialized:
   - PENDING: Target window still open or target date not yet arrived.
   - EVALUATED: Matched first valid observation in target window.
   - TARGET_UNAVAILABLE: Window definitively closed without any report in window.
6. Honest Matched Benchmarking: Evaluates XGBoost and Persistence baseline on exactly
   the same matched forecast origins and realized target prices.
7. Lightweight Persistence: Zero heavyweight dependencies; uses standard library SQLite
   with ACID transactions, WAL mode, and structured tables.
"""

import sys
import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    get_shadow_db_path,
    is_shadow_evaluation_enabled,
    is_horizon_forecasting_enabled,
    is_horizon_14d_enabled,
    is_horizon_30d_enabled,
)
from inference import RiceInferencePipeline, InsufficientHistoryError


# ============================================================
# 1. DATABASE SCHEMA & INITIALIZATION
# ============================================================

def init_shadow_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Initializes the SQLite schema for shadow forecast tracking and mandi observations.
    
    Creates:
    - mandi_observations: Real historical and newly ingested mandi records.
    - shadow_forecasts: Stored point-in-time forecasts and evaluation outcomes.
    """
    resolved_path = Path(db_path) if db_path else get_shadow_db_path()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(resolved_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        # 1. Real mandi observations table
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

        # 2. Shadow forecasts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_forecasts (
                forecast_id TEXT PRIMARY KEY,
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
                created_at TEXT NOT NULL,
                status TEXT NOT NULL, -- 'PENDING', 'EVALUATED', 'TARGET_UNAVAILABLE'
                matched_observation_date TEXT,
                matched_actual_price REAL,
                actual_horizon_offset INTEGER,
                absolute_error REAL,
                percentage_error REAL,
                persistence_pred_price REAL,
                persistence_absolute_error REAL,
                persistence_percentage_error REAL,
                evaluated_at TEXT,
                finalization_reason TEXT,
                cohort TEXT DEFAULT 'RETROSPECTIVE'
            );
        """)

        # Migration check: add cohort column if table already existed without it
        try:
            conn.execute("ALTER TABLE shadow_forecasts ADD COLUMN cohort TEXT DEFAULT 'RETROSPECTIVE';")
        except sqlite3.OperationalError:
            pass

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_lookup 
            ON shadow_forecasts (market, variety, status, horizon_days);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_origin 
            ON shadow_forecasts (origin_date, horizon_days);
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_cohort 
            ON shadow_forecasts (cohort, status, horizon_days);
        """)

        # 3. Persistent operational audit log table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shadow_audit_log (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                execution_type TEXT NOT NULL,
                status TEXT NOT NULL,
                source_label TEXT,
                latest_observation_date TEXT,
                reporting_lag_days INTEGER,
                is_stale INTEGER,
                drift_detected INTEGER,
                raw_rows INTEGER,
                accepted_rows INTEGER,
                rejected_rows INTEGER,
                forecasts_generated INTEGER,
                forecasts_skipped INTEGER,
                outcomes_evaluated INTEGER,
                outcomes_pending INTEGER,
                duration_seconds REAL,
                details_json TEXT
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_log_ts
            ON shadow_audit_log (timestamp, execution_type);
        """)

    return resolved_path


def get_db_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Returns an active SQLite connection configured with Row factory."""
    resolved_path = init_shadow_db(db_path)
    conn = sqlite3.connect(resolved_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# 2. BATCH INGESTION OF NEW MANDI OBSERVATIONS
# ============================================================

def normalize_observations_df(df: pd.DataFrame) -> pd.DataFrame:
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


def ingest_mandi_observations(
    data: Union[pd.DataFrame, str, Path],
    db_path: Optional[Union[str, Path]] = None,
    return_stats: bool = False,
) -> Union[int, Dict[str, Union[int, str, None]]]:
    """
    Ingests newly reported mandi observations into the persistent database.
    
    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Observations DataFrame or path to CSV file.
    db_path : Path or str, optional
        Target SQLite database path.
    return_stats : bool, default False
        If True, returns a detailed dictionary with ingestion statistics.
        If False, returns integer count of accepted records (backward-compatible).

    Returns
    -------
    int or dict
        Count of observations inserted or updated, or dictionary of statistics.
    """
    if isinstance(data, (str, Path)):
        p = Path(data)
        if not p.exists():
            raise FileNotFoundError(f"Observations file not found at {p}")
        df = pd.read_csv(p)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise TypeError(f"Expected DataFrame or Path, received {type(data)}")

    total_input_rows = len(df)
    clean_df = normalize_observations_df(df)
    if clean_df.empty:
        if return_stats:
            return {
                "total_input_rows": total_input_rows,
                "accepted_count": 0,
                "rejected_count": total_input_rows,
                "min_date": None,
                "max_date": None,
                "markets_count": 0,
                "varieties_count": 0,
            }
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

    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO mandi_observations (market, variety, reported_date, modal_price, arrivals, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(market, variety, reported_date) DO UPDATE SET
                modal_price = excluded.modal_price,
                arrivals = excluded.arrivals;
        """, records)
        conn.commit()

    if return_stats:
        return {
            "total_input_rows": total_input_rows,
            "accepted_count": len(records),
            "rejected_count": total_input_rows - len(records),
            "min_date": clean_df["reported_date"].min(),
            "max_date": clean_df["reported_date"].max(),
            "markets_count": int(clean_df["market"].nunique()),
            "varieties_count": int(clean_df["variety"].nunique()),
        }
    return len(records)


# ============================================================
# 3. IDEMPOTENT SHADOW FORECAST GENERATION
# ============================================================

def make_forecast_id(
    market: str,
    variety: str,
    origin_date: str,
    horizon_days: int,
    model_version: str = "1.0.0",
) -> str:
    """Generates a deterministic unique identifier for a forecast."""
    key = f"{market.strip().lower()}|{variety.strip().lower()}|{origin_date.strip()}|{int(horizon_days)}|{model_version.strip()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def generate_shadow_forecasts(
    history_df: pd.DataFrame,
    horizons: Tuple[int, ...] = (14, 30),
    origin_date: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    pipeline: Optional[RiceInferencePipeline] = None,
    force_enabled: bool = False,
    cohort: str = "RETROSPECTIVE",
) -> Dict[str, Union[int, str, list]]:
    """
    Generates point-in-time shadow forecasts for 14-day and 30-day horizons.
    
    Idempotent: Skips forecasts that already exist in the database for the given
    (market, variety, origin_date, horizon, model_version).
    """
    if not force_enabled and not is_shadow_evaluation_enabled():
        raise RuntimeError(
            "Shadow evaluation is currently disabled. Set AGROVISION_ENABLE_SHADOW_EVALUATION=true to enable."
        )

    if pipeline is None:
        pipeline = RiceInferencePipeline()

    clean_df = history_df.copy()
    # Normalize column names to standard pipeline format
    rename_dict = {}
    for c in clean_df.columns:
        c_low = str(c).strip().lower()
        if c_low in ("market", "market_name") and "Market Name" not in clean_df.columns:
            rename_dict[c] = "Market Name"
        elif c_low == "variety" and "Variety" not in clean_df.columns:
            rename_dict[c] = "Variety"
        elif c_low in ("reported_date", "date") and DATE_COLUMN not in clean_df.columns:
            rename_dict[c] = DATE_COLUMN
        elif c_low in ("modal_price", "price") and PRICE_COLUMN not in clean_df.columns:
            rename_dict[c] = PRICE_COLUMN
        elif c_low in ("arrivals", "arrivals_tonnes") and ARRIVAL_COLUMN not in clean_df.columns:
            rename_dict[c] = ARRIVAL_COLUMN

    clean_df = clean_df.rename(columns=rename_dict)
    clean_df[DATE_COLUMN] = pd.to_datetime(clean_df[DATE_COLUMN])
    clean_df = clean_df.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)

    if origin_date is None:
        origin_date = clean_df[DATE_COLUMN].max().strftime("%Y-%m-%d")

    origin_dt = pd.to_datetime(origin_date)
    # Strictly isolate history to <= origin_date (Zero future leakage!)
    clean_df = clean_df[clean_df[DATE_COLUMN] <= origin_dt].copy()

    # Also ingest these observations into the database so they are known
    ingest_mandi_observations(clean_df, db_path=db_path)

    generated_records = []
    skipped_count = 0
    now_ts = datetime.now(timezone.utc).isoformat()

    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    # Find existing forecast IDs to skip duplicate work
    cursor.execute("SELECT forecast_id FROM shadow_forecasts;")
    existing_ids = {row["forecast_id"] for row in cursor.fetchall()}

    prediction_errors = []

    # Group by Market + Variety
    for (mkt, var), group in clean_df.groupby(GROUP_COLUMNS):
        if len(group) < 30:
            # Insufficient history for feature lags; skip group
            continue

        for h in horizons:
            if h not in (14, 30):
                continue

            model_ver = "1.0.0"
            fid = make_forecast_id(mkt, var, origin_date, h, model_ver)
            if fid in existing_ids:
                skipped_count += 1
                continue

            try:
                # Direct prediction call using point-in-time observations (internal evaluation bypasses public switches)
                pred = pipeline.predict_horizon(
                    df=group,
                    horizon_days=h,
                    market=mkt,
                    variety=var,
                    prediction_date=origin_date,
                    bypass_switches=True,
                )
            except Exception as e:
                prediction_errors.append(f"{mkt}/{var}/{h}: {str(e)}")
                continue

            cur_price = float(pred["current_modal_price"])
            pred_change = float(pred["predicted_price_change"])
            pred_price = float(pred["predicted_modal_price"])
            req_target_dt = pred["requested_target_date"]
            win_start = req_target_dt
            win_end = (pd.to_datetime(req_target_dt) + pd.Timedelta(days=3)).strftime("%Y-%m-%d")
            m_name = pred["model_name"]

            generated_records.append((
                fid,
                mkt,
                var,
                origin_date,
                h,
                cur_price,
                pred_change,
                pred_price,
                req_target_dt,
                win_start,
                win_end,
                m_name,
                model_ver,
                now_ts,
                "PENDING",
                cohort,
            ))

    if generated_records:
        cursor.executemany("""
            INSERT OR IGNORE INTO shadow_forecasts (
                forecast_id, market, variety, origin_date, horizon_days,
                origin_modal_price, predicted_price_change, predicted_modal_price,
                requested_target_date, target_window_start, target_window_end,
                model_name, model_version, created_at, status, cohort
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, generated_records)
        conn.commit()

    conn.close()

    return {
        "origin_date": origin_date,
        "cohort": cohort,
        "generated_count": len(generated_records),
        "skipped_existing_count": skipped_count,
        "horizons": list(horizons),
        "errors_count": len(prediction_errors),
        "errors": prediction_errors[:10],
    }


# ============================================================
# 4. DELAYED OUTCOME EVALUATION
# ============================================================

def evaluate_pending_shadow_forecasts(
    db_path: Optional[Union[str, Path]] = None,
    recheck_unavailable: bool = False,
) -> Dict[str, int]:
    """
    Evaluates pending shadow forecasts against subsequently arrived observations.
    
    Target Window Semantics (strictly matching src/horizon_targets.py):
    - [t + H, t + H + 3] calendar days.
    - If valid observation exists inside window: matches earliest observation -> EVALUATED.
    - If no observation inside window AND latest observation > (t + H + 3):
      target window definitively closed -> TARGET_UNAVAILABLE.
    - If no observation inside window AND latest observation <= (t + H + 3):
      window still open or date not yet reached (ingestion delay) -> stays PENDING.
    
    Parameters
    ----------
    db_path : Path or str, optional
        Target SQLite database path.
    recheck_unavailable : bool, default False
        If True, re-evaluates forecasts previously marked TARGET_UNAVAILABLE in case
        late-arriving observation backfills cover the target window.
    """
    conn = get_db_connection(db_path)
    cursor = conn.cursor()

    status_filter = "status IN ('PENDING', 'TARGET_UNAVAILABLE')" if recheck_unavailable else "status = 'PENDING'"
    cursor.execute(f"""
        SELECT 
            forecast_id, market, variety, origin_date, horizon_days,
            origin_modal_price, predicted_modal_price, requested_target_date,
            target_window_start, target_window_end, status
        FROM shadow_forecasts
        WHERE {status_filter};
    """)
    target_rows = cursor.fetchall()

    if not target_rows:
        conn.close()
        return {"evaluated_count": 0, "unavailable_count": 0, "still_pending_count": 0, "recovered_late_count": 0}

    evaluated_updates = []
    unavailable_updates = []
    still_pending_count = 0
    recovered_late_count = 0
    now_ts = datetime.now(timezone.utc).isoformat()

    for row in target_rows:
        fid = row["forecast_id"]
        mkt = row["market"]
        var = row["variety"]
        orig_dt_str = row["origin_date"]
        h_days = row["horizon_days"]
        orig_price = float(row["origin_modal_price"])
        pred_price = float(row["predicted_modal_price"])
        win_start = row["target_window_start"]
        win_end = row["target_window_end"]
        prior_status = row["status"]

        # Check for first valid observation in window [win_start, win_end]
        cursor.execute("""
            SELECT reported_date, modal_price 
            FROM mandi_observations
            WHERE market = ? AND variety = ? AND reported_date >= ? AND reported_date <= ?
            ORDER BY reported_date ASC
            LIMIT 1;
        """, (mkt, var, win_start, win_end))
        matched = cursor.fetchone()

        if matched:
            matched_dt_str = matched["reported_date"]
            matched_price = float(matched["modal_price"])
            offset = int((pd.to_datetime(matched_dt_str) - pd.to_datetime(orig_dt_str)).days)

            # Model errors
            abs_err = float(abs(matched_price - pred_price))
            pct_err = float((abs_err / matched_price) * 100.0) if matched_price != 0 else np.nan

            # Persistence errors (Baseline predicts origin_modal_price)
            pers_pred = orig_price
            pers_abs_err = float(abs(matched_price - pers_pred))
            pers_pct_err = float((pers_abs_err / matched_price) * 100.0) if matched_price != 0 else np.nan

            reason = "Late observation arrived and resolved unavailable target" if prior_status == "TARGET_UNAVAILABLE" else "Matched valid observation in window"
            if prior_status == "TARGET_UNAVAILABLE":
                recovered_late_count += 1

            evaluated_updates.append((
                matched_dt_str,
                matched_price,
                offset,
                round(abs_err, 2),
                round(pct_err, 2) if not np.isnan(pct_err) else None,
                pers_pred,
                round(pers_abs_err, 2),
                round(pers_pct_err, 2) if not np.isnan(pers_pct_err) else None,
                now_ts,
                reason,
                fid,
            ))
        else:
            if prior_status == "TARGET_UNAVAILABLE":
                # Stays unavailable
                continue

            # Check latest available observation for this group
            cursor.execute("""
                SELECT MAX(reported_date) as max_date 
                FROM mandi_observations 
                WHERE market = ? AND variety = ?;
            """, (mkt, var))
            latest_res = cursor.fetchone()
            latest_date = latest_res["max_date"] if latest_res and latest_res["max_date"] else None

            if latest_date and latest_date > win_end:
                # Window closed without an observation
                reason = f"Window closed on {win_end}; latest observation is {latest_date}"
                unavailable_updates.append((now_ts, reason, fid))
            else:
                # Window still open or target date not yet reached (ingestion delay)
                still_pending_count += 1

    # Execute batch updates
    if evaluated_updates:
        cursor.executemany("""
            UPDATE shadow_forecasts SET
                status = 'EVALUATED',
                matched_observation_date = ?,
                matched_actual_price = ?,
                actual_horizon_offset = ?,
                absolute_error = ?,
                percentage_error = ?,
                persistence_pred_price = ?,
                persistence_absolute_error = ?,
                persistence_percentage_error = ?,
                evaluated_at = ?,
                finalization_reason = ?
            WHERE forecast_id = ?;
        """, evaluated_updates)

    if unavailable_updates:
        cursor.executemany("""
            UPDATE shadow_forecasts SET
                status = 'TARGET_UNAVAILABLE',
                evaluated_at = ?,
                finalization_reason = ?
            WHERE forecast_id = ?;
        """, unavailable_updates)

    conn.commit()
    conn.close()

    return {
        "evaluated_count": len(evaluated_updates),
        "unavailable_count": len(unavailable_updates),
        "still_pending_count": still_pending_count,
        "recovered_late_count": recovered_late_count,
    }


# ============================================================
# 5. HONEST LIVE PERFORMANCE METRICS CALCULATION
# ============================================================

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Safe MAPE excluding exact zeros."""
    valid = y_true != 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def compute_shadow_evaluation_metrics(
    db_path: Optional[Union[str, Path]] = None,
    horizon_days: Optional[int] = None,
    cohort: Optional[str] = None,
) -> Dict[str, Union[dict, str]]:
    """
    Computes rigorous live evaluation metrics comparing XGBoost vs Persistence baseline.
    
    Includes:
    - Counts: total, evaluated, pending, unavailable.
    - Global metrics: MAE, RMSE, R2, MAPE, tolerance +/- Rs.25/qtl.
    - Statistical bounds: 95% Confidence Intervals for MAE when N >= 10.
    - Percentiles: p50 (median error), p90 error.
    - Granular subsets: Unchanged prices (flat) vs Changed prices (movement).
    - Breakdown by actual horizon offset (e.g. 14, 15, 16, 17 days).
    - Breakdown by top markets and varieties.
    - Optional cohort filtering ('PROSPECTIVE' vs 'RETROSPECTIVE').
    """
    conn = get_db_connection(db_path)
    
    query = "SELECT * FROM shadow_forecasts"
    conditions = []
    params = []
    if horizon_days is not None:
        conditions.append("horizon_days = ?")
        params.append(int(horizon_days))
    if cohort is not None:
        conditions.append("cohort = ?")
        params.append(str(cohort))

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return {
            "status": "empty",
            "message": "No shadow forecasts recorded in database.",
            "metrics": {},
        }

    results = {}
    horizons = [horizon_days] if horizon_days is not None else [14, 30]

    for h in horizons:
        sub = df[df["horizon_days"] == h]
        if sub.empty:
            continue

        n_total = len(sub)
        n_eval = int((sub["status"] == "EVALUATED").sum())
        n_pend = int((sub["status"] == "PENDING").sum())
        n_unavail = int((sub["status"] == "TARGET_UNAVAILABLE").sum())

        eval_sub = sub[sub["status"] == "EVALUATED"].copy()

        if n_eval == 0:
            results[f"{h}d"] = {
                "counts": {
                    "total_forecasts": n_total,
                    "evaluated_forecasts": 0,
                    "pending_forecasts": n_pend,
                    "unavailable_target_forecasts": n_unavail,
                },
                "status": "pending_outcomes",
                "message": "All generated forecasts are currently awaiting outcome dates or observations.",
            }
            continue

        y_true = eval_sub["matched_actual_price"].values.astype(float)
        y_model = eval_sub["predicted_modal_price"].values.astype(float)
        y_base = eval_sub["persistence_pred_price"].values.astype(float)

        err_model = np.abs(y_true - y_model)
        err_base = np.abs(y_true - y_base)

        mae_m = float(mean_absolute_error(y_true, y_model))
        mae_b = float(mean_absolute_error(y_true, y_base))

        rmse_m = float(np.sqrt(mean_squared_error(y_true, y_model)))
        rmse_b = float(np.sqrt(mean_squared_error(y_true, y_base)))

        # R2 score only meaningful when target variance > 0 and sample size >= 2
        var_y = float(np.var(y_true))
        r2_m = float(r2_score(y_true, y_model)) if (len(y_true) >= 2 and var_y > 1e-5) else np.nan
        r2_b = float(r2_score(y_true, y_base)) if (len(y_true) >= 2 and var_y > 1e-5) else np.nan

        mape_m = safe_mape(y_true, y_model)
        mape_b = safe_mape(y_true, y_base)

        within25_m = float(np.mean(err_model <= 25.0) * 100.0)
        within25_b = float(np.mean(err_base <= 25.0) * 100.0)

        p50_m, p90_m = np.percentile(err_model, [50, 90])
        p50_b, p90_b = np.percentile(err_base, [50, 90])

        # 95% Confidence Intervals for MAE (when N >= 10)
        ci_m = None
        ci_b = None
        ci_delta = None
        if n_eval >= 10:
            se_m = float(np.std(err_model, ddof=1) / np.sqrt(n_eval))
            ci_m = [round(mae_m - 1.96 * se_m, 2), round(mae_m + 1.96 * se_m, 2)]

            se_b = float(np.std(err_base, ddof=1) / np.sqrt(n_eval))
            ci_b = [round(mae_b - 1.96 * se_b, 2), round(mae_b + 1.96 * se_b, 2)]

            paired_diff = err_model - err_base
            se_diff = float(np.std(paired_diff, ddof=1) / np.sqrt(n_eval))
            delta_mae = mae_m - mae_b
            ci_delta = [round(delta_mae - 1.96 * se_diff, 2), round(delta_mae + 1.96 * se_diff, 2)]

        # Subsets: Flat (unchanged) vs Movement (changed)
        actual_change = y_true - y_base
        flat_mask = np.isclose(actual_change, 0.0, atol=1e-5)
        move_mask = ~flat_mask

        subsets_info = {}
        if np.any(flat_mask):
            subsets_info["flat_market_days"] = {
                "count": int(np.sum(flat_mask)),
                "pct_of_evaluated": round(float(np.mean(flat_mask) * 100.0), 2),
                "model_mae": round(float(mean_absolute_error(y_true[flat_mask], y_model[flat_mask])), 2),
                "persistence_mae": 0.0,
            }
        if np.any(move_mask):
            subsets_info["movement_market_days"] = {
                "count": int(np.sum(move_mask)),
                "pct_of_evaluated": round(float(np.mean(move_mask) * 100.0), 2),
                "model_mae": round(float(mean_absolute_error(y_true[move_mask], y_model[move_mask])), 2),
                "persistence_mae": round(float(mean_absolute_error(y_true[move_mask], y_base[move_mask])), 2),
                "model_rmse": round(float(np.sqrt(mean_squared_error(y_true[move_mask], y_model[move_mask]))), 2),
                "persistence_rmse": round(float(np.sqrt(mean_squared_error(y_true[move_mask], y_base[move_mask]))), 2),
            }

        # Offset breakdown
        offset_counts = eval_sub["actual_horizon_offset"].value_counts().to_dict()
        offset_dist = {int(k): int(v) for k, v in sorted(offset_counts.items())}

        # Market breakdown
        mkt_metrics = {}
        for mkt, m_group in eval_sub.groupby("market"):
            if len(m_group) >= 5:
                yt_m = m_group["matched_actual_price"].values
                ym_m = m_group["predicted_modal_price"].values
                yb_m = m_group["persistence_pred_price"].values
                mkt_metrics[mkt] = {
                    "count": len(m_group),
                    "model_mae": round(float(mean_absolute_error(yt_m, ym_m)), 2),
                    "persistence_mae": round(float(mean_absolute_error(yt_m, yb_m)), 2),
                    "model_rmse": round(float(np.sqrt(mean_squared_error(yt_m, ym_m))), 2),
                    "persistence_rmse": round(float(np.sqrt(mean_squared_error(yt_m, yb_m))), 2),
                }

        # Variety breakdown
        var_metrics = {}
        for var, v_group in eval_sub.groupby("variety"):
            if len(v_group) >= 5:
                yt_v = v_group["matched_actual_price"].values
                ym_v = v_group["predicted_modal_price"].values
                yb_v = v_group["persistence_pred_price"].values
                var_metrics[var] = {
                    "count": len(v_group),
                    "model_mae": round(float(mean_absolute_error(yt_v, ym_v)), 2),
                    "persistence_mae": round(float(mean_absolute_error(yt_v, yb_v)), 2),
                    "model_rmse": round(float(np.sqrt(mean_squared_error(yt_v, ym_v))), 2),
                    "persistence_rmse": round(float(np.sqrt(mean_squared_error(yt_v, yb_v))), 2),
                }

        results[f"{h}d"] = {
            "counts": {
                "total_forecasts": n_total,
                "evaluated_forecasts": n_eval,
                "pending_forecasts": n_pend,
                "unavailable_target_forecasts": n_unavail,
            },
            "model_performance": {
                "mae_rs_qtl": round(mae_m, 2),
                "mae_95_ci": ci_m,
                "rmse_rs_qtl": round(rmse_m, 2),
                "r2_score": round(r2_m, 4) if not np.isnan(r2_m) else None,
                "mape_pct": round(mape_m, 2) if not np.isnan(mape_m) else None,
                "within_25_rs_pct": round(within25_m, 2),
                "median_error_p50": round(float(p50_m), 2),
                "error_p90": round(float(p90_m), 2),
            },
            "persistence_baseline": {
                "mae_rs_qtl": round(mae_b, 2),
                "mae_95_ci": ci_b,
                "rmse_rs_qtl": round(rmse_b, 2),
                "r2_score": round(r2_b, 4) if not np.isnan(r2_b) else None,
                "mape_pct": round(mape_b, 2) if not np.isnan(mape_b) else None,
                "within_25_rs_pct": round(within25_b, 2),
                "median_error_p50": round(float(p50_b), 2),
                "error_p90": round(float(p90_b), 2),
            },
            "delta_model_minus_persistence": {
                "mae": round(mae_m - mae_b, 2),
                "mae_delta_95_ci": ci_delta,
                "rmse": round(rmse_m - rmse_b, 2),
                "r2": round(r2_m - r2_b, 4) if (not np.isnan(r2_m) and not np.isnan(r2_b)) else None,
            },
            "subsets": subsets_info,
            "actual_horizon_offset_distribution": offset_dist,
            "market_breakdown": mkt_metrics,
            "variety_breakdown": var_metrics,
            "statistical_caveats": (
                "Overlapping forecast origins create serial autocorrelation; simple i.i.d. "
                "statistical significance tests are invalid. Evaluate across full seasonal cycles."
            ),
        }

    return results


# ============================================================
# 6. SHADOW STATUS & HEALTH
# ============================================================

def get_shadow_status(db_path: Optional[Union[str, Path]] = None) -> dict:
    """Returns operational status, database readiness, and count summaries."""
    resolved_path = Path(db_path) if db_path else get_shadow_db_path()
    db_exists = resolved_path.exists()

    status_dict = {
        "shadow_evaluation_enabled": is_shadow_evaluation_enabled(),
        "database_path": str(resolved_path),
        "database_exists": db_exists,
        "latest_evaluation_timestamp": None,
        "counts": {
            "total_observations": 0,
            "forecasts_14d": {"total": 0, "pending": 0, "evaluated": 0, "unavailable": 0},
            "forecasts_30d": {"total": 0, "pending": 0, "evaluated": 0, "unavailable": 0},
            "by_cohort": {
                "retrospective": {"total": 0, "pending": 0, "evaluated": 0, "unavailable": 0},
                "prospective": {"total": 0, "pending": 0, "evaluated": 0, "unavailable": 0},
            },
        },
    }

    if not db_exists:
        return status_dict

    try:
        conn = get_db_connection(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) as cnt FROM mandi_observations;")
        status_dict["counts"]["total_observations"] = cursor.fetchone()["cnt"]

        cursor.execute("SELECT MAX(created_at) as latest_ingest, MAX(reported_date) as latest_obs FROM mandi_observations;")
        obs_meta = cursor.fetchone()
        status_dict["latest_successful_ingestion_timestamp"] = obs_meta["latest_ingest"] if obs_meta else None
        status_dict["latest_actual_observation_date"] = obs_meta["latest_obs"] if obs_meta else None

        cursor.execute("SELECT MAX(evaluated_at) as latest FROM shadow_forecasts WHERE evaluated_at IS NOT NULL;")
        latest_res = cursor.fetchone()
        status_dict["latest_evaluation_timestamp"] = latest_res["latest"] if latest_res else None

        if status_dict["latest_evaluation_timestamp"]:
            try:
                eval_dt = pd.to_datetime(status_dict["latest_evaluation_timestamp"])
                if eval_dt.tzinfo is None:
                    eval_dt = eval_dt.replace(tzinfo=timezone.utc)
                now_dt = datetime.now(timezone.utc)
                status_dict["time_since_last_evaluation_seconds"] = int((now_dt - eval_dt).total_seconds())
            except Exception:
                status_dict["time_since_last_evaluation_seconds"] = None
        else:
            status_dict["time_since_last_evaluation_seconds"] = None

        for h in (14, 30):
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status='EVALUATED' THEN 1 ELSE 0 END) as evaluated,
                    SUM(CASE WHEN status='TARGET_UNAVAILABLE' THEN 1 ELSE 0 END) as unavailable
                FROM shadow_forecasts WHERE horizon_days = ?;
            """, (h,))
            h_row = cursor.fetchone()
            if h_row:
                status_dict["counts"][f"forecasts_{h}d"] = {
                    "total": h_row["total"] or 0,
                    "pending": h_row["pending"] or 0,
                    "evaluated": h_row["evaluated"] or 0,
                    "unavailable": h_row["unavailable"] or 0,
                }

        for coh in ("RETROSPECTIVE", "PROSPECTIVE"):
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status='PENDING' THEN 1 ELSE 0 END) as pending,
                    SUM(CASE WHEN status='EVALUATED' THEN 1 ELSE 0 END) as evaluated,
                    SUM(CASE WHEN status='TARGET_UNAVAILABLE' THEN 1 ELSE 0 END) as unavailable
                FROM shadow_forecasts WHERE UPPER(cohort) = ?;
            """, (coh,))
            c_row = cursor.fetchone()
            if c_row:
                status_dict["counts"]["by_cohort"][coh.lower()] = {
                    "total": c_row["total"] or 0,
                    "pending": c_row["pending"] or 0,
                    "evaluated": c_row["evaluated"] or 0,
                    "unavailable": c_row["unavailable"] or 0,
                }

        # Query latest audit log entry if table exists
        try:
            cursor.execute("""
                SELECT log_id, timestamp, execution_type, status, source_label,
                       latest_observation_date, reporting_lag_days, is_stale,
                       drift_detected, forecasts_generated, outcomes_evaluated
                FROM shadow_audit_log
                ORDER BY log_id DESC LIMIT 1;
            """)
            audit_row = cursor.fetchone()
            if audit_row:
                status_dict["latest_audit_log"] = dict(audit_row)
            else:
                status_dict["latest_audit_log"] = None
        except sqlite3.OperationalError:
            status_dict["latest_audit_log"] = None

        conn.close()
    except Exception as e:
        status_dict["error"] = str(e)

    return status_dict


def record_audit_log_entry(
    db_path: Optional[Union[str, Path]] = None,
    execution_type: str = "PROSPECTIVE_DAILY",
    status: str = "SUCCESS",
    source_label: Optional[str] = None,
    latest_observation_date: Optional[str] = None,
    reporting_lag_days: Optional[int] = None,
    is_stale: bool = False,
    drift_detected: bool = False,
    raw_rows: int = 0,
    accepted_rows: int = 0,
    rejected_rows: int = 0,
    forecasts_generated: int = 0,
    forecasts_skipped: int = 0,
    outcomes_evaluated: int = 0,
    outcomes_pending: int = 0,
    duration_seconds: float = 0.0,
    details: Optional[dict] = None,
) -> int:
    """Inserts a structured operational execution record into shadow_audit_log."""
    resolved_path = init_shadow_db(db_path)
    with sqlite3.connect(resolved_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO shadow_audit_log (
                timestamp, execution_type, status, source_label, latest_observation_date,
                reporting_lag_days, is_stale, drift_detected, raw_rows, accepted_rows,
                rejected_rows, forecasts_generated, forecasts_skipped, outcomes_evaluated,
                outcomes_pending, duration_seconds, details_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            datetime.now(timezone.utc).isoformat(),
            execution_type,
            status,
            source_label,
            latest_observation_date,
            reporting_lag_days,
            1 if is_stale else 0,
            1 if drift_detected else 0,
            raw_rows,
            accepted_rows,
            rejected_rows,
            forecasts_generated,
            forecasts_skipped,
            outcomes_evaluated,
            outcomes_pending,
            round(duration_seconds, 2),
            json.dumps(details or {}),
        ))
        conn.commit()
        return cursor.lastrowid


def backup_shadow_database(
    db_path: Optional[Union[str, Path]] = None,
    backup_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Creates an atomic, consistent online point-in-time snapshot of the shadow database.
    Uses SQLite's online backup API, safe during concurrent WAL reads and writes.
    """
    resolved_db = Path(db_path) if db_path else get_shadow_db_path()
    if not resolved_db.exists():
        raise FileNotFoundError(f"Shadow database not found at {resolved_db}")

    if backup_path is None:
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = resolved_db.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        target_path = backup_dir / f"{resolved_db.stem}_backup_{timestamp_str}.db"
    else:
        target_path = Path(backup_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(resolved_db)
    dst_conn = sqlite3.connect(target_path)
    try:
        with dst_conn:
            src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()

    return target_path
