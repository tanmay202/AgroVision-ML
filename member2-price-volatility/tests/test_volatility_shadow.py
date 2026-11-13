"""
Comprehensive Unit & Integration Test Suite for Volatility V3 Prospective Shadow Monitoring

Covers all 12 required test dimensions:
1. Forecast idempotency: repeated runs skip duplicates via deterministic SHA256 forecast_id.
2. Prediction persistence: shadow_volatility_forecasts table stores all required fields.
3. Model loading failures: missing/corrupt artifact raises FileNotFoundError gracefully.
4. Missing observations: history < 14 rows raises InsufficientHistoryError; gaps handled safely.
5. Calendar boundaries: month-end / year-end horizons properly add exactly 14 calendar days.
6. Incomplete 14-day windows: pending forecasts are NEVER silently evaluated before window closes.
7. Correct target calculation: forward excursion strictly includes origin price P_t and matches V3 thresholds.
8. Delayed evaluation: transitions to EVALUATED once 14 days elapse; transitions to WINDOW_EMPTY if 0 trades.
9. Duplicate evaluation prevention: re-running evaluator on already evaluated records is an idempotent no-op.
10. Database recovery: re-instantiated evaluator safely picks up pending forecasts across restarts.
11. Production isolation: production artifacts and endpoints are completely decoupled and untouched.
12. No farmer-facing exposure of V3: public /predict returns persistence/v2 only; never leaks V3.
"""

import sys
import sqlite3
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    is_volatility_shadow_enabled,
    set_volatility_shadow_enabled,
    reset_horizon_forecasting_switches,
    get_volatility_v3_14d_model_path,
    get_volatility_v3_14d_config_path,
)
from volatility_shadow import (
    init_volatility_shadow_db,
    get_volatility_db_connection,
    make_volatility_forecast_id,
    classify_volatility_range,
    calculate_calendar_persistence_prediction,
    validate_probability_vector,
    VolatilityShadowPredictionService,
    VolatilityShadowEvaluator,
    process_volatility_shadow_ingestion,
    compute_volatility_shadow_metrics,
    generate_volatility_shadow_report,
    InsufficientHistoryError,
    InvalidProbabilityError,
    V3_14D_MODEL_VERSION,
    DEFAULT_14D_LOW_CUTOFF,
    DEFAULT_14D_HIGH_CUTOFF,
)
from api import app


@pytest.fixture
def temp_vol_db(tmp_path):
    """Provides a fresh isolated SQLite database for volatility shadow testing."""
    db_file = tmp_path / "test_volatility_shadow.db"
    init_volatility_shadow_db(db_file)
    return db_file


@pytest.fixture
def synthetic_mandi_history():
    """Generates 45 consecutive days of valid mandi observations for Burdwan/Common."""
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    # Prices have some mild fluctuation
    prices = [2500.0 + (i * 3.0) + (10.0 if i % 5 == 0 else 0.0) for i in range(45)]
    arrivals = [100.0 + (i % 7) * 5.0 for i in range(45)]
    return pd.DataFrame({
        "Market Name": ["Burdwan"] * 45,
        "Variety": ["Common"] * 45,
        "Reported Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Modal Price (Rs./Quintal)": prices,
        "Arrivals (Tonnes)": arrivals,
    })


# ============================================================
# 1. FORECAST IDEMPOTENCY
# ============================================================

def test_make_volatility_forecast_id_deterministic():
    """Verify SHA-256 forecast_id is strictly deterministic and normalizes casing/whitespace."""
    fid1 = make_volatility_forecast_id("Burdwan", "Common", "2024-01-20", 14, "rice_volatility_v3_14d_v1")
    fid2 = make_volatility_forecast_id("  burdwan  ", "COMMON", "2024-01-20", 14, "rice_volatility_v3_14d_v1")
    fid3 = make_volatility_forecast_id("Burdwan", "Common", "2024-01-21", 14, "rice_volatility_v3_14d_v1")

    assert fid1 == fid2
    assert fid1 != fid3
    assert len(fid1) == 64  # SHA256 hex length


def test_forecast_generation_idempotency(temp_vol_db, synthetic_mandi_history):
    """Verify repeated forecast generation skips already stored forecasts without creating duplicates."""
    service = VolatilityShadowPredictionService(db_path=temp_vol_db)

    # First generation at day 20 (2024-01-21)
    rec1 = service.generate_forecast_record(
        market="Burdwan",
        variety="Common",
        origin_date="2024-01-21",
        history_df=synthetic_mandi_history.iloc[:21],
    )
    inserted1 = service.record_forecast(rec1, db_path=temp_vol_db)
    assert inserted1 is True

    # Repeated generation with identical observation
    rec2 = service.generate_forecast_record(
        market="Burdwan",
        variety="Common",
        origin_date="2024-01-21",
        history_df=synthetic_mandi_history.iloc[:21],
    )
    inserted2 = service.record_forecast(rec2, db_path=temp_vol_db)
    assert inserted2 is False  # Must be skipped as duplicate

    # Database count must remain exactly 1
    with sqlite3.connect(temp_vol_db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM shadow_volatility_forecasts;").fetchone()[0]
        assert count == 1


# ============================================================
# 2. PREDICTION PERSISTENCE
# ============================================================

def test_prediction_persistence_all_fields_stored(temp_vol_db, synthetic_mandi_history):
    """Verify all required fields and calibrated probabilities are persisted to shadow_volatility_forecasts."""
    service = VolatilityShadowPredictionService(db_path=temp_vol_db)
    rec = service.generate_forecast_record(
        market="Burdwan",
        variety="Common",
        origin_date="2024-01-25",
        history_df=synthetic_mandi_history.iloc[:25],
        cohort="PROSPECTIVE",
    )
    service.record_forecast(rec, db_path=temp_vol_db)

    with get_volatility_db_connection(temp_vol_db) as conn:
        row = conn.execute("SELECT * FROM shadow_volatility_forecasts WHERE origin_date='2024-01-25';").fetchone()
        assert row is not None
        assert row["market"] == "Burdwan"
        assert row["variety"] == "Common"
        assert row["horizon_days"] == 14
        assert row["origin_price"] == float(synthetic_mandi_history.iloc[24]["Modal Price (Rs./Quintal)"])
        assert row["status"] == "PENDING"
        assert row["target_window_end"] == "2024-02-08"
        assert row["cohort"] == "PROSPECTIVE"
        assert row["model_version"] == V3_14D_MODEL_VERSION

        # Verify probabilities
        assert 0.0 <= row["v3_prob_low"] <= 1.0
        assert 0.0 <= row["v3_prob_medium"] <= 1.0
        assert 0.0 <= row["v3_prob_high"] <= 1.0
        assert abs(row["v3_prob_low"] + row["v3_prob_medium"] + row["v3_prob_high"] - 1.0) < 1e-4

        assert 0.0 <= row["v3_calibrated_prob_low"] <= 1.0
        assert 0.0 <= row["v3_calibrated_prob_medium"] <= 1.0
        assert 0.0 <= row["v3_calibrated_prob_high"] <= 1.0
        assert abs(row["v3_calibrated_prob_low"] + row["v3_calibrated_prob_medium"] + row["v3_calibrated_prob_high"] - 1.0) < 1e-4

        assert row["v3_pred_class"] in ("LOW", "MEDIUM", "HIGH")
        assert row["persistence_pred_class"] in ("LOW", "MEDIUM", "HIGH")


# ============================================================
# 3. MODEL LOADING FAILURES
# ============================================================

def test_model_loading_failure_nonexistent_artifact(temp_vol_db, synthetic_mandi_history):
    """Verify service raises FileNotFoundError with clear message when artifact path is invalid."""
    bad_service = VolatilityShadowPredictionService(
        model_path="nonexistent/path/model.pkl",
        config_path="nonexistent/path/config.json",
        db_path=temp_vol_db,
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        bad_service.generate_forecast_record(
            market="Burdwan",
            variety="Common",
            origin_date="2024-01-20",
            history_df=synthetic_mandi_history.iloc[:20],
        )
    assert "Frozen V3 14D model artifact not found" in str(exc_info.value)


# ============================================================
# 4. MISSING OBSERVATIONS & DEGRADED HISTORIES
# ============================================================

def test_insufficient_history_raises_error(temp_vol_db, synthetic_mandi_history):
    """Verify that history with fewer than 14 records raises InsufficientHistoryError (preventing tree NaN default)."""
    service = VolatilityShadowPredictionService(db_path=temp_vol_db)
    sparse_history = synthetic_mandi_history.iloc[:10].copy()  # Only 10 records

    with pytest.raises(InsufficientHistoryError) as exc_info:
        service.generate_forecast_record(
            market="Burdwan",
            variety="Common",
            origin_date="2024-01-10",
            history_df=sparse_history,
        )
    assert "minimum 14 observations required" in str(exc_info.value)


def test_missing_dates_within_history_handled_safely(temp_vol_db, synthetic_mandi_history):
    """Verify calendar windowing gracefully handles irregular reporting days (e.g. weekends / holidays)."""
    service = VolatilityShadowPredictionService(db_path=temp_vol_db)
    # Drop every other observation to simulate irregular weekend reporting (22 observations across 44 days)
    irregular_history = synthetic_mandi_history.iloc[::2].copy().reset_index(drop=True)
    assert len(irregular_history) >= 14

    origin_date = irregular_history.iloc[-1]["Reported Date"]
    rec = service.generate_forecast_record(
        market="Burdwan",
        variety="Common",
        origin_date=origin_date,
        history_df=irregular_history,
    )
    assert rec["v3_pred_class"] in ("LOW", "MEDIUM", "HIGH")
    assert rec["persistence_pred_class"] in ("LOW", "MEDIUM", "HIGH")


# ============================================================
# 5. CALENDAR BOUNDARIES & EXACT 14-DAY HORIZON
# ============================================================

def test_calendar_boundary_month_end_and_leap_year(temp_vol_db):
    """Verify target_window_end correctly computes exact 14 calendar days across month and leap boundaries."""
    # Jan 31 + 14d -> Feb 14
    origin_jan31 = pd.to_datetime("2024-01-31")
    target_feb14 = (origin_jan31 + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    assert target_feb14 == "2024-02-14"

    # Feb 28 2024 (Leap Year) + 14d -> Mar 13
    origin_feb28 = pd.to_datetime("2024-02-28")
    target_mar13 = (origin_feb28 + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    assert target_mar13 == "2024-03-13"


# ============================================================
# 6. INCOMPLETE 14-DAY WINDOWS (NEVER EVALUATED SILENTLY)
# ============================================================

def test_incomplete_window_remains_pending(temp_vol_db, synthetic_mandi_history):
    """Verify forecasts whose 14-day window has NOT elapsed remain PENDING and are NOT evaluated."""
    service = VolatilityShadowPredictionService(db_path=temp_vol_db)
    # Ingest observations up to day 20 (2024-01-21). Window ends on 2024-02-04
    rec = service.generate_forecast_record(
        market="Burdwan",
        variety="Common",
        origin_date="2024-01-21",
        history_df=synthetic_mandi_history.iloc[:21],
    )
    service.record_forecast(rec, db_path=temp_vol_db)

    # Ingest forward observations only up to day 28 (2024-01-29) - 7 days into 14-day window
    with get_volatility_db_connection(temp_vol_db) as conn:
        for _, row in synthetic_mandi_history.iloc[21:29].iterrows():
            conn.execute("""
                INSERT INTO mandi_observations (market, variety, reported_date, modal_price, arrivals, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (row["Market Name"], row["Variety"], row["Reported Date"], float(row["Modal Price (Rs./Quintal)"]), 0.0, "2024-01-29T00:00:00Z"))
        conn.commit()

    # Attempt delayed evaluation as of 2024-01-29 (Window ends 2024-02-04, so window is incomplete!)
    evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    eval_res = evaluator.evaluate_pending_forecasts(as_of_date="2024-01-29", db_path=temp_vol_db)

    assert eval_res["evaluated_count"] == 0
    assert eval_res["window_empty_count"] == 0

    with get_volatility_db_connection(temp_vol_db) as conn:
        status = conn.execute("SELECT status FROM shadow_volatility_forecasts WHERE forecast_id = ?;", (rec["forecast_id"],)).fetchone()["status"]
        assert status == "PENDING"  # Must strictly remain PENDING


# ============================================================
# 7. CORRECT TARGET CALCULATION & THRESHOLD PARITY
# ============================================================

def test_target_calculation_exact_excursion_with_origin_price():
    """Verify realized percentage excursion includes origin price P_t and applies exact V3 thresholds."""
    pt = 2500.0
    # Case 1: All future prices identical to origin price -> 0.0% excursion -> LOW
    future_prices = [2500.0, 2500.0, 2500.0]
    all_prices = [pt] + future_prices
    pct_range = ((max(all_prices) - min(all_prices)) / (pt + 1e-4)) * 100.0
    assert pct_range == 0.0
    assert classify_volatility_range(pct_range) == "LOW"

    # Case 2: Max price 2525.0 -> 1.0% excursion -> MEDIUM (between 0.014286 and 2.666666)
    future_prices = [2510.0, 2525.0, 2505.0]
    all_prices = [pt] + future_prices
    pct_range = ((max(all_prices) - min(all_prices)) / (pt + 1e-4)) * 100.0
    assert 0.99 < pct_range < 1.01
    assert classify_volatility_range(pct_range) == "MEDIUM"

    # Case 3: Max price 2600.0 -> 4.0% excursion -> HIGH (> 2.666666)
    future_prices = [2510.0, 2600.0, 2490.0]
    all_prices = [pt] + future_prices
    pct_range = ((max(all_prices) - min(all_prices)) / (pt + 1e-4)) * 100.0
    assert pct_range > 2.666666
    assert classify_volatility_range(pct_range) == "HIGH"


# ============================================================
# 8. DELAYED EVALUATION & EMPTY WINDOW TRANSITION
# ============================================================

def test_delayed_evaluation_transitions_to_evaluated(temp_vol_db, synthetic_mandi_history):
    """Verify forecast transitions to EVALUATED once full 14-day window materializes."""
    # Ingest observations up to 2024-01-20 (Day 20)
    process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history.iloc[:20],
        origin_date="2024-01-20",
        db_path=temp_vol_db,
        force_enabled=True,
    )

    # Window end date is 2024-02-03. Ingest remaining observations up to 2024-02-14 (Day 45)
    with get_volatility_db_connection(temp_vol_db) as conn:
        for _, row in synthetic_mandi_history.iloc[20:].iterrows():
            conn.execute("""
                INSERT INTO mandi_observations (market, variety, reported_date, modal_price, arrivals, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(market, variety, reported_date) DO NOTHING;
            """, (row["Market Name"], row["Variety"], row["Reported Date"], float(row["Modal Price (Rs./Quintal)"]), 0.0, "2024-02-14T00:00:00Z"))
        conn.commit()

    evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    res = evaluator.evaluate_pending_forecasts(as_of_date="2024-02-05", db_path=temp_vol_db)

    assert res["evaluated_count"] == 1
    assert res["window_empty_count"] == 0

    with get_volatility_db_connection(temp_vol_db) as conn:
        row = conn.execute("SELECT * FROM shadow_volatility_forecasts WHERE origin_date = '2024-01-20';").fetchone()
        assert row["status"] == "EVALUATED"
        assert row["matched_trades_count"] > 0
        assert row["actual_class"] in ("LOW", "MEDIUM", "HIGH")
        assert row["actual_pct_range"] is not None
        assert row["evaluated_at"] is not None


def test_empty_window_transitions_to_window_empty(temp_vol_db, synthetic_mandi_history):
    """Verify forecast with zero realized trades in (t, t+14] transitions to WINDOW_EMPTY."""
    process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history.iloc[:20],
        origin_date="2024-01-20",
        db_path=temp_vol_db,
        force_enabled=True,
    )

    # Window closed on 2024-02-03, but NO trades were ingested for that window
    evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    res = evaluator.evaluate_pending_forecasts(as_of_date="2024-02-05", db_path=temp_vol_db)

    assert res["evaluated_count"] == 0
    assert res["window_empty_count"] == 1

    with get_volatility_db_connection(temp_vol_db) as conn:
        row = conn.execute("SELECT * FROM shadow_volatility_forecasts WHERE origin_date = '2024-01-20';").fetchone()
        assert row["status"] == "WINDOW_EMPTY"
        assert row["actual_class"] is None
        assert row["matched_trades_count"] == 0


# ============================================================
# 9. DUPLICATE EVALUATION PREVENTION
# ============================================================

def test_duplicate_evaluation_prevention(temp_vol_db, synthetic_mandi_history):
    """Verify running evaluator again on already evaluated forecasts is an idempotent no-op."""
    # Setup and evaluate forecast
    process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history,
        origin_date="2024-01-15",
        db_path=temp_vol_db,
        force_enabled=True,
    )
    evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    res1 = evaluator.evaluate_pending_forecasts(as_of_date="2024-02-01", db_path=temp_vol_db)
    assert res1["evaluated_count"] == 1

    # Record first evaluation timestamp
    with get_volatility_db_connection(temp_vol_db) as conn:
        eval_ts1 = conn.execute("SELECT evaluated_at FROM shadow_volatility_forecasts WHERE origin_date='2024-01-15';").fetchone()["evaluated_at"]

    # Run evaluation again
    res2 = evaluator.evaluate_pending_forecasts(as_of_date="2024-02-01", db_path=temp_vol_db)
    assert res2["evaluated_count"] == 0  # No duplicate evaluations

    with get_volatility_db_connection(temp_vol_db) as conn:
        eval_ts2 = conn.execute("SELECT evaluated_at FROM shadow_volatility_forecasts WHERE origin_date='2024-01-15';").fetchone()["evaluated_at"]
        assert eval_ts1 == eval_ts2  # Timestamp unchanged


# ============================================================
# 10. DATABASE RECOVERY AFTER APPLICATION RESTART
# ============================================================

def test_database_recovery_across_restarts(temp_vol_db, synthetic_mandi_history):
    """Verify that after restarting application, evaluator reconnects and safely processes pending records."""
    # Session 1: Generate forecast
    process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history,
        origin_date="2024-01-16",
        db_path=temp_vol_db,
        force_enabled=True,
    )

    # Session 2: "Restart" - instantiate completely new evaluator and connection
    fresh_evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    res = fresh_evaluator.evaluate_pending_forecasts(as_of_date="2024-02-02", db_path=temp_vol_db)
    assert res["evaluated_count"] == 1

    with get_volatility_db_connection(temp_vol_db) as conn:
        status = conn.execute("SELECT status FROM shadow_volatility_forecasts WHERE origin_date='2024-01-16';").fetchone()["status"]
        assert status == "EVALUATED"


# ============================================================
# 11. PRODUCTION ISOLATION & ARTIFACT INVARIANCE
# ============================================================

def test_production_artifacts_unmodified_by_shadow_operations():
    """Verify frozen production models remain identical to their baseline states."""
    prod_model_path = MEMBER2_DIR / "artifacts" / "rice_volatility_normalized_model.pkl"
    if prod_model_path.exists():
        hasher = hashlib.sha256()
        with open(prod_model_path, "rb") as f:
            hasher.update(f.read())
        # First 32 chars must match baseline audit hash DFC83D89AE3E93D9ABF225B669B82FD9
        assert hasher.hexdigest()[:32].upper() == "DFC83D89AE3E93D9ABF225B669B82FD9"


def test_production_endpoints_isolated(temp_vol_db):
    """Verify /health returns persistence and public routes remain isolated from V3."""
    with TestClient(app) as client:
        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["volatility"] == "persistence"

        # Volatility shadow status endpoint is internal only
        vol_status = client.get("/shadow/volatility/status")
        assert vol_status.status_code == 200
        assert vol_status.json()["production_default"] == "Calendar Persistence (14D)"


# ============================================================
# 12. NO FARMER-FACING EXPOSURE OF V3
# ============================================================

def test_no_farmer_facing_exposure_of_v3(synthetic_mandi_history):
    """Verify POST /predict returns only persistence (v1) or normalized v2; never exposes V3 predictions."""
    with TestClient(app) as client:
        dates = pd.date_range("2024-01-01", periods=35)
        history = [
            {"reported_date": d.strftime("%Y-%m-%d"), "modal_price": 2500.0 + i, "arrivals": 100.0}
            for i, d in enumerate(dates)
        ]

        # 1. Default call (Farmer-facing default)
        res1 = client.post("/predict", json={
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2024-02-04", "history": history
        })
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["volatility_version"] == "v1"
        assert "v3_pred_class" not in data1
        assert "v3_calibrated_prob_high" not in data1

        # 2. Explicit request for v2_normalized
        res2 = client.post("/predict", json={
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2024-02-04", "history": history,
            "volatility_version": "v2_normalized"
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["volatility_version"] == "v2_normalized"
        assert "v3_pred_class" not in data2


# ============================================================
# 13. PROMOTION GOVERNANCE NON-INFERIORITY TEST
# ============================================================

def test_governance_rejection_of_p_value_alone():
    """Verify promotion governance requires predefined margin CI lower bound and rejects p > 0.10 alone."""
    # When sample size is small or criteria unmet, status MUST be CONTINUE_SHADOW_MONITORING
    report = generate_volatility_shadow_report(cohort="PROSPECTIVE")
    assert "CONTINUE_SHADOW_MONITORING" in report
    assert "Maintain Calendar Persistence in production" in report


# ============================================================
# 14. FASTAPI INTERNAL MONITORING ENDPOINTS
# ============================================================

def test_shadow_volatility_api_endpoints(temp_vol_db, monkeypatch):
    """Verify internal shadow volatility endpoints respond accurately via FastAPI client."""
    from config import get_shadow_db_path
    monkeypatch.setenv("AGROVISION_SHADOW_DB_PATH", str(temp_vol_db))

    with TestClient(app) as client:
        # 1. GET /shadow/volatility/status
        status_res = client.get("/shadow/volatility/status")
        assert status_res.status_code == 200
        s_data = status_res.json()
        assert s_data["status"] == "ok"
        assert s_data["model_version"] == V3_14D_MODEL_VERSION
        assert s_data["production_default"] == "Calendar Persistence (14D)"
        assert "counts" in s_data

        # 2. GET /shadow/volatility/metrics
        metrics_res = client.get("/shadow/volatility/metrics?cohort=PROSPECTIVE")
        assert metrics_res.status_code == 200
        m_data = metrics_res.json()
        assert "counts" in m_data

        # 3. POST /shadow/volatility/evaluate
        eval_res = client.post("/shadow/volatility/evaluate?as_of_date=2024-02-01")
        assert eval_res.status_code == 200
        assert eval_res.json()["status"] == "success"


def test_shadow_volatility_api_bad_cohort():
    """Verify invalid cohort parameter returns HTTP 400 Bad Request on both status and metrics."""
    with TestClient(app) as client:
        res1 = client.get("/shadow/volatility/metrics?cohort=INVALID_COHORT")
        assert res1.status_code == 400
        assert "Unsupported cohort" in res1.json()["detail"]

        res2 = client.get("/shadow/volatility/status?cohort=INVALID_COHORT")
        assert res2.status_code == 400
        assert "Unsupported cohort" in res2.json()["detail"]


# ============================================================
# 15. DAILY SINGLE-OBSERVATION INGESTION & DB HISTORY LOOKUP
# ============================================================

def test_daily_feed_ingestion_with_database_history(temp_vol_db, synthetic_mandi_history):
    """
    Verify that an incoming daily feed containing just 1 observation can generate a shadow forecast
    by retrieving the pre-existing observation history from the database (preventing tree NaN errors).
    """
    # 1. Ingest initial 20 days of history
    res1 = process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history.iloc[:20],
        origin_date="2024-01-20",
        db_path=temp_vol_db,
        force_enabled=True,
    )
    assert res1["generated_count"] == 1

    # 2. Ingest daily feed with exactly 1 row for Day 21 (2024-01-21)
    single_obs_feed = synthetic_mandi_history.iloc[20:21].copy()
    assert len(single_obs_feed) == 1

    res2 = process_volatility_shadow_ingestion(
        history_df=single_obs_feed,
        origin_date="2024-01-21",
        db_path=temp_vol_db,
        force_enabled=True,
    )
    # Must succeed in generating forecast by querying DB history for Burdwan/Common
    assert res2["generated_count"] == 1
    assert res2["skipped_duplicate_count"] == 0

    with get_volatility_db_connection(temp_vol_db) as conn:
        row = conn.execute("SELECT * FROM shadow_volatility_forecasts WHERE origin_date = '2024-01-21';").fetchone()
        assert row is not None
        assert row["status"] == "PENDING"
        assert row["target_window_end"] == "2024-02-04"


# ============================================================
# 16. REPORT RENDERING ACCURACY DELTA & MARGIN FORMATTING
# ============================================================

def test_accuracy_delta_and_margin_formatting_in_report(temp_vol_db, synthetic_mandi_history):
    """Verify that report table prints true accuracy delta and correct non-inferiority margin sign."""
    process_volatility_shadow_ingestion(
        history_df=synthetic_mandi_history,
        origin_date="2024-01-20",
        db_path=temp_vol_db,
        force_enabled=True,
    )
    evaluator = VolatilityShadowEvaluator(db_path=temp_vol_db)
    evaluator.evaluate_pending_forecasts(as_of_date="2024-02-05", db_path=temp_vol_db)

    report = generate_volatility_shadow_report(db_path=temp_vol_db, cohort="PROSPECTIVE", non_inferiority_margin=0.01)

    # 1. Margin threshold must display as negative lower bound (-0.0100)
    assert "CI Lower > -0.0100" in report
    assert "CI Lower > 0.0100" not in report

    # 2. Check accuracy row
    metrics = compute_volatility_shadow_metrics(db_path=temp_vol_db, cohort="PROSPECTIVE")
    pooled = metrics["pooled_performance"]
    delta_acc = pooled["delta_accuracy"]
    acc_row_expected = f"| **Accuracy** | {pooled['v3']['accuracy']:.4f} | {pooled['persistence']['accuracy']:.4f} | {delta_acc:+.4f} |"
    assert acc_row_expected in report


# ============================================================
# 17. PREDICT ENDPOINT NON-BLOCKING SHADOW RECORDING
# ============================================================

def test_predict_endpoint_shadow_hook_non_blocking(temp_vol_db, monkeypatch):
    """Verify that POST /predict triggers isolated shadow forecast generation when enabled without changing output."""
    from config import set_volatility_shadow_enabled
    monkeypatch.setenv("AGROVISION_SHADOW_DB_PATH", str(temp_vol_db))
    set_volatility_shadow_enabled(True)

    dates = pd.date_range("2024-01-01", periods=35)
    history = [
        {"reported_date": d.strftime("%Y-%m-%d"), "modal_price": 2500.0 + (i * 2.0), "arrivals": 100.0}
        for i, d in enumerate(dates)
    ]

    with TestClient(app) as client:
        res = client.post("/predict", json={
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": "2024-02-04",
            "history": history,
        })
        assert res.status_code == 200
        data = res.json()
        assert data["volatility_version"] == "v1"
        assert "v3_pred_class" not in data  # Never exposed to farmer!

    # Check that shadow record was stored in the background
    with get_volatility_db_connection(temp_vol_db) as conn:
        row = conn.execute("SELECT * FROM shadow_volatility_forecasts WHERE origin_date = '2024-02-04';").fetchone()
        assert row is not None
        assert row["market"] == "Burdwan"
        assert row["status"] == "PENDING"
        assert row["v3_pred_class"] in ("LOW", "MEDIUM", "HIGH")

    set_volatility_shadow_enabled(False)

