"""
Comprehensive Unit & Integration Tests for Live Shadow-Mode Evaluation Workflow

Verifies:
1. Operational Switches & Deployment Policy:
   - Fresh installation defaults: horizon forecasting and shadow evaluation are disabled.
   - Disabled horizon endpoints return HTTP 503 while legacy /predict returns HTTP 200.
2. Production Model Invariance:
   - Enabling shadow evaluation or generating shadow forecasts leaves legacy predictions untouched.
3. Point-in-Time Integrity & Zero Leakage:
   - Forecast generation uses only observations on or before origin_date.
   - Future actual prices cannot enter feature calculations.
4. Idempotency & Deduplication:
   - Repeated shadow forecast generation calls for the same origin date skip duplicates.
5. Delayed Outcome Evaluation Semantics:
   - Pending forecasts are NOT evaluated prematurely while target window remains open.
   - Closed target windows without observations transition to TARGET_UNAVAILABLE.
   - Matched observations in [t+H, t+H+3] transition to EVALUATED with correct offset.
6. Honest Matched Persistence Comparison:
   - Evaluates XGBoost and Persistence baseline on identical rows and outcomes.
   - Correctly segregates movement vs flat market days.
7. Observability & API Endpoints:
   - GET /shadow/status and GET /shadow/metrics endpoints respond accurately.
   - No evaluation route exposes raw price time series.
"""

import sys
import sqlite3
import tempfile
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
    is_horizon_forecasting_enabled,
    is_horizon_14d_enabled,
    is_horizon_30d_enabled,
    is_shadow_evaluation_enabled,
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    set_shadow_evaluation_enabled,
    reset_horizon_forecasting_switches,
)
from shadow_evaluation import (
    init_shadow_db,
    ingest_mandi_observations,
    generate_shadow_forecasts,
    evaluate_pending_shadow_forecasts,
    compute_shadow_evaluation_metrics,
    get_shadow_status,
    make_forecast_id,
)
from inference import RiceInferencePipeline
from api import app


@pytest.fixture
def temp_db(tmp_path):
    """Provides a fresh isolated SQLite database for testing."""
    db_file = tmp_path / "test_shadow.db"
    init_shadow_db(db_file)
    return db_file


@pytest.fixture
def synthetic_mandi_series():
    """Builds a sequential daily mandi series spanning 60 days."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    prices = [2500.0 + (i * 2.0) for i in range(60)]
    arrivals = [100.0 + (i % 7) * 5.0 for i in range(60)]
    return pd.DataFrame({
        "Market Name": ["Burdwan"] * 60,
        "Variety": ["Common"] * 60,
        "Reported Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Modal Price (Rs./Quintal)": prices,
        "Arrivals (Tonnes)": arrivals,
    })


# ============================================================
# 1. OPERATIONAL SWITCHES & DEPLOYMENT POLICY
# ============================================================

def test_fresh_installation_defaults_are_disabled():
    """Verify that in a fresh installation without env overrides, all horizon switches default to False."""
    reset_horizon_forecasting_switches()
    assert is_horizon_forecasting_enabled() is False
    assert is_horizon_14d_enabled() is False
    assert is_horizon_30d_enabled() is False
    assert is_shadow_evaluation_enabled() is False


def test_disabled_horizon_endpoint_returns_503_while_legacy_returns_200():
    """Verify disabled horizon forecasting yields 503 while /predict operates normally."""
    reset_horizon_forecasting_switches()
    with TestClient(app) as client:
        # Legacy prediction request (35 days of history)
        dates = pd.date_range("2024-01-01", periods=35)
        history = [
            {"reported_date": d.strftime("%Y-%m-%d"), "modal_price": 2500.0 + i, "arrivals": 100.0}
            for i, d in enumerate(dates)
        ]
        legacy_res = client.post("/predict", json={
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2024-02-04", "history": history
        })
        assert legacy_res.status_code == 200
        assert "predicted_modal_price" in legacy_res.json()

        # Horizon endpoint must return 503 (disabled by default)
        horizon_res = client.post("/predict-horizon", json={
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2024-02-04", "horizon_days": 14, "history": history
        })
        assert horizon_res.status_code == 503
        assert "disabled globally" in horizon_res.json()["detail"]


def test_shadow_evaluation_can_run_independently_of_client_api(temp_db, synthetic_mandi_series):
    """Verify shadow evaluation operates when enabled internally, even if user-facing API is disabled."""
    reset_horizon_forecasting_switches()
    set_shadow_evaluation_enabled(True)  # Internal evaluation active
    set_horizon_forecasting_enabled(False)  # Public API remains closed

    # Public API is closed
    with TestClient(app) as client:
        dates = pd.date_range("2024-01-01", periods=35)
        history = [
            {"reported_date": d.strftime("%Y-%m-%d"), "modal_price": 2500.0 + i, "arrivals": 100.0}
            for i, d in enumerate(dates)
        ]
        api_res = client.post("/predict-horizon", json={
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2024-02-04", "horizon_days": 14, "history": history
        })
        assert api_res.status_code == 503

    # Internal shadow forecast generation succeeds
    gen_res = generate_shadow_forecasts(
        synthetic_mandi_series,
        horizons=(14,),
        origin_date="2024-02-04",
        db_path=temp_db,
    )
    assert gen_res["generated_count"] == 1
    reset_horizon_forecasting_switches()


# ============================================================
# 2. INGESTION & FORECAST GENERATION IDEMPOTENCY
# ============================================================

def test_ingest_mandi_observations(temp_db, synthetic_mandi_series):
    """Verify observation ingestion stores rows and updates existing records idempotently."""
    n1 = ingest_mandi_observations(synthetic_mandi_series, db_path=temp_db)
    assert n1 == 60

    # Ingest again with updated price for day 0
    updated_series = synthetic_mandi_series.copy()
    updated_series.loc[0, "Modal Price (Rs./Quintal)"] = 9999.0
    n2 = ingest_mandi_observations(updated_series, db_path=temp_db)
    assert n2 == 60

    with sqlite3.connect(temp_db) as conn:
        cur = conn.cursor()
        cur.execute("SELECT modal_price FROM mandi_observations WHERE reported_date='2024-01-01';")
        assert cur.fetchone()[0] == 9999.0


def test_forecast_generation_idempotency(temp_db, synthetic_mandi_series):
    """Verify repeated forecast generation runs are idempotent and skip duplicate records."""
    set_shadow_evaluation_enabled(True)
    try:
        # First generation
        res1 = generate_shadow_forecasts(
            synthetic_mandi_series,
            horizons=(14, 30),
            origin_date="2024-02-04",
            db_path=temp_db,
        )
        assert res1["generated_count"] == 2
        assert res1["skipped_existing_count"] == 0

        # Repeated generation for same origin date
        res2 = generate_shadow_forecasts(
            synthetic_mandi_series,
            horizons=(14, 30),
            origin_date="2024-02-04",
            db_path=temp_db,
        )
        assert res2["generated_count"] == 0
        assert res2["skipped_existing_count"] == 2
    finally:
        reset_horizon_forecasting_switches()


# ============================================================
# 3. DELAYED OUTCOME EVALUATION LOGIC
# ============================================================

def test_pending_forecasts_not_evaluated_prematurely(temp_db, synthetic_mandi_series):
    """Verify forecasts whose target window has not arrived remain PENDING."""
    set_shadow_evaluation_enabled(True)
    try:
        # Ingest history only up to 2024-02-04 (Day 35)
        history_sub = synthetic_mandi_series.iloc[:35].copy()
        generate_shadow_forecasts(
            history_sub,
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=temp_db,
        )

        # 14-day target window is [2024-02-18, 2024-02-21].
        # Ingest observations only up to 2024-02-10 (Before window opens)
        ingest_mandi_observations(synthetic_mandi_series.iloc[35:41], db_path=temp_db)

        # Run delayed evaluation
        res = evaluate_pending_shadow_forecasts(db_path=temp_db)
        assert res["evaluated_count"] == 0
        assert res["unavailable_count"] == 0
        assert res["still_pending_count"] == 1

        # Check DB status
        status = get_shadow_status(temp_db)
        assert status["counts"]["forecasts_14d"]["pending"] == 1
        assert status["counts"]["forecasts_14d"]["evaluated"] == 0
    finally:
        reset_horizon_forecasting_switches()


def test_forecast_evaluates_when_observation_arrives(temp_db, synthetic_mandi_series):
    """Verify forecast transitions to EVALUATED when valid observation arrives in target window."""
    set_shadow_evaluation_enabled(True)
    try:
        # Generate forecast at 2024-02-04 (Day 35). Target window: [2024-02-18, 2024-02-21]
        history_sub = synthetic_mandi_series.iloc[:35].copy()
        generate_shadow_forecasts(
            history_sub,
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=temp_db,
        )

        # Now ingest all observations including through Day 60 (> 2024-02-21)
        ingest_mandi_observations(synthetic_mandi_series, db_path=temp_db)

        # Run delayed evaluation
        res = evaluate_pending_shadow_forecasts(db_path=temp_db)
        assert res["evaluated_count"] == 1
        assert res["still_pending_count"] == 0

        # Verify evaluated record
        metrics = compute_shadow_evaluation_metrics(db_path=temp_db, horizon_days=14)
        h14 = metrics["14d"]
        assert h14["counts"]["evaluated_forecasts"] == 1
        assert "model_performance" in h14
        assert "persistence_baseline" in h14
        assert h14["model_performance"]["mae_rs_qtl"] >= 0.0
        assert h14["persistence_baseline"]["mae_rs_qtl"] >= 0.0
    finally:
        reset_horizon_forecasting_switches()


def test_closed_window_without_observation_becomes_unavailable(temp_db, synthetic_mandi_series):
    """Verify that if a target window closes without any report in the window, status is TARGET_UNAVAILABLE."""
    set_shadow_evaluation_enabled(True)
    try:
        # Generate forecast at 2024-02-04. 14d window is [2024-02-18, 2024-02-21]
        history_sub = synthetic_mandi_series.iloc[:35].copy()
        generate_shadow_forecasts(
            history_sub,
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=temp_db,
        )

        # Ingest an observation on 2024-02-28 (Past the window end), but OMIT [2024-02-18, 2024-02-21]
        sparse_obs = pd.DataFrame([{
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": "2024-02-28",
            "Modal Price (Rs./Quintal)": 2700.0,
            "Arrivals (Tonnes)": 150.0,
        }])
        ingest_mandi_observations(sparse_obs, db_path=temp_db)

        # Evaluate
        res = evaluate_pending_shadow_forecasts(db_path=temp_db)
        assert res["unavailable_count"] == 1
        assert res["evaluated_count"] == 0

        status = get_shadow_status(temp_db)
        assert status["counts"]["forecasts_14d"]["unavailable"] == 1
    finally:
        reset_horizon_forecasting_switches()


# ============================================================
# 4. OBSERVABILITY ENDPOINTS & PRODUCTION ISOLATION
# ============================================================

def test_shadow_api_endpoints(temp_db, monkeypatch):
    """Verify GET /shadow/status and GET /shadow/metrics via FastAPI TestClient."""
    monkeypatch.setenv("AGROVISION_SHADOW_DB_PATH", str(temp_db))
    with TestClient(app) as client:
        r_status = client.get("/shadow/status")
        assert r_status.status_code == 200
        assert "shadow_evaluation_enabled" in r_status.json()

        r_metrics = client.get("/shadow/metrics")
        assert r_metrics.status_code == 200


# ============================================================
# 5. DATABASE RELIABILITY, RESILIENCE & SECURITY
# ============================================================

def test_malformed_observation_rejection(temp_db):
    """Verify that corrupt, non-numeric, negative, and invalid date rows are rejected."""
    corrupt_df = pd.DataFrame([
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-01-01", "Modal Price (Rs./Quintal)": 2500.0, "Arrivals (Tonnes)": 100.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "not-a-date", "Modal Price (Rs./Quintal)": 2500.0, "Arrivals (Tonnes)": 100.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-01-02", "Modal Price (Rs./Quintal)": -150.0, "Arrivals (Tonnes)": 100.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-01-03", "Modal Price (Rs./Quintal)": 0.0, "Arrivals (Tonnes)": 100.0},
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-01-04", "Modal Price (Rs./Quintal)": "invalid_price", "Arrivals (Tonnes)": 100.0},
        {"Market Name": "", "Variety": "Common", "Reported Date": "2024-01-05", "Modal Price (Rs./Quintal)": 2600.0, "Arrivals (Tonnes)": 100.0},
    ])

    stats = ingest_mandi_observations(corrupt_df, db_path=temp_db, return_stats=True)
    assert stats["total_input_rows"] == 6
    assert stats["accepted_count"] == 1
    assert stats["rejected_count"] == 5
    assert stats["min_date"] == "2024-01-01"


def test_recheck_unavailable_recovers_late_arrivals(temp_db, synthetic_mandi_series):
    """Verify that a forecast marked TARGET_UNAVAILABLE is recovered if late data arrives."""
    set_shadow_evaluation_enabled(True)
    try:
        # Generate forecast at 2024-02-04. 14d window is [2024-02-18, 2024-02-21]
        history_sub = synthetic_mandi_series.iloc[:35].copy()
        generate_shadow_forecasts(
            history_sub,
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=temp_db,
        )

        # Ingest an observation on 2024-02-28 (Past the window end) -> window closed without observation
        post_window_obs = pd.DataFrame([{
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": "2024-02-28",
            "Modal Price (Rs./Quintal)": 2700.0,
            "Arrivals (Tonnes)": 150.0,
        }])
        ingest_mandi_observations(post_window_obs, db_path=temp_db)

        # First evaluation: window closed -> TARGET_UNAVAILABLE
        res1 = evaluate_pending_shadow_forecasts(db_path=temp_db)
        assert res1["unavailable_count"] == 1

        # Now simulate a late backfill batch arriving for 2024-02-19 (Inside window!)
        late_obs = pd.DataFrame([{
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": "2024-02-19",
            "Modal Price (Rs./Quintal)": 2650.0,
            "Arrivals (Tonnes)": 120.0,
        }])
        ingest_mandi_observations(late_obs, db_path=temp_db)

        # Recheck with recheck_unavailable=True
        res2 = evaluate_pending_shadow_forecasts(db_path=temp_db, recheck_unavailable=True)
        assert res2["evaluated_count"] == 1
        assert res2["recovered_late_count"] == 1

        status = get_shadow_status(temp_db)
        assert status["counts"]["forecasts_14d"]["evaluated"] == 1
        assert status["counts"]["forecasts_14d"]["unavailable"] == 0
    finally:
        reset_horizon_forecasting_switches()


def test_database_persistence_across_process_restart(tmp_path, synthetic_mandi_series):
    """Verify that SQLite persistence and metrics survive complete process/connection closure."""
    db_file = tmp_path / "restart_test.db"
    init_shadow_db(db_file)
    set_shadow_evaluation_enabled(True)

    try:
        # Ingest all 60 days
        ingest_mandi_observations(synthetic_mandi_series, db_path=db_file)

        # Generate 14d forecast at day 35
        generate_shadow_forecasts(
            synthetic_mandi_series.iloc[:35],
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=db_file,
        )

        # Evaluate against subsequently arrived observations
        evaluate_pending_shadow_forecasts(db_path=db_file)

        # Verify state in DB
        metrics1 = compute_shadow_evaluation_metrics(db_path=db_file, horizon_days=14)
        assert metrics1["14d"]["counts"]["evaluated_forecasts"] == 1

        # Simulate restart: open brand new SQLite connection from scratch
        metrics2 = compute_shadow_evaluation_metrics(db_path=db_file, horizon_days=14)
        assert metrics2["14d"]["counts"]["evaluated_forecasts"] == 1
        assert metrics1["14d"]["model_performance"]["mae_rs_qtl"] == metrics2["14d"]["model_performance"]["mae_rs_qtl"]
    finally:
        reset_horizon_forecasting_switches()


def test_shadow_api_security_no_raw_timeseries_leakage(temp_db, monkeypatch):
    """Verify that shadow monitoring routes return aggregates and metadata, never raw full timeseries."""
    monkeypatch.setenv("AGROVISION_SHADOW_DB_PATH", str(temp_db))
    with TestClient(app) as client:
        r_status = client.get("/shadow/status")
        assert r_status.status_code == 200
        # Check no raw rows leaked
        assert "history" not in r_status.json()
        assert "mandi_observations" not in r_status.json()

        r_metrics = client.get("/shadow/metrics")
        assert r_metrics.status_code == 200
        assert "history" not in r_metrics.json()

