"""
Unit & Integration Tests for Prospective Live Shadow Data Connector

Verifies:
1. Canonical market and variety mapping (exact, alias, and unmapped label tracking).
2. DataGovInConnector diagnostics and graceful keyless fallback behavior.
3. Record normalization, validation, corruption rejection, and audit reporting.
4. Data freshness auditing (reporting lag vs allowable threshold).
5. Statistical distribution drift detection against historical training baseline.
6. Single-instance process locking (ShadowExecutionLock) and stale-lock recovery.
7. Cohort isolation between prospective and retrospective shadow forecasts.
8. Observability API cohort filtering and schema validation.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from data_connector import (
    map_market_name,
    map_variety_name,
    DataGovInConnector,
    DataConnectorAuthError,
    normalize_and_map_records,
    audit_data_freshness_and_drift,
    ShadowExecutionLock,
    LockAcquisitionError,
    HISTORICAL_TRAINING_BASELINE,
    VARIETY_TRAINING_BASELINES,
    MARKET_TRAINING_BASELINES,
)
from shadow_evaluation import (
    init_shadow_db,
    ingest_mandi_observations,
    generate_shadow_forecasts,
    compute_shadow_evaluation_metrics,
    get_shadow_status,
    record_audit_log_entry,
    backup_shadow_database,
)
from config import (
    set_shadow_evaluation_enabled,
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    reset_horizon_forecasting_switches,
)
from api import app


# ============================================================
# 1. CANONICAL MARKET & VARIETY MAPPINGS
# ============================================================

def test_market_canonical_and_alias_mapping():
    """Verify exact canonical, alias, and unmapped market recognition."""
    # Canonical exact match
    m, known = map_market_name("Burdwan")
    assert known is True
    assert m == "Burdwan"

    # Known Agmarknet alias
    m, known = map_market_name("bardhaman")
    assert known is True
    assert m == "Burdwan"

    m, known = map_market_name("diamond harbour")
    assert known is True
    assert m == "Diamond Harbour(South 24-pgs)"

    # Case-insensitive canonical match
    m, known = map_market_name("kalna")
    assert known is True
    assert m == "Kalna"

    # Unmapped novel market
    m, known = map_market_name("Unmapped Mandi New")
    assert known is False
    assert m == "Unmapped Mandi New"


def test_variety_canonical_and_alias_mapping():
    """Verify exact canonical, alias, and unmapped variety recognition."""
    # Exact canonical match
    v, known = map_variety_name("Common")
    assert known is True
    assert v == "Common"

    # Agmarknet feed aliases
    v, known = map_variety_name("paddy (dhan) (common)")
    assert known is True
    assert v == "Common"

    v, known = map_variety_name("paddy(dhan)(fine)")
    assert known is True
    assert v == "Fine"

    v, known = map_variety_name("sona masoori")
    assert known is True
    assert v == "Sona Mansoori Non Basmati"

    # Unmapped variety
    v, known = map_variety_name("Exotic Basmati 1121")
    assert known is False
    assert v == "Exotic Basmati 1121"


# ============================================================
# 2. OFFICIAL DATA.GOV.IN CONNECTOR DIAGNOSTICS
# ============================================================

def test_datagovin_connector_unauthenticated_diagnostics(monkeypatch):
    """Verify data.gov.in connector behavior when no API key is configured."""
    monkeypatch.delenv("DATA_GOV_IN_API_KEY", raising=False)
    monkeypatch.delenv("AGMARKNET_API_KEY", raising=False)

    connector = DataGovInConnector(api_key=None)
    diag = connector.check_connection()

    assert diag["accessible"] is False
    assert diag["auth_configured"] is False
    assert diag["status_code"] == 400
    assert diag["error_category"] == "missing_api_key"
    assert "Authorization key missing" in diag["message"]
    assert "9ef84268-d588-465a-a308-a864a43d0070" in diag["resource_id"]

    # Direct fetch should raise DataConnectorAuthError
    with pytest.raises(DataConnectorAuthError) as exc_info:
        connector.fetch_records()
    assert "DATA_GOV_IN_API_KEY is not set" in str(exc_info.value)


# ============================================================
# 3. NORMALIZATION & VALIDATION PIPELINE
# ============================================================

def test_normalize_and_map_records_rejects_corrupt_data():
    """Verify cleaning drops invalid dates, non-positive prices, and tracks unmapped labels."""
    raw_data = pd.DataFrame([
        # Valid row 1
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-02-01", "Modal Price (Rs./Quintal)": 2500.0, "Arrivals (Tonnes)": 100.0},
        # Valid row 2 (alias variety & alias market)
        {"Market Name": "bardhaman", "Variety": "paddy (dhan) (common)", "Reported Date": "2024-02-02", "Modal Price (Rs./Quintal)": 2520.0, "Arrivals (Tonnes)": 110.0},
        # Corrupt row 1: unparseable date
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "not-a-date", "Modal Price (Rs./Quintal)": 2500.0, "Arrivals (Tonnes)": 50.0},
        # Corrupt row 2: zero price
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-02-03", "Modal Price (Rs./Quintal)": 0.0, "Arrivals (Tonnes)": 50.0},
        # Corrupt row 3: negative price
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-02-04", "Modal Price (Rs./Quintal)": -50.0, "Arrivals (Tonnes)": 50.0},
        # Unmapped novel market row (valid date and price)
        {"Market Name": "Unknown Mandi Z", "Variety": "Novel Rice Type", "Reported Date": "2024-02-01", "Modal Price (Rs./Quintal)": 3000.0, "Arrivals (Tonnes)": 20.0},
        # Duplicate row of row 1 (should overwrite / dedup)
        {"Market Name": "Burdwan", "Variety": "Common", "Reported Date": "2024-02-01", "Modal Price (Rs./Quintal)": 2510.0, "Arrivals (Tonnes)": 105.0},
    ])

    clean_df, audit = normalize_and_map_records(raw_data, source_tag="test_feed")

    assert audit["raw_input_rows"] == 7
    assert audit["rejected_invalid_dates"] == 1
    assert audit["rejected_invalid_prices"] == 2
    assert audit["rejected_corrupt_rows"] == 3
    assert audit["batch_duplicates_resolved"] == 1
    assert audit["accepted_valid_rows"] == 3

    assert "Unknown Mandi Z" in audit["unmapped_mandi_labels"]
    assert "Novel Rice Type" in audit["unmapped_variety_labels"]

    # Verify column structure
    expected_cols = ["market", "variety", "reported_date", "modal_price", "arrivals", "source_id", "ingestion_timestamp"]
    assert list(clean_df.columns) == expected_cols

    # Verify alias mapping succeeded
    burdwan_rows = clean_df[clean_df["market"] == "Burdwan"]
    assert len(burdwan_rows) == 2
    assert all(burdwan_rows["variety"] == "Common")


# ============================================================
# 4. DATA FRESHNESS & DISTRIBUTION DRIFT AUDIT
# ============================================================

def test_data_freshness_audit_fresh_and_stale():
    """Verify reporting freshness detection against calendar thresholds."""
    ref_date = "2024-02-15"

    # Fresh batch: 3 days lag (<= 14)
    fresh_df = pd.DataFrame({
        "market": ["Burdwan"], "variety": ["Common"], "reported_date": ["2024-02-12"], "modal_price": [2500.0]
    })
    audit_fresh = audit_data_freshness_and_drift(fresh_df, reference_date=ref_date, max_allowable_lag_days=14)
    assert audit_fresh["freshness"]["is_stale"] is False
    assert audit_fresh["freshness"]["reporting_lag_days"] == 3

    # Stale batch: 20 days lag (> 14)
    stale_df = pd.DataFrame({
        "market": ["Burdwan"], "variety": ["Common"], "reported_date": ["2024-01-26"], "modal_price": [2500.0]
    })
    audit_stale = audit_data_freshness_and_drift(stale_df, reference_date=ref_date, max_allowable_lag_days=14)
    assert audit_stale["freshness"]["is_stale"] is True
    assert audit_stale["freshness"]["reporting_lag_days"] == 20
    assert "STALE DATA WARNING" in audit_stale["freshness"]["message"]


def test_distribution_drift_detection():
    """Verify statistical drift flags large shifts from historical training baseline."""
    # Nominal batch close to historical baseline median ~2380
    nominal_df = pd.DataFrame({
        "market": ["Burdwan"] * 20, "variety": ["Common"] * 20,
        "reported_date": ["2024-02-01"] * 20,
        "modal_price": np.random.normal(2450.0, 100.0, 20),
    })
    audit_nominal = audit_data_freshness_and_drift(nominal_df)
    assert audit_nominal["drift"]["drift_detected"] is False

    # Shifted batch: median Rs.4200 (+76% above baseline 2380)
    drifted_df = pd.DataFrame({
        "market": ["Burdwan"] * 20, "variety": ["Common"] * 20,
        "reported_date": ["2024-02-01"] * 20,
        "modal_price": np.random.normal(4200.0, 100.0, 20),
    })
    audit_drift = audit_data_freshness_and_drift(drifted_df)
    assert audit_drift["drift"]["drift_detected"] is True
    assert "DRIFT DETECTED" in audit_drift["drift"]["message"]
    assert audit_drift["drift"]["median_pct_shift"] > 35.0


# ============================================================
# 5. PROCESS LOCKING & CONCURRENCY CONTROL
# ============================================================

def test_shadow_execution_lock_mutex(tmp_path):
    """Verify mutual exclusion prevents concurrent daily executions."""
    lock_file = tmp_path / "test_exec.lock"

    lock1 = ShadowExecutionLock(lock_path=lock_file)
    lock2 = ShadowExecutionLock(lock_path=lock_file)

    # First acquisition succeeds
    assert lock1.acquire() is True
    assert lock_file.exists()

    # Second acquisition by same or concurrent process raises LockAcquisitionError
    with pytest.raises(LockAcquisitionError) as exc_info:
        lock2.acquire()
    assert "Active lock held by PID" in str(exc_info.value)

    # Clean release allows acquisition
    lock1.release()
    assert not lock_file.exists()
    assert lock2.acquire() is True
    lock2.release()


def test_shadow_execution_stale_lock_recovery(tmp_path):
    """Verify expired/stale lock files are cleanly reclaimed."""
    lock_file = tmp_path / "stale_exec.lock"
    # Write lock with fake PID and old timestamp (2 hours old)
    old_payload = {
        "pid": 999999,
        "timestamp": time.time() - 7200,
        "time_iso": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
    }
    lock_file.write_text(json.dumps(old_payload), encoding="utf-8")

    # With stale_timeout_sec=3600 (1 hr), lock should be recognized as stale and acquired
    lock = ShadowExecutionLock(lock_path=lock_file, stale_timeout_sec=3600)
    assert lock.acquire() is True
    lock.release()


# ============================================================
# 6. COHORT ISOLATION IN PERSISTENCE & EVALUATION
# ============================================================

def test_prospective_and_retrospective_cohort_isolation(tmp_path):
    """Verify that prospective and retrospective shadow forecasts are strictly isolated."""
    db_file = tmp_path / "cohort_test.db"
    init_shadow_db(db_file)
    set_shadow_evaluation_enabled(True)
    set_horizon_forecasting_enabled(True)
    set_horizon_14d_enabled(True)
    set_horizon_30d_enabled(True)

    try:
        # Build 60 days of synthetic data
        dates = pd.date_range("2024-01-01", periods=60, freq="D")
        obs_df = pd.DataFrame({
            "Market Name": ["Burdwan"] * 60,
            "Variety": ["Common"] * 60,
            "Reported Date": [d.strftime("%Y-%m-%d") for d in dates],
            "Modal Price (Rs./Quintal)": [2500.0 + i for i in range(60)],
            "Arrivals (Tonnes)": [100.0] * 60,
        })
        ingest_mandi_observations(obs_df, db_path=db_file)

        # Generate RETROSPECTIVE forecast at day 35
        gen_ret = generate_shadow_forecasts(
            obs_df.iloc[:35],
            horizons=(14,),
            origin_date="2024-02-04",
            db_path=db_file,
            force_enabled=True,
            cohort="RETROSPECTIVE",
        )
        assert gen_ret["generated_count"] == 1

        # Generate PROSPECTIVE forecast at day 40
        gen_pro = generate_shadow_forecasts(
            obs_df.iloc[:40],
            horizons=(14,),
            origin_date="2024-02-09",
            db_path=db_file,
            force_enabled=True,
            cohort="PROSPECTIVE",
        )
        assert gen_pro["generated_count"] == 1

        # Check status contains cohort breakdown
        status = get_shadow_status(db_file)
        assert "by_cohort" in status["counts"]
        assert status["counts"]["by_cohort"]["retrospective"]["total"] == 1
        assert status["counts"]["by_cohort"]["prospective"]["total"] == 1

        # Check cohort filtering in evaluation metrics
        m_ret = compute_shadow_evaluation_metrics(db_path=db_file, horizon_days=14, cohort="RETROSPECTIVE")
        assert m_ret["14d"]["counts"]["total_forecasts"] == 1

        m_pro = compute_shadow_evaluation_metrics(db_path=db_file, horizon_days=14, cohort="PROSPECTIVE")
        assert m_pro["14d"]["counts"]["total_forecasts"] == 1
    finally:
        reset_horizon_forecasting_switches()


# ============================================================
# 7. OBSERVABILITY API COHORT FILTERING
# ============================================================

def test_api_shadow_cohort_filtering(tmp_path, monkeypatch):
    """Verify FastAPI GET /shadow/metrics filters by cohort and rejects invalid cohort."""
    db_file = tmp_path / "api_cohort.db"
    init_shadow_db(db_file)
    monkeypatch.setenv("AGROVISION_SHADOW_DB_PATH", str(db_file))

    with TestClient(app) as client:
        # Valid cohort query
        r_pro = client.get("/shadow/metrics?cohort=PROSPECTIVE")
        assert r_pro.status_code == 200

        r_ret = client.get("/shadow/metrics?cohort=RETROSPECTIVE")
        assert r_ret.status_code == 200

        r_all = client.get("/shadow/metrics?cohort=ALL")
        assert r_all.status_code == 200

        # Invalid cohort query returns 400
        r_inv = client.get("/shadow/metrics?cohort=INVALID_COHORT")
        assert r_inv.status_code == 400
        assert "Unsupported cohort" in r_inv.json()["detail"]


# ============================================================
# 8. GROUP-AWARE DRIFT & DATA QUALITY GATING
# ============================================================

def test_group_aware_variety_drift_avoids_composition_false_alarm():
    """Verify that a batch of higher-tier Super Fine rice does NOT trigger a false positive drift alert."""
    # 20 observations of Super Fine at price Rs. 3400
    # Global median = 3400 (+42.9% vs global baseline 2380), but only +9.7% vs Super Fine variety baseline (3100)
    # and +21.4% vs Jhargram baseline (2800)
    batch_df = pd.DataFrame({
        "market": ["Jhargram"] * 20,
        "variety": ["Super Fine"] * 20,
        "reported_date": ["2024-02-01"] * 20,
        "modal_price": [3400.0] * 20,
    })
    audit = audit_data_freshness_and_drift(batch_df)
    assert audit["drift"]["drift_detected"] is False
    assert audit["drift"]["composition_shift_only"] is True
    assert "COMPOSITION SHIFT OBSERVED" in audit["drift"]["message"]


def test_group_aware_drift_detects_real_group_shock():
    """Verify that a legitimate price shock in a specific variety triggers group drift detection."""
    # 20 observations of Common rice at Rs. 3900 (+56% vs Common baseline 2500)
    batch_df = pd.DataFrame({
        "market": ["Burdwan"] * 20,
        "variety": ["Common"] * 20,
        "reported_date": ["2024-02-01"] * 20,
        "modal_price": [3900.0] * 20,
    })
    audit = audit_data_freshness_and_drift(batch_df)
    assert audit["drift"]["drift_detected"] is True
    assert len(audit["drift"]["group_drift_anomalies"]) > 0
    assert "Common" in audit["drift"]["group_drift_anomalies"][0]


def test_empty_batch_handling_in_audit():
    """Verify that an empty DataFrame safely returns empty_batch status without raising exceptions."""
    empty_df = pd.DataFrame(columns=["market", "variety", "reported_date", "modal_price"])
    audit = audit_data_freshness_and_drift(empty_df)
    assert audit["status"] == "empty_batch"
    assert audit["freshness"]["is_stale"] is True
    assert audit["drift"]["drift_detected"] is False


def test_credential_sanitization_in_connector_messages():
    """Verify that DataGovInConnector._sanitize_message strictly redacts the API key."""
    fake_key = "secret_api_key_xyz_987"
    connector = DataGovInConnector(api_key=fake_key)
    leak_string = f"Error fetching url https://api.data.gov.in/resource/123?api-key={fake_key}&format=json"
    clean_string = connector._sanitize_message(leak_string)
    assert fake_key not in clean_string
    assert "[REDACTED_API_KEY]" in clean_string


def test_persistent_shadow_audit_log_and_backup(tmp_path):
    """Verify that shadow_audit_log records survive restarts and online backup snapshot works."""
    db_file = tmp_path / "audit_test.db"
    init_shadow_db(db_file)

    # Ingest a sample row so mandi_observations is populated
    obs_df = pd.DataFrame([{
        "Market Name": "Burdwan", "Variety": "Common",
        "Reported Date": "2024-02-01", "Modal Price (Rs./Quintal)": 2500.0,
        "Arrivals (Tonnes)": 100.0,
    }])
    ingest_mandi_observations(obs_df, db_path=db_file)

    # Record audit log entry
    log_id = record_audit_log_entry(
        db_path=db_file,
        execution_type="PROSPECTIVE_DAILY",
        status="SUCCESS",
        source_label="test_source",
        latest_observation_date="2024-02-01",
        reporting_lag_days=5,
        is_stale=False,
        drift_detected=False,
        raw_rows=1,
        accepted_rows=1,
        rejected_rows=0,
        forecasts_generated=2,
        forecasts_skipped=0,
        outcomes_evaluated=0,
        outcomes_pending=2,
        duration_seconds=1.23,
    )
    assert log_id >= 1

    # Check get_shadow_status returns persistent audit and observation metadata
    status = get_shadow_status(db_file)
    assert status["latest_actual_observation_date"] == "2024-02-01"
    assert status["latest_successful_ingestion_timestamp"] is not None
    assert status["latest_audit_log"] is not None
    assert status["latest_audit_log"]["status"] == "SUCCESS"
    assert status["latest_audit_log"]["source_label"] == "test_source"

    # Test atomic online backup
    backup_file = backup_shadow_database(db_path=db_file)
    assert backup_file.exists()
    assert backup_file.stat().st_size > 0

    # Verify backup database can be read and verified
    backup_status = get_shadow_status(backup_file)
    assert backup_status["latest_actual_observation_date"] == "2024-02-01"
    assert backup_status["counts"]["total_observations"] == 1


def test_daily_prospective_workflow_stale_gating(tmp_path):
    """Verify daily prospective workflow blocks forecast generation on stale feeds unless allowed."""
    sys.path.insert(0, str(MEMBER2_DIR / "experiments"))
    from run_shadow_evaluation import run_daily_prospective_workflow

    db_file = tmp_path / "stale_gate.db"
    init_shadow_db(db_file)

    # Stale CSV with dates from 2024
    stale_csv = tmp_path / "stale_feed.csv"
    stale_csv.write_text(
        "Market Name,Variety,Reported Date,Modal Price (Rs./Quintal),Arrivals (Tonnes)\n"
        "Burdwan,Common,2024-01-15,2500.0,100.0\n"
        "Burdwan,Common,2024-01-16,2510.0,105.0\n",
        encoding="utf-8"
    )

    # Run without allow_stale -> should block prospective forecast generation and record WARNING_STALE
    rc = run_daily_prospective_workflow(
        db_path=db_file,
        input_csv=stale_csv,
        max_lag_days=14,
        allow_stale=False,
        create_backup=False,
    )
    assert rc == 0
    status = get_shadow_status(db_file)
    assert status["counts"]["by_cohort"]["prospective"]["total"] == 0
    assert status["latest_audit_log"]["status"] == "WARNING_STALE"
    assert status["latest_audit_log"]["is_stale"] == 1


