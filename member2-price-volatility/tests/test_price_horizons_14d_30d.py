"""
Unit and integration tests for 14-day and 30-day Rice Price Forecasting Workflow.

Verifies:
1. Frozen production price model and configuration remain 100% unaltered.
2. 14-day and 30-day artifacts load with exact 48-feature parity and ordering.
3. Target generation correctly applies calendar horizons and 3-day tolerance windows.
4. No cross-group target leakage across markets or varieties.
5. Temporal purging strictly eliminates split-boundary leakage.
6. Pipeline predict_horizon produces valid schemas and price reconstructions.
7. Error handling for unsupported horizons and insufficient history (<30 rows).
8. Strict backward compatibility for default predict() and predict_normalized_volatility().
9. FastAPI POST /predict-horizon, GET /health, and POST /predict schemas.
10. AgroVisionPredictor bridge horizon integration.
"""

import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
REPO_ROOT = MEMBER2_DIR.parent
ARTIFACTS_DIR = MEMBER2_DIR / "artifacts"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT / "shared"))

from inference import RiceInferencePipeline, InsufficientHistoryError
from horizon_targets import (
    compute_single_horizon_targets,
    build_dual_horizon_dataset,
    apply_horizon_temporal_purge,
)
from predict_bridge import AgroVisionPredictor
from api import app
from fastapi.testclient import TestClient
from config import (
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    reset_horizon_forecasting_switches,
)


FROZEN_PRICE_MODEL_SHA256 = "b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993"


@pytest.fixture(autouse=True)
def configure_horizon_switches_for_suite():
    """Ensure horizon switches are enabled for the duration of this specific test module."""
    set_horizon_forecasting_enabled(True)
    set_horizon_14d_enabled(True)
    set_horizon_30d_enabled(True)
    yield
    reset_horizon_forecasting_switches()


@pytest.fixture(scope="module")
def pipeline():
    return RiceInferencePipeline()


@pytest.fixture
def synthetic_history_35():
    """Valid sequential 35-observation history for Burdwan / Common."""
    dates = pd.date_range("2023-01-01", periods=35, freq="D")
    prices = [2500.0 + i * 1.5 for i in range(35)]
    arrivals = [100.0 + (i % 5) * 10.0 for i in range(35)]
    return pd.DataFrame({
        "Market Name": ["Burdwan"] * 35,
        "Variety": ["Common"] * 35,
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": prices,
        "Arrivals (Tonnes)": arrivals,
    })


# ============================================================
# 1. FROZEN PRODUCTION ARTIFACT INTEGRITY
# ============================================================

def test_frozen_production_price_model_and_config_unaltered():
    """Verify that rice_price_model.pkl and rice_price_final_config.json are strictly untouched."""
    model_path = ARTIFACTS_DIR / "rice_price_model.pkl"
    config_path = ARTIFACTS_DIR / "rice_price_final_config.json"

    assert model_path.exists(), "Missing frozen rice_price_model.pkl"
    assert config_path.exists(), "Missing frozen rice_price_final_config.json"

    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert actual_hash == FROZEN_PRICE_MODEL_SHA256, (
        f"CRITICAL SAFETY VIOLATION: rice_price_model.pkl checksum changed!\n"
        f"Expected: {FROZEN_PRICE_MODEL_SHA256}\nActual:   {actual_hash}"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert cfg["commodity"] == "rice"
    assert len(cfg["features"]) == 48
    assert cfg["train_period"] == ["2002-05-28", "2021-01-31"]
    assert cfg["final_holdout_period"] == ["2023-02-01", "2024-02-01"]


# ============================================================
# 2. HORIZON ARTIFACTS VALIDATION
# ============================================================

@pytest.mark.parametrize("horizon_days", [14, 30])
def test_horizon_artifacts_exist_and_have_exact_feature_parity(horizon_days):
    """Verify that 14d and 30d models exist and match the 48-feature specification."""
    model_path = ARTIFACTS_DIR / f"rice_price_model_{horizon_days}d.pkl"
    config_path = ARTIFACTS_DIR / f"rice_price_model_{horizon_days}d_config.json"

    assert model_path.exists(), f"Missing horizon model artifact: {model_path}"
    assert config_path.exists(), f"Missing horizon config artifact: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    with open(ARTIFACTS_DIR / "rice_price_final_config.json", "r", encoding="utf-8") as f:
        prod_cfg = json.load(f)

    assert cfg["horizon_days"] == horizon_days
    assert cfg["max_tolerance_days"] == 3
    assert len(cfg["features"]) == 48
    assert cfg["features"] == prod_cfg["features"], "48-feature order must match frozen production model"


# ============================================================
# 3. TARGET CONSTRUCTION & TOLERANCE WINDOW
# ============================================================

def test_horizon_target_construction_and_tolerance_window():
    """Verify target matches first observation in [t + H, t + H + 3] calendar days."""
    dates = pd.to_datetime([
        "2023-01-01",  # Row 0
        "2023-01-05",  # Row 1
        "2023-01-15",  # Exactly 14 days after Row 0
        "2023-01-18",  # Exactly 17 days after Row 0 (boundary of 14+3)
        "2023-01-31",  # Exactly 30 days after Row 0
        "2023-02-03",  # Exactly 33 days after Row 0 (boundary of 30+3)
        "2023-02-15",  # Outside tolerance window for Row 1
    ])
    prices = [2000.0, 2020.0, 2050.0, 2070.0, 2100.0, 2120.0, 2200.0]
    df = pd.DataFrame({
        "Market Name": ["M1"] * 7,
        "Variety": ["V1"] * 7,
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": prices,
    })

    df_dual = build_dual_horizon_dataset(df, max_tolerance_days=3)

    # Row 0 (2023-01-01):
    # 14-day target: first observation in [2023-01-15, 2023-01-18] -> 2023-01-15 (P=2050.0, delta=+50.0)
    assert df_dual["future_modal_price_14d"].iloc[0] == 2050.0
    assert df_dual["future_price_change_14d"].iloc[0] == 50.0
    assert df_dual["actual_horizon_14d"].iloc[0] == 14.0

    # 30-day target for Row 0:
    # First observation in [2023-01-31, 2023-02-03] -> 2023-01-31 (P=2100.0, delta=+100.0)
    assert df_dual["future_modal_price_30d"].iloc[0] == 2100.0
    assert df_dual["future_price_change_30d"].iloc[0] == 100.0
    assert df_dual["actual_horizon_30d"].iloc[0] == 30.0

    # Row 1 (2023-01-05):
    # 14-day window: [2023-01-19, 2023-01-22]. Available dates: 2023-01-18 (too early) and 2023-01-31 (too late).
    # Must receive NaN!
    assert pd.isna(df_dual["future_price_change_14d"].iloc[1])


# ============================================================
# 4. CROSS-GROUP ISOLATION
# ============================================================

def test_no_cross_group_target_leakage():
    """Verify observations from Market A never serve as targets for Market B."""
    df = pd.DataFrame({
        "Market Name": ["MarketA", "MarketA", "MarketB", "MarketB"],
        "Variety": ["Common", "Common", "Common", "Common"],
        "Reported Date": pd.to_datetime(["2023-01-01", "2023-01-15", "2023-01-01", "2023-01-25"]),
        "Modal Price (Rs./Quintal)": [2000.0, 2100.0, 3000.0, 3200.0],
    })

    df_14 = compute_single_horizon_targets(df, horizon_days=14, max_tolerance_days=3)

    # Market A row 0 (2023-01-01) has target in Market A on 2023-01-15
    row_a = df_14[(df_14["Market Name"] == "MarketA") & (df_14["Reported Date"] == "2023-01-01")].iloc[0]
    assert row_a["future_modal_price_14d"] == 2100.0
    assert row_a["future_price_change_14d"] == 100.0

    # Market B row 2 (2023-01-01) has no observation in [2023-01-15, 2023-01-18] in Market B (next is 2023-01-25).
    # It must NOT steal Market A's 2023-01-15 observation!
    row_b = df_14[(df_14["Market Name"] == "MarketB") & (df_14["Reported Date"] == "2023-01-01")].iloc[0]
    assert pd.isna(row_b["future_price_change_14d"]), "Market B incorrectly assigned Market A observation!"


# ============================================================
# 5. TEMPORAL PURGING
# ============================================================

def test_temporal_purging_prevents_split_leakage():
    """Verify that rows whose future targets touch or cross split boundaries are dropped."""
    df = pd.DataFrame({
        "Market Name": ["M1", "M1"],
        "Variety": ["V1", "V1"],
        "Reported Date": pd.to_datetime(["2021-01-10", "2021-01-25"]),
        "Modal Price (Rs./Quintal)": [2000.0, 2010.0],
        # Row 0 target is 2021-01-24 (< 2021-02-01: SAFE)
        # Row 1 target is 2021-02-08 (>= 2021-02-01: LEAKAGE -> MUST PURGE)
        "target_date_14d": pd.to_datetime(["2021-01-24", "2021-02-08"]),
    })

    purged_df, n_purged = apply_horizon_temporal_purge(df, split_date="2021-02-01", horizon_days=14)

    assert n_purged == 1
    assert len(purged_df) == 1
    assert purged_df["Reported Date"].iloc[0] == pd.to_datetime("2021-01-10")


# ============================================================
# 6. PIPELINE INFERENCE & RECONSTRUCTION
# ============================================================

@pytest.mark.parametrize("horizon_days", [14, 30])
def test_pipeline_predict_horizon_output_schema_and_reconstruction(pipeline, synthetic_history_35, horizon_days):
    """Verify predict_horizon returns valid schema, correct model name, and price reconstruction."""
    res = pipeline.predict_horizon(synthetic_history_35, horizon_days=horizon_days)

    assert res["market"] == "Burdwan"
    assert res["variety"] == "Common"
    assert res["horizon_days"] == horizon_days
    assert res["model_name"] == f"rice-price-xgb-{horizon_days}d"
    assert "target_date_convention" in res

    # Reconstruction invariant: predicted_modal_price = current_modal_price + predicted_price_change
    expected_price = round(res["current_modal_price"] + res["predicted_price_change"], 2)
    assert np.isclose(res["predicted_modal_price"], expected_price, atol=0.01)


def test_pipeline_predict_horizon_dispatch_via_predict_kwargs(pipeline, synthetic_history_35):
    """Verify passing horizon_days kwarg to predict() routes to predict_horizon."""
    res = pipeline.predict(synthetic_history_35, horizon_days=14)
    assert res["horizon_days"] == 14
    assert res["model_name"] == "rice-price-xgb-14d"


# ============================================================
# 7. INSUFFICIENT HISTORY & INVALID HORIZON ERROR HANDLING
# ============================================================

def test_predict_horizon_raises_insufficient_history(pipeline, synthetic_history_35):
    """Fewer than 30 observations must raise InsufficientHistoryError."""
    short_df = synthetic_history_35.iloc[:25]
    with pytest.raises(InsufficientHistoryError):
        pipeline.predict_horizon(short_df, horizon_days=14)


@pytest.mark.parametrize("bad_horizon", [7, 10, 45, "invalid"])
def test_predict_horizon_raises_value_error_for_unsupported_horizon(pipeline, synthetic_history_35, bad_horizon):
    """Non 14/30 horizons must raise ValueError."""
    with pytest.raises(ValueError):
        pipeline.predict_horizon(synthetic_history_35, horizon_days=bad_horizon)


# ============================================================
# 8. BACKWARD COMPATIBILITY OF DEFAULT PREDICT
# ============================================================

def test_default_predict_remains_backward_compatible(pipeline, synthetic_history_35):
    """Calling predict() without horizon_days must preserve standard 1-7 day output."""
    res_default = pipeline.predict(synthetic_history_35)

    assert "predicted_modal_price" in res_default
    assert "predicted_volatility_class" in res_default
    assert "predicted_volatility_range" in res_default
    assert "horizon_days" not in res_default
    assert "model_name" not in res_default

    # Test v2_normalized volatility routing still intact
    res_v2 = pipeline.predict(synthetic_history_35, volatility_version="v2_normalized")
    assert res_v2["volatility_version"] == "v2_normalized"
    assert "normalized_volatility_class" in res_v2
    assert "normalized_volatility_probabilities" in res_v2


# ============================================================
# 9. FASTAPI ENDPOINTS & BACKWARD COMPATIBILITY
# ============================================================

def test_fastapi_predict_horizon_endpoints_and_health():
    """Verify POST /predict-horizon, POST /predict, and GET /health via TestClient."""
    with TestClient(app) as client:
        # 1. Health check
        h_res = client.get("/health")
        assert h_res.status_code == 200
        assert h_res.json() == {"status": "ok", "model": "rice-price-xgb", "volatility": "persistence"}

        dates = pd.date_range("2023-01-01", periods=35, freq="D")
        history = [
            {"reported_date": str(d.date()), "modal_price": 2500.0 + i, "arrivals": 100.0}
            for i, d in enumerate(dates)
        ]

        # 2. Standard /predict endpoint
        p_res = client.post("/predict", json={
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": "2023-02-05",
            "history": history
        })
        assert p_res.status_code == 200
        p_json = p_res.json()
        assert "predicted_modal_price" in p_json
        assert "predicted_volatility_class" in p_json

        # 3. Horizon endpoints: 14d and 30d
        for h in [14, 30]:
            h_res = client.post("/predict-horizon", json={
                "market": "Burdwan",
                "variety": "Common",
                "prediction_date": "2023-02-05",
                "horizon_days": h,
                "history": history
            })
            assert h_res.status_code == 200
            h_json = h_res.json()
            assert h_json["horizon_days"] == h
            assert h_json["model_name"] == f"rice-price-xgb-{h}d"
            assert "predicted_modal_price" in h_json
            assert "current_modal_price" in h_json

        # 4. Invalid horizon
        bad_res = client.post("/predict-horizon", json={
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": "2023-02-05",
            "horizon_days": 7,
            "history": history
        })
        assert bad_res.status_code == 400

        # 5. Short history (< 30)
        short_res = client.post("/predict-horizon", json={
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": "2023-02-05",
            "horizon_days": 14,
            "history": history[:20]
        })
        assert short_res.status_code == 400


# ============================================================
# 10. BRIDGE INTEGRATION
# ============================================================

def test_bridge_predict_price_horizon(synthetic_history_35):
    """Verify AgroVisionPredictor.predict_price_horizon() exposes horizon predictions."""
    bridge = AgroVisionPredictor()

    res14 = bridge.predict_price_horizon(synthetic_history_35, horizon_days=14, commodity="rice")
    assert res14["horizon_days"] == 14
    assert res14["model_name"] == "rice-price-xgb-14d"

    res30 = bridge.predict_price_horizon(synthetic_history_35, horizon_days=30, commodity="rice")
    assert res30["horizon_days"] == 30
    assert res30["model_name"] == "rice-price-xgb-30d"

    # Standard bridge predict_price remains backward compatible
    res_std = bridge.predict_price(synthetic_history_35, commodity="rice")
    assert "predicted_price_rs_quintal" in res_std
    assert "volatility" in res_std


# ============================================================
# 11. OPERATIONAL RELEASE SWITCHES & HEALTH/METRICS
# ============================================================

def test_horizon_switches_and_observability():
    """Verify operational switches disable endpoints with 503 without affecting default /predict."""
    with TestClient(app) as client:
        dates = pd.date_range("2023-01-01", periods=35, freq="D")
        history = [
            {"reported_date": str(d.date()), "modal_price": 2500.0 + i, "arrivals": 100.0}
            for i, d in enumerate(dates)
        ]
        payload_14 = {
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2023-02-05", "horizon_days": 14, "history": history
        }
        payload_30 = {
            "market": "Burdwan", "variety": "Common",
            "prediction_date": "2023-02-05", "horizon_days": 30, "history": history
        }

        try:
            # 1. Global disable
            set_horizon_forecasting_enabled(False)
            r_glob = client.post("/predict-horizon", json=payload_14)
            assert r_glob.status_code == 503
            assert "disabled globally" in r_glob.json()["detail"]

            # Default endpoint remains operational even when horizon is disabled
            r_legacy = client.post("/predict", json={
                "market": "Burdwan", "variety": "Common",
                "prediction_date": "2023-02-05", "history": history
            })
            assert r_legacy.status_code == 200

            # 2. Per-horizon switches: disable 14d only
            set_horizon_forecasting_enabled(True)
            set_horizon_14d_enabled(False)
            set_horizon_30d_enabled(True)

            r_14_off = client.post("/predict-horizon", json=payload_14)
            assert r_14_off.status_code == 503
            assert "14-day horizon forecasting is currently disabled" in r_14_off.json()["detail"]

            r_30_on = client.post("/predict-horizon", json=payload_30)
            assert r_30_on.status_code == 200
            assert r_30_on.json()["horizon_days"] == 30

            # 3. Disable 30d only
            set_horizon_14d_enabled(True)
            set_horizon_30d_enabled(False)

            r_14_on = client.post("/predict-horizon", json=payload_14)
            assert r_14_on.status_code == 200
            assert r_14_on.json()["horizon_days"] == 14

            r_30_off = client.post("/predict-horizon", json=payload_30)
            assert r_30_off.status_code == 503
            assert "30-day horizon forecasting is currently disabled" in r_30_off.json()["detail"]

            # 4. Observability endpoints: /health/horizons and /metrics/horizons
            set_horizon_forecasting_enabled(True)
            set_horizon_14d_enabled(True)
            set_horizon_30d_enabled(True)
            r_health = client.get("/health/horizons")
            assert r_health.status_code == 200
            h_data = r_health.json()
            assert h_data["status"] == "ok"
            assert h_data["horizon_14d"]["ready"] is True
            assert h_data["horizon_30d"]["ready"] is True

            # When reset to default (disabled by default in fresh installations):
            reset_horizon_forecasting_switches()
            r_health_default = client.get("/health/horizons")
            assert r_health_default.status_code == 200
            assert r_health_default.json()["master_switch_enabled"] is False
            assert r_health_default.json()["horizon_14d"]["ready"] is False

            r_metrics = client.get("/metrics/horizons")
            assert r_metrics.status_code == 200
            m_data = r_metrics.json()
            assert "14d" in m_data
            assert "30d" in m_data
            assert "requests" in m_data["14d"]

        finally:
            reset_horizon_forecasting_switches()


# ============================================================
# 12. NO STATE CONTAMINATION & METADATA INVARIANTS
# ============================================================

def test_no_state_contamination_and_target_date_math(pipeline, synthetic_history_35):
    """Verify calling predict_horizon leaves subsequent default predict results completely unaffected."""
    # Baseline default prediction
    pred_before = pipeline.predict(synthetic_history_35)

    # Call 14-day and 30-day horizon predictions
    h14 = pipeline.predict_horizon(synthetic_history_35, horizon_days=14)
    h30 = pipeline.predict_horizon(synthetic_history_35, horizon_days=30)

    # Verify target date math
    origin_dt = pd.to_datetime(h14["prediction_date"])
    expected_target_14 = (origin_dt + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    expected_target_30 = (origin_dt + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    assert h14["requested_target_date"] == expected_target_14
    assert h30["requested_target_date"] == expected_target_30
    assert h14["model_version"] == "1.0.0"
    assert h30["model_version"] == "1.0.0"
    assert h14["release_status"] == "opt-in"
    assert h30["release_status"] == "experimental"
    assert "limitations" in h14 and len(h14["limitations"]) > 10
    assert "limitations" in h30 and len(h30["limitations"]) > 10

    # Repeat default prediction and verify exact equality with pred_before
    pred_after = pipeline.predict(synthetic_history_35)
    assert pred_before["predicted_modal_price"] == pred_after["predicted_modal_price"]
    assert pred_before["predicted_volatility_class"] == pred_after["predicted_volatility_class"]

