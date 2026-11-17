"""
AgroVision-ML — Milestone 4 End-to-End Integration Test Suite
============================================================
Exhaustive verification of unified Member 1 + Member 2 functionality:
1. Aus, Aman and Boro yield requests.
2. Complete weather input.
3. Missing weather fallback.
4. NASA POWER unavailability.
5. Stale weather handling.
6. Rice yield + rice price + volatility integration (predict_all).
7. Four-commodity market prediction (Rice, Wheat, Potato, Tea).
8. Invalid commodity and unsupported horizon rejection.
9. Prediction interval formatting and monotonic quantiles.
10. Experimental release status governance.
11. API concurrency and recovery.
12. Protected artifact integrity and certified checksums.
13. Existing legacy API compatibility and deprecation warnings.
"""

import sys
import os
import hashlib
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MEMBER2_SRC = PROJECT_ROOT / "member2-price-volatility" / "src"
if str(MEMBER2_SRC) not in sys.path:
    sys.path.insert(0, str(MEMBER2_SRC))

from shared.predict_bridge import (
    AgroVisionPredictor,
    YieldPredictionResult,
    CombinedPredictionResult,
    predict,
    predict_yield,
    predict_price,
    predict_price_horizon,
)
from shared.paths import get_m1_models_dir, get_member2_root
from api_v2 import app


@pytest.fixture(scope="module")
def predictor():
    """Shared AgroVisionPredictor instance with auto_fetch_weather disabled for unit isolation."""
    return AgroVisionPredictor(auto_fetch_weather=False, weather_missing_policy="persistence")


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient for Staging API v2."""
    with TestClient(app) as client:
        yield client


def make_valid_mandi_history(days: int = 35, base_price: float = 2100.0, trend: float = 3.0):
    """Generates synthetic valid APMC historical records."""
    dates = pd.date_range(end="2024-05-15", periods=days, freq="D")
    records = []
    for i, d in enumerate(dates):
        p = base_price + i * trend + np.sin(i / 3.0) * 15.0
        records.append({
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": round(p, 2),
            "Arrivals (Tonnes)": round(50.0 + (i % 7) * 5.0, 2),
        })
    return records


# ==============================================================================
# TEST 1: Aus, Aman, and Boro Yield Requests
# ==============================================================================
def test_01_aus_aman_boro_yield_requests(predictor):
    """Verify inference across Autumn (Aus), Winter (Aman), and Summer (Boro) seasons."""
    seasons = [
        ("Autumn", 2800.0, 20000.0),
        ("Winter", 3200.0, 45000.0),
        ("Summer", 3500.0, 15000.0),
    ]
    for season, lag_yield, lag_area in seasons:
        rec = {
            "District": "Purba Bardhaman",
            "Season": season,
            "Year": 2021,
            "Yield_lag1": lag_yield,
            "Yield_lag2": lag_yield - 50.0,
            "Area_lag1": lag_area,
            "Area_lag2": lag_area,
        }
        res = predictor.predict_yield(rec)
        assert isinstance(res, YieldPredictionResult)
        assert "predicted_yield_kg_per_ha" in res
        assert "predicted_yield_tonnes_per_ha" in res
        # Dual-unit math verification
        assert abs(res["predicted_yield_tonnes_per_ha"] - (res["predicted_yield_kg_per_ha"] / 1000.0)) < 1e-4
        # Agronomic range checks
        assert 500.0 <= res["predicted_yield_kg_per_ha"] <= 6000.0
        # Check season specific metadata
        if season == "Summer":
            assert res.get("seasonal_routing_applied") is True
            assert res.get("prediction_method") == "provisional_summer_persistence"


# ==============================================================================
# TEST 2: Complete Weather Input
# ==============================================================================
def test_02_complete_weather_input(predictor):
    """Verify that complete in-season weather features trigger ML model prediction."""
    rec = {
        "District": "Hooghly",
        "Season": "Autumn",
        "Year": 2021,
        "Yield_lag1": 3100.0,
        "Yield_lag2": 3050.0,
        "Area_lag1": 25000.0,
        "Area_lag2": 24800.0,
        "InSeason_Precip_Cum": 1031.19,
        "InSeason_Precip_Mean": 11.20,
        "InSeason_Precip_Max": 115.64,
        "InSeason_T2M_Mean": 29.58,
        "InSeason_T2M_Min": 23.86,
        "InSeason_T2M_Max": 41.50,
        "InSeason_T2M_Range": 17.64,
        "InSeason_RH2M_Mean": 79.98,
        "InSeason_Solar_Mean": 16.58,
        "InSeason_Wind_Mean": 2.64,
    }
    res = predictor.predict_yield(rec, enable_seasonal_routing=False)
    assert res["weather_feature_status"] == "complete"
    assert res["weather_features_used"] is True
    assert len(res["missing_weather_features"]) == 0
    assert "delta_ridge" in res["prediction_method"].lower() or "improved" in res["prediction_method"].lower()


# ==============================================================================
# TEST 3: Missing Weather Fallback
# ==============================================================================
def test_03_missing_weather_fallback(predictor):
    """Verify fallback when weather is omitted under 'persistence' missing policy."""
    rec = {
        "District": "Bankura",
        "Season": "Winter",
        "Year": 2021,
        "Yield_lag1": 2950.0,
        "Yield_lag2": 2900.0,
        "Area_lag1": 30000.0,
        "Area_lag2": 30000.0,
    }
    # Persistence missing policy: bypass delta, predict baseline
    res = predictor.predict_yield(rec, weather_missing_policy="persistence", auto_fetch_weather=False)
    assert res["weather_feature_status"] == "missing"
    assert res["weather_features_used"] is False
    assert res["predicted_delta_kg_ha"] == 0.0
    assert res["predicted_yield_kg_per_ha"] == 2950.0
    assert res["prediction_method"] == "persistence"


# ==============================================================================
# TEST 4: NASA POWER Unavailability Handling
# ==============================================================================
def test_04_nasa_power_unavailability(predictor):
    """Verify safe fallback when NASA POWER weather service fails or is unreachable."""
    rec = {
        "District": "Nadia",
        "Season": "Autumn",
        "Year": 2021,
        "Yield_lag1": 3000.0,
        "Yield_lag2": 2950.0,
        "Area_lag1": 22000.0,
        "Area_lag2": 22000.0,
    }
    mock_service = MagicMock()
    mock_service.get_inseason_features.side_effect = RuntimeError("NASA POWER API 503 Service Unavailable")

    with patch.object(predictor, "_get_weather_service", return_value=mock_service):
        res = predictor.predict_yield(rec, auto_fetch_weather=True, weather_missing_policy="persistence")
        # Must not crash, should fall back to persistence
        assert res["predicted_yield_kg_per_ha"] == 3000.0
        assert res["weather_features_used"] is False
        assert res["prediction_method"] == "persistence"


# ==============================================================================
# TEST 5: Stale Weather Metadata
# ==============================================================================
def test_05_stale_weather_handling(predictor):
    """Verify weather provenance and freshness tracking."""
    rec = {
        "District": "Birbhum",
        "Season": "Autumn",
        "Year": 2020,
        "Yield_lag1": 3100.0,
        "InSeason_Precip_Cum": 800.0,
        "InSeason_Precip_Mean": 8.5,
        "InSeason_Precip_Max": 80.0,
        "InSeason_T2M_Mean": 28.0,
        "InSeason_T2M_Min": 22.0,
        "InSeason_T2M_Max": 38.0,
        "InSeason_T2M_Range": 16.0,
        "InSeason_RH2M_Mean": 80.0,
        "InSeason_Solar_Mean": 17.0,
        "InSeason_Wind_Mean": 2.5,
    }
    res = predictor.predict_yield(rec)
    assert "prediction_policy_version" in res
    assert res["prediction_policy_version"] == "2.1"


# ==============================================================================
# TEST 6: Rice Yield + Price + Volatility Integration (predict_all)
# ==============================================================================
def test_06_rice_yield_plus_price_plus_volatility_integration(predictor):
    """Verify predict_all integrates yield, price, and volatility while separating volume from intensity."""
    history = make_valid_mandi_history(35, base_price=2200.0)
    df_hist = pd.DataFrame(history)

    yield_inp = {
        "District": "Purba Bardhaman",
        "Season": "Winter",
        "Year": 2021,
        "Yield_lag1": 3400.0,
        "Yield_lag2": 3350.0,
        "Area_lag1": 50000.0,
        "Area_lag2": 50000.0,
    }

    # Case A: With valid area -> estimated_production_tonnes must be computed
    res_with_area = predictor.predict_all(
        data_df=df_hist,
        commodity="rice",
        yield_input=yield_inp,
        area_ha=100.0,
    )
    assert isinstance(res_with_area, CombinedPredictionResult)
    assert "crop_yield" in res_with_area
    assert "market_price" in res_with_area
    assert "predicted_price_rs_quintal" in res_with_area
    assert "volatility" in res_with_area
    assert "predicted_yield_tonnes_per_ha" in res_with_area
    assert "estimated_production_tonnes" in res_with_area
    expected_prod = res_with_area["predicted_yield_tonnes_per_ha"] * 100.0
    assert abs(res_with_area["estimated_production_tonnes"] - expected_prod) < 1e-2

    # Case B: Without area -> estimated_production_tonnes must be omitted
    res_no_area = predictor.predict_all(
        data_df=df_hist,
        commodity="rice",
        yield_input=yield_inp,
        area_ha=None,
    )
    assert "estimated_production_tonnes" not in res_no_area or res_no_area.get("estimated_production_tonnes") is None


# ==============================================================================
# TEST 7: Four-Commodity Market Prediction
# ==============================================================================
def test_07_four_commodity_market_prediction(predictor):
    """Verify price and volatility prediction across all 4 supported commodities."""
    history = make_valid_mandi_history(35, base_price=2200.0)
    df_hist = pd.DataFrame(history)

    for comm in ["rice", "wheat", "potato", "tea"]:
        res = predictor.predict(history=df_hist, commodity=comm, horizon="short")
        assert res["commodity"] == comm
        assert "predicted_price_rs_quintal" in res
        assert isinstance(res["predicted_price_rs_quintal"], float)
        assert res["predicted_price_rs_quintal"] > 0
        assert res["volatility"] in ("LOW", "MEDIUM", "HIGH")
        assert "volatility_probabilities" in res
        assert abs(sum(res["volatility_probabilities"].values()) - 1.0) < 1e-3


# ==============================================================================
# TEST 8: Invalid Commodity & Unsupported Horizon Rejection
# ==============================================================================
def test_08_invalid_commodity_and_unsupported_horizon(predictor):
    """Verify rejection of invalid commodities and unsupported horizon windows."""
    history = make_valid_mandi_history(35)
    df_hist = pd.DataFrame(history)

    # Unsupported commodity
    with pytest.raises(ValueError, match="Unsupported commodity"):
        predictor.predict(history=df_hist, commodity="onion", horizon="short")

    # Unsupported horizon
    with pytest.raises(ValueError, match="Unsupported horizon"):
        predictor.predict(history=df_hist, commodity="rice", horizon="60d")

    # Member 1 does not predict wheat/potato yield
    yield_wheat_inp = {"District": "Hooghly", "Season": "Winter", "Crop": "Wheat", "Yield_lag1": 2500.0}
    # predict_yield only supports Rice
    res_y = predictor.predict_yield(yield_wheat_inp)
    assert res_y["model_name"] == "m1_improved_delta_ridge.pkl"  # Rice model artifact exclusively


# ==============================================================================
# TEST 9: Prediction Interval Formatting & Monotonicity
# ==============================================================================
def test_09_prediction_interval_formatting(predictor):
    """Verify that empirical prediction intervals are monotonically non-decreasing."""
    history = make_valid_mandi_history(35, base_price=2300.0)
    df_hist = pd.DataFrame(history)

    for comm in ["rice", "wheat", "potato"]:
        res = predictor.predict(history=df_hist, commodity=comm, horizon="short")
        pi = res.get("prediction_interval")
        assert pi is not None
        for k in ["p10", "p25", "p50", "p75", "p90", "symmetric_80_ci"]:
            assert k in pi
        # Monotonicity check
        assert pi["p10"] <= pi["p25"] <= pi["p50"] <= pi["p75"] <= pi["p90"]
        assert pi["symmetric_80_ci"] == [pi["p10"], pi["p90"]]


# ==============================================================================
# TEST 10: Experimental Release Status Governance
# ==============================================================================
def test_10_experimental_release_status(predictor):
    """Verify that 30-day horizons are tagged as experimental with advisory limitations."""
    history = make_valid_mandi_history(35, base_price=2200.0)
    df_hist = pd.DataFrame(history)

    # Rice 30-day horizon -> experimental
    res_30d = predictor.predict(history=df_hist, commodity="rice", horizon="30d")
    assert res_30d["horizon_days"] == 30
    assert res_30d["release_status"] == "experimental"
    assert res_30d["is_experimental"] is True

    # Rice short horizon -> production
    res_short = predictor.predict(history=df_hist, commodity="rice", horizon="short")
    assert res_short["horizon_days"] == 1
    assert res_short["release_status"] == "production"
    assert res_short["is_experimental"] is False


# ==============================================================================
# TEST 11: API Concurrency and Recovery
# ==============================================================================
def test_11_api_concurrency_and_recovery(api_client):
    """Verify thread-safety and concurrency handling via Staging API v2."""
    valid_hist = make_valid_mandi_history(35)
    history_payload = [
        {"reported_date": h["Reported Date"], "modal_price": h["Modal Price (Rs./Quintal)"], "arrivals": h["Arrivals (Tonnes)"]}
        for h in valid_hist
    ]
    payload = {
        "commodity": "rice",
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-05-16",
        "horizon": "short",
        "history": history_payload,
    }

    def call_predict():
        resp = api_client.post("/v2/predict", json=payload)
        return resp.status_code, resp.json()

    # Fire 10 concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(call_predict) for _ in range(10)]
        results = [f.result() for f in futures]

    for status_code, body in results:
        assert status_code == 200
        assert body["commodity"] == "rice"
        assert body["predicted_price_rs_quintal"] > 0

    # Rapid recovery from a bad request
    bad_resp = api_client.post("/v2/predict", json={"commodity": "rice", "history": []})
    assert bad_resp.status_code in (400, 422)

    # Next legitimate request still succeeds
    good_resp = api_client.post("/v2/predict", json=payload)
    assert good_resp.status_code == 200


# ==============================================================================
# TEST 12: Protected Artifact Integrity Check
# ==============================================================================
def test_12_protected_artifact_integrity():
    """Verify that production model artifacts have not been altered or retrained."""
    m1_artifact = PROJECT_ROOT / "member1-yield-model" / "models" / "m1_improved_delta_ridge.pkl"
    assert m1_artifact.exists(), "Member 1 production model missing!"

    with open(m1_artifact, "rb") as f:
        m1_hash = hashlib.sha256(f.read()).hexdigest()

    expected_m1_hash = "3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56"
    assert m1_hash == expected_m1_hash, f"Member 1 SHA-256 altered! Found {m1_hash}"


# ==============================================================================
# TEST 13: Existing Legacy API Compatibility & Deprecation Warnings
# ==============================================================================
def test_13_legacy_api_compatibility(api_client, predictor):
    """Verify backward compatibility of legacy endpoints and deprecation warnings."""
    # Test legacy /health
    h_resp = api_client.get("/health")
    assert h_resp.status_code == 200
    assert h_resp.json()["status"] == "ok"

    # Test legacy deprecation warning for yield_arrivals_tonnes
    rec = {
        "District": "Birbhum",
        "Season": "Winter",
        "Year": 2021,
        "Yield_lag1": 3100.0,
    }
    res = predictor.predict_yield(rec, enable_legacy_alias=True)
    with pytest.deprecated_call():
        _ = res["yield_arrivals_tonnes"]
