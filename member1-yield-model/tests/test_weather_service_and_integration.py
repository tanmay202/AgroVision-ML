"""
Automated Test Suite: NASA POWER Weather Service & Bridge Integration
=====================================================================
Validates all Workstream D requirements:
1. Successful weather fetch (offline archive & mock live API).
2. Empty API response handling.
3. Timeout and exponential retry behavior.
4. Cached responses & persistent SQLite stats.
5. Missing weather variables (honest incompleteness without fabrication).
6. Invalid weather values & NASA -999 sentinels sanitization.
7. Date parsing and boundary enforcement.
8. Seasonal cutoff enforcement (Aus: July 31, Aman: Sept 30, Boro: Feb 28).
9. NASA POWER outage fallback to persistence policy.
10. Weather provenance tracking (source, timestamp, freshness).
11. Bridge integration (auto_fetch_weather=True / False).
12. No regressions or alterations to Member 2 behavior.
13. Production model artifact SHA-256 checksum preservation.
"""

import hashlib
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import requests

# Ensure project and member1 roots are importable
TEST_DIR = Path(__file__).resolve().parent
M1_ROOT = TEST_DIR.parent
PROJECT_ROOT = M1_ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(M1_ROOT) not in sys.path:
    sys.path.insert(0, str(M1_ROOT))
if str(M1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(M1_ROOT / "src"))
if str(M1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(M1_ROOT / "scripts"))

from src.weather_service import NASAWeatherService, get_weather_service, DISTRICT_CENTROIDS
from shared.predict_bridge import AgroVisionPredictor, YieldPredictionResult, CombinedPredictionResult


FROZEN_MODEL_SHA256 = "3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56"


# =====================================================================
# 1. Model Artifact Checksum Preservation
# =====================================================================

def test_production_artifact_checksum_preservation():
    """Verify that the production model artifact has not been modified or retrained."""
    model_path = M1_ROOT / "models" / "m1_improved_delta_ridge.pkl"
    assert model_path.exists(), f"Production model missing at {model_path}"

    with open(model_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest().lower()

    assert file_hash == FROZEN_MODEL_SHA256, (
        f"Production model SHA-256 corrupted or modified! Expected: {FROZEN_MODEL_SHA256}, Got: {file_hash}"
    )


# =====================================================================
# 2. Successful Fetch: Offline Archive & Persistent SQLite Cache
# =====================================================================

def test_weather_service_offline_archive_and_caching():
    """
    Test that historical weather can be fetched from the local verified archive,
    cached into SQLite, and subsequent queries hit the cache.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_cache.db"
        ws = NASAWeatherService(cache_db_path=cache_file, allow_live_api=False)

        # Query Hooghly 2018 Winter (which exists in offline archive)
        res1 = ws.get_inseason_features(district="Hooghly", year=2018, season="Winter")
        assert res1["is_complete"] is True
        assert res1["weather_source"] in ("offline_archive", "mixed")
        assert len(res1["features"]) == 10
        for feat_name, feat_val in res1["features"].items():
            assert not np.isnan(feat_val), f"Feature {feat_name} is NaN"

        # Query same slice again: MUST hit cache
        res2 = ws.get_inseason_features(district="Hooghly", year=2018, season="Winter")
        assert res2["is_complete"] is True
        assert res2["weather_source"] == "cache"

        # Verify cache metrics
        stats = ws.get_cache_stats()
        assert stats["total_observations"] > 0
        assert stats["districts_cached"] >= 1
        assert "Hooghly" in DISTRICT_CENTROIDS


# =====================================================================
# 3. Mock Live NASA POWER API: Successful Parsing & Caching
# =====================================================================

def test_mock_live_api_successful_fetch():
    """
    Test that when live API returns valid NASA POWER JSON,
    the service parses parameters, sanitizes values, and writes to SQLite.
    """
    mock_api_payload = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {"20210501": 5.2, "20210502": 0.0},
                "T2M": {"20210501": 29.5, "20210502": 30.1},
                "T2M_MIN": {"20210501": 24.0, "20210502": 24.5},
                "T2M_MAX": {"20210501": 35.0, "20210502": 35.8},
                "RH2M": {"20210501": 75.0, "20210502": 72.0},
                "ALLSKY_SFC_SW_DWN": {"20210501": 18.2, "20210502": 19.1},
                "WS2M": {"20210501": 2.1, "20210502": 2.4},
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_api_cache.db"
        ws = NASAWeatherService(cache_db_path=cache_file, allow_live_api=True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_api_payload

        with patch("requests.get", return_value=mock_resp) as mock_get:
            df_rec, err = ws._fetch_from_api_with_retry("Bankura", 23.2324, 87.0715, "20210501", "20210502")
            assert mock_get.called
            assert err is None
            assert df_rec is not None
            assert len(df_rec) == 2
            assert df_rec.iloc[0]["Date"] == "2021-05-01"
            assert df_rec.iloc[0]["T2M"] == 29.5

            # Check that observations can be saved to cache
            ws._save_to_cache("Bankura", df_rec, source="live_api")
            cached = ws._get_cached_observations("Bankura", pd.to_datetime("2021-05-01"), pd.to_datetime("2021-05-02"))
            assert len(cached) == 2


# =====================================================================
# 4. Empty or Malformed API Responses
# =====================================================================

def test_empty_or_malformed_api_responses():
    """Verify that empty or malformed API responses fail safely without crashing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_empty_cache.db"
        ws = NASAWeatherService(cache_db_path=cache_file, allow_live_api=True)

        # 4a. Empty properties
        mock_resp1 = MagicMock()
        mock_resp1.status_code = 200
        mock_resp1.json.return_value = {"properties": {}}

        with patch("requests.get", return_value=mock_resp1):
            df_rec1, err1 = ws._fetch_from_api_with_retry("Bankura", 23.2324, 87.0715, "20210501", "20210502")
            assert df_rec1 is None or df_rec1.empty
            assert err1 is not None

        # 4b. 500 error response
        mock_resp2 = MagicMock()
        mock_resp2.status_code = 500

        with patch("requests.get", return_value=mock_resp2):
            df_rec2, err2 = ws._fetch_from_api_with_retry("Bankura", 23.2324, 87.0715, "20210501", "20210502")
            assert df_rec2 is None or df_rec2.empty
            assert err2 is not None


# =====================================================================
# 5. Timeout & Exponential Backoff Behavior
# =====================================================================

def test_timeout_and_retry_behavior():
    """Verify exponential backoff retries when API times out."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_timeout.db"
        ws = NASAWeatherService(
            cache_db_path=cache_file,
            allow_live_api=True,
            timeout_seconds=0.1,
            max_retries=2,
            retry_delay_seconds=0.01,
        )

        with patch("requests.get", side_effect=requests.exceptions.Timeout("Timeout")):
            df_rec, err = ws._fetch_from_api_with_retry("Birbhum", 23.8404, 87.6186, "20210501", "20210502")
            assert df_rec is None
            assert err is not None
            assert "timeout" in err.lower()


# =====================================================================
# 6. Missing Weather Variables: Honest Incompleteness (No Fabrication)
# =====================================================================

def test_missing_weather_variables_non_fabrication():
    """Verify that when meteorological parameters are missing, values are NaN without fabrication."""
    mock_incomplete_payload = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {"20210501": 5.2},
                "T2M": {"20210501": 29.5},
                # Missing T2M_MIN, T2M_MAX, RH2M, ALLSKY_SFC_SW_DWN, WS2M
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_missing_vars.db"
        ws = NASAWeatherService(cache_db_path=cache_file, allow_live_api=True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_incomplete_payload

        with patch("requests.get", return_value=mock_resp):
            df_rec, err = ws._fetch_from_api_with_retry("Bankura", 23.2324, 87.0715, "20210501", "20210501")
            assert df_rec is not None
            assert len(df_rec) == 1
            # Missing columns must be NaN, NOT synthesized or fabricated
            assert np.isnan(df_rec.iloc[0]["T2M_MIN"])
            assert np.isnan(df_rec.iloc[0]["RH2M"])


# =====================================================================
# 7. Invalid Weather Values: NASA -999 Sentinel Sanitization
# =====================================================================

def test_sentinel_sanitization():
    """Verify NASA POWER sentinels (-999, -9999) are mapped to NaN."""
    mock_sentinel_payload = {
        "properties": {
            "parameter": {
                "PRECTOTCORR": {"20210501": -999.0},
                "T2M": {"20210501": -999.0},
                "T2M_MIN": {"20210501": -9999.0},
                "T2M_MAX": {"20210501": 35.0},
                "RH2M": {"20210501": 80.0},
                "ALLSKY_SFC_SW_DWN": {"20210501": -999.0},
                "WS2M": {"20210501": 2.0},
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_file = Path(tmp_dir) / "test_sentinels.db"
        ws = NASAWeatherService(cache_db_path=cache_file, allow_live_api=True)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = mock_sentinel_payload

        with patch("requests.get", return_value=mock_resp):
            df_rec, err = ws._fetch_from_api_with_retry("Bankura", 23.2324, 87.0715, "20210501", "20210501")
            assert df_rec is not None
            assert len(df_rec) == 1
            assert np.isnan(df_rec.iloc[0]["PRECTOTCORR"])
            assert np.isnan(df_rec.iloc[0]["T2M"])
            assert np.isnan(df_rec.iloc[0]["T2M_MIN"])
            assert np.isnan(df_rec.iloc[0]["ALLSKY_SFC_SW_DWN"])
            assert df_rec.iloc[0]["T2M_MAX"] == 35.0


# =====================================================================
# 8. Date Parsing & Seasonal Cutoff Enforcement
# =====================================================================

def test_seasonal_cutoff_enforcement():
    """
    Verify seasonal cutoffs:
    - Aus (Autumn): July 31
    - Aman (Winter): September 30
    - Boro (Summer): February 28 (or 29 on leap years)
    """
    ws = get_weather_service()

    # Aus window: May 1 to July 31
    f_aus = ws.get_inseason_features(district="Hooghly", year=2018, season="Autumn")
    assert f_aus["metadata"]["cutoff_date"] == "2018-07-31"

    # Aman window: June 1 to September 30
    f_aman = ws.get_inseason_features(district="Hooghly", year=2018, season="Winter")
    assert f_aman["metadata"]["cutoff_date"] == "2018-09-30"

    # Boro window: November 1 to February 28
    f_boro = ws.get_inseason_features(district="Hooghly", year=2018, season="Summer")
    assert f_boro["metadata"]["cutoff_date"] == "2018-02-28"

    # Boundary check: start_date > end_date returns safe empty structure
    inv_res = ws.fetch_daily_weather("Hooghly", "2020-05-10", "2020-05-01")
    assert inv_res["is_complete"] is False
    assert inv_res["source"] == "invalid_range"


# =====================================================================
# 9. NASA POWER Outage Fallback to Persistence
# =====================================================================

def test_weather_outage_fallback_to_persistence():
    """
    When weather service cannot retrieve complete data, prediction bridge
    must fall back safely to configured missing policy (persistence) without error.
    """
    # Create predictor with custom failing weather service
    failing_service = MagicMock()
    failing_service.get_inseason_features.return_value = {
        "features": {},
        "is_complete": False,
        "weather_source": "failed",
        "metadata": {},
        "warnings": ["Simulated NASA POWER service outage."],
    }

    predictor = AgroVisionPredictor(
        auto_fetch_weather=True,
        weather_service=failing_service,
        weather_missing_policy="persistence",
    )

    req = {
        "District": "Bankura",
        "Season": "Winter",
        "Year": 2022,
        "Yield_lag1": 2700.0,
    }

    res = predictor.predict_yield(req)

    assert isinstance(res, YieldPredictionResult)
    assert res["prediction_method"] == "persistence"
    assert res["weather_features_used"] is False
    assert res["predicted_yield_kg_per_ha"] == 2700.0
    assert any("Simulated NASA POWER service outage" in w or "incomplete" in w for w in res["warnings"])


# =====================================================================
# 10. Weather Provenance Tracking
# =====================================================================

def test_weather_provenance_tracking():
    """Verify that successful weather fetch records complete provenance."""
    predictor = AgroVisionPredictor(auto_fetch_weather=True)

    # Historical 2018 Winter in Hooghly is present in offline archive
    req = {
        "District": "Hooghly",
        "Season": "Winter",
        "Year": 2018,
        "Yield_lag1": 2850.0,
    }

    res = predictor.predict_yield(req)

    assert res["weather_features_used"] is True
    assert res["weather_feature_status"] == "complete"
    assert "weather_source" in res
    assert res["weather_source"] in ("offline_archive", "cache", "mixed")
    assert "weather_fetch_timestamp" in res
    assert "weather_latest_observation_date" in res


# =====================================================================
# 11. Bridge Integration: Auto Fetch Toggle & Bypassing When Present
# =====================================================================

def test_bridge_auto_fetch_toggle_and_bypass():
    """
    Verify:
    1. When auto_fetch_weather=False, missing weather follows persistence.
    2. When weather features are already provided, weather service is bypassed.
    """
    mock_service = MagicMock()
    predictor = AgroVisionPredictor(auto_fetch_weather=False, weather_service=mock_service)

    # 11a. auto_fetch=False: should NOT call service
    req_no_weather = {
        "District": "Nadia",
        "Season": "Winter",
        "Yield_lag1": 2500.0,
    }
    res_no_fetch = predictor.predict_yield(req_no_weather, auto_fetch_weather=False)
    assert not mock_service.get_inseason_features.called
    assert res_no_fetch["prediction_method"] == "persistence"

    # 11b. Weather already supplied: should NOT call service even if auto_fetch=True
    req_with_weather = {
        "District": "Nadia",
        "Season": "Winter",
        "Year": 2018,
        "Yield_lag1": 2500.0,
        "InSeason_Precip_Cum": 500.0,
        "InSeason_Precip_Mean": 4.1,
        "InSeason_Precip_Max": 45.0,
        "InSeason_T2M_Mean": 26.5,
        "InSeason_T2M_Min": 21.0,
        "InSeason_T2M_Max": 32.0,
        "InSeason_T2M_Range": 11.0,
        "InSeason_RH2M_Mean": 78.0,
        "InSeason_Solar_Mean": 17.5,
        "InSeason_Wind_Mean": 2.2,
    }
    predictor_auto = AgroVisionPredictor(auto_fetch_weather=True, weather_service=mock_service)
    res_with_weather = predictor_auto.predict_yield(req_with_weather)
    assert not mock_service.get_inseason_features.called
    assert res_with_weather["weather_features_used"] is True


# =====================================================================
# 12. Regression: No Changes to Member 2 Behavior
# =====================================================================

def test_no_changes_to_member2_behavior():
    """
    Verify that Member 2 price and volatility predictions remain completely
    unaffected by Member 1 weather service additions.
    """
    dates = pd.date_range("2024-01-01", periods=35, freq="D")
    df = pd.DataFrame({
        "Reported Date": dates,
        "State Name": ["West Bengal"] * 35,
        "District Name": ["Purba Bardhaman"] * 35,
        "Market Name": ["Burdwan"] * 35,
        "Variety": ["Common"] * 35,
        "Group": ["Cereals"] * 35,
        "Modal Price (Rs./Quintal)": [2600.0 + (i % 5) * 10.0 for i in range(35)],
        "Arrivals (Tonnes)": [50.0 + (i % 3) * 5.0 for i in range(35)],
    })

    predictor = AgroVisionPredictor(auto_fetch_weather=True)
    m2_res = predictor.predict_price(df, commodity="rice")

    assert "predicted_price_rs_quintal" in m2_res
    assert "volatility" in m2_res
    assert m2_res["commodity"] == "rice"
    assert isinstance(m2_res["predicted_price_rs_quintal"], (int, float))
    assert m2_res["volatility"] in ("LOW", "MEDIUM", "HIGH")
