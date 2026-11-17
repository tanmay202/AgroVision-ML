"""
AgroVision Model 1 Expansion: Multi-Crop Yield Integration Tests
================================================================
Comprehensive test suite verifying:
1. Wheat yield inference and output contract.
2. Potato yield inference and output contract.
3. Rice production model preservation and contract.
4. Universal predict_yield routing across Rice, Wheat, Potato.
5. Rejection of unsupported yield commodities (Tea, etc.).
6. Weather preprocessor and seasonal window enforcement.
7. Weather fallback and provenance tracking.
8. Conditional crop production (tonnes) strictly requiring area_ha > 0.
9. Bit-for-bit SHA-256 checksum verification of frozen Rice production model.
10. Non-regression of Member 2 multi-commodity price forecasting bridge.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.predict_bridge import (
    AgroVisionPredictor,
    predict_yield,
    predict_rice_yield,
    predict_wheat_yield,
    predict_potato_yield,
    predict_price,
)
from shared.paths import get_m1_models_dir


@pytest.fixture(scope="module")
def predictor():
    return AgroVisionPredictor(auto_fetch_weather=True)


def test_01_frozen_rice_model_checksum_unmodified():
    """Verify frozen Rice production model has NOT been modified or overwritten."""
    rice_model_path = get_m1_models_dir() / "m1_improved_delta_ridge.pkl"
    assert rice_model_path.exists(), "Rice production model artifact missing"
    
    sha256 = hashlib.sha256(open(rice_model_path, "rb").read()).hexdigest()
    certified_hash = "3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56"
    assert sha256 == certified_hash, (
        f"CRITICAL: Rice production model checksum mismatch! Expected {certified_hash}, got {sha256}"
    )


def test_02_rice_yield_prediction_contract(predictor):
    """Verify Rice yield prediction satisfies the common output contract."""
    sample_input = {
        "District": "Hooghly",
        "Season": "Winter",
        "Year": 2018,
        "Yield_lag1": 2950.0,
        "Yield_lag2": 2880.0,
    }
    res = predictor.predict_rice_yield(sample_input, area_ha=150.0)
    
    assert res["crop"] == "rice"
    assert res["district"] == "Hooghly"
    assert res["season"] == "Winter"
    assert res["release_status"] == "PRODUCTION"
    assert res["predicted_yield_kg_per_ha"] > 1000.0
    assert res["predicted_yield_tonnes_per_ha"] == round(res["predicted_yield_kg_per_ha"] / 1000.0, 4)
    assert res["estimated_production_tonnes"] == round(res["predicted_yield_tonnes_per_ha"] * 150.0, 4)
    assert "prediction_interval" in res
    assert "weather_provenance" in res


def test_03_wheat_yield_prediction_contract(predictor):
    """Verify Wheat yield prediction satisfies the common output contract."""
    sample_input = {
        "District": "Murshidabad",
        "Season": "Rabi",
        "Year": 2018,
        "Yield_lag1": 2450.0,
        "Yield_lag2": 2400.0,
        "Area_ha": 500.0,
    }
    res = predictor.predict_wheat_yield(sample_input)
    
    assert res["crop"] == "wheat"
    assert res["district"] == "Murshidabad"
    assert res["season"] == "Rabi"
    assert res["release_status"] in ("PRODUCTION READY", "EXPERIMENTAL")
    assert 1000.0 <= res["predicted_yield_kg_per_ha"] <= 4500.0
    assert res["predicted_yield_tonnes_per_ha"] == round(res["predicted_yield_kg_per_ha"] / 1000.0, 4)
    assert res["estimated_production_tonnes"] is not None
    assert res["estimated_production_tonnes"] > 0
    assert "p10_kg_per_ha" in res["prediction_interval"]
    assert "p90_kg_per_ha" in res["prediction_interval"]
    assert res["prediction_interval"]["p10_kg_per_ha"] < res["prediction_interval"]["p90_kg_per_ha"]


def test_04_potato_yield_prediction_contract(predictor):
    """Verify Potato yield prediction satisfies the common output contract."""
    sample_input = {
        "District": "Hooghly",
        "Season": "Rabi",
        "Year": 2018,
        "Yield_lag1": 24000.0,
        "Yield_lag2": 23500.0,
        "Area_ha": 1200.0,
    }
    res = predictor.predict_potato_yield(sample_input)
    
    assert res["crop"] == "potato"
    assert res["district"] == "Hooghly"
    assert res["season"] == "Rabi"
    assert res["release_status"] in ("PRODUCTION READY", "EXPERIMENTAL")
    assert 10000.0 <= res["predicted_yield_kg_per_ha"] <= 45000.0
    assert res["predicted_yield_tonnes_per_ha"] == round(res["predicted_yield_kg_per_ha"] / 1000.0, 4)
    assert res["estimated_production_tonnes"] is not None
    assert "p10_kg_per_ha" in res["prediction_interval"]
    assert "p90_kg_per_ha" in res["prediction_interval"]
    assert res["prediction_interval"]["p10_kg_per_ha"] < res["prediction_interval"]["p90_kg_per_ha"]


def test_05_predict_yield_universal_router(predictor):
    """Verify predict_yield routes correctly across commodities and crops."""
    # 1. Rice via default
    r_res = predictor.predict_yield({"District": "Bankura", "Season": "Winter", "Year": 2018, "Yield_lag1": 2600.0})
    assert r_res["crop"] == "rice"

    # 2. Wheat via commodity="wheat"
    w_res = predictor.predict_yield(
        {"District": "Bankura", "Season": "Rabi", "Year": 2018, "Yield_lag1": 2100.0},
        commodity="wheat"
    )
    assert w_res["crop"] == "wheat"

    # 3. Potato via crop="potato"
    p_res = predictor.predict_yield(
        {"District": "Hooghly", "Season": "Rabi", "Year": 2018, "Yield_lag1": 25000.0},
        crop="potato"
    )
    assert p_res["crop"] == "potato"


def test_06_unsupported_yield_commodity_raises_error(predictor):
    """Verify that unsupported commodities (like Tea) explicitly raise ValueError."""
    with pytest.raises(ValueError, match="is not supported for crop yield prediction"):
        predictor.predict_yield({"District": "Jalpaiguri"}, commodity="tea")

    with pytest.raises(ValueError, match="is not supported for crop yield prediction"):
        predictor.predict_yield({"District": "Jalpaiguri"}, crop="cotton")


def test_07_weather_cutoff_window_enforcement():
    """Verify crop-specific season cutoffs in weather preprocessor."""
    sys.path.insert(0, str((PROJECT_ROOT / "member1-yield-model" / "src").resolve()))
    from weather_preprocessor import get_season_window

    # Wheat: Nov 15 (year-1) to Jan 31 (year)
    w_start, w_cutoff, w_days = get_season_window(2019, "Rabi", crop="wheat")
    assert w_start == pd.Timestamp("2018-11-15")
    assert w_cutoff.strftime("%Y-%m-%d") == "2019-01-31"
    assert w_days == 78

    # Potato: Nov 1 (year-1) to Jan 31 (year)
    p_start, p_cutoff, p_days = get_season_window(2019, "Rabi", crop="potato")
    assert p_start == pd.Timestamp("2018-11-01")
    assert p_cutoff.strftime("%Y-%m-%d") == "2019-01-31"
    assert p_days == 92

    # Rice: Summer (Nov 1 to Feb 28)
    r_start, r_cutoff, r_days = get_season_window(2019, "Summer", crop="rice")
    assert r_start == pd.Timestamp("2018-11-01")
    assert r_cutoff.strftime("%Y-%m-%d") == "2019-02-28"
    assert r_days == 120


def test_08_missing_weather_persistence_fallback(predictor):
    """Verify safe fallback to persistence when weather cannot be acquired."""
    # Non-existent future year with auto_fetch_weather=False
    pred_no_weather = AgroVisionPredictor(auto_fetch_weather=False)
    
    w_res = pred_no_weather.predict_wheat_yield({
        "District": "Nadia",
        "Year": 2035,
        "Yield_lag1": 2345.6,
    })
    assert w_res["prediction_method"] == "wheat_persistence_fallback"
    assert w_res["predicted_yield_kg_per_ha"] == 2345.6
    assert any("Weather-informed model inactive" in w for w in w_res["warnings"])

    p_res = pred_no_weather.predict_potato_yield({
        "District": "Bardhaman",
        "Year": 2035,
        "Yield_lag1": 26543.2,
    })
    assert p_res["prediction_method"] == "potato_persistence_fallback"
    assert p_res["predicted_yield_kg_per_ha"] == 26543.2
    assert any("Weather-informed model inactive" in w for w in p_res["warnings"])


def test_09_production_calculation_conditional_on_area(predictor):
    """Verify that estimated_production_tonnes is only computed when area_ha > 0."""
    # When area is None, production is None
    res_no_area = predictor.predict_wheat_yield({"District": "Nadia", "Year": 2018, "Yield_lag1": 2200.0})
    assert res_no_area.get("estimated_production_tonnes") is None

    # When area is 0 or negative, production is None
    res_zero_area = predictor.predict_potato_yield({"District": "Hooghly", "Year": 2018, "Yield_lag1": 25000.0}, area_ha=0.0)
    assert res_zero_area.get("estimated_production_tonnes") is None

    # When valid area is provided
    res_valid_area = predictor.predict_potato_yield({"District": "Hooghly", "Year": 2018, "Yield_lag1": 25000.0}, area_ha=100.0)
    assert res_valid_area.get("estimated_production_tonnes") == round(res_valid_area["predicted_yield_tonnes_per_ha"] * 100.0, 2)


def test_10_member2_price_bridge_non_regression():
    """Verify that Member 2 multi-commodity price predictions continue working without regressions."""
    dates = pd.date_range(end="2024-05-15", periods=35)
    history = [
        {
            "Market Name": "Sainthia",
            "Variety": "Common",
            "Reported Date": str(d.date()),
            "Modal Price (Rs./Quintal)": 2200.0 + i * 5,
            "Arrivals (Tonnes)": 50.0
        }
        for i, d in enumerate(dates)
    ]
    price_res = predict_price(history, commodity="rice")
    assert "predicted_price" in price_res or "modal_price" in price_res or "predicted_modal_price" in price_res
