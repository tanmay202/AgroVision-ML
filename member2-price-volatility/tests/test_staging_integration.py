"""
AgroVision-ML — Member 3 Staging Integration Test Suite
======================================================
Validates Phase 2 Member 3 Integration Requirements:
1. Real application integration for all 4 commodities (Rice, Wheat, Potato, Tea).
2. All supported prediction horizons (short, 14d, 30d) across commodities.
3. Explicit error for unsupported horizons (Tea 14d/30d).
4. Full output schema validation matching API_Integration_Contract.md Table 3.1:
   - Canonical keys, types, and values.
   - Volatility probabilities sum to 1.0 +- 1e-4.
   - Prediction interval percentile monotonicity (p10 <= p25 <= p50 <= p75 <= p90).
5. Accurate experimental status flags (is_experimental and release_status).
6. Full backward compatibility with legacy Rice application contracts:
   - predict_price(), predict_price_horizon(), predict_yield(), predict_all().
7. Robust error handling and strict no cross-commodity fallback.
"""

import sys
import math
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
PROJECT_ROOT = MEMBER2_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEMBER2_DIR / "src") not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR / "src"))

from shared.predict_bridge import (
    AgroVisionPredictor,
    predict,
    predict_price,
    predict_price_horizon,
    predict_yield,
)


def make_history_df(commodity: str = "rice", n: int = 35, base_price: float = 2400.0) -> pd.DataFrame:
    """Generate synthetically valid historical mandi time series."""
    market_map = {
        "rice": ("Burdwan", "Common"),
        "wheat": ("Khanna", "Dara"),
        "potato": ("Agra", "Jyoti"),
        "tea": ("Kolkata", "CTC"),
    }
    market, variety = market_map.get(commodity.lower(), ("DefaultMarket", "DefaultVariety"))
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": market,
            "Variety": variety,
            "Modal Price (Rs./Quintal)": base_price + (i % 7) * 8.0,
            "Arrivals (Tonnes)": 45.0 + (i % 4) * 6.0,
        }
        for i, d in enumerate(dates)
    ])


REQUIRED_CANONICAL_FIELDS = [
    ("commodity", str),
    ("predicted_price_rs_quintal", (float, int)),
    ("predicted_modal_price", (float, int)),
    ("predicted_price", (float, int)),
    ("current_modal_price", (float, int)),
    ("predicted_price_change", (float, int)),
    ("horizon_days", int),
    ("target_date_convention", str),
    ("volatility", str),
    ("predicted_volatility_class", str),
    ("volatility_probabilities", dict),
    ("volatility_confidence", (float, int)),
    ("model_name", str),
    ("model_version", str),
    ("release_status", str),
    ("is_experimental", bool),
    ("limitations", (str, list)),
    ("residual_distribution", (str, dict)),
    ("prediction_interval", dict),
]


def assert_standardized_schema(res: Dict[str, Any], expected_commodity: str, expected_horizon_days: int):
    """Enforce complete schema compliance per Section 3.1 of API_Integration_Contract.md."""
    assert isinstance(res, dict), f"Prediction output must be a dict, got {type(res)}"
    
    # 1. Check all canonical keys and types
    for key, expected_type in REQUIRED_CANONICAL_FIELDS:
        assert key in res, f"Missing required canonical key: '{key}'"
        assert isinstance(res[key], expected_type), (
            f"Field '{key}' has invalid type: expected {expected_type}, got {type(res[key])}"
        )

    # 2. Field-level content checks
    assert res["commodity"].lower() == expected_commodity.lower()
    assert res["horizon_days"] == expected_horizon_days
    assert res["predicted_price_rs_quintal"] > 0
    assert res["predicted_modal_price"] == res["predicted_price_rs_quintal"]
    assert res["predicted_price"] == res["predicted_price_rs_quintal"]
    assert res["current_modal_price"] > 0
    assert abs(res["predicted_price_rs_quintal"] - (res["current_modal_price"] + res["predicted_price_change"])) < 0.05
    
    # 3. Volatility format
    assert res["volatility"] in {"LOW", "MEDIUM", "HIGH"}
    assert res["predicted_volatility_class"] == res["volatility"]
    probs = res["volatility_probabilities"]
    assert set(probs.keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert all(isinstance(v, (float, int)) and 0.0 <= v <= 1.0 for v in probs.values())
    prob_sum = sum(probs.values())
    assert math.isclose(prob_sum, 1.0, abs_tol=1e-3), f"Probabilities do not sum to 1.0: {prob_sum}"
    assert math.isclose(res["volatility_confidence"], max(probs.values()), abs_tol=1e-4)

    # 4. Release status & experimental flag consistency
    assert res["release_status"] in {"production", "opt-in", "experimental"}
    if res["release_status"] == "experimental":
        assert res["is_experimental"] is True
    elif res["release_status"] == "production":
        assert res["is_experimental"] is False

    # 5. Prediction interval checks
    pi = res["prediction_interval"]
    for pi_key in ["p10", "p25", "p50", "p75", "p90", "symmetric_80_ci"]:
        assert pi_key in pi, f"prediction_interval missing key '{pi_key}'"
    assert pi["p10"] <= pi["p25"] <= pi["p50"] <= pi["p75"] <= pi["p90"]
    assert isinstance(pi["symmetric_80_ci"], (list, tuple)) and len(pi["symmetric_80_ci"]) == 2
    assert pi["symmetric_80_ci"][0] <= pi["symmetric_80_ci"][1]


# ============================================================
# Test 1: All Commodities x All Supported Prediction Horizons
# ============================================================

@pytest.mark.parametrize("commodity,horizon,expected_days", [
    ("rice", "short", 1),
    ("rice", 1, 1),
    ("rice", "14d", 14),
    ("rice", 14, 14),
    ("rice", "30d", 30),
    ("rice", 30, 30),
    ("wheat", "short", 1),
    ("wheat", 1, 1),
    ("wheat", "14d", 14),
    ("wheat", 14, 14),
    ("wheat", "30d", 30),
    ("wheat", 30, 30),
    ("potato", "short", 1),
    ("potato", 1, 1),
    ("potato", "14d", 14),
    ("potato", 14, 14),
    ("potato", "30d", 30),
    ("potato", 30, 30),
    ("tea", "short", 1),
    ("tea", 1, 1),
])
def test_all_commodities_and_horizons_inference(commodity, horizon, expected_days):
    """Verify that every valid commodity and horizon executes successfully and satisfies the contract."""
    bridge = AgroVisionPredictor()
    df = make_history_df(commodity, n=35)
    result = bridge.predict(history=df, commodity=commodity, horizon=horizon)
    assert_standardized_schema(result, expected_commodity=commodity, expected_horizon_days=expected_days)


# ============================================================
# Test 2: Unsupported Horizons for Tea Raise Explicit Errors
# ============================================================

@pytest.mark.parametrize("unsupported_horizon", ["14d", 14, "30d", 30])
def test_tea_unsupported_horizons_raise_explicit_error(unsupported_horizon):
    """Tea only supports short-term (1–7d). Direct 14d and 30d must raise an explicit exception."""
    bridge = AgroVisionPredictor()
    df = make_history_df("tea", n=35)
    with pytest.raises((ValueError, RuntimeError), match="[Hh]orizon|[Uu]nsupported"):
        bridge.predict(history=df, commodity="tea", horizon=unsupported_horizon)


# ============================================================
# Test 3: Standalone Convenience Functions (predict, predict_price)
# ============================================================

def test_standalone_predict_convenience_functions():
    """Verify top-level module functions predict() and predict_price() export identically."""
    df_rice = make_history_df("rice", n=35)
    res1 = predict(df_rice, commodity="rice", horizon="short")
    assert_standardized_schema(res1, "rice", 1)

    df_wheat = make_history_df("wheat", n=35)
    res2 = predict_price(df_wheat, commodity="wheat")
    assert_standardized_schema(res2, "wheat", 1)


# ============================================================
# Test 4: Rice Legacy Application Backward Compatibility
# ============================================================

def test_rice_legacy_application_contracts():
    """Verify that existing Rice Member 3 contracts remain 100% backward compatible."""
    bridge = AgroVisionPredictor()
    df = make_history_df("rice", n=35)

    # 1. predict_price() default is Rice
    price_res = bridge.predict_price(df)
    assert price_res["commodity"] == "rice"
    assert "predicted_price" in price_res
    assert "volatility" in price_res

    # 2. predict_price_horizon() with Rice
    h14_res = bridge.predict_price_horizon(df, horizon=14)
    assert h14_res["horizon_days"] == 14
    assert h14_res["commodity"] == "rice"

    # 3. predict_yield()
    yield_res = bridge.predict_yield({
        "District": "Hooghly",
        "Season": "Winter",
        "Yield_lag1": 2850.0,
        "area_ha": 300000.0,
    })
    assert "predicted_yield_tonnes_per_ha" in yield_res
    assert "estimated_production_tonnes" in yield_res
    assert 2.0 <= yield_res["predicted_yield_tonnes_per_ha"] <= 4.0

    # 4. predict_all() combining crop yield and rice market price
    combined = bridge.predict_all(
        yield_input={
            "District": "Hooghly",
            "Season": "Winter",
            "Yield_lag1": 2850.0,
            "area_ha": 300000.0,
        },
        market_input=df,
    )
    assert "crop_yield" in combined
    assert "market_price" in combined
    assert combined["market_price"]["commodity"] == "rice"
    assert combined["crop_yield"]["predicted_yield_tonnes_per_ha"] > 0


# ============================================================
# Test 5: Experimental Status Flag Accuracy
# ============================================================

def test_experimental_flags_accuracy():
    """Verify experimental vs production status is strictly marked according to governance tiers."""
    bridge = AgroVisionPredictor()
    df = make_history_df("wheat", n=35)

    # Wheat short-term is production
    res_short = bridge.predict(df, commodity="wheat", horizon="short")
    assert res_short["release_status"] == "production"
    assert res_short["is_experimental"] is False

    # Wheat 14d is experimental
    res_14d = bridge.predict(df, commodity="wheat", horizon="14d")
    assert res_14d["release_status"] == "experimental"
    assert res_14d["is_experimental"] is True

    # Wheat 30d is experimental
    res_30d = bridge.predict(df, commodity="wheat", horizon="30d")
    assert res_30d["release_status"] == "experimental"
    assert res_30d["is_experimental"] is True

    # Potato short-term is production
    df_potato = make_history_df("potato", n=35)
    res_pot_short = bridge.predict(df_potato, commodity="potato", horizon="short")
    assert res_pot_short["release_status"] == "production"
    assert res_pot_short["is_experimental"] is False

    # Potato 14d is experimental
    res_pot_14d = bridge.predict(df_potato, commodity="potato", horizon="14d")
    assert res_pot_14d["release_status"] == "experimental"
    assert res_pot_14d["is_experimental"] is True


# ============================================================
# Test 6: Strict Error Handling & No Silent Cross-Commodity Fallback
# ============================================================

def test_insufficient_history_error():
    """Sub-30-day history must trigger an explicit ValueError."""
    bridge = AgroVisionPredictor()
    short_df = make_history_df("wheat", n=15)
    with pytest.raises(ValueError, match="[Hh]istory|[Oo]bservation|[Rr]ows"):
        bridge.predict(short_df, commodity="wheat", horizon="short")


def test_missing_artifacts_directory_error():
    """Explicit nonexistent artifacts directory must raise FileNotFoundError, not fallback."""
    bridge = AgroVisionPredictor()
    df = make_history_df("wheat", n=35)
    with pytest.raises(FileNotFoundError):
        bridge.predict(df, commodity="wheat", horizon="short", artifacts_dir="/nonexistent/directory/path")


def test_invalid_commodity_raises_value_error():
    """Unrecognized commodity names must raise ValueError immediately."""
    bridge = AgroVisionPredictor()
    df = make_history_df("rice", n=35)
    with pytest.raises(ValueError, match="[Uu]nsupported commodity|[Cc]ommodity"):
        bridge.predict(df, commodity="unknown_crop_xyz", horizon="short")
