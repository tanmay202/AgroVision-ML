"""
Comprehensive Automated Test Suite: AgroVision Prediction Bridge Contract Repair
================================================================================
Validates all requirements of Step 2:
1. Correct kg/ha output.
2. Correct tonnes/ha output.
3. Unit conversion consistency (tonnes/ha = kg/ha / 1000).
4. Single prediction (dict).
5. Batch prediction (list of dicts and pandas DataFrame).
6. predict_all() output contract.
7. Separation of crop yield and market arrivals.
8. Deprecated alias behavior (yield_arrivals_tonnes with DeprecationWarning and configurable toggle).
9. Missing Area_ha behavior (estimated_production_tonnes omitted when area unavailable).
10. Optional crop production calculation (production tonnes = yield tonnes/ha * Area_ha).
11. Member 2 price prediction regression.
12. Member 2 volatility prediction regression.
13. Existing bridge integration (load_model, get_required_features).
14. Error handling for invalid predictions (no silent fallbacks to plausible values).
"""

import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure monorepo root is on sys.path
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import get_project_root, get_member1_root, get_member2_root
from shared.predict_bridge import (
    AgroVisionPredictor,
    YieldPredictionResult,
    CombinedPredictionResult,
    DEPRECATION_WARNING_MSG,
)


@pytest.fixture
def predictor():
    """Default AgroVisionPredictor instance."""
    return AgroVisionPredictor()


@pytest.fixture
def sample_yield_input():
    return {
        "District": "Hooghly",
        "Season": "Winter",
        "Yield_lag1": 2850.0,
        "InSeason_Precip_Cum": 120.0,
        "InSeason_T2M_Mean": 24.5,
    }


@pytest.fixture
def sample_rice_history():
    """Historical trading sessions suitable for Member 2 RiceInferencePipeline."""
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
    return df


# =========================================================================
# 1, 2, 3: Canonical Output Fields & Unit Conversion Consistency
# =========================================================================

def test_canonical_yield_units_and_mathematical_consistency(predictor, sample_yield_input):
    """
    Requirements 1, 2, 3:
    - predicted_yield_kg_per_ha must exist and represent kg/ha (~1500 - 4000 for WB rice).
    - predicted_yield_tonnes_per_ha must exist and represent tonnes/ha (~1.5 - 4.0).
    - tonnes_per_ha = kg_per_ha / 1000.0 must be mathematically exact within rounding precision.
    """
    res = predictor.predict_yield(sample_yield_input)

    assert isinstance(res, YieldPredictionResult)
    assert "predicted_yield_kg_per_ha" in res
    assert "predicted_yield_tonnes_per_ha" in res

    kg_ha = res["predicted_yield_kg_per_ha"]
    t_ha = res["predicted_yield_tonnes_per_ha"]

    assert isinstance(kg_ha, (int, float))
    assert isinstance(t_ha, (int, float))

    # Agronomic range verification for West Bengal rice
    assert 1500.0 <= kg_ha <= 4000.0, f"Expected WB rice yield in [1500, 4000] kg/ha, got {kg_ha}"
    assert 1.5 <= t_ha <= 4.0, f"Expected WB rice yield in [1.5, 4.0] tonnes/ha, got {t_ha}"

    # Exact unit conversion consistency: tonnes = kg / 1000
    assert abs(t_ha - (kg_ha / 1000.0)) < 1e-4
    assert np.isclose(kg_ha, t_ha * 1000.0, atol=0.1)


# =========================================================================
# 4, 5: Single, Batch, and DataFrame Prediction
# =========================================================================

def test_single_prediction_dict(predictor, sample_yield_input):
    """Requirement 4: Single dictionary prediction."""
    res = predictor.predict_yield(sample_yield_input)
    assert isinstance(res, dict)
    assert "predicted_yield_kg_per_ha" in res
    assert "predicted_yield_tonnes_per_ha" in res
    assert res["model_name"] == "m1_improved_delta_ridge.pkl"


def test_batch_prediction_list_of_dicts(predictor):
    """Requirement 5: Batch prediction with list of dictionaries."""
    records = [
        {"District": "Hooghly", "Season": "Winter", "Yield_lag1": 2800.0},
        {"District": "Bankura", "Season": "Summer", "Yield_lag1": 2900.0},
        {"District": "Birbhum", "Season": "Autumn", "Yield_lag1": 2600.0},
    ]
    results = predictor.predict_yield(records)
    assert isinstance(results, list)
    assert len(results) == 3

    for r in results:
        assert isinstance(r, YieldPredictionResult)
        assert "predicted_yield_kg_per_ha" in r
        assert "predicted_yield_tonnes_per_ha" in r
        assert abs(r["predicted_yield_tonnes_per_ha"] - (r["predicted_yield_kg_per_ha"] / 1000.0)) < 1e-4


def test_dataframe_prediction(predictor):
    """Requirement 5: Batch prediction with pandas DataFrame input."""
    df_in = pd.DataFrame([
        {"District": "Hooghly", "Season": "Winter", "Yield_lag1": 2800.0},
        {"District": "Purba Bardhaman", "Season": "Winter", "Yield_lag1": 3100.0},
    ])
    df_out = predictor.predict_yield(df_in)
    assert isinstance(df_out, pd.DataFrame)
    assert "predicted_yield_kg_per_ha" in df_out.columns
    assert "predicted_yield_tonnes_per_ha" in df_out.columns
    assert len(df_out) == 2

    for _, row in df_out.iterrows():
        assert abs(row["predicted_yield_tonnes_per_ha"] - (row["predicted_yield_kg_per_ha"] / 1000.0)) < 1e-4


# =========================================================================
# 6, 7: predict_all() Contract & Semantic Separation of Yield and Arrivals
# =========================================================================

def test_predict_all_output_contract_and_domain_separation(predictor, sample_rice_history, sample_yield_input):
    """
    Requirements 6 & 7:
    - predict_all() returns structured, distinctly separated domains:
      * 'crop_yield': Member 1 crop productivity intensity
      * 'market_price': Member 2 mandi price and volatility
    - Crop yield (tonnes/ha) must NEVER be conflated with mandi arrivals (tonnes).
    - Top-level canonical fields exposed cleanly.
    """
    result = predictor.predict_all(
        sample_rice_history,
        commodity="rice",
        yield_input=sample_yield_input,
    )

    assert isinstance(result, CombinedPredictionResult)

    # 1. Verify separated sub-domains
    assert "crop_yield" in result
    assert "market_price" in result

    yield_domain = result["crop_yield"]
    market_domain = result["market_price"]

    # Member 1 domain checks
    assert "predicted_yield_kg_per_ha" in yield_domain
    assert "predicted_yield_tonnes_per_ha" in yield_domain
    assert yield_domain["error"] is None
    assert 1.5 <= yield_domain["predicted_yield_tonnes_per_ha"] <= 4.0

    # Member 2 domain checks
    assert "predicted_price_rs_quintal" in market_domain
    assert "volatility" in market_domain
    assert market_domain["commodity"] == "rice"
    assert market_domain["error"] is None
    assert market_domain["predicted_price_rs_quintal"] > 2000.0
    assert market_domain["volatility"] in ["LOW", "MEDIUM", "HIGH"]

    # 2. Verify top-level canonical fields
    assert result["predicted_yield_kg_per_ha"] == yield_domain["predicted_yield_kg_per_ha"]
    assert result["predicted_yield_tonnes_per_ha"] == yield_domain["predicted_yield_tonnes_per_ha"]
    assert result["predicted_price_rs_quintal"] == market_domain["predicted_price_rs_quintal"]
    assert result["volatility"] == market_domain["volatility"]

    # 3. Verify separation: Crop yield is productivity (~2.8 t/ha),
    # while mandi arrivals are market volumes (~50 tonnes). They must NOT be conflated.
    latest_arrival = sample_rice_history["Arrivals (Tonnes)"].iloc[-1]
    assert not np.isclose(yield_domain["predicted_yield_tonnes_per_ha"], latest_arrival, rtol=0.2)


# =========================================================================
# 8: Deprecated Legacy Alias Behavior & Migration Warning
# =========================================================================

def test_deprecated_legacy_alias_emits_warning(predictor, sample_yield_input):
    """
    Requirement 8:
    - Accessing legacy 'yield_arrivals_tonnes' key must emit a DeprecationWarning.
    - Warning must explain conflation and cite removal policy.
    - Value must match predicted_yield_tonnes_per_ha.
    """
    res = predictor.predict_yield(sample_yield_input)

    # Key exists in dict when legacy aliases are enabled
    assert "yield_arrivals_tonnes" in res

    # Accessing via __getitem__ must emit DeprecationWarning
    with pytest.deprecated_call(match="yield_arrivals_tonnes.*deprecated"):
        val = res["yield_arrivals_tonnes"]

    assert val == res["predicted_yield_tonnes_per_ha"]

    # Accessing via .get() must also emit DeprecationWarning
    with pytest.deprecated_call(match="yield_arrivals_tonnes.*deprecated"):
        val_get = res.get("yield_arrivals_tonnes")

    assert val_get == res["predicted_yield_tonnes_per_ha"]


def test_configurable_legacy_alias_behavior(sample_yield_input):
    """
    Requirement 8:
    - Disabling legacy alias via enable_legacy_alias=False omits the key entirely.
    """
    # 1. Instance-level disable
    strict_predictor = AgroVisionPredictor(enable_legacy_alias=False)
    strict_res = strict_predictor.predict_yield(sample_yield_input)
    assert "yield_arrivals_tonnes" not in strict_res
    with pytest.raises(KeyError):
        _ = strict_res["yield_arrivals_tonnes"]

    # 2. Per-call disable override
    default_predictor = AgroVisionPredictor()
    call_strict_res = default_predictor.predict_yield(sample_yield_input, enable_legacy_alias=False)
    assert "yield_arrivals_tonnes" not in call_strict_res


# =========================================================================
# 9, 10: Optional Crop Production Calculation & Missing Area Behavior
# =========================================================================

def test_optional_crop_production_calculation(predictor, sample_yield_input):
    """
    Requirement 10:
    - When valid Area_ha is explicitly provided:
      estimated_production_tonnes = predicted_yield_tonnes_per_ha * Area_ha
    """
    # Test via input_dict Area_ha
    input_with_area = dict(sample_yield_input)
    input_with_area["Area_ha"] = 500.0  # 500 hectares

    res = predictor.predict_yield(input_with_area)
    assert "estimated_production_tonnes" in res
    assert "area_ha" in res
    assert res["area_ha"] == 500.0

    expected_prod = round(res["predicted_yield_tonnes_per_ha"] * 500.0, 4)
    assert np.isclose(res["estimated_production_tonnes"], expected_prod)

    # Test via explicit area_ha parameter
    res_param = predictor.predict_yield(sample_yield_input, area_ha=250.0)
    assert "estimated_production_tonnes" in res_param
    expected_prod_param = round(res_param["predicted_yield_tonnes_per_ha"] * 250.0, 4)
    assert np.isclose(res_param["estimated_production_tonnes"], expected_prod_param)


def test_missing_or_invalid_area_omits_production(predictor, sample_yield_input):
    """
    Requirement 9:
    - When Area_ha is missing, None, negative, or zero:
      'estimated_production_tonnes' must be strictly omitted.
    - Area_lag1 and Area_lag2 must NOT be mistakenly used as current area.
    """
    # 1. Completely missing area
    res_no_area = predictor.predict_yield(sample_yield_input)
    assert "estimated_production_tonnes" not in res_no_area

    # 2. Area_lag1 / Area_lag2 present (must NOT be used as current area)
    input_with_lags = dict(sample_yield_input)
    input_with_lags["Area_lag1"] = 12000.0
    input_with_lags["Area_lag2"] = 11500.0
    res_lags = predictor.predict_yield(input_with_lags)
    assert "estimated_production_tonnes" not in res_lags

    # 3. Negative area (invalid agronomic input)
    input_neg_area = dict(sample_yield_input)
    input_neg_area["Area_ha"] = -100.0
    res_neg = predictor.predict_yield(input_neg_area)
    assert "estimated_production_tonnes" not in res_neg
    assert any("Non-positive area" in w for w in res_neg["warnings"])

    # 4. Zero area
    input_zero_area = dict(sample_yield_input)
    input_zero_area["Area_ha"] = 0.0
    res_zero = predictor.predict_yield(input_zero_area)
    assert "estimated_production_tonnes" not in res_zero


# =========================================================================
# 11, 12: Member 2 Price & Volatility Regression Protection
# =========================================================================

def test_member2_price_prediction_regression(predictor, sample_rice_history):
    """Requirement 11: Member 2 price forecasting functionality is completely preserved."""
    res_price = predictor.predict_price(
        sample_rice_history,
        commodity="rice",
        market="Burdwan",
        variety="Common",
        prediction_date=sample_rice_history["Reported Date"].iloc[-1],
    )
    assert "predicted_price_rs_quintal" in res_price
    assert res_price["predicted_price_rs_quintal"] > 0
    assert res_price["commodity"] == "rice"
    assert "market" in res_price


def test_member2_volatility_prediction_regression(predictor, sample_rice_history):
    """Requirement 12: Member 2 volatility forecasting functionality is completely preserved."""
    res_vol = predictor.predict_price(sample_rice_history, commodity="rice")
    assert "volatility" in res_vol
    assert res_vol["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert "predicted_volatility_range" in res_vol


# =========================================================================
# 13: Existing Bridge Integration & Feature Discovery
# =========================================================================

def test_bridge_required_features_discovery(predictor):
    """Requirement 13: get_required_features() exposes both model requirements."""
    feats = predictor.get_required_features()
    assert "yield_features" in feats
    assert "price_features" in feats
    assert "volatility_features" in feats
    assert "rice_price_features" in feats
    assert len(feats["yield_features"]) == 20
    assert len(feats["rice_price_features"]) == 48


# =========================================================================
# 14: Error Handling For Invalid Predictions
# =========================================================================

def test_error_handling_for_invalid_prediction_inputs(predictor):
    """
    Requirement 14:
    - Passing non-dictionary/non-DataFrame input raises informative TypeError.
    - Missing models raise RuntimeError without silent fabrication.
    """
    # 1. Invalid input type
    with pytest.raises(TypeError, match="Expected dict, list of dicts, or DataFrame"):
        predictor.predict_yield(12345)

    # 2. Non-existent model path raises RuntimeError
    bad_predictor = AgroVisionPredictor(m1_model_path="non_existent_model_file.pkl")
    with pytest.raises(RuntimeError, match="Member 1 yield model or predict module is not available"):
        bad_predictor.predict_yield({"District": "Hooghly", "Season": "Winter"})


# =========================================================================
# 15: Edge Cases & Production Consistency Verifications
# =========================================================================

def test_crop_production_dataframe_and_single_dict_consistency(predictor, sample_yield_input):
    """
    Verify numerical consistency:
    - Single dict prediction with area_ha
    - List of dicts prediction with area_ha
    - DataFrame prediction with area_ha
    All must produce the exact same estimated_production_tonnes, exactly matching
    predicted_yield_tonnes_per_ha * area_ha.
    """
    area = 100.0
    input_with_area = dict(sample_yield_input)
    input_with_area["Area_ha"] = area

    # 1. Single dict
    res_dict = predictor.predict_yield(input_with_area)
    # 2. List of dicts
    res_list = predictor.predict_yield([input_with_area])
    # 3. DataFrame
    df_in = pd.DataFrame([input_with_area])
    res_df = predictor.predict_yield(df_in)

    dict_prod = res_dict["estimated_production_tonnes"]
    list_prod = res_list[0]["estimated_production_tonnes"]
    df_prod = float(res_df["estimated_production_tonnes"].iloc[0])

    expected_prod = round(res_dict["predicted_yield_tonnes_per_ha"] * area, 4)

    assert dict_prod == expected_prod, f"Dict production {dict_prod} != {expected_prod}"
    assert list_prod == expected_prod, f"List production {list_prod} != {expected_prod}"
    assert df_prod == expected_prod, f"DF production {df_prod} != {expected_prod}"


def test_predict_yield_dataframe_explicit_area_ha_parameter_override(predictor, sample_yield_input):
    """
    Verify that explicit area_ha parameter overrides or populates Area_ha in DataFrame.
    """
    # DataFrame with pre-existing Area_ha = 50.0
    df_in = pd.DataFrame([dict(sample_yield_input, Area_ha=50.0)])
    # Override with area_ha = 200.0
    res_df = predictor.predict_yield(df_in, area_ha=200.0)

    t_ha = res_df["predicted_yield_tonnes_per_ha"].iloc[0]
    expected_prod = round(t_ha * 200.0, 4)
    assert np.isclose(res_df["estimated_production_tonnes"].iloc[0], expected_prod)


def test_yield_prediction_result_pop_and_copy(sample_yield_input):
    """
    Verify pop() and copy() behavior on YieldPredictionResult:
    - pop('yield_arrivals_tonnes') emits DeprecationWarning when enabled
    - pop('yield_arrivals_tonnes') raises KeyError when disabled
    - copy() preserves YieldPredictionResult class and legacy alias toggle
    """
    predictor_default = AgroVisionPredictor(enable_legacy_alias=True)
    res = predictor_default.predict_yield(sample_yield_input)

    # Copy preserves type and behavior
    res_copy = res.copy()
    assert isinstance(res_copy, YieldPredictionResult)
    assert res_copy._enable_legacy_alias is True

    # pop() on enabled emits DeprecationWarning
    with pytest.deprecated_call(match="yield_arrivals_tonnes.*deprecated"):
        val_pop = res.pop("yield_arrivals_tonnes")
    assert val_pop == res["predicted_yield_tonnes_per_ha"]

    # pop() on disabled raises KeyError
    predictor_strict = AgroVisionPredictor(enable_legacy_alias=False)
    res_strict = predictor_strict.predict_yield(sample_yield_input)
    with pytest.raises(KeyError):
        res_strict.pop("yield_arrivals_tonnes")


def test_predict_all_crop_yield_domain_isolation_and_no_premature_warnings(predictor, sample_rice_history, sample_yield_input):
    """
    Verify:
    1. predict_all() does NOT prematurely trigger DeprecationWarning during execution
    2. 'crop_yield' sub-domain does NOT contain 'yield_arrivals_tonnes'
    3. Accessing top-level 'yield_arrivals_tonnes' still emits DeprecationWarning
    """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", DeprecationWarning)
        res = predictor.predict_all(
            sample_rice_history,
            commodity="rice",
            yield_input=sample_yield_input,
        )
        # Ensure no deprecation warnings were emitted during predict_all execution itself
        dep_w = [item for item in w if issubclass(item.category, DeprecationWarning) and "yield_arrivals_tonnes" in str(item.message)]
        assert len(dep_w) == 0, f"Unexpected deprecation warnings during predict_all: {dep_w}"

    # Verify crop_yield domain is strictly isolated
    assert "yield_arrivals_tonnes" not in res["crop_yield"]
    assert "predicted_yield_kg_per_ha" in res["crop_yield"]
    assert "predicted_yield_tonnes_per_ha" in res["crop_yield"]

    # Verify accessing top-level legacy key emits DeprecationWarning
    with pytest.deprecated_call(match="yield_arrivals_tonnes.*deprecated"):
        _ = res["yield_arrivals_tonnes"]


def test_area_edge_cases(predictor, sample_yield_input):
    """
    Verify extreme and unusual area edge cases:
    - Fractional area (0.0001 ha)
    - Numeric string area ('150.5')
    - Invalid string area ('abc')
    - Extreme float (1e6 ha)
    """
    # Fractional area
    res_frac = predictor.predict_yield(sample_yield_input, area_ha=0.0001)
    assert "estimated_production_tonnes" in res_frac
    assert res_frac["estimated_production_tonnes"] == round(res_frac["predicted_yield_tonnes_per_ha"] * 0.0001, 4)

    # Numeric string area
    res_str = predictor.predict_yield(sample_yield_input, area_ha="150.5")
    assert "estimated_production_tonnes" in res_str
    assert res_str["estimated_production_tonnes"] == round(res_str["predicted_yield_tonnes_per_ha"] * 150.5, 4)

    # Invalid string area
    res_invalid = predictor.predict_yield(sample_yield_input, area_ha="abc")
    assert "estimated_production_tonnes" not in res_invalid
    assert any("Non-numeric" in w for w in res_invalid["warnings"])

    # Extreme area
    res_huge = predictor.predict_yield(sample_yield_input, area_ha=100000.0)
    assert "estimated_production_tonnes" in res_huge
    assert res_huge["estimated_production_tonnes"] == round(res_huge["predicted_yield_tonnes_per_ha"] * 100000.0, 4)

