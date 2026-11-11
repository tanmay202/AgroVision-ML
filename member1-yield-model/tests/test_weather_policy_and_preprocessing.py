"""
Automated Test Suite: Weather Integration, In-Season Preprocessing, and Seasonal Policy
=======================================================================================
Step 3 Automated Verifications:
PART B & D: Weather availability policies ("persistence" vs "model_imputation"),
            weather completeness detection, provenance metadata reporting.
PART C: Reusable in-season weather preprocessor, strict cutoff gating (Aus, Aman, Boro),
        leakage prevention, post-cutoff exclusion, formula aggregation parity.
PART E: Provisional seasonal routing configuration (Summer-to-persistence policy),
        routing provenance, bridge propagation, single and batch consistency.
PART F: Canonical prediction contracts, dual-unit conversion, legacy alias compatibility,
        optional production calculation.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
M1_ROOT = TEST_DIR.parent
MONOREPO_ROOT = M1_ROOT.parent

if str(M1_ROOT) not in sys.path:
    sys.path.insert(0, str(M1_ROOT))
if str(M1_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(M1_ROOT / "scripts"))
if str(M1_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(M1_ROOT / "src"))
if str(MONOREPO_ROOT) not in sys.path:
    sys.path.insert(0, str(MONOREPO_ROOT))

from src.weather_preprocessor import (
    preprocess_inseason_weather,
    get_season_window,
    normalize_district_name,
    normalize_season_name,
    INSEASON_FEATURE_COLUMNS,
    SEASON_CONFIG,
)
from scripts.predict_rice_yield import (
    RiceYieldPredictor,
    predict_rice_yield,
    load_model,
    predict_yield,
    INSEASON_WEATHER_FEATURES,
    POLICY_VERSION,
    PROVISIONAL_SUMMER_ROUTING_POLICY,
)
from shared.predict_bridge import (
    AgroVisionPredictor,
    YieldPredictionResult,
    preprocess_inseason_weather as bridge_preprocess,
)


# =============================================================================
# Deterministic Fixtures (Synthetic, Zero Live API Calls)
# =============================================================================

@pytest.fixture
def complete_weather_features():
    """Exact 10 in-season weather features with valid realistic values."""
    return {
        'InSeason_Precip_Cum': 850.5,
        'InSeason_Precip_Mean': 9.24,
        'InSeason_Precip_Max': 75.2,
        'InSeason_T2M_Mean': 27.8,
        'InSeason_T2M_Min': 22.1,
        'InSeason_T2M_Max': 33.6,
        'InSeason_T2M_Range': 11.5,
        'InSeason_RH2M_Mean': 81.2,
        'InSeason_Solar_Mean': 17.5,
        'InSeason_Wind_Mean': 2.1,
    }


@pytest.fixture
def base_rice_input():
    """Historical baseline features without current weather."""
    return {
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': 2850.0,
        'Yield_lag2': 2800.0,
        'Area_lag1': 320000.0,
        'Area_lag2': 318000.0,
        'Precip_lag1': 1400.0,
        'T2M_lag1': 25.5,
        'RH2M_lag1': 75.0,
        'NDVI_lag1': 0.62,
    }


@pytest.fixture
def full_weather_input(base_rice_input, complete_weather_features):
    """Full input with all 8 historical lags and all 10 in-season weather features."""
    rec = dict(base_rice_input)
    rec.update(complete_weather_features)
    return rec


@pytest.fixture
def synthetic_daily_weather_series():
    """Generate daily weather covering Autumn, Winter, and Summer seasons for Hooghly."""
    dates = pd.date_range("2018-01-01", "2018-12-31", freq="D")
    records = []
    for i, d in enumerate(dates):
        records.append({
            "Date": d.strftime("%Y-%m-%d"),
            "District": "Hooghly",
            "PRECTOTCORR": float(i % 15),
            "T2M": 26.0 + (i % 8) * 0.5,
            "T2M_MIN": 20.0 + (i % 5) * 0.4,
            "T2M_MAX": 32.0 + (i % 6) * 0.6,
            "RH2M": 70.0 + (i % 25),
            "ALLSKY_SFC_SW_DWN": 15.0 + (i % 10),
            "WS2M": 1.5 + (i % 4) * 0.3,
        })
    return pd.DataFrame(records)


# =============================================================================
# PART B & D TESTS: Weather Availability Policy & Provenance
# =============================================================================

def test_complete_weather_uses_ml_model(full_weather_input):
    """When all 10 in-season features are available, the ML model is used."""
    predictor = RiceYieldPredictor(weather_missing_policy="persistence", enable_seasonal_routing=False)
    res = predictor.predict(full_weather_input)

    assert res["weather_feature_status"] == "complete"
    assert res["weather_features_used"] is True
    assert res["missing_weather_features"] == []
    assert res["prediction_method"] == predictor.model_name.replace(".pkl", "")
    assert res["predicted_delta_kg_ha"] != 0.0
    assert res["prediction_policy_version"] == POLICY_VERSION


def test_missing_weather_persistence_policy(base_rice_input):
    """When all 10 weather features are missing under persistence policy, return baseline yield with delta 0."""
    predictor = RiceYieldPredictor(weather_missing_policy="persistence", enable_seasonal_routing=False)
    res = predictor.predict(base_rice_input)

    assert res["weather_feature_status"] == "missing"
    assert res["weather_features_used"] is False
    assert len(res["missing_weather_features"]) == 10
    assert res["prediction_method"] == "persistence"
    assert res["predicted_delta_kg_ha"] == 0.0
    assert res["predicted_yield_kg_per_ha"] == base_rice_input["Yield_lag1"]
    assert any("In-season weather features unavailable; persistence fallback applied" in w for w in res["warnings"])


def test_partial_weather_persistence_policy(base_rice_input):
    """When some weather features are provided but incomplete, persistence fallback is applied."""
    predictor = RiceYieldPredictor(weather_missing_policy="persistence", enable_seasonal_routing=False)
    partial_input = dict(base_rice_input)
    partial_input["InSeason_Precip_Cum"] = 450.0
    partial_input["InSeason_T2M_Mean"] = 26.5

    res = predictor.predict(partial_input)

    assert res["weather_feature_status"] == "incomplete"
    assert res["weather_features_used"] is False
    assert len(res["missing_weather_features"]) == 8
    assert "InSeason_Precip_Mean" in res["missing_weather_features"]
    assert "InSeason_Precip_Cum" not in res["missing_weather_features"]
    assert res["prediction_method"] == "persistence"
    assert res["predicted_delta_kg_ha"] == 0.0
    assert any("Incomplete in-season weather features" in w for w in res["warnings"])


def test_explicit_model_imputation_mode(base_rice_input):
    """When model_imputation is explicitly selected, pipeline imputation is preserved with clear warning."""
    predictor = RiceYieldPredictor(weather_missing_policy="model_imputation", enable_seasonal_routing=False)
    res = predictor.predict(base_rice_input)

    assert res["weather_feature_status"] == "missing"
    assert res["weather_features_used"] is False  # Never falsely describe imputed as weather-observed
    assert res["prediction_method"] == "model_imputation"
    assert res["predicted_delta_kg_ha"] != 0.0
    assert any("values filled using training-set median imputation" in w for w in res["warnings"])
    assert any("Warning: This prediction does NOT represent observed weather" in w for w in res["warnings"])


def test_batch_independent_weather_handling(base_rice_input, complete_weather_features):
    """Batch inputs with different weather availability work independently for each row."""
    # Row 0: Complete weather
    r0 = dict(base_rice_input)
    r0.update(complete_weather_features)
    r0["District"] = "Hooghly"
    r0["Season"] = "Winter"

    # Row 1: Completely missing weather
    r1 = dict(base_rice_input)
    r1["District"] = "Hooghly"
    r1["Season"] = "Winter"

    # Row 2: Partial weather
    r2 = dict(base_rice_input)
    r2["District"] = "Hooghly"
    r2["Season"] = "Winter"
    r2["InSeason_Precip_Cum"] = 500.0

    predictor = RiceYieldPredictor(weather_missing_policy="persistence", enable_seasonal_routing=False)
    df_batch = pd.DataFrame([r0, r1, r2])
    out_df = predictor.predict(df_batch)

    # Row 0 uses ML delta model
    assert out_df.iloc[0]["weather_feature_status"] == "complete"
    assert bool(out_df.iloc[0]["weather_features_used"]) is True
    assert out_df.iloc[0]["predicted_delta_kg_ha"] != 0.0
    assert out_df.iloc[0]["prediction_method"] != "persistence"

    # Row 1 uses persistence fallback
    assert out_df.iloc[1]["weather_feature_status"] == "missing"
    assert bool(out_df.iloc[1]["weather_features_used"]) is False
    assert out_df.iloc[1]["predicted_delta_kg_ha"] == 0.0
    assert out_df.iloc[1]["prediction_method"] == "persistence"

    # Row 2 uses persistence fallback due to incomplete weather
    assert out_df.iloc[2]["weather_feature_status"] == "incomplete"
    assert bool(out_df.iloc[2]["weather_features_used"]) is False
    assert out_df.iloc[2]["predicted_delta_kg_ha"] == 0.0
    assert out_df.iloc[2]["prediction_method"] == "persistence"


# =============================================================================
# PART C TESTS: In-Season Weather Preprocessing & Cutoff Gating
# =============================================================================

def test_aus_cutoff_enforcement():
    """Autumn (Aus): Cutoff is strictly July 31. Observations after July 31 are excluded."""
    start_date, cutoff_date, expected_days = get_season_window(2018, "Autumn")
    assert start_date == pd.Timestamp("2018-05-01")
    assert cutoff_date.date() == pd.Timestamp("2018-07-31").date()
    assert expected_days == 92

    # Daily data including pre-season (April), in-season (May-July), and post-cutoff (August-September)
    dates = pd.date_range("2018-04-01", "2018-09-30", freq="D")
    records = []
    for d in dates:
        records.append({
            "Date": d.strftime("%Y-%m-%d"),
            "District": "Hooghly",
            "PRECTOTCORR": 10.0,
            "T2M": 28.0,
            "T2M_MIN": 24.0,
            "T2M_MAX": 32.0,
            "RH2M": 80.0,
            "ALLSKY_SFC_SW_DWN": 18.0,
            "WS2M": 2.0,
        })

    prep = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Aus")
    meta = prep["metadata"]

    assert meta["observed_days"] == 92
    assert meta["excluded_post_cutoff_count"] == 61  # 31 in Aug + 30 in Sep
    assert meta["excluded_pre_season_count"] == 30    # 30 in April
    assert meta["max_observation_date"] == "2018-07-31"
    assert prep["is_complete"] is True
    # In-season precipitation sum for 92 days @ 10 mm/day = 920.0 mm
    assert prep["features"]["InSeason_Precip_Cum"] == 920.0


def test_aman_cutoff_enforcement():
    """Winter (Aman): Cutoff is strictly September 30. Observations after Sep 30 are excluded."""
    start_date, cutoff_date, expected_days = get_season_window(2018, "Winter")
    assert start_date == pd.Timestamp("2018-06-01")
    assert cutoff_date.date() == pd.Timestamp("2018-09-30").date()
    assert expected_days == 122

    # Include October observations
    dates = pd.date_range("2018-06-01", "2018-10-31", freq="D")
    records = [{
        "Date": d.strftime("%Y-%m-%d"),
        "District": "Purba Bardhaman",
        "PRECTOTCORR": 5.0, "T2M": 27.0, "T2M_MIN": 22.0, "T2M_MAX": 32.0,
        "RH2M": 85.0, "ALLSKY_SFC_SW_DWN": 16.0, "WS2M": 1.8,
    } for d in dates]

    prep = preprocess_inseason_weather(records, district="Purba Bardhaman", year=2018, season="Aman")
    assert prep["metadata"]["observed_days"] == 122
    assert prep["metadata"]["excluded_post_cutoff_count"] == 31  # October days excluded
    assert prep["metadata"]["max_observation_date"] == "2018-09-30"
    assert prep["is_complete"] is True
    assert prep["features"]["InSeason_Precip_Cum"] == 122 * 5.0


def test_boro_cutoff_enforcement():
    """Summer (Boro): Starts Nov 1 of (year-1), cutoff strictly Feb 28 of (year)."""
    start_date, cutoff_date, expected_days = get_season_window(2018, "Summer")
    assert start_date == pd.Timestamp("2017-11-01")
    assert cutoff_date.date() == pd.Timestamp("2018-02-28").date()
    assert expected_days == 120

    dates = pd.date_range("2017-11-01", "2018-04-30", freq="D")
    records = [{
        "Date": d.strftime("%Y-%m-%d"),
        "District": "Bankura",
        "PRECTOTCORR": 2.0, "T2M": 22.0, "T2M_MIN": 15.0, "T2M_MAX": 29.0,
        "RH2M": 65.0, "ALLSKY_SFC_SW_DWN": 19.0, "WS2M": 2.2,
    } for d in dates]

    prep = preprocess_inseason_weather(records, district="Bankura", year=2018, season="Boro")
    assert prep["metadata"]["observed_days"] == 120
    assert prep["metadata"]["max_observation_date"] == "2018-02-28"
    assert prep["metadata"]["excluded_post_cutoff_count"] == 61  # March + April
    assert prep["is_complete"] is True


def test_future_date_rejection_and_max_allowed_date():
    """Observations past cutoff or user-specified max_allowed_date are strictly excluded."""
    dates = pd.date_range("2018-05-01", "2018-07-31", freq="D")
    records = [{
        "Date": d.strftime("%Y-%m-%d"),
        "District": "Hooghly",
        "PRECTOTCORR": 10.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0,
        "RH2M": 75.0, "ALLSKY_SFC_SW_DWN": 18.0, "WS2M": 2.0
    } for d in dates]

    # Constrain to June 30 barrier
    prep = preprocess_inseason_weather(
        records, district="Hooghly", year=2018, season="Autumn", max_allowed_date="2018-06-30"
    )
    assert prep["metadata"]["observed_days"] == 61  # May (31) + June (30)
    assert prep["metadata"]["max_observation_date"] == "2018-06-30"
    assert any("Leakage protection: Excluded 31 observation(s)" in w for w in prep["warnings"])


def test_invalid_date_and_sentinel_handling():
    """Unparseable dates are safely dropped; NASA POWER -999 sentinels are converted to NaN."""
    records = [
        {"Date": "invalid-date", "PRECTOTCORR": 10.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 75.0, "ALLSKY_SFC_SW_DWN": 18.0, "WS2M": 2.0},
        {"Date": "2018-05-01", "PRECTOTCORR": -999.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 75.0, "ALLSKY_SFC_SW_DWN": 18.0, "WS2M": 2.0},
        {"Date": "2018-05-02", "PRECTOTCORR": 15.0, "T2M": -999, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 75.0, "ALLSKY_SFC_SW_DWN": 18.0, "WS2M": 2.0},
    ]
    prep = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Autumn")
    assert any("Found 1 observation(s) with invalid or unparseable dates" in w for w in prep["warnings"])
    assert prep["metadata"]["observed_days"] == 2
    # PRECTOTCORR sum ignores NaN from -999 and sums 15.0
    assert prep["features"]["InSeason_Precip_Cum"] == 15.0


def test_exact_feature_names_and_formula_parity():
    """Aggregated features match the exact 10 training feature definitions and math formulas."""
    records = [
        {"Date": "2018-05-01", "PRECTOTCORR": 10.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 70.0, "ALLSKY_SFC_SW_DWN": 15.0, "WS2M": 2.0},
        {"Date": "2018-05-02", "PRECTOTCORR": 20.0, "T2M": 27.0, "T2M_MIN": 22.0, "T2M_MAX": 32.0, "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 17.0, "WS2M": 4.0},
    ]
    prep = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Autumn")
    feats = prep["features"]

    assert list(feats.keys()) == INSEASON_FEATURE_COLUMNS
    assert feats["InSeason_Precip_Cum"] == 30.0
    assert feats["InSeason_Precip_Mean"] == 15.0
    assert feats["InSeason_Precip_Max"] == 20.0
    assert feats["InSeason_T2M_Mean"] == 26.0
    assert feats["InSeason_T2M_Min"] == 20.0
    assert feats["InSeason_T2M_Max"] == 32.0
    assert feats["InSeason_T2M_Range"] == 12.0  # 32.0 - 20.0
    assert feats["InSeason_RH2M_Mean"] == 75.0
    assert feats["InSeason_Solar_Mean"] == 16.0
    assert feats["InSeason_Wind_Mean"] == 3.0


# =============================================================================
# PART E TESTS: Seasonal Policy Routing Configuration & Provenance
# =============================================================================

def test_summer_provisional_routing_enabled(full_weather_input):
    """Under provisional routing, Summer yield is routed to persistence even with complete weather."""
    predictor = RiceYieldPredictor(enable_seasonal_routing=True, weather_missing_policy="persistence")
    summer_input = dict(full_weather_input)
    summer_input["Season"] = "Summer"

    res = predictor.predict(summer_input)

    assert res["seasonal_routing_applied"] is True
    assert res["seasonal_routing_policy"] == PROVISIONAL_SUMMER_ROUTING_POLICY
    assert res["prediction_method"] == PROVISIONAL_SUMMER_ROUTING_POLICY
    assert res["predicted_delta_kg_ha"] == 0.0
    assert res["predicted_yield_kg_per_ha"] == summer_input["Yield_lag1"]
    assert res["weather_features_used"] is False  # Bypassed by seasonal policy
    assert any("Provisional seasonal routing policy applied" in w for w in res["warnings"])
    assert any("motivated by historical 2017-2019 test set evaluations" in w for w in res["warnings"])
    assert any("provisional" in w.lower() for w in res["warnings"])


def test_autumn_and_winter_not_intercepted_by_seasonal_routing(full_weather_input):
    """Autumn and Winter with complete weather run through ML model regardless of seasonal routing setting."""
    predictor = RiceYieldPredictor(enable_seasonal_routing=True, weather_missing_policy="persistence")

    for season in ["Autumn", "Winter"]:
        inp = dict(full_weather_input)
        inp["Season"] = season
        res = predictor.predict(inp)

        assert res["seasonal_routing_applied"] is False
        assert res["seasonal_routing_policy"] == PROVISIONAL_SUMMER_ROUTING_POLICY
        assert res["weather_features_used"] is True
        assert res["weather_feature_status"] == "complete"
        assert res["predicted_delta_kg_ha"] != 0.0


def test_seasonal_routing_disabled_allows_summer_model(full_weather_input):
    """When seasonal routing is explicitly disabled, Summer runs through ML model if weather is complete."""
    predictor = RiceYieldPredictor(enable_seasonal_routing=False, weather_missing_policy="persistence")
    summer_input = dict(full_weather_input)
    summer_input["Season"] = "Summer"

    res = predictor.predict(summer_input)

    assert res["seasonal_routing_applied"] is False
    assert res["seasonal_routing_policy"] == "disabled"
    assert res["weather_features_used"] is True
    assert res["prediction_method"] != PROVISIONAL_SUMMER_ROUTING_POLICY
    assert res["predicted_delta_kg_ha"] != 0.0


def test_bridge_seasonal_routing_propagation(full_weather_input):
    """AgroVisionPredictor correctly passes seasonal routing configuration down to RiceYieldPredictor."""
    # 1. Default AgroVisionPredictor has provisional routing enabled
    bridge_default = AgroVisionPredictor()
    summer_inp = dict(full_weather_input)
    summer_inp["Season"] = "Summer"

    res_default = bridge_default.predict_yield(summer_inp)
    assert res_default["seasonal_routing_applied"] is True
    assert res_default["prediction_method"] == PROVISIONAL_SUMMER_ROUTING_POLICY
    assert res_default["predicted_delta_kg_ha"] == 0.0

    # 2. Explicitly disabled routing through bridge instance
    bridge_disabled = AgroVisionPredictor(enable_seasonal_routing=False)
    res_disabled = bridge_disabled.predict_yield(summer_inp)
    assert res_disabled["seasonal_routing_applied"] is False
    assert res_disabled["predicted_delta_kg_ha"] != 0.0

    # 3. Per-call override through predict_yield()
    res_call_override = bridge_default.predict_yield(summer_inp, enable_seasonal_routing=False)
    assert res_call_override["seasonal_routing_applied"] is False
    assert res_call_override["predicted_delta_kg_ha"] != 0.0


def test_single_and_batch_routing_consistency(full_weather_input):
    """Single-record dict, list of dicts, and DataFrame produce identical predictions and routing provenance."""
    bridge = AgroVisionPredictor(enable_seasonal_routing=True)
    summer_inp = dict(full_weather_input)
    summer_inp["Season"] = "Summer"

    # Single dict
    res_single = bridge.predict_yield(summer_inp)

    # List of dicts
    res_list = bridge.predict_yield([summer_inp])

    # DataFrame
    res_df = bridge.predict_yield(pd.DataFrame([summer_inp]))

    assert res_single["predicted_yield_kg_per_ha"] == res_list[0]["predicted_yield_kg_per_ha"]
    assert res_single["predicted_yield_kg_per_ha"] == res_df.iloc[0]["predicted_yield_kg_per_ha"]
    assert res_single["prediction_method"] == res_list[0]["prediction_method"]
    assert res_single["prediction_method"] == res_df.iloc[0]["prediction_method"]
    assert res_single["seasonal_routing_applied"] == res_list[0]["seasonal_routing_applied"]
    assert res_single["seasonal_routing_applied"] == res_df.iloc[0]["seasonal_routing_applied"]


# =============================================================================
# PART F & INTEGRATION TESTS
# =============================================================================

def test_bridge_convenience_preprocessor_export(synthetic_daily_weather_series):
    """Convenience preprocess_inseason_weather export on shared.predict_bridge functions identically."""
    res_direct = preprocess_inseason_weather(synthetic_daily_weather_series, "Hooghly", 2018, "Winter")
    res_bridge = bridge_preprocess(synthetic_daily_weather_series, "Hooghly", 2018, "Winter")

    assert res_direct["features"] == res_bridge["features"]
    assert res_direct["metadata"]["observed_days"] == res_bridge["metadata"]["observed_days"]
    assert res_direct["is_complete"] == res_bridge["is_complete"]


def test_optional_crop_production_preserved_with_new_provenance(full_weather_input):
    """When Area_ha is provided, estimated_production_tonnes is calculated alongside new provenance metadata."""
    bridge = AgroVisionPredictor()
    inp = dict(full_weather_input)
    inp["Area_ha"] = 50.0

    res = bridge.predict_yield(inp)
    assert "estimated_production_tonnes" in res
    assert res["estimated_production_tonnes"] == round(res["predicted_yield_tonnes_per_ha"] * 50.0, 4)
    assert "prediction_method" in res
    assert "weather_feature_status" in res
    assert "seasonal_routing_applied" in res


def test_weather_preprocessor_mismatched_district_filters_correctly():
    """Daily weather dataset containing only 'Bankura' records must yield 0 observed days when queried for 'Hooghly'."""
    df_bankura = pd.DataFrame([{
        "Date": "2018-07-01",
        "District": "Bankura",
        "PRECTOTCORR": 10.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0,
        "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 15.0, "WS2M": 2.0,
    }])
    res = preprocess_inseason_weather(df_bankura, district="Hooghly", year=2018, season="Winter")
    assert res["metadata"]["observed_days"] == 0
    assert res["is_complete"] is False
    assert np.isnan(res["features"]["InSeason_Precip_Cum"])
    assert any("No rows matching district 'Hooghly'" in w for w in res["warnings"])


def test_weather_preprocessor_timezone_aware_dates():
    """Timezone-aware ISO timestamp strings or datetime objects are safely parsed without comparison crashes."""
    records = [{
        "Date": "2018-07-01T00:00:00Z",
        "District": "Hooghly",
        "PRECTOTCORR": 12.0, "T2M": 26.0, "T2M_MIN": 21.0, "T2M_MAX": 31.0,
        "RH2M": 82.0, "ALLSKY_SFC_SW_DWN": 16.0, "WS2M": 2.5,
    }]
    res = preprocess_inseason_weather(
        records, district="Hooghly", year=2018, season="Winter", max_allowed_date="2018-08-01T00:00:00+00:00"
    )
    assert res["metadata"]["observed_days"] == 1
    assert res["is_complete"] is True
    assert res["features"]["InSeason_Precip_Cum"] == 12.0


def test_weather_preprocessor_sub_daily_timestamps_aggregation():
    """Multiple observations with time components on the same date are normalized to a single daily average."""
    records = [
        {"Date": "2018-07-01 06:00:00", "District": "Hooghly", "PRECTOTCORR": 10.0, "T2M": 24.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 10.0, "WS2M": 2.0},
        {"Date": "2018-07-01 18:00:00", "District": "Hooghly", "PRECTOTCORR": 20.0, "T2M": 28.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 70.0, "ALLSKY_SFC_SW_DWN": 20.0, "WS2M": 4.0},
    ]
    res = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Winter")
    assert res["metadata"]["observed_days"] == 1
    assert res["features"]["InSeason_Precip_Cum"] == 15.0  # (10 + 20) / 2
    assert res["features"]["InSeason_T2M_Mean"] == 26.0   # (24 + 28) / 2


def test_weather_preprocessor_physical_consistency_warning():
    """When T2M_MIN exceeds T2M_MAX in observation records, an explicit data quality warning is emitted."""
    records = [
        {"Date": "2018-07-01", "District": "Hooghly", "PRECTOTCORR": 5.0, "T2M": 25.0, "T2M_MIN": 35.0, "T2M_MAX": 20.0, "RH2M": 75.0, "ALLSKY_SFC_SW_DWN": 15.0, "WS2M": 2.0},
    ]
    res = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Winter")
    assert any("Physical inconsistency: T2M_MIN exceeds T2M_MAX" in w for w in res["warnings"])


def test_case_insensitive_and_alias_inference_input():
    """Input dictionaries with lowercase keys ('district', 'season', 'yield_lag1') are properly parsed."""
    predictor = RiceYieldPredictor(weather_missing_policy="persistence", enable_seasonal_routing=False)
    inp = {
        "district": "hooghly",
        "season": "winter",
        "yield_lag1": 2850.0,
        "area_ha": 120.0,
    }
    res = predictor.predict(inp)
    assert res["baseline_yield_kg_ha"] == 2850.0
    assert res["is_yield_lag1_imputed"] is False
    assert res["predicted_yield_kg_per_ha"] == 2850.0
    assert res["prediction_method"] == "persistence"
    assert res["estimated_production_tonnes"] == round(2.85 * 120.0, 4)
    # Ensure no warnings about Unknown District or Unknown Season
    assert not any("Unseen District" in w for w in res["warnings"])
    assert not any("Unseen Season" in w for w in res["warnings"])


def test_bridge_predict_all_policy_overrides():
    """predict_all correctly passes enable_seasonal_routing and weather_missing_policy to predict_yield."""
    bridge = AgroVisionPredictor(enable_seasonal_routing=True, weather_missing_policy="persistence")
    tea_df = pd.DataFrame([{
        "Date": "2019-01-01",
        "Market": "Siliguri",
        "Arrivals_Tonnes": 50.0,
        "Avg_Price_Rs_Kg": 140.0,
    }])
    summer_yield_input = {
        "District": "Hooghly",
        "Season": "Summer",
        "Yield_lag1": 3100.0,
    }
    # With default routing, Summer routes to persistence
    res1 = bridge.predict_all(tea_df, commodity="tea", yield_input=summer_yield_input)
    assert res1["crop_yield"]["seasonal_routing_applied"] is True
    assert res1["crop_yield"]["prediction_method"] == PROVISIONAL_SUMMER_ROUTING_POLICY

    # With override enable_seasonal_routing=False, Summer persistence is not applied
    res2 = bridge.predict_all(
        tea_df, commodity="tea", yield_input=summer_yield_input, enable_seasonal_routing=False
    )
    assert res2["crop_yield"]["seasonal_routing_applied"] is False
    assert res2["crop_yield"]["prediction_method"] == "persistence"  # because weather missing


def test_dataframe_batch_has_seasonal_routing_policy_column(full_weather_input):
    """DataFrame batch prediction output contains seasonal_routing_policy column matching dict schema."""
    predictor = RiceYieldPredictor(enable_seasonal_routing=True)
    df = pd.DataFrame([full_weather_input])
    out_df = predictor.predict(df)
    assert "seasonal_routing_policy" in out_df.columns
    assert out_df.iloc[0]["seasonal_routing_policy"] == PROVISIONAL_SUMMER_ROUTING_POLICY


def test_composite_and_noun_first_district_aliases():
    """Verify noun-first and phonetic district aliases map to canonical districts without warnings."""
    from src.weather_preprocessor import normalize_district_name
    test_cases = {
        'Burdwan East': 'Purba Bardhaman',
        'Burdwan West': 'Paschim Bardhaman',
        'Bardhaman East': 'Purba Bardhaman',
        'Bardhaman West': 'Paschim Bardhaman',
        'Midnapore East': 'Purba Medinipur',
        'Midnapore West': 'Paschim Medinipur',
        'North 24 Paraganas': 'North 24 Parganas',
        'South 24 Paraganas': 'South 24 Parganas',
    }
    for alias, canon in test_cases.items():
        assert normalize_district_name(alias) == canon
        assert RiceYieldPredictor._normalize_district(alias) == canon

    # Verify end-to-end baseline yield lookup without 'Unseen District' warning
    predictor = RiceYieldPredictor(weather_missing_policy="persistence")
    res = predictor.predict({"District": "Burdwan East", "Season": "Winter"})
    assert res["baseline_yield_kg_ha"] == 2966.30
    assert not any("Unseen District" in w for w in res["warnings"])


def test_sub_daily_conflicting_rainfall_observations():
    """Verify sub-daily observations with extreme outlier differences are averaged cleanly."""
    records = [
        {"Date": "2018-06-01 06:00:00", "District": "Hooghly", "PRECTOTCORR": 0.0, "T2M": 25.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 15.0, "WS2M": 2.0},
        {"Date": "2018-06-01 18:00:00", "District": "Hooghly", "PRECTOTCORR": 250.0, "T2M": 27.0, "T2M_MIN": 20.0, "T2M_MAX": 30.0, "RH2M": 85.0, "ALLSKY_SFC_SW_DWN": 17.0, "WS2M": 2.5},
    ]
    res = preprocess_inseason_weather(records, district="Hooghly", year=2018, season="Winter")
    assert res["metadata"]["observed_days"] == 1
    assert res["features"]["InSeason_Precip_Cum"] == 125.0  # (0 + 250) / 2
    assert res["features"]["InSeason_Precip_Mean"] == 125.0
    assert res["features"]["InSeason_T2M_Mean"] == 26.0
    assert any("Duplicate date entries found" in w for w in res["warnings"])


def test_post_cutoff_weather_with_model_imputation_policy():
    """Verify observations post-cutoff are excluded before evaluating missingness under model_imputation."""
    # Observations entirely post-Aman cutoff (Sept 30)
    post_cutoff_records = [
        {"Date": "2018-10-05", "District": "Hooghly", "PRECTOTCORR": 25.0, "T2M": 28.0, "T2M_MIN": 22.0, "T2M_MAX": 33.0, "RH2M": 85.0, "ALLSKY_SFC_SW_DWN": 16.0, "WS2M": 2.0},
        {"Date": "2018-10-15", "District": "Hooghly", "PRECTOTCORR": 10.0, "T2M": 27.0, "T2M_MIN": 21.0, "T2M_MAX": 32.0, "RH2M": 80.0, "ALLSKY_SFC_SW_DWN": 18.0, "WS2M": 1.8},
    ]
    w_res = preprocess_inseason_weather(post_cutoff_records, district="Hooghly", year=2018, season="Winter")
    assert w_res["is_complete"] is False
    assert w_res["metadata"]["observed_days"] == 0
    assert w_res["metadata"]["excluded_post_cutoff_count"] == 2

    # Predict with model_imputation
    input_data = {"District": "Hooghly", "Season": "Winter", "Yield_lag1": 2800.0}
    input_data.update(w_res["features"])
    predictor = RiceYieldPredictor(weather_missing_policy="model_imputation")
    pred = predictor.predict(input_data)
    assert pred["prediction_method"] == "model_imputation"
    assert pred["weather_feature_status"] == "missing"
    assert pred["weather_features_used"] is False
    assert any("Warning: This prediction does NOT represent observed weather" in w for w in pred["warnings"])


