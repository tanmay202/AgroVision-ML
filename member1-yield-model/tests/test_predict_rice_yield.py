"""
Automated Test Suite for AgroVision Model 1 Rice Yield Prediction Pipeline
==========================================================================
Tests cover:
1. Loading all saved candidate models.
2. Single-record, list, and batch DataFrame prediction interfaces.
3. Unit conversion accuracy (kg/ha to tonnes/ha).
4. Missing input features and fallback imputation logic.
5. Missing Yield_lag1 fallback hierarchy (lag2 -> DS mean -> state mean).
6. String null representations ('nan', 'NaN', 'null', 'None', '', '   ') in Yield_lag1.
7. Infinite float handling (inf, -inf) in numerical features without sklearn crash.
8. Non-positive baseline rejection (Yield_lag1, Yield_lag2 <= 0).
9. District and Season normalization across casing and aliases (Burdwan, 24 Paraganas North, Boro, Aman).
10. Unseen districts and unseen seasons (OHE handle_unknown='ignore').
11. Invalid agronomic inputs (negative area, precip, out-of-bounds NDVI).
12. Agronomic yield clamping ([500, 6000] kg/ha) and unclamp mode.
13. Reproduction of test-set benchmark predictions.
14. Non-existent model path error raising.
15. Legacy model path alias mapping (yield_model.pkl).
16. Seasonal policy routing (Summer persistence fallback).
17. CLI execution via subprocess with arguments and JSON string (--data).
18. Shared bridge API compatibility (load_model, predict_yield, AgroVisionPredictor).
"""

import sys
import json
import subprocess
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
BASE = TEST_DIR.parent
PROJECT_ROOT = BASE.parent
SCRIPTS_DIR = BASE / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict_rice_yield import (
    RiceYieldPredictor,
    predict_rice_yield,
    load_model,
    predict_yield,
    ALL_FEATURES,
    DS_MEANS,
    D_MEANS,
    S_MEANS,
    STATE_MEAN,
    AGRONOMIC_MIN_YIELD,
    AGRONOMIC_MAX_YIELD
)


@pytest.fixture(scope="module")
def default_predictor():
    return RiceYieldPredictor(model_type='improved_ridge')


@pytest.fixture(scope="module")
def sample_valid_input():
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
        'InSeason_Precip_Cum': 950.0,
        'InSeason_Precip_Mean': 7.8,
        'InSeason_Precip_Max': 85.0,
        'InSeason_T2M_Mean': 27.2,
        'InSeason_T2M_Min': 22.0,
        'InSeason_T2M_Max': 33.5,
        'InSeason_T2M_Range': 11.5,
        'InSeason_RH2M_Mean': 80.5,
        'InSeason_Solar_Mean': 18.2,
        'InSeason_Wind_Mean': 1.8
    }


def test_candidate_models_loading():
    """Verify all 3 candidate models load and instantiate cleanly."""
    for mtype in ['improved_ridge', 'improved_xgb_cons', 'improved_xgb']:
        p = RiceYieldPredictor(model_type=mtype)
        assert p.pipeline is not None
        assert p.model_name.endswith('.pkl')


def test_predict_single_dict(default_predictor, sample_valid_input):
    """Test inference on single dictionary input."""
    res = default_predictor.predict(sample_valid_input)
    assert isinstance(res, dict)
    assert "predicted_yield_kg_ha" in res
    assert "predicted_yield_tonnes_ha" in res
    assert "predicted_delta_kg_ha" in res
    assert "baseline_yield_kg_ha" in res
    assert res["is_yield_lag1_imputed"] is False
    assert res["baseline_yield_kg_ha"] == 2850.0

    # Verify tonnes/ha conversion exactness: tonnes = kg / 1000
    assert abs(res["predicted_yield_tonnes_ha"] - (res["predicted_yield_kg_ha"] / 1000.0)) < 1e-4

    # Verify delta arithmetic: predicted = baseline + delta
    assert abs(res["predicted_yield_kg_ha"] - (res["baseline_yield_kg_ha"] + res["predicted_delta_kg_ha"])) < 0.05


def test_predict_dataframe_batch(default_predictor, sample_valid_input):
    """Test batch prediction using pandas DataFrame."""
    df_in = pd.DataFrame([sample_valid_input, sample_valid_input])
    df_out = default_predictor.predict(df_in)
    assert isinstance(df_out, pd.DataFrame)
    assert len(df_out) == 2
    assert "predicted_yield_kg_ha" in df_out.columns
    assert "predicted_yield_tonnes_ha" in df_out.columns
    assert "prediction_warnings" in df_out.columns


def test_predict_list_of_dicts(default_predictor, sample_valid_input):
    """Test batch prediction using list of dictionaries."""
    data_list = [sample_valid_input, sample_valid_input]
    res_list = default_predictor.predict(data_list, return_dict=True)
    assert isinstance(res_list, list)
    assert len(res_list) == 2
    assert isinstance(res_list[0], dict)


def test_missing_numerical_features_imputation(default_predictor):
    """Test that missing non-critical numerical features are imputed without crashing."""
    sparse_input = {
        'District': 'Birbhum',
        'Season': 'Autumn',
        'Yield_lag1': 2600.0
    }
    res = default_predictor.predict(sparse_input)
    assert res["is_yield_lag1_imputed"] is False
    assert res["baseline_yield_kg_ha"] == 2600.0
    assert AGRONOMIC_MIN_YIELD <= res["predicted_yield_kg_ha"] <= AGRONOMIC_MAX_YIELD


def test_missing_yield_lag1_fallback_hierarchy(default_predictor):
    """Test fallback hierarchy when Yield_lag1 is missing."""
    # 1. Fallback to Yield_lag2
    inp1 = {'District': 'Bankura', 'Season': 'Summer', 'Yield_lag2': 2950.0}
    res1 = default_predictor.predict(inp1)
    assert res1["is_yield_lag1_imputed"] is True
    assert res1["baseline_yield_kg_ha"] == 2950.0
    assert any("imputed using Yield_lag2" in w for w in res1["warnings"])

    # 2. Fallback to DS_MEANS for known District and Season
    inp2 = {'District': 'Bankura', 'Season': 'Summer'}
    res2 = default_predictor.predict(inp2)
    assert res2["is_yield_lag1_imputed"] is True
    assert res2["baseline_yield_kg_ha"] == DS_MEANS["Bankura_Summer"]
    assert any("historical Bankura Summer mean" in w for w in res2["warnings"])

    # 3. Fallback to STATE_MEAN for unseen District & Season
    inp3 = {'District': 'UnknownLand', 'Season': 'UnknownSeason'}
    res3 = default_predictor.predict(inp3)
    assert res3["is_yield_lag1_imputed"] is True
    assert res3["baseline_yield_kg_ha"] == STATE_MEAN
    assert any("imputed using state mean" in w for w in res3["warnings"])


def test_string_null_and_nan_imputation(default_predictor):
    """Test that string representations of null/nan safely trigger imputation."""
    for null_val in ['nan', 'NaN', 'null', 'None', 'N/A', '', '   ']:
        res = default_predictor.predict({
            'District': 'Hooghly',
            'Season': 'Winter',
            'Yield_lag1': null_val
        })
        assert res["is_yield_lag1_imputed"] is True
        assert not np.isnan(res["predicted_yield_kg_ha"])
        assert not np.isnan(res["predicted_yield_tonnes_ha"])
        assert res["baseline_yield_kg_ha"] == DS_MEANS["Hooghly_Winter"]


def test_infinite_float_sanitization(default_predictor):
    """Test that inf, -inf in numerical features do not crash pipeline."""
    bad_inf_input = {
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': 2800.0,
        'InSeason_Precip_Cum': float('inf'),
        'InSeason_T2M_Mean': float('-inf')
    }
    res = default_predictor.predict(bad_inf_input)
    assert not np.isnan(res["predicted_yield_kg_ha"])
    assert any("Infinite value in column" in w for w in res["warnings"])


def test_non_positive_baseline_rejection(default_predictor):
    """Test that non-positive Yield_lag1 and Yield_lag2 (<= 0) are rejected."""
    # Yield_lag1 <= 0 falls back to lag2
    res1 = default_predictor.predict({
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': 0.0,
        'Yield_lag2': 2750.0
    })
    assert res1["is_yield_lag1_imputed"] is True
    assert res1["baseline_yield_kg_ha"] == 2750.0

    # Yield_lag1 <= 0 and Yield_lag2 <= 0 falls back to DS_MEANS
    res2 = default_predictor.predict({
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': -50.0,
        'Yield_lag2': 0.0
    })
    assert res2["is_yield_lag1_imputed"] is True
    assert res2["baseline_yield_kg_ha"] == DS_MEANS["Hooghly_Winter"]


def test_district_and_season_normalization(default_predictor):
    """Test that casing variations and regional aliases are properly normalized."""
    # Lowercase & uppercase
    res_lower = default_predictor.predict({'District': 'hooghly', 'Season': 'winter', 'Yield_lag1': 2800.0})
    res_title = default_predictor.predict({'District': 'Hooghly', 'Season': 'Winter', 'Yield_lag1': 2800.0})
    assert res_lower["predicted_delta_kg_ha"] == res_title["predicted_delta_kg_ha"]

    # Regional aliases: Burdwan -> Bardhaman, 24 Paraganas North -> North 24 Parganas
    res_burdwan = default_predictor.predict({'District': 'burdwan', 'Season': 'Autumn', 'Yield_lag1': 2900.0})
    res_bardhaman = default_predictor.predict({'District': 'Bardhaman', 'Season': 'Autumn', 'Yield_lag1': 2900.0})
    assert res_burdwan["predicted_delta_kg_ha"] == res_bardhaman["predicted_delta_kg_ha"]

    # Agronomic seasonal aliases: Boro -> Summer, Aman -> Winter, Aus -> Autumn
    res_boro = default_predictor.predict({'District': 'Birbhum', 'Season': 'Boro', 'Yield_lag1': 3100.0})
    res_summer = default_predictor.predict({'District': 'Birbhum', 'Season': 'Summer', 'Yield_lag1': 3100.0})
    assert res_boro["predicted_delta_kg_ha"] == res_summer["predicted_delta_kg_ha"]


def test_unseen_district_and_season(default_predictor):
    """Test that unseen categorical values do not throw exceptions (OHE handle_unknown='ignore')."""
    inp = {
        'District': 'NonExistentDistrict',
        'Season': 'Spring',
        'Yield_lag1': 2500.0
    }
    res = default_predictor.predict(inp)
    assert isinstance(res, dict)
    assert any("Unseen District" in w for w in res["warnings"])
    assert any("Unseen Season" in w for w in res["warnings"])
    assert AGRONOMIC_MIN_YIELD <= res["predicted_yield_kg_ha"] <= AGRONOMIC_MAX_YIELD


def test_agronomic_bounds_clamping():
    """Verify that yields outside realistic agronomic boundaries are clamped and flagged."""
    low_predictor = RiceYieldPredictor(clamp_yield=True, min_yield=1000.0)
    res_low = low_predictor.predict({'District': 'Purulia', 'Season': 'Autumn', 'Yield_lag1': 50.0})
    assert res_low["predicted_yield_kg_ha"] == 1000.0
    assert any("below agronomic minimum" in w for w in res_low["warnings"])

    unclamped_predictor = RiceYieldPredictor(clamp_yield=False)
    res_unclamped = unclamped_predictor.predict({'District': 'Purulia', 'Season': 'Autumn', 'Yield_lag1': 50.0})
    assert res_unclamped["predicted_yield_kg_ha"] < 1000.0

    high_predictor = RiceYieldPredictor(clamp_yield=True, max_yield=6000.0)
    res_high = high_predictor.predict({'District': 'Hooghly', 'Season': 'Summer', 'Yield_lag1': 10000.0})
    assert res_high["predicted_yield_kg_ha"] == 6000.0
    assert any("above agronomic maximum" in w for w in res_high["warnings"])


def test_invalid_input_ranges_generate_warnings(default_predictor):
    """Test that biologically / physically implausible input features trigger warnings."""
    bad_input = {
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': 2500.0,
        'Area_lag1': -500.0,
        'Precip_lag1': -20.0,
        'NDVI_lag1': 1.8,
        'InSeason_T2M_Mean': 85.0,
        'InSeason_RH2M_Mean': 120.0
    }
    res = default_predictor.predict(bad_input)
    warnings_text = " ".join(res["warnings"])
    assert "Negative area" in warnings_text
    assert "Negative precipitation" in warnings_text
    assert "NDVI out of range" in warnings_text
    assert "Extreme temperature" in warnings_text
    assert "Relative humidity out of [0, 100]%" in warnings_text


def test_reproduction_accuracy(default_predictor):
    """Verify inference on a test set record matches expected delta within numerical tolerance."""
    inseason_path = BASE / "data/processed/rice_yield_inseason_features_wb.csv"
    df = pd.read_csv(inseason_path)
    test_rows = df[df['Year'] == 2017].sort_values(['District', 'Season']).reset_index(drop=True)
    sample_row = test_rows.iloc[0].to_dict()

    res = default_predictor.predict(sample_row)
    expected_delta = default_predictor.pipeline.predict(pd.DataFrame([sample_row])[ALL_FEATURES])[0]

    assert abs(res["predicted_delta_kg_ha"] - expected_delta) < 0.01
    assert abs(res["predicted_yield_kg_ha"] - (sample_row["Yield_lag1"] + expected_delta)) < 0.01


def test_nonexistent_model_path_raises_error():
    """Verify that specifying an explicit invalid model path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RiceYieldPredictor(model_path="nonexistent_delta_model.pkl")


def test_legacy_bridge_model_alias():
    """Verify that passing legacy yield_model.pkl resolves to improved delta ridge."""
    legacy_path = BASE / "models" / "yield_model.pkl"
    predictor = RiceYieldPredictor(model_path=legacy_path)
    assert predictor.model_name == "m1_improved_delta_ridge.pkl"


def test_seasonal_policy_routing():
    """Verify that enable_seasonal_routing bypasses delta for Summer and preserves delta for Autumn."""
    routed_predictor = RiceYieldPredictor(enable_seasonal_routing=True)

    # Summer: delta should be 0.0, predicted yield == baseline
    res_summer = routed_predictor.predict({
        'District': 'Hooghly',
        'Season': 'Summer',
        'Yield_lag1': 3200.0,
        'InSeason_Precip_Cum': 50.0
    })
    assert res_summer["predicted_delta_kg_ha"] == 0.0
    assert res_summer["predicted_yield_kg_ha"] == 3200.0
    assert any("Seasonal routing active" in w for w in res_summer["warnings"])

    # Autumn: delta should NOT be 0.0
    res_autumn = routed_predictor.predict({
        'District': 'Hooghly',
        'Season': 'Autumn',
        'Yield_lag1': 2500.0,
        'InSeason_Precip_Cum': 600.0
    })
    assert res_autumn["predicted_delta_kg_ha"] != 0.0


def test_shared_bridge_api():
    """Verify load_model and predict_yield functions meet the integration contract."""
    model_info = load_model()
    assert "pipeline" in model_info
    assert "feature_names" in model_info
    assert len(model_info["feature_names"]) == 20

    sample = {'District': 'Hooghly', 'Season': 'Winter', 'Yield_lag1': 2800.0}
    y_tonnes = predict_yield(model_info, sample)
    assert isinstance(y_tonnes, float)
    assert 0.5 <= y_tonnes <= 6.0


def test_agrovision_predictor_bridge_integration():
    """Verify end-to-end integration via shared.predict_bridge.AgroVisionPredictor."""
    from shared.predict_bridge import AgroVisionPredictor
    bridge = AgroVisionPredictor()
    res = bridge.predict_yield({'District': 'Hooghly', 'Season': 'Winter', 'Yield_lag1': 2800.0})
    assert "predicted_yield_kg_per_ha" in res
    assert "predicted_yield_tonnes_per_ha" in res
    assert abs(res["predicted_yield_tonnes_per_ha"] - (res["predicted_yield_kg_per_ha"] / 1000.0)) < 1e-4
    assert "yield_arrivals_tonnes" in res
    assert 0.5 <= res["predicted_yield_tonnes_per_ha"] <= 6.0
    assert 0.5 <= res["yield_arrivals_tonnes"] <= 6.0


def test_cli_execution_with_flags():
    """Test predict_rice_yield.py CLI script execution via subprocess."""
    cli_path = SCRIPTS_DIR / "predict_rice_yield.py"
    cmd = [
        sys.executable,
        str(cli_path),
        "--district", "Hooghly",
        "--season", "Winter",
        "--yield-lag1", "2800.0"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "AGROVISION RICE YIELD PREDICTION RESULT" in proc.stdout
    assert "Predicted Yield (kg/ha):" in proc.stdout
    assert "Predicted Yield (tonnes/ha):" in proc.stdout


def test_cli_execution_with_data_json():
    """Test CLI execution using raw JSON string via --data."""
    cli_path = SCRIPTS_DIR / "predict_rice_yield.py"
    json_data = json.dumps({"District": "Bankura", "Season": "Autumn", "Yield_lag1": 2700.0})
    cmd = [
        sys.executable,
        str(cli_path),
        "--data", json_data
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "predicted_yield_kg_ha" in proc.stdout
    assert "predicted_yield_tonnes_ha" in proc.stdout
