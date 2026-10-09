"""
AgroVision — Member 2 Audit Regression Test Suite

Verifies:
1. Production inference loads the intended 48-feature model.
2. Missing or incompatible artifacts produce clear errors instead of silently loading a legacy model.
3. The shared bridge produces compatible outputs using RiceInferencePipeline.
4. API, inference pipeline, and shared bridge predictions agree for identical inputs.
5. The production volatility estimator (persistence baseline) is used instead of the discarded classifier.
6. Feature parity and historical point-in-time leakage safeguards remain intact.
7. Pipeline execution cannot overwrite production artifacts.
8. Existing Tea functionality and relevant legacy callers continue to work.
"""

import sys
import os
import json
import hashlib
import joblib
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from inference import RiceInferencePipeline, InsufficientHistoryError
from api import app
from shared.predict_bridge import AgroVisionPredictor
from models.predict import predict as legacy_predict


@pytest.fixture
def sample_rice_history():
    """Generates 35 valid observations for Rice."""
    dates = pd.date_range("2024-01-01", periods=35)
    records = []
    for i, d in enumerate(dates):
        records.append({
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": 2500.0 + (i % 5) * 10,
            "Arrivals (Tonnes)": 150.0 + i
        })
    return pd.DataFrame(records)


@pytest.fixture
def sample_tea_dataframe():
    """Generates 1 valid row for Tea with legacy features."""
    return pd.DataFrame([{
        "Modal Price (Rs./Quintal)": 250.0,
        "lag_1": 240.0,
        "lag_7": 235.0,
        "lag_14": 230.0,
        "lag_30": 225.0,
        "rolling_mean_7": 242.0,
        "rolling_mean_14": 240.0,
        "rolling_mean_30": 238.0,
        "rolling_std_7": 5.1,
        "rolling_std_14": 6.2,
        "year": 2024,
        "day": 15,
        "month": 3,
        "day_of_week": 2,
        "week_of_year": 11,
        "arrival_lag_1": 50.0,
        "arrival_lag_7": 48.0,
        "arrival_rolling_mean_7": 49.0,
        "arrival_rolling_mean_14": 47.0,
        "price_pct_change": 0.5,
        "arrival_pct_change": 1.2,
        "price_spread": 10.0,
        "price_position_in_range": 0.5
    }])


# ============================================================
# 1. Production inference loads intended 48-feature model
# ============================================================
def test_production_inference_loads_48_feature_model():
    pipeline = RiceInferencePipeline()
    assert pipeline.price_model is not None
    assert getattr(pipeline.price_model, "n_features_in_", None) == 48
    assert len(pipeline.features_list) == 48
    assert pipeline.config.get("commodity") == "rice"
    
    # Key 48-feature checks (momentum, trend, streak)
    for expected_feat in [
        "lag_1", "lag_30", "price_change_1", "price_change_14",
        "accel_1_7", "trend_slope_7", "ewma_7", "zscore_7",
        "move_count_7", "streak_zero", "last_non_zero_move"
    ]:
        assert expected_feat in pipeline.features_list


# ============================================================
# 2. Missing or incompatible artifacts produce clear errors
# ============================================================
def test_missing_or_incompatible_artifacts_raise_clear_errors(tmp_path):
    # Test A: Missing directory / files
    empty_dir = tmp_path / "empty_artifacts"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        RiceInferencePipeline(artifacts_dir=empty_dir)

    # Test B: Incompatible feature count (legacy 20-feature model)
    incompatible_dir = tmp_path / "legacy_artifacts"
    incompatible_dir.mkdir()
    
    from xgboost import XGBRegressor
    dummy_model = XGBRegressor(n_estimators=10)
    X_dummy = np.random.rand(20, 20)
    y_dummy = np.random.rand(20)
    dummy_model.fit(X_dummy, y_dummy)
    joblib.dump(dummy_model, incompatible_dir / "rice_price_model.pkl")
    
    # Matching 48 config
    valid_cfg = {
        "commodity": "rice",
        "features": [f"feat_{i}" for i in range(48)]
    }
    with open(incompatible_dir / "rice_price_final_config.json", "w") as f:
        json.dump(valid_cfg, f)
        
    with pytest.raises(ValueError) as excinfo:
        RiceInferencePipeline(artifacts_dir=incompatible_dir)
    assert "Incompatible model artifact" in str(excinfo.value)
    assert "48 features" in str(excinfo.value)


# ============================================================
# 3. Shared bridge produces compatible outputs using RiceInferencePipeline
# ============================================================
def test_shared_bridge_compatible_outputs_rice(sample_rice_history):
    predictor = AgroVisionPredictor()
    result = predictor.predict_price(sample_rice_history, commodity="rice")
    
    assert "predicted_price_rs_quintal" in result
    assert "volatility" in result
    assert "commodity" in result
    assert result["commodity"] == "rice"
    assert isinstance(result["predicted_price_rs_quintal"], (int, float))
    assert result["predicted_price_rs_quintal"] > 100.0
    assert result["volatility"] in ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# 4. Predictions agree across API, inference pipeline, and bridge
# ============================================================
def test_predictions_agreement_api_pipeline_bridge(sample_rice_history):
    # Pipeline prediction
    pipeline = RiceInferencePipeline()
    pred_date = sample_rice_history["Reported Date"].iloc[-1]
    pipe_res = pipeline.predict(
        sample_rice_history,
        market="Burdwan",
        variety="Common",
        prediction_date=pred_date
    )

    # API prediction via TestClient
    history_payload = []
    for _, row in sample_rice_history.iterrows():
        history_payload.append({
            "reported_date": row["Reported Date"],
            "modal_price": float(row["Modal Price (Rs./Quintal)"]),
            "arrivals": float(row["Arrivals (Tonnes)"])
        })
    api_payload = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": pred_date,
        "history": history_payload
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=api_payload)
        assert response.status_code == 200
        api_res = response.json()

    # Bridge prediction
    bridge = AgroVisionPredictor()
    bridge_res = bridge.predict_price(
        sample_rice_history,
        commodity="rice",
        market="Burdwan",
        variety="Common",
        prediction_date=pred_date
    )

    # Validate exact agreement
    assert np.isclose(pipe_res["predicted_modal_price"], api_res["predicted_modal_price"], atol=1e-2)
    assert np.isclose(pipe_res["predicted_modal_price"], bridge_res["predicted_price_rs_quintal"], atol=1e-2)
    assert pipe_res["predicted_volatility_class"] == api_res["predicted_volatility_class"] == bridge_res["volatility"]
    assert np.isclose(pipe_res["predicted_volatility_range"], api_res["predicted_volatility_range"], atol=1e-2)


# ============================================================
# 5. Production volatility estimator uses persistence baseline
# ============================================================
def test_production_volatility_uses_persistence_baseline(sample_rice_history):
    pipeline = RiceInferencePipeline()
    predictor = AgroVisionPredictor()
    
    # Test case A: Last 5 prices equal -> Range 0 -> LOW
    df_low = sample_rice_history.copy()
    df_low.iloc[-5:, df_low.columns.get_loc("Modal Price (Rs./Quintal)")] = 2500.0
    res_low = pipeline.predict(df_low, "Burdwan", "Common", df_low["Reported Date"].iloc[-1])
    assert res_low["predicted_volatility_class"] == "LOW"
    assert res_low["predicted_volatility_range"] == 0.0

    # Test case B: Last 5 prices swing by 30 -> Range 30 -> MEDIUM
    df_med = sample_rice_history.copy()
    df_med.iloc[-5:, df_med.columns.get_loc("Modal Price (Rs./Quintal)")] = 2500.0
    df_med.iloc[-1, df_med.columns.get_loc("Modal Price (Rs./Quintal)")] = 2530.0
    res_med = pipeline.predict(df_med, "Burdwan", "Common", df_med["Reported Date"].iloc[-1])
    assert res_med["predicted_volatility_class"] == "MEDIUM"
    assert res_med["predicted_volatility_range"] == 30.0

    # Test case C: Last 5 prices swing by 120 -> Range 120 -> HIGH
    df_high = sample_rice_history.copy()
    df_high.iloc[-5:, df_high.columns.get_loc("Modal Price (Rs./Quintal)")] = 2500.0
    df_high.iloc[-1, df_high.columns.get_loc("Modal Price (Rs./Quintal)")] = 2620.0
    res_high = pipeline.predict(df_high, "Burdwan", "Common", df_high["Reported Date"].iloc[-1])
    assert res_high["predicted_volatility_class"] == "HIGH"
    assert res_high["predicted_volatility_range"] == 120.0


# ============================================================
# 6. Feature parity and point-in-time safeguards remain intact
# ============================================================
def test_point_in_time_future_immunity(sample_rice_history):
    pipeline = RiceInferencePipeline()
    target_date = sample_rice_history["Reported Date"].iloc[30]
    
    # Base prediction
    base_res = pipeline.predict(sample_rice_history, "Burdwan", "Common", target_date)
    
    # Poison observations AFTER target_date
    poisoned_df = sample_rice_history.copy()
    poisoned_df.iloc[31:, poisoned_df.columns.get_loc("Modal Price (Rs./Quintal)")] = 99999.0
    poisoned_df.iloc[31:, poisoned_df.columns.get_loc("Arrivals (Tonnes)")] = 99999.0
    
    poisoned_res = pipeline.predict(poisoned_df, "Burdwan", "Common", target_date)
    
    assert base_res["predicted_modal_price"] == poisoned_res["predicted_modal_price"]
    assert base_res["predicted_volatility_range"] == poisoned_res["predicted_volatility_range"]
    assert base_res["predicted_volatility_class"] == poisoned_res["predicted_volatility_class"]


# ============================================================
# 7. Pipeline execution cannot overwrite production artifacts
# ============================================================
def test_production_artifacts_remain_unmodified_under_pipeline_run():
    m2_art = PROJECT_ROOT / "artifacts"
    prod_model_path = m2_art / "rice_price_model.pkl"
    prod_cfg_path = m2_art / "rice_price_final_config.json"
    
    assert prod_model_path.exists()
    assert prod_cfg_path.exists()
    
    hash_model_before = hashlib.sha256(prod_model_path.read_bytes()).hexdigest()
    hash_cfg_before = hashlib.sha256(prod_cfg_path.read_bytes()).hexdigest()
    
    # Attempt save_models with commodity="rice" without test directory override
    from models.save_final_models import save_models
    from config import set_commodity, reset_artifacts_dir
    set_commodity("rice")
    reset_artifacts_dir()
    
    dummy_train = pd.DataFrame([{
        "Modal Price (Rs./Quintal)": 2000.0,
        "future_modal_price": 2010.0,
        "lag_1": 1990.0, "lag_7": 1980.0, "lag_14": 1970.0, "lag_30": 1960.0,
        "rolling_mean_7": 1990.0, "rolling_mean_14": 1980.0, "rolling_mean_30": 1970.0,
        "rolling_std_7": 5.0, "rolling_std_14": 6.0,
        "year": 2023, "day": 1, "month": 1, "day_of_week": 0, "week_of_year": 1,
        "arrival_lag_1": 10.0, "arrival_lag_7": 10.0,
        "arrival_rolling_mean_7": 10.0, "arrival_rolling_mean_14": 10.0,
        "price_pct_change": 0.5, "arrival_pct_change": 0.2
    }] * 35)
    
    save_models(train_df=dummy_train, vol_train_df=None, vol_test_df=None)
    
    hash_model_after = hashlib.sha256(prod_model_path.read_bytes()).hexdigest()
    hash_cfg_after = hashlib.sha256(prod_cfg_path.read_bytes()).hexdigest()
    
    # Assert production artifact hashes are strictly unchanged
    assert hash_model_before == hash_model_after, "Production rice_price_model.pkl was modified!"
    assert hash_cfg_before == hash_cfg_after, "Production rice_price_final_config.json was modified!"
    
    # Verify the pipeline run was safely diverted
    for diverted_name in [
        "rice_price_model_pipeline_run.pkl",
        "rice_volatility_model_pipeline_run.pkl",
        "rice_feature_config_pipeline_run.json",
    ]:
        p = m2_art / diverted_name
        if p.exists():
            p.unlink()


# ============================================================
# 8. Tea functionality and relevant legacy callers continue to work
# ============================================================
def test_tea_functionality_and_legacy_callers(sample_tea_dataframe):
    # Test AgroVisionPredictor with Tea
    predictor = AgroVisionPredictor()
    tea_res = predictor.predict_price(sample_tea_dataframe, commodity="tea")
    assert "predicted_price_rs_quintal" in tea_res
    assert tea_res["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert tea_res["commodity"] == "tea"
    assert isinstance(tea_res["predicted_price_rs_quintal"], (int, float))

    # Test get_required_features
    req_feats = predictor.get_required_features()
    assert "yield_features" in req_feats
    assert "price_features" in req_feats
    assert "volatility_features" in req_feats
    assert "rice_price_features" in req_feats
    assert len(req_feats["rice_price_features"]) == 48

    # Test legacy predict API function
    leg_res = legacy_predict(sample_tea_dataframe, commodity="tea")
    assert "predicted_price" in leg_res
    assert "volatility" in leg_res
    assert leg_res["commodity"] == "tea"
