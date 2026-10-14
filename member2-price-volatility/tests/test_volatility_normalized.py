"""
AgroVision Member 2 — Price-Normalized Volatility Model Tests

Comprehensive test suite verifying:
1. Artifact Loading: Checks existence, types, metadata, threshold values, and feature list.
2. Training/Inference Feature Parity: Verifies mathematical parity (< 1e-5) across all 35 features.
3. Temporal Leakage Protection: Proves future data cannot contaminate feature generation or predictions.
4. Insufficient History Handling: Asserts graceful rejection when observations < 30.
5. Backward Compatibility: Ensures default calls without version flag return legacy persistence baseline.
6. Versioned ML Inference: Asserts 'v2_normalized' prediction schema, probabilities, and class sanity.
7. API & Bridge Integration: Validates FastAPI and AgroVisionPredictor support for versioned volatility.
"""

import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi.testclient import TestClient

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
REPO_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from inference import RiceInferencePipeline, InsufficientHistoryError
from api import app
from feature_engineering import extract_volatility_features, EXPERIMENTAL_FEATURE_COLS
from shared.predict_bridge import AgroVisionPredictor


@pytest.fixture
def synthetic_history_40():
    """Generates 40 rows of deterministic time-series history."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    records = []
    for i, d in enumerate(dates):
        records.append({
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": 2400.0 + (i % 7) * 12.0,
            "Arrivals (Tonnes)": 120.0 + i * 1.5,
        })
    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# 1. Artifact Loading & Schema Integrity
# -----------------------------------------------------------------------------

def test_normalized_volatility_artifacts_loading():
    pipeline = RiceInferencePipeline()
    pipeline._load_normalized_volatility_model()

    assert pipeline._normalized_vol_model is not None
    assert pipeline._normalized_vol_config is not None
    assert len(pipeline._normalized_vol_features) == 35
    assert pipeline._normalized_vol_classes == ["LOW", "MEDIUM", "HIGH"]

    # Verify thresholds
    thresh = pipeline._normalized_vol_thresholds
    assert "low_medium_threshold" in thresh
    assert "medium_high_threshold" in thresh
    assert thresh["low_medium_threshold"] < thresh["medium_high_threshold"]
    assert np.isclose(thresh["low_medium_threshold"], 0.0142857, atol=1e-4)
    assert np.isclose(thresh["medium_high_threshold"], 1.9607843, atol=1e-4)


# -----------------------------------------------------------------------------
# 2. Training vs Inference Feature Parity
# -----------------------------------------------------------------------------

def test_feature_parity_training_vs_inference(synthetic_history_40):
    pipeline = RiceInferencePipeline()
    pipeline._load_normalized_volatility_model()

    # Batch extraction (experiment codebase)
    batch_feat = extract_volatility_features(
        synthetic_history_40,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        arrival_col="Arrivals (Tonnes)",
    )

    # Inference pipeline extraction
    inf_feat = pipeline._generate_normalized_volatility_features(synthetic_history_40)

    # Check last observation (where inference evaluates)
    batch_last = batch_feat.iloc[-1]
    inf_last = inf_feat.iloc[-1]

    for col in pipeline._normalized_vol_features:
        assert col in batch_last, f"Column missing from batch: {col}"
        assert col in inf_last, f"Column missing from inference: {col}"
        val_b = float(batch_last[col])
        val_i = float(inf_last[col])
        diff = abs(val_b - val_i)
        assert diff < 1e-5, f"Feature parity failure on {col}: batch={val_b}, inf={val_i}, diff={diff}"


# -----------------------------------------------------------------------------
# 3. Temporal Leakage Protection
# -----------------------------------------------------------------------------

def test_normalized_volatility_temporal_leakage_protection(synthetic_history_40):
    pipeline = RiceInferencePipeline()

    # Clean run on day 35
    cutoff_date = synthetic_history_40.iloc[34]["Reported Date"]
    clean_history = synthetic_history_40.iloc[:35].copy()

    pred_clean = pipeline.predict(
        clean_history,
        market="Burdwan",
        variety="Common",
        prediction_date=cutoff_date,
        volatility_version="v2_normalized",
    )

    # Poisoned history: full 40 days (containing 5 future days after cutoff)
    # Even if future prices fluctuate wildly, filtering by prediction_date must isolate history
    poisoned_history = synthetic_history_40.copy()
    poisoned_history.loc[35:, "Modal Price (Rs./Quintal)"] = 9999.0

    pred_poisoned = pipeline.predict(
        poisoned_history,
        market="Burdwan",
        variety="Common",
        prediction_date=cutoff_date,
        volatility_version="v2_normalized",
    )

    assert pred_clean["predicted_modal_price"] == pred_poisoned["predicted_modal_price"]
    assert pred_clean["predicted_volatility_class"] == pred_poisoned["predicted_volatility_class"]
    assert pred_clean["normalized_volatility_class"] == pred_poisoned["normalized_volatility_class"]
    assert pred_clean["historical_pct_range_5"] == pred_poisoned["historical_pct_range_5"]
    assert pred_clean["normalized_volatility_probabilities"] == pred_poisoned["normalized_volatility_probabilities"]


# -----------------------------------------------------------------------------
# 4. Insufficient History Handling
# -----------------------------------------------------------------------------

def test_normalized_volatility_insufficient_history(synthetic_history_40):
    pipeline = RiceInferencePipeline()
    short_df = synthetic_history_40.iloc[:20].copy()

    with pytest.raises(InsufficientHistoryError):
        pipeline.predict(
            short_df,
            market="Burdwan",
            variety="Common",
            prediction_date="2024-01-20",
            volatility_version="v2_normalized",
        )


# -----------------------------------------------------------------------------
# 5. Backward Compatibility (Default vs Versioned)
# -----------------------------------------------------------------------------

def test_backward_compatibility_default_call(synthetic_history_40):
    pipeline = RiceInferencePipeline()

    # Default call without volatility_version
    result_default = pipeline.predict(synthetic_history_40)

    # Verify standard legacy dictionary keys
    expected_keys = {
        "market",
        "variety",
        "prediction_date",
        "predicted_modal_price",
        "predicted_volatility_range",
        "predicted_volatility_class",
    }
    assert set(result_default.keys()) == expected_keys
    assert result_default["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
    assert "volatility_version" not in result_default

    # Explicit v1 call
    result_v1 = pipeline.predict(synthetic_history_40, volatility_version="v1")
    assert result_v1 == result_default


# -----------------------------------------------------------------------------
# 6. Versioned ML Inference Schema and Values
# -----------------------------------------------------------------------------

def test_versioned_ml_inference_schema_and_probabilities(synthetic_history_40):
    pipeline = RiceInferencePipeline()

    result = pipeline.predict(synthetic_history_40, volatility_version="v2_normalized")

    assert result["volatility_version"] == "v2_normalized"
    assert result["normalized_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["predicted_volatility_class"] == result["normalized_volatility_class"]

    probs = result["normalized_volatility_probabilities"]
    assert isinstance(probs, dict)
    assert set(probs.keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    for p in probs.values():
        assert 0.0 <= p <= 1.0

    assert isinstance(result["historical_pct_range_5"], float)
    assert result["historical_pct_range_5"] >= 0.0
    assert "thresholds" in result

    # Dedicated method test
    res_dedicated = pipeline.predict_normalized_volatility(synthetic_history_40)
    assert res_dedicated == result


# -----------------------------------------------------------------------------
# 7. FastAPI Endpoint Integration
# -----------------------------------------------------------------------------

def test_api_predict_versioned_and_default(synthetic_history_40):
    with TestClient(app) as client:
        history_records = [
            {
                "reported_date": row["Reported Date"],
                "modal_price": float(row["Modal Price (Rs./Quintal)"]),
                "arrivals": float(row["Arrivals (Tonnes)"]),
            }
            for _, row in synthetic_history_40.iterrows()
        ]

        base_payload = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": synthetic_history_40.iloc[-1]["Reported Date"],
            "history": history_records,
        }

        # 1. Default request
        resp_default = client.post("/predict", json=base_payload)
        assert resp_default.status_code == 200
        data_default = resp_default.json()
        assert data_default["volatility_version"] == "v1"
        assert data_default["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
        assert data_default["normalized_volatility_class"] is None

        # 2. Versioned request
        payload_v2 = dict(base_payload)
        payload_v2["volatility_version"] = "v2_normalized"
        resp_v2 = client.post("/predict", json=payload_v2)
        assert resp_v2.status_code == 200
        data_v2 = resp_v2.json()
        assert data_v2["volatility_version"] == "v2_normalized"
        assert data_v2["normalized_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
        assert data_v2["predicted_volatility_class"] == data_v2["normalized_volatility_class"]
        assert data_v2["normalized_volatility_probabilities"] is not None
        assert data_v2["historical_pct_range_5"] is not None
        assert data_v2["thresholds"] is not None

        # 3. Health check remains backward compatible
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json() == {
            "status": "ok",
            "model": "rice-price-xgb",
            "volatility": "persistence",
        }


# -----------------------------------------------------------------------------
# 8. AgroVision Bridge Integration
# -----------------------------------------------------------------------------

def test_bridge_predict_versioned_passthrough(synthetic_history_40):
    predictor = AgroVisionPredictor()

    hist_for_bridge = synthetic_history_40.rename(columns={
        "Reported Date": "reported_date",
        "Modal Price (Rs./Quintal)": "modal_price",
        "Arrivals (Tonnes)": "arrivals",
    })

    # Default bridge call
    bridge_default = predictor.predict_price(
        data_df=hist_for_bridge,
        commodity="rice",
        market="Burdwan",
        variety="Common",
        prediction_date=synthetic_history_40.iloc[-1]["Reported Date"],
    )
    assert bridge_default["commodity"] == "rice"
    assert bridge_default["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert "volatility_version" not in bridge_default

    # Versioned bridge call
    bridge_v2 = predictor.predict_price(
        data_df=hist_for_bridge,
        commodity="rice",
        market="Burdwan",
        variety="Common",
        prediction_date=synthetic_history_40.iloc[-1]["Reported Date"],
        volatility_version="v2_normalized",
    )
    assert bridge_v2["commodity"] == "rice"
    assert bridge_v2["volatility_version"] == "v2_normalized"
    assert bridge_v2["normalized_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
    assert bridge_v2["volatility"] == bridge_v2["normalized_volatility_class"]
    assert bridge_v2["historical_pct_range_5"] is not None
    assert "LOW" in bridge_v2["normalized_volatility_probabilities"]
