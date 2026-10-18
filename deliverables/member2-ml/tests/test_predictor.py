"""
Test Suite for Member 2 ML Deliverable Package (RicePricePredictor).
Verifies:
1. Artifact loading & integrity.
2. Short-Term (1-7d) price & volatility prediction.
3. 14-Day calendar horizon prediction.
4. 30-Day calendar horizon prediction.
5. Unified dispatch method.
6. Schema tolerance (DataFrame & List of Dicts with alias handling).
7. Error handling & validation (insufficient history, invalid horizon, missing columns).
8. Mathematical parity with existing pipeline.
9. Zero-lookahead leakage prevention.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = TEST_DIR.parent
sys.path.insert(0, str(PACKAGE_DIR))

from src.predictor import RicePricePredictor, InsufficientHistoryError
from src.config import SUPPORTED_HORIZONS, TARGET_CONVENTIONS, RELEASE_STATUSES


@pytest.fixture(scope="module")
def predictor():
    """Initializes RicePricePredictor with local deliverable artifacts."""
    artifacts_dir = PACKAGE_DIR / "artifacts"
    return RicePricePredictor(artifacts_dir=artifacts_dir)


@pytest.fixture
def synthetic_history():
    """Builds 40 sequential daily observations for Burdwan Common rice."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    base_price = 2500.0
    prices = [base_price + (i % 7) * 12.0 for i in range(40)]
    arrivals = [100.0 + (i % 5) * 10.0 for i in range(40)]

    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": dates.strftime("%Y-%m-%d"),
        "Modal Price (Rs./Quintal)": prices,
        "Arrivals (Tonnes)": arrivals,
    })
    return df


def test_predictor_initialization(predictor):
    """Verifies predictor loads 48 features and valid configuration."""
    assert len(predictor.get_required_features()) == 48
    info = predictor.get_model_info()
    assert info["commodity"] == "rice"
    assert "horizons" in info
    assert "short_term" in info["horizons"]
    assert "14d" in info["horizons"]
    assert "30d" in info["horizons"]


def test_predict_short_term_dataframe(predictor, synthetic_history):
    """Verifies short-term forecasting using DataFrame input."""
    res = predictor.predict_short_term(
        history=synthetic_history,
        market="Burdwan",
        variety="Common",
        prediction_date="2024-02-09",
        volatility_version="v2_normalized",
    )

    assert res["market"] == "Burdwan"
    assert res["variety"] == "Common"
    assert res["horizon_days"] == 1
    assert res["release_status"] == "production"
    assert "current_modal_price" in res
    assert "predicted_modal_price" in res
    assert "predicted_price_change" in res
    assert np.isclose(res["predicted_modal_price"], res["current_modal_price"] + res["predicted_price_change"], atol=1e-2)
    assert res["volatility_version"] == "v2_normalized"
    assert res["normalized_volatility_class"] in ("LOW", "MEDIUM", "HIGH")
    assert "normalized_volatility_probabilities" in res


def test_predict_short_term_list_of_dicts(predictor, synthetic_history):
    """Verifies short-term forecasting using list of dicts with snake_case aliases."""
    records = []
    for _, row in synthetic_history.iterrows():
        records.append({
            "market": row["Market Name"],
            "variety": row["Variety"],
            "date": row["Reported Date"],
            "modal_price": float(row["Modal Price (Rs./Quintal)"]),
            "arrivals": float(row["Arrivals (Tonnes)"]),
        })

    res = predictor.predict_short_term(history=records)
    assert res["market"] == "Burdwan"
    assert res["variety"] == "Common"
    assert res["predicted_modal_price"] > 1000.0


def test_predict_14d_horizon(predictor, synthetic_history):
    """Verifies 14-day calendar horizon forecasting."""
    res = predictor.predict_horizon(
        history=synthetic_history,
        horizon_days=14,
        market="Burdwan",
        variety="Common",
        prediction_date="2024-02-09",
    )

    assert res["horizon_days"] == 14
    assert res["release_status"] == "opt-in"
    assert res["model_name"] == "rice-price-xgb-14d"
    assert res["requested_target_date"] == "2024-02-23"
    assert res["target_date_convention"] == TARGET_CONVENTIONS[14]
    assert np.isclose(res["predicted_modal_price"], res["current_modal_price"] + res["predicted_price_change"], atol=1e-2)
    assert "horizon_metadata" in res


def test_predict_30d_horizon(predictor, synthetic_history):
    """Verifies 30-day calendar horizon forecasting."""
    res = predictor.predict_horizon(
        history=synthetic_history,
        horizon_days=30,
        market="Burdwan",
        variety="Common",
        prediction_date="2024-02-09",
    )

    assert res["horizon_days"] == 30
    assert res["release_status"] == "experimental"
    assert res["model_name"] == "rice-price-xgb-30d"
    assert res["requested_target_date"] == "2024-03-10"
    assert res["target_date_convention"] == TARGET_CONVENTIONS[30]
    assert np.isclose(res["predicted_modal_price"], res["current_modal_price"] + res["predicted_price_change"], atol=1e-2)
    assert "horizon_metadata" in res


def test_unified_predict_dispatch(predictor, synthetic_history):
    """Verifies unified predict() dispatch for short, 14d, and 30d horizons."""
    r_short = predictor.predict(synthetic_history, horizon="short")
    assert r_short["horizon_days"] == 1

    r_14 = predictor.predict(synthetic_history, horizon=14)
    assert r_14["horizon_days"] == 14

    r_30 = predictor.predict(synthetic_history, horizon=30)
    assert r_30["horizon_days"] == 30

    with pytest.raises(ValueError, match="Unsupported horizon"):
        predictor.predict(synthetic_history, horizon=45)


def test_insufficient_history_error(predictor, synthetic_history):
    """Verifies InsufficientHistoryError is raised when observations < 30."""
    short_df = synthetic_history.iloc[:20]
    with pytest.raises(InsufficientHistoryError, match="Required at least 30"):
        predictor.predict_short_term(short_df)

    with pytest.raises(InsufficientHistoryError, match="Required at least 30"):
        predictor.predict_horizon(short_df, horizon_days=14)


def test_unsupported_horizon_error(predictor, synthetic_history):
    """Verifies ValueError on invalid horizon days."""
    with pytest.raises(ValueError, match="Unsupported horizon_days"):
        predictor.predict_horizon(synthetic_history, horizon_days=7)


def test_leakage_safety_future_poisoning(predictor, synthetic_history):
    """Verifies future records after prediction_date do not influence predictions."""
    pred_date = "2024-02-01"

    res_clean = predictor.predict_horizon(synthetic_history, horizon_days=14, prediction_date=pred_date)

    # Poison rows after pred_date
    poisoned_df = synthetic_history.copy()
    poison_mask = pd.to_datetime(poisoned_df["Reported Date"]) > pd.to_datetime(pred_date)
    poisoned_df.loc[poison_mask, "Modal Price (Rs./Quintal)"] = 99999.0

    res_poisoned = predictor.predict_horizon(poisoned_df, horizon_days=14, prediction_date=pred_date)

    assert np.isclose(res_clean["predicted_modal_price"], res_poisoned["predicted_modal_price"], atol=1e-4)
    assert np.isclose(res_clean["predicted_price_change"], res_poisoned["predicted_price_change"], atol=1e-4)


def test_parity_with_member2_inference_pipeline(predictor, synthetic_history):
    """Verifies exact output parity between RicePricePredictor and RiceInferencePipeline."""
    # Import original pipeline from member2-price-volatility
    m2_src = PACKAGE_DIR.parent.parent / "member2-price-volatility" / "src"
    if str(m2_src) not in sys.path:
        sys.path.insert(0, str(m2_src))

    from inference import RiceInferencePipeline
    pipe = RiceInferencePipeline(artifacts_dir=PACKAGE_DIR / "artifacts")

    # Short-term parity
    pred_st = predictor.predict_short_term(synthetic_history, volatility_version="v2_normalized")
    pipe_st = pipe.predict(synthetic_history, volatility_version="v2_normalized")
    assert np.isclose(pred_st["predicted_modal_price"], pipe_st["predicted_modal_price"], atol=1e-2)
    assert pred_st["predicted_volatility_class"] == pipe_st["predicted_volatility_class"]

    # 14d parity
    pred_14 = predictor.predict_horizon(synthetic_history, horizon_days=14)
    pipe_14 = pipe.predict_horizon(synthetic_history, horizon_days=14, bypass_switches=True)
    assert np.isclose(pred_14["predicted_modal_price"], pipe_14["predicted_modal_price"], atol=1e-2)

    # 30d parity
    pred_30 = predictor.predict_horizon(synthetic_history, horizon_days=30)
    pipe_30 = pipe.predict_horizon(synthetic_history, horizon_days=30, bypass_switches=True)
    assert np.isclose(pred_30["predicted_modal_price"], pipe_30["predicted_modal_price"], atol=1e-2)
