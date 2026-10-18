"""
AgroVision-ML — Safe Client Integration Example for Rice Price Forecasting
Demonstrating:
1. Default legacy prediction (1-7 day observation target via POST /predict and pipeline.predict)
2. Opt-in 14-day calendar forecast (POST /predict-horizon and pipeline.predict_horizon)
3. Experimental 30-day calendar forecast (POST /predict-horizon and bridge.predict_price_horizon)
4. Comprehensive error handling (insufficient history, unsupported horizon, disabled switches)
5. Structured downstream consumption (forecast origin, target date, model identity, release policy, limitations)
"""

import sys
import json
from pathlib import Path
import pandas as pd
from fastapi.testclient import TestClient

# Path resolution
EXAMPLE_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = EXAMPLE_DIR.parent
REPO_ROOT = MEMBER2_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
SHARED_DIR = REPO_ROOT / "shared"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))
sys.path.insert(0, str(SHARED_DIR))

from api import app
from inference import RiceInferencePipeline, InsufficientHistoryError
from config import (
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    reset_horizon_forecasting_switches,
)
from predict_bridge import AgroVisionPredictor


def build_sample_observations(n_days: int = 35, start_price: float = 2400.0) -> list:
    """Helper to construct synthetic daily observations for API requests."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    history = []
    for i, d in enumerate(dates):
        history.append({
            "reported_date": d.strftime("%Y-%m-%d"),
            "modal_price": float(start_price + (i * 5.0)),
            "arrivals": float(120.0 + (i % 7) * 10.0),
        })
    return history


def format_dataframe_from_history(history: list, market: str = "Burdwan", variety: str = "Common") -> pd.DataFrame:
    """Helper to format pandas DataFrame for pipeline or bridge calls."""
    df = pd.DataFrame(history)
    df["Market Name"] = market
    df["Variety"] = variety
    df.rename(columns={"reported_date": "Reported Date", "modal_price": "Modal Price (Rs./Quintal)", "arrivals": "Arrivals (Tonnes)"}, inplace=True)
    return df


def demonstrate_api_client():
    """Demonstrates interacting with the FastAPI service using TestClient / requests."""
    print("=" * 80)
    print("PART 1: HTTP API CLIENT INTEGRATION (FastAPI Endpoints)")
    print("=" * 80)

    history = build_sample_observations(35, start_price=2500.0)
    forecast_origin = history[-1]["reported_date"]  # "2024-02-04"

    with TestClient(app) as client:
        # ---------------------------------------------------------
        # 1. Legacy Default Prediction (POST /predict)
        # ---------------------------------------------------------
        print("\n[Step 1.1] Requesting Legacy Default Prediction (POST /predict)...")
        legacy_payload = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": forecast_origin,
            "history": history,
        }
        resp = client.post("/predict", json=legacy_payload)
        print(f"  HTTP Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"  Target: Next valid observation within 1-7 days (continuity constraint)")
            print(f"  Predicted Price: Rs.{data['predicted_modal_price']:.2f} / quintal")
            print(f"  Volatility Class: {data['predicted_volatility_class']}")
            print(f"  (Note: 'horizon_days' is NOT present in legacy response)")

        # Verify default-disabled state in fresh installation
        reset_horizon_forecasting_switches()
        payload_14d = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": forecast_origin,
            "horizon_days": 14,
            "history": history,
        }
        resp_fresh = client.post("/predict-horizon", json=payload_14d)
        print(f"\n[Opt-In Policy Check] Default fresh installation status for POST /predict-horizon:")
        print(f"  HTTP Status: {resp_fresh.status_code} (Expected 503 - Disabled by default for safety)")
        print(f"  Detail: {resp_fresh.json()['detail']}")

        # Explicitly activate horizon switches (Operator Opt-In Deployment)
        print("\n[Operator Action] Activating AGROVISION_ENABLE_HORIZON_FORECASTING and horizon switches...")
        set_horizon_forecasting_enabled(True)
        set_horizon_14d_enabled(True)
        set_horizon_30d_enabled(True)

        # ---------------------------------------------------------
        # 2. Opt-in 14-Day Calendar Forecast (POST /predict-horizon)
        # ---------------------------------------------------------
        print("\n[Step 1.2] Requesting Direct 14-Day Forecast (POST /predict-horizon, horizon_days=14)...")
        resp_14d = client.post("/predict-horizon", json=payload_14d)
        print(f"  HTTP Status: {resp_14d.status_code}")
        if resp_14d.status_code == 200:
            d14 = resp_14d.json()
            print(f"  Forecast Origin Date : {d14['prediction_date']}")
            print(f"  Requested Target Date: {d14['requested_target_date']} (t + 14 calendar days)")
            print(f"  Current Modal Price  : Rs.{d14['current_modal_price']:.2f} / quintal")
            print(f"  Predicted Price Change: Rs.{d14['predicted_price_change']:+.2f} / quintal")
            print(f"  Predicted Modal Price: Rs.{d14['predicted_modal_price']:.2f} / quintal")
            print(f"  Model Identity       : {d14['model_name']} (version {d14['model_version']})")
            print(f"  Release Status       : {d14['release_status']}")
            print(f"  Target Window Policy : {d14['target_date_convention']}")
            print(f"  Operational Caution  : {d14['limitations']}")

        # ---------------------------------------------------------
        # 3. Experimental 30-Day Calendar Forecast (POST /predict-horizon)
        # ---------------------------------------------------------
        print("\n[Step 1.3] Requesting Direct 30-Day Forecast (POST /predict-horizon, horizon_days=30)...")
        payload_30d = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": forecast_origin,
            "horizon_days": 30,
            "history": history,
        }
        resp_30d = client.post("/predict-horizon", json=payload_30d)
        print(f"  HTTP Status: {resp_30d.status_code}")
        if resp_30d.status_code == 200:
            d30 = resp_30d.json()
            print(f"  Forecast Origin Date : {d30['prediction_date']}")
            print(f"  Requested Target Date: {d30['requested_target_date']} (t + 30 calendar days)")
            print(f"  Predicted Modal Price: Rs.{d30['predicted_modal_price']:.2f} / quintal")
            print(f"  Model Identity       : {d30['model_name']} (version {d30['model_version']})")
            print(f"  Release Status       : {d30['release_status']} (interpret with caution on flat mandis)")
            print(f"  Operational Caution  : {d30['limitations']}")

        # ---------------------------------------------------------
        # 4. Error Handling Scenarios
        # ---------------------------------------------------------
        print("\n[Step 1.4] Demonstrating API Error Handling...")

        # 4a. Unsupported Horizon
        bad_payload = {**payload_14d, "horizon_days": 7}
        resp_bad = client.post("/predict-horizon", json=bad_payload)
        print(f"  [4a. Bad Horizon (7d)] Expected HTTP 400 -> Got HTTP {resp_bad.status_code}: {resp_bad.json()['detail']}")

        # 4b. Insufficient History (< 30 observations)
        short_payload = {**payload_14d, "history": history[:20]}
        resp_short = client.post("/predict-horizon", json=short_payload)
        print(f"  [4b. Insufficient History (20 obs)] Expected HTTP 400 -> Got HTTP {resp_short.status_code}: {resp_short.json()['detail']}")

        # 4c. Feature Disabled via Operational Switch
        set_horizon_forecasting_enabled(False)
        resp_disabled = client.post("/predict-horizon", json=payload_14d)
        print(f"  [4c. Globally Disabled Switch] Expected HTTP 503 -> Got HTTP {resp_disabled.status_code}: {resp_disabled.json()['detail']}")
        reset_horizon_forecasting_switches()

        # ---------------------------------------------------------
        # 5. Horizon Health & Observability Metrics
        # ---------------------------------------------------------
        print("\n[Step 1.5] Inspecting Horizon Observability Endpoints...")
        h_health = client.get("/health/horizons").json()
        print(f"  GET /health/horizons: 14d ready={h_health['horizon_14d']['ready']}, 30d ready={h_health['horizon_30d']['ready']}")
        
        h_metrics = client.get("/metrics/horizons").json()
        print(f"  GET /metrics/horizons: 14d stats={h_metrics['14d']}, 30d stats={h_metrics['30d']}")


def demonstrate_python_sdk():
    """Demonstrates direct Python library usage via RiceInferencePipeline and AgroVisionPredictor bridge."""
    print("\n" + "=" * 80)
    print("PART 2: PYTHON SDK & SHARED BRIDGE INTEGRATION")
    print("=" * 80)

    set_horizon_forecasting_enabled(True)
    set_horizon_14d_enabled(True)
    set_horizon_30d_enabled(True)

    history = build_sample_observations(35, start_price=2500.0)
    df = format_dataframe_from_history(history)

    # 1. Direct Pipeline Call
    print("\n[Step 2.1] Calling RiceInferencePipeline.predict_horizon()...")
    pipeline = RiceInferencePipeline()
    res_pipe_14 = pipeline.predict_horizon(df, horizon_days=14)
    print(f"  Pipeline 14d Result: Modal Price = Rs.{res_pipe_14['predicted_modal_price']}, Target Date = {res_pipe_14['requested_target_date']}")

    # 2. AgroVisionPredictor Bridge Call
    print("\n[Step 2.2] Calling AgroVisionPredictor.predict_price_horizon() bridge...")
    bridge = AgroVisionPredictor()
    res_bridge_30 = bridge.predict_price_horizon(df, horizon_days=30, commodity="rice")
    print(f"  Bridge 30d Result  : Modal Price = Rs.{res_bridge_30['predicted_modal_price']}, Target Date = {res_bridge_30['requested_target_date']}")
    print(f"  Model Identifier   : {res_bridge_30['model_name']}")

    # 3. Standard bridge predict_price() remains backward-compatible
    res_std = bridge.predict_price(df, commodity="rice")
    print(f"\n[Step 2.3] Calling AgroVisionPredictor.predict_price() (Legacy Invariant)...")
    print(f"  Legacy Bridge Result: Modal Price = Rs.{res_std['predicted_price_rs_quintal']}, Volatility = {res_std['volatility']}")


def main():
    print("Running AgroVision-ML Client Integration Showcase...")
    demonstrate_api_client()
    demonstrate_python_sdk()
    print("\n" + "=" * 80)
    print("INTEGRATION SHOWCASE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
