import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from api import app
from config import (
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    reset_horizon_forecasting_switches,
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_cors_headers_present(client):
    """Verify CORS headers are returned for cross-origin frontend requests."""
    response = client.options(
        "/predict-horizon",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        }
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_serve_ui_endpoint(client):
    """Verify GET / and GET /ui serve the AgroVision Farmer App UI."""
    resp_root = client.get("/")
    assert resp_root.status_code == 200
    assert "AgroVision" in resp_root.text
    assert "Rice Price Forecasting" in resp_root.text

    resp_ui = client.get("/ui")
    assert resp_ui.status_code == 200
    assert "AgroVision" in resp_ui.text


def test_get_history_endpoint_valid(client):
    """Verify GET /history returns authentic observations matching HistoricalObservation schema."""
    response = client.get("/history?market=Burdwan&variety=Common&limit=35")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 30
    for obs in data:
        assert "reported_date" in obs
        assert "modal_price" in obs
        assert "arrivals" in obs
        assert isinstance(obs["modal_price"], (int, float))
        assert isinstance(obs["arrivals"], (int, float))

    # Verify chronological ascending order
    dates = [o["reported_date"] for o in data]
    assert dates == sorted(dates)


def test_get_history_insufficient_or_missing(client):
    """Verify GET /history returns 400 when insufficient observations exist."""
    response = client.get("/history?market=NonExistentMarket&variety=RareVariety&limit=35")
    assert response.status_code == 400
    assert "Insufficient history" in response.json()["detail"]


def test_get_markets_endpoint(client):
    """Verify GET /markets returns available markets and varieties."""
    response = client.get("/markets")
    assert response.status_code == 200
    data = response.json()
    assert data["commodity"] == "rice"
    assert "markets" in data
    assert len(data["markets"]) > 0
    burdwan = next((m for m in data["markets"] if m["market"] == "Burdwan"), None)
    assert burdwan is not None
    assert "Common" in burdwan["varieties"]


def test_short_term_prediction_workflow(client):
    """Verify existing short-term prediction (1-7 days) uses genuine history and returns PredictResponse."""
    # Step 1: Obtain genuine history through backend interface
    hist_resp = client.get("/history?market=Burdwan&variety=Common&limit=35")
    assert hist_resp.status_code == 200
    history = hist_resp.json()

    # Step 2: Call POST /predict exactly as UI does
    payload = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": history[-1]["reported_date"],
        "history": history,
        "volatility_version": "v1"
    }
    pred_resp = client.post("/predict", json=payload)
    assert pred_resp.status_code == 200
    res = pred_resp.json()
    assert res["market"] == "Burdwan"
    assert res["variety"] == "Common"
    assert "predicted_modal_price" in res
    assert res["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]
    assert "predicted_volatility_range" in res


def test_14d_horizon_prediction_workflow(client):
    """Verify 14-day horizon forecasting sends horizon_days=14 and returns PredictHorizonResponse."""
    set_horizon_forecasting_enabled(True)
    set_horizon_14d_enabled(True)

    try:
        hist_resp = client.get("/history?market=Burdwan&variety=Common&limit=35")
        assert hist_resp.status_code == 200
        history = hist_resp.json()
        origin_date = history[-1]["reported_date"]

        payload = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": origin_date,
            "horizon_days": 14,
            "history": history
        }
        pred_resp = client.post("/predict-horizon", json=payload)
        assert pred_resp.status_code == 200
        res = pred_resp.json()
        assert res["market"] == "Burdwan"
        assert res["variety"] == "Common"
        assert res["horizon_days"] == 14
        assert res["prediction_date"] == origin_date
        assert res["model_name"] == "rice-price-xgb-14d"
        assert res["release_status"] == "opt-in"
        assert "First observation in [t + 14, t + 17]" in res["target_date_convention"]
        assert "predicted_modal_price" in res
        assert "predicted_price_change" in res
        assert "limitations" in res
    finally:
        reset_horizon_forecasting_switches()


def test_30d_horizon_prediction_workflow(client):
    """Verify 30-day horizon forecasting sends horizon_days=30 and returns experimental metadata."""
    set_horizon_forecasting_enabled(True)
    set_horizon_30d_enabled(True)

    try:
        hist_resp = client.get("/history?market=Burdwan&variety=Common&limit=35")
        assert hist_resp.status_code == 200
        history = hist_resp.json()
        origin_date = history[-1]["reported_date"]

        payload = {
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": origin_date,
            "horizon_days": 30,
            "history": history
        }
        pred_resp = client.post("/predict-horizon", json=payload)
        assert pred_resp.status_code == 200
        res = pred_resp.json()
        assert res["market"] == "Burdwan"
        assert res["variety"] == "Common"
        assert res["horizon_days"] == 30
        assert res["prediction_date"] == origin_date
        assert res["model_name"] == "rice-price-xgb-30d"
        assert res["release_status"] == "experimental"
        assert "First observation in [t + 30, t + 33]" in res["target_date_convention"]
        assert "predicted_modal_price" in res
        assert "predicted_price_change" in res
        assert "limitations" in res
    finally:
        reset_horizon_forecasting_switches()


def test_disabled_horizon_endpoints_produce_503(client):
    """Verify that when switches are disabled, meaningful 503 responses are generated."""
    reset_horizon_forecasting_switches()

    hist_resp = client.get("/history?market=Burdwan&variety=Common&limit=35")
    history = hist_resp.json()

    payload_14 = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": history[-1]["reported_date"],
        "horizon_days": 14,
        "history": history
    }
    resp_14 = client.post("/predict-horizon", json=payload_14)
    assert resp_14.status_code == 503
    assert "disabled" in resp_14.json()["detail"].lower()

    # Legacy predict STILL works even when horizon forecasting is disabled!
    payload_legacy = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": history[-1]["reported_date"],
        "history": history
    }
    resp_legacy = client.post("/predict", json=payload_legacy)
    assert resp_legacy.status_code == 200


def test_ui_html_structure_invariants():
    """Verify that ui/index.html includes the required UI controls, governance tags, and handlers."""
    ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    assert ui_path.exists()
    content = ui_path.read_text(encoding="utf-8")

    # Horizon selector buttons
    assert 'id="btnHorizonShort"' in content
    assert 'id="btnHorizon14"' in content
    assert 'id="btnHorizon30"' in content

    # 30-day experimental labeling
    assert "Experimental" in content or "EXPERIMENTAL" in content

    # Target window notices
    assert "Target Window Policy" in content
    assert "t + 14" in content or "[t + 14, t + 17]" in content

    # Error handling & loading UI
    assert 'id="forecastLoading"' in content
    assert 'id="forecastError"' in content
    assert 'retryForecast()' in content

    # Stale response protection logic
    assert "activeRequestId" in content
    assert "AbortController" in content

    # Real history loading
    assert "loadValidHistory" in content
    assert "/history" in content
