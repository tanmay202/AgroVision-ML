import pytest
from fastapi.testclient import TestClient
import pandas as pd
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from api import app, pipeline_instance

# TestClient automatically triggers startup events if used as a context manager
@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def get_valid_history(days=40):
    dates = pd.date_range("2024-01-01", periods=days)
    history = []
    for i, d in enumerate(dates):
        history.append({
            "reported_date": d.strftime("%Y-%m-%d"),
            "modal_price": 2000.0 + i * 10,
            "arrivals": 100.0 + i
        })
    return history

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "rice-price-xgb",
        "volatility": "persistence"
    }

def test_valid_predict(client):
    history = get_valid_history(35)
    payload = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-02-04",
        "history": history
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["market"] == "Burdwan"
    assert data["variety"] == "Common"
    assert data["prediction_date"] == "2024-02-04"
    assert "predicted_modal_price" in data
    assert "predicted_volatility_range" in data
    assert data["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]

def test_insufficient_history(client):
    # Only 10 days of history
    history = get_valid_history(10)
    payload = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-01-10",
        "history": history
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Insufficient history" in response.json()["detail"]

def test_missing_fields(client):
    # Missing 'market' field
    payload = {
        "variety": "Common",
        "prediction_date": "2024-01-10",
        "history": get_valid_history(35)
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic validation error

def test_empty_history(client):
    payload = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-01-10",
        "history": []
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Empty history" in response.json()["detail"]

def test_future_rows_cannot_affect_prediction(client):
    # Generate 40 days of history
    history_clean = get_valid_history(40)
    
    # Prediction date is day 35
    prediction_date = history_clean[34]["reported_date"]
    
    payload_clean = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": prediction_date,
        "history": history_clean[:35] # Only provide history up to day 35
    }
    
    response_clean = client.post("/predict", json=payload_clean)
    assert response_clean.status_code == 200
    
    # Now provide the same target prediction date, but provide all 40 days in history (5 future days)
    # The future days should be filtered out by the inference pipeline
    payload_poisoned = {
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": prediction_date,
        "history": history_clean
    }
    
    response_poisoned = client.post("/predict", json=payload_poisoned)
    assert response_poisoned.status_code == 200
    
    assert response_clean.json() == response_poisoned.json()

def test_model_loaded_only_once():
    # Because we're in tests, we can verify that pipeline_instance is the same object
    # across multiple requests
    from api import app, pipeline_instance
    with TestClient(app) as client:
        # Load happened on startup
        from api import pipeline_instance as pl1
        assert pl1 is not None
        
        client.get("/health")
        from api import pipeline_instance as pl2
        
        client.get("/health")
        from api import pipeline_instance as pl3
        
        assert pl1 is pl2
        assert pl2 is pl3
