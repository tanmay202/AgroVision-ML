"""
AgroVision-ML — Universal Multi-Commodity Staging API Verification
==================================================================
Tests all Phase 6 staging deployment requirements:
- Multi-commodity routing (Rice, Wheat, Potato, Tea)
- All supported horizons (short, 14d, 30d)
- Historical input validation (< 30 rows -> 400)
- Missing commodity / invalid payload rejection
- Real-time observability telemetry and latencies
- Live ingestion endpoints with alias normalization
- Prospective shadow monitoring status and outcome evaluation
- Concurrent request execution
- API latency (< 500ms)
- Security: zero credential leakage in responses or logs
- 100% backward compatibility with legacy Rice endpoints
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pytest
from fastapi.testclient import TestClient
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TEST_DIR.parent
PROJECT_ROOT = MEMBER2_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEMBER2_DIR) not in sys.path:
    sys.path.insert(0, str(MEMBER2_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from api_v2 import app
from config import is_horizon_14d_enabled, is_horizon_30d_enabled


@pytest.fixture(scope="module")
def staging_client():
    """Provides a shared TestClient for the Staging API with lifespan execution."""
    with TestClient(app) as client:
        yield client


def make_history(days: int = 35, base_price: float = 2000.0, trend: float = 5.0):
    """Generates synthetic valid APMC historical observations."""
    dates = pd.date_range("2024-01-01", periods=days)
    return [
        {
            "reported_date": d.strftime("%Y-%m-%d"),
            "modal_price": round(base_price + i * trend, 2),
            "arrivals": round(50.0 + (i % 7) * 2.0, 2),
        }
        for i, d in enumerate(dates)
    ]


# ============================================================
# 1. HEALTH AND METRICS ENDPOINTS
# ============================================================

def test_v2_health_all(staging_client):
    """Verifies that GET /v2/health returns overall system health and all 4 commodities."""
    res = staging_client.get("/v2/health")
    assert res.status_code == 200
    data = res.json()
    assert "overall_status" in data
    assert "commodities" in data
    for comm in ["rice", "wheat", "potato", "tea"]:
        assert comm in data["commodities"]
        assert data["commodities"][comm]["status"] in ("healthy", "degraded")


def test_v2_health_per_commodity(staging_client):
    """Verifies that GET /v2/health/{commodity} returns independent status for each commodity."""
    for comm in ["rice", "wheat", "potato", "tea"]:
        res = staging_client.get(f"/v2/health/{comm}")
        assert res.status_code == 200
        data = res.json()
        assert data["commodity"] == comm
        assert "model_version" in data
        assert "ingestion_success_rate_pct" in data


def test_v2_health_unknown_commodity(staging_client):
    """Verifies that GET /v2/health/unknown returns 404."""
    res = staging_client.get("/v2/health/soybean")
    assert res.status_code == 404


def test_v2_metrics_endpoints(staging_client):
    """Verifies that GET /v2/metrics returns telemetry for all commodities."""
    res = staging_client.get("/v2/metrics")
    assert res.status_code == 200
    data = res.json()
    for comm in ["rice", "wheat", "potato", "tea"]:
        assert comm in data
        assert "ingestion" in data[comm]
        assert "predictions" in data[comm]


# ============================================================
# 2. MULTI-COMMODITY PREDICTIONS
# ============================================================

def test_v2_predict_rice_short(staging_client):
    """Verifies Rice short-term prediction with interval quantiles."""
    payload = {
        "commodity": "rice",
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-02-04",
        "horizon": "short",
        "history": make_history(35, base_price=2200.0),
        "return_intervals": True,
    }
    t0 = time.perf_counter()
    res = staging_client.post("/v2/predict", json=payload)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res.status_code == 200
    data = res.json()
    assert data["commodity"] == "rice"
    assert data["predicted_modal_price"] > 0
    assert data["predicted_price_rs_quintal"] == data["predicted_modal_price"]
    assert data["horizon_days"] == 1
    assert data["release_status"] == "production"
    assert not data["is_experimental"]
    assert "prediction_interval" in data
    assert latency_ms < 500.0, f"Latency {latency_ms:.2f}ms exceeded 500ms SLA"


def test_v2_predict_wheat_short_and_14d(staging_client):
    """Verifies Wheat short-term (production) and 14-day (experimental) predictions."""
    history = make_history(35, base_price=2400.0)

    # 1. Short-term
    res_short = staging_client.post("/v2/predict", json={
        "commodity": "wheat",
        "market": "Bahraich",
        "variety": "Dara",
        "prediction_date": "2024-02-04",
        "horizon": "short",
        "history": history,
    })
    assert res_short.status_code == 200
    d_short = res_short.json()
    assert d_short["commodity"] == "wheat"
    assert d_short["release_status"] == "production"
    assert not d_short["is_experimental"]

    # 2. 14-day horizon (Experimental V2)
    res_14d = staging_client.post("/v2/predict", json={
        "commodity": "wheat",
        "market": "Bahraich",
        "variety": "Dara",
        "prediction_date": "2024-02-04",
        "horizon": 14,
        "history": history,
    })
    assert res_14d.status_code == 200
    d_14d = res_14d.json()
    assert d_14d["commodity"] == "wheat"
    assert d_14d["horizon_days"] == 14
    assert d_14d["release_status"] == "experimental"
    assert d_14d["is_experimental"] is True
    assert "prediction_interval" in d_14d


def test_v2_predict_potato_30d(staging_client):
    """Verifies Potato 30-day (experimental) prediction."""
    res = staging_client.post("/v2/predict", json={
        "commodity": "potato",
        "market": "Agra",
        "variety": "Desi",
        "prediction_date": "2024-02-04",
        "horizon": "30d",
        "history": make_history(35, base_price=1100.0),
    })
    assert res.status_code == 200
    data = res.json()
    assert data["commodity"] == "potato"
    assert data["horizon_days"] == 30
    assert data["release_status"] == "experimental"
    assert data["is_experimental"] is True


def test_v2_predict_tea(staging_client):
    """Verifies Tea short-term prediction."""
    res = staging_client.post("/v2/predict", json={
        "commodity": "tea",
        "market": "Siliguri",
        "variety": "CTC",
        "prediction_date": "2024-02-04",
        "history": make_history(35, base_price=210.0, trend=0.5),
    })
    assert res.status_code == 200
    data = res.json()
    assert data["commodity"] == "tea"
    assert data["model_name"] == "tea-price-xgb"
    assert data["release_status"] == "production"
    assert not data["is_experimental"]


# ============================================================
# 3. INPUT VALIDATION & ERROR REJECTION
# ============================================================

def test_v2_predict_insufficient_history(staging_client):
    """Verifies that history < 30 observations returns HTTP 400 Bad Request."""
    res = staging_client.post("/v2/predict", json={
        "commodity": "rice",
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-02-04",
        "history": make_history(25),  # 25 < 30
    })
    assert res.status_code == 400
    assert "minimum 30 observations required" in res.json()["detail"].lower()


def test_v2_predict_unsupported_commodity(staging_client):
    """Verifies that an unsupported commodity returns HTTP 400."""
    res = staging_client.post("/v2/predict", json={
        "commodity": "coffee",
        "market": "Coorg",
        "variety": "Arabica",
        "prediction_date": "2024-02-04",
        "history": make_history(35),
    })
    assert res.status_code == 400
    assert "unsupported commodity" in res.json()["detail"].lower()


# ============================================================
# 4. INGESTION AND SHADOW MONITORING ENDPOINTS
# ============================================================

def test_v2_ingest_and_status(staging_client):
    """Verifies live batch observation ingestion and status reporting."""
    batch_records = [
        {"Market": "BURDWAN", "Variety": "COMMON", "Reported Date": "2024-02-05", "Modal Price (Rs./Quintal)": "2250", "Arrivals (Tonnes)": "120"},
        {"Market": "BURDWAN", "Variety": "COMMON", "Reported Date": "2024-02-06", "Modal Price (Rs./Quintal)": "2260", "Arrivals (Tonnes)": "115"},
    ]
    res_ingest = staging_client.post("/v2/ingest/rice", json={"records": batch_records})
    assert res_ingest.status_code == 200
    data_ingest = res_ingest.json()
    assert data_ingest["status"] == "success"
    assert data_ingest["inserted_count"] >= 0

    res_status = staging_client.get("/v2/ingestion/status/rice")
    assert res_status.status_code == 200
    assert "total_records" in res_status.json()


def test_v2_shadow_endpoints(staging_client):
    """Verifies shadow monitoring status and metrics endpoints."""
    res_status = staging_client.get("/v2/shadow/wheat/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert "forecasts" in data_status
    assert "observations" in data_status

    res_metrics = staging_client.get("/v2/shadow/wheat/metrics")
    assert res_metrics.status_code == 200
    assert "horizons" in res_metrics.json()

    res_eval = staging_client.post("/v2/shadow/wheat/evaluate", params={"as_of_date": "2024-02-10"})
    assert res_eval.status_code == 200


# ============================================================
# 5. CONCURRENT REQUESTS & LATENCY BENCHMARK
# ============================================================

def test_v2_concurrent_predictions(staging_client):
    """Verifies thread-safety and SLA compliance under 10 concurrent prediction requests."""
    payloads = [
        {
            "commodity": comm,
            "market": "TestMarket",
            "variety": "TestVariety",
            "prediction_date": "2024-02-04",
            "history": make_history(35, base_price=2000.0),
        }
        for comm in ["rice", "wheat", "potato", "tea"] * 3
    ][:10]

    def _call(p):
        t0 = time.perf_counter()
        r = staging_client.post("/v2/predict", json=p)
        lat = (time.perf_counter() - t0) * 1000.0
        return r.status_code, lat

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(_call, payloads))

    for status_code, lat in results:
        assert status_code == 200
        assert lat < 800.0  # Under concurrent load, allow up to 800ms


# ============================================================
# 6. SECURITY: CREDENTIAL PROTECTION
# ============================================================

def test_v2_security_no_api_key_leakage(staging_client):
    """Verifies that API keys and internal credentials are never exposed in error responses."""
    res = staging_client.post("/v2/predict", json={
        "commodity": "rice",
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-02-04",
        "history": [],  # Causes error
    })
    error_text = res.text.lower()
    for sensitive in ["api_key", "secret", "password", "token", "579b464db66ec23bdd000001"]:
        assert sensitive not in error_text


# ============================================================
# 7. BACKWARD COMPATIBILITY WITH RICE LEGACY ENDPOINTS
# ============================================================

def test_legacy_rice_endpoints_backward_compatibility(staging_client):
    """Verifies that legacy /health, /predict, and /predict-horizon continue to work unchanged."""
    # 1. GET /health
    res_health = staging_client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json() == {
        "status": "ok",
        "model": "rice-price-xgb",
        "volatility": "persistence",
    }

    # 2. POST /predict
    history = make_history(35, base_price=2100.0)
    res_predict = staging_client.post("/predict", json={
        "market": "Burdwan",
        "variety": "Common",
        "prediction_date": "2024-02-04",
        "history": history,
    })
    assert res_predict.status_code == 200
    d_pred = res_predict.json()
    assert d_pred["market"] == "Burdwan"
    assert d_pred["predicted_modal_price"] > 0
    assert "predicted_volatility_class" in d_pred

    # 3. POST /predict-horizon (if 14d is enabled)
    if is_horizon_14d_enabled():
        res_horizon = staging_client.post("/predict-horizon", json={
            "market": "Burdwan",
            "variety": "Common",
            "prediction_date": "2024-02-04",
            "horizon_days": 14,
            "history": history,
        })
        assert res_horizon.status_code == 200
        d_h = res_horizon.json()
        assert d_h["horizon_days"] == 14
        assert d_h["current_modal_price"] > 0
