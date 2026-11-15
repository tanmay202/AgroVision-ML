"""
AgroVision — Wheat and Potato V2 Production Test Suite
Validates:
1. Complete V2 artifact creation for Wheat and Potato.
2. Manifest SHA-256 checksum integrity.
3. Strict protection and preservation of Rice and V1 baselines (SHA-256 equality).
4. Direct calendar-horizon models (14d and 30d) and empirical residual intervals.
5. Multi-commodity prediction runtime and backward compatibility with V1.
6. Shared bridge integration for Member 3 consumers.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import get_artifacts_dir
from models.predict import predict
from shared.predict_bridge import AgroVisionPredictor


def sha256_of_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# 1. Artifact Completeness
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_v2_artifacts_exist(comm):
    v2_dir = get_artifacts_dir() / comm / "v2"
    assert v2_dir.is_dir(), f"{v2_dir} does not exist"
    
    expected_files = [
        f"{comm}_price_model.pkl",
        f"{comm}_price_14d_model.pkl",
        f"{comm}_price_30d_model.pkl",
        f"{comm}_volatility_model.pkl",
        f"{comm}_feature_config.json",
        f"{comm}_volatility_config.json",
        f"{comm}_residual_distributions.json",
        f"{comm}_v2_metrics.json",
        "manifest.json"
    ]
    for fn in expected_files:
        fpath = v2_dir / fn
        assert fpath.is_file(), f"Missing V2 artifact: {fpath}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"


@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_v2_manifest_integrity(comm):
    v2_dir = get_artifacts_dir() / comm / "v2"
    manifest_path = v2_dir / "manifest.json"
    assert manifest_path.is_file(), f"Missing manifest at {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest.get("commodity") == comm
    assert manifest.get("version") == "2.0.0"
    artifacts = manifest.get("artifacts", {})
    
    for fn, meta in artifacts.items():
        fpath = v2_dir / fn
        assert fpath.is_file(), f"Manifest lists {fn} but it does not exist on disk"
        disk_sha = sha256_of_file(fpath)
        assert disk_sha == meta["sha256"], f"SHA256 mismatch for {fn}: {disk_sha} vs {meta['sha256']}"


def test_protected_rice_and_v1_artifacts_untouched():
    """Verify that all protected Rice artifacts and V1 baselines remain identical to baseline."""
    checksums_path = get_artifacts_dir() / "baseline_sha256_checksums.json"
    assert checksums_path.is_file(), f"Missing baseline checksums file: {checksums_path}"
    
    with open(checksums_path, "r", encoding="utf-8") as f:
        recorded_hashes = json.load(f)
        
    for rel_path, expected_hash in recorded_hashes.items():
        full_path = get_artifacts_dir() / rel_path
        assert full_path.is_file(), f"Protected file missing: {full_path}"
        actual_hash = sha256_of_file(full_path)
        assert actual_hash == expected_hash, f"CRITICAL: Protected file {rel_path} was modified! Hash {actual_hash} != {expected_hash}"


# ============================================================
# 2. Multi-Commodity Runtime & Inference Parity
# ============================================================

def make_sample_history(commodity, n_days=35):
    dates = pd.date_range("2024-01-01", periods=n_days)
    base_p = 2000.0 if commodity == "wheat" else 1000.0
    return pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "TestMandi",
            "Variety": "TestVar",
            "Modal Price (Rs./Quintal)": base_p + (i % 5) * 10,
            "Arrivals (Tonnes)": 50.0 + i
        }
        for i, d in enumerate(dates)
    ])


@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_v2_prediction_from_history(comm):
    df = make_sample_history(comm, 35)
    res = predict(df, commodity=comm, version="v2")
    
    assert "predicted_price" in res
    assert "volatility" in res
    assert res["commodity"] == comm
    assert res["version"] == "v2"
    assert res["predicted_price"] > 0
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert "volatility_probabilities" in res
    probs = res["volatility_probabilities"]
    assert set(probs.keys()) == {"LOW", "MEDIUM", "HIGH"}
    assert pytest.approx(sum(probs.values()), abs=1e-2) == 1.0


@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_v2_prediction_single_observation(comm):
    """Ensure edge case of 1 observation works without error."""
    df1 = pd.DataFrame([{
        "Reported Date": "2024-01-01",
        "Market Name": "SoloMandi",
        "Variety": "SoloVar",
        "Modal Price (Rs./Quintal)": 1950.0 if comm == "wheat" else 850.0,
        "Arrivals (Tonnes)": 15.0
    }])
    res = predict(df1, commodity=comm, version="v2")
    assert "predicted_price" in res
    assert not np.isnan(res["predicted_price"])
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]


@pytest.mark.parametrize("comm", ["wheat", "potato"])
@pytest.mark.parametrize("horizon", [14, 30])
def test_direct_horizons_with_prediction_intervals(comm, horizon):
    df = make_sample_history(comm, 35)
    res = predict(df, commodity=comm, version="v2", horizon_days=horizon)
    
    assert res["horizon_days"] == horizon
    assert "prediction_interval" in res
    interval = res["prediction_interval"]
    assert "p10" in interval and "p90" in interval
    assert interval["p10"] <= res["predicted_price"] or interval["p10"] <= interval["p90"]
    assert interval["p90"] >= interval["p10"]


@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_v1_v2_version_switch(comm):
    df = make_sample_history(comm, 35)
    res_v1 = predict(df, commodity=comm, version="v1")
    res_v2 = predict(df, commodity=comm, version="v2")
    
    assert res_v1["version"] == "v1"
    assert res_v2["version"] == "v2"
    assert res_v1["predicted_price"] > 0
    assert res_v2["predicted_price"] > 0


# ============================================================
# 3. Shared Predict Bridge Integration
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_shared_bridge_v2_price_and_horizons(comm):
    bridge = AgroVisionPredictor()
    df = make_sample_history(comm, 35)
    
    # Short-term price
    res_price = bridge.predict_price(df, commodity=comm)
    assert res_price["commodity"] == comm
    assert res_price["predicted_price_rs_quintal"] > 0
    assert res_price["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    
    # 14-day horizon
    res_14 = bridge.predict_price_horizon(df, horizon_days=14, commodity=comm)
    assert res_14["commodity"] == comm
    assert res_14["horizon_days"] == 14
    assert res_14["predicted_price_rs_quintal"] > 0
    assert "prediction_interval" in res_14
    
    # 30-day horizon
    res_30 = bridge.predict_price_horizon(df, horizon_days=30, commodity=comm)
    assert res_30["commodity"] == comm
    assert res_30["horizon_days"] == 30
    assert res_30["predicted_price_rs_quintal"] > 0
    assert "prediction_interval" in res_30
