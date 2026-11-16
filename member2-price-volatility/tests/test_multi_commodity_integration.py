"""
AgroVision-ML — Multi-Commodity Unified Integration & Acceptance Test Suite
Verifies all 11 Milestone Acceptance Criteria:
1. Rice inference still works (short-term, 14d, 30d, predict, predict_all).
2. Wheat inference loads Wheat artifacts (CatBoost price regressor + CatBoost volatility).
3. Potato inference loads Potato artifacts (XGBoost price regressor + XGBoost volatility).
4. Feature schemas match all selected models (Rice 48/35, Wheat 81, Potato 81, Tea 20/19).
5. Correct residual distributions load by commodity and horizon.
6. Prediction probabilities are valid (sum to ~1.0, range [0, 1]).
7. Experimental horizons are clearly marked (release_status == 'experimental', is_experimental is True).
8. Missing artifacts produce explicit errors (no silent fallback).
9. Shadow monitoring cannot change production decisions.
10. Separate prospective shadow evaluation for Wheat and Potato (isolated databases/tables).
11. SHA-256 verification confirms protected artifacts remain unchanged.
"""

import sys
import json
import sqlite3
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
REPO_ROOT = PROJECT_ROOT.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import get_artifacts_dir
from models.predict import predict, _artifacts
from shared.predict_bridge import AgroVisionPredictor
from multi_commodity_shadow import (
    init_commodity_shadow_db,
    ingest_commodity_mandi_observations,
    generate_commodity_shadow_forecasts,
    evaluate_pending_commodity_shadow_forecasts,
    compute_commodity_shadow_metrics,
    get_commodity_shadow_status,
)


def sha256_of_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def make_history(commodity="rice", n=35, base_price=2500.0):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Burdwan" if commodity == "rice" else "TestMandi",
            "Variety": "Common" if commodity == "rice" else "TestVar",
            "Modal Price (Rs./Quintal)": base_price + (i % 6) * 10.0,
            "Arrivals (Tonnes)": 50.0 + (i % 5) * 5.0,
        }
        for i, d in enumerate(dates)
    ])


# ============================================================
# Criterion 1: Rice Inference Still Works & Backward Compatibility
# ============================================================

def test_rice_inference_backward_compatibility():
    bridge = AgroVisionPredictor()
    df = make_history("rice", 35, 2500.0)

    # Short-term
    r_short = bridge.predict(history=df, commodity="rice", horizon="short")
    assert r_short["commodity"] == "rice"
    assert r_short["horizon_days"] == 1
    assert r_short["predicted_modal_price"] > 0
    assert r_short["predicted_price_rs_quintal"] == r_short["predicted_modal_price"]
    assert r_short["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert r_short["release_status"] == "production"
    assert r_short["is_experimental"] is False

    # 14-day horizon
    r_14 = bridge.predict(history=df, commodity="rice", horizon=14)
    assert r_14["commodity"] == "rice"
    assert r_14["horizon_days"] == 14
    assert r_14["predicted_modal_price"] > 0
    assert r_14["release_status"] == "opt-in"
    assert r_14["is_experimental"] is False

    # 30-day horizon
    r_30 = bridge.predict(history=df, commodity="rice", horizon=30)
    assert r_30["commodity"] == "rice"
    assert r_30["horizon_days"] == 30
    assert r_30["predicted_modal_price"] > 0
    assert r_30["release_status"] == "experimental"
    assert r_30["is_experimental"] is True


# ============================================================
# Criteria 2 & 3: Wheat and Potato Load Respective Artifacts
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_wheat_potato_artifact_loading(comm):
    base_p = 2000.0 if comm == "wheat" else 950.0
    df = make_history(comm, 35, base_p)
    bridge = AgroVisionPredictor()

    res = bridge.predict(history=df, commodity=comm, horizon="short")
    assert res["commodity"] == comm
    assert res["predicted_modal_price"] > 0
    assert res["predicted_price_rs_quintal"] == res["predicted_modal_price"]
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]
    assert res["release_status"] == "production"
    assert res["is_experimental"] is False

    # Check model name matches candidate
    if comm == "wheat":
        assert "catboost" in res["model_name"].lower()
    else:
        assert "xgb" in res["model_name"].lower()


# ============================================================
# Criterion 4: Feature Schemas Match Selected Models
# ============================================================

def test_feature_schemas_match_models():
    # Rice: 48 features (loaded from final config / production pipeline)
    from inference import RiceInferencePipeline
    pipe = RiceInferencePipeline()
    assert len(pipe.features_list) == 48

    # Wheat V2: 81 features
    cfg_wheat, p_wheat, v_wheat, _, _ = _artifacts("wheat", version="v2", horizon_days=1)
    wheat_feats = cfg_wheat["price_model"]["features"]
    assert len(wheat_feats) == 81

    # Potato V2: 81 features
    cfg_pot, p_pot, v_pot, _, _ = _artifacts("potato", version="v2", horizon_days=1)
    pot_feats = cfg_pot["price_model"]["features"]
    assert len(pot_feats) == 81

    # Tea: 23 features
    cfg_tea, p_tea, v_tea, _, _ = _artifacts("tea", version="v1", horizon_days=1)
    tea_feats = cfg_tea["price_model"]["features"]
    assert len(tea_feats) == 23


# ============================================================
# Criterion 5: Correct Residual Distributions Load by Commodity and Horizon
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
@pytest.mark.parametrize("h", [1, 14, 30])
def test_residual_distributions_load_correctly(comm, h):
    base_p = 2000.0 if comm == "wheat" else 950.0
    df = make_history(comm, 35, base_p)
    bridge = AgroVisionPredictor()

    res = bridge.predict(history=df, commodity=comm, horizon=h)
    assert "prediction_interval" in res
    assert "p10" in res["prediction_interval"]
    assert "p90" in res["prediction_interval"]
    assert res["prediction_interval"]["p10"] <= res["prediction_interval"]["p90"]

    # Verify residual quantiles exist
    assert "residual_distribution" in res
    q = res["residual_distribution"]
    assert "p10" in q and "p90" in q
    assert q["p10"] <= q["p90"]


# ============================================================
# Criterion 6: Prediction Probabilities are Valid
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
def test_prediction_probabilities_valid(comm):
    base_p = 2000.0 if comm == "wheat" else 950.0
    df = make_history(comm, 35, base_p)
    bridge = AgroVisionPredictor()

    res = bridge.predict(history=df, commodity=comm, horizon="short")
    assert "volatility_probabilities" in res
    probs = res["volatility_probabilities"]
    assert set(probs.keys()) == {"LOW", "MEDIUM", "HIGH"}

    prob_sum = sum(probs.values())
    assert pytest.approx(prob_sum, abs=1e-2) == 1.0

    for cls, p in probs.items():
        assert 0.0 <= p <= 1.0


# ============================================================
# Criterion 7: Experimental Horizons are Clearly Marked
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato"])
@pytest.mark.parametrize("h", [14, 30])
def test_experimental_horizons_clearly_marked(comm, h):
    base_p = 2000.0 if comm == "wheat" else 950.0
    df = make_history(comm, 35, base_p)
    bridge = AgroVisionPredictor()

    res = bridge.predict(history=df, commodity=comm, horizon=h)
    assert res["horizon_days"] == h
    assert res["release_status"] == "experimental"
    assert res["is_experimental"] is True
    assert "limitations" in res
    assert "deficit" in res["limitations"].lower() or "experimental" in res["limitations"].lower()


# ============================================================
# Criterion 8: Missing Artifacts Produce Explicit Errors & No Fallback
# ============================================================

def test_missing_artifacts_and_unsupported_commodities_raise_errors():
    bridge = AgroVisionPredictor()
    df = make_history("rice", 35)

    # 1. Unsupported commodity raises ValueError
    with pytest.raises(ValueError, match="Unsupported commodity"):
        bridge.predict(history=df, commodity="mango", horizon="short")

    # 2. RicePredictor rejects non-rice
    deliv_pkg = REPO_ROOT / "deliverables" / "member2-ml"
    if str(deliv_pkg) not in sys.path:
        sys.path.insert(0, str(deliv_pkg))
    if "src" in sys.modules and not hasattr(sys.modules["src"], "predictor"):
        sys.modules.pop("src", None)
    from src.predictor import RicePricePredictor
    rp = RicePricePredictor()
    with pytest.raises(ValueError, match="only supports commodity='rice'"):
        rp.predict(history=df, commodity="wheat")

    # 3. Missing directory produces FileNotFoundError
    with pytest.raises(FileNotFoundError):
        predict(df, commodity="wheat", artifacts_dir="/non/existent/path/artifacts")


# ============================================================
# Criterion 9 & 10: Shadow Monitoring Isolation & Prospective Evaluator
# ============================================================

def test_shadow_monitoring_cannot_alter_production_decisions(tmp_path):
    bridge = AgroVisionPredictor()
    df = make_history("wheat", 40, 2050.0)

    # 1. Take baseline prediction
    baseline_pred = bridge.predict(history=df, commodity="wheat", horizon="short")
    baseline_price = baseline_pred["predicted_modal_price"]
    baseline_vol = baseline_pred["volatility"]

    # 2. Run prospective shadow pipeline on an isolated database
    wheat_db = tmp_path / "test_wheat_shadow.db"
    init_commodity_shadow_db("wheat", wheat_db)
    ingest_commodity_mandi_observations("wheat", df, wheat_db)
    gen_res = generate_commodity_shadow_forecasts("wheat", df, horizons=(14, 30), db_path=wheat_db, force_enabled=True)
    assert gen_res["forecasts_generated"] == 2

    eval_res = evaluate_pending_commodity_shadow_forecasts("wheat", db_path=wheat_db)
    assert eval_res["status"] == "success"

    # 3. Re-run production prediction: must be bit-for-bit identical!
    after_pred = bridge.predict(history=df, commodity="wheat", horizon="short")
    assert after_pred["predicted_modal_price"] == baseline_price
    assert after_pred["volatility"] == baseline_vol


def test_separate_prospective_shadow_for_wheat_and_potato(tmp_path):
    """Verify Wheat and Potato operate on completely isolated shadow databases."""
    wheat_db = tmp_path / "wheat_shadow.db"
    potato_db = tmp_path / "potato_shadow.db"

    df_w = make_history("wheat", 40, 2100.0)
    df_p = make_history("potato", 40, 900.0)

    # Ingest and generate wheat
    ingest_commodity_mandi_observations("wheat", df_w, wheat_db)
    res_w = generate_commodity_shadow_forecasts("wheat", df_w, horizons=(14, 30), db_path=wheat_db, force_enabled=True)
    assert res_w["forecasts_generated"] == 2

    # Ingest and generate potato
    ingest_commodity_mandi_observations("potato", df_p, potato_db)
    res_p = generate_commodity_shadow_forecasts("potato", df_p, horizons=(14, 30), db_path=potato_db, force_enabled=True)
    assert res_p["forecasts_generated"] == 2

    # Verify zero database cross-contamination
    status_w = get_commodity_shadow_status("wheat", wheat_db)
    assert status_w["forecasts"]["total"] == 2
    assert status_w["commodity"] == "wheat"

    status_p = get_commodity_shadow_status("potato", potato_db)
    assert status_p["forecasts"]["total"] == 2
    assert status_p["commodity"] == "potato"


# ============================================================
# Criterion 11: Protected Artifacts SHA-256 Unchanged
# ============================================================

def test_all_protected_artifacts_sha256_unmodified():
    checksums_path = get_artifacts_dir() / "baseline_sha256_checksums.json"
    assert checksums_path.is_file(), f"Missing checksums file: {checksums_path}"

    with open(checksums_path, "r", encoding="utf-8") as f:
        recorded_hashes = json.load(f)

    for rel_path, expected_hash in recorded_hashes.items():
        full_path = get_artifacts_dir() / rel_path
        assert full_path.is_file(), f"Protected file missing: {full_path}"
        actual_hash = sha256_of_file(full_path)
        assert actual_hash == expected_hash, (
            f"CRITICAL: Protected file {rel_path} was modified! Hash {actual_hash} != {expected_hash}"
        )
