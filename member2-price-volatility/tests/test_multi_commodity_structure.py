"""
AgroVision — Member 2: Multi-Commodity Structure & Compatibility Test Suite

Verifies:
1. Commodity separation across commodities/, artifacts/, data/raw/, and data/processed/.
2. config.py dynamic resolution for all commodities (Rice, Wheat, Potato, Tea).
3. models.predict multi-commodity execution on both precomputed features and raw history.
4. shared.predict_bridge compatibility for all commodities.
5. pipeline.py discover_commodities finds all commodities.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
REPO_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

from config import (
    set_commodity,
    get_commodity,
    raw_data_path,
    cleaned_data_path,
    price_model_path,
    volatility_model_path,
    feature_config_path,
    volatility_config_path,
    group_mappings_path,
    commodity_raw_dir,
    commodity_processed_dir,
    commodity_artifacts_dir,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    get_artifacts_dir,
)
from pipeline import discover_commodities
from models.predict import predict
from shared.predict_bridge import AgroVisionPredictor


# ============================================================
# 1. Directory Structure Separation
# ============================================================

def test_commodity_documentation_directories_exist():
    comm_dir = PROJECT_ROOT / "commodities"
    assert comm_dir.is_dir(), "commodities/ directory must exist"
    assert (comm_dir / "README.md").is_file(), "commodities/README.md must exist"

    for c in ["rice", "wheat", "potato", "tea"]:
        sub = comm_dir / c
        assert sub.is_dir(), f"commodities/{c}/ directory must exist"
        readme = sub / "README.md"
        assert readme.is_file(), f"commodities/{c}/README.md must exist"
        assert readme.stat().st_size > 200, f"commodities/{c}/README.md should not be empty"


def test_commodity_artifacts_directories_exist():
    art_dir = get_artifacts_dir()
    assert (art_dir / "README.md").is_file(), "artifacts/README.md must exist"

    for c in ["rice", "wheat", "potato", "tea"]:
        sub = art_dir / c
        assert sub.is_dir(), f"artifacts/{c}/ directory must exist"
        assert (sub / "README.md").is_file(), f"artifacts/{c}/README.md must exist"
        assert (sub / f"{c}_price_model.pkl").is_file(), f"artifacts/{c}/{c}_price_model.pkl must exist"
        assert (sub / f"{c}_volatility_model.pkl").is_file() or (sub / f"{c}_volatility_normalized_model.pkl").is_file(), f"artifacts/{c}/ volatility model must exist"
        assert (sub / f"{c}_feature_config.json").is_file() or (sub / f"{c}_price_final_config.json").is_file(), f"artifacts/{c}/ feature config must exist"


def test_commodity_raw_data_directories_exist():
    for c in ["rice", "wheat", "potato", "tea"]:
        sub = RAW_DATA_DIR / c
        assert sub.is_dir(), f"data/raw/{c}/ directory must exist"
        csv_file = sub / f"{c}.csv"
        assert csv_file.is_file(), f"data/raw/{c}/{c}.csv must exist"
        assert csv_file.stat().st_size > 1000, f"data/raw/{c}/{c}.csv must not be empty"

    assert (RAW_DATA_DIR / "README.md").is_file(), "data/raw/README.md must exist"


def test_commodity_processed_data_directories_exist():
    for c in ["rice", "wheat", "potato", "tea"]:
        sub = PROCESSED_DATA_DIR / c
        assert sub.is_dir(), f"data/processed/{c}/ directory must exist"
        assert (sub / f"{c}_cleaned.csv").is_file(), f"data/processed/{c}/{c}_cleaned.csv must exist"
        assert (sub / f"{c}_features_final.csv").is_file(), f"data/processed/{c}/{c}_features_final.csv must exist"

    assert (PROCESSED_DATA_DIR / "README.md").is_file(), "data/processed/README.md must exist"


# ============================================================
# 2. Config Path Resolution for All Commodities
# ============================================================

@pytest.mark.parametrize("comm", ["rice", "wheat", "potato", "tea"])
def test_config_paths_resolve_for_all_commodities(comm):
    set_commodity(comm)
    assert get_commodity() == comm

    raw = raw_data_path()
    assert raw.exists(), f"raw_data_path() failed to find raw CSV for '{comm}': {raw}"

    cleaned = cleaned_data_path()
    assert cleaned.exists(), f"cleaned_data_path() failed for '{comm}': {cleaned}"

    price_m = price_model_path()
    assert price_m.exists(), f"price_model_path() failed for '{comm}': {price_m}"

    vol_m = volatility_model_path()
    assert vol_m.exists(), f"volatility_model_path() failed for '{comm}': {vol_m}"

    feat_cfg = feature_config_path()
    assert feat_cfg.exists(), f"feature_config_path() failed for '{comm}': {feat_cfg}"

    mapping = group_mappings_path(comm)
    assert mapping.exists(), f"group_mappings_path() failed for '{comm}': {mapping}"


def test_discover_commodities_finds_all():
    found = discover_commodities()
    for expected in ["rice", "wheat", "potato", "tea"]:
        assert expected in found, f"discover_commodities() did not find '{expected}' in {found}"


# ============================================================
# 3. Multi-Commodity Prediction (Wheat, Potato, Rice, Tea)
# ============================================================

def test_predict_wheat_from_raw_history():
    """Verify Wheat prediction from 35 raw daily observations."""
    dates = pd.date_range("2024-01-01", periods=35)
    history = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Asansol",
            "Variety": "Kalyan",
            "Modal Price (Rs./Quintal)": 1800.0 + (i % 4) * 15,
            "Arrivals (Tonnes)": 30.0 + i
        }
        for i, d in enumerate(dates)
    ])

    res = predict(history, commodity="wheat")
    assert "predicted_price" in res
    assert "volatility" in res
    assert res["commodity"] == "wheat"
    assert res["predicted_price"] > 100.0
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_potato_from_raw_history():
    """Verify Potato prediction from 35 raw daily observations."""
    dates = pd.date_range("2024-01-01", periods=35)
    history = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Champadanga",
            "Variety": "Jyoti",
            "Modal Price (Rs./Quintal)": 950.0 + (i % 5) * 10,
            "Arrivals (Tonnes)": 120.0 + i
        }
        for i, d in enumerate(dates)
    ])

    res = predict(history, commodity="potato")
    assert "predicted_price" in res
    assert "volatility" in res
    assert res["commodity"] == "potato"
    assert res["predicted_price"] > 50.0
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_rice_from_raw_history():
    """Verify Rice prediction from 35 raw daily observations."""
    dates = pd.date_range("2024-01-01", periods=35)
    history = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Modal Price (Rs./Quintal)": 2500.0 + (i % 3) * 20,
            "Arrivals (Tonnes)": 100.0 + i
        }
        for i, d in enumerate(dates)
    ])

    res = predict(history, commodity="rice")
    assert "predicted_price" in res
    assert "volatility" in res
    assert res["commodity"] == "rice"
    assert res["predicted_price"] > 500.0
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# 4. Shared Bridge Integration for All Commodities
# ============================================================

@pytest.mark.parametrize("comm", ["wheat", "potato", "rice", "tea"])
def test_shared_bridge_predict_price_all_commodities(comm):
    bridge = AgroVisionPredictor()

    if comm == "tea":
        sample = pd.DataFrame([{
            "Modal Price (Rs./Quintal)": 250.0,
            "lag_1": 240.0, "lag_7": 235.0, "lag_14": 230.0, "lag_30": 225.0,
            "rolling_mean_7": 242.0, "rolling_mean_14": 240.0, "rolling_mean_30": 238.0,
            "rolling_std_7": 5.1, "rolling_std_14": 6.2,
            "year": 2024, "day": 15, "month": 3, "day_of_week": 2, "week_of_year": 11,
            "arrival_lag_1": 50.0, "arrival_lag_7": 48.0,
            "arrival_rolling_mean_7": 49.0, "arrival_rolling_mean_14": 47.0,
            "price_pct_change": 0.5, "arrival_pct_change": 1.2,
            "price_spread": 10.0, "price_position_in_range": 0.5
        }])
    else:
        dates = pd.date_range("2024-01-01", periods=35)
        base_price = 2500.0 if comm == "rice" else (1800.0 if comm == "wheat" else 950.0)
        sample = pd.DataFrame([
            {
                "Reported Date": d.strftime("%Y-%m-%d"),
                "Market Name": "TestMarket",
                "Variety": "TestVariety",
                "Modal Price (Rs./Quintal)": base_price + (i % 4) * 10,
                "Arrivals (Tonnes)": 50.0 + i
            }
            for i, d in enumerate(dates)
        ])

    res = bridge.predict_price(sample, commodity=comm)
    assert "predicted_price_rs_quintal" in res
    assert "volatility" in res
    assert res["commodity"] == comm
    assert isinstance(res["predicted_price_rs_quintal"], (int, float))
    assert res["predicted_price_rs_quintal"] > 0
    assert res["volatility"] in ["LOW", "MEDIUM", "HIGH"]


# ============================================================
# 5. Robustness: Short-History Predictions & Separated Rice Model
# ============================================================

def test_predict_wheat_short_history():
    """Verify Wheat prediction succeeds with < 30 observations (e.g. 5 days and 1 day)."""
    # 5 observations
    dates = pd.date_range("2024-01-01", periods=5)
    hist5 = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Asansol",
            "Variety": "Kalyan",
            "Modal Price (Rs./Quintal)": 1800.0 + i * 10,
            "Arrivals (Tonnes)": 20.0 + i
        }
        for i, d in enumerate(dates)
    ])
    res5 = predict(hist5, commodity="wheat")
    assert "predicted_price" in res5
    assert not np.isnan(res5["predicted_price"])
    assert res5["volatility"] in ["LOW", "MEDIUM", "HIGH"]

    # 1 observation
    hist1 = pd.DataFrame([{
        "Reported Date": "2024-01-01",
        "Market Name": "Asansol",
        "Variety": "Kalyan",
        "Modal Price (Rs./Quintal)": 1850.0,
        "Arrivals (Tonnes)": 25.0
    }])
    res1 = predict(hist1, commodity="wheat")
    assert "predicted_price" in res1
    assert not np.isnan(res1["predicted_price"])
    assert res1["volatility"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_potato_short_history():
    """Verify Potato prediction succeeds with < 30 observations (e.g. 7 days and 1 day)."""
    # 7 observations
    dates = pd.date_range("2024-01-01", periods=7)
    hist7 = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Champadanga",
            "Variety": "Jyoti",
            "Modal Price (Rs./Quintal)": 950.0 + i * 5,
            "Arrivals (Tonnes)": 100.0 + i
        }
        for i, d in enumerate(dates)
    ])
    res7 = predict(hist7, commodity="potato")
    assert "predicted_price" in res7
    assert not np.isnan(res7["predicted_price"])
    assert res7["volatility"] in ["LOW", "MEDIUM", "HIGH"]

    # 1 observation
    hist1 = pd.DataFrame([{
        "Reported Date": "2024-01-01",
        "Market Name": "Champadanga",
        "Variety": "Jyoti",
        "Modal Price (Rs./Quintal)": 920.0,
        "Arrivals (Tonnes)": 110.0
    }])
    res1 = predict(hist1, commodity="potato")
    assert "predicted_price" in res1
    assert not np.isnan(res1["predicted_price"])
    assert res1["volatility"] in ["LOW", "MEDIUM", "HIGH"]


def test_rice_price_predictor_with_separated_artifacts():
    """Verify that deliverables RicePricePredictor loads from artifacts/rice cleanly."""
    deliverables_path = REPO_ROOT / "deliverables" / "member2-ml"
    saved_src = sys.modules.get("src")
    try:
        sys.path.insert(0, str(deliverables_path))
        sys.modules.pop("src", None)
        import src.predictor as deliv_pred
        RicePricePredictor = deliv_pred.RicePricePredictor
    finally:
        if str(deliverables_path) in sys.path:
            sys.path.remove(str(deliverables_path))
        if saved_src is not None:
            sys.modules["src"] = saved_src

    rice_art_dir = get_artifacts_dir() / "rice"
    predictor = RicePricePredictor(artifacts_dir=rice_art_dir)
    assert len(predictor.features_list) == 48

    # Verify prediction execution
    dates = pd.date_range("2024-01-01", periods=35)
    sample = pd.DataFrame([
        {
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Modal Price (Rs./Quintal)": 2500.0 + (i % 3) * 15,
            "Arrivals (Tonnes)": 100.0 + i,
            "Min Price (Rs./Quintal)": 2400.0,
            "Max Price (Rs./Quintal)": 2600.0,
        }
        for i, d in enumerate(dates)
    ])
    res = predictor.predict(sample)
    assert "predicted_modal_price" in res
    assert res["predicted_modal_price"] > 1000.0
    assert res["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]


def test_config_paths_prefer_separated_subdirectories():
    """Verify that config paths resolve to separated commodity subdirectories."""
    for comm in ["wheat", "potato", "rice", "tea"]:
        set_commodity(comm)
        cleaned = cleaned_data_path()
        assert str(comm) in str(cleaned), f"cleaned_data_path for {comm} should be in {comm} subfolder, got {cleaned}"

        price_m = price_model_path()
        assert str(comm) in str(price_m), f"price_model_path for {comm} should be in {comm} subfolder, got {price_m}"
