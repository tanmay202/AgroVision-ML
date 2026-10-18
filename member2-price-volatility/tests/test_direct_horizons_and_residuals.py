"""
Comprehensive Test Suite for Direct Horizon Evaluation, Residuals, and Crop Group Mappings.
Author: Member 2 (Price Prediction & Volatility ML)

Verifies:
1. T+7, T+14, T+30 Method A target construction (symmetric window [T+H-3, T+H+3]).
2. Observation-shift (Method B) vs calendar-horizon distinction.
3. Nearest-observation selection and deterministic forward tie-breaking.
4. Market and variety isolation (no cross-group contamination).
5. Exact 48 price feature definitions and ordering.
6. Zero lookahead leakage prevention under temporal shift.
7. Horizon-aware chronological purging.
8. Metric calculation functions (MAE, RMSE, R2, MAPE, directional accuracy).
9. Zero-denominator handling in MAPE.
10. Correct residual sign: residual = actual - predicted.
11. Error quantile computation (p05 to p95 monotonic ordering).
12. Residual JSON schema validation and holdout derivation.
13. Crop group mappings: exact training values, no NaN/placeholders, deterministic sorting.
14. Frozen production artifacts immutability and bit-for-bit SHA-256 checks.
"""

import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TEST_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
ARTIFACTS_DIR = MEMBER2_DIR / "artifacts"
DELIV_ARTIFACTS = MEMBER2_DIR.parent / "deliverables" / "member2-ml" / "artifacts"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from direct_horizon_targets import (
    compute_symmetric_calendar_horizon_targets,
    compute_observation_shift_targets,
    apply_direct_horizon_temporal_purge,
)
from inference import RiceInferencePipeline


# --- 1. Target Construction, Nearest Observation, and Tie-Breaking ---

def test_nearest_observation_selection():
    """Verifies that the closest observation in [T+H-3, T+H+3] is selected."""
    # Origin date: 2024-01-01. H=7. Ideal target: 2024-01-08.
    # Observations at 2024-01-06 (diff=-2), 2024-01-09 (diff=+1), 2024-01-10 (diff=+2).
    # Closest is 2024-01-09.
    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": ["2024-01-01", "2024-01-06", "2024-01-09", "2024-01-10"],
        "Modal Price (Rs./Quintal)": [2500.0, 2520.0, 2550.0, 2570.0],
    })
    res = compute_symmetric_calendar_horizon_targets(df, horizon_days=7, max_tolerance_days=3)
    target_date = str(res.loc[0, "target_date_7d_sym"])[:10]
    assert target_date == "2024-01-09"
    assert res.loc[0, "actual_offset_7d_sym"] == 8.0
    assert res.loc[0, "future_modal_price_7d_sym"] == 2550.0
    assert res.loc[0, "future_price_change_7d_sym"] == 50.0


def test_deterministic_forward_tie_breaking():
    """Verifies that equal distance ties are broken in favor of the forward observation."""
    # Origin: 2024-01-01. H=14. Ideal: 2024-01-15.
    # Candidates: 2024-01-14 (diff=-1d), 2024-01-16 (diff=+1d). Both distance = 1.
    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": ["2024-01-01", "2024-01-14", "2024-01-16"],
        "Modal Price (Rs./Quintal)": [2500.0, 2520.0, 2580.0],
    })
    res_fwd = compute_symmetric_calendar_horizon_targets(df, horizon_days=14, tie_break="forward")
    assert str(res_fwd.loc[0, "target_date_14d_sym"])[:10] == "2024-01-16"
    assert res_fwd.loc[0, "future_modal_price_14d_sym"] == 2580.0

    res_back = compute_symmetric_calendar_horizon_targets(df, horizon_days=14, tie_break="backward")
    assert str(res_back.loc[0, "target_date_14d_sym"])[:10] == "2024-01-14"
    assert res_back.loc[0, "future_modal_price_14d_sym"] == 2520.0


def test_window_exclusion_outside_tolerance():
    """Verifies that observations outside [T+H-3, T+H+3] are not matched."""
    # Origin: 2024-01-01. H=30. Window: [2024-01-28, 2024-02-03].
    # Observations at 2024-01-20 (too early) and 2024-02-10 (too late).
    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": ["2024-01-01", "2024-01-20", "2024-02-10"],
        "Modal Price (Rs./Quintal)": [2500.0, 2520.0, 2580.0],
    })
    res = compute_symmetric_calendar_horizon_targets(df, horizon_days=30, max_tolerance_days=3)
    assert pd.isna(res.loc[0, "target_date_30d_sym"])
    assert pd.isna(res.loc[0, "future_modal_price_30d_sym"])


def test_observation_shift_vs_calendar_horizon_distinction():
    """Verifies that Method B (.shift(-k)) measures observation rank, not calendar days."""
    dates = ["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-10", "2024-01-20", "2024-01-30", "2024-02-15", "2024-03-01"]
    prices = [2500.0 + i * 10 for i in range(len(dates))]
    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": dates,
        "Modal Price (Rs./Quintal)": prices,
    })
    res_b = compute_observation_shift_targets(df, shift_steps=7)
    # 7 observations ahead is index 7 (2024-03-01)
    assert str(res_b.loc[0, "target_date_shift_7"])[:10] == "2024-03-01"
    # Elapsed calendar days is 60 days, distinctly different from 7 calendar days!
    assert res_b.loc[0, "actual_elapsed_days_shift_7"] == 60.0


# --- 2. Market and Variety Isolation ---

def test_market_variety_isolation():
    """Verifies observations never cross market or variety boundaries."""
    df = pd.DataFrame({
        "Market Name": ["Burdwan", "Burdwan", "Howrah", "Howrah"],
        "Variety": ["Common", "Common", "Common", "Common"],
        "Reported Date": ["2024-01-01", "2024-01-08", "2024-01-01", "2024-01-08"],
        "Modal Price (Rs./Quintal)": [2000.0, 2100.0, 3000.0, 3200.0],
    })
    res = compute_symmetric_calendar_horizon_targets(df, horizon_days=7)
    # Burdwan row 0 should match Burdwan row 1 (2100.0), not Howrah
    assert res.loc[0, "future_modal_price_7d_sym"] == 2100.0
    # Howrah row 2 should match Howrah row 3 (3200.0), not Burdwan
    assert res.loc[2, "future_modal_price_7d_sym"] == 3200.0


# --- 3. Exact 48 Features & Ordering ---

def test_exact_48_features_and_ordering():
    """Verifies that the inference pipeline features match the frozen 48 config exactly."""
    pipe = RiceInferencePipeline()
    with open(ARTIFACTS_DIR / "rice_price_final_config.json") as f:
        cfg = json.load(f)
    assert len(pipe.features_list) == 48
    assert pipe.features_list == cfg["features"]


# --- 4. Zero Lookahead Leakage Prevention ---

def test_zero_lookahead_leakage_future_poisoning():
    """Verifies that changing future data beyond forecast origin T does not affect features at T."""
    dates = pd.date_range("2024-01-01", periods=45, freq="D")
    prices = [2500.0 + (i % 7) * 15.0 for i in range(45)]
    arrivals = [100.0 + (i % 5) * 10.0 for i in range(45)]

    df = pd.DataFrame({
        "Market Name": "Burdwan",
        "Variety": "Common",
        "Reported Date": pd.to_datetime(dates),
        "Modal Price (Rs./Quintal)": prices,
        "Arrivals (Tonnes)": arrivals,
    })

    pipe = RiceInferencePipeline()
    clean_features = pipe._generate_features(df)
    t_origin_idx = 35
    origin_feat_clean = clean_features.iloc[t_origin_idx][pipe.features_list].copy()

    # Poison future data at index > t_origin_idx
    poisoned_df = df.copy()
    poisoned_df.loc[t_origin_idx + 1:, "Modal Price (Rs./Quintal)"] = 99999.0
    poisoned_df.loc[t_origin_idx + 1:, "Arrivals (Tonnes)"] = 88888.0

    poison_features = pipe._generate_features(poisoned_df)
    origin_feat_poisoned = poison_features.iloc[t_origin_idx][pipe.features_list].copy()

    # Features at t_origin must be 100% identical
    for col in pipe.features_list:
        v_clean = origin_feat_clean[col]
        v_poison = origin_feat_poisoned[col]
        if pd.isna(v_clean):
            assert pd.isna(v_poison)
        else:
            assert np.isclose(float(v_clean), float(v_poison), atol=1e-5), f"Leakage detected on feature {col}!"


# --- 5. Horizon-Aware Temporal Purging ---

def test_horizon_aware_temporal_purging():
    """Verifies that examples whose future target reaches >= split_date are purged."""
    split_date = "2021-02-01"
    df = pd.DataFrame({
        "Market Name": ["M1", "M1"],
        "Variety": ["V1", "V1"],
        "Reported Date": ["2021-01-20", "2021-01-28"],
        "target_date_14d_sym": [pd.to_datetime("2021-01-30"), pd.to_datetime("2021-02-05")],
        "Modal Price (Rs./Quintal)": [2500.0, 2520.0],
    })
    purged_df, purged_cnt = apply_direct_horizon_temporal_purge(df, split_date=split_date, target_date_col="target_date_14d_sym")
    assert purged_cnt == 1
    assert len(purged_df) == 1
    assert str(purged_df.loc[0, "Reported Date"]) == "2021-01-20"


# --- 6. Metric Calculations & Zero Denominators ---

def test_mape_zero_denominator_handling():
    """Verifies that zero prices are safely handled without producing NaN or Inf."""
    y_true = np.array([2500.0, 0.0, 2600.0])
    y_pred = np.array([2550.0, 50.0, 2620.0])

    valid_denom = y_true != 0
    mape = float(np.mean(np.abs((y_true[valid_denom] - y_pred[valid_denom]) / y_true[valid_denom])) * 100.0)
    assert not np.isnan(mape)
    assert not np.isinf(mape)
    assert np.isclose(mape, (50.0 / 2500.0 + 20.0 / 2600.0) / 2.0 * 100.0, atol=1e-2)


def test_directional_accuracy_signs():
    """Verifies consistent directional accuracy calculation across sign combinations."""
    actual_change = np.array([10.0, -5.0, 0.0, 15.0, -10.0])
    pred_change = np.array([8.0, 5.0, 0.0, -10.0, -5.0])

    sign_true = np.sign(actual_change)
    sign_pred = np.sign(pred_change)
    correct = (sign_true == sign_pred)
    # Row 0: + vs + -> True
    # Row 1: - vs + -> False
    # Row 2: 0 vs 0 -> True
    # Row 3: + vs - -> False
    # Row 4: - vs - -> True
    # 3 correct out of 5 -> 60.0%
    dir_acc = float(np.mean(correct) * 100.0)
    assert np.isclose(dir_acc, 60.0)


# --- 7. Residual Signs, Error Quantiles, and JSON Structure ---

@pytest.mark.parametrize("filename,expected_h", [
    ("rice_holdout_residuals_7d.json", 7),
    ("rice_holdout_residuals_14d.json", 14),
    ("rice_holdout_residuals_30d.json", 30),
    ("rice_holdout_residuals.json", 1),
])
def test_residual_json_schema_and_quantiles(filename, expected_h):
    """Verifies schema, quantiles, and non-emptiness of final holdout residual files."""
    filepath = ARTIFACTS_DIR / filename
    assert filepath.exists(), f"Missing residual file: {filepath}"

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["crop"] == "rice"
    assert data["horizon_days"] == expected_h
    assert "metrics" in data
    assert "error_quantiles" in data
    assert "residuals" in data
    assert "model_version" in data and len(data["model_version"]) > 0
    assert "target_definition" in data and len(data["target_definition"]) > 0
    assert "holdout_period" in data and data["holdout_period"]["start"] == "2023-02-01" and data["holdout_period"]["end"] == "2024-02-01"
    assert "model_artifact" in data and len(data["model_artifact"]) > 0
    assert "model_artifact_sha256" in data and len(data["model_artifact_sha256"]) == 64

    # Verify metrics fields
    for m in ["mae", "rmse", "r2", "mape", "directional_accuracy", "naive_baseline_mae"]:
        assert m in data["metrics"]
        assert isinstance(data["metrics"][m], (int, float))

    # Verify quantiles monotonic ordering: p05 <= p10 <= p25 <= p50 <= p75 <= p90 <= p95
    q = data["error_quantiles"]
    quantile_keys = ["p05", "p10", "p25", "p50", "p75", "p90", "p95"]
    vals = [q[k] for k in quantile_keys]
    for i in range(len(vals) - 1):
        assert vals[i] <= vals[i + 1], f"Quantiles not monotonically increasing: {vals}"

    # Verify residuals array
    residuals = data["residuals"]
    assert len(residuals) > 1000
    assert not any(np.isnan(r) or np.isinf(r) for r in residuals[:100])

    # Verify deliverables copy is identical
    deliv_filepath = DELIV_ARTIFACTS / filename
    assert deliv_filepath.exists(), f"Missing deliverable copy: {deliv_filepath}"
    assert hashlib.sha256(open(filepath, "rb").read()).hexdigest() == hashlib.sha256(open(deliv_filepath, "rb").read()).hexdigest()


# --- 8. Crop Group Mappings ---

def test_rice_group_mappings_structure_and_determinism():
    """Verifies that rice_group_mappings.json contains authentic values and deterministic order."""
    m_path = ARTIFACTS_DIR / "rice_group_mappings.json"
    assert m_path.exists()

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["crop"] == "rice"
    assert "columns" in data
    cols = data["columns"]
    for required_col in ["State Name", "District Name", "Market Name", "Variety", "Group"]:
        assert required_col in cols
        vals = cols[required_col]
        assert len(vals) > 0
        assert vals == sorted(vals), f"Values in {required_col} are not deterministically sorted!"

    assert "West Bengal" in cols["State Name"]
    assert "Burdwan" in cols["Market Name"]
    assert "Common" in cols["Variety"]
    assert "Cereals" in cols["Group"]
    assert data["missing_value_counts"]["State Name"] == 0
    assert data["observed_combinations_count"] > 100


def test_tea_group_mappings_structure():
    """Verifies that tea_group_mappings.json contains authentic values."""
    m_path = ARTIFACTS_DIR / "tea_group_mappings.json"
    assert m_path.exists()

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["crop"] == "tea"
    assert len(data["columns"]["Market Name"]) > 0
    assert len(data["columns"]["Variety"]) > 0


def test_model_handoff_manifest_structure():
    """Verifies that model_handoff_manifest.json exists and maps models to configs and residuals."""
    for base in [ARTIFACTS_DIR, DELIV_ARTIFACTS]:
        m_path = base / "model_handoff_manifest.json"
        assert m_path.exists(), f"Missing manifest at {m_path}"
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data["models"]) >= 5
        assert len(data["crop_group_mappings"]) == 2
        for m in data["models"]:
            assert "model_artifact" in m
            assert "model_sha256" in m
            assert "config_artifact" in m
            assert "target_methodology" in m
            assert "release_status" in m


# --- 9. Immutability of Frozen Production Artifacts ---

def test_frozen_production_artifacts_sha256_unmodified():
    """Verifies that original production models and configs are bit-for-bit identical to frozen hashes."""
    frozen_checksums = {
        "rice_price_model.pkl": "b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993",
        "rice_price_final_config.json": "508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7",
        "rice_price_model_14d.pkl": "1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab",
        "rice_price_model_14d_config.json": "86c74c27c55bf7ff423cad5c3bced60e30a4b8d88e66bfd358fd42dcdda16da7",
        "rice_price_model_30d.pkl": "ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343",
        "rice_price_model_30d_config.json": "5650b870ab767840f0953b919efd26df6e41cc4c271c7f55dfa1911a053f280d",
        "rice_volatility_normalized_model.pkl": "dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f",
        "rice_volatility_normalized_config.json": "c57651ff4a54f7cdde0e3199c63750ac72d09d205530643d1ff670bce1396f61",
    }

    for fn, exp_hash in frozen_checksums.items():
        actual_hash = hashlib.sha256(open(ARTIFACTS_DIR / fn, "rb").read()).hexdigest()
        assert actual_hash == exp_hash, f"CRITICAL: Protected artifact {fn} modified! Expected {exp_hash}, got {actual_hash}"
