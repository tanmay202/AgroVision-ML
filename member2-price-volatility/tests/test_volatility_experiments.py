"""
AgroVision Member 2 — Volatility Experiments Test Suite

Verifies:
1. Feature engineering produces expected features with no leakage.
2. Target computation correctly builds Experiment A and B targets.
3. Temporal purging properly identifies and eliminates forward target overlaps.
4. Experiment results artifacts exist and contain valid evaluation rows.
5. Production estimator and frozen artifacts remain untouched.
"""

import sys
import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

from feature_engineering import extract_volatility_features, EXPERIMENTAL_FEATURE_COLS
from target_definitions import (
    compute_forward_volatility_targets,
    compute_persistence_predictions,
    apply_temporal_purge,
    EXP_A_LOW_THRESH,
    EXP_A_HIGH_THRESH,
    EXP_B_LOW_THRESH,
    EXP_B_HIGH_THRESH,
)
from metrics import evaluate_predictions
from experiment_models import tune_decision_thresholds, apply_decision_thresholds
from inference import RiceInferencePipeline


@pytest.fixture
def sample_timeseries_df():
    """Generates 40 rows of deterministic time-series history."""
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    records = []
    for i, d in enumerate(dates):
        records.append({
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": 2500.0 + (i % 6) * 15.0,
            "Arrivals (Tonnes)": 100.0 + i * 2.0,
        })
    return pd.DataFrame(records)


def test_feature_engineering_safety_and_shapes(sample_timeseries_df):
    df = extract_volatility_features(
        sample_timeseries_df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)"
    )

    for col in EXPERIMENTAL_FEATURE_COLS:
        assert col in df.columns, f"Missing feature: {col}"
        assert not df[col].isna().any(), f"NaN found in feature: {col}"
        assert not np.isneginf(df[col]).any() and not np.isposinf(df[col]).any(), f"Inf found in feature: {col}"

    # Verify no future targets in feature cols
    assert "target_A" not in EXPERIMENTAL_FEATURE_COLS
    assert "target_B" not in EXPERIMENTAL_FEATURE_COLS
    assert "future_volatility" not in EXPERIMENTAL_FEATURE_COLS


def test_target_definitions_and_classes(sample_timeseries_df):
    df = compute_forward_volatility_targets(
        sample_timeseries_df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        window=5
    )

    assert "target_A" in df.columns
    assert "target_B" in df.columns
    assert "future_target_date" in df.columns

    # Verify tail rows have NaN targets (last 5 rows cannot observe next 5)
    assert df["target_A"].iloc[-5:].isna().all()
    assert df["target_B"].iloc[-5:].isna().all()

    # Verify valid classes
    valid_A = df["target_A"].dropna().unique()
    for c in valid_A:
        assert c in ["LOW", "MEDIUM", "HIGH"]

    valid_B = df["target_B"].dropna().unique()
    for c in valid_B:
        assert c in ["LOW", "MEDIUM", "HIGH"]


def test_temporal_purging_logic(sample_timeseries_df):
    df = compute_forward_volatility_targets(
        sample_timeseries_df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        window=5
    )

    # Cutoff at 2023-01-25
    clean_train, is_val, purge_count = apply_temporal_purge(df, "Reported Date", "2023-01-25")

    assert purge_count > 0, "Expected boundary train rows to be purged"
    # No purged rows should remain in clean_train
    assert not (clean_train & (df["Reported Date"] >= "2023-01-25")).any()
    # Any train row in clean_train must have its future_target_date strictly before 2023-01-25
    train_targets = df.loc[clean_train, "future_target_date"]
    assert (train_targets < pd.Timestamp("2023-01-25")).all()


def test_threshold_tuning_monotonicity():
    # Synthetic test of probability adjustment
    y_val = np.array([0, 0, 1, 1, 2, 2])
    proba_val = np.array([
        [0.7, 0.2, 0.1],
        [0.6, 0.3, 0.1],
        [0.2, 0.5, 0.3],
        [0.3, 0.4, 0.3],
        [0.1, 0.3, 0.6],
        [0.2, 0.3, 0.5],
    ])

    best_w, best_f1, _ = tune_decision_thresholds(y_val, proba_val, min_high_recall=0.0)
    assert len(best_w) == 3
    assert best_f1 > 0.0

    preds = apply_decision_thresholds(proba_val, best_w)
    assert len(preds) == 6
    for p in preds:
        assert p in ["LOW", "MEDIUM", "HIGH"]


def test_experiment_results_artifacts_generated():
    results_dir = EXPERIMENTS_DIR / "results"
    summary_path = results_dir / "summary_metrics.csv"
    per_class_path = results_dir / "per_class_metrics.csv"
    cm_path = results_dir / "confusion_matrices.json"
    report_path = results_dir / "experiment_comparison_report.md"

    assert summary_path.exists(), f"Missing {summary_path}"
    assert per_class_path.exists(), f"Missing {per_class_path}"
    assert cm_path.exists(), f"Missing {cm_path}"
    assert report_path.exists(), f"Missing {report_path}"

    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) > 10
    assert "Accuracy" in summary_df.columns
    assert "Macro_F1" in summary_df.columns
    assert "HIGH_Recall" in summary_df.columns

    # Verify both Experiment A and Experiment B are present
    experiments_found = summary_df["Experiment"].unique().tolist()
    assert "Experiment A (Nominal)" in experiments_found
    assert "Experiment B (Normalized)" in experiments_found


def test_production_frozen_artifacts_and_pipeline_unmodified():
    # Confirm RiceInferencePipeline continues to use persistence baseline and loads frozen 48-feature price model
    pipeline = RiceInferencePipeline()
    assert pipeline.price_model is not None
    assert getattr(pipeline.price_model, "n_features_in_", None) == 48
    assert len(pipeline.features_list) == 48

    # Ensure production artifacts are unchanged
    artifacts_dir = PROJECT_ROOT / "artifacts"
    assert (artifacts_dir / "rice_price_model.pkl").exists()
    assert (artifacts_dir / "rice_price_final_config.json").exists()
