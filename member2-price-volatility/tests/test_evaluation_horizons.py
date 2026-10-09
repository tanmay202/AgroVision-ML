"""
Unit and integration tests for Rice Price Forecast Horizon Evaluation.

Verifies:
1. Forecast horizon calculation and assignment (days_to_next).
2. Metric calculation correctness (MAE, RMSE, R², MAPE, 100-MAPE, tolerance accuracy).
3. Zero-actual price handling in MAPE.
4. Point-in-time leakage prevention (future target price is never a feature).
5. Unsupported status of 14-day forecasts under the frozen model.
6. Artifact preservation and deterministic reproduction of evaluation results.
"""

import sys
import hashlib
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = TESTS_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
REPO_ROOT = MEMBER2_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from evaluation.evaluate_forecast_horizons import (
    calculate_mape,
    compute_regression_metrics,
    load_and_prepare_evaluation_dataset,
    run_evaluation,
)
from inference import RiceInferencePipeline
from config import DATE_COLUMN, PRICE_COLUMN, GROUP_COLUMNS


# ============================================================
# 1. HORIZON ASSIGNMENT TESTS
# ============================================================

def test_forecast_horizon_assignment_logic():
    """Verify days_to_next accurately calculates elapsed calendar days to next observation."""
    dates = pd.to_datetime(["2023-05-01", "2023-05-02", "2023-05-09", "2023-05-12"])
    df = pd.DataFrame({
        "Market Name": ["M1"] * 4,
        "Variety": ["V1"] * 4,
        DATE_COLUMN: dates,
        PRICE_COLUMN: [2000.0, 2010.0, 2020.0, 2030.0],
    })

    next_date = df.groupby(GROUP_COLUMNS)[DATE_COLUMN].shift(-1)
    days_to_next = (next_date - df[DATE_COLUMN]).dt.days

    # 2023-05-01 -> 2023-05-02 is 1 day
    assert days_to_next.iloc[0] == 1
    # 2023-05-02 -> 2023-05-09 is 7 days
    assert days_to_next.iloc[1] == 7
    # 2023-05-09 -> 2023-05-12 is 3 days
    assert days_to_next.iloc[2] == 3
    # Last row has no next observation
    assert pd.isna(days_to_next.iloc[3])


def test_holdout_horizons_strictly_within_1_to_7_days():
    """Verify that all evaluated holdout rows satisfy 1 <= forecast_horizon_days <= 7."""
    eval_df, _ = load_and_prepare_evaluation_dataset(
        holdout_start="2023-02-01",
        holdout_end="2024-02-01",
    )
    horizons = eval_df["forecast_horizon_days"]
    assert (horizons >= 1).all(), "Found horizon < 1 day!"
    assert (horizons <= 7).all(), "Found horizon > 7 days!"
    assert 1 in horizons.values, "1-day horizon must exist in holdout"
    assert 7 in horizons.values, "7-day horizon must exist in holdout"


def test_fourteen_day_horizon_is_unsupported_in_dataset():
    """Verify that 14-day horizon does not exist in the holdout dataset."""
    eval_df, _ = load_and_prepare_evaluation_dataset(
        holdout_start="2023-02-01",
        holdout_end="2024-02-01",
    )
    assert (eval_df["forecast_horizon_days"] == 14).sum() == 0, (
        "14-day horizon unexpectedly present in holdout dataset!"
    )


# ============================================================
# 2. METRIC CALCULATION TESTS
# ============================================================

def test_compute_regression_metrics_known_values():
    """Verify metric calculations against hand-computed deterministic values."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 300.0, 420.0])
    # Absolute errors: 10, 10, 0, 20 -> MAE = 10.0
    # Squared errors: 100, 100, 0, 400 -> Mean = 150 -> RMSE = sqrt(150) = 12.2474
    # Percentage errors: 10%, 5%, 0%, 5% -> MAPE = 5.0%
    # 100 - MAPE = 95.0%

    metrics = compute_regression_metrics(y_true, y_pred, tolerance_rs=25.0)

    assert metrics["N"] == 4
    assert np.isclose(metrics["MAE"], 10.0)
    assert np.isclose(metrics["RMSE"], np.sqrt(150.0))
    assert np.isclose(metrics["MAPE"], 5.0)
    assert np.isclose(metrics["100 - MAPE (%)"], 95.0)
    assert np.isclose(metrics["Tolerance_Accuracy (%)"], 100.0)

    # Test tighter tolerance of 15.0: errors are 10, 10, 0 (<=15) and 20 (>15) -> 75%
    metrics_tight = compute_regression_metrics(y_true, y_pred, tolerance_rs=15.0)
    assert np.isclose(metrics_tight["Tolerance_Accuracy (%)"], 75.0)


def test_mape_zero_actual_handling():
    """Verify calculate_mape safely excludes zero actual values without throwing ZeroDivisionError."""
    # When zero actual is mixed with non-zeros
    y_true = np.array([0.0, 100.0, 200.0])
    y_pred = np.array([10.0, 110.0, 190.0])
    # Valid non-zeros: 10/100 (10%) and 10/200 (5%) -> Mean = 7.5%
    mape = calculate_mape(y_true, y_pred)
    assert np.isclose(mape, 7.5)

    # When all actuals are zero
    y_zero = np.array([0.0, 0.0])
    y_pred_zero = np.array([10.0, 20.0])
    assert np.isnan(calculate_mape(y_zero, y_pred_zero))


def test_empty_dataset_metric_handling():
    """Verify compute_regression_metrics handles empty inputs gracefully."""
    metrics = compute_regression_metrics([], [])
    assert metrics["N"] == 0
    assert np.isnan(metrics["MAE"])
    assert np.isnan(metrics["RMSE"])


# ============================================================
# 3. LEAKAGE PREVENTION & INTEGRITY TESTS
# ============================================================

def test_target_price_excluded_from_model_features():
    """Verify that target price and future columns are strictly absent from features."""
    pipeline = RiceInferencePipeline()
    leaky_targets = [
        "future_modal_price",
        "actual_modal_price",
        "future_price_change",
        "future_price_pct_change",
        "future_log_return",
        "days_to_next",
        "forecast_horizon_days",
    ]
    for target in leaky_targets:
        assert target not in pipeline.features_list, (
            f"LEAKAGE: '{target}' found in production feature list!"
        )


def test_point_in_time_prediction_immunity():
    """Verify that observations after the prediction date do not affect predictions."""
    pipeline = RiceInferencePipeline()
    dates = pd.date_range("2024-01-01", periods=35)
    records = []
    for i, d in enumerate(dates):
        records.append({
            "Market Name": "Burdwan",
            "Variety": "Common",
            "Reported Date": d.strftime("%Y-%m-%d"),
            "Modal Price (Rs./Quintal)": 2500.0 + (i % 4) * 10,
            "Arrivals (Tonnes)": 100.0 + i,
        })
    df_clean = pd.DataFrame(records)
    target_date = dates[30].strftime("%Y-%m-%d")

    # Baseline prediction at target_date
    res_base = pipeline.predict(df_clean, "Burdwan", "Common", target_date)

    # Poison future data (dates 31 to 34)
    df_poisoned = df_clean.copy()
    df_poisoned.loc[31:, "Modal Price (Rs./Quintal)"] = 88888.0
    df_poisoned.loc[31:, "Arrivals (Tonnes)"] = 99999.0

    res_poisoned = pipeline.predict(df_poisoned, "Burdwan", "Common", target_date)

    assert res_base["predicted_modal_price"] == res_poisoned["predicted_modal_price"]


# ============================================================
# 4. FROZEN PRODUCTION ARTIFACT INTEGRITY & REPRODUCIBILITY
# ============================================================

def test_frozen_production_artifacts_unmodified():
    """Verify that the production model artifact is intact and unchanged."""
    pipeline = RiceInferencePipeline()
    assert getattr(pipeline.price_model, "n_features_in_", None) == 48
    assert len(pipeline.features_list) == 48
    assert pipeline.config.get("commodity") == "rice"


def test_evaluation_reproduces_documented_metrics():
    """Verify that evaluation reproduces actual 1-day, 7-day, and overall holdout metrics."""
    summary_df, eval_df, _ = run_evaluation(
        holdout_start="2023-02-01",
        holdout_end="2024-02-01",
        tolerance_rs=25.0,
    )

    # 1. Total holdout rows must equal 20,465
    assert len(eval_df) == 20465

    # 2. Check 1-Day Horizon metrics
    h1 = summary_df[summary_df["Horizon"] == "1-Day"].iloc[0]
    assert h1["N"] == 17419
    assert np.isclose(h1["MAE (₹/qtl)"], 15.75, atol=0.05)
    assert np.isclose(h1["RMSE (₹/qtl)"], 69.14, atol=0.05)
    assert np.isclose(h1["R2"], 0.9869, atol=0.001)
    assert np.isclose(h1["MAPE (%)"], 0.46, atol=0.05)
    assert np.isclose(h1["100 - MAPE (%)"], 99.54, atol=0.05)
    assert np.isclose(h1["Within ₹25/qtl (%)"], 85.76, atol=0.05)

    # 3. Check 7-Day Horizon metrics
    h7 = summary_df[summary_df["Horizon"] == "7-Day"].iloc[0]
    assert h7["N"] == 56
    assert np.isclose(h7["MAE (₹/qtl)"], 46.05, atol=0.1)
    assert np.isclose(h7["RMSE (₹/qtl)"], 123.31, atol=0.1)
    assert np.isclose(h7["R2"], 0.9314, atol=0.001)
    assert np.isclose(h7["MAPE (%)"], 1.28, atol=0.05)
    assert np.isclose(h7["100 - MAPE (%)"], 98.72, atol=0.05)
    assert np.isclose(h7["Within ₹25/qtl (%)"], 64.29, atol=0.1)

    # 4. Check Overall Holdout metrics
    hov = summary_df[summary_df["Horizon"] == "Overall (Holdout)"].iloc[0]
    assert hov["N"] == 20465
    assert np.isclose(hov["MAE (₹/qtl)"], 18.98, atol=0.05)
    assert np.isclose(hov["RMSE (₹/qtl)"], 85.45, atol=0.05)
    assert np.isclose(hov["R2"], 0.9805, atol=0.001)
    assert np.isclose(hov["MAPE (%)"], 0.54, atol=0.05)
    assert np.isclose(hov["100 - MAPE (%)"], 99.46, atol=0.05)
    assert np.isclose(hov["Within ₹25/qtl (%)"], 84.09, atol=0.05)
