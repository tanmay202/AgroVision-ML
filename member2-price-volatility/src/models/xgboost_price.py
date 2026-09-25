"""
AgroVision — XGBoost Price Difference Model

Experiment:
    Predict the future absolute price difference instead of directly
    predicting the future absolute price.

Target:
    future_price_change = future_modal_price - current_price

Reconstruction:
    predicted_future_price =
        current_price + predicted_price_change

Evaluation:
    - MAE
    - RMSE
    - R²
    - MAPE

Baselines:
    1. Current-price persistence
    2. lag_1 previous-price persistence

Important:
    This file is an EXPERIMENT.
    Do not change save_final_models.py until this experiment is evaluated.
"""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import pandas as pd
from xgboost import XGBRegressor

from config import (
    PRICE_COLUMN,
    PRICE_TARGET,
    PRICE_FEATURES,
    train_path,
    test_path,
)

from utils import (
    calculate_metrics,
    print_metrics,
)


# ============================================================
# SAFE METRIC ACCESS
# ============================================================

def get_metric(metrics, name):
    """
    Safely retrieve a metric regardless of capitalization.
    """

    target = str(name).strip().lower()

    for key, value in metrics.items():

        if str(key).strip().lower() == target:

            try:
                return float(value)

            except (TypeError, ValueError):
                raise ValueError(
                    f"Metric '{name}' is not numeric: {value}"
                )

    raise KeyError(
        f"Metric '{name}' not found. "
        f"Available metrics: {list(metrics.keys())}"
    )


# ============================================================
# PERCENTAGE IMPROVEMENT
# ============================================================

def calculate_improvement(
    baseline_value,
    model_value
):
    """
    Calculate percentage improvement.

    Positive = model is better.
    Negative = model is worse.
    """

    if baseline_value == 0:
        return 0.0

    return (
        (baseline_value - model_value)
        / baseline_value
    ) * 100.0


# ============================================================
# PRINT COMPARISON
# ============================================================

def print_comparison(
    model_metrics,
    baseline_metrics,
    baseline_name
):
    """
    Compare XGBoost against a baseline.
    """

    model_mae = get_metric(
        model_metrics,
        "MAE"
    )

    model_rmse = get_metric(
        model_metrics,
        "RMSE"
    )

    model_mape = get_metric(
        model_metrics,
        "MAPE"
    )

    baseline_mae = get_metric(
        baseline_metrics,
        "MAE"
    )

    baseline_rmse = get_metric(
        baseline_metrics,
        "RMSE"
    )

    baseline_mape = get_metric(
        baseline_metrics,
        "MAPE"
    )

    mae_improvement = calculate_improvement(
        baseline_mae,
        model_mae
    )

    rmse_improvement = calculate_improvement(
        baseline_rmse,
        model_rmse
    )

    mape_improvement = calculate_improvement(
        baseline_mape,
        model_mape
    )

    print("\n" + "=" * 60)
    print(f"XGBOOST vs {baseline_name}")
    print("=" * 60)

    print(
        f"   MAE improvement : "
        f"{mae_improvement:+.2f}%"
    )

    print(
        f"   RMSE improvement: "
        f"{rmse_improvement:+.2f}%"
    )

    print(
        f"   MAPE improvement: "
        f"{mape_improvement:+.2f}%"
    )

    if mae_improvement > 0:

        print(
            "   RESULT: XGBoost improves "
            "over this baseline on MAE."
        )

    elif mae_improvement < 0:

        print(
            "   RESULT: XGBoost performs "
            "worse than this baseline on MAE."
        )

    else:

        print(
            "   RESULT: XGBoost and the baseline "
            "have identical MAE."
        )


# ============================================================
# TRAIN PRICE-DIFFERENCE MODEL
# ============================================================

def train_xgboost_price(
    train_df=None,
    test_df=None
):
    """
    Train XGBoost to predict future price difference.

    Returns:
        model, reconstructed-price metrics
    """

    # ========================================================
    # LOAD DATA
    # ========================================================

    if train_df is None:

        train_file = train_path()

        if not train_file.exists():

            raise FileNotFoundError(
                f"Training file not found: {train_file}"
            )

        train_df = pd.read_csv(
            train_file
        )

    if test_df is None:

        test_file = test_path()

        if not test_file.exists():

            raise FileNotFoundError(
                f"Test file not found: {test_file}"
            )

        test_df = pd.read_csv(
            test_file
        )

    print("\n" + "=" * 60)
    print("XGBOOST PRICE-DIFFERENCE MODEL")
    print("=" * 60)

    # ========================================================
    # VALIDATE REQUIRED COLUMNS
    # ========================================================

    required_columns = [
        PRICE_COLUMN,
        PRICE_TARGET,
    ]

    for column in required_columns:

        if column not in train_df.columns:

            raise ValueError(
                f"Training data missing required column: "
                f"{column}"
            )

        if column not in test_df.columns:

            raise ValueError(
                f"Test data missing required column: "
                f"{column}"
            )

    # ========================================================
    # SELECT FEATURES
    # ========================================================

    features = [
        feature
        for feature in PRICE_FEATURES
        if feature in train_df.columns
        and feature in test_df.columns
    ]

    if not features:

        raise ValueError(
            "No common PRICE_FEATURES found "
            "in train and test data."
        )

    missing_features = [
        feature
        for feature in PRICE_FEATURES
        if feature not in train_df.columns
        or feature not in test_df.columns
    ]

    if missing_features:

        print(
            "   WARNING: Missing features:"
        )

        for feature in missing_features:
            print(f"      - {feature}")

    features = list(
        dict.fromkeys(features)
    )

    print(
        f"   Features: {len(features)}"
    )

    # ========================================================
    # PREPARE DATA
    # ========================================================

    train_columns = list(
        dict.fromkeys(
            features
            + [
                PRICE_TARGET,
                PRICE_COLUMN,
            ]
        )
    )

    test_columns = list(
        dict.fromkeys(
            features
            + [
                PRICE_TARGET,
                PRICE_COLUMN,
                "lag_1",
            ]
        )
    )

    train_columns = [
        column
        for column in train_columns
        if column in train_df.columns
    ]

    test_columns = [
        column
        for column in test_columns
        if column in test_df.columns
    ]

    train_subset = train_df[
        train_columns
    ].copy()

    test_subset = test_df[
        test_columns
    ].copy()

    # ========================================================
    # REMOVE INVALID TARGET ROWS
    # ========================================================

    train_before = len(
        train_subset
    )

    train_subset = train_subset.dropna(
        subset=[
            PRICE_TARGET,
            PRICE_COLUMN,
        ]
    ).copy()

    test_before = len(
        test_subset
    )

    test_subset = test_subset.dropna(
        subset=[
            PRICE_TARGET,
            PRICE_COLUMN,
        ]
    ).copy()

    print(
        f"   Train rows: {len(train_subset)}"
    )

    print(
        f"   Test rows : {len(test_subset)}"
    )

    print(
        f"   Dropped training rows: "
        f"{train_before - len(train_subset)}"
    )

    print(
        f"   Dropped test rows: "
        f"{test_before - len(test_subset)}"
    )

    # ========================================================
    # CREATE PRICE-DIFFERENCE TARGET
    # ========================================================

    train_subset["future_price_change"] = (
        train_subset[PRICE_TARGET]
        - train_subset[PRICE_COLUMN]
    )

    test_subset["future_price_change"] = (
        test_subset[PRICE_TARGET]
        - test_subset[PRICE_COLUMN]
    )

    # ========================================================
    # TRAINING DATA
    # ========================================================

    X_train = train_subset[
        features
    ]

    y_train = train_subset[
        "future_price_change"
    ]

    # ========================================================
    # TEST DATA
    # ========================================================

    X_test = test_subset[
        features
    ]

    y_test_actual_price = test_subset[
        PRICE_TARGET
    ]

    current_prices = test_subset[
        PRICE_COLUMN
    ]

    # ========================================================
    # TRAIN XGBOOST
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST ON FUTURE PRICE DIFFERENCE")
    print("=" * 60)

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:pseudohubererror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "   Training complete."
    )

    # ========================================================
    # PREDICT PRICE DIFFERENCE
    # ========================================================

    predicted_price_change = model.predict(
        X_test
    )

    # ========================================================
    # RECONSTRUCT FUTURE PRICE
    # ========================================================

    predicted_future_price = (
        current_prices.values
        + predicted_price_change
    )

    # ========================================================
    # SANITY CHECK
    # ========================================================

    if pd.Series(
        predicted_future_price
    ).isna().any():

        raise ValueError(
            "NaN detected in reconstructed "
            "price predictions."
        )

    # ========================================================
    # MODEL EVALUATION
    # ========================================================

    model_metrics = calculate_metrics(
        y_test_actual_price,
        predicted_future_price
    )

    print_metrics(
        model_metrics,
        "XGBoost Price-Difference → Price"
    )

    # ========================================================
    # CURRENT PRICE BASELINE
    # ========================================================

    current_baseline_metrics = calculate_metrics(
        y_test_actual_price,
        current_prices
    )

    print_metrics(
        current_baseline_metrics,
        "Baseline (Current Price)"
    )

    print_comparison(
        model_metrics,
        current_baseline_metrics,
        "CURRENT PRICE BASELINE"
    )

    # ========================================================
    # lag_1 BASELINE
    # ========================================================

    if "lag_1" in test_subset.columns:

        lag_mask = (
            test_subset[
                "lag_1"
            ].notna()
        )

        if lag_mask.sum() > 0:

            lag_actual = (
                y_test_actual_price
                .loc[lag_mask]
            )

            lag_predictions = (
                test_subset.loc[
                    lag_mask,
                    "lag_1",
                ]
            )

            lag_metrics = calculate_metrics(
                lag_actual,
                lag_predictions
            )

            print_metrics(
                lag_metrics,
                "Baseline (lag_1)"
            )

            # ------------------------------------------------
            # Compare XGBoost on SAME rows
            # ------------------------------------------------

            xgb_on_lag_rows = (
                pd.Series(
                    predicted_future_price,
                    index=test_subset.index
                )
                .loc[lag_mask]
            )

            xgb_lag_metrics = calculate_metrics(
                lag_actual,
                xgb_on_lag_rows
            )

            print_metrics(
                xgb_lag_metrics,
                "XGBoost on lag_1 common rows"
            )

            print_comparison(
                xgb_lag_metrics,
                lag_metrics,
                "lag_1 BASELINE"
            )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                model.feature_importances_
            ),
        }
    ).sort_values(
        by="importance",
        ascending=False
    )

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    # ========================================================
    # TARGET STATISTICS
    # ========================================================

    print("\n" + "=" * 60)
    print("PRICE-DIFFERENCE TARGET STATISTICS")
    print("=" * 60)

    print(
        f"   Mean change   : "
        f"₹{y_train.mean():.2f}"
    )

    print(
        f"   Median change : "
        f"₹{y_train.median():.2f}"
    )

    print(
        f"   Std change    : "
        f"₹{y_train.std():.2f}"
    )

    print(
        f"   Min change    : "
        f"₹{y_train.min():.2f}"
    )

    print(
        f"   Max change    : "
        f"₹{y_train.max():.2f}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("PRICE-DIFFERENCE MODEL SUMMARY")
    print("=" * 60)

    print(
        f"   MAE  : "
        f"{get_metric(model_metrics, 'MAE'):.2f}"
    )

    print(
        f"   RMSE : "
        f"{get_metric(model_metrics, 'RMSE'):.2f}"
    )

    print(
        f"   R²   : "
        f"{get_metric(model_metrics, 'R2'):.4f}"
    )

    print(
        f"   MAPE : "
        f"{get_metric(model_metrics, 'MAPE'):.2f}%"
    )

    return model, model_metrics


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    train_xgboost_price()


if __name__ == "__main__":
    main()