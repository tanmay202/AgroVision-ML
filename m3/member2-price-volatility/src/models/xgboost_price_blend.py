"""
AgroVision — XGBoost Price Blend Experiment

Goal:
    Combine the current-price baseline with the XGBoost
    predicted price correction.

XGBoost target:
    future_price_change =
        future_modal_price - current_price

Blend:
    predicted_future_price =
        current_price + alpha * predicted_price_change

Alpha:
    alpha = 0.00  -> current-price baseline
    alpha = 1.00  -> full XGBoost correction

Alpha is selected using a chronological validation split
inside the training data.

The final test set is used only after alpha is selected.

IMPORTANT:
    This is an EXPERIMENT.
    Do not modify save_final_models.py yet.
"""

import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    PRICE_FEATURES,
    GROUP_COLUMNS,
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

    Example:
        "MAE", "mae", "Mae"

    are treated as the same metric.
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
# CREATE PRICE-DIFFERENCE TARGET
# ============================================================

def add_target(df):
    """
    Create:

        future_price_change =
            future_modal_price - current_price
    """

    df = df.copy()

    df["future_price_change"] = (
        df[PRICE_TARGET]
        - df[PRICE_COLUMN]
    )

    return df


# ============================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ============================================================

def validation_split(df, val_ratio=0.20):
    """
    Perform an 80/20 chronological split separately
    for every Market Name + Variety group.

    This prevents future observations from being used
    to predict earlier observations.
    """

    train_parts = []
    val_parts = []

    for _, group in df.groupby(
        GROUP_COLUMNS
    ):

        group = group.sort_values(
            DATE_COLUMN
        ).reset_index(
            drop=True
        )

        split_idx = int(
            len(group)
            * (1 - val_ratio)
        )

        # Very small groups
        if split_idx < 1:

            val_parts.append(group)
            continue

        # No validation rows possible
        if split_idx >= len(group):

            train_parts.append(group)
            continue

        # Earlier observations -> training
        train_parts.append(
            group.iloc[:split_idx]
        )

        # Later observations -> validation
        val_parts.append(
            group.iloc[split_idx:]
        )

    # --------------------------------------------------------
    # Combine groups
    # --------------------------------------------------------

    if train_parts:

        train_part = pd.concat(
            train_parts,
            ignore_index=True
        )

    else:

        train_part = pd.DataFrame(
            columns=df.columns
        )

    if val_parts:

        val_part = pd.concat(
            val_parts,
            ignore_index=True
        )

    else:

        val_part = pd.DataFrame(
            columns=df.columns
        )

    return train_part, val_part


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

def create_model():
    """
    Create the price-difference XGBoost model.
    """

    return XGBRegressor(
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("XGBOOST PRICE BLEND EXPERIMENT")
    print("=" * 60)

    # ========================================================
    # LOAD TRAINING AND TEST DATA
    # ========================================================

    train_file = train_path()
    test_file = test_path()

    if not train_file.exists():

        raise FileNotFoundError(
            f"Training file not found: {train_file}"
        )

    if not test_file.exists():

        raise FileNotFoundError(
            f"Test file not found: {test_file}"
        )

    train_df = pd.read_csv(
        train_file
    )

    test_df = pd.read_csv(
        test_file
    )

    # ========================================================
    # VALIDATE IMPORTANT COLUMNS
    # ========================================================

    required_columns = [
        DATE_COLUMN,
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
    # SELECT COMMON FEATURES
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
            "in training and test data."
        )

    missing_features = [
        feature
        for feature in PRICE_FEATURES
        if feature not in train_df.columns
        or feature not in test_df.columns
    ]

    if missing_features:

        print(
            "\nWARNING: Some PRICE_FEATURES are missing:"
        )

        for feature in missing_features:

            print(
                f"   - {feature}"
            )

    features = list(
        dict.fromkeys(features)
    )

    print(
        f"   Features: {len(features)}"
    )

    # ========================================================
    # CREATE PRICE-DIFFERENCE TARGET
    # ========================================================

    train_df = add_target(
        train_df
    )

    test_df = add_target(
        test_df
    )

    target_columns = [
        PRICE_COLUMN,
        PRICE_TARGET,
        "future_price_change",
    ]

    # ========================================================
    # REMOVE INVALID TARGET ROWS
    # ========================================================

    train_df = train_df.dropna(
        subset=target_columns
    ).copy()

    test_df = test_df.dropna(
        subset=target_columns
    ).copy()

    if len(train_df) == 0:

        raise ValueError(
            "No valid training rows remain."
        )

    if len(test_df) == 0:

        raise ValueError(
            "No valid test rows remain."
        )

    # ========================================================
    # CHRONOLOGICAL VALIDATION SPLIT
    # ========================================================

    subtrain_df, val_df = validation_split(
        train_df,
        val_ratio=0.20
    )

    # Remove rows with missing features

    subtrain_df = subtrain_df.dropna(
        subset=features
    ).copy()

    val_df = val_df.dropna(
        subset=features
    ).copy()

    print(
        f"   Subtrain rows: {len(subtrain_df)}"
    )

    print(
        f"   Validation rows: {len(val_df)}"
    )

    if len(subtrain_df) == 0:

        raise ValueError(
            "No valid subtraining rows remain."
        )

    if len(val_df) == 0:

        raise ValueError(
            "No valid validation rows remain."
        )

    # ========================================================
    # TRAIN TEMPORARY MODEL
    # ========================================================

    print("\n" + "=" * 60)
    print("TRAINING VALIDATION MODEL")
    print("=" * 60)

    validation_model = create_model()

    validation_model.fit(
        subtrain_df[features],
        subtrain_df["future_price_change"]
    )

    print(
        "   Training complete."
    )

    # ========================================================
    # VALIDATION PREDICTION
    # ========================================================

    validation_predicted_change = (
        validation_model.predict(
            val_df[features]
        )
    )

    validation_current_price = (
        val_df[PRICE_COLUMN]
        .to_numpy()
    )

    validation_actual_price = (
        val_df[PRICE_TARGET]
        .to_numpy()
    )

    # ========================================================
    # SEARCH BEST ALPHA
    # ========================================================

    print("\n" + "=" * 60)
    print("SEARCHING BEST BLEND WEIGHT")
    print("=" * 60)

    results = []

    # 0.00, 0.05, ..., 1.00
    alphas = np.arange(
        0.00,
        1.001,
        0.05
    )

    for alpha in alphas:

        # ----------------------------------------------------
        # Blend prediction
        # ----------------------------------------------------

        predicted_price = (
            validation_current_price
            + alpha
            * validation_predicted_change
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        metrics = calculate_metrics(
            validation_actual_price,
            predicted_price
        )

        results.append(
            {
                "alpha": round(
                    float(alpha),
                    2
                ),
                "MAE": get_metric(
                    metrics,
                    "MAE"
                ),
                "RMSE": get_metric(
                    metrics,
                    "RMSE"
                ),
                "R2": get_metric(
                    metrics,
                    "R2"
                ),
                "MAPE": get_metric(
                    metrics,
                    "MAPE"
                ),
            }
        )

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT BEST ALPHA USING VALIDATION MAE
    # ========================================================

    best_row = results_df.loc[
        results_df["MAE"].idxmin()
    ]

    best_alpha = float(
        best_row["alpha"]
    )

    best_validation_mae = float(
        best_row["MAE"]
    )

    print("\n" + "=" * 60)
    print("BEST ALPHA")
    print("=" * 60)

    print(
        f"   Alpha: {best_alpha:.2f}"
    )

    print(
        f"   Validation MAE: "
        f"{best_validation_mae:.2f}"
    )

    # ========================================================
    # TRAIN FINAL MODEL USING ALL TRAINING DATA
    # ========================================================

    train_df = train_df.dropna(
        subset=features
    ).copy()

    test_df = test_df.dropna(
        subset=features
    ).copy()

    if len(train_df) == 0:

        raise ValueError(
            "No valid training rows after "
            "feature filtering."
        )

    if len(test_df) == 0:

        raise ValueError(
            "No valid test rows after "
            "feature filtering."
        )

    print("\n" + "=" * 60)
    print("TRAINING FINAL XGBOOST")
    print("=" * 60)

    final_model = create_model()

    final_model.fit(
        train_df[features],
        train_df["future_price_change"]
    )

    print(
        "   Training complete."
    )

    # ========================================================
    # TEST PREDICTION
    # ========================================================

    predicted_change = (
        final_model.predict(
            test_df[features]
        )
    )

    current_prices = (
        test_df[PRICE_COLUMN]
        .to_numpy()
    )

    actual_prices = (
        test_df[PRICE_TARGET]
        .to_numpy()
    )

    # ========================================================
    # FULL XGBOOST CORRECTION
    # ========================================================

    xgb_predictions = (
        current_prices
        + predicted_change
    )

    xgb_metrics = calculate_metrics(
        actual_prices,
        xgb_predictions
    )

    print_metrics(
        xgb_metrics,
        "XGBoost Full Correction"
    )

    # ========================================================
    # BLENDED MODEL
    # ========================================================

    blended_predictions = (
        current_prices
        + best_alpha
        * predicted_change
    )

    blended_metrics = calculate_metrics(
        actual_prices,
        blended_predictions
    )

    print_metrics(
        blended_metrics,
        f"Blended Model (alpha={best_alpha:.2f})"
    )

    # ========================================================
    # CURRENT PRICE BASELINE
    # ========================================================

    baseline_metrics = calculate_metrics(
        actual_prices,
        current_prices
    )

    print_metrics(
        baseline_metrics,
        "Baseline (Current Price)"
    )

    # ========================================================
    # EXTRACT METRICS SAFELY
    # ========================================================

    baseline_mae = get_metric(
        baseline_metrics,
        "MAE"
    )

    xgb_mae = get_metric(
        xgb_metrics,
        "MAE"
    )

    blended_mae = get_metric(
        blended_metrics,
        "MAE"
    )

    baseline_rmse = get_metric(
        baseline_metrics,
        "RMSE"
    )

    blended_rmse = get_metric(
        blended_metrics,
        "RMSE"
    )

    baseline_r2 = get_metric(
        baseline_metrics,
        "R2"
    )

    blended_r2 = get_metric(
        blended_metrics,
        "R2"
    )

    baseline_mape = get_metric(
        baseline_metrics,
        "MAPE"
    )

    blended_mape = get_metric(
        blended_metrics,
        "MAPE"
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)

    print(
        f"Current Price MAE : "
        f"{baseline_mae:.2f}"
    )

    print(
        f"XGBoost MAE       : "
        f"{xgb_mae:.2f}"
    )

    print(
        f"Blended MAE       : "
        f"{blended_mae:.2f}"
    )

    print()

    print(
        f"Current Price RMSE: "
        f"{baseline_rmse:.2f}"
    )

    print(
        f"Blended RMSE     : "
        f"{blended_rmse:.2f}"
    )

    print()

    print(
        f"Current Price R²  : "
        f"{baseline_r2:.4f}"
    )

    print(
        f"Blended R²        : "
        f"{blended_r2:.4f}"
    )

    print()

    print(
        f"Current Price MAPE: "
        f"{baseline_mape:.2f}%"
    )

    print(
        f"Blended MAPE      : "
        f"{blended_mape:.2f}%"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)

    if blended_mae < baseline_mae:

        improvement = (
            (
                baseline_mae
                - blended_mae
            )
            / baseline_mae
        ) * 100.0

        print(
            "   Blend BEATS current-price baseline."
        )

        print(
            f"   MAE improvement: "
            f"{improvement:.2f}%"
        )

    elif blended_mae > baseline_mae:

        difference = (
            blended_mae
            - baseline_mae
        )

        print(
            "   Blend does NOT beat "
            "current-price baseline."
        )

        print(
            f"   MAE difference: "
            f"+{difference:.2f}"
        )

    else:

        print(
            "   Blend has EXACTLY the same "
            "MAE as current-price baseline."
        )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": (
                final_model.feature_importances_
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

    y_train = train_df[
        "future_price_change"
    ]

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

    print("\n" + "=" * 60)
    print("BLEND EXPERIMENT COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()