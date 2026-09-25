"""
AgroVision — XGBoost Price Model

Trains an XGBoost regressor for price forecasting.
Uses shared metrics from utils.

LEAKAGE FIX (v2.0):
    - Uses corrected PRICE_FEATURES (no current-row prices)
    - XGBoost handles NaN natively, so we don't drop rows
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from config import (
    TRAIN_PATH,
    TEST_PATH,
    PRICE_TARGET,
    PRICE_FEATURES,
    train_path,
    test_path,
)
from utils import calculate_metrics, print_metrics


def train_xgboost_price(train_df=None, test_df=None):
    """
    Train an XGBoost price model and evaluate on test set.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame, optional
        If None, reads from CSV files.

    Returns
    -------
    tuple of (model, metrics_dict)
    """
    if train_df is None:
        t_path = train_path()
        te_path = test_path()
        if not t_path.exists():
            raise FileNotFoundError(f"Missing {t_path}")
        if not te_path.exists():
            raise FileNotFoundError(f"Missing {te_path}")
        train_df = pd.read_csv(t_path)
        test_df = pd.read_csv(te_path)

    print("\n" + "=" * 60)
    print("XGBOOST PRICE MODEL")
    print("=" * 60)

    # Select required columns
    available_features = [f for f in PRICE_FEATURES if f in train_df.columns]
    missing = set(PRICE_FEATURES) - set(available_features)
    if missing:
        print(f"   WARNING: Missing features (will skip): {missing}")

    features = available_features
    required = features + [PRICE_TARGET]

    train_subset = train_df[[c for c in required if c in train_df.columns]].copy()
    test_subset = test_df[[c for c in required if c in test_df.columns]].copy()

    # Drop rows where the TARGET is missing (required for evaluation)
    train_subset = train_subset.dropna(subset=[PRICE_TARGET])
    test_subset = test_subset.dropna(subset=[PRICE_TARGET])

    # XGBoost handles NaN in features natively — no need to drop feature NaNs
    print(f"   Train rows: {len(train_subset)}")
    print(f"   Test rows : {len(test_subset)}")
    print(f"   Features  : {len(features)}")

    X_train = train_subset[features]
    y_train = train_subset[PRICE_TARGET]
    X_test = test_subset[features]
    y_test = test_subset[PRICE_TARGET]

    # Train
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    print("   Training complete.")

    # Evaluate
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    print_metrics(metrics, "XGBoost Price")

    # Baseline comparison (use lag_1 as naive baseline)
    if "lag_1" in test_subset.columns:
        baseline_mask = test_subset["lag_1"].notna()
        if baseline_mask.sum() > 0:
            baseline_metrics = calculate_metrics(
                y_test[baseline_mask],
                test_subset.loc[baseline_mask, "lag_1"]
            )
            print_metrics(baseline_metrics, "Baseline (lag_1 = previous price)")

    # Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    print(importance.head(10).to_string(index=False))

    return model, metrics


def main():
    train_xgboost_price()


if __name__ == "__main__":
    main()