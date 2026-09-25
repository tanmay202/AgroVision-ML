"""
AgroVision — Random Forest Price Model

Trains a Random Forest regressor for price forecasting.
Uses shared metrics from utils.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from config import (
    TRAIN_PATH,
    TEST_PATH,
    PRICE_TARGET,
    PRICE_FEATURES,
    train_path,
    test_path,
)
from utils import calculate_metrics, print_metrics


def train_rf_price(train_df=None, test_df=None):
    """
    Train a Random Forest price model and evaluate.

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
    print("RANDOM FOREST PRICE MODEL")
    print("=" * 60)

    available_features = [f for f in PRICE_FEATURES if f in train_df.columns]
    required = available_features + [PRICE_TARGET]

    # Drop rows with any NaN (RF doesn't handle NaN natively)
    train_subset = train_df[[c for c in required if c in train_df.columns]].dropna().copy()
    test_subset = test_df[[c for c in required if c in test_df.columns]].dropna().copy()

    print(f"   Train rows: {len(train_subset)}")
    print(f"   Test rows : {len(test_subset)}")

    X_train = train_subset[available_features]
    y_train = train_subset[PRICE_TARGET]
    X_test = test_subset[available_features]
    y_test = test_subset[PRICE_TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(X_train, y_train)
    print("   Training complete.")

    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    print_metrics(metrics, "Random Forest Price")

    # Feature importance
    importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)
    print(importance.head(10).to_string(index=False))

    return model, metrics


def main():
    train_rf_price()


if __name__ == "__main__":
    main()