"""Leakage-safe Rice price model evaluated on the original price scale."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from xgboost import XGBRegressor

from config import PRICE_COLUMN, PRICE_FEATURES, PRICE_TARGET, train_path, test_path
from utils import calculate_metrics, print_metrics


def _require_columns(df, columns, label):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def make_price_model():
    """Shared specification for evaluated and saved price models."""
    return XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        objective="reg:pseudohubererror", random_state=42, n_jobs=-1,
        tree_method="hist",
    )


def train_xgboost_price(train_df=None, test_df=None):
    """Train on unmodified future-price changes and score reconstructed prices.

    Current modal price is deliberately not a feature. It is used solely for
    reconstruction and for the persistence baseline.
    """
    if train_df is None:
        train_df = pd.read_csv(train_path())
    if test_df is None:
        test_df = pd.read_csv(test_path())

    required = PRICE_FEATURES + [PRICE_TARGET, PRICE_COLUMN]
    _require_columns(train_df, required, "Training data")
    _require_columns(test_df, required, "Test data")
    train = train_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    test = test_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    if train.empty or test.empty:
        raise ValueError("Price train/test data is empty after target validation.")

    # No clipping: all evaluation and fitting stays on the observed scale.
    y_train = train[PRICE_TARGET] - train[PRICE_COLUMN]
    model = make_price_model()
    model.fit(train[PRICE_FEATURES], y_train)

    predicted_change = model.predict(test[PRICE_FEATURES])
    predicted_price = test[PRICE_COLUMN].to_numpy() + predicted_change
    actual_price = test[PRICE_TARGET].to_numpy()
    persistence = test[PRICE_COLUMN].to_numpy()

    metrics = calculate_metrics(actual_price, predicted_price)
    metrics["persistence"] = calculate_metrics(actual_price, persistence)
    lag_mask = test["lag_1"].notna().to_numpy()
    if lag_mask.any():
        metrics["lag_1"] = calculate_metrics(
            actual_price[lag_mask], test.loc[lag_mask, "lag_1"].to_numpy()
        )

    movement_mask = actual_price != persistence
    metrics["movement_rows"] = int(movement_mask.sum())
    if movement_mask.any():
        metrics["movement"] = calculate_metrics(
            actual_price[movement_mask], predicted_price[movement_mask]
        )
        metrics["movement_persistence"] = calculate_metrics(
            actual_price[movement_mask], persistence[movement_mask]
        )

    importance = pd.DataFrame({"feature": PRICE_FEATURES,
                               "importance": model.feature_importances_})
    importance = importance.sort_values("importance", ascending=False)
    metrics["feature_importance"] = importance.to_dict("records")

    print("\n" + "=" * 60)
    print("XGBOOST PRICE-CHANGE MODEL (unclipped target)")
    print("=" * 60)
    print(f"   Features: {len(PRICE_FEATURES)}")
    print(f"   Train rows: {len(train)}; test rows: {len(test)}")
    print(f"   Target change distribution: min={y_train.min():.2f}, "
          f"p01={y_train.quantile(.01):.2f}, median={y_train.median():.2f}, "
          f"p99={y_train.quantile(.99):.2f}, max={y_train.max():.2f}")
    print_metrics(metrics, "XGBoost reconstructed future price")
    print_metrics(metrics["persistence"], "Current-price persistence baseline")
    if "lag_1" in metrics:
        print_metrics(metrics["lag_1"], "lag_1 baseline")
    print(f"\nMovement rows: {metrics['movement_rows']} / {len(test)}")
    if "movement" in metrics:
        print_metrics(metrics["movement"], "XGBoost movement rows")
        print_metrics(metrics["movement_persistence"], "Persistence movement rows")
    print("\nTop feature importance:")
    print(importance.head(10).to_string(index=False))
    return model, metrics


if __name__ == "__main__":
    train_xgboost_price()
