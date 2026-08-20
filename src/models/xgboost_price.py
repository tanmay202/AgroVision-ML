import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    TRAIN_PATH,
    TEST_PATH,
    PRICE_TARGET,
    PRICE_COLUMN,
    PRICE_FEATURES,
)


def calculate_mape(actual, predicted):
    """Calculate MAPE while ignoring zero actual values."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    mape = calculate_mape(
        actual,
        predicted
    )

    return mae, rmse, r2, mape


def main():
    # --------------------------------------------------
    # 1. Load data
    # --------------------------------------------------
    if not TRAIN_PATH.exists():
        print(f"ERROR: Missing {TRAIN_PATH}")
        return

    if not TEST_PATH.exists():
        print(f"ERROR: Missing {TEST_PATH}")
        return

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print("=" * 60)
    print("INITIAL DATA")
    print("=" * 60)

    print(f"Train rows: {len(train_df)}")
    print(f"Test rows : {len(test_df)}")

    # --------------------------------------------------
    # 2. Select columns
    # --------------------------------------------------
    required_columns = PRICE_FEATURES + [PRICE_TARGET]

    train_df = train_df[required_columns].copy()
    test_df = test_df[required_columns].copy()

    # --------------------------------------------------
    # 3. Remove rows with unavailable features
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)

    print(
        "Training missing rows before removal:",
        train_df.isna().any(axis=1).sum()
    )

    print(
        "Testing missing rows before removal:",
        test_df.isna().any(axis=1).sum()
    )

    train_df = train_df.dropna(
        subset=PRICE_FEATURES + [PRICE_TARGET]
    ).copy()

    test_df = test_df.dropna(
        subset=PRICE_FEATURES + [PRICE_TARGET]
    ).copy()

    print(
        "\nTraining rows after removal:",
        len(train_df)
    )

    print(
        "Testing rows after removal:",
        len(test_df)
    )

    # --------------------------------------------------
    # 4. X / y
    # --------------------------------------------------
    X_train = train_df[PRICE_FEATURES].copy()
    y_train = train_df[PRICE_TARGET].copy()

    X_test = test_df[PRICE_FEATURES].copy()
    y_test = test_df[PRICE_TARGET].copy()

    # --------------------------------------------------
    # 5. Feature order check
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ORDER CHECK")
    print("=" * 60)

    if list(X_train.columns) == list(X_test.columns):
        print("PASS: Train/test feature order matches.")
    else:
        print("ERROR: Train/test feature order does not match.")
        return

    # --------------------------------------------------
    # 6. Train XGBoost
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

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

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------
    # 7. Predict
    # --------------------------------------------------
    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 8. XGBoost metrics
    # --------------------------------------------------
    xgb_mae, xgb_rmse, xgb_r2, xgb_mape = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # 9. Baseline
    # --------------------------------------------------
    baseline_predictions = test_df[PRICE_COLUMN]

    base_mae, base_rmse, base_r2, base_mape = calculate_metrics(
        y_test,
        baseline_predictions
    )

    # --------------------------------------------------
    # 10. Compare
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET COMPARISON")
    print("=" * 60)

    print("\nBaseline:")
    print(f"MAE  : {base_mae:.2f}")
    print(f"RMSE : {base_rmse:.2f}")
    print(f"R2   : {base_r2:.4f}")
    print(f"MAPE : {base_mape:.2f}%")

    print("\nXGBoost:")
    print(f"MAE  : {xgb_mae:.2f}")
    print(f"RMSE : {xgb_rmse:.2f}")
    print(f"R2   : {xgb_r2:.4f}")
    print(f"MAPE : {xgb_mape:.2f}%")

    # --------------------------------------------------
    # 11. Improvement over baseline
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE IMPROVEMENT")
    print("=" * 60)

    mae_improvement = (
        (base_mae - xgb_mae)
        / base_mae
    ) * 100 if base_mae != 0 else np.nan

    mape_improvement = (
        (base_mape - xgb_mape)
        / base_mape
    ) * 100 if base_mape != 0 else np.nan

    print(
        f"MAE improvement : {mae_improvement:.2f}%"
    )

    print(
        f"MAPE improvement: {mape_improvement:.2f}%"
    )

    # --------------------------------------------------
    # 12. Feature importance
    # --------------------------------------------------
    importance = pd.DataFrame({
        "feature": PRICE_FEATURES,
        "importance": model.feature_importances_
    }).sort_values(
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


if __name__ == "__main__":
    main()