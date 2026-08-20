import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    # 1. Load train/test data
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
    # 2. Keep required columns
    # --------------------------------------------------
    required_columns = PRICE_FEATURES + [PRICE_TARGET]

    train_df = train_df[required_columns].copy()
    test_df = test_df[required_columns].copy()

    # --------------------------------------------------
    # 3. Handle missing feature rows
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
    # 4. Create X and y
    # --------------------------------------------------
    X_train = train_df[PRICE_FEATURES].copy()
    y_train = train_df[PRICE_TARGET].copy()

    X_test = test_df[PRICE_FEATURES].copy()
    y_test = test_df[PRICE_TARGET].copy()

    # Safety check: make sure train/test feature order
    # is exactly identical.
    print("\n" + "=" * 60)
    print("FEATURE ORDER CHECK")
    print("=" * 60)

    print("Feature count:", len(PRICE_FEATURES))
    print("Train feature count:", X_train.shape[1])
    print("Test feature count:", X_test.shape[1])

    if list(X_train.columns) == list(X_test.columns):
        print("PASS: Train/test feature order matches.")
    else:
        print("ERROR: Train/test feature order does not match.")
        return

    # --------------------------------------------------
    # 5. Train Random Forest
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------
    # 6. Generate predictions
    # --------------------------------------------------
    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 7. Random Forest metrics
    # --------------------------------------------------
    rf_mae, rf_rmse, rf_r2, rf_mape = calculate_metrics(
        y_test,
        predictions
    )

    # --------------------------------------------------
    # 8. Baseline using CURRENT PRICE
    # --------------------------------------------------
    baseline_predictions = test_df[PRICE_COLUMN]

    base_mae, base_rmse, base_r2, base_mape = calculate_metrics(
        y_test,
        baseline_predictions
    )

    # --------------------------------------------------
    # 9. Compare
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST SET COMPARISON")
    print("=" * 60)

    print("\nBaseline:")
    print(f"MAE  : {base_mae:.2f}")
    print(f"RMSE : {base_rmse:.2f}")
    print(f"R2   : {base_r2:.4f}")
    print(f"MAPE : {base_mape:.2f}%")

    print("\nRandom Forest:")
    print(f"MAE  : {rf_mae:.2f}")
    print(f"RMSE : {rf_rmse:.2f}")
    print(f"R2   : {rf_r2:.4f}")
    print(f"MAPE : {rf_mape:.2f}%")

    # --------------------------------------------------
    # 10. Improvement
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE IMPROVEMENT")
    print("=" * 60)

    if base_mae != 0:
        mae_improvement = (
            (base_mae - rf_mae)
            / base_mae
        ) * 100
    else:
        mae_improvement = np.nan

    if base_mape != 0:
        mape_improvement = (
            (base_mape - rf_mape)
            / base_mape
        ) * 100
    else:
        mape_improvement = np.nan

    print(
        f"MAE improvement : {mae_improvement:.2f}%"
    )

    print(
        f"MAPE improvement: {mape_improvement:.2f}%"
    )

    # --------------------------------------------------
    # 11. Feature importance
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