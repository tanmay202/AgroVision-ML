import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import (
    TRAIN_PATH,
    TEST_PATH,
    ERROR_ANALYSIS_PATH,
    PRICE_TARGET,
    PRICE_COLUMN,
    DATE_COLUMN,
    PRICE_FEATURES,
)


def main():
    # --------------------------------------------------
    # 1. Load
    # --------------------------------------------------
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    test_df[DATE_COLUMN] = pd.to_datetime(
        test_df[DATE_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------
    # 2. Keep metadata for analysis
    # --------------------------------------------------
    metadata = test_df[
        [
            "State Name",
            "District Name",
            "Market Name",
            "Variety",
            DATE_COLUMN,
            PRICE_COLUMN,
            PRICE_TARGET,
        ]
    ].copy()

    # --------------------------------------------------
    # 3. Keep ML data
    # --------------------------------------------------
    train_ml = train_df[
        PRICE_FEATURES + [PRICE_TARGET]
    ].dropna().copy()

    test_ml = test_df[
        PRICE_FEATURES + [PRICE_TARGET]
    ].dropna().copy()

    X_train = train_ml[PRICE_FEATURES]
    y_train = train_ml[PRICE_TARGET]

    X_test = test_ml[PRICE_FEATURES]
    y_test = test_ml[PRICE_TARGET]

    # --------------------------------------------------
    # 4. Train Random Forest
    # --------------------------------------------------
    rf_model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    rf_model.fit(
        X_train,
        y_train
    )

    rf_predictions = rf_model.predict(X_test)

    # --------------------------------------------------
    # 5. Train XGBoost
    # --------------------------------------------------
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    xgb_model.fit(
        X_train,
        y_train
    )

    xgb_predictions = xgb_model.predict(X_test)

    # --------------------------------------------------
    # 6. Align metadata to test rows
    # --------------------------------------------------
    metadata = metadata.loc[
        test_ml.index
    ].reset_index(drop=True)

    y_test = y_test.reset_index(drop=True)

    # --------------------------------------------------
    # 7. Build error table
    # --------------------------------------------------
    results = metadata.copy()

    results["random_forest_prediction"] = rf_predictions
    results["xgboost_prediction"] = xgb_predictions

    results["baseline_error"] = (
        results[PRICE_TARGET]
        - results[PRICE_COLUMN]
    )

    results["random_forest_error"] = (
        results[PRICE_TARGET]
        - results["random_forest_prediction"]
    )

    results["xgboost_error"] = (
        results[PRICE_TARGET]
        - results["xgboost_prediction"]
    )

    results["baseline_abs_error"] = (
        results["baseline_error"].abs()
    )

    results["random_forest_abs_error"] = (
        results["random_forest_error"].abs()
    )

    results["xgboost_abs_error"] = (
        results["xgboost_error"].abs()
    )

    # --------------------------------------------------
    # 8. Price movement
    # --------------------------------------------------
    results["actual_price_change"] = (
        results[PRICE_TARGET]
        - results[PRICE_COLUMN]
    )

    results["actual_price_change_pct"] = (
        results["actual_price_change"]
        / results[PRICE_COLUMN]
    ) * 100

    # --------------------------------------------------
    # 9. Largest errors
    # --------------------------------------------------
    print("=" * 60)
    print("LARGEST BASELINE ERRORS")
    print("=" * 60)

    print(
        results.nlargest(
            10,
            "baseline_abs_error"
        )[
            [
                "Market Name",
                "Variety",
                DATE_COLUMN,
                PRICE_COLUMN,
                PRICE_TARGET,
                "baseline_abs_error",
                "actual_price_change_pct",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("LARGEST RANDOM FOREST ERRORS")
    print("=" * 60)

    print(
        results.nlargest(
            10,
            "random_forest_abs_error"
        )[
            [
                "Market Name",
                "Variety",
                DATE_COLUMN,
                PRICE_COLUMN,
                PRICE_TARGET,
                "random_forest_prediction",
                "random_forest_abs_error",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 60)
    print("LARGEST XGBOOST ERRORS")
    print("=" * 60)

    print(
        results.nlargest(
            10,
            "xgboost_abs_error"
        )[
            [
                "Market Name",
                "Variety",
                DATE_COLUMN,
                PRICE_COLUMN,
                PRICE_TARGET,
                "xgboost_prediction",
                "xgboost_abs_error",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 10. Error statistics
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("ERROR STATISTICS")
    print("=" * 60)

    print(
        results[
            [
                "baseline_abs_error",
                "random_forest_abs_error",
                "xgboost_abs_error",
            ]
        ].describe()
    )

    # --------------------------------------------------
    # 11. Save
    # --------------------------------------------------
    results.to_csv(
        ERROR_ANALYSIS_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {ERROR_ANALYSIS_PATH}")
    print(f"Rows: {len(results)}")


if __name__ == "__main__":
    main()