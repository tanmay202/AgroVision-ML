import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    TRAIN_PATH,
    TEST_PATH,
    VOLATILITY_DATASET_TRAIN_PATH,
    VOLATILITY_DATASET_TEST_PATH,
    DATE_COLUMN,
    PRICE_TARGET,
    VOLATILITY_TARGET,
    PRICE_FEATURES,
    VOLATILITY_FEATURES,
)


def main():
    print("=" * 60)
    print("FINAL TEMPORAL VALIDATION")
    print("=" * 60)

    results = []

    # --------------------------------------------------
    # 1. Price temporal validation
    # --------------------------------------------------
    train_price = pd.read_csv(TRAIN_PATH)
    test_price = pd.read_csv(TEST_PATH)

    train_price[DATE_COLUMN] = pd.to_datetime(
        train_price[DATE_COLUMN]
    )

    test_price[DATE_COLUMN] = pd.to_datetime(
        test_price[DATE_COLUMN]
    )

    train_end = train_price[DATE_COLUMN].max()
    test_start = test_price[DATE_COLUMN].min()

    print("\nPRICE MODEL")

    print(f"Training end : {train_end}")
    print(f"Testing start: {test_start}")

    if train_end < test_start:
        print("PASS: Price train/test chronology is valid.")
        results.append(True)
    else:
        print("FAIL: Price temporal overlap detected.")
        results.append(False)

    if PRICE_TARGET in PRICE_FEATURES:
        print("FAIL: Price target appears in features.")
        results.append(False)
    else:
        print("PASS: Price target is not a feature.")
        results.append(True)

    # --------------------------------------------------
    # 2. Volatility leakage validation
    # --------------------------------------------------
    print("\nVOLATILITY MODEL")

    if VOLATILITY_TARGET in VOLATILITY_FEATURES:
        print("FAIL: Volatility target appears in features.")
        results.append(False)
    else:
        print("PASS: Volatility target is not a feature.")
        results.append(True)

    if "price_pct_change" in VOLATILITY_FEATURES:
        print("FAIL: price_pct_change appears in volatility features.")
        results.append(False)
    else:
        print("PASS: price_pct_change is not used as a volatility feature.")
        results.append(True)

    # --------------------------------------------------
    # 3. Check exact expected columns
    # --------------------------------------------------
    train_vol = pd.read_csv(VOLATILITY_DATASET_TRAIN_PATH)
    test_vol = pd.read_csv(VOLATILITY_DATASET_TEST_PATH)

    expected_columns = VOLATILITY_FEATURES + [VOLATILITY_TARGET]

    train_columns_match = (
        list(train_vol.columns) == expected_columns
    )

    test_columns_match = (
        list(test_vol.columns) == expected_columns
    )

    if train_columns_match:
        print("PASS: Training volatility columns match expected schema.")
        results.append(True)
    else:
        print("FAIL: Training volatility columns do not match expected schema.")
        print("\nActual:", train_vol.columns.tolist())
        print("Expected:", expected_columns)
        results.append(False)

    if test_columns_match:
        print("PASS: Testing volatility columns match expected schema.")
        results.append(True)
    else:
        print("FAIL: Testing volatility columns do not match expected schema.")
        print("\nActual:", test_vol.columns.tolist())
        print("Expected:", expected_columns)
        results.append(False)

    # --------------------------------------------------
    # 4. Required feature presence
    # --------------------------------------------------
    missing_train_features = [
        f for f in VOLATILITY_FEATURES
        if f not in train_vol.columns
    ]

    missing_test_features = [
        f for f in VOLATILITY_FEATURES
        if f not in test_vol.columns
    ]

    print("\nREQUIRED FEATURE CHECK")

    if not missing_train_features:
        print("PASS: All required training features exist.")
        results.append(True)
    else:
        print("FAIL: Missing training features:", missing_train_features)
        results.append(False)

    if not missing_test_features:
        print("PASS: All required testing features exist.")
        results.append(True)
    else:
        print("FAIL: Missing testing features:", missing_test_features)
        results.append(False)

    # --------------------------------------------------
    # 5. Missing values
    # --------------------------------------------------
    print("\nMISSING VALUE CHECK")

    price_train_missing = (
        train_price[PRICE_FEATURES]
        .isna().sum().sum()
    )

    price_test_missing = (
        test_price[PRICE_FEATURES]
        .isna().sum().sum()
    )

    print(
        "Price train feature missing values:",
        price_train_missing
    )

    print(
        "Price test feature missing values:",
        price_test_missing
    )

    print(
        "\nNote: These are expected historical-feature "
        "gaps before enough lag/rolling observations exist."
    )

    # --------------------------------------------------
    # 6. Dynamic summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")

    if all(results):
        print("\nOVERALL: ALL CHECKS PASSED")
    else:
        print("\nOVERALL: SOME CHECKS FAILED — review output above.")


if __name__ == "__main__":
    main()