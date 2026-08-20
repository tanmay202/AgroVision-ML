import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from config import (
    TEST_PATH,
    VOLATILITY_DATASET_TEST_PATH,
    VOLATILITY_CONFIG_PATH,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
)


def main():
    if not TEST_PATH.exists():
        print(f"ERROR: File not found: {TEST_PATH}")
        return

    if not VOLATILITY_CONFIG_PATH.exists():
        print(
            f"ERROR: Volatility config not found: "
            f"{VOLATILITY_CONFIG_PATH}\n"
            f"Run create_volatility_target.py first."
        )
        return

    # --------------------------------------------------
    # Load thresholds from saved config
    # --------------------------------------------------
    with open(
        VOLATILITY_CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        vol_config = json.load(file)

    low_threshold = vol_config["low_medium_threshold_percent"]
    high_threshold = vol_config["medium_high_threshold_percent"]

    print("=" * 60)
    print("VOLATILITY THRESHOLDS (from config)")
    print("=" * 60)
    print(f"Low/Medium : {low_threshold}%")
    print(f"Medium/High: {high_threshold}%")

    # --------------------------------------------------
    # Classify function
    # --------------------------------------------------
    def classify_volatility(value):
        if pd.isna(value):
            return pd.NA

        if value == 0:
            return "LOW"

        if value <= low_threshold:
            return "LOW"

        if value <= high_threshold:
            return "MEDIUM"

        return "HIGH"

    # --------------------------------------------------
    # Load test data
    # --------------------------------------------------
    df = pd.read_csv(TEST_PATH)

    # --------------------------------------------------
    # Calculate the movement that actually occurred.
    # This is used ONLY to create the evaluation label.
    # It is NOT included as a model feature.
    # --------------------------------------------------
    df["abs_price_pct_change"] = (
        df["price_pct_change"].abs()
    )

    df[VOLATILITY_TARGET] = (
        df["abs_price_pct_change"]
        .apply(classify_volatility)
    )

    print("\n" + "=" * 60)
    print("TEST VOLATILITY DATASET")
    print("=" * 60)

    print(f"Initial rows: {len(df)}")

    # Remove rows without target
    before = len(df)

    df = df.dropna(
        subset=[VOLATILITY_TARGET]
    ).copy()

    print(
        "Rows removed because target is missing:",
        before - len(df)
    )

    # Remove rows with unavailable historical features
    missing_rows = (
        df[VOLATILITY_FEATURES]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        "Rows with missing features:",
        missing_rows
    )

    df = df.dropna(
        subset=VOLATILITY_FEATURES
    ).copy()

    result = df[
        VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    ].copy()

    print("\n" + "=" * 60)
    print("FINAL TEST VOLATILITY DATASET")
    print("=" * 60)

    print(f"Rows: {len(result)}")
    print(f"Features: {len(VOLATILITY_FEATURES)}")

    print("\nClass distribution:")
    print(result[VOLATILITY_TARGET].value_counts())

    print("\nPercentages:")
    print(
        (
            result[VOLATILITY_TARGET]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    result.to_csv(
        VOLATILITY_DATASET_TEST_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {VOLATILITY_DATASET_TEST_PATH}")


if __name__ == "__main__":
    main()