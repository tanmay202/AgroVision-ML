"""
AgroVision — Forward Volatility Dataset Builder (Test)

Creates the volatility test dataset using thresholds computed from the
training set.

Target:
    Future volatility = price range (max - min) of the NEXT 5 prices
    within each Market + Variety group.

The date column is retained as metadata for chronological validation.
It is NOT a model feature.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    test_path,
    volatility_dataset_test_path,
    volatility_config_path,
)
from volatility_utils import (
    VOLATILITY_WINDOW,
    add_future_volatility,
    classify_volatility,
)


def build_volatility_test(df=None):
    """
    Build the final volatility test dataset.

    Columns:
        [DATE_COLUMN] + VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    """

    # ------------------------------------------------------------
    # Validate configuration
    # ------------------------------------------------------------
    if DATE_COLUMN in VOLATILITY_FEATURES:
        raise ValueError(
            f"'{DATE_COLUMN}' must not be listed in VOLATILITY_FEATURES."
        )

    # ------------------------------------------------------------
    # Load training thresholds
    # ------------------------------------------------------------
    config_path = volatility_config_path()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Volatility config not found: {config_path}\n"
            "Run create_volatility_target.py first."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        vol_config = json.load(f)

    low_threshold = vol_config["low_medium_threshold"]
    high_threshold = vol_config["medium_high_threshold"]

    # ------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------
    if df is None:
        path = test_path()

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        df = pd.read_csv(path)

    df = df.copy()

    print("\n" + "=" * 60)
    print("STEP 7: FORWARD VOLATILITY DATASET (TEST)")
    print("=" * 60)

    print(
        f"   Thresholds: "
        f"Low/Med={low_threshold:.6f}, "
        f"Med/High={high_threshold:.6f}"
    )

    # ------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------
    required = (
        [DATE_COLUMN, PRICE_COLUMN]
        + GROUP_COLUMNS
        + VOLATILITY_FEATURES
    )

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns in test data: {missing}"
        )

    # ------------------------------------------------------------
    # FUTURE VOLATILITY TARGET
    #
    # Uses the same price-range definition as training:
    #   future_volatility = max(price[t+1..t+W]) - min(price[t+1..t+W])
    # ------------------------------------------------------------
    df = add_future_volatility(df, GROUP_COLUMNS, DATE_COLUMN, PRICE_COLUMN)

    print(
        f"   Future volatility: price range of "
        f"next {VOLATILITY_WINDOW} observations"
    )

    # ------------------------------------------------------------
    # Classify using TRAINING thresholds
    # ------------------------------------------------------------
    df[VOLATILITY_TARGET] = classify_volatility(
        df["future_volatility"], low_threshold, high_threshold,
        VOLATILITY_CLASSES,
    )

    # ------------------------------------------------------------
    # Remove rows without future target
    # ------------------------------------------------------------
    before = len(df)

    df = df.dropna(
        subset=[VOLATILITY_TARGET]
    ).copy()

    print(
        f"   Dropped {before - len(df)} rows "
        "without future volatility target"
    )

    # ------------------------------------------------------------
    # Keep feature NaNs for XGBoost
    # ------------------------------------------------------------
    df[VOLATILITY_FEATURES] = (
        df[VOLATILITY_FEATURES]
        .replace([float("inf"), float("-inf")], pd.NA)
    )

    # ------------------------------------------------------------
    # Final dataset
    # ------------------------------------------------------------
    result = df[
        [DATE_COLUMN]
        + VOLATILITY_FEATURES
        + [VOLATILITY_TARGET]
    ].copy()

    print(
        f"\n   Final rows: {len(result)}"
    )

    # Parse date for range display
    result[DATE_COLUMN] = pd.to_datetime(
        result[DATE_COLUMN], errors="coerce"
    )

    if result[DATE_COLUMN].notna().any():
        print(
            f"   Date range: "
            f"{result[DATE_COLUMN].min().date()} -> "
            f"{result[DATE_COLUMN].max().date()}"
        )

    print("\n   Class distribution:")
    print(
        result[VOLATILITY_TARGET]
        .value_counts()
        .reindex(VOLATILITY_CLASSES)
        .fillna(0)
        .astype(int)
        .to_string()
    )

    # Verify all 3 classes exist
    counts = result[VOLATILITY_TARGET].value_counts()
    empty = [c for c in VOLATILITY_CLASSES if counts.get(c, 0) == 0]
    if empty:
        print(
            f"\n   WARNING: Classes with zero rows in test: {empty}"
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    out_path = volatility_dataset_test_path()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        out_path,
        index=False
    )

    print(
        f"\n   Saved to: {out_path}"
    )

    return result


def main():
    build_volatility_test()


if __name__ == "__main__":
    main()
