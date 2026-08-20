import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    VOLATILITY_TRAIN_PATH,
    VOLATILITY_DATASET_TRAIN_PATH,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
)


def main():
    # --------------------------------------------------
    # 1. Check input file
    # --------------------------------------------------
    if not VOLATILITY_TRAIN_PATH.exists():
        print(f"ERROR: File not found: {VOLATILITY_TRAIN_PATH}")
        return

    # --------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------
    df = pd.read_csv(VOLATILITY_TRAIN_PATH)

    print("=" * 60)
    print("INITIAL VOLATILITY DATASET")
    print("=" * 60)

    print(f"Rows: {len(df)}")

    # --------------------------------------------------
    # 3. Validate required columns
    # --------------------------------------------------
    required_columns = VOLATILITY_FEATURES + [
        VOLATILITY_TARGET,
        "price_pct_change",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Missing required columns:")
        print(missing_columns)
        return

    # --------------------------------------------------
    # 4. Remove rows without volatility target
    # --------------------------------------------------
    before = len(df)

    df = df.dropna(
        subset=[VOLATILITY_TARGET]
    ).copy()

    print(
        "Rows removed because target is missing:",
        before - len(df)
    )

    # --------------------------------------------------
    # 5. Check missing feature rows
    # --------------------------------------------------
    missing_feature_rows = (
        df[VOLATILITY_FEATURES]
        .isna()
        .any(axis=1)
        .sum()
    )

    print(
        "Rows with missing features:",
        missing_feature_rows
    )

    # --------------------------------------------------
    # 6. Remove rows with missing historical features
    # --------------------------------------------------
    df = df.dropna(
        subset=VOLATILITY_FEATURES
    ).copy()

    # --------------------------------------------------
    # 7. Create final dataset
    #
    # IMPORTANT:
    # price_pct_change is NOT selected here.
    # --------------------------------------------------
    result = df[
        VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    ].copy()

    # --------------------------------------------------
    # 8. Verify leakage protection
    # --------------------------------------------------
    if "price_pct_change" in result.columns:
        print(
            "ERROR: price_pct_change is still present!"
        )
        return

    print("\n" + "=" * 60)
    print("FINAL VOLATILITY DATASET")
    print("=" * 60)

    print(f"Rows: {len(result)}")
    print(f"Features: {len(VOLATILITY_FEATURES)}")
    print(f"Target: {VOLATILITY_TARGET}")

    # --------------------------------------------------
    # 9. Class distribution
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    print(
        result[VOLATILITY_TARGET]
        .value_counts()
    )

    print("\nPercentages:")

    print(
        (
            result[VOLATILITY_TARGET]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # --------------------------------------------------
    # 10. Target validation
    # --------------------------------------------------
    allowed_classes = {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    actual_classes = set(
        result[VOLATILITY_TARGET].unique()
    )

    unexpected_classes = (
        actual_classes - allowed_classes
    )

    print("\n" + "=" * 60)
    print("TARGET VALIDATION")
    print("=" * 60)

    if unexpected_classes:
        print(
            "ERROR: Unexpected classes:",
            unexpected_classes
        )
        return

    print(
        "PASS: Only LOW, MEDIUM, HIGH classes exist."
    )

    # --------------------------------------------------
    # 11. Final column check
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL COLUMN CHECK")
    print("=" * 60)

    print("Columns:")
    for i, column in enumerate(result.columns, start=1):
        print(f"{i:2}. {column}")

    expected_columns = VOLATILITY_FEATURES + [VOLATILITY_TARGET]

    if list(result.columns) == expected_columns:
        print(
            "\nPASS: Final column order is correct."
        )
    else:
        print(
            "\nERROR: Final column order is incorrect."
        )
        return

    # --------------------------------------------------
    # 12. Save
    # --------------------------------------------------
    VOLATILITY_DATASET_TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        VOLATILITY_DATASET_TRAIN_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(
        f"Saved to: {VOLATILITY_DATASET_TRAIN_PATH}"
    )


if __name__ == "__main__":
    main()