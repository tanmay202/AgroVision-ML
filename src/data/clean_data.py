import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    RAW_DATA_PATH,
    CLEANED_DATA_PATH,
    DATE_COLUMN,
    NUMERIC_COLUMNS,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
)


def main():
    # --------------------------------------------------
    # 1. Load raw dataset
    # --------------------------------------------------
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    print("=" * 60)
    print("INITIAL DATA")
    print("=" * 60)
    print(f"Rows: {len(df)}")

    # --------------------------------------------------
    # 2. Convert date using explicit format
    # --------------------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        format="%d %b %Y",
        errors="coerce"
    )

    # Fallback: if most dates failed, retry with automatic format detection
    if df[DATE_COLUMN].isna().sum() > len(df) * 0.5:
        print("   [INFO] Explicit date format failed for >50% rows, trying auto-detection...")
        df[DATE_COLUMN] = pd.to_datetime(
            pd.read_csv(RAW_DATA_PATH)[DATE_COLUMN],
            infer_datetime_format=True,
            errors="coerce"
        )

    invalid_dates = df[DATE_COLUMN].isna().sum()

    print("\n" + "=" * 60)
    print("DATE CLEANING")
    print("=" * 60)
    print(f"Invalid dates after explicit parsing: {invalid_dates}")

    # --------------------------------------------------
    # 3. Convert numeric columns
    # --------------------------------------------------
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    print("\n" + "=" * 60)
    print("NUMERIC VALIDATION")
    print("=" * 60)

    for column in NUMERIC_COLUMNS:
        invalid_count = df[column].isna().sum()
        print(f"{column}: {invalid_count} invalid values")

    # --------------------------------------------------
    # 4. Remove rows with invalid dates
    # --------------------------------------------------
    before = len(df)

    df = df.dropna(subset=[DATE_COLUMN])

    removed_dates = before - len(df)

    print("\nRows removed because of invalid dates:", removed_dates)

    # --------------------------------------------------
    # 5. Remove duplicate rows
    # --------------------------------------------------
    before = len(df)

    df = df.drop_duplicates()

    removed_duplicates = before - len(df)

    print("Duplicate rows removed:", removed_duplicates)

    # --------------------------------------------------
    # 6. Remove impossible numeric values
    # --------------------------------------------------

    # Arrival cannot be negative
    before = len(df)

    df = df[df[ARRIVAL_COLUMN] >= 0]

    print(
        "Negative arrival rows removed:",
        before - len(df)
    )

    # Prices must be strictly positive
    before = len(df)

    df = df[
        (df[MIN_PRICE_COLUMN] > 0)
        & (df[MAX_PRICE_COLUMN] > 0)
        & (df[PRICE_COLUMN] > 0)
    ]

    print(
        "Zero or negative price rows removed:",
        before - len(df)
    )

    # --------------------------------------------------
    # 7. Remove structurally inconsistent price rows
    # --------------------------------------------------
    before = len(df)

    df = df[
        df[MIN_PRICE_COLUMN]
        <= df[MAX_PRICE_COLUMN]
    ]

    print(
        "Rows where Min Price > Max Price removed:",
        before - len(df)
    )

    # Modal price should normally lie between min and max.
    before = len(df)

    df = df[
        (df[PRICE_COLUMN] >= df[MIN_PRICE_COLUMN])
        & (df[PRICE_COLUMN] <= df[MAX_PRICE_COLUMN])
    ]

    print(
        "Rows with Modal Price outside Min/Max removed:",
        before - len(df)
    )

    # --------------------------------------------------
    # 8. Sort chronologically
    # --------------------------------------------------
    df = df.sort_values(
        by=[
            "Market Name",
            "Variety",
            DATE_COLUMN
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 9. Check remaining missing values
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL MISSING VALUE CHECK")
    print("=" * 60)

    print(df.isna().sum())

    # --------------------------------------------------
    # 10. Final dataset information
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANED DATASET")
    print("=" * 60)

    print(f"Final rows: {len(df)}")
    print(f"Final columns: {len(df.columns)}")
    print(f"Minimum date: {df[DATE_COLUMN].min()}")
    print(f"Maximum date: {df[DATE_COLUMN].max()}")

    # --------------------------------------------------
    # 11. Save cleaned dataset
    # --------------------------------------------------
    CLEANED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    print("\nCleaned dataset saved to:")
    print(CLEANED_DATA_PATH)


if __name__ == "__main__":
    main()