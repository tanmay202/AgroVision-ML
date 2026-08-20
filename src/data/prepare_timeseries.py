import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    CLEANED_DATA_PATH,
    TIMESERIES_PATH,
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load cleaned dataset
    # --------------------------------------------------
    if not CLEANED_DATA_PATH.exists():
        print(f"ERROR: File not found: {CLEANED_DATA_PATH}")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)

    # --------------------------------------------------
    # 2. Convert date again after reading CSV
    # --------------------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    print("=" * 60)
    print("INITIAL DATA")
    print("=" * 60)
    print(f"Rows: {len(df)}")

    # --------------------------------------------------
    # 3. Sort chronologically within each market/variety
    # --------------------------------------------------
    df = df.sort_values(
        by=GROUP_COLUMNS + [DATE_COLUMN]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. Create future-price target
    # --------------------------------------------------
    # shift(-1) means:
    # current row -> next chronological observation
    df[PRICE_TARGET] = (
        df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]
        .shift(-1)
    )

    # --------------------------------------------------
    # 5. Record how many days ahead each target is
    # --------------------------------------------------
    next_date = (
        df.groupby(GROUP_COLUMNS)[DATE_COLUMN]
        .shift(-1)
    )

    df["days_to_next"] = (
        next_date - df[DATE_COLUMN]
    ).dt.days

    # --------------------------------------------------
    # 6. Inspect target creation
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TARGET EXAMPLE")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        DATE_COLUMN,
        PRICE_COLUMN,
        PRICE_TARGET,
        "days_to_next",
    ]

    print(df[example_columns].head(15))

    # --------------------------------------------------
    # 7. Count rows without a future observation
    # --------------------------------------------------
    rows_without_target = df[PRICE_TARGET].isna().sum()

    print("\nRows without future target:", rows_without_target)

    # --------------------------------------------------
    # 8. Remove rows that cannot have a target
    # --------------------------------------------------
    df = df.dropna(
        subset=[PRICE_TARGET]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 9. Forecast horizon statistics
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FORECAST HORIZON (days_to_next)")
    print("=" * 60)
    print(df["days_to_next"].describe())

    # --------------------------------------------------
    # 10. Check temporal ordering
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TIME-SERIES CHECK")
    print("=" * 60)

    check = df.groupby(GROUP_COLUMNS)[DATE_COLUMN].diff()

    print("Negative time differences:", (check.dt.total_seconds() < 0).sum())
    print("Rows remaining:", len(df))

    # --------------------------------------------------
    # 11. Save prepared dataset
    # --------------------------------------------------
    TIMESERIES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        TIMESERIES_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)
    print(f"Saved to: {TIMESERIES_PATH}")

    print("\nFinal columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()