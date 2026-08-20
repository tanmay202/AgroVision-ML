import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    TIMESERIES_PATH,
    LAG_FEATURES_PATH,
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load time-series dataset
    # --------------------------------------------------
    if not TIMESERIES_PATH.exists():
        print(f"ERROR: File not found: {TIMESERIES_PATH}")
        return

    df = pd.read_csv(TIMESERIES_PATH)

    # --------------------------------------------------
    # 2. Convert date
    # --------------------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------
    # 3. Sort chronologically
    # --------------------------------------------------
    df = df.sort_values(
        by=GROUP_COLUMNS + [DATE_COLUMN]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. Create lag features
    # --------------------------------------------------
    grouped_price = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]

    df["lag_1"] = grouped_price.shift(1)
    df["lag_7"] = grouped_price.shift(7)
    df["lag_14"] = grouped_price.shift(14)
    df["lag_30"] = grouped_price.shift(30)

    # --------------------------------------------------
    # 5. Display examples
    # --------------------------------------------------
    print("=" * 60)
    print("LAG FEATURE EXAMPLES")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        DATE_COLUMN,
        PRICE_COLUMN,
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        PRICE_TARGET,
    ]

    print(df[example_columns].head(40))

    # --------------------------------------------------
    # 6. Count missing values in lag features
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("LAG FEATURE MISSING VALUES")
    print("=" * 60)

    lag_columns = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30"
    ]

    print(df[lag_columns].isna().sum())

    # --------------------------------------------------
    # 7. Save result
    # --------------------------------------------------
    LAG_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        LAG_FEATURES_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {LAG_FEATURES_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()