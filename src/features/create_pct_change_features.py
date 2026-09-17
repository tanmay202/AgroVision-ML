import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    ARRIVAL_FEATURES_PATH,
    FEATURES_FINAL_PATH,
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not ARRIVAL_FEATURES_PATH.exists():
        print(f"ERROR: File not found: {ARRIVAL_FEATURES_PATH}")
        return

    df = pd.read_csv(ARRIVAL_FEATURES_PATH)

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
    # 4. Previous price and arrival
    # --------------------------------------------------
    grouped_price = df.groupby(
        GROUP_COLUMNS
    )[PRICE_COLUMN]

    grouped_arrival = df.groupby(
        GROUP_COLUMNS
    )[ARRIVAL_COLUMN]

    previous_price = grouped_price.shift(1)
    previous_arrival = grouped_arrival.shift(1)

    # --------------------------------------------------
    # 5. Price percentage change
    # --------------------------------------------------
    df["price_pct_change"] = (
        (df[PRICE_COLUMN] - previous_price)
        / previous_price
    ) * 100

    # --------------------------------------------------
    # 6. Arrival percentage change
    # --------------------------------------------------
    df["arrival_pct_change"] = (
        (df[ARRIVAL_COLUMN] - previous_arrival)
        / previous_arrival
    ) * 100

    # --------------------------------------------------
    # 7. Replace infinite values
    # --------------------------------------------------
    df["price_pct_change"] = (
        df["price_pct_change"]
        .replace([float("inf"), -float("inf")], 0.0)
    )

    df["arrival_pct_change"] = (
        df["arrival_pct_change"]
        .replace([float("inf"), -float("inf")], 0.0)
    )

    # --------------------------------------------------
    # 8. Display examples
    # --------------------------------------------------
    print("=" * 60)
    print("PERCENTAGE CHANGE EXAMPLES")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        DATE_COLUMN,
        PRICE_COLUMN,
        "price_pct_change",
        ARRIVAL_COLUMN,
        "arrival_pct_change",
        PRICE_TARGET,
    ]

    print(df[example_columns].head(40))

    # --------------------------------------------------
    # 9. Check missing values
    # --------------------------------------------------
    pct_columns = [
        "price_pct_change",
        "arrival_pct_change"
    ]

    print("\n" + "=" * 60)
    print("PERCENTAGE FEATURE MISSING VALUES")
    print("=" * 60)

    print(df[pct_columns].isna().sum())

    # --------------------------------------------------
    # 10. Basic statistics
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PERCENTAGE FEATURE STATISTICS")
    print("=" * 60)

    print(df[pct_columns].describe())

    # --------------------------------------------------
    # 11. Save final feature dataset
    # --------------------------------------------------
    FEATURES_FINAL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FEATURES_FINAL_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {FEATURES_FINAL_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nFinal feature columns:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()