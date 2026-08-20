import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    ROLLING_FEATURES_PATH,
    DATE_FEATURES_PATH,
    DATE_COLUMN,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not ROLLING_FEATURES_PATH.exists():
        print(f"ERROR: File not found: {ROLLING_FEATURES_PATH}")
        return

    df = pd.read_csv(ROLLING_FEATURES_PATH)

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
    # 4. Create date features
    # --------------------------------------------------
    df["year"] = df[DATE_COLUMN].dt.year

    df["day"] = df[DATE_COLUMN].dt.day

    df["month"] = df[DATE_COLUMN].dt.month

    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek

    df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

    # --------------------------------------------------
    # 5. Display examples
    # --------------------------------------------------
    print("=" * 60)
    print("DATE FEATURE EXAMPLES")
    print("=" * 60)

    example_columns = [
        DATE_COLUMN,
        "year",
        "day",
        "month",
        "day_of_week",
        "week_of_year"
    ]

    print(df[example_columns].head(15))

    # --------------------------------------------------
    # 6. Check missing values
    # --------------------------------------------------
    date_features = [
        "year",
        "day",
        "month",
        "day_of_week",
        "week_of_year"
    ]

    print("\n" + "=" * 60)
    print("DATE FEATURE MISSING VALUES")
    print("=" * 60)

    print(df[date_features].isna().sum())

    # --------------------------------------------------
    # 7. Check ranges
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("DATE FEATURE RANGES")
    print("=" * 60)

    for column in date_features:
        print(
            f"{column}: "
            f"{df[column].min()} -> {df[column].max()}"
        )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------
    DATE_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        DATE_FEATURES_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {DATE_FEATURES_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()