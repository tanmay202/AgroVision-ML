import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    LAG_FEATURES_PATH,
    ROLLING_FEATURES_PATH,
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not LAG_FEATURES_PATH.exists():
        print(f"ERROR: File not found: {LAG_FEATURES_PATH}")
        return

    df = pd.read_csv(LAG_FEATURES_PATH)

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
    # 4. Create grouped past-price series
    # --------------------------------------------------
    grouped_price = df.groupby(
        GROUP_COLUMNS
    )[PRICE_COLUMN]

    # IMPORTANT:
    # shift(1) means only previous observations
    # are available to the rolling calculation.
    past_price = grouped_price.shift(1)

    # --------------------------------------------------
    # 5. Rolling mean features
    # --------------------------------------------------
    df["rolling_mean_7"] = (
        past_price
        .groupby(
            [
                df["Market Name"],
                df["Variety"]
            ]
        )
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    df["rolling_mean_14"] = (
        past_price
        .groupby(
            [
                df["Market Name"],
                df["Variety"]
            ]
        )
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    df["rolling_mean_30"] = (
        past_price
        .groupby(
            [
                df["Market Name"],
                df["Variety"]
            ]
        )
        .transform(
            lambda x: x.rolling(30).mean()
        )
    )

    # --------------------------------------------------
    # 6. Rolling standard deviation features
    # --------------------------------------------------
    df["rolling_std_7"] = (
        past_price
        .groupby(
            [
                df["Market Name"],
                df["Variety"]
            ]
        )
        .transform(
            lambda x: x.rolling(7).std()
        )
    )

    df["rolling_std_14"] = (
        past_price
        .groupby(
            [
                df["Market Name"],
                df["Variety"]
            ]
        )
        .transform(
            lambda x: x.rolling(14).std()
        )
    )

    # --------------------------------------------------
    # 7. Display examples
    # --------------------------------------------------
    print("=" * 60)
    print("ROLLING FEATURE EXAMPLES")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        DATE_COLUMN,
        PRICE_COLUMN,
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_14",
        PRICE_TARGET,
    ]

    print(df[example_columns].head(40))

    # --------------------------------------------------
    # 8. Missing values
    # --------------------------------------------------
    rolling_columns = [
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_30",
        "rolling_std_7",
        "rolling_std_14"
    ]

    print("\n" + "=" * 60)
    print("ROLLING FEATURE MISSING VALUES")
    print("=" * 60)

    print(df[rolling_columns].isna().sum())

    # --------------------------------------------------
    # 9. Save
    # --------------------------------------------------
    ROLLING_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        ROLLING_FEATURES_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {ROLLING_FEATURES_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()