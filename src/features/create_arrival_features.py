import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_FEATURES_PATH,
    ARRIVAL_FEATURES_PATH,
    DATE_COLUMN,
    ARRIVAL_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not DATE_FEATURES_PATH.exists():
        print(f"ERROR: File not found: {DATE_FEATURES_PATH}")
        return

    df = pd.read_csv(DATE_FEATURES_PATH)

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
    # 4. Create grouped arrival series
    # --------------------------------------------------
    grouped_arrival = df.groupby(
        GROUP_COLUMNS
    )[ARRIVAL_COLUMN]

    # --------------------------------------------------
    # 5. Arrival lag features
    # --------------------------------------------------
    df["arrival_lag_1"] = grouped_arrival.shift(1)

    df["arrival_lag_7"] = grouped_arrival.shift(7)

    # --------------------------------------------------
    # 6. Arrival change feature
    # --------------------------------------------------
    previous_arrival = grouped_arrival.shift(1)

    df["arrival_change"] = (
        df[ARRIVAL_COLUMN] - previous_arrival
    )

    # --------------------------------------------------
    # 7. Past arrival series for rolling features
    # --------------------------------------------------
    past_arrival = grouped_arrival.shift(1)

    grouped_past_arrival = past_arrival.groupby(
        [
            df["Market Name"],
            df["Variety"]
        ]
    )

    # --------------------------------------------------
    # 8. Rolling arrival mean
    # --------------------------------------------------
    df["arrival_rolling_mean_7"] = (
        grouped_past_arrival
        .transform(
            lambda x: x.rolling(7).mean()
        )
    )

    df["arrival_rolling_mean_14"] = (
        grouped_past_arrival
        .transform(
            lambda x: x.rolling(14).mean()
        )
    )

    # --------------------------------------------------
    # 9. Display examples
    # --------------------------------------------------
    print("=" * 60)
    print("ARRIVAL FEATURE EXAMPLES")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        DATE_COLUMN,
        ARRIVAL_COLUMN,
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_change",
        "arrival_rolling_mean_7",
        "arrival_rolling_mean_14",
        PRICE_TARGET,
    ]

    print(df[example_columns].head(40))

    # --------------------------------------------------
    # 10. Missing values
    # --------------------------------------------------
    arrival_features = [
        "arrival_lag_1",
        "arrival_lag_7",
        "arrival_change",
        "arrival_rolling_mean_7",
        "arrival_rolling_mean_14"
    ]

    print("\n" + "=" * 60)
    print("ARRIVAL FEATURE MISSING VALUES")
    print("=" * 60)

    print(df[arrival_features].isna().sum())

    # --------------------------------------------------
    # 11. Save dataset
    # --------------------------------------------------
    ARRIVAL_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        ARRIVAL_FEATURES_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {ARRIVAL_FEATURES_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()