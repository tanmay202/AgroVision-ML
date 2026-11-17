"""
AgroVision — Arrival Features

Creates arrival-based features (lags, rolling means).

LEAKAGE FIX (v2.0):
    arrival_change now uses only past values:
    arrival_change = arrival_lag_1 - arrival_lag_2
    (previously used current arrival, which is unknown at prediction time)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    date_features_path,
    arrival_features_path,
)


def create_arrivals(df=None):
    """
    Create arrival lag and rolling features.
    """
    if df is None:
        path = date_features_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 3d: ARRIVAL FEATURES")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    grouped_arrival = df.groupby(group_cols)[ARRIVAL_COLUMN]

    # Arrival lags
    df["arrival_lag_1"] = grouped_arrival.shift(1)
    df["arrival_lag_7"] = grouped_arrival.shift(7)

    # Arrival change: difference between two PAST values (no leakage)
    arrival_lag_2 = grouped_arrival.shift(2)
    df["arrival_change"] = df["arrival_lag_1"] - arrival_lag_2

    # Rolling arrival features (using shifted past values)
    past_arrival = grouped_arrival.shift(1)
    group_keys = [df[c] for c in group_cols]
    grouped_past = past_arrival.groupby(group_keys)

    df["arrival_rolling_mean_7"] = grouped_past.transform(
        lambda x: x.rolling(7).mean()
    )
    df["arrival_rolling_mean_14"] = grouped_past.transform(
        lambda x: x.rolling(14).mean()
    )

    arrival_features = ["arrival_lag_1", "arrival_lag_7", "arrival_change",
                        "arrival_rolling_mean_7", "arrival_rolling_mean_14"]
    print(f"   Created: {', '.join(arrival_features)}")
    print(f"   Missing values: {df[arrival_features].isna().sum().to_dict()}")
    print(f"   Rows: {len(df)}")

    out_path = arrival_features_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    create_arrivals()


if __name__ == "__main__":
    main()