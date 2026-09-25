"""
AgroVision — Rolling Features

Creates rolling mean and std features from past price observations.
Uses shift(1) to ensure no current-row leakage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    lag_features_path,
    rolling_features_path,
)


def create_rolling(df=None):
    """
    Create rolling mean and std features.

    All rolling calculations use shift(1) to prevent leakage:
    only past observations are included.
    """
    if df is None:
        path = lag_features_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 3b: ROLLING FEATURES")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    # Use shift(1) so only PAST prices are available
    grouped_price = df.groupby(group_cols)[PRICE_COLUMN]
    past_price = grouped_price.shift(1)

    # Group the shifted series for rolling calculations
    group_keys = [df[c] for c in group_cols]
    grouped_past = past_price.groupby(group_keys)

    # Rolling means
    df["rolling_mean_7"] = grouped_past.transform(lambda x: x.rolling(7).mean())
    df["rolling_mean_14"] = grouped_past.transform(lambda x: x.rolling(14).mean())
    df["rolling_mean_30"] = grouped_past.transform(lambda x: x.rolling(30).mean())

    # Rolling standard deviations
    df["rolling_std_7"] = grouped_past.transform(lambda x: x.rolling(7).std())
    df["rolling_std_14"] = grouped_past.transform(lambda x: x.rolling(14).std())

    rolling_cols = ["rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
                    "rolling_std_7", "rolling_std_14"]
    print(f"   Created: {', '.join(rolling_cols)}")
    print(f"   Missing values: {df[rolling_cols].isna().sum().to_dict()}")
    print(f"   Rows: {len(df)}")

    out_path = rolling_features_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    create_rolling()


if __name__ == "__main__":
    main()