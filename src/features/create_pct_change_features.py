"""
AgroVision — Percentage Change Features

Computes price and arrival percentage changes from previous values.

Note:
    price_pct_change = (price - previous_price) / previous_price * 100
    This uses shift(1) so it represents the change FROM the previous
    observation TO the current one. Safe for the price model (it's a
    known-at-prediction-time feature since it uses past prices).

    For the VOLATILITY model, price_pct_change is intentionally
    EXCLUDED in config.VOLATILITY_FEATURES because volatility is
    directly derived from it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    arrival_features_path,
    features_final_path,
)


def create_pct_changes(df=None):
    """
    Create percentage change features.

    Parameters
    ----------
    df : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
    """
    if df is None:
        path = arrival_features_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 3e: PERCENTAGE CHANGE FEATURES")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    # Previous values (shifted)
    previous_price = df.groupby(group_cols)[PRICE_COLUMN].shift(1)
    previous_arrival = df.groupby(group_cols)[ARRIVAL_COLUMN].shift(1)

    # Price percentage change
    df["price_pct_change"] = (
        (df[PRICE_COLUMN] - previous_price) / previous_price
    ) * 100

    # Arrival percentage change
    df["arrival_pct_change"] = (
        (df[ARRIVAL_COLUMN] - previous_arrival) / previous_arrival
    ) * 100

    # Replace infinite values (from division by zero)
    df["price_pct_change"] = df["price_pct_change"].replace(
        [float("inf"), -float("inf")], 0.0
    )
    df["arrival_pct_change"] = df["arrival_pct_change"].replace(
        [float("inf"), -float("inf")], 0.0
    )

    pct_cols = ["price_pct_change", "arrival_pct_change"]
    print(f"   Created: {', '.join(pct_cols)}")
    print(f"   Missing values: {df[pct_cols].isna().sum().to_dict()}")
    print(f"   Rows: {len(df)}")

    out_path = features_final_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    create_pct_changes()


if __name__ == "__main__":
    main()