"""
AgroVision — Lag Features

Creates price lag features for each market/variety group.
Lags use shift(N) which only looks at past observations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    timeseries_path,
    lag_features_path,
)


def create_lags(df=None):
    """
    Create price lag features.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Time-series data. If None, reads from CSV.

    Returns
    -------
    pd.DataFrame
        Dataset with lag features added.
    """
    if df is None:
        path = timeseries_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 3a: LAG FEATURES")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    grouped_price = df.groupby(group_cols)[PRICE_COLUMN]

    df["lag_1"] = grouped_price.shift(1)
    df["lag_7"] = grouped_price.shift(7)
    df["lag_14"] = grouped_price.shift(14)
    df["lag_30"] = grouped_price.shift(30)

    lag_columns = ["lag_1", "lag_7", "lag_14", "lag_30"]
    print(f"   Created: {', '.join(lag_columns)}")
    print(f"   Missing values: {df[lag_columns].isna().sum().to_dict()}")
    print(f"   Rows: {len(df)}")

    out_path = lag_features_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    create_lags()


if __name__ == "__main__":
    main()