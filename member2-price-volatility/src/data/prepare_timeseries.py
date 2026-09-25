"""
AgroVision — Time-Series Target Preparation

Creates the forecasting target:
  - future_modal_price = next observation's price (within same group)
  - Filters to rows where next observation is within FORECAST_HORIZON_MAX_DAYS
  - Records days_to_next for transparency
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
    FORECAST_HORIZON_MAX_DAYS,
    cleaned_data_path,
    timeseries_path,
)


def prepare(df=None):
    """
    Create future-price target and filter by forecast horizon.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Cleaned data. If None, reads from cleaned CSV.

    Returns
    -------
    pd.DataFrame
        Dataset with future_modal_price target.
    """
    if df is None:
        path = cleaned_data_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 2: TIME-SERIES TARGET")
    print("=" * 60)

    # Ensure date is datetime
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    # Sort chronologically within each group
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    print(f"   Initial rows: {len(df)}")

    # Create future-price target: 
    df[PRICE_TARGET] = (
    df.groupby(group_cols)[PRICE_COLUMN]
    .shift(-1)
    )

    # Create percentage-change target
    df["future_price_pct_change"] = (
    (df[PRICE_TARGET] - df[PRICE_COLUMN])
    / df[PRICE_COLUMN]
    )
    
    
    # Log-return target
    df["future_log_return"] = np.log(
    df[PRICE_TARGET] / df[PRICE_COLUMN]
    )



    # Record how many days ahead each target is
    next_date = df.groupby(group_cols)[DATE_COLUMN].shift(-1)
    df["days_to_next"] = (next_date - df[DATE_COLUMN]).dt.days

    # Remove rows without a target (last row per group)
    before = len(df)
    df = df.dropna(subset=[PRICE_TARGET]).reset_index(drop=True)
    print(f"   Dropped {before - len(df)} rows without future target (last per group)")

    # Filter by forecast horizon
    before = len(df)
    df = df[df["days_to_next"] <= FORECAST_HORIZON_MAX_DAYS].reset_index(drop=True)
    filtered = before - len(df)
    print(f"   Dropped {filtered} rows where days_to_next > {FORECAST_HORIZON_MAX_DAYS}")

    # Report forecast horizon statistics
    print(f"\n   Forecast horizon (days_to_next):")
    print(f"     Mean  : {df['days_to_next'].mean():.1f}")
    print(f"     Median: {df['days_to_next'].median():.0f}")
    print(f"     Max   : {df['days_to_next'].max():.0f}")

    # Temporal check
    check = df.groupby(group_cols)[DATE_COLUMN].diff()
    neg_diffs = (check.dt.total_seconds() < 0).sum()
    if neg_diffs > 0:
        print(f"   WARNING: {neg_diffs} negative time differences found!")
    else:
        print(f"   PASS: No negative time differences.")

    print(f"\n   Final rows: {len(df)}")

    # Save
    out_path = timeseries_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    prepare()


if __name__ == "__main__":
    main()