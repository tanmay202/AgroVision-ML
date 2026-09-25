"""
AgroVision — Data Cleaning

Cleans raw mandi CSV data:
  1. Parse dates (with fallback for unknown formats)
  2. Convert numeric columns
  3. Remove invalid/impossible values
  4. Sort chronologically
  5. Save cleaned dataset
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    NUMERIC_COLUMNS,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    raw_data_path,
    cleaned_data_path,
)


def clean(df=None, commodity=None):
    """
    Clean a raw mandi DataFrame.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Raw data. If None, reads from raw CSV for current commodity.
    commodity : str, optional
        Commodity name override.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    if commodity:
        from config import set_commodity
        set_commodity(commodity)

    raw_path = raw_data_path()

    if df is None:
        if not raw_path.exists():
            raise FileNotFoundError(f"Dataset not found at {raw_path}")
        df = pd.read_csv(raw_path)

    print("=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)
    print(f"   Initial rows: {len(df)}")

    # --------------------------------------------------
    # 1. Parse dates
    # --------------------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        format="%d %b %Y",
        errors="coerce"
    )

    # Fallback: if most dates failed, retry with mixed format
    if df[DATE_COLUMN].isna().sum() > len(df) * 0.5:
        print("   [INFO] Explicit date format failed for >50% rows, trying auto-detection...")
        raw_dates = pd.read_csv(raw_path)[DATE_COLUMN]
        df[DATE_COLUMN] = pd.to_datetime(
            raw_dates,
            format="mixed",
            dayfirst=True,
            errors="coerce"
        )

    invalid_dates = df[DATE_COLUMN].isna().sum()
    print(f"   Invalid dates: {invalid_dates}")

    # --------------------------------------------------
    # 2. Convert numeric columns
    # --------------------------------------------------
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # --------------------------------------------------
    # 3. Remove rows with invalid dates
    # --------------------------------------------------
    before = len(df)
    df = df.dropna(subset=[DATE_COLUMN])
    print(f"   Dropped {before - len(df)} rows with invalid dates")

    # --------------------------------------------------
    # 4. Remove duplicate rows
    # --------------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    print(f"   Dropped {before - len(df)} duplicate rows")

    # --------------------------------------------------
    # 5. Remove impossible values
    # --------------------------------------------------
    before = len(df)
    df = df[df[ARRIVAL_COLUMN] >= 0]
    print(f"   Dropped {before - len(df)} rows with negative arrivals")

    before = len(df)
    df = df[
        (df[MIN_PRICE_COLUMN] > 0)
        & (df[MAX_PRICE_COLUMN] > 0)
        & (df[PRICE_COLUMN] > 0)
    ]
    print(f"   Dropped {before - len(df)} rows with zero/negative prices")

    # --------------------------------------------------
    # 6. Remove structurally inconsistent prices
    # --------------------------------------------------
    before = len(df)
    df = df[df[MIN_PRICE_COLUMN] <= df[MAX_PRICE_COLUMN]]
    print(f"   Dropped {before - len(df)} rows where Min > Max price")

    before = len(df)
    df = df[
        (df[PRICE_COLUMN] >= df[MIN_PRICE_COLUMN])
        & (df[PRICE_COLUMN] <= df[MAX_PRICE_COLUMN])
    ]
    print(f"   Dropped {before - len(df)} rows with Modal outside Min/Max")

    # --------------------------------------------------
    # 7. Sort chronologically within each group
    # --------------------------------------------------
    sort_cols = [c for c in GROUP_COLUMNS if c in df.columns] + [DATE_COLUMN]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    # --------------------------------------------------
    # 8. Report & save
    # --------------------------------------------------
    print(f"\n   Final rows: {len(df)}")
    print(f"   Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")

    out_path = cleaned_data_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    clean()


if __name__ == "__main__":
    main()