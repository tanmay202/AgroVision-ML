"""
AgroVision - Data Cleaner
Cleans the tea mandi price/arrival dataset.

Steps:
    1. Parse dates and extract temporal features
    2. Strip whitespace from string columns
    3. Remove zero/negative arrivals and prices
    4. Remove outliers (beyond 3 std deviations)
    5. Standardize string columns
"""

import os
import numpy as np
import pandas as pd

from src.config import COLUMNS, PROCESSED_DATA_DIR, CLEANED_DATASET_FILE


def clean_data(df):
    """
    Clean the tea mandi DataFrame.

    Args:
        df: Raw DataFrame from data_loader

    Returns:
        Cleaned DataFrame with temporal features added
    """
    col = COLUMNS
    print("\n[CLEAN] Cleaning data...")
    df = df.copy()

    # 1. Strip whitespace from string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    # 2. Parse the Reported Date column
    if col["date"] in df.columns:
        df[col["date"]] = pd.to_datetime(df[col["date"]], errors="coerce")
        invalid_dates = df[col["date"]].isna().sum()
        if invalid_dates > 0:
            print(f"   Dropped {invalid_dates} rows with invalid dates")
            df = df.dropna(subset=[col["date"]])

        # Extract temporal components
        df["Year"] = df[col["date"]].dt.year
        df["Month"] = df[col["date"]].dt.month
        df["Day"] = df[col["date"]].dt.day
        df["DayOfWeek"] = df[col["date"]].dt.dayofweek  # 0=Monday, 6=Sunday
        df["WeekOfYear"] = df[col["date"]].dt.isocalendar().week.astype(int)
        df["Quarter"] = df[col["date"]].dt.quarter
        print(f"   Extracted: Year, Month, Day, DayOfWeek, WeekOfYear, Quarter")

    # 3. Remove zero/negative arrivals
    if col["arrivals"] in df.columns:
        before = len(df)
        df = df[df[col["arrivals"]] > 0].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"   Removed {dropped} rows with zero/negative arrivals")

    # 4. Remove zero/negative prices
    price_cols = [col["min_price"], col["max_price"], col["modal_price"]]
    price_cols = [c for c in price_cols if c in df.columns]
    for pc in price_cols:
        before = len(df)
        df = df[df[pc] > 0].copy()
        dropped = before - len(df)
        if dropped > 0:
            print(f"   Removed {dropped} rows with zero/negative {pc}")

    # 5. Standardize state and district names
    if col["state"] in df.columns:
        df[col["state"]] = df[col["state"]].str.title()
    if col["district"] in df.columns:
        df[col["district"]] = df[col["district"]].str.title()
    if col["market"] in df.columns:
        df[col["market"]] = df[col["market"]].str.title()

    # 6. Remove arrival outliers per variety (beyond 3 std deviations)
    if col["arrivals"] in df.columns and col["variety"] in df.columns:
        before = len(df)
        df = df.groupby(col["variety"], group_keys=False).apply(
            _remove_outliers, col_name=col["arrivals"]
        )
        removed = before - len(df)
        if removed > 0:
            print(f"   Removed {removed} arrival outliers (>3 std per variety)")

    # 7. Compute price spread and price range
    if col["max_price"] in df.columns and col["min_price"] in df.columns:
        df["Price_Spread"] = df[col["max_price"]] - df[col["min_price"]]
        print(f"   Created: Price_Spread (Max - Min price)")

    if col["modal_price"] in df.columns and col["min_price"] in df.columns:
        df["Price_Ratio"] = df[col["modal_price"]] / df[col["min_price"]]
        print(f"   Created: Price_Ratio (Modal / Min price)")

    # Sort by date
    if col["date"] in df.columns:
        df = df.sort_values(col["date"]).reset_index(drop=True)

    print(f"\n   [OK] Cleaned data: {len(df):,} rows, {len(df.columns)} columns")

    return df


def _remove_outliers(group, col_name, n_std=3.0):
    """Remove rows where value is beyond n_std standard deviations from mean."""
    mean = group[col_name].mean()
    std = group[col_name].std()
    if std == 0 or np.isnan(std):
        return group
    return group[
        (group[col_name] >= mean - n_std * std)
        & (group[col_name] <= mean + n_std * std)
    ]


def clean_and_save(data):
    """
    Full cleaning pipeline.

    Args:
        data: dict from data_loader.load_all_data()

    Returns:
        Cleaned DataFrame, also saved to disk
    """
    print("\n" + "=" * 60)
    print("STEP 2: CLEANING DATA")
    print("=" * 60)

    cleaned = clean_data(data["main"])

    # Save to disk
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    cleaned.to_csv(CLEANED_DATASET_FILE, index=False)
    print(f"\n[SAVE] Saved cleaned dataset to: {CLEANED_DATASET_FILE}")

    return cleaned
