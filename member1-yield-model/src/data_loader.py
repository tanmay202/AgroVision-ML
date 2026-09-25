"""
AgroVision - Data Loader
Loads and validates the tea mandi price/arrival CSV data.
"""

import os
import sys
import pandas as pd

from src.config import RAW_DATA_FILE, COLUMNS


def load_data(filepath=None):
    """
    Load the tea mandi data CSV.

    Args:
        filepath: Path to CSV file. Defaults to RAW_DATA_FILE from config.

    Returns:
        pd.DataFrame with tea mandi data
    """
    filepath = filepath or RAW_DATA_FILE

    if not os.path.exists(filepath):
        print(f"[ERROR] Data file not found: {filepath}")
        print(f"   -> Place your CSV file in the data/raw/ folder")
        sys.exit(1)

    print(f"[LOAD] Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   Rows: {len(df):,} | Columns: {list(df.columns)}")

    # Validate expected columns
    expected = set(COLUMNS.values())
    actual = set(df.columns)
    missing = expected - actual

    if missing:
        print(f"[WARN] Missing columns: {sorted(missing)}")
        print(f"   Found: {sorted(actual)}")
    else:
        print(f"   [OK] All expected columns present")

    return df


def load_all_data():
    """
    Load all available data sources.
    Returns dict with 'main' key containing the DataFrame.
    """
    print("=" * 60)
    print("STEP 1: LOADING RAW DATA")
    print("=" * 60)

    data = {
        "main": load_data(),
    }

    print(f"\n[OK] Data loading complete. {len(data['main']):,} rows loaded.")
    return data
