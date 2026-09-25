"""
AgroVision — Time-Based Train/Test Split

Performs a per-group chronological 80/20 split:
  For each (Market, Variety) group, the first 80% of
  observations (by date) go to train, the last 20% to test.

This prevents temporal leakage across groups with
different date ranges.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    features_final_path,
    train_path,
    test_path,
)


def split(df=None, test_ratio=0.2):
    """
    Perform per-group chronological train/test split.

    Parameters
    ----------
    df : pd.DataFrame, optional
    test_ratio : float
        Fraction of each group to use for testing.

    Returns
    -------
    train_df, test_df : tuple of pd.DataFrame
    """
    if df is None:
        path = features_final_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 4: TRAIN/TEST SPLIT")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    # Per-group chronological split
    train_parts = []
    test_parts = []

    if group_cols:
        for name, group in df.groupby(group_cols):
            group = group.sort_values(DATE_COLUMN).reset_index(drop=True)
            split_idx = int(len(group) * (1 - test_ratio))
            if split_idx < 1:
                # Group too small — put all in train
                train_parts.append(group)
                continue
            train_parts.append(group.iloc[:split_idx])
            test_parts.append(group.iloc[split_idx:])
    else:
        # No group columns — global split
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
        split_idx = int(len(df) * (1 - test_ratio))
        train_parts.append(df.iloc[:split_idx])
        test_parts.append(df.iloc[split_idx:])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame()

    print(f"   Total rows : {len(df)}")
    print(f"   Train rows : {len(train_df)}")
    print(f"   Test rows  : {len(test_df)}")

    if len(train_df) > 0 and len(test_df) > 0:
        train_end = train_df[DATE_COLUMN].max()
        test_start = test_df[DATE_COLUMN].min()
        print(f"\n   Train period: {train_df[DATE_COLUMN].min()} to {train_end}")
        print(f"   Test period : {test_start} to {test_df[DATE_COLUMN].max()}")

    # Save
    t_path = train_path()
    te_path = test_path()
    t_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(t_path, index=False)
    test_df.to_csv(te_path, index=False)
    print(f"\n   Saved: {t_path}")
    print(f"   Saved: {te_path}")

    return train_df, test_df


def main():
    split()


if __name__ == "__main__":
    main()