"""
AgroVision — Date Features

Extracts calendar features from the date column.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    rolling_features_path,
    date_features_path,
)


def create_dates(df=None):
    """
    Create calendar features from the date column.
    """
    if df is None:
        path = rolling_features_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 3c: DATE FEATURES")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    df["year"] = df[DATE_COLUMN].dt.year
    df["day"] = df[DATE_COLUMN].dt.day
    df["month"] = df[DATE_COLUMN].dt.month
    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek
    df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

    date_features = ["year", "day", "month", "day_of_week", "week_of_year"]
    print(f"   Created: {', '.join(date_features)}")
    print(f"   Rows: {len(df)}")

    out_path = date_features_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df


def main():
    create_dates()


if __name__ == "__main__":
    main()