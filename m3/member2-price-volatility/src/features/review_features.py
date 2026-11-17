import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    FEATURES_FINAL_PATH,
    PRICE_TARGET,
)


def main():
    if not FEATURES_FINAL_PATH.exists():
        print(f"ERROR: File not found: {FEATURES_FINAL_PATH}")
        return

    df = pd.read_csv(FEATURES_FINAL_PATH)

    excluded_columns = [
        PRICE_TARGET,
        "Reported Date",
        "State Name",
        "District Name",
        "Market Name",
        "Variety",
        "Group",
        "days_to_next",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    print("=" * 60)
    print("FINAL FEATURE REVIEW")
    print("=" * 60)

    print(f"Rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Number of candidate ML features: {len(feature_columns)}")

    print("\nCandidate ML features:")
    for i, column in enumerate(feature_columns, start=1):
        print(f"{i:2}. {column}")

    print("\nTarget:")
    print(PRICE_TARGET)

    print("\nFeature missing values:")
    print(df[feature_columns].isna().sum())

    print("\nTarget missing values:")
    print(df[PRICE_TARGET].isna().sum())


if __name__ == "__main__":
    main()