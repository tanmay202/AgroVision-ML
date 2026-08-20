import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    FEATURES_FINAL_PATH,
    TRAIN_PATH,
    TEST_PATH,
    DATE_COLUMN,
)


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not FEATURES_FINAL_PATH.exists():
        print(f"ERROR: File not found: {FEATURES_FINAL_PATH}")
        return

    df = pd.read_csv(FEATURES_FINAL_PATH)

    # --------------------------------------------------
    # 2. Convert date
    # --------------------------------------------------
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------
    # 3. Sort by date
    # --------------------------------------------------
    df = df.sort_values(
        by=DATE_COLUMN
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 4. Time-based 80/20 split
    # --------------------------------------------------
    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    # --------------------------------------------------
    # 5. Display split information
    # --------------------------------------------------
    print("=" * 60)
    print("TIME-AWARE TRAIN / TEST SPLIT")
    print("=" * 60)

    print(f"Total rows : {len(df)}")
    print(f"Train rows : {len(train_df)}")
    print(f"Test rows  : {len(test_df)}")

    print("\nTRAIN PERIOD")
    print(f"Start: {train_df[DATE_COLUMN].min()}")
    print(f"End  : {train_df[DATE_COLUMN].max()}")

    print("\nTEST PERIOD")
    print(f"Start: {test_df[DATE_COLUMN].min()}")
    print(f"End  : {test_df[DATE_COLUMN].max()}")

    # --------------------------------------------------
    # 6. Leakage check
    # --------------------------------------------------
    train_end = train_df[DATE_COLUMN].max()
    test_start = test_df[DATE_COLUMN].min()

    print("\n" + "=" * 60)
    print("LEAKAGE CHECK")
    print("=" * 60)

    print(f"Latest training date : {train_end}")
    print(f"Earliest testing date: {test_start}")

    if train_end < test_start:
        print("PASS: Training data comes before test data.")
    else:
        print("WARNING: Temporal overlap detected.")

    # --------------------------------------------------
    # 7. Save split datasets
    # --------------------------------------------------
    TRAIN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_PATH,
        index=False
    )

    test_df.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nSaved:")
    print(TRAIN_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()