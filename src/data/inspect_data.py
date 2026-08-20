import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import RAW_DATA_PATH, DATE_COLUMN


def main():
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {RAW_DATA_PATH}")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    # --------------------------------------------------
    # 1. Basic shape
    # --------------------------------------------------
    print("=" * 60)
    print("1. DATASET SHAPE")
    print("=" * 60)
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # --------------------------------------------------
    # 2. Column names
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("2. COLUMN NAMES")
    print("=" * 60)

    for column in df.columns:
        print(f"- {column}")

    # --------------------------------------------------
    # 3. Data types
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("3. DATA TYPES")
    print("=" * 60)
    print(df.dtypes)

    # --------------------------------------------------
    # 4. Missing values
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("4. MISSING VALUES")
    print("=" * 60)

    missing = df.isna().sum()

    print(missing)

    # --------------------------------------------------
    # 5. Duplicate rows
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("5. DUPLICATES")
    print("=" * 60)

    duplicate_count = df.duplicated().sum()

    print(f"Duplicate rows: {duplicate_count}")

    # --------------------------------------------------
    # 6. Numerical statistics
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("6. NUMERICAL STATISTICS")
    print("=" * 60)

    print(df.describe())

    # --------------------------------------------------
    # 7. Unique values
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("7. UNIQUE VALUES")
    print("=" * 60)

    categorical_columns = [
        "State Name",
        "District Name",
        "Market Name",
        "Variety",
        "Group"
    ]

    for column in categorical_columns:
        print(f"{column}: {df[column].nunique()} unique values")

    # --------------------------------------------------
    # 8. Date information
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("8. DATE INFORMATION")
    print("=" * 60)

    dates = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    print(f"Minimum date: {dates.min()}")
    print(f"Maximum date: {dates.max()}")
    print(f"Invalid dates: {dates.isna().sum()}")

    # --------------------------------------------------
    # 9. First five rows
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("9. FIRST FIVE ROWS")
    print("=" * 60)

    print(df.head())


if __name__ == "__main__":
    main()