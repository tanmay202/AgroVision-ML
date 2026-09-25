import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAW_DATA_PATH, DATE_COLUMN


def main():
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {RAW_DATA_PATH}")
        return

    import pandas as pd

    df = pd.read_csv(RAW_DATA_PATH)

    print("\nDataset loaded successfully!")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nReported Date sample:")
    print(df[DATE_COLUMN].head(10))

    print("\nReported Date data type:")
    print(df[DATE_COLUMN].dtype)


if __name__ == "__main__":
    main()