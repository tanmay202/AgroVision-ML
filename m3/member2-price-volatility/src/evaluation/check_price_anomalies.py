import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    CLEANED_DATA_PATH,
    PRICE_COLUMNS,
    PRICE_COLUMN,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
)


def main():
    if not CLEANED_DATA_PATH.exists():
        print(f"ERROR: File not found: {CLEANED_DATA_PATH}")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)

    print("=" * 60)
    print("PRICE ANOMALY CHECK")
    print("=" * 60)

    for column in PRICE_COLUMNS:
        print(f"\n{column}")

        print(f"Minimum: {df[column].min()}")
        print(f"Maximum: {df[column].max()}")
        print(f"Median : {df[column].median()}")

    # --------------------------------------------------
    # Look for unusually large prices
    # --------------------------------------------------
    threshold = 100000

    suspicious = df[
        df[PRICE_COLUMN] > threshold
    ].copy()

    print("\n" + "=" * 60)
    print(f"MODAL PRICE > {threshold}")
    print("=" * 60)

    print(f"Rows found: {len(suspicious)}")

    if len(suspicious) > 0:
        print(
            suspicious[
                [
                    "State Name",
                    "District Name",
                    "Market Name",
                    "Variety",
                    MIN_PRICE_COLUMN,
                    MAX_PRICE_COLUMN,
                    PRICE_COLUMN,
                    "Reported Date",
                ]
            ].to_string(index=False)
        )

    # --------------------------------------------------
    # Check price consistency again
    # --------------------------------------------------
    inconsistent = df[
        (df[PRICE_COLUMN] < df[MIN_PRICE_COLUMN])
        | (df[PRICE_COLUMN] > df[MAX_PRICE_COLUMN])
    ]

    print("\n" + "=" * 60)
    print("PRICE CONSISTENCY CHECK")
    print("=" * 60)

    print(
        "Rows with Modal Price outside Min/Max:",
        len(inconsistent)
    )


if __name__ == "__main__":
    main()