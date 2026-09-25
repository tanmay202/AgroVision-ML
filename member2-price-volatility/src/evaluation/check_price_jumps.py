"""
AgroVision — Temporal Price Jump Check

Finds unusually large consecutive price changes within
each Market + Variety time series.

This script DOES NOT modify the dataset.
It only reports suspicious observations for review.
"""

import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    cleaned_data_path,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Flag changes of 100% or more.
# Example:
#   1000 -> 2000 = +100%
#   2000 -> 1000 = -50%  (not flagged)
#
# We use this only as a diagnostic threshold for now.
JUMP_THRESHOLD = 1.0


def main():

    path = cleaned_data_path()

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    df = pd.read_csv(path)

    print("=" * 60)
    print("TEMPORAL PRICE JUMP CHECK")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required = (
        GROUP_COLUMNS
        + [
            DATE_COLUMN,
            PRICE_COLUMN,
        ]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        print(
            "ERROR: Missing columns:"
        )

        for column in missing:
            print(f"   - {column}")

        return

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df = df.copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            DATE_COLUMN,
            PRICE_COLUMN,
        ]
    )

    df = df.sort_values(
        GROUP_COLUMNS + [DATE_COLUMN]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Previous price within each group
    # --------------------------------------------------------

    df["previous_price"] = (
        df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]
        .shift(1)
    )

    df["price_change_pct"] = (
        (
            df[PRICE_COLUMN]
            - df["previous_price"]
        )
        / df["previous_price"]
    )

    df["abs_price_change_pct"] = (
        df["price_change_pct"].abs()
    )

    # --------------------------------------------------------
    # Find suspicious jumps
    # --------------------------------------------------------

    suspicious = df[
        df["abs_price_change_pct"]
        >= JUMP_THRESHOLD
    ].copy()

    print(
        f"\nThreshold: "
        f"{JUMP_THRESHOLD * 100:.0f}% absolute change"
    )

    print(
        f"Suspicious rows: "
        f"{len(suspicious)}"
    )

    if len(suspicious) == 0:

        print(
            "\nNo extreme consecutive price jumps found."
        )

        return

    # --------------------------------------------------------
    # Display suspicious rows
    # --------------------------------------------------------

    columns = [
        DATE_COLUMN,
        *GROUP_COLUMNS,
        "previous_price",
        PRICE_COLUMN,
        "price_change_pct",
    ]

    suspicious = suspicious[
        columns
    ].sort_values(
        "price_change_pct"
    )

    print("\n" + "=" * 60)
    print("SUSPICIOUS PRICE JUMPS")
    print("=" * 60)

    print(
        suspicious.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(
        f"Large downward jumps: "
        f"{(
            suspicious['price_change_pct'] <= -1
        ).sum()}"
    )

    print(
        f"Large upward jumps: "
        f"{(
            suspicious['price_change_pct'] >= 1
        ).sum()}"
    )

    print(
        "\nNOTE:"
    )

    print(
        "These rows have NOT been removed."
    )

    print(
        "Review them before modifying the training data."
    )


if __name__ == "__main__":
    main()