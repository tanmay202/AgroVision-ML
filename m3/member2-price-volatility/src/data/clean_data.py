"""
AgroVision — Data Cleaning

Cleans raw mandi CSV data for Member 2 (Price + Volatility):

1. Parse dates
2. Convert numeric columns
3. Remove invalid dates
4. Remove exact duplicate rows
5. Remove negative arrivals
6. Remove rows with invalid modal price
7. Treat zero/negative Min/Max prices as missing
8. Remove genuinely inconsistent Min/Max/Modal relationships
9. Remove duplicate Market + Variety + Date observations
10. Sort chronologically within each Market + Variety group
11. Report extreme price jumps without deleting them
12. Save cleaned dataset
"""

import sys
from pathlib import Path

# Allow importing config.py from member2-price-volatility/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    NUMERIC_COLUMNS,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    raw_data_path,
    cleaned_data_path,
)


def clean(df=None, commodity=None):
    """
    Clean a raw mandi DataFrame.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Raw data. If None, reads the raw CSV for the current commodity.

    commodity : str, optional
        Commodity name override.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """

    # --------------------------------------------------
    # 0. Set commodity if provided
    # --------------------------------------------------
    if commodity:
        from config import set_commodity
        set_commodity(commodity)

    raw_path = raw_data_path()

    # --------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------
    if df is None:
        if not raw_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at: {raw_path}"
            )

        df = pd.read_csv(raw_path)

    # Make a copy so the original DataFrame is not modified
    df = df.copy()

    print("=" * 60)
    print("STEP 1: DATA CLEANING")
    print("=" * 60)

    initial_rows = len(df)
    print(f"   Initial rows: {initial_rows}")

    # --------------------------------------------------
    # 2. Validate required columns
    # --------------------------------------------------
    required_columns = [
        DATE_COLUMN,
        PRICE_COLUMN,
        ARRIVAL_COLUMN,
        MIN_PRICE_COLUMN,
        MAX_PRICE_COLUMN,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------
    # 3. Parse dates
    # --------------------------------------------------
    print("\n   [1/10] Parsing dates...")

    original_dates = df[DATE_COLUMN].copy()

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        format="%d %b %Y",
        errors="coerce",
    )

    # Fallback for unexpected date formats
    if df[DATE_COLUMN].isna().sum() > len(df) * 0.5:
        print(
            "   [INFO] Explicit date format failed for >50% "
            "of rows. Trying automatic date detection..."
        )

        df[DATE_COLUMN] = pd.to_datetime(
            original_dates,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )

    invalid_dates = int(df[DATE_COLUMN].isna().sum())

    print(f"   Invalid dates: {invalid_dates}")

    # --------------------------------------------------
    # 4. Convert numeric columns
    # --------------------------------------------------
    print("\n   [2/10] Converting numeric columns...")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------
    # 5. Remove invalid dates
    # --------------------------------------------------
    print("\n   [3/10] Removing invalid dates...")

    before = len(df)

    df = df.dropna(
        subset=[DATE_COLUMN]
    )

    print(
        f"   Dropped {before - len(df)} rows "
        f"with invalid dates"
    )

    # --------------------------------------------------
    # 6. Remove exact duplicate rows
    # --------------------------------------------------
    print("\n   [4/10] Removing exact duplicate rows...")

    before = len(df)

    df = df.drop_duplicates()

    print(
        f"   Dropped {before - len(df)} "
        f"exact duplicate rows"
    )

    # --------------------------------------------------
    # 7. Remove negative arrivals
    # --------------------------------------------------
    print("\n   [5/10] Validating arrivals...")

    before = len(df)

    df = df[
        df[ARRIVAL_COLUMN].notna()
        & (df[ARRIVAL_COLUMN] >= 0)
    ].copy()

    print(
        f"   Dropped {before - len(df)} rows "
        f"with invalid/negative arrivals"
    )

    # --------------------------------------------------
    # 8. Modal price must be positive
    # --------------------------------------------------
    print("\n   [6/10] Validating modal price...")

    before = len(df)

    df = df[
        df[PRICE_COLUMN].notna()
        & (df[PRICE_COLUMN] > 0)
    ].copy()

    print(
        f"   Dropped {before - len(df)} rows "
        f"with invalid modal price"
    )

    # --------------------------------------------------
    # 9. Treat zero/negative Min and Max as missing
    # --------------------------------------------------
    print(
        "\n   [7/10] Handling missing Min/Max prices..."
    )

    min_missing = (
        df[MIN_PRICE_COLUMN] <= 0
    ).sum()

    max_missing = (
        df[MAX_PRICE_COLUMN] <= 0
    ).sum()

    print(
        f"   Min price <= 0 treated as missing: "
        f"{int(min_missing)}"
    )

    print(
        f"   Max price <= 0 treated as missing: "
        f"{int(max_missing)}"
    )

    df.loc[
        df[MIN_PRICE_COLUMN] <= 0,
        MIN_PRICE_COLUMN,
    ] = pd.NA

    df.loc[
        df[MAX_PRICE_COLUMN] <= 0,
        MAX_PRICE_COLUMN,
    ] = pd.NA

    # --------------------------------------------------
    # 10. Validate Min / Max / Modal relationships
    #
    # Only validate the relationship when BOTH Min
    # and Max are available.
    #
    # Examples:
    # Min=0, Max=0, Modal=1950
    # -> KEEP
    #
    # Min=2500, Max=2400, Modal=2450
    # -> REMOVE
    #
    # Min=2500, Max=3000, Modal=3500
    # -> REMOVE
    # --------------------------------------------------
    print(
        "\n   [8/10] Checking price consistency..."
    )

    has_valid_range = (
        df[MIN_PRICE_COLUMN].notna()
        & df[MAX_PRICE_COLUMN].notna()
    )

    inconsistent_price_mask = (
        has_valid_range
        & (
            (df[MIN_PRICE_COLUMN] > df[MAX_PRICE_COLUMN])
            |
            (df[PRICE_COLUMN] < df[MIN_PRICE_COLUMN])
            |
            (df[PRICE_COLUMN] > df[MAX_PRICE_COLUMN])
        )
    )

    inconsistent_count = int(
        inconsistent_price_mask.sum()
    )

    print(
        f"   Inconsistent Min/Max/Modal rows: "
        f"{inconsistent_count}"
    )

    before = len(df)

    df = df[
        ~inconsistent_price_mask
    ].copy()

    print(
        f"   Dropped {before - len(df)} "
        f"structurally inconsistent price rows"
    )

    # --------------------------------------------------
    # 11. Remove duplicate observations for the same
    #     Market + Variety + Date
    # --------------------------------------------------
    print(
        "\n   [9/10] Removing duplicate "
        "Market + Variety + Date observations..."
    )

    group_cols = [
        column
        for column in GROUP_COLUMNS
        if column in df.columns
    ]

    duplicate_key_columns = (
        group_cols + [DATE_COLUMN]
    )

    before = len(df)

    df = df.drop_duplicates(
        subset=duplicate_key_columns,
        keep="first",
    ).copy()

    print(
        f"   Dropped {before - len(df)} "
        f"duplicate Market + Variety + Date observations"
    )

    # --------------------------------------------------
    # 12. Sort chronologically
    # --------------------------------------------------
    sort_columns = (
        group_cols + [DATE_COLUMN]
    )

    df = (
        df.sort_values(
            by=sort_columns
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 13. Report extreme price jumps
    #
    # IMPORTANT:
    # These are NOT deleted.
    #
    # Large jumps may represent real volatility.
    # They will be investigated by the volatility
    # pipeline.
    # --------------------------------------------------
    print(
        "\n   [10/10] Checking extreme price jumps..."
    )

    extreme_count = 0

    if group_cols:

        previous_price = (
            df.groupby(group_cols)[PRICE_COLUMN]
            .shift(1)
        )

        price_change_pct = (
            (
                df[PRICE_COLUMN]
                - previous_price
            )
            / previous_price
        ).abs() * 100

        extreme_mask = (
            previous_price.notna()
            & (price_change_pct > 300)
        )

        extreme_count = int(
            extreme_mask.sum()
        )

        print(
            f"   Extreme price jumps >300%: "
            f"{extreme_count}"
        )

        print(
            "   [INFO] Extreme jumps are FLAGGED only "
            "and are NOT removed."
        )

    # --------------------------------------------------
    # 14. Final cleanup
    # --------------------------------------------------
    df = df.reset_index(drop=True)

    # --------------------------------------------------
    # 15. Final report
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

    print(f"   Initial rows : {initial_rows}")
    print(f"   Final rows   : {len(df)}")

    print(
        f"   Date range   : "
        f"{df[DATE_COLUMN].min()} "
        f"to "
        f"{df[DATE_COLUMN].max()}"
    )

    print(
        f"   Markets      : "
        f"{df['Market Name'].nunique()}"
        if "Market Name" in df.columns
        else ""
    )

    print(
        f"   Varieties    : "
        f"{df['Variety'].nunique()}"
        if "Variety" in df.columns
        else ""
    )

    print(
        f"   Extreme jumps >300%: "
        f"{extreme_count}"
    )

    # --------------------------------------------------
    # 16. Save cleaned dataset
    # --------------------------------------------------
    out_path = cleaned_data_path()

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out_path,
        index=False,
    )

    print(
        f"\n   Saved cleaned dataset to:"
        f"\n   {out_path}"
    )

    return df


def main():
    """Run cleaning for the currently configured commodity."""
    clean()


if __name__ == "__main__":
    main()
