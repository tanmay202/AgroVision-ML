"""
AgroVision — Volatility Dataset Builder (Training)

Creates the final volatility training dataset by:
  - Selecting VOLATILITY_FEATURES + target
  - Keeping the date column (config.DATE_COLUMN, "Reported Date") as
    METADATA ONLY, so the trainer can do a chronological validation split.
    It is never a model feature.
  - Removing rows with missing features
  - Verifying price_pct_change is excluded (leakage protection)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    DATE_COLUMN,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    volatility_train_path,
    volatility_dataset_train_path,
)


def build_volatility_train(df=None):
    """
    Build the final volatility training dataset.

    Parameters
    ----------
    df : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
        Columns: [DATE_COLUMN] + VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    """
    if DATE_COLUMN in VOLATILITY_FEATURES:
        raise ValueError(
            f"'{DATE_COLUMN}' must not be listed in VOLATILITY_FEATURES; "
            "it is kept only as metadata for time-ordered splitting."
        )

    if df is None:
        path = volatility_train_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 6: VOLATILITY DATASET (TRAIN)")
    print("=" * 60)

    # Validate columns (date is required so the split can be chronological)
    required = [DATE_COLUMN] + VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove rows without target
    before = len(df)
    df = df.dropna(subset=[VOLATILITY_TARGET]).copy()
    print(f"   Dropped {before - len(df)} rows without volatility target")

    # Remove rows with missing features
    before = len(df)
    df = df.dropna(subset=VOLATILITY_FEATURES).copy()
    print(f"   Dropped {before - len(df)} rows with missing features")

    # Select final columns: date (metadata) first, then features, then target
    result = df[[DATE_COLUMN] + VOLATILITY_FEATURES + [VOLATILITY_TARGET]].copy()

    # Leakage check
    if "price_pct_change" in result.columns:
        raise ValueError("LEAKAGE: price_pct_change is still present!")

    dates = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    n_bad = int(dates.isna().sum())
    if n_bad:
        print(f"   WARNING: {n_bad} rows have an unparseable '{DATE_COLUMN}'")
    if dates.notna().any():
        print(f"   Date range: {dates.min().date()} -> {dates.max().date()}")

    print(f"\n   Final rows: {len(result)}")
    print(f"   Features: {len(VOLATILITY_FEATURES)} (+ '{DATE_COLUMN}' kept as metadata)")
    print(f"   Class distribution:")
    print(result[VOLATILITY_TARGET].value_counts().to_string())

    # Save
    out_path = volatility_dataset_train_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\n   Saved to: {out_path}")

    return result


def main():
    build_volatility_train()


if __name__ == "__main__":
    main()