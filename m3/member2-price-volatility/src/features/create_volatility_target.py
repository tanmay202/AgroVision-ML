"""
AgroVision — Forward Volatility Target Creation

Creates a FUTURE volatility target (LOW/MEDIUM/HIGH).
Target = price range (max - min) of the NEXT 5 prices within each
Market + Variety group. Thresholds come ONLY from the training data.

This replaces the old "std of next 3 pct_changes" which collapsed
to only 2 classes because 83% of consecutive prices are identical.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import pandas as pd

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    ARTIFACTS_DIR,
    train_path,
    volatility_train_path,
    volatility_config_path,
)
from volatility_utils import (
    VOLATILITY_WINDOW,
    add_future_volatility,
    compute_thresholds,
    classify_volatility,
    print_vol_diagnostics,
)


def create_volatility(df=None):
    """
    Returns
    -------
    tuple (training dataframe, low_threshold, high_threshold)
    """
    if df is None:
        path = train_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 5: FORWARD VOLATILITY TARGET")
    print("=" * 60)

    required = [DATE_COLUMN, PRICE_COLUMN] + GROUP_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Future volatility (shared with the test builder)
    df = add_future_volatility(df, GROUP_COLUMNS, DATE_COLUMN)

    print("\n   Diagnostics:")
    print_vol_diagnostics(df)

    # Thresholds from TRAINING data only
    low_threshold, high_threshold, method = compute_thresholds(
        df["future_volatility"]
    )
    n_valid = int(df["future_volatility"].notna().sum())

    print(f"\n   Future volatility window: next {VOLATILITY_WINDOW} observations")
    print(f"   Threshold method      : {method}")
    print(f"   Low/Medium threshold  : {low_threshold:.8f}")
    print(f"   Medium/High threshold : {high_threshold:.8f}")
    print(f"   Valid future-vol rows : {n_valid}")

    # Classify
    df[VOLATILITY_TARGET] = classify_volatility(
        df["future_volatility"], low_threshold, high_threshold,
        VOLATILITY_CLASSES,
    )

    counts = df[VOLATILITY_TARGET].value_counts().reindex(VOLATILITY_CLASSES)
    print("\n   Class distribution:")
    print(counts.fillna(0).astype(int).to_string())

    empty = [c for c in VOLATILITY_CLASSES if not counts.get(c, 0)]
    if empty:
        raise ValueError(f"Volatility classes with zero rows: {empty}")

    # Save thresholds (full precision - no rounding to 0)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vol_config = {
        "threshold_method": method,
        "volatility_definition": (
            f"price range (max - min) of next {VOLATILITY_WINDOW} observations"
        ),
        "volatility_window": VOLATILITY_WINDOW,
        "low_medium_threshold": low_threshold,
        "medium_high_threshold": high_threshold,
        "classes": VOLATILITY_CLASSES,
    }
    config_path = volatility_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vol_config, f, indent=4)
    print(f"\n   Thresholds saved to: {config_path}")

    out_path = volatility_train_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"   Saved to: {out_path}")

    return df, low_threshold, high_threshold


def main():
    create_volatility()


if __name__ == "__main__":
    main()