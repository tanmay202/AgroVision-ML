"""
AgroVision — Volatility Target Creation

Computes volatility labels (LOW/MEDIUM/HIGH) from
absolute price percentage changes.

Thresholds are data-driven (tertile quantiles of
non-zero movements in the training set).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from config import (
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    ARTIFACTS_DIR,
    train_path,
    volatility_train_path,
    volatility_config_path,
)


PRICE_CHANGE_COLUMN = "price_pct_change"


def create_volatility(df=None):
    """
    Create volatility labels for the training set.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Training data. If None, reads from train CSV.

    Returns
    -------
    tuple of (pd.DataFrame, float, float)
        (training data with volatility column, low_threshold, high_threshold)
    """
    if df is None:
        path = train_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 5: VOLATILITY TARGET")
    print("=" * 60)

    if PRICE_CHANGE_COLUMN not in df.columns:
        raise ValueError(
            f"Column '{PRICE_CHANGE_COLUMN}' not found. "
            f"Run pct_change features first."
        )

    # Absolute price movement
    df["abs_price_pct_change"] = df[PRICE_CHANGE_COLUMN].abs()

    # Data-driven thresholds from non-zero movements
    non_zero = df.loc[
        df["abs_price_pct_change"] > 0, "abs_price_pct_change"
    ].dropna()

    low_threshold = float(non_zero.quantile(1 / 3))
    high_threshold = float(non_zero.quantile(2 / 3))

    print(f"   Low/Medium threshold : {low_threshold:.4f}%")
    print(f"   Medium/High threshold: {high_threshold:.4f}%")
    print(f"   Zero-change rows: {(df['abs_price_pct_change'] == 0).sum()}")
    print(f"   Non-zero rows: {len(non_zero)}")

    # Save thresholds
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    vol_config = {
        "threshold_method": (
            "training-data quantiles of non-zero "
            "absolute price percentage changes"
        ),
        "low_medium_threshold_percent": round(low_threshold, 4),
        "medium_high_threshold_percent": round(high_threshold, 4),
        "classes": VOLATILITY_CLASSES,
    }
    config_path = volatility_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vol_config, f, indent=4)
    print(f"   Thresholds saved to: {config_path}")

    # Classify
    def classify(value):
        if pd.isna(value):
            return pd.NA
        if value == 0 or value <= low_threshold:
            return VOLATILITY_CLASSES[0]  # LOW
        if value <= high_threshold:
            return VOLATILITY_CLASSES[1]  # MEDIUM
        return VOLATILITY_CLASSES[2]  # HIGH

    df[VOLATILITY_TARGET] = df["abs_price_pct_change"].apply(classify)

    # Distribution
    print(f"\n   Class distribution:")
    print(df[VOLATILITY_TARGET].value_counts().to_string())

    # Save
    out_path = volatility_train_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n   Saved to: {out_path}")

    return df, low_threshold, high_threshold


def main():
    create_volatility()


if __name__ == "__main__":
    main()