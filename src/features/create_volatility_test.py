"""
AgroVision — Volatility Dataset Builder (Test)

Creates the volatility test dataset using thresholds
computed from the training set (saved in volatility_config.json).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from config import (
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    test_path,
    volatility_dataset_test_path,
    volatility_config_path,
)


def build_volatility_test(df=None):
    """
    Build the final volatility test dataset.

    Parameters
    ----------
    df : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
    """
    config_path = volatility_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Volatility config not found: {config_path}\n"
            f"Run create_volatility_target.py first."
        )

    # Load thresholds from training config
    with open(config_path, "r", encoding="utf-8") as f:
        vol_config = json.load(f)

    low_threshold = vol_config["low_medium_threshold_percent"]
    high_threshold = vol_config["medium_high_threshold_percent"]

    if df is None:
        path = test_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("STEP 7: VOLATILITY DATASET (TEST)")
    print("=" * 60)

    print(f"   Thresholds: Low/Med={low_threshold}%, Med/High={high_threshold}%")

    # Classify
    if "price_pct_change" not in df.columns:
        raise ValueError(
            "price_pct_change column not found. "
            "Ensure feature engineering ran correctly."
        )

    def classify(value):
        if pd.isna(value):
            return pd.NA
        if value == 0 or value <= low_threshold:
            return VOLATILITY_CLASSES[0]
        if value <= high_threshold:
            return VOLATILITY_CLASSES[1]
        return VOLATILITY_CLASSES[2]

    df["abs_price_pct_change"] = df["price_pct_change"].abs()
    df[VOLATILITY_TARGET] = df["abs_price_pct_change"].apply(classify)

    # Remove missing
    before = len(df)
    df = df.dropna(subset=[VOLATILITY_TARGET]).copy()
    df = df.dropna(subset=VOLATILITY_FEATURES).copy()
    print(f"   Dropped {before - len(df)} rows with missing values")

    result = df[VOLATILITY_FEATURES + [VOLATILITY_TARGET]].copy()

    print(f"\n   Final rows: {len(result)}")
    print(f"   Class distribution:")
    print(result[VOLATILITY_TARGET].value_counts().to_string())

    # Save
    out_path = volatility_dataset_test_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\n   Saved to: {out_path}")

    return result


def main():
    build_volatility_test()


if __name__ == "__main__":
    main()