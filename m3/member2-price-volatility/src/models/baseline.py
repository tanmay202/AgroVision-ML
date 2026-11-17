"""
AgroVision — Baseline Price Model

Uses the previous price (lag_1) as the prediction for future price.
This is the simplest possible baseline: "next price = last known price".
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    FEATURES_FINAL_PATH,
    DATE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
    features_final_path,
)
from utils import calculate_metrics, print_metrics


def run_baseline(df=None):
    """
    Evaluate the naive baseline: predicted_price = lag_1 (previous price).

    Returns
    -------
    dict with metrics
    """
    if df is None:
        path = features_final_path()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)

    print("\n" + "=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    df = df.sort_values(by=group_cols + [DATE_COLUMN]).reset_index(drop=True)

    # Baseline: lag_1 = previous price predicts future price
    if "lag_1" not in df.columns:
        print("   ERROR: lag_1 feature not found. Run feature engineering first.")
        return None

    # Only evaluate where we have both actual and baseline
    mask = df[PRICE_TARGET].notna() & df["lag_1"].notna()
    actual = df.loc[mask, PRICE_TARGET]
    predicted = df.loc[mask, "lag_1"]

    print(f"   Strategy: predicted_future_price = lag_1 (previous price)")
    print(f"   Rows evaluated: {len(actual)}")

    metrics = calculate_metrics(actual, predicted)
    print_metrics(metrics, "Baseline (lag_1)")

    return metrics


def main():
    run_baseline()


if __name__ == "__main__":
    main()