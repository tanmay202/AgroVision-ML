"""
AgroVision — Shared Volatility Utilities

Shared functions for computing forward volatility and classifying it.
Used by both create_volatility_target.py (training) and
create_volatility_test.py (test).

Volatility Definition (v2 — price-range based):
    Forward volatility = max(next W prices) - min(next W prices)

    This replaces the old "std of next W pct_changes" which collapsed
    to zero for ~70% of rows because 83% of consecutive prices are
    identical in the tea dataset.  The price-range definition is:
      - Non-zero whenever ANY of the next W prices differs from another.
      - Intuitive: "how much could the price swing in the next W days?"
      - Robust to the zero-change problem.
"""

import numpy as np
import pandas as pd

# Kept for backward compatibility with other pipeline steps.
# NOT used for volatility computation anymore.
PRICE_CHANGE_COLUMN = "price_pct_change"

# Number of future observations used to compute forward volatility.
# Increased from 3 to 5 for robustness.
# With 83% zero-change rate, P(all 5 same) ≈ 0.83^4 ≈ 47%  (vs ~70% at W=3).
VOLATILITY_WINDOW = 5


def add_future_volatility(df, group_cols, date_col, price_col=None):
    """
    Compute forward price range as the volatility target.

    For row t:
        future_volatility = max(price[t+1..t+W]) - min(price[t+1..t+W])

    Parameters
    ----------
    df : pd.DataFrame
    group_cols : list of str
        Columns defining each time series (e.g. ["Market Name", "Variety"]).
    date_col : str
    price_col : str, optional
        Defaults to "Modal Price (Rs./Quintal)".

    Returns
    -------
    pd.DataFrame with ``future_volatility`` column added.
    """
    if price_col is None:
        price_col = "Modal Price (Rs./Quintal)"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    grouped = df.groupby(group_cols)[price_col]

    # Collect next W prices for each row
    future_cols = []
    for i in range(1, VOLATILITY_WINDOW + 1):
        col_name = f"_future_price_{i}"
        df[col_name] = grouped.shift(-i)
        future_cols.append(col_name)

    future_prices = df[future_cols]

    # Require ALL W future prices to be present (strict window)
    valid_mask = future_prices.notna().all(axis=1)

    df["future_volatility"] = np.nan
    df.loc[valid_mask, "future_volatility"] = (
        future_prices.loc[valid_mask].max(axis=1).values
        - future_prices.loc[valid_mask].min(axis=1).values
    )

    # Clean up temporary columns
    df = df.drop(columns=future_cols)

    return df


def compute_thresholds(series):
    """
    Compute LOW / MEDIUM / HIGH thresholds from forward volatility values.

    Strategy:
        1. If p33 and p66 naturally separate (no zero-mass problem),
           use standard quantiles.
        2. Otherwise (common case: many exact zeros), use a zero-aware
           split:
             LOW    = volatility == 0  (price stayed flat)
             MEDIUM = 0 < volatility <= median(non-zero values)
             HIGH   = volatility > median(non-zero values)

    Returns
    -------
    (low_threshold, high_threshold, method_description)
    """
    valid = series.dropna()
    if len(valid) == 0:
        raise ValueError("No valid forward volatility values.")

    p33 = float(valid.quantile(0.33))
    p66 = float(valid.quantile(0.66))

    # Happy path: natural separation
    if p33 > 0 and p33 < p66:
        return p33, p66, "standard quantiles (p33/p66)"

    # ---- Zero-aware split ----
    nonzero = valid[valid > 0]
    if len(nonzero) < 10:
        raise ValueError(
            f"Only {len(nonzero)} non-zero volatility values. "
            "Data may be too sparse for 3-class classification."
        )

    # LOW  = exact zero  →  use half the smallest non-zero as cutoff
    smallest_nonzero = float(nonzero.min())
    low_threshold = smallest_nonzero / 2.0

    # MEDIUM / HIGH split at the median of non-zero values
    high_threshold = float(nonzero.median())

    # Safety: ensure meaningful separation
    if low_threshold >= high_threshold:
        high_threshold = float(nonzero.quantile(0.75))

    zero_frac = (valid == 0).mean() * 100

    method = (
        f"zero-aware split "
        f"({zero_frac:.0f}% zeros -> LOW, "
        f"non-zeros split at median -> MEDIUM/HIGH)"
    )

    return low_threshold, high_threshold, method


def classify_volatility(series, low_threshold, high_threshold, classes):
    """
    Classify volatility into LOW / MEDIUM / HIGH.

    Parameters
    ----------
    series : pd.Series
        Forward volatility values.
    low_threshold : float
    high_threshold : float
    classes : list of str
        [LOW_label, MEDIUM_label, HIGH_label]

    Returns
    -------
    pd.Series of class labels (or pd.NA for missing values).
    """
    def _classify(value):
        if pd.isna(value):
            return pd.NA
        if value <= low_threshold:
            return classes[0]   # LOW
        if value <= high_threshold:
            return classes[1]   # MEDIUM
        return classes[2]       # HIGH

    return series.apply(_classify)


def print_vol_diagnostics(df):
    """Print diagnostics for the ``future_volatility`` column."""
    if "future_volatility" not in df.columns:
        print("   WARNING: 'future_volatility' column not found.")
        return

    vol = df["future_volatility"]
    valid = vol.dropna()
    n_total = len(vol)
    n_valid = len(valid)
    n_zero = int((valid == 0).sum())

    print(f"   Total rows        : {n_total}")
    print(f"   Valid volatility  : {n_valid}")
    print(f"   Missing (tail)    : {n_total - n_valid}")
    print(f"   Exact zeros       : {n_zero} ({n_zero / max(n_valid, 1) * 100:.1f}%)")

    if n_valid > 0:
        print(f"   Mean              : {valid.mean():.2f}")
        print(f"   Median            : {valid.median():.2f}")
        print(f"   Std               : {valid.std():.2f}")
        print(f"   Min               : {valid.min():.2f}")
        print(f"   Max               : {valid.max():.2f}")
        print(f"   p25               : {valid.quantile(0.25):.2f}")
        print(f"   p75               : {valid.quantile(0.75):.2f}")

    nonzero = valid[valid > 0]
    if len(nonzero) > 0:
        print(f"\n   Non-zero values   : {len(nonzero)}")
        print(f"     Mean            : {nonzero.mean():.2f}")
        print(f"     Median          : {nonzero.median():.2f}")
        print(f"     Min             : {nonzero.min():.2f}")
        print(f"     Max             : {nonzero.max():.2f}")
