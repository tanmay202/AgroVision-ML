"""
AgroVision Member 2 — Volatility Target Definitions & Temporal Purging

Defines targets for both Experiment A and Experiment B:
- Experiment A: Preserved Nominal 5-Observation Range
- Experiment B: Price-Normalized 5-Observation Percentage Range
- Training-only threshold derivation
- Temporal purging based on 5-observation forward horizon
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add src to path for shared constants if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import VOLATILITY_CLASSES, VOLATILITY_LABEL_MAP, VOLATILITY_REVERSE_LABEL_MAP
from volatility_utils import compute_thresholds, classify_volatility


# Static Experiment A thresholds
EXP_A_LOW_THRESH = 0.5
EXP_A_HIGH_THRESH = 50.0

# Experiment B training-derived thresholds
EXP_B_LOW_THRESH = 0.014286   # zero cutoff
EXP_B_HIGH_THRESH = 1.960784  # median non-zero percentage range in pre-2021 training set


def compute_forward_volatility_targets(df, group_cols, date_col, price_col, window=5):
    """
    Computes both nominal forward range and normalized forward percentage range.
    Also calculates the observation date of the window's final future record for purging.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    grouped = df.groupby(group_cols)[price_col]

    # Collect next W future prices
    future_cols = []
    for i in range(1, window + 1):
        col_name = f"_future_price_{i}"
        df[col_name] = grouped.shift(-i)
        future_cols.append(col_name)

    future_prices = df[future_cols]
    valid_window = future_prices.notna().all(axis=1)

    # 1. Target A: Nominal Forward Price Range
    df["future_nominal_range"] = np.nan
    df.loc[valid_window, "future_nominal_range"] = (
        future_prices.loc[valid_window].max(axis=1).values
        - future_prices.loc[valid_window].min(axis=1).values
    )
    df["target_A"] = classify_volatility(
        df["future_nominal_range"],
        EXP_A_LOW_THRESH,
        EXP_A_HIGH_THRESH,
        VOLATILITY_CLASSES
    )

    # 2. Target B: Price-Normalized Forward Percentage Range
    df["future_pct_range"] = np.nan
    df.loc[valid_window, "future_pct_range"] = (
        (df.loc[valid_window, "future_nominal_range"] / df.loc[valid_window, price_col]) * 100.0
    )
    df["target_B"] = classify_volatility(
        df["future_pct_range"],
        EXP_B_LOW_THRESH,
        EXP_B_HIGH_THRESH,
        VOLATILITY_CLASSES
    )

    # 3. Observation date of 5th future price for exact temporal purging
    grouped_date = df.groupby(group_cols)[date_col]
    df["future_target_date"] = grouped_date.shift(-window)

    # Drop temporary columns
    df = df.drop(columns=future_cols)

    return df


def compute_persistence_predictions(df, group_cols, price_col):
    """
    Computes persistence predictions for both Experiment A and Experiment B
    based on the past 5 observations [t-4..t].
    """
    df = df.copy()
    grouped = df.groupby(group_cols)[price_col]

    p5_max = grouped.transform(lambda s: s.rolling(5).max())
    p5_min = grouped.transform(lambda s: s.rolling(5).min())
    past_5_range = p5_max - p5_min
    past_5_pct_range = (past_5_range / (df[price_col] + 1e-4)) * 100.0

    df["persistence_pred_A"] = classify_volatility(
        past_5_range,
        EXP_A_LOW_THRESH,
        EXP_A_HIGH_THRESH,
        VOLATILITY_CLASSES
    )
    df["persistence_pred_B"] = classify_volatility(
        past_5_pct_range,
        EXP_B_LOW_THRESH,
        EXP_B_HIGH_THRESH,
        VOLATILITY_CLASSES
    )

    return df


def apply_temporal_purge(df, date_col, val_start_date):
    """
    Returns boolean mask (train_mask, val_mask, purge_count)
    where train rows whose 5th forward observation date extends into or past
    val_start_date are purged.
    """
    val_start = pd.Timestamp(val_start_date)
    is_train_cand = df[date_col] < val_start
    is_val = df[date_col] >= val_start

    # If future_target_date is on or after val_start, target leaks into evaluation window
    leaks = is_train_cand & (df["future_target_date"] >= val_start)
    clean_train = is_train_cand & (~leaks)
    purge_count = int(leaks.sum())

    return clean_train, is_val, purge_count
