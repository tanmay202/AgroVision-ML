"""
AgroVision Member 2 — Volatility V3 Target Definitions & Temporal Purging

Defines calendar-horizon targets for V3 price volatility prediction:
- Target 7D: 7-calendar-day forward price-normalized percentage range
- Target 14D: 14-calendar-day forward price-normalized percentage range
- Symmetric historical calendar persistence baselines (past 7d and past 14d)
- Exact calendar horizon bounds (max offset strictly bounded by 7 and 14 days)
- Horizon-aware temporal boundary purging with zero lookahead contamination
- Strict training-only threshold derivation (zero-move cutoff + median non-zero split)
- Graceful handling of irregular trading days and insufficient future history
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Standard volatility classification labels
VOLATILITY_CLASSES = ["LOW", "MEDIUM", "HIGH"]
VOLATILITY_LABEL_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
VOLATILITY_REVERSE_LABEL_MAP = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

# Default zero cutoff threshold (distinguishes flat/zero movement from active trading)
ZERO_MOVE_CUTOFF = 0.014286

# Pre-computed training thresholds (< 2023-01-27) for reference & verification
V3_7D_THRESHOLDS = {
    "low_medium_threshold": 0.014286,
    "medium_high_threshold": 2.083333,
    "horizon_days": 7,
    "method": "zero-aware training split (< 2023-01-27 pre-holdout)",
}

V3_14D_THRESHOLDS = {
    "low_medium_threshold": 0.014286,
    "medium_high_threshold": 2.666666,
    "horizon_days": 14,
    "method": "zero-aware training split (< 2023-01-27 pre-holdout)",
}


def classify_volatility_series(
    series: pd.Series,
    low_thresh: float,
    high_thresh: float,
    classes: Optional[List[str]] = None,
) -> pd.Series:
    """
    Classifies a continuous percentage range into discrete classes:
      - LOW: pct_range <= low_thresh
      - MEDIUM: low_thresh < pct_range <= high_thresh
      - HIGH: pct_range > high_thresh
      - NaN if input is NaN
    """
    if classes is None:
        classes = VOLATILITY_CLASSES

    result = pd.Series(index=series.index, dtype=object)
    valid_mask = series.notna()

    c_low, c_med, c_high = classes[0], classes[1], classes[2]
    val = series[valid_mask]

    res_valid = pd.Series(index=val.index, dtype=object)
    res_valid[val <= low_thresh] = c_low
    res_valid[(val > low_thresh) & (val <= high_thresh)] = c_med
    res_valid[val > high_thresh] = c_high

    result[valid_mask] = res_valid
    return result


def derive_thresholds_from_training(
    train_pct_series: pd.Series,
    low_cutoff: float = ZERO_MOVE_CUTOFF,
) -> Dict[str, float]:
    """
    Derives LOW/MEDIUM and MEDIUM/HIGH thresholds strictly from training data:
    1. LOW threshold: fixed zero/near-zero cutoff (default 0.014286%).
    2. HIGH threshold: median of strictly non-zero percentage moves (> low_cutoff)
       in the training distribution.
    """
    valid = train_pct_series.dropna()
    non_zero = valid[valid > low_cutoff]
    if len(non_zero) == 0:
        raise ValueError("Cannot derive thresholds: No non-zero moves in training series.")

    high_thresh = float(non_zero.median())
    return {
        "low_medium_threshold": float(low_cutoff),
        "medium_high_threshold": high_thresh,
        "n_training_samples": int(len(valid)),
        "zero_percentage": float((valid <= low_cutoff).mean() * 100.0),
        "non_zero_median": high_thresh,
    }


def compute_v3_calendar_targets(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    price_col: str,
    horizons: List[int] = [7, 14],
    min_future_obs: int = 1,
    thresholds_dict: Optional[Dict[int, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Computes strict calendar-horizon targets and temporal dates for V3.

    Forecasting Semantics & Design Invariants:
    1. Exact Observation Window:
       For origin row i at date t, the forward window is [t, t + H calendar days].
       Price excursions are measured over all trading observations within [t, t + H].
       The base price P_t anchors the excursion:
         forward_pct_range = (max(P_window) - min(P_window)) / P_t * 100.0
    2. Horizon Bounding:
       Unlike V2 where .shift(-5) reached up to 86 days in holdout and 13.8 years in train,
       the observation date of any future trade is strictly bounded by t + H calendar days.
       The exact date of the latest observation used in the target is recorded in
       `target_date_{H}d` for temporal boundary purging.
    3. Missing Trading Days:
       Mandis trade irregularly (weekends closed, holidays, seasonal variations).
       Calendar-day windowing naturally collects all actual trading days that occurred
       in [t, t + H].
    4. Insufficient Future History:
       If zero future trading days occurred in (t, t + H], the forward volatility cannot
       be observed (e.g. mandi closed for season). Such rows receive NaN.
       `min_future_obs` parameter enforces the required minimum number of future trades
       in (t, t + H] (default 1).
    5. Symmetric Historical Persistence:
       Persistence over the past H calendar days [t - H, t]:
         past_pct_range_{H}d = (max(P_past_window) - min(P_past_window)) / P_t * 100.0
       This ensures mathematical and temporal symmetry between what is predicted
       and what is observed historically.
    """
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    df_out[price_col] = pd.to_numeric(df_out[price_col], errors="coerce")

    # Sort strictly by groups and chronological date
    df_out = df_out.sort_values(group_cols + [date_col]).reset_index(drop=True)

    n_rows = len(df_out)
    dates_int = df_out[date_col].values.astype("datetime64[D]").astype(np.int64)
    raw_dates = df_out[date_col].values
    prices = df_out[price_col].values.astype(np.float64)

    # Identify contiguous group boundaries
    group_series = df_out[group_cols[0]].astype(str)
    for c in group_cols[1:]:
        group_series = group_series + "___" + df_out[c].astype(str)
    group_keys = group_series.to_numpy()

    diff = np.where(group_keys[:-1] != group_keys[1:])[0] + 1
    group_starts = np.r_[0, diff]
    group_ends = np.r_[diff, n_rows]

    # Preallocate output arrays for each horizon
    target_data = {}
    for H in horizons:
        h_str = f"{H}d"
        target_data[f"future_pct_range_{h_str}"] = np.full(n_rows, np.nan)
        target_data[f"future_target_date_{h_str}"] = np.full(n_rows, np.datetime64("NaT"), dtype="datetime64[ns]")
        target_data[f"future_obs_count_{h_str}"] = np.zeros(n_rows, dtype=np.int32)
        target_data[f"past_pct_range_{h_str}"] = np.full(n_rows, np.nan)
        target_data[f"past_obs_count_{h_str}"] = np.zeros(n_rows, dtype=np.int32)

    # Process each market x variety group
    for g_s, g_e in zip(group_starts, group_ends):
        g_dates = dates_int[g_s:g_e]
        g_prices = prices[g_s:g_e]
        g_raw = raw_dates[g_s:g_e]
        n_group = g_e - g_s

        for H in horizons:
            h_str = f"{H}d"
            # Forward calendar bounds: [i, end_H) where dates <= date[i] + H
            end_H = np.searchsorted(g_dates, g_dates + H, side="right")
            # Past calendar bounds: [start_H, i] where dates >= date[i] - H
            start_H = np.searchsorted(g_dates, g_dates - H, side="left")

            for i in range(n_group):
                pt = g_prices[i]
                if np.isnan(pt) or pt <= 0:
                    continue

                idx_global = g_s + i

                # Forward window: observations from i+1 to end_H[i]
                future_slice_prices = g_prices[i+1:end_H[i]]
                valid_f_mask = (~np.isnan(future_slice_prices)) & (future_slice_prices > 0)
                n_future_obs = int(np.sum(valid_f_mask))
                target_data[f"future_obs_count_{h_str}"][idx_global] = n_future_obs

                if n_future_obs >= min_future_obs:
                    valid_f_prices = future_slice_prices[valid_f_mask]
                    all_f_prices = np.r_[pt, valid_f_prices]
                    p_max = np.max(all_f_prices)
                    p_min = np.min(all_f_prices)
                    target_data[f"future_pct_range_{h_str}"][idx_global] = ((p_max - p_min) / (pt + 1e-4)) * 100.0
                    future_slice_dates = g_raw[i+1:end_H[i]]
                    target_data[f"future_target_date_{h_str}"][idx_global] = future_slice_dates[valid_f_mask][-1]

                # Historical persistence window: observations from start_H[i] to i
                past_slice_prices = g_prices[start_H[i]:i]
                valid_p_mask = (~np.isnan(past_slice_prices)) & (past_slice_prices > 0)
                n_past_obs = int(np.sum(valid_p_mask))
                target_data[f"past_obs_count_{h_str}"][idx_global] = n_past_obs

                if n_past_obs >= 1:
                    valid_p_prices = past_slice_prices[valid_p_mask]
                    all_p_prices = np.r_[valid_p_prices, pt]
                    p_p_max = np.max(all_p_prices)
                    p_p_min = np.min(all_p_prices)
                    target_data[f"past_pct_range_{h_str}"][idx_global] = ((p_p_max - p_p_min) / (pt + 1e-4)) * 100.0

    # Add columns to output dataframe
    for col_name, arr in target_data.items():
        df_out[col_name] = arr

    # Also compute legacy past 5-observation persistence for baseline comparison
    grouped_p = df_out.groupby(group_cols, sort=False)[price_col]
    p5_max = grouped_p.transform(lambda s: s.rolling(5).max())
    p5_min = grouped_p.transform(lambda s: s.rolling(5).min())
    df_out["past_5_pct_range"] = ((p5_max - p5_min) / (df_out[price_col] + 1e-4)) * 100.0

    # Classify targets and persistence
    for H in horizons:
        h_str = f"{H}d"
        # Determine thresholds
        if thresholds_dict and H in thresholds_dict:
            th = thresholds_dict[H]
            l_th = th["low_medium_threshold"]
            h_th = th["medium_high_threshold"]
        elif H == 7:
            l_th, h_th = V3_7D_THRESHOLDS["low_medium_threshold"], V3_7D_THRESHOLDS["medium_high_threshold"]
        elif H == 14:
            l_th, h_th = V3_14D_THRESHOLDS["low_medium_threshold"], V3_14D_THRESHOLDS["medium_high_threshold"]
        else:
            l_th, h_th = ZERO_MOVE_CUTOFF, 2.0

        # Discrete classification
        df_out[f"target_{h_str}"] = classify_volatility_series(
            df_out[f"future_pct_range_{h_str}"], l_th, h_th, VOLATILITY_CLASSES
        )
        df_out[f"persistence_pred_{h_str}"] = classify_volatility_series(
            df_out[f"past_pct_range_{h_str}"], l_th, h_th, VOLATILITY_CLASSES
        )
        df_out[f"persistence_pred_5obs_{h_str}"] = classify_volatility_series(
            df_out["past_5_pct_range"], l_th, h_th, VOLATILITY_CLASSES
        )

    return df_out


def apply_v3_temporal_purge(
    df: pd.DataFrame,
    date_col: str,
    target_date_col: str,
    split_date: str,
) -> Tuple[pd.Series, pd.Series, int]:
    """
    Applies strict chronological boundary purging for V3:
    1. is_train_cand: rows with Reported Date < split_date.
    2. is_eval: rows with Reported Date >= split_date.
    3. leaks: train candidate rows whose future target date reaches into or across split_date
       (i.e. target_date >= split_date).
    4. clean_train: train candidate rows that do NOT leak into eval window.

    Returns:
      (clean_train_mask, is_eval_mask, purge_count)
    """
    split_ts = pd.Timestamp(split_date)
    if split_ts.tz is not None:
        split_ts = split_ts.tz_localize(None)

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    target_dates = pd.to_datetime(df[target_date_col], errors="coerce")
    if target_dates.dt.tz is not None:
        target_dates = target_dates.dt.tz_localize(None)

    is_train_cand = dates < split_ts
    is_eval = dates >= split_ts

    # Check for target contamination (only valid target dates >= split_ts can leak)
    leaks = is_train_cand & target_dates.notna() & (target_dates >= split_ts)
    clean_train = is_train_cand & (~leaks)
    purge_count = int(leaks.sum())

    return clean_train, is_eval, purge_count
