"""
AgroVision-ML — Member 2: Direct Horizon Target Construction & Purging

Implements:
1. Method A: Symmetric Calendar Horizon Targets [T + H - 3, T + H + 3]
   - Closest observation to requested target date T + H.
   - Deterministic tie-breaking: when distances are equal, chooses the forward
     observation (T + H + delta), reflecting subsequent market conditions.
2. Method B: Observation-Shift Targets (.shift(-k))
   - Shift by observation count within sorted Market Name + Variety group.
   - Records actual elapsed calendar days to demonstrate divergence from calendar horizons.
3. Temporal Purging:
   - Purges training and validation observations whose future target dates reach across
     chronological split boundaries (e.g. >= 2021-02-01 or >= 2023-02-01).
"""

from typing import List, Optional, Tuple, Literal
import numpy as np
import pandas as pd


def compute_symmetric_calendar_horizon_targets(
    df: pd.DataFrame,
    horizon_days: int,
    max_tolerance_days: int = 3,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Reported Date",
    price_col: str = "Modal Price (Rs./Quintal)",
    tie_break: Literal["forward", "backward"] = "forward",
) -> pd.DataFrame:
    """
    Method A: Computes symmetric calendar-horizon targets for forecast origin T.
    Permitted window: [T + H - max_tolerance_days, T + H + max_tolerance_days].
    Picks the observation closest to T + H. Ties broken deterministically.
    """
    if group_cols is None:
        group_cols = ["Market Name", "Variety"]

    active_group_cols = [c for c in group_cols if c in df.columns]
    if not active_group_cols:
        raise ValueError(f"Required group columns {group_cols} not found in DataFrame.")

    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out[price_col] = pd.to_numeric(df_out[price_col], errors="coerce")

    df_out = df_out.sort_values(active_group_cols + [date_col]).reset_index(drop=True)

    h_prefix = f"{horizon_days}d"
    target_date_col = f"target_date_{h_prefix}_sym"
    future_price_col = f"future_modal_price_{h_prefix}_sym"
    target_change_col = f"future_price_change_{h_prefix}_sym"
    actual_offset_col = f"actual_offset_{h_prefix}_sym"
    offset_error_col = f"offset_error_{h_prefix}_sym"

    df_out[target_date_col] = pd.NaT
    df_out[future_price_col] = np.nan
    df_out[target_change_col] = np.nan
    df_out[actual_offset_col] = np.nan
    df_out[offset_error_col] = np.nan

    W = max_tolerance_days
    H = horizon_days

    for _, group in df_out.groupby(active_group_cols, sort=False):
        n = len(group)
        if n == 0:
            continue

        dates = group[date_col].values
        prices = group[price_col].values
        orig_indices = group.index.values

        ideal_dates = dates + np.timedelta64(H, "D")
        win_min_dates = dates + np.timedelta64(H - W, "D")
        win_max_dates = dates + np.timedelta64(H + W, "D")

        idx_start = np.searchsorted(dates, win_min_dates, side="left")
        idx_end = np.searchsorted(dates, win_max_dates, side="right")

        matched_idx = np.full(n, -1, dtype=int)

        for i in range(n):
            s = idx_start[i]
            e = idx_end[i]
            if s >= e:
                continue

            cands = np.arange(s, e)
            cand_dates = dates[cands]
            diffs = (cand_dates - ideal_dates[i]).astype("timedelta64[D]").astype(int)
            abs_diffs = np.abs(diffs)
            min_diff = abs_diffs.min()

            best_mask = (abs_diffs == min_diff)
            best_cands = cands[best_mask]

            if len(best_cands) == 1:
                matched_idx[i] = best_cands[0]
            else:
                cand_diffs = diffs[best_mask]
                if tie_break == "forward":
                    forward_pos = np.where(cand_diffs >= 0)[0]
                    if len(forward_pos) > 0:
                        matched_idx[i] = best_cands[forward_pos[0]]
                    else:
                        matched_idx[i] = best_cands[0]
                else:
                    backward_pos = np.where(cand_diffs <= 0)[0]
                    if len(backward_pos) > 0:
                        matched_idx[i] = best_cands[backward_pos[-1]]
                    else:
                        matched_idx[i] = best_cands[0]

        valid_mask = matched_idx >= 0
        if np.any(valid_mask):
            valid_orig_idx = orig_indices[valid_mask]
            chosen_matched = matched_idx[valid_mask]
            target_dates = dates[chosen_matched]
            target_prices = prices[chosen_matched]
            origin_dates = dates[valid_mask]
            origin_prices = prices[valid_mask]

            offsets = (target_dates - origin_dates).astype("timedelta64[D]").astype(int)
            offset_errs = offsets - H
            price_changes = target_prices - origin_prices

            df_out.loc[valid_orig_idx, target_date_col] = target_dates
            df_out.loc[valid_orig_idx, future_price_col] = target_prices
            df_out.loc[valid_orig_idx, target_change_col] = price_changes
            df_out.loc[valid_orig_idx, actual_offset_col] = offsets
            df_out.loc[valid_orig_idx, offset_error_col] = offset_errs

    return df_out


def compute_observation_shift_targets(
    df: pd.DataFrame,
    shift_steps: int,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Reported Date",
    price_col: str = "Modal Price (Rs./Quintal)",
) -> pd.DataFrame:
    """
    Method B: Computes diagnostic observation-shift targets (.shift(-k)).
    Records the actual elapsed calendar days to show discrepancy with calendar time.
    """
    if group_cols is None:
        group_cols = ["Market Name", "Variety"]

    active_group_cols = [c for c in group_cols if c in df.columns]
    if not active_group_cols:
        raise ValueError(f"Required group columns {group_cols} not found in DataFrame.")

    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out[price_col] = pd.to_numeric(df_out[price_col], errors="coerce")

    df_out = df_out.sort_values(active_group_cols + [date_col]).reset_index(drop=True)

    prefix = f"shift_{shift_steps}"
    target_date_col = f"target_date_{prefix}"
    future_price_col = f"future_modal_price_{prefix}"
    target_change_col = f"future_price_change_{prefix}"
    elapsed_days_col = f"actual_elapsed_days_{prefix}"

    grouped = df_out.groupby(active_group_cols, sort=False)
    df_out[target_date_col] = grouped[date_col].shift(-shift_steps)
    df_out[future_price_col] = grouped[price_col].shift(-shift_steps)
    df_out[target_change_col] = df_out[future_price_col] - df_out[price_col]
    df_out[elapsed_days_col] = (df_out[target_date_col] - df_out[date_col]).dt.days

    return df_out


def apply_direct_horizon_temporal_purge(
    df: pd.DataFrame,
    split_date: str,
    target_date_col: str,
    date_col: str = "Reported Date",
) -> Tuple[pd.DataFrame, int]:
    """
    Purges any row whose matched future target date reaches into or past the split boundary.
    """
    split_dt = pd.to_datetime(split_date)

    if target_date_col in df.columns:
        leaks = df[target_date_col].notna() & (df[target_date_col] >= split_dt)
        purged_df = df[~leaks].copy()
        purged_count = int(leaks.sum())
    else:
        purged_df = df.copy()
        purged_count = 0

    return purged_df.reset_index(drop=True), purged_count
