"""
AgroVision — Member 2: Direct Horizon Target Generation (14-Day & 30-Day)

Provides isolated, point-in-time target generation and temporal purging for
14-calendar-day and 30-calendar-day rice modal price forecasting.

Key Guarantees:
1. Group Isolation: Observations are strictly grouped by Market Name + Variety.
   No cross-market or cross-variety target assignment.
2. Chronological Ordering: Enforced within each group before target lookup.
3. Calendar Target Window: For origin date t and horizon H in {14, 30}:
   Selects the FIRST valid observation on or after (t + H days), subject to a
   maximum target-date tolerance of max_tolerance_days (default: 3 days).
   Permitted target window: [t + H, t + H + max_tolerance_days].
4. Missing Target Handling: Forecast origins with no observation inside the
   permitted window receive NaN and are dropped from model training.
5. Target Calculation:
   future_price_change_Hd = future_modal_price - origin_modal_price.
6. Temporal Purging: Ensures no training or validation label extends across
   chronological split boundaries.
7. Zero Side-Effects: Completely isolated from the existing 1-7 day production
   pipeline.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_single_horizon_targets(
    df: pd.DataFrame,
    horizon_days: int,
    max_tolerance_days: int = 3,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Reported Date",
    price_col: str = "Modal Price (Rs./Quintal)",
) -> pd.DataFrame:
    """
    Computes direct calendar-horizon target for a single horizon H (e.g., 14 or 30 days).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned or feature-engineered time-series DataFrame.
    horizon_days : int
        Target calendar horizon in days (e.g. 14 or 30).
    max_tolerance_days : int, default=3
        Maximum allowed calendar days past the ideal target date to accept an observation.
    group_cols : list of str, optional
        Grouping keys. Defaults to ['Market Name', 'Variety'].
    date_col : str
        Reported date column name.
    price_col : str
        Modal price column name.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame augmented with:
        - target_date_{H}d: Timestamp of matched future observation.
        - future_modal_price_{H}d: Future modal price.
        - future_price_change_{H}d: Price delta (future_price - origin_price).
        - actual_horizon_{H}d: Calendar days from origin to matched target observation.
    """
    if group_cols is None:
        group_cols = ["Market Name", "Variety"]

    active_group_cols = [c for c in group_cols if c in df.columns]
    if not active_group_cols:
        raise ValueError(f"Required group columns {group_cols} not found in DataFrame.")

    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])
    df_out[price_col] = pd.to_numeric(df_out[price_col], errors="coerce")

    # Sort strictly by group and date
    df_out = df_out.sort_values(active_group_cols + [date_col]).reset_index(drop=True)

    h_prefix = f"{horizon_days}d"
    target_date_col = f"target_date_{h_prefix}"
    future_price_col = f"future_modal_price_{h_prefix}"
    target_change_col = f"future_price_change_{h_prefix}"
    actual_horizon_col = f"actual_horizon_{h_prefix}"

    # Initialize columns
    df_out[target_date_col] = pd.NaT
    df_out[future_price_col] = np.nan
    df_out[target_change_col] = np.nan
    df_out[actual_horizon_col] = np.nan

    target_delta = np.timedelta64(horizon_days, "D")
    max_delta = np.timedelta64(horizon_days + max_tolerance_days, "D")

    # Group-wise target matching via numpy searchsorted for high efficiency
    matched_indices = []
    matched_target_dates = []
    matched_prices = []
    matched_offsets = []

    for _, group in df_out.groupby(active_group_cols, sort=False):
        if len(group) == 0:
            continue

        dates = group[date_col].values
        prices = group[price_col].values
        orig_indices = group.index.values

        target_dates_ideal = dates + target_delta
        max_dates_allowed = dates + max_delta

        # Find first observation >= target_date_ideal
        idx = np.searchsorted(dates, target_dates_ideal, side="left")

        valid = idx < len(dates)
        idx_bounded = np.where(valid, idx, 0)
        actual_target_dates = dates[idx_bounded]

        # Check tolerance window: target_date_ideal <= actual_date <= max_date_allowed
        in_window = valid & (actual_target_dates <= max_dates_allowed)

        if not np.any(in_window):
            continue

        selected_orig_idx = orig_indices[in_window]
        selected_target_dates = actual_target_dates[in_window]
        selected_prices = prices[idx_bounded[in_window]]
        selected_offsets = (selected_target_dates - dates[in_window]) / np.timedelta64(1, "D")

        matched_indices.extend(selected_orig_idx)
        matched_target_dates.extend(selected_target_dates)
        matched_prices.extend(selected_prices)
        matched_offsets.extend(selected_offsets)

    if matched_indices:
        df_out.loc[matched_indices, target_date_col] = pd.to_datetime(matched_target_dates)
        df_out.loc[matched_indices, future_price_col] = matched_prices
        df_out.loc[matched_indices, actual_horizon_col] = matched_offsets
        df_out.loc[matched_indices, target_change_col] = (
            df_out.loc[matched_indices, future_price_col] - df_out.loc[matched_indices, price_col]
        )

    return df_out


def build_dual_horizon_dataset(
    df: pd.DataFrame,
    max_tolerance_days: int = 3,
    group_cols: Optional[List[str]] = None,
    date_col: str = "Reported Date",
    price_col: str = "Modal Price (Rs./Quintal)",
) -> pd.DataFrame:
    """
    Builds both 14-day and 30-day targets on the given DataFrame.
    """
    df_14 = compute_single_horizon_targets(
        df,
        horizon_days=14,
        max_tolerance_days=max_tolerance_days,
        group_cols=group_cols,
        date_col=date_col,
        price_col=price_col,
    )
    df_dual = compute_single_horizon_targets(
        df_14,
        horizon_days=30,
        max_tolerance_days=max_tolerance_days,
        group_cols=group_cols,
        date_col=date_col,
        price_col=price_col,
    )
    return df_dual


def apply_horizon_temporal_purge(
    df: pd.DataFrame,
    split_date: str,
    horizon_days: int,
    max_tolerance_days: int = 3,
    date_col: str = "Reported Date",
) -> Tuple[pd.DataFrame, int]:
    """
    Applies temporal purging to remove examples whose future target reaches into
    or across the split boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Training or validation subset ending on or before split_date.
    split_date : str
        Start date of the next chronological evaluation partition (e.g. '2021-02-01' or '2023-02-01').
    horizon_days : int
        Horizon in days (14 or 30).
    max_tolerance_days : int
        Maximum tolerance days (3).
    date_col : str
        Date column name.

    Returns
    -------
    purged_df : pd.DataFrame
        DataFrame with boundary-leaking rows removed.
    purged_count : int
        Number of rows removed by purging.
    """
    target_date_col = f"target_date_{horizon_days}d"
    split_dt = pd.to_datetime(split_date)

    if target_date_col in df.columns:
        # Purge any row where target date is on or after the split boundary
        leaks = df[target_date_col].notna() & (df[target_date_col] >= split_dt)
        purged_df = df[~leaks].copy()
        purged_count = int(leaks.sum())
    else:
        # Fallback based on maximum possible horizon offset if target_date column is absent
        purge_window = horizon_days + max_tolerance_days
        cutoff_date = split_dt - pd.Timedelta(days=purge_window)
        dates = pd.to_datetime(df[date_col])
        leaks = dates > cutoff_date
        purged_df = df[~leaks].copy()
        purged_count = int(leaks.sum())

    return purged_df.reset_index(drop=True), purged_count
