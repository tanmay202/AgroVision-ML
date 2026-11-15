"""
AgroVision — Multi-Commodity Advanced Feature Engineering V2
Provides comprehensive point-in-time candidate features for Price Forecasting and Volatility Modeling.
Strictly zero lookahead: all features computed from information available at or before observation time t.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

GROUP_COLUMNS = ["Market Name", "Variety"]
DATE_COLUMN = "Reported Date"
PRICE_COLUMN = "Modal Price (Rs./Quintal)"
ARRIVAL_COLUMN = "Arrivals (Tonnes)"
MIN_PRICE_COLUMN = "Min Price (Rs./Quintal)"
MAX_PRICE_COLUMN = "Max Price (Rs./Quintal)"


def compute_slope_weights(window: int) -> np.ndarray:
    """Least-squares slope weights for rolling window."""
    x = np.arange(window) - (window - 1) / 2.0
    return x / np.sum(x**2)


def generate_candidate_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes candidate features across all categories (A through G).
    Guarantees strict point-in-time calculation (shift(1) where appropriate).
    """
    df = df.copy()
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    if not group_cols:
        raise ValueError(f"Required group columns {GROUP_COLUMNS} not found.")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    df = df.sort_values(group_cols + [DATE_COLUMN]).reset_index(drop=True)

    grouped_p = df.groupby(group_cols)[PRICE_COLUMN]
    
    # ------------------------------------------------------------
    # A. PRICE LAGS (shifted past prices relative to origin t)
    # ------------------------------------------------------------
    # lag_1 is price at t-1 (1 observation ago)
    df["lag_1"] = grouped_p.shift(1)
    df["lag_2"] = grouped_p.shift(2)
    df["lag_3"] = grouped_p.shift(3)
    df["lag_5"] = grouped_p.shift(5)
    df["lag_7"] = grouped_p.shift(7)
    df["lag_10"] = grouped_p.shift(10)
    df["lag_14"] = grouped_p.shift(14)
    df["lag_21"] = grouped_p.shift(21)
    df["lag_30"] = grouped_p.shift(30)

    # ------------------------------------------------------------
    # B. ROLLING STATISTICS (strictly over past observations shifted by 1)
    # ------------------------------------------------------------
    past_price = grouped_p.shift(1)
    grp_past = past_price.groupby([df[c] for c in group_cols])

    # Rolling means
    df["rolling_mean_3"] = grp_past.transform(lambda x: x.rolling(3).mean())
    df["rolling_mean_5"] = grp_past.transform(lambda x: x.rolling(5).mean())
    df["rolling_mean_7"] = grp_past.transform(lambda x: x.rolling(7).mean())
    df["rolling_mean_14"] = grp_past.transform(lambda x: x.rolling(14).mean())
    df["rolling_mean_30"] = grp_past.transform(lambda x: x.rolling(30).mean())

    # Rolling stds
    df["rolling_std_5"] = grp_past.transform(lambda x: x.rolling(5).std()).fillna(0)
    df["rolling_std_7"] = grp_past.transform(lambda x: x.rolling(7).std()).fillna(0)
    df["rolling_std_14"] = grp_past.transform(lambda x: x.rolling(14).std()).fillna(0)
    df["rolling_std_30"] = grp_past.transform(lambda x: x.rolling(30).std()).fillna(0)

    # Rolling medians
    df["rolling_median_7"] = grp_past.transform(lambda x: x.rolling(7).median())
    df["rolling_median_14"] = grp_past.transform(lambda x: x.rolling(14).median())
    df["rolling_min_7"] = grp_past.transform(lambda x: x.rolling(7).min())
    df["rolling_max_7"] = grp_past.transform(lambda x: x.rolling(7).max())

    # ------------------------------------------------------------
    # C. MOMENTUM & ACCELERATION (differences up to origin t)
    # ------------------------------------------------------------
    # P_t - P_{t-k}
    df["price_change_1"] = df[PRICE_COLUMN] - df["lag_1"]
    df["price_change_3"] = df[PRICE_COLUMN] - df["lag_3"]
    df["price_change_7"] = df[PRICE_COLUMN] - df["lag_7"]
    df["price_change_14"] = df[PRICE_COLUMN] - df["lag_14"]

    df["momentum_3"] = df["price_change_3"] / (df["lag_3"] + 1e-4) * 100.0
    df["momentum_7"] = df["price_change_7"] / (df["lag_7"] + 1e-4) * 100.0
    df["momentum_14"] = df["price_change_14"] / (df["lag_14"] + 1e-4) * 100.0

    df["acceleration_3_7"] = df["price_change_3"] - (df["price_change_7"] - df["price_change_3"])
    df["acceleration_7_14"] = df["price_change_7"] - (df["price_change_14"] - df["price_change_7"])
    df["accel_1_7"] = df["price_change_1"] - (df["price_change_7"] / 7.0)
    df["accel_1_14"] = df["price_change_1"] - (df["price_change_14"] / 14.0)

    # ------------------------------------------------------------
    # D. TREND & EWMA
    # ------------------------------------------------------------
    # Slope of past prices including current price P_t
    w7 = compute_slope_weights(7)
    df["trend_slope_7"] = sum(w7[i] * (df[PRICE_COLUMN] if i == 6 else grouped_p.shift(6 - i)) for i in range(7))
    w14 = compute_slope_weights(14)
    df["trend_slope_14"] = sum(w14[i] * (df[PRICE_COLUMN] if i == 13 else grouped_p.shift(13 - i)) for i in range(14))
    w30 = compute_slope_weights(30)
    df["trend_slope_30"] = sum(w30[i] * (df[PRICE_COLUMN] if i == 29 else grouped_p.shift(29 - i)) for i in range(30))

    df["ewma_5"] = grouped_p.transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df["ewma_7"] = grouped_p.transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["ewma_14"] = grouped_p.transform(lambda x: x.ewm(span=14, adjust=False).mean())
    df["ewma_30"] = grouped_p.transform(lambda x: x.ewm(span=30, adjust=False).mean())

    df["price_to_rolling_mean_7"] = df[PRICE_COLUMN] / (df["rolling_mean_7"] + 1e-4)
    df["price_to_rolling_mean_14"] = df[PRICE_COLUMN] / (df["rolling_mean_14"] + 1e-4)
    df["gap_mean_7"] = df[PRICE_COLUMN] - df["rolling_mean_7"]
    df["gap_mean_14"] = df[PRICE_COLUMN] - df["rolling_mean_14"]
    df["price_zscore_7"] = (df[PRICE_COLUMN] - df["rolling_mean_7"]) / (df["rolling_std_7"] + 1e-4)
    df["price_zscore_14"] = (df[PRICE_COLUMN] - df["rolling_mean_14"]) / (df["rolling_std_14"] + 1e-4)

    # ------------------------------------------------------------
    # E. ARRIVAL FEATURES
    # ------------------------------------------------------------
    if ARRIVAL_COLUMN in df.columns:
        grouped_arr = df.groupby(group_cols)[ARRIVAL_COLUMN]
        df["arrival_lag_1"] = grouped_arr.shift(1)
        df["arrival_lag_3"] = grouped_arr.shift(3)
        df["arrival_lag_7"] = grouped_arr.shift(7)

        past_arr = grouped_arr.shift(1)
        grp_past_arr = past_arr.groupby([df[c] for c in group_cols])
        df["arrival_rolling_mean_7"] = grp_past_arr.transform(lambda x: x.rolling(7).mean())
        df["arrival_rolling_mean_14"] = grp_past_arr.transform(lambda x: x.rolling(14).mean())
        df["arrival_rolling_std_7"] = grp_past_arr.transform(lambda x: x.rolling(7).std()).fillna(0)

        df["arrival_pct_change"] = grouped_arr.pct_change(fill_method=None) * 100.0
        df["price_pct_change"] = grouped_p.pct_change(fill_method=None) * 100.0
        df["arrival_price_interaction"] = np.log1p(np.maximum(df[ARRIVAL_COLUMN], 0)) * df["price_pct_change"]
    else:
        df["price_pct_change"] = grouped_p.pct_change(fill_method=None) * 100.0

    # ------------------------------------------------------------
    # F. TIME AND SEASONALITY (omits raw day-of-month to avoid overfitting)
    # ------------------------------------------------------------
    df["year"] = df[DATE_COLUMN].dt.year
    df["month"] = df[DATE_COLUMN].dt.month
    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek
    df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)
    df["quarter"] = df[DATE_COLUMN].dt.quarter
    day_of_year = df[DATE_COLUMN].dt.dayofyear
    df["annual_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["annual_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # ------------------------------------------------------------
    # G. PRICE STRUCTURE (Spread, positions, ratios)
    # ------------------------------------------------------------
    if MIN_PRICE_COLUMN in df.columns and MAX_PRICE_COLUMN in df.columns:
        df["price_spread"] = df[MAX_PRICE_COLUMN] - df[MIN_PRICE_COLUMN]
        spread_safe = np.maximum(df["price_spread"], 1e-4)
        df["price_position_in_range"] = np.clip((df[PRICE_COLUMN] - df[MIN_PRICE_COLUMN]) / spread_safe, 0.0, 1.0)
        df["min_to_modal_ratio"] = df[MIN_PRICE_COLUMN] / (df[PRICE_COLUMN] + 1e-4)
        df["modal_to_max_ratio"] = df[PRICE_COLUMN] / (df[MAX_PRICE_COLUMN] + 1e-4)

    # ------------------------------------------------------------
    # H. FLAT-PRICE & STREAK DYNAMICS (Key for Persistence Conundrum)
    # ------------------------------------------------------------
    is_move = (df["price_change_1"].abs() > 1e-4).astype(int)
    is_pos = (df["price_change_1"] > 1e-4).astype(int)
    is_neg = (df["price_change_1"] < -1e-4).astype(int)

    group_keys = [df[c] for c in group_cols]
    df["move_count_7"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
    df["move_count_14"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(14).sum())
    df["move_count_30"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(30).sum())
    df["pos_count_7"] = is_pos.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
    df["neg_count_7"] = is_neg.groupby(group_keys).transform(lambda x: x.rolling(7).sum())

    # Streaks
    block_id = (~(is_move == 0)).groupby(group_keys).cumsum()
    df["streak_zero"] = (is_move == 0).groupby(group_keys + [block_id]).cumsum()

    last_move = df["price_change_1"].replace(0, np.nan)
    df["last_non_zero_move"] = last_move.groupby(group_keys).ffill().fillna(0)
    df["last_move_abs"] = df["last_non_zero_move"].abs()

    return df
