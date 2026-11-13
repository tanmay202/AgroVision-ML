"""
AgroVision Member 2 — Volatility V3 Feature Engineering

Point-in-time safe historical feature engineering for V3 calendar-horizon volatility modeling.
Strictly uses information available at or before observation time t:
- Historical lagged percentage changes (1, 2, 5 days)
- Rolling standard deviations (5, 7, 14, 30 days)
- Historical normalized price ranges (past 5, 10, 20 observations)
- Calendar-aligned historical price ranges (past 7 and 14 calendar days)
- EWMA volatility estimates
- Volatility regime trends, ratios, and streak dynamics
- Calendar and arrival signals
- 38 total features, all strictly point-in-time
"""

import sys
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Full feature column list for V3
V3_FEATURE_COLS = [
    # Historical observation-based ranges & normalized scale
    "past_5_range",
    "past_5_pct_range",
    "past_10_range",
    "past_10_pct_range",
    "past_20_range",
    "past_20_pct_range",
    # Calendar-aligned historical price ranges
    "past_7d_range",
    "past_7d_pct_range",
    "past_14d_pct_range",
    # Rolling standard deviations
    "rolling_std_5",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_30",
    "rolling_std_pct_5",
    "rolling_std_pct_14",
    # Lagged percentage changes
    "pct_change_1",
    "pct_change_2",
    "pct_change_5",
    "abs_pct_change_1",
    "abs_pct_change_5",
    # EWMA volatility
    "ewma_abs_diff_5",
    "ewma_abs_diff_14",
    "ewma_pct_vol_5",
    "ewma_pct_vol_14",
    # Volatility trends and dynamics
    "vol_trend_5_10",
    "vol_pct_trend_5_10",
    "vol_std_ratio_5_14",
    "nonzero_moves_5",
    "nonzero_moves_10",
    "streak_zero",
    "max_daily_pct_jump_5",
    # Scale and calendar
    "price_level",
    "price_to_mean_7",
    "month",
    "day_of_week",
    "week_of_year",
    # Arrivals
    "arrival_lag_1",
    "arrival_rolling_mean_7",
]


def extract_v3_volatility_features(
    df: pd.DataFrame,
    group_cols: List[str],
    date_col: str,
    price_col: str,
    arrival_col: str = "Arrivals (Tonnes)",
) -> pd.DataFrame:
    """
    Extracts all 38 V3 point-in-time features strictly using past information <= t.
    Guarantees zero future lookahead contamination.
    """
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    df_out[price_col] = pd.to_numeric(df_out[price_col], errors="coerce")
    df_out = df_out.sort_values(group_cols + [date_col]).reset_index(drop=True)

    group_series = df_out[group_cols[0]].astype(str)
    for c in group_cols[1:]:
        group_series = group_series + "___" + df_out[c].astype(str)

    grouped_p = df_out.groupby(group_cols, sort=False)[price_col]

    # 1. Historical price lags
    lag_1 = grouped_p.shift(1)
    lag_2 = grouped_p.shift(2)
    lag_5 = grouped_p.shift(5)

    # 2. Lagged percentage changes (strictly past)
    df_out["pct_change_1"] = ((df_out[price_col] - lag_1) / (lag_1 + 1e-4)) * 100.0
    df_out["pct_change_2"] = ((lag_1 - lag_2) / (lag_2 + 1e-4)) * 100.0
    df_out["pct_change_5"] = ((df_out[price_col] - lag_5) / (lag_5 + 1e-4)) * 100.0
    df_out["abs_pct_change_1"] = df_out["pct_change_1"].abs()
    df_out["abs_pct_change_5"] = df_out["pct_change_5"].abs()

    # 3. Observation-based historical rolling price ranges
    p5_max = grouped_p.transform(lambda s: s.rolling(5).max())
    p5_min = grouped_p.transform(lambda s: s.rolling(5).min())
    df_out["past_5_range"] = p5_max - p5_min
    df_out["past_5_pct_range"] = (df_out["past_5_range"] / (df_out[price_col] + 1e-4)) * 100.0

    p10_max = grouped_p.transform(lambda s: s.rolling(10).max())
    p10_min = grouped_p.transform(lambda s: s.rolling(10).min())
    df_out["past_10_range"] = p10_max - p10_min
    df_out["past_10_pct_range"] = (df_out["past_10_range"] / (df_out[price_col] + 1e-4)) * 100.0

    p20_max = grouped_p.transform(lambda s: s.rolling(20).max())
    p20_min = grouped_p.transform(lambda s: s.rolling(20).min())
    df_out["past_20_range"] = p20_max - p20_min
    df_out["past_20_pct_range"] = (df_out["past_20_range"] / (df_out[price_col] + 1e-4)) * 100.0

    # 4. Calendar-aligned historical price ranges (if not already computed by target def)
    if "past_pct_range_7d" not in df_out.columns or "past_pct_range_14d" not in df_out.columns:
        # Fast vectorized calendar range computation
        n_rows = len(df_out)
        dates_int = df_out[date_col].values.astype("datetime64[D]").astype(np.int64)
        prices = df_out[price_col].values.astype(np.float64)

        group_series = df_out[group_cols[0]].astype(str)
        for c in group_cols[1:]:
            group_series = group_series + "___" + df_out[c].astype(str)
        group_keys = group_series.to_numpy()
        diff = np.where(group_keys[:-1] != group_keys[1:])[0] + 1
        group_starts = np.r_[0, diff]
        group_ends = np.r_[diff, n_rows]

        p7_arr = np.full(n_rows, 0.0)
        p7_nom_arr = np.full(n_rows, 0.0)
        p14_arr = np.full(n_rows, 0.0)

        for g_s, g_e in zip(group_starts, group_ends):
            g_dates = dates_int[g_s:g_e]
            g_prices = prices[g_s:g_e]
            n_group = g_e - g_s
            start_7 = np.searchsorted(g_dates, g_dates - 7, side="left")
            start_14 = np.searchsorted(g_dates, g_dates - 14, side="left")
            for i in range(n_group):
                pt = g_prices[i]
                if pt > 0:
                    if i - start_7[i] >= 1:
                        sub7 = g_prices[start_7[i]:i+1]
                        valid7 = sub7[(~np.isnan(sub7)) & (sub7 > 0)]
                        if len(valid7) > 1:
                            p7_nom = np.max(valid7) - np.min(valid7)
                            p7_nom_arr[g_s + i] = p7_nom
                            p7_arr[g_s + i] = (p7_nom / (pt + 1e-4)) * 100.0
                    if i - start_14[i] >= 1:
                        sub14 = g_prices[start_14[i]:i+1]
                        valid14 = sub14[(~np.isnan(sub14)) & (sub14 > 0)]
                        if len(valid14) > 1:
                            p14_arr[g_s + i] = ((np.max(valid14) - np.min(valid14)) / (pt + 1e-4)) * 100.0

        df_out["past_7d_range"] = p7_nom_arr
        df_out["past_7d_pct_range"] = p7_arr
        df_out["past_14d_pct_range"] = p14_arr
    else:
        df_out["past_7d_pct_range"] = df_out["past_pct_range_7d"].fillna(0.0)
        df_out["past_14d_pct_range"] = df_out["past_pct_range_14d"].fillna(0.0)
        df_out["past_7d_range"] = (df_out["past_7d_pct_range"] / 100.0) * df_out[price_col]

    # 5. Rolling standard deviations
    df_out["rolling_std_5"] = grouped_p.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df_out["rolling_std_7"] = grouped_p.transform(lambda s: s.rolling(7).std()).fillna(0.0)
    df_out["rolling_std_14"] = grouped_p.transform(lambda s: s.rolling(14).std()).fillna(0.0)
    df_out["rolling_std_30"] = grouped_p.transform(lambda s: s.rolling(30).std()).fillna(0.0)

    grouped_pct = df_out.groupby(group_cols, sort=False)["pct_change_1"]
    df_out["rolling_std_pct_5"] = grouped_pct.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df_out["rolling_std_pct_14"] = grouped_pct.transform(lambda s: s.rolling(14).std()).fillna(0.0)

    # 6. EWMA volatility
    diff_abs = grouped_p.diff().abs()
    df_out["ewma_abs_diff_5"] = diff_abs.groupby(group_series, sort=False).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df_out["ewma_abs_diff_14"] = diff_abs.groupby(group_series, sort=False).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    df_out["ewma_pct_vol_5"] = df_out["abs_pct_change_1"].groupby(group_series, sort=False).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df_out["ewma_pct_vol_14"] = df_out["abs_pct_change_1"].groupby(group_series, sort=False).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    # 7. Volatility trends & ratios
    df_out["vol_trend_5_10"] = df_out["past_5_range"] / (df_out["past_10_range"] + 1e-4)
    df_out["vol_pct_trend_5_10"] = df_out["past_5_pct_range"] / (df_out["past_10_pct_range"] + 1e-4)
    df_out["vol_std_ratio_5_14"] = df_out["rolling_std_5"] / (df_out["rolling_std_14"] + 1e-4)

    # Non-zero price moves (first observation in group has no prior price, so diff is 0)
    price_diff = grouped_p.diff().fillna(0.0)
    is_move = (price_diff != 0).astype(int)
    df_out["nonzero_moves_5"] = is_move.groupby(group_series, sort=False).transform(lambda s: s.rolling(5).sum()).fillna(0.0)
    df_out["nonzero_moves_10"] = is_move.groupby(group_series, sort=False).transform(lambda s: s.rolling(10).sum()).fillna(0.0)

    # Flat streak (consecutive trading days with zero price movement)
    def compute_flat_streak(is_flat_series: pd.Series) -> pd.Series:
        blocks = (~is_flat_series).cumsum()
        return is_flat_series.astype(int).groupby(blocks).cumsum()

    is_flat_diff = (price_diff == 0) & (grouped_p.shift(1).notna())
    df_out["streak_zero"] = is_flat_diff.groupby(group_series, sort=False).transform(compute_flat_streak).fillna(0.0)
    df_out["max_daily_pct_jump_5"] = df_out["abs_pct_change_1"].groupby(group_series, sort=False).transform(lambda s: s.rolling(5).max()).fillna(0.0)

    # 8. Price scale and calendar
    df_out["price_level"] = df_out[price_col].astype(float)
    rolling_mean_7 = grouped_p.transform(lambda s: s.rolling(7).mean())
    df_out["price_to_mean_7"] = df_out[price_col] / (rolling_mean_7 + 1e-4)

    df_out["month"] = df_out[date_col].dt.month
    df_out["day_of_week"] = df_out[date_col].dt.dayofweek
    df_out["week_of_year"] = df_out[date_col].dt.isocalendar().week.astype(int)

    # 9. Arrivals (strictly historical lag >= 1)
    if arrival_col in df_out.columns:
        grouped_arr = df_out.groupby(group_cols, sort=False)[arrival_col]
        df_out["arrival_lag_1"] = grouped_arr.shift(1).fillna(0.0)
        df_out["arrival_rolling_mean_7"] = grouped_arr.shift(1).transform(lambda s: s.rolling(7).mean()).fillna(0.0)
    else:
        df_out["arrival_lag_1"] = 0.0
        df_out["arrival_rolling_mean_7"] = 0.0

    # Ensure no inf or nan in feature columns
    df_out[V3_FEATURE_COLS] = df_out[V3_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df_out
