"""
AgroVision Member 2 — Volatility Feature Engineering

Point-in-time safe historical feature engineering for volatility modeling.
Strictly uses information available at or before observation time t:
- Historical lagged percentage changes
- Rolling standard deviations (price and returns)
- Normalized historical price ranges (past 5, 10, 20 observations)
- EWMA volatility estimates
- Volatility regime trends and streak dynamics
- Calendar and arrival signals
"""

import numpy as np
import pandas as pd


EXPERIMENTAL_FEATURE_COLS = [
    # Historical price ranges & normalized scale
    "past_5_range",
    "past_5_pct_range",
    "past_10_range",
    "past_10_pct_range",
    "past_20_range",
    "past_20_pct_range",
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


def extract_volatility_features(df, group_cols, date_col, price_col, arrival_col="Arrivals (Tonnes)"):
    """
    Generate all experimental volatility features on sorted time series data.
    Ensures zero forward lookahead leakage.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(group_cols + [date_col]).reset_index(drop=True)

    grouped_p = df.groupby(group_cols)[price_col]

    # 1. Historical price lags
    lag_1 = grouped_p.shift(1)
    lag_2 = grouped_p.shift(2)
    lag_5 = grouped_p.shift(5)

    # 2. Lagged percentage changes (strictly past)
    df["pct_change_1"] = ((df[price_col] - lag_1) / (lag_1 + 1e-4)) * 100.0
    df["pct_change_2"] = ((lag_1 - lag_2) / (lag_2 + 1e-4)) * 100.0
    df["pct_change_5"] = ((df[price_col] - lag_5) / (lag_5 + 1e-4)) * 100.0
    df["abs_pct_change_1"] = df["pct_change_1"].abs()
    df["abs_pct_change_5"] = df["pct_change_5"].abs()

    # 3. Historical rolling price ranges
    p5_max = grouped_p.transform(lambda s: s.rolling(5).max())
    p5_min = grouped_p.transform(lambda s: s.rolling(5).min())
    df["past_5_range"] = p5_max - p5_min
    df["past_5_pct_range"] = (df["past_5_range"] / (df[price_col] + 1e-4)) * 100.0

    p10_max = grouped_p.transform(lambda s: s.rolling(10).max())
    p10_min = grouped_p.transform(lambda s: s.rolling(10).min())
    df["past_10_range"] = p10_max - p10_min
    df["past_10_pct_range"] = (df["past_10_range"] / (df[price_col] + 1e-4)) * 100.0

    p20_max = grouped_p.transform(lambda s: s.rolling(20).max())
    p20_min = grouped_p.transform(lambda s: s.rolling(20).min())
    df["past_20_range"] = p20_max - p20_min
    df["past_20_pct_range"] = (df["past_20_range"] / (df[price_col] + 1e-4)) * 100.0

    # 4. Rolling standard deviations
    df["rolling_std_5"] = grouped_p.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df["rolling_std_7"] = grouped_p.transform(lambda s: s.rolling(7).std()).fillna(0.0)
    df["rolling_std_14"] = grouped_p.transform(lambda s: s.rolling(14).std()).fillna(0.0)
    df["rolling_std_30"] = grouped_p.transform(lambda s: s.rolling(30).std()).fillna(0.0)

    grouped_pct = df.groupby(group_cols)["pct_change_1"]
    df["rolling_std_pct_5"] = grouped_pct.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df["rolling_std_pct_14"] = grouped_pct.transform(lambda s: s.rolling(14).std()).fillna(0.0)

    # 5. EWMA volatility
    diff_abs = grouped_p.diff().abs()
    df["ewma_abs_diff_5"] = diff_abs.groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df["ewma_abs_diff_14"] = diff_abs.groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    df["ewma_pct_vol_5"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df["ewma_pct_vol_14"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    # 6. Volatility trends & ratios
    df["vol_trend_5_10"] = df["past_5_range"] / (df["past_10_range"] + 1e-4)
    df["vol_pct_trend_5_10"] = df["past_5_pct_range"] / (df["past_10_pct_range"] + 1e-4)
    df["vol_std_ratio_5_14"] = df["rolling_std_5"] / (df["rolling_std_14"] + 1e-4)

    is_move = (grouped_p.diff() != 0).astype(int)
    df["nonzero_moves_5"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(5).sum()).fillna(0.0)
    df["nonzero_moves_10"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(10).sum()).fillna(0.0)

    # Streak of unchanged price days
    def compute_flat_streak(series):
        is_flat = (series == 0)
        blocks = (~is_flat).cumsum()
        return is_flat.groupby(blocks).cumsum()

    df["streak_zero"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: compute_flat_streak(s == 0)).fillna(0.0)
    df["max_daily_pct_jump_5"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(5).max()).fillna(0.0)

    # 7. Price scale and calendar
    df["price_level"] = df[price_col].astype(float)
    rolling_mean_7 = grouped_p.transform(lambda s: s.rolling(7).mean())
    df["price_to_mean_7"] = df[price_col] / (rolling_mean_7 + 1e-4)

    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

    # 8. Arrivals (historical only)
    if arrival_col in df.columns:
        grouped_arr = df.groupby(group_cols)[arrival_col]
        df["arrival_lag_1"] = grouped_arr.shift(1).fillna(0.0)
        df["arrival_rolling_mean_7"] = grouped_arr.shift(1).transform(lambda s: s.rolling(7).mean()).fillna(0.0)
    else:
        df["arrival_lag_1"] = 0.0
        df["arrival_rolling_mean_7"] = 0.0

    # Ensure no inf or nan in feature columns
    df[EXPERIMENTAL_FEATURE_COLS] = df[EXPERIMENTAL_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
