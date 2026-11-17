"""
Feature Engineering for AgroVision-ML Rice Forecasting.
Produces:
- 48 point-in-time features for Short-term (1-7d), 14-day, and 30-day XGBoost price regressors.
- 35 point-in-time features for price-normalized volatility classification.
"""

from typing import Union, List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    COLUMN_ALIASES,
)


class InsufficientHistoryError(ValueError):
    """Raised when the input history has fewer than 30 valid observations."""
    pass


def normalize_input_data(
    history: Union[pd.DataFrame, List[Dict]],
    market: Optional[str] = None,
    variety: Optional[str] = None,
    prediction_date: Optional[Union[str, pd.Timestamp]] = None,
    min_observations: int = 30,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Validates, standardizes, sorts, and filters raw historical observations.

    Accepts:
    - pandas DataFrame or list of dicts.
    - Column names can be standard AgroVision names or standard snake_case aliases.

    Returns:
    - Cleaned and sorted pandas DataFrame with >= min_observations rows.
    - Resolved market string.
    - Resolved variety string.
    - Resolved prediction_date string (YYYY-MM-DD).
    """
    if history is None:
        raise ValueError("Input history cannot be None.")

    if isinstance(history, list):
        if len(history) == 0:
            raise InsufficientHistoryError("Empty history list provided.")
        df = pd.DataFrame(history)
    elif isinstance(history, pd.DataFrame):
        if history.empty:
            raise InsufficientHistoryError("Empty history DataFrame provided.")
        df = history.copy()
    else:
        raise TypeError(f"Unsupported history type: {type(history)}. Expected DataFrame or list of dicts.")

    # Standardize column names via aliases
    rename_map = {}
    for col in df.columns:
        c_clean = str(col).strip()
        c_low = c_clean.lower()
        if c_low in COLUMN_ALIASES and COLUMN_ALIASES[c_low] not in df.columns:
            rename_map[col] = COLUMN_ALIASES[c_low]
    if rename_map:
        df = df.rename(columns=rename_map)

    # Infer or set Market
    if market is None:
        if GROUP_COLUMNS[0] in df.columns and not df[GROUP_COLUMNS[0]].dropna().empty:
            market = str(df[GROUP_COLUMNS[0]].dropna().iloc[-1]).strip()
        else:
            raise ValueError(f"Market must be provided or present in '{GROUP_COLUMNS[0]}' column.")
    else:
        market = str(market).strip()

    # Infer or set Variety
    if variety is None:
        if GROUP_COLUMNS[1] in df.columns and not df[GROUP_COLUMNS[1]].dropna().empty:
            variety = str(df[GROUP_COLUMNS[1]].dropna().iloc[-1]).strip()
        else:
            raise ValueError(f"Variety must be provided or present in '{GROUP_COLUMNS[1]}' column.")
    else:
        variety = str(variety).strip()

    # Infill market/variety columns if not in df
    if GROUP_COLUMNS[0] not in df.columns:
        df[GROUP_COLUMNS[0]] = market
    if GROUP_COLUMNS[1] not in df.columns:
        df[GROUP_COLUMNS[1]] = variety

    # Required columns check
    for req_col in [DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN]:
        if req_col not in df.columns:
            raise ValueError(f"Required column missing from input: '{req_col}' (or supported alias).")

    # Infill / Parse Date
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    if df[DATE_COLUMN].isna().all():
        raise ValueError("Failed to parse valid dates in date column.")

    # Infer or format prediction_date
    if prediction_date is None:
        pred_dt = df[DATE_COLUMN].dropna().max()
        prediction_date = str(pred_dt.date())
    else:
        pred_dt = pd.to_datetime(prediction_date)
        prediction_date = str(pred_dt.date())

    # Filter by market and variety (case-insensitive fallback)
    mask = (
        (df[GROUP_COLUMNS[0]].astype(str).str.strip().str.lower() == market.lower()) &
        (df[GROUP_COLUMNS[1]].astype(str).str.strip().str.lower() == variety.lower())
    )
    group_df = df[mask].copy()
    if group_df.empty:
        raise ValueError(f"No observations found for Market: '{market}', Variety: '{variety}'.")

    # Filter to point-in-time <= prediction_date
    group_df = group_df[group_df[DATE_COLUMN] <= pred_dt]

    # Sort chronologically and drop duplicate dates
    group_df = group_df.sort_values(DATE_COLUMN).drop_duplicates(subset=[DATE_COLUMN], keep="last")

    # Validate observation count
    n_obs = len(group_df)
    if n_obs < min_observations:
        raise InsufficientHistoryError(
            f"Insufficient history for Market='{market}', Variety='{variety}'. "
            f"Required at least {min_observations} observations, found {n_obs}."
        )

    # Impute missing prices/arrivals using point-in-time forward-fill then backward-fill
    for col in [PRICE_COLUMN, ARRIVAL_COLUMN]:
        group_df[col] = pd.to_numeric(group_df[col], errors="coerce").ffill().bfill()

    return group_df.reset_index(drop=True), market, variety, prediction_date


def generate_48_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates all 48 point-in-time features required by the XGBoost rice price models.
    Guarantees strict zero-lookahead temporal integrity.
    """
    df = df.copy()
    grouped_price = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]

    # 1. Base pipeline lag and rolling features
    df["lag_1"] = grouped_price.shift(1)
    df["lag_7"] = grouped_price.shift(7)
    df["lag_14"] = grouped_price.shift(14)
    df["lag_30"] = grouped_price.shift(30)

    df["rolling_mean_7"] = grouped_price.shift(1).transform(lambda x: x.rolling(7).mean())
    df["rolling_mean_14"] = grouped_price.shift(1).transform(lambda x: x.rolling(14).mean())
    df["rolling_mean_30"] = grouped_price.shift(1).transform(lambda x: x.rolling(30).mean())
    df["rolling_std_7"] = grouped_price.shift(1).transform(lambda x: x.rolling(7).std()).fillna(0)
    df["rolling_std_14"] = grouped_price.shift(1).transform(lambda x: x.rolling(14).std()).fillna(0)

    df["year"] = df[DATE_COLUMN].dt.year
    df["day"] = df[DATE_COLUMN].dt.day
    df["month"] = df[DATE_COLUMN].dt.month
    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek
    df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

    grouped_arr = df.groupby(GROUP_COLUMNS)[ARRIVAL_COLUMN]
    df["arrival_lag_1"] = grouped_arr.shift(1)
    df["arrival_lag_7"] = grouped_arr.shift(7)
    df["arrival_rolling_mean_7"] = grouped_arr.shift(1).transform(lambda x: x.rolling(7).mean())
    df["arrival_rolling_mean_14"] = grouped_arr.shift(1).transform(lambda x: x.rolling(14).mean())

    df["price_pct_change"] = grouped_price.pct_change(fill_method=None) * 100.0
    df["arrival_pct_change"] = grouped_arr.pct_change(fill_method=None) * 100.0

    # 2. Advanced momentum, acceleration, and slope features
    grouped = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]

    df["price_change_1"] = grouped.diff(1)
    df["price_change_7"] = grouped.diff(7)
    df["price_change_14"] = grouped.diff(14)
    df["accel_1_7"] = df["price_change_1"] - df["price_change_7"]
    df["accel_1_14"] = df["price_change_1"] - df["price_change_14"]

    def get_slope_weights(window):
        x = np.arange(window) - (window - 1) / 2.0
        return x / np.sum(x**2)

    w7 = get_slope_weights(7)
    df["trend_slope_7"] = sum(w7[i] * grouped.shift(6 - i) for i in range(7))
    w14 = get_slope_weights(14)
    df["trend_slope_14"] = sum(w14[i] * grouped.shift(13 - i) for i in range(14))
    w30 = get_slope_weights(30)
    df["trend_slope_30"] = sum(w30[i] * grouped.shift(29 - i) for i in range(30))

    df["ewma_7"] = grouped.transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["ewma_14"] = grouped.transform(lambda x: x.ewm(span=14, adjust=False).mean())
    df["ewma_30"] = grouped.transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df["rolling_median_7"] = grouped.transform(lambda x: x.rolling(7).median())
    df["rolling_median_14"] = grouped.transform(lambda x: x.rolling(14).median())
    df["rolling_median_30"] = grouped.transform(lambda x: x.rolling(30).median())

    df["gap_mean_7"] = df[PRICE_COLUMN] - df["rolling_mean_7"]
    df["gap_mean_14"] = df[PRICE_COLUMN] - df["rolling_mean_14"]
    df["zscore_7"] = (df[PRICE_COLUMN] - df["rolling_mean_7"]) / (df["rolling_std_7"] + 1e-5)
    df["zscore_14"] = (df[PRICE_COLUMN] - df["rolling_mean_14"]) / (df["rolling_std_14"] + 1e-5)

    def compute_streak(data_df, condition_series):
        group_keys = [data_df[c] for c in GROUP_COLUMNS]
        block_id = (~condition_series).groupby(group_keys).cumsum()
        streak = condition_series.groupby(group_keys + [block_id]).cumsum()
        return streak

    is_move = (df["price_change_1"] != 0).astype(int)
    is_pos = (df["price_change_1"] > 0).astype(int)
    is_neg = (df["price_change_1"] < 0).astype(int)

    group_keys = [df[c] for c in GROUP_COLUMNS]
    df["move_count_7"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
    df["move_count_14"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(14).sum())
    df["move_count_30"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(30).sum())
    df["pos_count_7"] = is_pos.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
    df["neg_count_7"] = is_neg.groupby(group_keys).transform(lambda x: x.rolling(7).sum())

    df["streak_zero"] = compute_streak(df, is_move == 0)
    df["streak_pos"] = compute_streak(df, is_pos == 1)
    df["streak_neg"] = compute_streak(df, is_neg == 1)

    last_move = df["price_change_1"].replace(0, np.nan)
    df["last_non_zero_move"] = last_move.groupby(group_keys).ffill().fillna(0)
    df["last_move_abs"] = df["last_non_zero_move"].abs()

    return df


def generate_normalized_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the 35 point-in-time features required by the normalized volatility classifier.
    Features strictly use information at or before observation time t.
    """
    df = df.copy()
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    if not group_cols:
        group_cols = GROUP_COLUMNS
    price_col = PRICE_COLUMN if PRICE_COLUMN in df.columns else "modal_price"
    date_col = DATE_COLUMN if DATE_COLUMN in df.columns else "Reported Date"
    arrival_col = ARRIVAL_COLUMN if ARRIVAL_COLUMN in df.columns else "Arrivals (Tonnes)"

    df[date_col] = pd.to_datetime(df[date_col])

    grouped_p = df.groupby(group_cols)[price_col]

    lag_1 = grouped_p.shift(1)
    lag_2 = grouped_p.shift(2)
    lag_5 = grouped_p.shift(5)

    df["pct_change_1"] = ((df[price_col] - lag_1) / (lag_1 + 1e-4)) * 100.0
    df["pct_change_2"] = ((lag_1 - lag_2) / (lag_2 + 1e-4)) * 100.0
    df["pct_change_5"] = ((df[price_col] - lag_5) / (lag_5 + 1e-4)) * 100.0
    df["abs_pct_change_1"] = df["pct_change_1"].abs()
    df["abs_pct_change_5"] = df["pct_change_5"].abs()

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

    df["rolling_std_5"] = grouped_p.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df["rolling_std_7"] = grouped_p.transform(lambda s: s.rolling(7).std()).fillna(0.0)
    df["rolling_std_14"] = grouped_p.transform(lambda s: s.rolling(14).std()).fillna(0.0)
    df["rolling_std_30"] = grouped_p.transform(lambda s: s.rolling(30).std()).fillna(0.0)

    grouped_pct = df.groupby(group_cols)["pct_change_1"]
    df["rolling_std_pct_5"] = grouped_pct.transform(lambda s: s.rolling(5).std()).fillna(0.0)
    df["rolling_std_pct_14"] = grouped_pct.transform(lambda s: s.rolling(14).std()).fillna(0.0)

    diff_abs = grouped_p.diff().abs()
    df["ewma_abs_diff_5"] = diff_abs.groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df["ewma_abs_diff_14"] = diff_abs.groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    df["ewma_pct_vol_5"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=5).mean()).fillna(0.0)
    df["ewma_pct_vol_14"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.ewm(span=14).mean()).fillna(0.0)

    df["vol_trend_5_10"] = df["past_5_range"] / (df["past_10_range"] + 1e-4)
    df["vol_pct_trend_5_10"] = df["past_5_pct_range"] / (df["past_10_pct_range"] + 1e-4)
    df["vol_std_ratio_5_14"] = df["rolling_std_5"] / (df["rolling_std_14"] + 1e-4)

    is_move = (grouped_p.diff() != 0).astype(int)
    df["nonzero_moves_5"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(5).sum()).fillna(0.0)
    df["nonzero_moves_10"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(10).sum()).fillna(0.0)

    def compute_flat_streak(series):
        is_flat = (series == 0)
        blocks = (~is_flat).cumsum()
        return is_flat.groupby(blocks).cumsum()

    df["streak_zero"] = is_move.groupby([df[c] for c in group_cols]).transform(lambda s: compute_flat_streak(s == 0)).fillna(0.0)
    df["max_daily_pct_jump_5"] = df["abs_pct_change_1"].groupby([df[c] for c in group_cols]).transform(lambda s: s.rolling(5).max()).fillna(0.0)

    df["price_level"] = df[price_col].astype(float)
    rolling_mean_7 = grouped_p.transform(lambda s: s.rolling(7).mean())
    df["price_to_mean_7"] = df[price_col] / (rolling_mean_7 + 1e-4)

    df["month"] = df[date_col].dt.month
    df["day_of_week"] = df[date_col].dt.dayofweek
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)

    if arrival_col in df.columns:
        grouped_arr = df.groupby(group_cols)[arrival_col]
        df["arrival_lag_1"] = grouped_arr.shift(1).fillna(0.0)
        df["arrival_rolling_mean_7"] = grouped_arr.shift(1).transform(lambda s: s.rolling(7).mean()).fillna(0.0)
    else:
        df["arrival_lag_1"] = 0.0
        df["arrival_rolling_mean_7"] = 0.0

    return df
