"""
AgroVision - Feature Engineer
Creates ML-ready features from the cleaned tea mandi dataset.

Features created:
    - Lag features (previous arrivals/prices for same market+variety)
    - Rolling averages (7-day, 30-day)
    - Price features (spread, ratio, volatility)
    - Temporal features (month, quarter, day-of-week encoding)
    - Categorical encoding (state, district, market, variety)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import COLUMNS, FEATURES_DIR, FEATURE_MATRIX_FILE, TARGET_COLUMN


def create_lag_features(df, lags=None):
    """
    Create lag features for arrivals and modal price.
    For each (Market, Variety) group, shift values by N previous records.
    """
    if lags is None:
        lags = [1, 3, 7]

    col = COLUMNS
    group_cols = [col["market"], col["variety"]]
    group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        print("   [WARN] No group columns for lag features")
        return df

    df = df.sort_values(group_cols + [col["date"]])

    # Arrival lags
    if col["arrivals"] in df.columns:
        for lag in lags:
            col_name = f"Arrival_Lag_{lag}"
            df[col_name] = df.groupby(group_cols)[col["arrivals"]].shift(lag)
            print(f"   Created: {col_name}")

    # Modal price lags
    if col["modal_price"] in df.columns:
        for lag in lags:
            col_name = f"Price_Lag_{lag}"
            df[col_name] = df.groupby(group_cols)[col["modal_price"]].shift(lag)
            print(f"   Created: {col_name}")

    return df


def create_rolling_features(df):
    """Create rolling mean and std features for arrivals and price."""
    col = COLUMNS
    group_cols = [col["market"], col["variety"]]
    group_cols = [c for c in group_cols if c in df.columns]

    if not group_cols:
        return df

    df = df.sort_values(group_cols + [col["date"]])

    windows = [7, 14, 30]

    for window in windows:
        # Rolling mean arrivals
        if col["arrivals"] in df.columns:
            col_name = f"Arrival_RollMean_{window}"
            df[col_name] = df.groupby(group_cols)[col["arrivals"]].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            print(f"   Created: {col_name}")

        # Rolling mean price
        if col["modal_price"] in df.columns:
            col_name = f"Price_RollMean_{window}"
            df[col_name] = df.groupby(group_cols)[col["modal_price"]].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            print(f"   Created: {col_name}")

    # Rolling std (volatility) for price - 14 day window
    if col["modal_price"] in df.columns:
        df["Price_Volatility_14"] = df.groupby(group_cols)[col["modal_price"]].transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=2).std()
        )
        print(f"   Created: Price_Volatility_14")

    # Rolling std for arrivals
    if col["arrivals"] in df.columns:
        df["Arrival_Volatility_14"] = df.groupby(group_cols)[col["arrivals"]].transform(
            lambda x: x.shift(1).rolling(window=14, min_periods=2).std()
        )
        print(f"   Created: Arrival_Volatility_14")

    return df


def create_price_features(df):
    """Create derived price features."""
    col = COLUMNS

    # Log-transform arrivals (reduces skew)
    if col["arrivals"] in df.columns:
        df["Log_Arrivals"] = np.log1p(df[col["arrivals"]])
        print(f"   Created: Log_Arrivals")

    # Log-transform modal price
    if col["modal_price"] in df.columns:
        df["Log_Modal_Price"] = np.log1p(df[col["modal_price"]])
        print(f"   Created: Log_Modal_Price")

    # Price change from lag
    if "Price_Lag_1" in df.columns and col["modal_price"] in df.columns:
        df["Price_Change"] = df[col["modal_price"]] - df["Price_Lag_1"]
        df["Price_Pct_Change"] = df["Price_Change"] / (df["Price_Lag_1"] + 1)
        print(f"   Created: Price_Change, Price_Pct_Change")

    # Arrival change from lag
    if "Arrival_Lag_1" in df.columns and col["arrivals"] in df.columns:
        df["Arrival_Change"] = df[col["arrivals"]] - df["Arrival_Lag_1"]
        print(f"   Created: Arrival_Change")

    return df


def create_temporal_features(df):
    """Create cyclical and trend time features."""
    # Cyclical encoding for month (sin/cos to capture seasonality)
    if "Month" in df.columns:
        df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12)
        df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12)
        print(f"   Created: Month_Sin, Month_Cos (cyclical)")

    # Cyclical encoding for day of week
    if "DayOfWeek" in df.columns:
        df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
        df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)
        print(f"   Created: DayOfWeek_Sin, DayOfWeek_Cos")

    # Year trend
    if "Year" in df.columns:
        min_year = df["Year"].min()
        df["Year_Trend"] = df["Year"] - min_year
        print(f"   Created: Year_Trend (base year = {min_year})")

    # Is weekend flag
    if "DayOfWeek" in df.columns:
        df["Is_Weekend"] = (df["DayOfWeek"] >= 5).astype(int)
        print(f"   Created: Is_Weekend")

    return df


def encode_categorical_features(df):
    """Encode categorical columns using LabelEncoder for tree-based models."""
    col = COLUMNS
    categorical_cols = [col["state"], col["district"], col["market"], col["variety"]]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    for c in categorical_cols:
        le = LabelEncoder()
        df[f"{c}_Encoded"] = le.fit_transform(df[c].astype(str))
        n_unique = len(le.classes_)
        print(f"   Encoded: {c} -> {c}_Encoded ({n_unique} unique values)")

    return df


def get_feature_columns(df):
    """
    Identify which columns are usable as ML features.

    Excludes: target variable, raw string columns, date column.

    Returns:
        List of feature column names
    """
    col = COLUMNS

    # Columns to exclude from features
    exclude = {
        col["arrivals"],       # Target variable
        col["state"],          # Raw categorical (use encoded)
        col["district"],       # Raw categorical
        col["market"],         # Raw categorical
        col["variety"],        # Raw categorical
        col["group"],          # Raw categorical (only 1 value: Beverages)
        col["date"],           # Date object
        "Log_Arrivals",        # Derived from target
        "Arrival_Change",      # Derived from target
    }

    features = []
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype in ["int64", "float64", "int32", "float32", "int8"]:
            features.append(c)

    return features


def engineer_features(df):
    """
    Full feature engineering pipeline.

    Args:
        df: Cleaned DataFrame from data_cleaner

    Returns:
        DataFrame with engineered features, also saved to disk
    """
    print("\n" + "=" * 60)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 60)

    print("\n[FEAT] Creating lag features...")
    df = create_lag_features(df)

    print("\n[FEAT] Creating rolling features...")
    df = create_rolling_features(df)

    print("\n[FEAT] Creating price features...")
    df = create_price_features(df)

    print("\n[FEAT] Creating temporal features...")
    df = create_temporal_features(df)

    print("\n[FEAT] Encoding categorical features...")
    df = encode_categorical_features(df)

    # Drop rows with NaN from lag/rolling features
    feature_cols = get_feature_columns(df)
    before = len(df)
    df = df.dropna(subset=[c for c in feature_cols if c in df.columns])
    dropped = before - len(df)
    if dropped > 0:
        print(f"\n   Dropped {dropped} rows with NaN (from lags/rolling) -> {len(df):,} rows")

    # Final feature list
    feature_cols = get_feature_columns(df)
    print(f"\n   [STATS] Total feature columns: {len(feature_cols)}")
    for i, fc in enumerate(feature_cols):
        print(f"      {i+1:2d}. {fc}")

    # Save
    os.makedirs(FEATURES_DIR, exist_ok=True)
    df.to_csv(FEATURE_MATRIX_FILE, index=False)
    print(f"\n[SAVE] Saved feature matrix to: {FEATURE_MATRIX_FILE}")

    return df
