import pytest
import sys
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline
from config import cleaned_data_path, set_commodity, DATE_COLUMN, PRICE_COLUMN, GROUP_COLUMNS, ARRIVAL_COLUMN

set_commodity("rice")

def generate_training_pipeline_features(raw_df):
    """
    Replicate the exact feature generation logic used in training (Step 7 experiment).
    This serves as the ground truth.
    """
    df = raw_df.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(by=GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)
    
    grouped_price = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]
    
    # 1. Base pipeline features (matching create_lags, create_rolling, etc.)
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
    
    # 2. Advanced features (Step 7)
    df["price_change_1"] = grouped_price.diff(1)
    df["price_change_7"] = grouped_price.diff(7)
    df["price_change_14"] = grouped_price.diff(14)
    df["accel_1_7"] = df["price_change_1"] - df["price_change_7"]
    df["accel_1_14"] = df["price_change_1"] - df["price_change_14"]
    
    def get_slope_weights(window):
        x = np.arange(window) - (window - 1) / 2.0
        return x / np.sum(x**2)

    w7 = get_slope_weights(7)
    df["trend_slope_7"] = sum(w7[i] * grouped_price.shift(6 - i) for i in range(7))
    w14 = get_slope_weights(14)
    df["trend_slope_14"] = sum(w14[i] * grouped_price.shift(13 - i) for i in range(14))
    w30 = get_slope_weights(30)
    df["trend_slope_30"] = sum(w30[i] * grouped_price.shift(29 - i) for i in range(30))

    df["ewma_7"] = grouped_price.transform(lambda x: x.ewm(span=7, adjust=False).mean())
    df["ewma_14"] = grouped_price.transform(lambda x: x.ewm(span=14, adjust=False).mean())
    df["ewma_30"] = grouped_price.transform(lambda x: x.ewm(span=30, adjust=False).mean())
    df["rolling_median_7"] = grouped_price.transform(lambda x: x.rolling(7).median())
    df["rolling_median_14"] = grouped_price.transform(lambda x: x.rolling(14).median())
    df["rolling_median_30"] = grouped_price.transform(lambda x: x.rolling(30).median())
    
    df["gap_mean_7"] = df[PRICE_COLUMN] - df["rolling_mean_7"]
    df["gap_mean_14"] = df[PRICE_COLUMN] - df["rolling_mean_14"]
    df["zscore_7"] = (df[PRICE_COLUMN] - df["rolling_mean_7"]) / (df["rolling_std_7"] + 1e-5)
    df["zscore_14"] = (df[PRICE_COLUMN] - df["rolling_mean_14"]) / (df["rolling_std_14"] + 1e-5)
    
    def compute_streak(df, condition_series):
        group_keys = [df[c] for c in GROUP_COLUMNS]
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

@pytest.fixture(scope="module")
def dataset_and_pipeline():
    raw_df = pd.read_csv(cleaned_data_path())
    raw_df[DATE_COLUMN] = pd.to_datetime(raw_df[DATE_COLUMN])
    # Build ground truth features
    truth_df = generate_training_pipeline_features(raw_df)
    pipeline = RiceInferencePipeline()
    return raw_df, truth_df, pipeline

def test_feature_parity_multiple_dates(dataset_and_pipeline):
    raw_df, truth_df, pipeline = dataset_and_pipeline
    
    # Pick a few specific known Market/Variety/Date from the dataset where enough history exists
    # Find rows with lag_30 not null
    valid_rows = truth_df.dropna(subset=["lag_30", "rolling_mean_30"]).sample(5, random_state=42)
    
    for _, row in valid_rows.iterrows():
        market = row["Market Name"]
        variety = row["Variety"]
        pred_date = row[DATE_COLUMN]
        
        # Inference prediction
        inf_clean = pipeline._validate_and_clean_input(raw_df, market, variety, pred_date)
        inf_feat = pipeline._generate_features(inf_clean).iloc[-1]
        
        # 1. Compare all 48 features
        for f in pipeline.features_list:
            train_val = float(row[f]) if pd.notna(row[f]) else np.nan
            inf_val = float(inf_feat[f]) if pd.notna(inf_feat[f]) else np.nan
            
            if pd.isna(train_val) and pd.isna(inf_val):
                continue
            assert not pd.isna(train_val), f"Train NaN for {f}"
            assert not pd.isna(inf_val), f"Inf NaN for {f}"
            
            diff = abs(train_val - inf_val)
            assert diff < 1e-5, f"Mismatch for {f} on {pred_date}: Train={train_val}, Inf={inf_val}"
            
        # 2. Check Feature Ordering
        ordered_inf_keys = list(inf_feat[pipeline.features_list].index)
        assert ordered_inf_keys == pipeline.features_list, "Feature ordering mismatch"
        
        # 3. Model Prediction Parity
        # Predict using manual extraction from truth_df
        X_train_fmt = row[pipeline.features_list].to_frame().T
        train_pred_change = float(pipeline.price_model.predict(X_train_fmt)[0])
        train_pred_price = row[PRICE_COLUMN] + train_pred_change
        
        inf_res = pipeline.predict(raw_df, market, variety, pred_date)
        
        assert abs(train_pred_price - inf_res["predicted_modal_price"]) < 1e-2
        
        # 4. Volatility Parity
        past_5 = raw_df[(raw_df["Market Name"] == market) & 
                        (raw_df["Variety"] == variety) & 
                        (raw_df[DATE_COLUMN] <= pred_date)].sort_values(DATE_COLUMN)[PRICE_COLUMN].tail(5)
        expected_range = float(past_5.max() - past_5.min())
        assert abs(expected_range - inf_res["predicted_volatility_range"]) < 1e-4
        
        if expected_range <= 0.5:
            expected_class = "LOW"
        elif expected_range <= 50.0:
            expected_class = "MEDIUM"
        else:
            expected_class = "HIGH"
            
        assert expected_class == inf_res["predicted_volatility_class"]

def test_point_in_time_behavior(dataset_and_pipeline):
    raw_df, truth_df, pipeline = dataset_and_pipeline
    
    # Take a row
    valid_rows = truth_df.dropna(subset=["lag_30"]).sample(1, random_state=99)
    row = valid_rows.iloc[0]
    market = row["Market Name"]
    variety = row["Variety"]
    pred_date = row[DATE_COLUMN]
    
    # Inference normally
    res_normal = pipeline.predict(raw_df, market, variety, pred_date)
    
    # Inject fake FUTURE data
    future_date = pred_date + pd.Timedelta(days=5)
    fake_row = pd.DataFrame([{
        "Market Name": market,
        "Variety": variety,
        DATE_COLUMN: future_date,
        PRICE_COLUMN: 99999.0,
        ARRIVAL_COLUMN: 99999.0
    }])
    raw_df_poisoned = pd.concat([raw_df, fake_row], ignore_index=True)
    
    # Inference on poisoned data
    res_poisoned = pipeline.predict(raw_df_poisoned, market, variety, pred_date)
    
    # Must be identical
    assert res_normal["predicted_modal_price"] == res_poisoned["predicted_modal_price"]
    assert res_normal["predicted_volatility_class"] == res_poisoned["predicted_volatility_class"]
