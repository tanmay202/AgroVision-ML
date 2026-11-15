"""
Phase 3 & 4: Systematic Price Forecasting Experiments
Performs:
1. Feature Generation & Ablation across candidate feature groups
2. Model Exploration: XGBoost, CatBoost, HistGradientBoosting, Random Forest
3. Loss functions: pseudo-Huber, squared error, absolute error
4. Hyperparameter tuning via 3-Fold Chronological Walk-Forward Cross Validation
5. Dead-band sparsity analysis (zero-movement thresholding)
6. Out-of-sample evaluation on the untouched chronological holdout
7. Comparison against Persistence and V1 baseline
"""

import sys
sys.path.insert(0, "member2-price-volatility/src")

import json
import joblib
import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from features_v2 import generate_candidate_price_features

def run_price_experiments(commodity):
    print(f"\n=======================================================")
    print(f"=== PRICE FORECASTING EXPERIMENTS: {commodity.upper()} ===")
    print(f"=======================================================")
    
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    art_dir = Path(f"member2-price-volatility/artifacts/{commodity}")
    
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    print(f"Generating candidate features for {len(clean_df):,} rows...")
    t0 = time.time()
    feats_df = generate_candidate_price_features(clean_df)
    print(f"Features generated in {time.time()-t0:.1f}s")
    
    # Add target: next valid observation within 7 days
    group_cols = ["Market Name", "Variety"]
    date_col = "Reported Date"
    price_col = "Modal Price (Rs./Quintal)"
    
    feats_df[date_col] = pd.to_datetime(feats_df[date_col])
    feats_df = feats_df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    
    # Target
    feats_df["future_modal_price"] = feats_df.groupby(group_cols)[price_col].shift(-1)
    next_date = feats_df.groupby(group_cols)[date_col].shift(-1)
    feats_df["days_to_next"] = (next_date - feats_df[date_col]).dt.days
    
    # Filter by 30 history and 1 <= days_to_next <= 7
    hist_mask = feats_df.groupby(group_cols).cumcount() >= 30
    target_mask = feats_df["future_modal_price"].notna() & (feats_df["days_to_next"] >= 1) & (feats_df["days_to_next"] <= 7)
    
    valid_df = feats_df[hist_mask & target_mask].copy().reset_index(drop=True)
    valid_df["future_price_change"] = valid_df["future_modal_price"] - valid_df[price_col]
    print(f"Total model-ready rows: {len(valid_df):,}")
    
    # Split using the exact historical split date from V1
    train_orig = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    split_date = pd.to_datetime(train_orig[date_col]).max()
    print(f"Historical split date: {split_date.date()}")
    
    train_data = valid_df[valid_df[date_col] <= split_date].copy().reset_index(drop=True)
    holdout_data = valid_df[valid_df[date_col] > split_date].copy().reset_index(drop=True)
    print(f"Train rows: {len(train_data):,}, Holdout rows: {len(holdout_data):,}")
    
    # Define candidate feature sets for ablation
    v1_features = [
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "year", "month", "day_of_week", "week_of_year",
        "arrival_lag_1", "arrival_lag_7",
        "arrival_rolling_mean_7", "arrival_rolling_mean_14",
        "price_pct_change", "arrival_pct_change"
    ]
    # Filter to available in data
    v1_features = [f for f in v1_features if f in train_data.columns]
    
    candidate_features_all = [
        # Lags
        "lag_1", "lag_2", "lag_3", "lag_5", "lag_7", "lag_10", "lag_14", "lag_21", "lag_30",
        # Rolling
        "rolling_mean_3", "rolling_mean_5", "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_5", "rolling_std_7", "rolling_std_14", "rolling_std_30",
        "rolling_median_7", "rolling_median_14", "rolling_min_7", "rolling_max_7",
        # Momentum & Accel
        "price_change_1", "price_change_3", "price_change_7", "price_change_14",
        "momentum_3", "momentum_7", "momentum_14",
        "acceleration_3_7", "acceleration_7_14", "accel_1_7", "accel_1_14",
        # Trend & EWMA
        "trend_slope_7", "trend_slope_14", "trend_slope_30",
        "ewma_5", "ewma_7", "ewma_14", "ewma_30",
        "price_to_rolling_mean_7", "price_to_rolling_mean_14",
        "gap_mean_7", "gap_mean_14", "price_zscore_7", "price_zscore_14",
        # Arrivals
        "arrival_lag_1", "arrival_lag_3", "arrival_lag_7",
        "arrival_rolling_mean_7", "arrival_rolling_mean_14", "arrival_rolling_std_7",
        "arrival_pct_change", "price_pct_change", "arrival_price_interaction",
        # Time & Seasonality (no day of month!)
        "year", "month", "day_of_week", "week_of_year", "quarter", "annual_sin", "annual_cos",
        # Price structure
        "price_spread", "price_position_in_range", "min_to_modal_ratio", "modal_to_max_ratio",
        # Streaks
        "move_count_7", "move_count_14", "move_count_30", "pos_count_7", "neg_count_7",
        "streak_zero", "last_non_zero_move", "last_move_abs"
    ]
    candidate_features_all = [f for f in candidate_features_all if f in train_data.columns]
    print(f"V1 features count: {len(v1_features)}, Candidate features count: {len(candidate_features_all)}")
    
    # ------------------------------------------------------------
    # 3-Fold Chronological Walk-Forward Cross-Validation on Train
    # ------------------------------------------------------------
    print("\n--- 3-FOLD CHRONOLOGICAL WALK-FORWARD VALIDATION ---")
    train_dates = train_data[date_col].sort_values().reset_index(drop=True)
    # Split train into 4 chronological slices:
    # Slice 0 (base), Slice 1, Slice 2, Slice 3
    q25 = train_dates.quantile(0.40)
    q50 = train_dates.quantile(0.60)
    q75 = train_dates.quantile(0.80)
    
    cv_splits = [
        (train_data[date_col] < q25, (train_data[date_col] >= q25) & (train_data[date_col] < q50)),
        (train_data[date_col] < q50, (train_data[date_col] >= q50) & (train_data[date_col] < q75)),
        (train_data[date_col] < q75, train_data[date_col] >= q75),
    ]
    
    models_to_test = {
        "XGBoost_Default_V1Feats": (XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, n_jobs=-1), v1_features),
        "XGBoost_PseudoHuber_AllFeats": (XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, objective="reg:pseudohubererror", subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1), candidate_features_all),
        "XGBoost_MAE_AllFeats": (XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03, objective="reg:absoluteerror", subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1), candidate_features_all),
        "CatBoost_MAE_AllFeats": (CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, loss_function="MAE", verbose=0, random_seed=42), candidate_features_all),
        "HistGB_AllFeats": (HistGradientBoostingRegressor(max_iter=300, max_depth=6, learning_rate=0.03, loss="absolute_error", random_state=42), candidate_features_all),
        "RandomForest_AllFeats": (RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs=-1, random_state=42), candidate_features_all),
    }
    
    cv_results = {}
    for name, (model_template, feats) in models_to_test.items():
        fold_maes = []
        fold_rmses = []
        p_maes = []
        p_rmses = []
        
        for tr_idx, val_idx in cv_splits:
            tr_df = train_data[tr_idx]
            vl_df = train_data[val_idx]
            
            X_tr, y_tr = tr_df[feats], tr_df["future_price_change"]
            X_vl, y_vl = vl_df[feats], vl_df["future_price_change"]
            
            cur_price_vl = vl_df[price_col].to_numpy()
            act_future_vl = vl_df["future_modal_price"].to_numpy()
            
            # clone model
            from sklearn.base import clone
            m = clone(model_template)
            m.fit(X_tr, y_tr)
            
            pred_chg = m.predict(X_vl)
            pred_price = cur_price_vl + pred_chg
            
            fold_maes.append(mean_absolute_error(act_future_vl, pred_price))
            fold_rmses.append(root_mean_squared_error(act_future_vl, pred_price))
            p_maes.append(mean_absolute_error(act_future_vl, cur_price_vl))
            p_rmses.append(root_mean_squared_error(act_future_vl, cur_price_vl))
            
        cv_results[name] = {
            "cv_mae": np.mean(fold_maes),
            "cv_rmse": np.mean(fold_rmses),
            "p_cv_mae": np.mean(p_maes),
            "p_cv_rmse": np.mean(p_rmses),
        }
        print(f"   {name:30s} -> CV MAE: Rs.{np.mean(fold_maes):.2f} (Persist: Rs.{np.mean(p_maes):.2f}) | RMSE: Rs.{np.mean(fold_rmses):.2f} (Persist: Rs.{np.mean(p_rmses):.2f})")

    # Save CV results
    Path(f"reports/{commodity}/cv_price_results.json").write_text(json.dumps(cv_results, indent=2))
    return valid_df, train_data, holdout_data, candidate_features_all, v1_features

if __name__ == "__main__":
    import sys
    comm = sys.argv[1] if len(sys.argv) > 1 else "wheat"
    run_price_experiments(comm)
