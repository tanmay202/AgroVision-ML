"""
Phase 5: Direct Calendar Horizon Models (14-Day and 30-Day)
For Wheat and Potato.
Uses horizon_targets.py for:
  - 14-day: first observation in [T+14, T+17]
  - 30-day: first observation in [T+30, T+33]
Applies temporal purging at split boundary.
Trains independent models and computes empirical holdout residuals.
"""

import sys
sys.path.insert(0, "member2-price-volatility/src")

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from features_v2 import generate_candidate_price_features
from horizon_targets import compute_single_horizon_targets, apply_horizon_temporal_purge

def run_horizon_models(commodity):
    print(f"\n=======================================================")
    print(f"=== PHASE 5 DIRECT HORIZON FORECASTING: {commodity.upper()} ===")
    print(f"=======================================================")
    
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    
    print(f"Generating features for {len(clean_df):,} rows...")
    feats_df = generate_candidate_price_features(clean_df)
    
    group_cols = ["Market Name", "Variety"]
    date_col = "Reported Date"
    price_col = "Modal Price (Rs./Quintal)"
    
    # 30-history requirement
    feats_df[date_col] = pd.to_datetime(feats_df[date_col])
    feats_df = feats_df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    hist_mask = feats_df.groupby(group_cols).cumcount() >= 30
    ready_df = feats_df[hist_mask].copy().reset_index(drop=True)
    
    # Get historical split date
    train_orig = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    split_date = pd.to_datetime(train_orig[date_col]).max()
    print(f"Historical split date: {split_date.date()}")
    
    cand_feats = [c for c in ready_df.columns if c not in [
        "State Name", "District Name", "Market Name", "Variety", "Group",
        date_col, price_col, "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)",
    ]]
    
    results = {}
    
    for horizon in [14, 30]:
        h_prefix = f"{horizon}d"
        print(f"\n--- HORIZON: {horizon} CALENDAR DAYS ---")
        
        # 1. Target generation
        h_df = compute_single_horizon_targets(
            ready_df,
            horizon_days=horizon,
            max_tolerance_days=3,
            group_cols=group_cols,
            date_col=date_col,
            price_col=price_col,
        )
        
        target_col = f"future_price_change_{h_prefix}"
        future_price_col = f"future_modal_price_{h_prefix}"
        actual_h_col = f"actual_horizon_{h_prefix}"
        target_date_col = f"target_date_{h_prefix}"
        
        # Valid targets
        valid_h = h_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
        print(f"Total rows with valid {h_prefix} target: {len(valid_h):,} / {len(ready_df):,} ({len(valid_h)/len(ready_df)*100:.1f}%)")
        print(f"Actual calendar days offset: Mean={valid_h[actual_h_col].mean():.2f}d, Min={valid_h[actual_h_col].min():.0f}d, Max={valid_h[actual_h_col].max():.0f}d")
        
        # Split train and holdout
        tr_raw = valid_h[valid_h[date_col] <= split_date].copy().reset_index(drop=True)
        ho_data = valid_h[valid_h[date_col] > split_date].copy().reset_index(drop=True)
        
        # 2. Temporal Purging on Train: remove samples whose target_date >= split_date
        tr_data, purged_count = apply_horizon_temporal_purge(
            tr_raw,
            split_date=str(split_date.date()),
            horizon_days=horizon,
            max_tolerance_days=3,
            date_col=date_col,
        )
        print(f"Train rows before purge: {len(tr_raw):,}, Purged leaking rows: {purged_count:,}, Train rows after purge: {len(tr_data):,}")
        print(f"Holdout rows: {len(ho_data):,}")
        
        if len(tr_data) < 1000 or len(ho_data) < 500:
            print(f"INSUFFICIENT DATA for {commodity} at horizon {horizon}d. Skipping model creation.")
            continue
            
        # 3. Train Model
        # CatBoost MAE
        m = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, loss_function="MAE", verbose=0, random_seed=42)
        m.fit(tr_data[cand_feats], tr_data[target_col])
        
        pred_chg = m.predict(ho_data[cand_feats])
        cur_p = ho_data[price_col].to_numpy()
        act_p = ho_data[future_price_col].to_numpy()
        pred_p = cur_p + pred_chg
        
        # Metrics
        mae = mean_absolute_error(act_p, pred_p)
        rmse = root_mean_squared_error(act_p, pred_p)
        r2 = r2_score(act_p, pred_p)
        mape = np.mean(np.abs((act_p - pred_p) / act_p)) * 100.0
        
        p_mae = mean_absolute_error(act_p, cur_p)
        p_rmse = root_mean_squared_error(act_p, cur_p)
        p_r2 = r2_score(act_p, cur_p)
        p_mape = np.mean(np.abs((act_p - cur_p) / act_p)) * 100.0
        
        print(f"Model Results ({h_prefix}):")
        print(f"  MAE:  Rs.{mae:.2f} (Persistence: Rs.{p_mae:.2f})")
        print(f"  RMSE: Rs.{rmse:.2f} (Persistence: Rs.{p_rmse:.2f})")
        print(f"  R2:   {r2:.4f} (Persistence: {p_r2:.4f})")
        print(f"  MAPE: {mape:.2f}% (Persistence: {p_mape:.2f}%)")
        
        # 4. Residuals & Prediction Intervals
        residuals = act_p - pred_p
        res_quantiles = {
            "p05": float(np.percentile(residuals, 5)),
            "p10": float(np.percentile(residuals, 10)),
            "p25": float(np.percentile(residuals, 25)),
            "p50": float(np.percentile(residuals, 50)),
            "p75": float(np.percentile(residuals, 75)),
            "p90": float(np.percentile(residuals, 90)),
            "p95": float(np.percentile(residuals, 95)),
            "std": float(np.std(residuals)),
            "mean": float(np.mean(residuals)),
            "n_holdout": int(len(residuals)),
            "holdout_start": str(ho_data[date_col].min().date()),
            "holdout_end": str(ho_data[date_col].max().date()),
        }
        print(f"Residual quantiles: p10={res_quantiles['p10']:.2f}, p50={res_quantiles['p50']:.2f}, p90={res_quantiles['p90']:.2f}")
        
        results[h_prefix] = {
            "model": m,
            "features": cand_feats,
            "metrics": {
                "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
                "persistence_mae": p_mae, "persistence_rmse": p_rmse, "persistence_r2": p_r2,
            },
            "residuals": res_quantiles,
            "sample_sizes": {
                "train_rows": len(tr_data),
                "purged_rows": purged_count,
                "holdout_rows": len(ho_data),
            }
        }
        
    return results

if __name__ == "__main__":
    import sys
    comm = sys.argv[1] if len(sys.argv) > 1 else "wheat"
    run_horizon_models(comm)
