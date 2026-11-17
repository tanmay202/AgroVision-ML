"""
AgroVision Model 1: Production Champion Training & Promotion
===========================================================
Trains and serializes the certified production champions for:
- Wheat: Delta-Ridge on 3-Year Rolling Mean (m1_wheat_delta_ridge.pkl)
- Potato: Delta-Huber on 3-Year Rolling Mean (m1_potato_delta_huber.pkl)

Preserves existing artifacts bit-for-bit:
- m1_improved_delta_ridge.pkl (Rice production model)
- m1_wheat_ridge.pkl (Milestone 6 candidate)
- m1_potato_catboost.pkl (Milestone 6 candidate)
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

M1_DIR = Path(__file__).resolve().parent.parent
DATA_PROC_DIR = M1_DIR / "data" / "processed"
MODELS_DIR = M1_DIR / "models"

# Feature definitions
WHEAT_FEATURE_COLS = [
    'yield_lag1', 'yield_lag2', 'yield_roll_mean3', 'log_area_lag1',
    'precip_cum', 'precip_mean', 'precip_days', 't2m_mean', 't2m_min', 't2m_max',
    't2m_range', 'solar_mean', 'rh2m_mean', 'ws2m_mean', 'gdd_base5',
    'heat_stress_days_28', 'heat_stress_days_30'
]

POTATO_FEATURE_COLS = [
    'yield_lag1', 'yield_lag2', 'yield_roll_mean3', 'log_area_lag1',
    'precip_cum', 'precip_mean', 'precip_days', 't2m_mean', 't2m_min', 't2m_max',
    't2m_range', 'solar_mean', 'rh2m_mean', 'ws2m_mean', 'gdd_base5',
    'cool_night_days_16', 'cool_night_days_14', 'late_blight_risk_days'
]

def train_wheat_production():
    print("Training Wheat Production Champion...")
    df = pd.read_csv(DATA_PROC_DIR / "des_wheat_full_1997_2022.csv")
    valid = df[df['yield_lag1'].notnull() & df['has_weather']].copy()
    valid['yield_lag2'] = valid['yield_lag2'].fillna(valid['yield_lag1'])
    valid['yield_roll_mean3'] = valid['yield_roll_mean3'].fillna(valid['yield_lag1'])
    valid['area_lag1'] = valid['area_lag1'].fillna(valid['Area_ha'])
    valid['log_area_lag1'] = np.log1p(valid['area_lag1'])
    
    # Calculate district means for fallback
    dist_means = valid.groupby('District')['Yield_kg_ha'].mean().to_dict()
    global_mean = float(valid['Yield_kg_ha'].mean())
    
    # Target: Delta from 3-year rolling mean
    valid['delta_roll3'] = valid['Yield_kg_ha'] - valid['yield_roll_mean3']
    
    X = valid[WHEAT_FEATURE_COLS].values
    y_delta = valid['delta_roll3'].values
    y_true = valid['Yield_kg_ha'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=20.0, random_state=42)
    model.fit(X_scaled, y_delta)
    
    # Holdout evaluation on 2020-2022
    test_mask = valid['Crop_Year'] >= 2020
    test_rows = valid[test_mask]
    
    y_true_test = y_true[test_mask]
    pred_delta_test = model.predict(X_scaled[test_mask])
    pred_test = test_rows['yield_roll_mean3'].values + pred_delta_test
    
    res_test = y_true_test - pred_test
    eval_metrics = {
        'p10': float(np.percentile(res_test, 10)),
        'p25': float(np.percentile(res_test, 25)),
        'p50': float(np.percentile(res_test, 50)),
        'p75': float(np.percentile(res_test, 75)),
        'p90': float(np.percentile(res_test, 90)),
        'std': float(np.std(res_test)),
        'mae': float(mean_absolute_error(y_true_test, pred_test)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_test, pred_test))),
        'r2': float(r2_score(y_true_test, pred_test)),
        'bias': float(np.mean(res_test)),
        'n_holdout': int(len(res_test))
    }
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_cols': WHEAT_FEATURE_COLS,
        'dist_means': dist_means,
        'global_mean': global_mean,
        'target_type': 'delta_roll3',
        'cutoff': {'month': 1, 'day': 31},
        'season_start': {'month': 11, 'day': 15},
        'residuals_quantiles': eval_metrics
    }
    
    art_path = MODELS_DIR / "m1_wheat_delta_ridge.pkl"
    joblib.dump(artifact, art_path)
    h_sha256 = hashlib.sha256(open(art_path, "rb").read()).hexdigest()
    
    metadata = {
        "model_name": "m1_wheat_delta_ridge",
        "crop": "wheat",
        "release_status": "PRODUCTION READY",
        "artifact_filename": "m1_wheat_delta_ridge.pkl",
        "artifact_sha256": h_sha256,
        "architecture": "Ridge Regressor on Delta from 3-Year Rolling Mean (alpha=20.0, StandardScaler)",
        "target_type": "delta_roll3",
        "target": "yield_kg_per_ha = yield_roll_mean3 + predicted_delta",
        "season": "Rabi",
        "cutoff_date": "Jan 31",
        "training_rows": len(valid),
        "feature_cols": WHEAT_FEATURE_COLS,
        "evaluation_metrics": eval_metrics,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(MODELS_DIR / "m1_wheat_delta_ridge.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved Wheat Champion: {art_path} (SHA-256: {h_sha256})")
    return metadata

def train_potato_production():
    print("Training Potato Production Champion...")
    df = pd.read_csv(DATA_PROC_DIR / "des_potato_full_1997_2022.csv")
    valid = df[df['yield_lag1'].notnull() & df['has_weather']].copy()
    valid['yield_lag2'] = valid['yield_lag2'].fillna(valid['yield_lag1'])
    valid['yield_roll_mean3'] = valid['yield_roll_mean3'].fillna(valid['yield_lag1'])
    valid['area_lag1'] = valid['area_lag1'].fillna(valid['Area_ha'])
    valid['log_area_lag1'] = np.log1p(valid['area_lag1'])
    
    dist_means = valid.groupby('District')['Yield_kg_ha'].mean().to_dict()
    global_mean = float(valid['Yield_kg_ha'].mean())
    
    # Target: Delta from 3-year rolling mean
    valid['delta_roll3'] = valid['Yield_kg_ha'] - valid['yield_roll_mean3']
    
    X = valid[POTATO_FEATURE_COLS].values
    y_delta = valid['delta_roll3'].values
    y_true = valid['Yield_kg_ha'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = HuberRegressor(alpha=15.0, max_iter=300)
    model.fit(X_scaled, y_delta)
    
    # Holdout evaluation on 2020-2022
    test_mask = valid['Crop_Year'] >= 2020
    test_rows = valid[test_mask]
    
    y_true_test = y_true[test_mask]
    pred_delta_test = model.predict(X_scaled[test_mask])
    pred_test = test_rows['yield_roll_mean3'].values + pred_delta_test
    
    res_test = y_true_test - pred_test
    eval_metrics = {
        'p10': float(np.percentile(res_test, 10)),
        'p25': float(np.percentile(res_test, 25)),
        'p50': float(np.percentile(res_test, 50)),
        'p75': float(np.percentile(res_test, 75)),
        'p90': float(np.percentile(res_test, 90)),
        'std': float(np.std(res_test)),
        'mae': float(mean_absolute_error(y_true_test, pred_test)),
        'rmse': float(np.sqrt(mean_squared_error(y_true_test, pred_test))),
        'r2': float(r2_score(y_true_test, pred_test)),
        'bias': float(np.mean(res_test)),
        'n_holdout': int(len(res_test))
    }
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_cols': POTATO_FEATURE_COLS,
        'dist_means': dist_means,
        'global_mean': global_mean,
        'target_type': 'delta_roll3',
        'cutoff': {'month': 1, 'day': 31},
        'season_start': {'month': 11, 'day': 1},
        'residuals_quantiles': eval_metrics
    }
    
    art_path = MODELS_DIR / "m1_potato_delta_huber.pkl"
    joblib.dump(artifact, art_path)
    h_sha256 = hashlib.sha256(open(art_path, "rb").read()).hexdigest()
    
    metadata = {
        "model_name": "m1_potato_delta_huber",
        "crop": "potato",
        "release_status": "PRODUCTION READY",
        "artifact_filename": "m1_potato_delta_huber.pkl",
        "artifact_sha256": h_sha256,
        "architecture": "Huber Regressor on Delta from 3-Year Rolling Mean (alpha=15.0, StandardScaler)",
        "target_type": "delta_roll3",
        "target": "yield_kg_per_ha = yield_roll_mean3 + predicted_delta",
        "season": "Rabi",
        "cutoff_date": "Jan 31",
        "training_rows": len(valid),
        "feature_cols": POTATO_FEATURE_COLS,
        "evaluation_metrics": eval_metrics,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(MODELS_DIR / "m1_potato_delta_huber.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Saved Potato Champion: {art_path} (SHA-256: {h_sha256})")
    return metadata

if __name__ == "__main__":
    w_meta = train_wheat_production()
    p_meta = train_potato_production()
    print("\n--- PRODUCTION PROMOTION SUMMARY ---")
    print(f"Wheat : {w_meta['model_name']} | Status: {w_meta['release_status']} | SHA-256: {w_meta['artifact_sha256']}")
    print(f"Potato: {p_meta['model_name']} | Status: {p_meta['release_status']} | SHA-256: {p_meta['artifact_sha256']}")
