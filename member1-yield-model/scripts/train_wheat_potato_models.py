"""
AgroVision Model 1: Training and Artifact Generation for Wheat and Potato Yield Models
======================================================================================
Trains crop-specific champion models:
- Wheat: Regularized Ridge Regressor (Cutoff Jan 31)
- Potato: CatBoost Regressor (Cutoff Jan 31)

Saves standalone artifacts in member1-yield-model/models/ alongside JSON metadata.
Guarantees bit-for-bit isolation: DOES NOT TOUCH existing Rice production models.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import catboost as cb

# Base paths
SCRIPTS_DIR = Path(__file__).resolve().parent
M1_DIR = SCRIPTS_DIR.parent
DATA_RAW_DIR = M1_DIR / "data" / "raw"
DATA_PROC_DIR = M1_DIR / "data" / "processed"
MODELS_DIR = M1_DIR / "models"
WEATHER_DIR = DATA_RAW_DIR / "weather"

DISTRICT_MAP = {
    '24 PARAGANAS NORTH': 'North 24 Parganas',
    '24 PARAGANAS SOUTH': 'South 24 Parganas',
    'ALIPURDUAR': 'Alipurduar',
    'BANKURA': 'Bankura',
    'BARDHAMAN': 'Purba Bardhaman',
    'BIRBHUM': 'Birbhum',
    'COOCHBEHAR': 'Cooch Behar',
    'DARJEELING': 'Darjeeling',
    'DINAJPUR DAKSHIN': 'Dakshin Dinajpur',
    'DINAJPUR UTTAR': 'Uttar Dinajpur',
    'HOOGHLY': 'Hooghly',
    'HOWRAH': 'Howrah',
    'JALPAIGURI': 'Jalpaiguri',
    'JHARGRAM': 'Jhargram',
    'KALIMPONG': 'Kalimpong',
    'MALDAH': 'Maldah',
    'MEDINIPUR EAST': 'Purba Medinipur',
    'MEDINIPUR WEST': 'Paschim Medinipur',
    'MURSHIDABAD': 'Murshidabad',
    'NADIA': 'Nadia',
    'PASCHIM BARDHAMAN': 'Paschim Bardhaman',
    'PURBA BARDHAMAN': 'Purba Bardhaman',
    'PURULIA': 'Purulia',
}

WEATHER_FILE_MAP = {
    'alipurduar': 'Alipurduar',
    'bankura': 'Bankura',
    'birbhum': 'Birbhum',
    'cooch_behar': 'Cooch Behar',
    'dakshin_dinajpur': 'Dakshin Dinajpur',
    'darjeeling': 'Darjeeling',
    'hooghly': 'Hooghly',
    'howrah': 'Howrah',
    'jalpaiguri': 'Jalpaiguri',
    'jhargram': 'Jhargram',
    'kalimpong': 'Kalimpong',
    'maldah': 'Maldah',
    'murshidabad': 'Murshidabad',
    'nadia': 'Nadia',
    'north_24_parganas': 'North 24 Parganas',
    'paschim_bardhaman': 'Paschim Bardhaman',
    'paschim_medinipur': 'Paschim Medinipur',
    'purba_bardhaman': 'Purba Bardhaman',
    'purba_medinipur': 'Purba Medinipur',
    'purulia': 'Purulia',
    'south_24_parganas': 'South 24 Parganas',
    'uttar_dinajpur': 'Uttar Dinajpur',
}

def load_daily_weather():
    params = ['PRECTOTCORR', 'T2M', 'T2M_MIN', 'T2M_MAX', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M']
    records = []
    for jf in sorted(WEATHER_DIR.glob('*.json')):
        stem = jf.stem
        if stem not in WEATHER_FILE_MAP:
            continue
        dist = WEATHER_FILE_MAP[stem]
        with open(jf, 'r') as f:
            d = json.load(f)
        pdata = d['properties']['parameter']
        dates = list(pdata[params[0]].keys())
        for dt in dates:
            rec = {'District': dist, 'Date': dt}
            for p in params:
                val = pdata[p].get(dt, np.nan)
                rec[p] = val if val != -999 else np.nan
            records.append(rec)
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    return df

def clean_raw_data(filepath, crop_name):
    df = pd.read_csv(filepath)
    df['District_Raw'] = df['District_Name'].str.strip()
    df['District'] = df['District_Raw'].map(lambda x: DISTRICT_MAP.get(x, x))
    
    thresh = 10.0 if crop_name == 'Wheat' else 100.0
    df['yield_kg_per_ha'] = np.where(df['yield'] < thresh, df['yield'] * 1000.0, df['yield'] * 10.0)
    df['yield_tonnes_per_ha'] = df['yield_kg_per_ha'] / 1000.0
    df['production_tonnes'] = np.where(df['yield'] < thresh, df['Production'], df['Production'] / 100.0)
    df['area_ha'] = df['Area']
    
    df = df.drop_duplicates(subset=['District_Raw', 'Crop_Year', 'Season'], keep='first').copy()
    return df

def extract_weather(weather_df, district, year, start_m, start_d, cutoff_m, cutoff_d):
    start_date = pd.Timestamp(year - 1, start_m, start_d)
    cutoff_date = pd.Timestamp(year, cutoff_m, cutoff_d)
    
    sub = weather_df[(weather_df['District'] == district) & 
                     (weather_df['Date'] >= start_date) & 
                     (weather_df['Date'] <= cutoff_date)]
    if len(sub) < 30:
        return None
    
    t2m = sub['T2M']
    tmax = sub['T2M_MAX']
    tmin = sub['T2M_MIN']
    precip = sub['PRECTOTCORR']
    rh = sub['RH2M']
    solar = sub['ALLSKY_SFC_SW_DWN']
    ws = sub['WS2M']
    
    return {
        'precip_cum': float(precip.sum()),
        'precip_mean': float(precip.mean()),
        'precip_max': float(precip.max()),
        'precip_days': int((precip >= 1.0).sum()),
        't2m_mean': float(t2m.mean()),
        't2m_min': float(tmin.mean()),
        't2m_max': float(tmax.mean()),
        't2m_range': float((tmax - tmin).mean()),
        'heat_stress_days_28': int((tmax >= 28.0).sum()),
        'heat_stress_days_30': int((tmax >= 30.0).sum()),
        'cool_night_days_16': int((tmin <= 16.0).sum()),
        'cool_night_days_14': int((tmin <= 14.0).sum()),
        'late_blight_risk_days': int(((rh >= 80.0) & (t2m >= 10.0) & (t2m <= 22.0)).sum()),
        'gdd_base5': float(np.maximum(0, t2m - 5.0).sum()),
        'solar_mean': float(solar.mean()),
        'rh2m_mean': float(rh.mean()),
        'ws2m_mean': float(ws.mean()),
    }

def build_wheat_model(weather_df):
    print("Building Wheat Model...")
    w_df = clean_raw_data(DATA_RAW_DIR / "Wheat.csv", "Wheat")
    
    rows = []
    for dist, grp in w_df.groupby('District'):
        grp = grp.sort_values('Crop_Year').copy()
        grp['yield_lag1'] = grp['yield_kg_per_ha'].shift(1)
        grp['yield_lag2'] = grp['yield_kg_per_ha'].shift(2)
        grp['yield_roll_mean3'] = grp['yield_kg_per_ha'].shift(1).rolling(3, min_periods=1).mean()
        grp['area_lag1'] = grp['area_ha'].shift(1)
        
        for idx, row in grp.iterrows():
            yr = int(row['Crop_Year'])
            wf = extract_weather(weather_df, dist, yr, 11, 15, 1, 31)
            rec = row.to_dict()
            if wf is not None:
                rec.update(wf)
                rec['has_weather'] = True
            else:
                rec['has_weather'] = False
            rows.append(rec)
            
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROC_DIR / "wheat_yield_features_wb.csv", index=False)
    
    valid = df[df['yield_lag1'].notnull() & df['has_weather']].copy()
    valid['yield_lag2'] = valid['yield_lag2'].fillna(valid['yield_lag1'])
    valid['yield_roll_mean3'] = valid['yield_roll_mean3'].fillna(valid['yield_lag1'])
    valid['area_lag1'] = valid['area_lag1'].fillna(valid['area_ha'])
    valid['log_area_lag1'] = np.log1p(valid['area_lag1'])
    
    # District target encoding
    dist_means = valid.groupby('District')['yield_kg_per_ha'].mean().to_dict()
    global_mean = float(valid['yield_kg_per_ha'].mean())
    valid['dist_mean'] = valid['District'].map(dist_means).fillna(global_mean)
    
    feature_cols = [
        'yield_lag1', 'yield_lag2', 'yield_roll_mean3', 'log_area_lag1', 'dist_mean',
        'precip_cum', 'precip_mean', 'precip_days', 't2m_mean', 't2m_min', 't2m_max',
        't2m_range', 'solar_mean', 'rh2m_mean', 'ws2m_mean', 'gdd_base5',
        'heat_stress_days_28', 'heat_stress_days_30'
    ]
    
    X = valid[feature_cols].values
    y = valid['yield_kg_per_ha'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = Ridge(alpha=10.0, random_state=42)
    model.fit(X_scaled, y)
    
    # Holdout error distribution (residuals on 2015-2019)
    test_mask = valid['Crop_Year'] >= 2015
    y_test = y[test_mask]
    y_pred_test = model.predict(X_scaled[test_mask])
    res_test = y_test - y_pred_test
    
    residuals_quantiles = {
        'p10': float(np.percentile(res_test, 10)),
        'p25': float(np.percentile(res_test, 25)),
        'p50': float(np.percentile(res_test, 50)),
        'p75': float(np.percentile(res_test, 75)),
        'p90': float(np.percentile(res_test, 90)),
        'std': float(np.std(res_test)),
        'mae': float(np.mean(np.abs(res_test))),
        'rmse': float(np.sqrt(np.mean(res_test**2))),
        'r2': float(1.0 - (np.sum(res_test**2) / np.sum((y_test - np.mean(y_test))**2))),
        'n_holdout': int(len(res_test))
    }
    
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'dist_means': dist_means,
        'global_mean': global_mean,
        'cutoff': {'month': 1, 'day': 31},
        'season_start': {'month': 11, 'day': 15},
        'residuals_quantiles': residuals_quantiles
    }
    
    artifact_path = MODELS_DIR / "m1_wheat_ridge.pkl"
    joblib.dump(artifact, artifact_path)
    
    sha256 = hashlib.sha256(open(artifact_path, 'rb').read()).hexdigest()
    
    metadata = {
        "model_name": "m1_wheat_ridge",
        "crop": "wheat",
        "release_status": "EXPERIMENTAL",
        "artifact_filename": "m1_wheat_ridge.pkl",
        "artifact_sha256": sha256,
        "architecture": "Ridge Regression (StandardScaler + Target Encoded District)",
        "hyperparameters": {"alpha": 10.0, "random_state": 42},
        "target": "yield_kg_per_ha",
        "season": "Rabi",
        "season_start": "Nov 15",
        "cutoff_date": "Jan 31",
        "training_rows": len(valid),
        "feature_cols": feature_cols,
        "evaluation_metrics": residuals_quantiles,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(MODELS_DIR / "m1_wheat_ridge.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Wheat model saved: {artifact_path} (SHA-256: {sha256})")
    return metadata

def build_potato_model(weather_df):
    print("Building Potato Model...")
    p_df = clean_raw_data(DATA_RAW_DIR / "Potato.csv", "Potato")
    p_main = p_df[p_df['Season'].isin(['Rabi', 'Whole Year', 'Winter'])].copy()
    
    rows = []
    for dist, grp in p_main.groupby('District'):
        grp = grp.sort_values('Crop_Year').copy()
        grp['yield_lag1'] = grp['yield_kg_per_ha'].shift(1)
        grp['yield_lag2'] = grp['yield_kg_per_ha'].shift(2)
        grp['yield_roll_mean3'] = grp['yield_kg_per_ha'].shift(1).rolling(3, min_periods=1).mean()
        grp['area_lag1'] = grp['area_ha'].shift(1)
        
        for idx, row in grp.iterrows():
            yr = int(row['Crop_Year'])
            wf = extract_weather(weather_df, dist, yr, 11, 1, 1, 31)
            rec = row.to_dict()
            if wf is not None:
                rec.update(wf)
                rec['has_weather'] = True
            else:
                rec['has_weather'] = False
            rows.append(rec)
            
    df = pd.DataFrame(rows)
    df.to_csv(DATA_PROC_DIR / "potato_yield_features_wb.csv", index=False)
    
    valid = df[df['yield_lag1'].notnull() & df['has_weather']].copy()
    valid['yield_lag2'] = valid['yield_lag2'].fillna(valid['yield_lag1'])
    valid['yield_roll_mean3'] = valid['yield_roll_mean3'].fillna(valid['yield_lag1'])
    valid['area_lag1'] = valid['area_lag1'].fillna(valid['area_ha'])
    valid['log_area_lag1'] = np.log1p(valid['area_lag1'])
    
    dist_means = valid.groupby('District')['yield_kg_per_ha'].mean().to_dict()
    global_mean = float(valid['yield_kg_per_ha'].mean())
    valid['dist_mean'] = valid['District'].map(dist_means).fillna(global_mean)
    
    feature_cols = [
        'yield_lag1', 'yield_lag2', 'yield_roll_mean3', 'log_area_lag1', 'dist_mean',
        'precip_cum', 'precip_mean', 'precip_days', 't2m_mean', 't2m_min', 't2m_max',
        't2m_range', 'solar_mean', 'rh2m_mean', 'ws2m_mean', 'gdd_base5',
        'cool_night_days_16', 'cool_night_days_14', 'late_blight_risk_days'
    ]
    
    X = valid[feature_cols].values
    y = valid['yield_kg_per_ha'].values
    
    model = cb.CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.05,
        verbose=0,
        random_seed=42
    )
    model.fit(X, y)
    
    # Residuals on 2015-2019
    test_mask = valid['Crop_Year'] >= 2015
    y_test = y[test_mask]
    y_pred_test = model.predict(X[test_mask])
    res_test = y_test - y_pred_test
    
    residuals_quantiles = {
        'p10': float(np.percentile(res_test, 10)),
        'p25': float(np.percentile(res_test, 25)),
        'p50': float(np.percentile(res_test, 50)),
        'p75': float(np.percentile(res_test, 75)),
        'p90': float(np.percentile(res_test, 90)),
        'std': float(np.std(res_test)),
        'mae': float(np.mean(np.abs(res_test))),
        'rmse': float(np.sqrt(np.mean(res_test**2))),
        'r2': float(1.0 - (np.sum(res_test**2) / np.sum((y_test - np.mean(y_test))**2))),
        'n_holdout': int(len(res_test))
    }
    
    artifact = {
        'model': model,
        'feature_cols': feature_cols,
        'dist_means': dist_means,
        'global_mean': global_mean,
        'cutoff': {'month': 1, 'day': 31},
        'season_start': {'month': 11, 'day': 1},
        'residuals_quantiles': residuals_quantiles
    }
    
    artifact_path = MODELS_DIR / "m1_potato_catboost.pkl"
    joblib.dump(artifact, artifact_path)
    
    sha256 = hashlib.sha256(open(artifact_path, 'rb').read()).hexdigest()
    
    metadata = {
        "model_name": "m1_potato_catboost",
        "crop": "potato",
        "release_status": "EXPERIMENTAL",
        "artifact_filename": "m1_potato_catboost.pkl",
        "artifact_sha256": sha256,
        "architecture": "CatBoost Regressor (depth=4, lr=0.05, 100 iters)",
        "hyperparameters": {"iterations": 100, "depth": 4, "learning_rate": 0.05, "random_seed": 42},
        "target": "yield_kg_per_ha",
        "season": "Rabi",
        "season_start": "Nov 1",
        "cutoff_date": "Jan 31",
        "training_rows": len(valid),
        "feature_cols": feature_cols,
        "evaluation_metrics": residuals_quantiles,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    
    with open(MODELS_DIR / "m1_potato_catboost.metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Potato model saved: {artifact_path} (SHA-256: {sha256})")
    return metadata

if __name__ == '__main__':
    print("Loading daily weather archive...")
    w_daily = load_daily_weather()
    print("Weather archive loaded:", len(w_daily), "daily records.")
    
    w_meta = build_wheat_model(w_daily)
    p_meta = build_potato_model(w_daily)
    
    print("\n--- MODEL CREATION SUMMARY ---")
    print(f"Wheat Model: {w_meta['model_name']} | Status: {w_meta['release_status']} | SHA-256: {w_meta['artifact_sha256']}")
    print(f"Potato Model: {p_meta['model_name']} | Status: {p_meta['release_status']} | SHA-256: {p_meta['artifact_sha256']}")
