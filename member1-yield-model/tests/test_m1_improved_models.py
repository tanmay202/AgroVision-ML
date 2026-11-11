"""
Unit and Regression Test Suite for AgroVision Model 1 Improved Models
====================================================================
Tests:
1. Integrity of untouched test dataset (192 rows, perfect alignment, zero leakage).
2. Preservation and non-regression of original models (XGBoost and Ridge).
3. Loading and inference sanity of newly trained candidate models.
4. Validation that Improved Delta Ridge and Improved Ensemble strictly beat Persistence Baseline.
5. Plausible agronomic bounds on all predicted yields.
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

BASE = Path(__file__).resolve().parent.parent
TARGET = 'Rice_Yield_kg_ha'

@pytest.fixture(scope="module")
def dataset_splits():
    inseason_path = BASE / "data/processed/rice_yield_inseason_features_wb.csv"
    test_base_path = BASE / "data/processed/test.csv"
    
    df = pd.read_csv(inseason_path)
    test_base = pd.read_csv(test_base_path)
    
    sort_cols = ['District', 'Year', 'Season']
    df = df.sort_values(by=sort_cols).reset_index(drop=True)
    test_base = test_base.sort_values(by=sort_cols).reset_index(drop=True)
    
    train_val = df[df['Year'] <= 2016].copy().reset_index(drop=True)
    test = df[df['Year'] >= 2017].copy().reset_index(drop=True)
    
    return {
        'df': df,
        'train_val': train_val,
        'test': test,
        'test_base': test_base
    }

@pytest.fixture(scope="module")
def models():
    model_dir = BASE / "models"
    return {
        'orig_ridge': joblib.load(model_dir / "m1_best_ridge.pkl"),
        'orig_xgb': joblib.load(model_dir / "m1_inseason_xgboost.pkl"),
        'improved_ridge': joblib.load(model_dir / "m1_improved_delta_ridge.pkl"),
        'improved_xgb': joblib.load(model_dir / "m1_improved_delta_xgboost.pkl"),
        'improved_xgb_cons': joblib.load(model_dir / "m1_improved_delta_xgboost_conservative.pkl"),
    }

def test_data_integrity(dataset_splits):
    test = dataset_splits['test']
    test_base = dataset_splits['test_base']
    train_val = dataset_splits['train_val']
    
    # 192 rows test set
    assert len(test) == 192, f"Expected 192 test rows, got {len(test)}"
    assert len(test_base) == 192, f"Expected 192 test_base rows, got {len(test_base)}"
    
    # Zero temporal overlap
    assert set(train_val['Year']).isdisjoint(set(test['Year'])), "Temporal overlap detected between train_val and test!"
    assert min(test['Year']) == 2017
    assert max(test['Year']) == 2019
    assert max(train_val['Year']) == 2016
    
    # Exact row alignment
    assert (test['District'].values == test_base['District'].values).all()
    assert (test['Year'].values == test_base['Year'].values).all()
    assert (test['Season'].values == test_base['Season'].values).all()
    assert np.allclose(test[TARGET].values, test_base[TARGET].values)
    
    # No NaNs in critical test columns
    assert test[TARGET].isna().sum() == 0
    assert test['Yield_lag1'].isna().sum() == 0

def test_original_models_preservation(dataset_splits, models):
    test = dataset_splits['test']
    test_base = dataset_splits['test_base']
    y_test = test[TARGET].values
    
    num_ridge = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean',
        'Area_lag1', 'Area_lag2', 'Precip_lag1',
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1'
    ]
    cat_cols = ['District', 'Season']
    
    # Ridge
    orig_ridge = models['orig_ridge']
    pred_r = orig_ridge.predict(test_base[num_ridge + cat_cols])
    mae_r = mean_absolute_error(y_test, pred_r)
    r2_r = r2_score(y_test, pred_r)
    assert abs(mae_r - 288.00) < 0.5, f"Original Ridge MAE drifted: {mae_r}"
    assert abs(r2_r - 0.5812) < 0.01, f"Original Ridge R2 drifted: {r2_r}"
    
    # XGBoost
    num_mandi = [c for c in test.columns if 'Mandi' in c and 'lag' in c]
    num_inseason = [c for c in test.columns if c.startswith('InSeason_')]
    num_xgb = num_ridge + num_mandi + num_inseason
    orig_xgb = models['orig_xgb']
    pred_x = orig_xgb.predict(test[num_xgb + cat_cols])
    mae_x = mean_absolute_error(y_test, pred_x)
    r2_x = r2_score(y_test, pred_x)
    assert abs(mae_x - 277.37) < 0.5, f"Original XGBoost MAE drifted: {mae_x}"
    assert abs(r2_x - 0.6043) < 0.01, f"Original XGBoost R2 drifted: {r2_x}"

def test_improved_models_performance(dataset_splits, models):
    test = dataset_splits['test']
    y_test = test[TARGET].values
    te_lag1 = test['Yield_lag1'].values
    
    num_clean = [
        'Yield_lag1', 'Yield_lag2',
        'Area_lag1', 'Area_lag2',
        'Precip_lag1', 'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
        'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
        'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
        'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
    ]
    cat_cols = ['District', 'Season']
    
    # Persistence Baseline
    persist_mae = mean_absolute_error(y_test, te_lag1)
    persist_rmse = np.sqrt(mean_squared_error(y_test, te_lag1))
    persist_r2 = r2_score(y_test, te_lag1)
    persist_mape = mean_absolute_percentage_error(y_test, te_lag1) * 100
    
    assert abs(persist_mae - 249.46) < 0.5
    assert abs(persist_r2 - 0.6115) < 0.01
    
    # Improved Delta Ridge
    pred_delta_ridge = te_lag1 + models['improved_ridge'].predict(test[num_clean + cat_cols])
    ridge_mae = mean_absolute_error(y_test, pred_delta_ridge)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, pred_delta_ridge))
    ridge_r2 = r2_score(y_test, pred_delta_ridge)
    ridge_mape = mean_absolute_percentage_error(y_test, pred_delta_ridge) * 100
    
    # Check that Delta Ridge BEATS Persistence on all metrics
    assert ridge_mae < persist_mae, f"Delta Ridge MAE ({ridge_mae:.2f}) did not beat Persistence ({persist_mae:.2f})"
    assert ridge_rmse < persist_rmse, f"Delta Ridge RMSE ({ridge_rmse:.2f}) did not beat Persistence ({persist_rmse:.2f})"
    assert ridge_r2 > persist_r2, f"Delta Ridge R2 ({ridge_r2:.4f}) did not beat Persistence ({persist_r2:.4f})"
    assert ridge_mape < persist_mape, f"Delta Ridge MAPE ({ridge_mape:.2f}%) did not beat Persistence ({persist_mape:.2f}%)"
    
    # Improved Delta XGBoost Conservative
    pred_delta_xgb_c = te_lag1 + models['improved_xgb_cons'].predict(test[num_clean + cat_cols])
    xgb_c_mae = mean_absolute_error(y_test, pred_delta_xgb_c)
    assert xgb_c_mae < persist_mae, f"Conservative Delta XGBoost MAE ({xgb_c_mae:.2f}) did not beat Persistence ({persist_mae:.2f})"
    
    # Improved Ensemble
    pred_ens = 0.5 * pred_delta_ridge + 0.5 * pred_delta_xgb_c
    ens_mae = mean_absolute_error(y_test, pred_ens)
    ens_r2 = r2_score(y_test, pred_ens)
    assert ens_mae < persist_mae
    assert ens_r2 > persist_r2

def test_predictions_agronomic_bounds(dataset_splits, models):
    test = dataset_splits['test']
    te_lag1 = test['Yield_lag1'].values
    num_clean = [
        'Yield_lag1', 'Yield_lag2',
        'Area_lag1', 'Area_lag2',
        'Precip_lag1', 'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
        'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
        'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
        'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
    ]
    cat_cols = ['District', 'Season']
    
    for mname in ['improved_ridge', 'improved_xgb', 'improved_xgb_cons']:
        model = models[mname]
        delta = model.predict(test[num_clean + cat_cols])
        pred_yield = te_lag1 + delta
        
        # Rice yield in West Bengal realistically ranges between 1000 and 5500 kg/ha
        assert np.all(pred_yield > 1000), f"Unrealistic low yield prediction in {mname}: min={pred_yield.min()}"
        assert np.all(pred_yield < 5500), f"Unrealistic high yield prediction in {mname}: max={pred_yield.max()}"
        assert not np.isnan(pred_yield).any(), f"NaN in predictions of {mname}"
