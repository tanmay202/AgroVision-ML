"""
AgroVision M1 Candidate Model Validation & Tuning
=================================================
STRICTLY uses data up to 2016 (Train 2000-2014, Val 2015-2016).
Does NOT touch or evaluate on the test set (2017-2019).
Uses chronological validation and rolling-origin validation.
Explores:
1. Feature selection (impact of removing Mandi, dropping Yield_hist_mean, weather subsets).
2. Delta/Residual formulation: predicting (Yield - Yield_lag1) vs direct Yield.
3. XGBoost regularization: max_depth, learning_rate, colsample, subsample, reg_alpha, reg_lambda, min_child_weight.
4. Regularized linear models (Ridge, ElasticNet, HuberRegressor).
5. Blended / Ensemble formulations.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent.parent
TARGET = "Rice_Yield_kg_ha"

def calc_metrics(y_true, y_pred, name="Model"):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp) * 100.0
    return {'Model': name, 'N': len(yt), 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def run_validation_experiments():
    # Load dataset
    df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    
    # Strictly training + validation: 2000 to 2016
    df_tv = df[df['Year'] <= 2016].copy()
    
    # Primary split: Train <= 2014, Val 2015-2016
    train = df_tv[df_tv['Year'] <= 2014].copy()
    val = df_tv[(df_tv['Year'] >= 2015) & (df_tv['Year'] <= 2016)].copy()

    # Note: newly split districts in 2016 have NaN in Yield_lag1
    # For fair baseline comparison, fill Yield_lag1 NaN with district historical mean or drop
    print(f"Train rows: {len(train)}, Val rows: {len(val)}")
    print(f"Val rows with NaN Yield_lag1: {val['Yield_lag1'].isna().sum()}")

    # Feature groups
    historical_all = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean',
        'Area_lag1', 'Area_lag2', 'Precip_lag1',
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1'
    ]
    historical_no_histmean = [
        'Yield_lag1', 'Yield_lag2',
        'Area_lag1', 'Area_lag2', 'Precip_lag1',
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1'
    ]
    mandi_cols = [c for c in df.columns if 'Mandi' in c and 'lag' in c]
    inseason_weather = [c for c in df.columns if c.startswith('InSeason_')]
    key_inseason = [
        'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_T2M_Range',
        'InSeason_T2M_Max', 'InSeason_RH2M_Mean', 'InSeason_Solar_Mean'
    ]
    cat_cols = ['District', 'Season']

    # Target arrays
    y_train = train[TARGET].values
    y_val = val[TARGET].values

    # Validation persistence baseline
    # For rows where Yield_lag1 is NaN in 2016, we can impute with district historical yield or train mean
    val_persist = val['Yield_lag1'].copy()
    val_persist_imputed = val_persist.fillna(val['Yield_hist_mean']).fillna(train[TARGET].mean()).values

    print("\n" + "=" * 60)
    print("VALIDATION BASELINES (2015-2016)")
    print("=" * 60)
    b1 = calc_metrics(y_val, val['Yield_lag1'].values, "Persistence (masked NaN, N=108)")
    b2 = calc_metrics(y_val, val_persist_imputed, "Persistence (imputed NaN, N=118)")
    b3 = calc_metrics(y_val, val['Yield_hist_mean'].values, "Historical Mean")
    print(b1)
    print(b2)
    print(b3)

    # -------------------------------------------------------------
    # EXPERIMENT A: Original XGBoost Reproduction on Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT A: Original XGBoost vs Variants on Validation")
    print("=" * 60)
    
    num_orig = historical_all + mandi_cols + inseason_weather
    ct_orig = ColumnTransformer([
        ('num', 'passthrough', num_orig),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    pipe_orig = Pipeline([
        ('pre', ct_orig),
        ('xgb', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42))
    ])
    pipe_orig.fit(train[num_orig + cat_cols], y_train)
    p_orig = pipe_orig.predict(val[num_orig + cat_cols])
    print(calc_metrics(y_val, p_orig, "Original XGBoost (all 58 feats, depth=5, lr=0.05)"))

    # -------------------------------------------------------------
    # EXPERIMENT B: Feature Ablation on Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Feature Ablation (Removing Mandi, HistMean, Noisy Weather)")
    print("=" * 60)

    feature_sets = {
        'Orig (All Features)': historical_all + mandi_cols + inseason_weather,
        'No Mandi': historical_all + inseason_weather,
        'No Mandi, No HistMean': historical_no_histmean + inseason_weather,
        'No Mandi, No HistMean, Key Weather': historical_no_histmean + key_inseason,
        'Lag1+Lag2+KeyWeather+Cat': ['Yield_lag1', 'Yield_lag2', 'Area_lag1'] + key_inseason,
        'Lag1+Lag2+PreplantWeather+InSeason': ['Yield_lag1', 'Yield_lag2', 'Precip_lag1', 'T2M_lag1', 'NDVI_lag1'] + key_inseason,
    }

    for fname, fcols in feature_sets.items():
        ct = ColumnTransformer([
            ('num', 'passthrough', fcols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        pipe = Pipeline([
            ('pre', ct),
            ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.03, max_depth=3,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42))
        ])
        pipe.fit(train[fcols + cat_cols], y_train)
        preds = pipe.predict(val[fcols + cat_cols])
        print(calc_metrics(y_val, preds, f"XGB (depth=3) - {fname}"))

    # -------------------------------------------------------------
    # EXPERIMENT C: Delta / Residual Formulation
    # Predicting Delta = Yield - Yield_lag1
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Delta Formulation (Predicting Delta = Yield - Yield_lag1)")
    print("=" * 60)

    # Impute missing lag1 in train with district median or shift
    train_lag1_filled = train['Yield_lag1'].fillna(train['Yield_hist_mean']).fillna(train[TARGET].mean())
    val_lag1_filled = val['Yield_lag1'].fillna(val['Yield_hist_mean']).fillna(train[TARGET].mean())

    y_train_delta = train[TARGET] - train_lag1_filled
    y_val_delta = val[TARGET] - val_lag1_filled

    for fname, fcols in [
        ('No Mandi', historical_all + inseason_weather),
        ('No Mandi, No HistMean, Key Weather', historical_no_histmean + key_inseason),
        ('Compact Features', ['Yield_lag1', 'Yield_lag2', 'Precip_lag1', 'T2M_lag1', 'NDVI_lag1'] + key_inseason)
    ]:
        ct = ColumnTransformer([
            ('num', 'passthrough', fcols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        for depth in [2, 3]:
            for lr in [0.02, 0.05]:
                for reg_l in [1.0, 5.0, 10.0]:
                    pipe = Pipeline([
                        ('pre', ct),
                        ('xgb', XGBRegressor(n_estimators=80, learning_rate=lr, max_depth=depth,
                                             reg_lambda=reg_l, subsample=0.8, colsample_bytree=0.8, random_state=42))
                    ])
                    pipe.fit(train[fcols + cat_cols], y_train_delta)
                    delta_preds = pipe.predict(val[fcols + cat_cols])
                    yield_preds = val_lag1_filled + delta_preds
                    m = calc_metrics(y_val, yield_preds, f"Delta XGB d={depth} lr={lr} l2={reg_l} ({fname})")
                    if m['MAE'] < 235:
                        print(m)

    # -------------------------------------------------------------
    # EXPERIMENT D: Ridge and Linear Models on Delta
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Regularized Linear Models on Delta and Direct Yield")
    print("=" * 60)

    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 200.0, 500.0]:
        ct = ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scl', StandardScaler())
            ]), historical_no_histmean + key_inseason),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ])
        
        # Delta Ridge
        pipe_r_delta = Pipeline([('pre', ct), ('ridge', Ridge(alpha=alpha))])
        pipe_r_delta.fit(train[historical_no_histmean + key_inseason + cat_cols], y_train_delta)
        delta_p = pipe_r_delta.predict(val[historical_no_histmean + key_inseason + cat_cols])
        yield_p = val_lag1_filled + delta_p
        m_rd = calc_metrics(y_val, yield_p, f"Delta Ridge alpha={alpha}")
        if m_rd['MAE'] < 230:
            print(m_rd)

        # Direct Ridge
        pipe_r_direct = Pipeline([('pre', ct), ('ridge', Ridge(alpha=alpha))])
        pipe_r_direct.fit(train[historical_no_histmean + key_inseason + cat_cols], y_train)
        yield_direct = pipe_r_direct.predict(val[historical_no_histmean + key_inseason + cat_cols])
        m_direct = calc_metrics(y_val, yield_direct, f"Direct Ridge alpha={alpha}")
        if m_direct['MAE'] < 240:
            print(m_direct)

    # -------------------------------------------------------------
    # EXPERIMENT E: Rolling-Origin Chronological Validation
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EXPERIMENT E: Rolling-Origin Time-Series CV (2012-2016)")
    print("=" * 60)

    splits = [
        (2012, [2013]),
        (2013, [2014]),
        (2014, [2015]),
        (2015, [2016]),
    ]

    models_to_test = {
        'Persistence': None,
        'Direct XGB Orig': ('direct', Pipeline([
            ('pre', ColumnTransformer([
                ('num', 'passthrough', num_orig),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ])),
            ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42))
        ])),
        'Delta XGB (Regularized)': ('delta', Pipeline([
            ('pre', ColumnTransformer([
                ('num', 'passthrough', historical_no_histmean + key_inseason),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
            ])),
            ('model', XGBRegressor(n_estimators=60, learning_rate=0.03, max_depth=2,
                                  reg_lambda=5.0, subsample=0.8, colsample_bytree=0.8, random_state=42))
        ])),
        'Delta Ridge (alpha=100)': ('delta', Pipeline([
            ('pre', ColumnTransformer([
                ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]),
                 historical_no_histmean + key_inseason),
                ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                                 ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
            ])),
            ('model', Ridge(alpha=100.0))
        ])),
        'Blended (0.7 Persistence + 0.3 Delta Ridge)': 'blend'
    }

    cv_results = {m: [] for m in models_to_test}

    for cutoff_yr, test_yrs in splits:
        tr_df = df_tv[df_tv['Year'] <= cutoff_yr].copy()
        va_df = df_tv[df_tv['Year'].isin(test_yrs)].copy()

        y_tr = tr_df[TARGET].values
        y_va = va_df[TARGET].values

        tr_lag1 = tr_df['Yield_lag1'].fillna(tr_df['Yield_hist_mean']).fillna(y_tr.mean()).values
        va_lag1 = va_df['Yield_lag1'].fillna(va_df['Yield_hist_mean']).fillna(y_tr.mean()).values

        y_tr_d = y_tr - tr_lag1
        y_va_d = y_va - va_lag1

        # Persistence
        m_p = calc_metrics(y_va, va_lag1)['MAE']
        cv_results['Persistence'].append(m_p)

        # Direct XGB
        _, pipe_dx = models_to_test['Direct XGB Orig']
        pipe_dx.fit(tr_df[num_orig + cat_cols], y_tr)
        p_dx = pipe_dx.predict(va_df[num_orig + cat_cols])
        cv_results['Direct XGB Orig'].append(calc_metrics(y_va, p_dx)['MAE'])

        # Delta XGB
        _, pipe_x = models_to_test['Delta XGB (Regularized)']
        pipe_x.fit(tr_df[historical_no_histmean + key_inseason + cat_cols], y_tr_d)
        p_x = va_lag1 + pipe_x.predict(va_df[historical_no_histmean + key_inseason + cat_cols])
        cv_results['Delta XGB (Regularized)'].append(calc_metrics(y_va, p_x)['MAE'])

        # Delta Ridge
        _, pipe_r = models_to_test['Delta Ridge (alpha=100)']
        pipe_r.fit(tr_df[historical_no_histmean + key_inseason + cat_cols], y_tr_d)
        p_r = va_lag1 + pipe_r.predict(va_df[historical_no_histmean + key_inseason + cat_cols])
        cv_results['Delta Ridge (alpha=100)'].append(calc_metrics(y_va, p_r)['MAE'])

        # Blend
        p_blend = 0.7 * va_lag1 + 0.3 * p_r
        cv_results['Blended (0.7 Persistence + 0.3 Delta Ridge)'].append(calc_metrics(y_va, p_blend)['MAE'])

    print("\n--- ROLLING-ORIGIN VALIDATION MAE (kg/ha) ---")
    df_cv = pd.DataFrame(cv_results, index=[f"Val {y[0]}" for _, y in splits])
    df_cv.loc['Mean MAE'] = df_cv.mean()
    print(df_cv.round(2).to_string())

if __name__ == '__main__':
    run_validation_experiments()
