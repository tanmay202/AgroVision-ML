"""
Hyperparameter Tuning and Model Selection on Validation Data
============================================================
STRICTLY uses only data up to 2016 (Train 2000-2014, Val 2015-2016, and Rolling CV 2012-2016).
Untouched test data (2017-2019) is NEVER seen or loaded in this script.
"""
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from xgboost import XGBRegressor

BASE = Path(r"C:\Arunava Personal\Arunava_New\AgroVision-ML\member1-yield-model")
TARGET = "Rice_Yield_kg_ha"

def calc_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp) * 100.0
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def run_tuning():
    df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    df_tv = df[df['Year'] <= 2016].copy()

    train = df_tv[df_tv['Year'] <= 2014].copy()
    val = df_tv[(df_tv['Year'] >= 2015) & (df_tv['Year'] <= 2016)].copy()

    # Features
    # Note: Mandi features are excluded because of severe missingness and distortion
    # Yield_hist_mean is excluded because of strong downward historical trend bias
    num_clean = [
        'Yield_lag1', 'Yield_lag2',
        'Area_lag1', 'Area_lag2',
        'Precip_lag1', 'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
        'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
        'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
        'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
    ]
    cat_cols = ['District', 'Season']

    # Impute missing lag1 for newly split districts in 2016 val using district yield mean or train mean
    train_lag1 = train['Yield_lag1'].fillna(train['Yield_hist_mean']).fillna(train[TARGET].mean())
    val_lag1 = val['Yield_lag1'].fillna(val['Yield_hist_mean']).fillna(train[TARGET].mean())

    y_train = train[TARGET].values
    y_val = val[TARGET].values

    y_train_delta = (train[TARGET] - train_lag1).values
    y_val_delta = (val[TARGET] - val_lag1).values

    # -----------------------------------------------------------------
    # 1. TUNING DELTA RIDGE
    # -----------------------------------------------------------------
    print("=" * 60)
    print("1. TUNING DELTA RIDGE ON VALIDATION")
    print("=" * 60)

    alphas = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150.0, 200.0, 300.0, 500.0]
    ridge_results = []

    for alpha in alphas:
        preprocessor = ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scl', StandardScaler())
            ]), num_clean),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols)
        ])
        pipe = Pipeline([('pre', preprocessor), ('model', Ridge(alpha=alpha, random_state=42))])
        pipe.fit(train[num_clean + cat_cols], y_train_delta)
        delta_preds = pipe.predict(val[num_clean + cat_cols])
        yield_preds = val_lag1.values + delta_preds
        m = calc_metrics(y_val, yield_preds)
        ridge_results.append({'alpha': alpha, **m})

    df_ridge = pd.DataFrame(ridge_results).sort_values('MAE')
    print(df_ridge.head(8).to_string(index=False))
    best_ridge_alpha = df_ridge.iloc[0]['alpha']
    print(f"Best Ridge Alpha: {best_ridge_alpha} (Val MAE: {df_ridge.iloc[0]['MAE']:.2f})")

    # -----------------------------------------------------------------
    # 2. TUNING DELTA XGBOOST
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. TUNING DELTA XGBOOST ON VALIDATION")
    print("=" * 60)

    param_grid = {
        'max_depth': [2, 3],
        'learning_rate': [0.02, 0.03, 0.05],
        'n_estimators': [50, 75, 100, 130],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'reg_lambda': [1.0, 5.0, 10.0],
        'reg_alpha': [0.0, 1.0, 5.0],
        'min_child_weight': [3, 5, 8]
    }

    # Grid search (sample or thorough search)
    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Total parameter combinations: {len(permutations_dicts)}")

    # Preprocessor for XGBoost
    pre_xgb = ColumnTransformer([
        ('num', 'passthrough', num_clean),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])
    X_tr = pre_xgb.fit_transform(train[num_clean + cat_cols])
    X_va = pre_xgb.transform(val[num_clean + cat_cols])

    xgb_results = []
    # Test all combinations (matrix operations in XGBoost are fast on 850 rows)
    for p in permutations_dicts:
        model = XGBRegressor(
            n_estimators=p['n_estimators'],
            learning_rate=p['learning_rate'],
            max_depth=p['max_depth'],
            subsample=p['subsample'],
            colsample_bytree=p['colsample_bytree'],
            reg_lambda=p['reg_lambda'],
            reg_alpha=p['reg_alpha'],
            min_child_weight=p['min_child_weight'],
            random_state=42,
            n_jobs=4
        )
        model.fit(X_tr, y_train_delta)
        delta_p = model.predict(X_va)
        yield_p = val_lag1.values + delta_p
        m = calc_metrics(y_val, yield_p)
        xgb_results.append({**p, **m})

    df_xgb = pd.DataFrame(xgb_results).sort_values('MAE')
    print("Top 10 XGBoost hyperparameter configurations:")
    print(df_xgb.head(10)[['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree', 'reg_lambda', 'reg_alpha', 'min_child_weight', 'MAE', 'R2', 'RMSE']].to_string(index=False))

    best_xgb_params = df_xgb.iloc[0].to_dict()
    print(f"\nBest XGBoost Val MAE: {best_xgb_params['MAE']:.2f} (R2={best_xgb_params['R2']:.4f})")

    # -----------------------------------------------------------------
    # 3. DIRECT XGBOOST WITH REGULARIZATION (for direct yield benchmark)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. TUNING REGULARIZED DIRECT XGBOOST (No Delta)")
    print("=" * 60)
    direct_results = []
    for depth in [2, 3, 4]:
        for lr in [0.02, 0.04, 0.06]:
            for n_est in [60, 100, 150]:
                for reg_l in [1.0, 5.0, 10.0]:
                    for mcw in [3, 5, 8]:
                        m_direct = XGBRegressor(
                            n_estimators=n_est, learning_rate=lr, max_depth=depth,
                            subsample=0.8, colsample_bytree=0.7, reg_lambda=reg_l,
                            min_child_weight=mcw, random_state=42, n_jobs=4
                        )
                        m_direct.fit(X_tr, y_train)
                        p_dir = m_direct.predict(X_va)
                        m = calc_metrics(y_val, p_dir)
                        direct_results.append({'max_depth': depth, 'lr': lr, 'n_est': n_est, 'reg_l': reg_l, 'mcw': mcw, **m})

    df_direct = pd.DataFrame(direct_results).sort_values('MAE')
    print("Top 5 Direct XGBoost configurations:")
    print(df_direct.head(5).to_string(index=False))

    # -----------------------------------------------------------------
    # 4. ROLLING-ORIGIN VALIDATION CONFIRMATION
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. ROLLING-ORIGIN CV (2012-2016) CONFIRMATION")
    print("=" * 60)
    splits = [
        (2012, [2013]),
        (2013, [2014]),
        (2014, [2015]),
        (2015, [2016]),
    ]
    cv_table = {'Persistence': [], 'Original XGBoost': [], 'Improved Delta Ridge': [], 'Improved Delta XGBoost': [], 'Improved Ensemble (50/50)': []}

    for cutoff_yr, test_yrs in splits:
        tr_df = df_tv[df_tv['Year'] <= cutoff_yr].copy()
        va_df = df_tv[df_tv['Year'].isin(test_yrs)].copy()

        y_tr = tr_df[TARGET].values
        y_va = va_df[TARGET].values

        tr_lag = tr_df['Yield_lag1'].fillna(tr_df['Yield_hist_mean']).fillna(y_tr.mean()).values
        va_lag = va_df['Yield_lag1'].fillna(va_df['Yield_hist_mean']).fillna(y_tr.mean()).values

        y_tr_delta = y_tr - tr_lag
        y_va_delta = y_va - va_lag

        # 1. Persistence
        cv_table['Persistence'].append(calc_metrics(y_va, va_lag)['MAE'])

        # 2. Original XGBoost pipeline
        num_orig = [c for c in tr_df.columns if c not in ['State_Name', 'District', 'Year', 'Season', 'Crop', TARGET]]
        ct_orig = ColumnTransformer([
            ('num', 'passthrough', num_orig),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        p_orig = Pipeline([
            ('pre', ct_orig),
            ('model', XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42))
        ])
        p_orig.fit(tr_df[num_orig + cat_cols], y_tr)
        pred_orig = p_orig.predict(va_df[num_orig + cat_cols])
        cv_table['Original XGBoost'].append(calc_metrics(y_va, pred_orig)['MAE'])

        # 3. Improved Delta Ridge (alpha=best)
        ct_ridge = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), num_clean),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                             ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols)
        ])
        p_ridge = Pipeline([('pre', ct_ridge), ('model', Ridge(alpha=best_ridge_alpha, random_state=42))])
        p_ridge.fit(tr_df[num_clean + cat_cols], y_tr_delta)
        pred_ridge = va_lag + p_ridge.predict(va_df[num_clean + cat_cols])
        cv_table['Improved Delta Ridge'].append(calc_metrics(y_va, pred_ridge)['MAE'])

        # 4. Improved Delta XGBoost (best params)
        ct_xgb = ColumnTransformer([
            ('num', 'passthrough', num_clean),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ])
        p_xgb = Pipeline([
            ('pre', ct_xgb),
            ('model', XGBRegressor(
                n_estimators=int(best_xgb_params['n_estimators']),
                learning_rate=float(best_xgb_params['learning_rate']),
                max_depth=int(best_xgb_params['max_depth']),
                subsample=float(best_xgb_params['subsample']),
                colsample_bytree=float(best_xgb_params['colsample_bytree']),
                reg_lambda=float(best_xgb_params['reg_lambda']),
                reg_alpha=float(best_xgb_params['reg_alpha']),
                min_child_weight=float(best_xgb_params['min_child_weight']),
                random_state=42
            ))
        ])
        p_xgb.fit(tr_df[num_clean + cat_cols], y_tr_delta)
        pred_xgb = va_lag + p_xgb.predict(va_df[num_clean + cat_cols])
        cv_table['Improved Delta XGBoost'].append(calc_metrics(y_va, pred_xgb)['MAE'])

        # 5. Ensemble (50% Ridge + 50% XGBoost)
        pred_ens = 0.5 * pred_ridge + 0.5 * pred_xgb
        cv_table['Improved Ensemble (50/50)'].append(calc_metrics(y_va, pred_ens)['MAE'])

    df_cv_summary = pd.DataFrame(cv_table, index=[f"Val {y[0]}" for _, y in splits])
    df_cv_summary.loc['Mean MAE'] = df_cv_summary.mean()
    print(df_cv_summary.round(2).to_string())

if __name__ == '__main__':
    run_tuning()
