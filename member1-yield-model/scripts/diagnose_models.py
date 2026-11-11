"""
Comprehensive Diagnostic Analysis for AgroVision M1
Inspects existing models:
- models/m1_inseason_xgboost.pkl
- models/m1_best_ridge.pkl
- Persistence Baseline
Analyzes errors by district, season, year, residual distributions, and feature importance.
Strictly read-only on existing models and datasets.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

BASE = Path(__file__).resolve().parent.parent
TARGET = "Rice_Yield_kg_ha"

def calc_metrics(y_true, y_pred):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mape = mean_absolute_percentage_error(yt, yp) * 100.0
    return {'N': len(yt), 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def run_diagnostics():
    print("=" * 70)
    print("AGROVISION M1 DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    # 1. Load Data
    inseason_df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    train_df = inseason_df[inseason_df['Year'] <= 2014].copy()
    val_df = inseason_df[(inseason_df['Year'] >= 2015) & (inseason_df['Year'] <= 2016)].copy()
    train_val_df = inseason_df[inseason_df['Year'] <= 2016].copy()
    test_df = inseason_df[inseason_df['Year'] >= 2017].copy()

    # Pre-planting baseline data (for Ridge baseline)
    test_base_df = pd.read_csv(BASE / "data/processed/test.csv")
    train_val_base_df = pd.concat([
        pd.read_csv(BASE / "data/processed/train.csv"),
        pd.read_csv(BASE / "data/processed/validation.csv")
    ], ignore_index=True)

    # Sort both strictly by ['District', 'Year', 'Season'] to guarantee identical row order
    test_base_df = test_base_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)
    test_df = test_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)
    train_val_base_df = train_val_base_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)
    train_val_df = train_val_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

    print(f"Data shapes: Train={train_df.shape}, Val={val_df.shape}, Train+Val={train_val_df.shape}, Test={test_df.shape}")

    # 2. Load Models
    ridge_path = BASE / "models/m1_best_ridge.pkl"
    xgb_path = BASE / "models/m1_inseason_xgboost.pkl"

    ridge_model = joblib.load(ridge_path)
    xgb_model = joblib.load(xgb_path)

    # Fix SimpleImputer version issue if present
    for name, transformer, cols in ridge_model.named_steps['preprocessor'].transformers_:
        if hasattr(transformer, 'named_steps'):
            for step in transformer.named_steps.values():
                if hasattr(step, '_fit_dtype') and not hasattr(step, '_fill_dtype'):
                    step._fill_dtype = step._fit_dtype

    # Features for Ridge
    num_features_ridge = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean',
        'Area_lag1', 'Area_lag2', 'Precip_lag1',
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1'
    ]
    cat_features = ['District', 'Season']

    # Features for XGBoost
    num_historical = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean',
        'Area_lag1', 'Area_lag2', 'Precip_lag1',
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
    ]
    num_mandi = [c for c in inseason_df.columns if 'Mandi' in c and 'lag' in c]
    num_inseason = [c for c in inseason_df.columns if c.startswith('InSeason_')]
    num_xgb = num_historical + num_mandi + num_inseason

    # 3. Predictions on Test
    X_test_ridge = test_base_df[num_features_ridge + cat_features]
    preds_ridge = ridge_model.predict(X_test_ridge)

    X_test_xgb = test_df[num_xgb + cat_features]
    preds_xgb = xgb_model.predict(X_test_xgb)

    y_test = test_df[TARGET].values
    preds_persist = test_df['Yield_lag1'].values

    # Train predictions to assess overfitting
    X_tv_xgb = train_val_df[num_xgb + cat_features]
    y_tv = train_val_df[TARGET].values
    preds_xgb_train = xgb_model.predict(X_tv_xgb)

    X_tv_ridge = train_val_base_df[num_features_ridge + cat_features]
    y_tv_ridge = train_val_base_df[TARGET].values
    preds_ridge_train = ridge_model.predict(X_tv_ridge)

    # Metrics
    m_persist = calc_metrics(y_test, preds_persist)
    m_ridge_test = calc_metrics(y_test, preds_ridge)
    m_xgb_test = calc_metrics(y_test, preds_xgb)

    m_ridge_train = calc_metrics(y_tv_ridge, preds_ridge_train)
    m_xgb_train = calc_metrics(y_tv, preds_xgb_train)

    print("\n--- OVERALL PERFORMANCE ON TEST (2017-2019) ---")
    print(f"Persistence: R2={m_persist['R2']:.4f}, MAE={m_persist['MAE']:.2f} kg/ha ({m_persist['MAE']/1000:.4f} t/ha), RMSE={m_persist['RMSE']:.2f}, MAPE={m_persist['MAPE']:.2f}%")
    print(f"Ridge (Exp1): R2={m_ridge_test['R2']:.4f}, MAE={m_ridge_test['MAE']:.2f} kg/ha ({m_ridge_test['MAE']/1000:.4f} t/ha), RMSE={m_ridge_test['RMSE']:.2f}, MAPE={m_ridge_test['MAPE']:.2f}%")
    print(f"XGBoost (Exp2): R2={m_xgb_test['R2']:.4f}, MAE={m_xgb_test['MAE']:.2f} kg/ha ({m_xgb_test['MAE']/1000:.4f} t/ha), RMSE={m_xgb_test['RMSE']:.2f}, MAPE={m_xgb_test['MAPE']:.2f}%")

    print("\n--- OVERFITTING ASSESSMENT (TRAIN+VAL vs TEST) ---")
    print(f"Ridge Train R2: {m_ridge_train['R2']:.4f}, MAE: {m_ridge_train['MAE']:.2f} | Test R2: {m_ridge_test['R2']:.4f}, MAE: {m_ridge_test['MAE']:.2f} | Gap R2: {m_ridge_train['R2'] - m_ridge_test['R2']:.4f}")
    print(f"XGBoost Train R2: {m_xgb_train['R2']:.4f}, MAE: {m_xgb_train['MAE']:.2f} | Test R2: {m_xgb_test['R2']:.4f}, MAE: {m_xgb_test['MAE']:.2f} | Gap R2: {m_xgb_train['R2'] - m_xgb_test['R2']:.4f}")

    # 4. Residual Analysis
    res_persist = y_test - preds_persist
    res_ridge = y_test - preds_ridge
    res_xgb = y_test - preds_xgb

    print("\n--- RESIDUAL STATISTICS (TEST) ---")
    for name, r in [('Persistence', res_persist), ('Ridge', res_ridge), ('XGBoost', res_xgb)]:
        print(f"{name:12s} Mean={np.mean(r):7.2f}, Std={np.std(r):7.2f}, Skew={pd.Series(r).skew():7.3f}, Kurt={pd.Series(r).kurt():7.3f}, Min={np.min(r):7.2f}, Max={np.max(r):7.2f}")

    # Build analysis dataframe
    df_eval = test_df[['District', 'Year', 'Season', TARGET, 'Yield_lag1']].copy()
    df_eval['Pred_Persist'] = preds_persist
    df_eval['Pred_Ridge'] = preds_ridge
    df_eval['Pred_XGB'] = preds_xgb
    df_eval['Err_Persist'] = np.abs(df_eval[TARGET] - df_eval['Pred_Persist'])
    df_eval['Err_Ridge'] = np.abs(df_eval[TARGET] - df_eval['Pred_Ridge'])
    df_eval['Err_XGB'] = np.abs(df_eval[TARGET] - df_eval['Pred_XGB'])

    # 5. Year-wise Analysis
    print("\n--- YEAR-WISE TEST PERFORMANCE (MAE kg/ha) ---")
    year_grp = df_eval.groupby('Year')
    for yr, g in year_grp:
        print(f"Year {yr} (N={len(g)}): Persist={g['Err_Persist'].mean():.2f}, Ridge={g['Err_Ridge'].mean():.2f}, XGB={g['Err_XGB'].mean():.2f}")

    # 6. Season-wise Analysis
    print("\n--- SEASON-WISE TEST PERFORMANCE (MAE kg/ha) ---")
    season_grp = df_eval.groupby('Season')
    for sn, g in season_grp:
        print(f"Season {sn:8s} (N={len(g)}): Persist={g['Err_Persist'].mean():.2f}, Ridge={g['Err_Ridge'].mean():.2f}, XGB={g['Err_XGB'].mean():.2f}")

    # 7. District-wise Analysis
    print("\n--- DISTRICT-WISE TEST PERFORMANCE (MAE kg/ha) ---")
    dist_summary = []
    dist_grp = df_eval.groupby('District')
    for d, g in dist_grp:
        dist_summary.append({
            'District': d,
            'N': len(g),
            'MAE_Persist': g['Err_Persist'].mean(),
            'MAE_Ridge': g['Err_Ridge'].mean(),
            'MAE_XGB': g['Err_XGB'].mean(),
            'Mean_Yield': g[TARGET].mean(),
            'Std_Yield': g[TARGET].std(),
            'Yield_Change': (g[TARGET] - g['Yield_lag1']).abs().mean()
        })
    df_dist = pd.DataFrame(dist_summary).sort_values(by='MAE_XGB', ascending=False)
    print(df_dist.to_string(index=False))

    # 8. Feature Importance of XGBoost
    xgb_reg = xgb_model.named_steps['model']
    ohe = xgb_model.named_steps['preprocessor'].transformers_[1][1]
    cat_cols_ohe = list(ohe.get_feature_names_out(cat_features))
    all_feature_names = num_xgb + cat_cols_ohe

    importances = xgb_reg.feature_importances_
    df_fi = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\n--- TOP 20 FEATURES IN XGBOOST ---")
    print(df_fi.head(20).to_string(index=False))
    print("\n--- BOTTOM 15 FEATURES IN XGBOOST ---")
    print(df_fi.tail(15).to_string(index=False))

    # 9. Correlation of features with target
    corr = train_val_df[num_xgb + [TARGET]].corr()[TARGET].sort_values(ascending=False)
    print("\n--- TOP POSITIVE CORRELATIONS WITH TARGET (TRAIN+VAL) ---")
    print(corr.head(15))
    print("\n--- TOP NEGATIVE CORRELATIONS WITH TARGET (TRAIN+VAL) ---")
    print(corr.tail(15))

if __name__ == '__main__':
    run_diagnostics()
