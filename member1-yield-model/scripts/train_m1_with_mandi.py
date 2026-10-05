import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

def calc_metrics(y_true, y_pred, name="Model"):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    mape = mean_absolute_percentage_error(y_t, y_p)
    return {'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data/processed/rice_yield_forecast_features_mandi_wb.csv"
    preds_path = base_dir / "data/processed/m1_mandi_predictions.csv"
    report_path = base_dir / "reports/m1_mandi_model_results.md"
    
    df = pd.read_csv(data_path)
    
    # Sort just in case
    df = df.sort_values(by=['Year', 'Season', 'District']).reset_index(drop=True)
    
    train = df[df['Year'] <= 2014].copy()
    val = df[(df['Year'] >= 2015) & (df['Year'] <= 2016)].copy()
    test = df[df['Year'] >= 2017].copy()
    
    target = 'Rice_Yield_kg_ha'
    
    # 1. DEFINE FEATURES
    num_features = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean', 
        'Area_lag1', 'Area_lag2', 'Precip_lag1', 
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
        # MANDI LAGS
        'Mandi_Arrivals_Total_lag1', 'Mandi_Arrivals_Mean_lag1',
        'Mandi_Price_Mean_lag1', 'Mandi_Price_Min_lag1', 'Mandi_Price_Max_lag1', 'Mandi_Price_Vol_lag1',
        'Mandi_Obs_Count_lag1', 'Mandi_Price_Range_lag1',
        'Mandi_Arrivals_Total_lag2', 'Mandi_Arrivals_Mean_lag2',
        'Mandi_Price_Mean_lag2', 'Mandi_Price_Min_lag2', 'Mandi_Price_Max_lag2', 'Mandi_Price_Vol_lag2',
        'Mandi_Obs_Count_lag2', 'Mandi_Price_Range_lag2'
    ]
    cat_features = ['District', 'Season']
    
    # Leakage Assertions
    forbidden = ['Rice_Yield_kg_ha', 'Production', 'yield', 'Rice_Yield_t_ha']
    for f in num_features + cat_features:
        assert f not in forbidden, f"Forbidden feature used: {f}"
        if 'Mandi' in f:
            assert 'lag' in f, f"Current-year mandi feature used! {f}"
            
    print("--- BASELINES (VALIDATION) ---")
    bl_val_lag1 = calc_metrics(val[target], val['Yield_lag1'], 'Persistence (Lag1) [Val]')
    bl_val_hist = calc_metrics(val[target], val['Yield_hist_mean'], 'Historical Mean [Val]')
    print(bl_val_lag1)
    
    X_train = train[num_features + cat_features]
    y_train = train[target]
    
    X_val = val[num_features + cat_features]
    y_val = val[target]
    
    # Preprocessor for Ridge / RF (Needs Imputation)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])
    
    xgb_preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ])
    
    models = {
        'Ridge': Pipeline(steps=[('preprocessor', preprocessor), ('model', Ridge(alpha=1.0))]),
        'RandomForest': Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
        'XGBoost': Pipeline(steps=[('preprocessor', xgb_preprocessor), ('model', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42))])
    }
    
    val_results = []
    print("\n--- ML MODELS (VALIDATION) ---")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        res = calc_metrics(y_val, preds, name)
        val_results.append(res)
        print(res)
        
    best_model_name = min(val_results, key=lambda x: x['MAE'])['Model']
    print(f"\nBest Model selected: {best_model_name}")
    
    best_pipeline = models[best_model_name]
    
    # Combine Train + Val
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = train_val[num_features + cat_features]
    y_train_val = train_val[target]
    
    X_test = test[num_features + cat_features]
    y_test = test[target]
    
    print("\nRetraining Best Model on TRAIN + VAL...")
    best_pipeline.fit(X_train_val, y_train_val)
    
    test_preds = best_pipeline.predict(X_test)
    test_res = calc_metrics(y_test, test_preds, f"{best_model_name} (Test)")
    
    print("\n--- FINAL TEST METRICS ---")
    bl_test_lag1 = calc_metrics(test[target], test['Yield_lag1'], 'Persistence (Lag1) [Test]')
    bl_test_hist = calc_metrics(test[target], test['Yield_hist_mean'], 'Historical Mean [Test]')
    print(bl_test_lag1)
    print(test_res)
    
    test_out = test.copy()
    test_out['Predicted'] = test_preds
    test_out['Residual'] = test_out[target] - test_out['Predicted']
    test_out.to_csv(preds_path, index=False)
    
    # Leakage Audit
    assert not train['Year'].isin(test['Year']).any(), "Temporal overlap between Train and Test!"
    assert not val['Year'].isin(test['Year']).any(), "Temporal overlap between Val and Test!"
    
    # Markdown Report
    report_content = f"""# Member 1 Rice Yield Model (WITH MANDI) Results

## 1. Feature List
- **Categorical:** {cat_features}
- **Numerical Lags (Yield/Env):** {len([f for f in num_features if 'Mandi' not in f])} features
- **Numerical Lags (Mandi):** {len([f for f in num_features if 'Mandi' in f])} features
*(Strictly pre-planting historical features. Zero current-season mandi data).*

## 2. Validation Metrics (Hyperparameter / Model Selection)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | {bl_val_lag1['MAE']:.1f} | {bl_val_lag1['RMSE']:.1f} | {bl_val_lag1['R2']:.3f} | {bl_val_lag1['MAPE']:.3f} |
| Historical Mean | {bl_val_hist['MAE']:.1f} | {bl_val_hist['RMSE']:.1f} | {bl_val_hist['R2']:.3f} | {bl_val_hist['MAPE']:.3f} |
| Ridge | {val_results[0]['MAE']:.1f} | {val_results[0]['RMSE']:.1f} | {val_results[0]['R2']:.3f} | {val_results[0]['MAPE']:.3f} |
| Random Forest | {val_results[1]['MAE']:.1f} | {val_results[1]['RMSE']:.1f} | {val_results[1]['R2']:.3f} | {val_results[1]['MAPE']:.3f} |
| XGBoost | {val_results[2]['MAE']:.1f} | {val_results[2]['RMSE']:.1f} | {val_results[2]['R2']:.3f} | {val_results[2]['MAPE']:.3f} |

**Selected Model:** {best_model_name}

## 3. Final Test Metrics (Out-of-Sample: 2017-2019)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | {bl_test_lag1['MAE']:.1f} | {bl_test_lag1['RMSE']:.1f} | {bl_test_lag1['R2']:.3f} | {bl_test_lag1['MAPE']:.3f} |
| **{best_model_name} (ML)** | **{test_res['MAE']:.1f}** | **{test_res['RMSE']:.1f}** | **{test_res['R2']:.3f}** | **{test_res['MAPE']:.3f}** |

"""
    if test_res['MAE'] < bl_test_lag1['MAE'] and test_res['MAE'] < bl_test_hist['MAE']:
        report_content += "> ✅ **Success:** The ML model successfully beat the baselines on the out-of-sample Test set.\n"
    else:
        report_content += "> ⚠️ **Warning:** The ML model failed to beat the best baseline. Adding mandi data did not provide enough forward-looking signal to overcome the strong autocorrelation of yield.\n"
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    main()
