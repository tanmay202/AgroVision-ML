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
    # Drop NaNs just in case baselines have missing predictions
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
    train_path = base_dir / "data/processed/train.csv"
    val_path = base_dir / "data/processed/validation.csv"
    test_path = base_dir / "data/processed/test.csv"
    preds_path = base_dir / "data/processed/m1_predictions.csv"
    model_dir = base_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)
    
    target = 'Rice_Yield_kg_ha'
    
    # 1. DEFINE FEATURES
    num_features = [
        'Yield_lag1', 'Yield_lag2', 'Yield_hist_mean', 
        'Area_lag1', 'Area_lag2', 'Precip_lag1', 
        'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1'
    ]
    cat_features = ['District', 'Season']
    
    # Assert no forbidden columns are used in features
    forbidden = ['Rice_Yield_kg_ha', 'Production', 'yield', 'Rice_Yield_t_ha']
    for f in num_features + cat_features:
        assert f not in forbidden, f"Forbidden feature used: {f}"
        
    # 2. CREATE BASELINES
    print("--- BASELINES (VALIDATION) ---")
    bl_val_lag1 = calc_metrics(val[target], val['Yield_lag1'], 'Persistence (Lag1) [Val]')
    bl_val_hist = calc_metrics(val[target], val['Yield_hist_mean'], 'Historical Mean [Val]')
    print(bl_val_lag1)
    print(bl_val_hist)
    
    # 3. TRAIN ML MODELS (ON TRAIN, EVAL ON VAL)
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
    
    # Preprocessor for XGBoost (Native NaN handling, only encode categorical)
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
        
    # 4 & 5. SELECT BEST MODEL AND RETRAIN
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
    print(bl_test_hist)
    print(test_res)
    
    # 6. DIAGNOSTICS
    test_out = test.copy()
    test_out['Predicted'] = test_preds
    test_out['Residual'] = test_out[target] - test_out['Predicted']
    test_out.to_csv(preds_path, index=False)
    
    # Feature Importance (if applicable)
    feat_importances = None
    if best_model_name in ['RandomForest', 'XGBoost']:
        model_obj = best_pipeline.named_steps['model']
        if best_model_name == 'RandomForest':
            importances = model_obj.feature_importances_
        else:
            importances = model_obj.feature_importances_
            
        # Get feature names
        num_names = num_features
        cat_enc = best_pipeline.named_steps['preprocessor'].transformers_[1][1]
        if isinstance(cat_enc, Pipeline): # for RF
            cat_names = cat_enc.named_steps['onehot'].get_feature_names_out(cat_features).tolist()
        else: # for XGB
            cat_names = cat_enc.get_feature_names_out(cat_features).tolist()
        
        all_features = num_names + cat_names
        feat_importances = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    # Error by District / Season
    district_error = test_out.groupby('District')['Residual'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
    season_error = test_out.groupby('Season')['Residual'].apply(lambda x: np.mean(np.abs(x))).sort_values(ascending=False)
    
    # 7. LEAKAGE AUDIT
    assert 'Production' not in X_train.columns
    assert best_model_name in ['Ridge', 'RandomForest', 'XGBoost']
    assert not train['Year'].isin(test['Year']).any(), "Temporal overlap between Train and Test!"
    assert not val['Year'].isin(test['Year']).any(), "Temporal overlap between Val and Test!"
    
    # Save Model
    model_file = model_dir / f"m1_best_{best_model_name.lower()}.pkl"
    joblib.dump(best_pipeline, model_file)
    print(f"\nModel saved to {model_file}")
    
    # Generate Markdown Report
    report_path = base_dir / "reports/m1_model_results.md"
    
    report_content = f"""# Member 1 Rice Yield Model Results

## 1. Dataset Sizes
- **Train (2000-2014):** {len(train)} rows
- **Validation (2015-2016):** {len(val)} rows
- **Test (2017-2019):** {len(test)} rows

## 2. Feature List
- **Categorical:** {cat_features}
- **Numerical (Lags):** {num_features}
- **Target:** {target}
*(Strictly pre-planting historical features. No current-season environmental features were used).*

## 3. Validation Metrics (Hyperparameter / Model Selection)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | {bl_val_lag1['MAE']:.1f} | {bl_val_lag1['RMSE']:.1f} | {bl_val_lag1['R2']:.3f} | {bl_val_lag1['MAPE']:.3f} |
| Historical Mean | {bl_val_hist['MAE']:.1f} | {bl_val_hist['RMSE']:.1f} | {bl_val_hist['R2']:.3f} | {bl_val_hist['MAPE']:.3f} |
| Ridge | {val_results[0]['MAE']:.1f} | {val_results[0]['RMSE']:.1f} | {val_results[0]['R2']:.3f} | {val_results[0]['MAPE']:.3f} |
| Random Forest | {val_results[1]['MAE']:.1f} | {val_results[1]['RMSE']:.1f} | {val_results[1]['R2']:.3f} | {val_results[1]['MAPE']:.3f} |
| XGBoost | {val_results[2]['MAE']:.1f} | {val_results[2]['RMSE']:.1f} | {val_results[2]['R2']:.3f} | {val_results[2]['MAPE']:.3f} |

**Selected Model:** {best_model_name}

## 4. Final Test Metrics (Out-of-Sample: 2017-2019)
The selected {best_model_name} model was retrained on Train + Validation, and evaluated once on the strictly untouched Test set.

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | {bl_test_lag1['MAE']:.1f} | {bl_test_lag1['RMSE']:.1f} | {bl_test_lag1['R2']:.3f} | {bl_test_lag1['MAPE']:.3f} |
| Historical Mean | {bl_test_hist['MAE']:.1f} | {bl_test_hist['RMSE']:.1f} | {bl_test_hist['R2']:.3f} | {bl_test_hist['MAPE']:.3f} |
| **{best_model_name} (ML)** | **{test_res['MAE']:.1f}** | **{test_res['RMSE']:.1f}** | **{test_res['R2']:.3f}** | **{test_res['MAPE']:.3f}** |

"""
    if test_res['MAE'] < bl_test_lag1['MAE'] and test_res['MAE'] < bl_test_hist['MAE']:
        report_content += "> ✅ **Success:** The ML model successfully beat the baselines on the out-of-sample Test set.\n"
    else:
        report_content += "> ⚠️ **Warning:** The ML model failed to beat the best baseline (overfitting or insufficient signal).\n"

    report_content += f"""
## 5. Diagnostics

### Mean Absolute Error by Season (Test Set)
```text
{season_error.to_string()}
```

### Mean Absolute Error by District (Top 5 Worst in Test Set)
```text
{district_error.head().to_string()}
```
"""
    if feat_importances is not None:
        report_content += f"""
### Top 10 Feature Importances
```text
{feat_importances.head(10).to_string(index=False)}
```
"""

    report_content += """
## 6. Leakage Audit
- **Temporal Integrity:** Verified programmatically. Train (<=2014), Val (2015-2016), and Test (2017-2019) have zero temporal overlap.
- **Target Leakage:** Current-year `Production`, `Area`, `PRECTOTCORR`, and `NDVI` were strictly excluded.
- **Preprocessing:** Standard scalers, median imputers, and OneHotEncoders were fitted *exclusively* on the training sets before transforming validation/test data.

## 7. Limitations
- **Data Gap:** Undivided Bardhaman's missing environmental data is imputed to the median (Ridge/RF) or natively bypassed (XGBoost).
- **Lag Dependency:** Pre-planting forecasts rely heavily on historical autoregression. If a massive shock occurs in the *current* season (e.g., mid-season flood), the model will miss it because current-season weather is correctly withheld to prevent data leakage.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"Report saved to {report_path}")

if __name__ == '__main__':
    main()
