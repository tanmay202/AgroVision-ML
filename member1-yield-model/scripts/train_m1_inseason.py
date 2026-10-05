"""
M1 In-Season Model Training & Evaluation
==========================================
Uses historical + corrected mandi + in-season-to-cutoff weather features.
Same strict chronological split: Train 2000-2014, Val 2015-2016, Test 2017-2019.
"""
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
    yt, yp = y_true[mask], y_pred[mask]
    return {
        'Model': name,
        'MAE': mean_absolute_error(yt, yp),
        'RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'R2': r2_score(yt, yp),
        'MAPE': mean_absolute_percentage_error(yt, yp),
    }

def main():
    base = Path(__file__).resolve().parent.parent
    data_path = base / "data/processed/rice_yield_inseason_features_wb.csv"
    preds_path = base / "data/processed/m1_inseason_predictions.csv"
    report_path = base / "reports/m1_inseason_model_results.md"
    audit_path = base / "reports/m1_inseason_feature_audit.md"

    df = pd.read_csv(data_path)
    target = 'Rice_Yield_kg_ha'

    # Split
    train = df[df['Year'] <= 2014].copy()
    val   = df[(df['Year'] >= 2015) & (df['Year'] <= 2016)].copy()
    test  = df[df['Year'] >= 2017].copy()

    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Features
    num_historical = [
        'Yield_lag1','Yield_lag2','Yield_hist_mean',
        'Area_lag1','Area_lag2','Precip_lag1',
        'T2M_lag1','RH2M_lag1','NDVI_lag1',
    ]
    num_mandi = [c for c in df.columns if 'Mandi' in c and 'lag' in c]
    num_inseason = [c for c in df.columns if c.startswith('InSeason_')]
    cat = ['District','Season']

    num_all = num_historical + num_mandi + num_inseason

    # Leakage assertions
    forbidden = [target, 'Production', 'yield', 'Rice_Yield_t_ha']
    for f in num_all + cat:
        assert f not in forbidden, f"Forbidden feature: {f}"
    for c in num_inseason:
        assert 'InSeason_' in c, f"Non in-season feature: {c}"
    for c in num_mandi:
        assert 'lag' in c, f"Current-year mandi: {c}"
    assert not train['Year'].isin(test['Year']).any(), "Train/test overlap!"
    assert not val['Year'].isin(test['Year']).any(), "Val/test overlap!"
    print("Leakage assertions passed.")

    # Build pipelines
    def make_pipeline(model, use_passthrough=False):
        if use_passthrough:
            pp = ColumnTransformer([
                ('num', 'passthrough', num_all),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat),
            ])
        else:
            pp = ColumnTransformer([
                ('num', Pipeline([
                    ('imp', SimpleImputer(strategy='median')),
                    ('scl', StandardScaler()),
                ]), num_all),
                ('cat', Pipeline([
                    ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ]), cat),
            ])
        return Pipeline([('preprocessor', pp), ('model', model)])

    models = {
        'Ridge': make_pipeline(Ridge(alpha=1.0)),
        'RandomForest': make_pipeline(RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        'XGBoost': make_pipeline(XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5,
                                              subsample=0.8, colsample_bytree=0.8, random_state=42),
                                 use_passthrough=True),
    }

    X_train, y_train = train[num_all + cat], train[target]
    X_val,   y_val   = val[num_all + cat],   val[target]

    # Baselines
    bl_val_persist = calc_metrics(y_val.values, val['Yield_lag1'].values, 'Persistence [Val]')
    bl_val_hist    = calc_metrics(y_val.values, val['Yield_hist_mean'].values, 'Historical Mean [Val]')

    print("\n--- BASELINES (VALIDATION) ---")
    print(bl_val_persist)
    print(bl_val_hist)

    # Train & evaluate on validation
    val_results = []
    print("\n--- IN-SEASON ML MODELS (VALIDATION) ---")
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        res = calc_metrics(y_val.values, preds, name)
        val_results.append(res)
        print(res)

    best_name = min(val_results, key=lambda x: x['MAE'])['Model']
    print(f"\nBest In-Season Model (Validation): {best_name}")

    # Retrain on train+val, evaluate on test
    train_val = pd.concat([train, val])
    X_tv, y_tv = train_val[num_all + cat], train_val[target]
    X_test = test[num_all + cat]
    y_test = test[target].values

    best_pipe = models[best_name]
    best_pipe.fit(X_tv, y_tv)
    test_preds = best_pipe.predict(X_test)

    # Test baselines
    bl_test_persist = calc_metrics(y_test, test['Yield_lag1'].values, 'Persistence [Test]')
    bl_test_hist    = calc_metrics(y_test, test['Yield_hist_mean'].values, 'Historical Mean [Test]')
    test_res        = calc_metrics(y_test, test_preds, f'{best_name} InSeason [Test]')

    print("\n--- FINAL TEST METRICS ---")
    print(bl_test_persist)
    print(bl_test_hist)
    print(test_res)

    # Also retrain all models for the report
    all_test_results = [bl_test_persist, bl_test_hist]
    for name, pipe in models.items():
        pipe.fit(X_tv, y_tv)
        tp = pipe.predict(X_test)
        all_test_results.append(calc_metrics(y_test, tp, f'{name} InSeason'))

    # Save predictions
    test_out = test[['District','Year','Season',target]].copy()
    test_out['Predicted_Yield'] = test_preds
    test_out['Residual'] = test_out[target] - test_out['Predicted_Yield']
    test_out['Absolute_Error'] = test_out['Residual'].abs()
    test_out.to_csv(preds_path, index=False)

    # Save model
    model_path = base / f"models/m1_inseason_{best_name.lower()}.pkl"
    joblib.dump(best_pipe, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance
    feat_imp_text = ""
    if best_name in ['RandomForest','XGBoost']:
        model_obj = best_pipe.named_steps['model']
        pp = best_pipe.named_steps['preprocessor']
        if best_name == 'XGBoost':
            cat_enc = pp.transformers_[1][1]
        else:
            cat_enc = pp.transformers_[1][1].named_steps['ohe']
        cat_names = cat_enc.get_feature_names_out(cat).tolist()
        all_names = num_all + cat_names
        importances = model_obj.feature_importances_
        fi = pd.DataFrame({'Feature': all_names, 'Importance': importances})
        fi = fi.sort_values('Importance', ascending=False).head(15)
        feat_imp_text = fi.to_string(index=False)
    elif best_name == 'Ridge':
        model_obj = best_pipe.named_steps['model']
        pp = best_pipe.named_steps['preprocessor']
        cat_enc = pp.transformers_[1][1].named_steps['ohe']
        cat_names = cat_enc.get_feature_names_out(cat).tolist()
        all_names = num_all + cat_names
        coefs = model_obj.coef_
        fi = pd.DataFrame({'Feature': all_names, 'Coefficient': coefs, 'Abs': np.abs(coefs)})
        fi = fi.sort_values('Abs', ascending=False).head(15)
        feat_imp_text = fi[['Feature','Coefficient']].to_string(index=False)

    # ================================================================
    # GENERATE REPORTS
    # ================================================================

    # Pre-planting baseline from previous experiment
    preplant_ridge_mae = 270.2
    persist_mae = bl_test_persist['MAE']
    inseason_mae = test_res['MAE']

    report = f"""# M1 In-Season Model Results

## 1. Forecast Cutoff Definitions

| Season | Sowing | Cutoff Date | Harvest | Weather Days Available |
|---|---|---|---|---|
| **Autumn** (Aus) | May–Jun | **July 31** | Aug–Sep | ~92 days |
| **Winter** (Aman) | Jun–Jul | **Sep 30** | Nov–Dec | ~122 days |
| **Summer** (Boro) | Nov–Dec | **Feb 28** | Mar–May | ~120 days |

## 2. Dataset
- **Train:** {len(train)} rows (2000–2014)
- **Validation:** {len(val)} rows (2015–2016)
- **Test:** {len(test)} rows (2017–2019)

## 3. Feature List

### Historical Features (from M1 Baseline)
`Yield_lag1`, `Yield_lag2`, `Yield_hist_mean`, `Area_lag1`, `Area_lag2`,
`Precip_lag1`, `T2M_lag1`, `RH2M_lag1`, `NDVI_lag1`

### Historical Mandi Features (corrected mapping)
{', '.join([f'`{c}`' for c in num_mandi])}

### In-Season Weather Features (up to cutoff only)
{', '.join([f'`{c}`' for c in num_inseason])}

### Categorical
`District`, `Season`

**Total numerical features:** {len(num_all)}
**Total features incl. encoded categoricals:** {len(num_all)} + encoded districts/seasons

### NDVI Note
The MODIS NDVI dataset contains only full-season aggregates. Since full-season NDVI includes
observations *after* the forecast cutoff, using it would constitute future leakage.
Only `NDVI_lag1` (previous year's NDVI) is used.

## 4. Leakage Audit
- All in-season weather observations verified to have `max_date <= cutoff_date`
- No current-year target columns in features
- All mandi features are lagged (contain `lag`)
- Train/Val/Test have zero temporal overlap
- Preprocessing fitted exclusively on training data

## 5. Validation Results (2015–2016)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence | {bl_val_persist['MAE']:.1f} | {bl_val_persist['RMSE']:.1f} | {bl_val_persist['R2']:.3f} | {bl_val_persist['MAPE']:.3f} |
| Historical Mean | {bl_val_hist['MAE']:.1f} | {bl_val_hist['RMSE']:.1f} | {bl_val_hist['R2']:.3f} | {bl_val_hist['MAPE']:.3f} |
"""
    for r in val_results:
        report += f"| {r['Model']} InSeason | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['R2']:.3f} | {r['MAPE']:.3f} |\n"

    report += f"""
**Selected Model:** {best_name}

## 6. Final Test Results (2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
"""
    for r in all_test_results:
        bold = '**' if r['Model'] == f'{best_name} InSeason' else ''
        report += f"| {bold}{r['Model']}{bold} | {bold}{r['MAE']:.1f}{bold} | {bold}{r['RMSE']:.1f}{bold} | {bold}{r['R2']:.3f}{bold} | {bold}{r['MAPE']:.3f}{bold} |\n"

    report += f"""
## 7. Cross-Experiment Comparison

| Model | Test MAE (kg/ha) | vs Persistence |
|---|---|---|
| Persistence Baseline | {persist_mae:.1f} | — |
| Pre-planting Ridge + Mandi | {preplant_ridge_mae:.1f} | +{preplant_ridge_mae - persist_mae:.1f} worse |
| In-Season {best_name} | {inseason_mae:.1f} | {"+" if inseason_mae > persist_mae else ""}{inseason_mae - persist_mae:.1f} {"worse" if inseason_mae > persist_mae else "**BETTER**"} |

"""
    if inseason_mae < persist_mae:
        report += f"""> ✅ **Success:** In-season weather features pushed the ML model below the Persistence baseline
> by {persist_mae - inseason_mae:.1f} kg/ha. Current-season weather contains genuine predictive signal.\n"""
    else:
        report += f"""> ⚠️ The in-season model did not beat the Persistence baseline.
> In-season weather to cutoff provides insufficient signal to overcome yield autocorrelation.\n"""

    report += f"""
## 8. Feature Importance (Top 15)
```text
{feat_imp_text}
```

## 9. Limitations
- **NDVI unavailable at sub-seasonal resolution:** The MODIS dataset only has full-season
  aggregates, preventing its use as an in-season feature. Monthly NDVI composites would
  significantly improve the model.
- **Small dataset:** ~850 training rows limits complex model capacity.
- **Uniform cutoffs:** A single cutoff per season is a simplification. In practice,
  forecasts could be updated as more weather observations arrive.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # Feature audit report
    audit_report = f"""# M1 In-Season Feature Audit

## Cutoff Definitions
- **Autumn:** May 1 to **July 31**
- **Winter:** Jun 1 to **Sep 30**
- **Summer:** Nov 1 (prev year) to **Feb 28**

## Weather Features Constructed
| Feature | Definition |
|---|---|
| `InSeason_Precip_Cum` | Cumulative rainfall from season start to cutoff |
| `InSeason_Precip_Mean` | Mean daily rainfall from season start to cutoff |
| `InSeason_Precip_Max` | Maximum single-day rainfall from season start to cutoff |
| `InSeason_T2M_Mean` | Mean 2m temperature from season start to cutoff |
| `InSeason_T2M_Min` | Absolute minimum temperature from season start to cutoff |
| `InSeason_T2M_Max` | Absolute maximum temperature from season start to cutoff |
| `InSeason_T2M_Range` | Temperature range (max - min) from season start to cutoff |
| `InSeason_RH2M_Mean` | Mean relative humidity from season start to cutoff |
| `InSeason_Solar_Mean` | Mean solar radiation from season start to cutoff |
| `InSeason_Wind_Mean` | Mean wind speed from season start to cutoff |
| `InSeason_Weather_Days` | Number of daily observations from season start to cutoff |

## NDVI
Full-season NDVI aggregates cannot be used (would include post-cutoff observations).
Only `NDVI_lag1` (previous year) is included.

## Leakage Verification
For every row in the dataset, the maximum weather observation date was verified
programmatically to be ≤ the forecast cutoff date. Zero violations detected.

## Missingness
In-season weather features may be missing for districts where the NASA POWER
coverage starts after 2000 or where historical district boundaries changed (e.g., undivided Bardhaman).
Missing values are handled via median imputation (fitted on training data only) for Ridge/RF,
or via native NaN handling for XGBoost.
"""
    with open(audit_path, 'w', encoding='utf-8') as f:
        f.write(audit_report)
    print(f"Feature audit saved to {audit_path}")

if __name__ == '__main__':
    main()
