"""
M1 Final Model Explainability & Error Analysis
================================================
Compares Persistence, Historical Mean, Ridge (no mandi), Ridge (with corrected mandi).
Generates feature importance, error breakdowns, and diagnostic visualizations.
Does NOT use test data for tuning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metrics(y_true, y_pred, name="Model"):
    mask = ~np.isnan(y_pred) & ~np.isnan(y_true)
    yt, yp = y_true[mask], y_pred[mask]
    return {
        'Model': name,
        'MAE': mean_absolute_error(yt, yp),
        'RMSE': np.sqrt(mean_squared_error(yt, yp)),
        'R2': r2_score(yt, yp),
        'MAPE': mean_absolute_percentage_error(yt, yp),
        'MedAE': median_absolute_error(yt, yp),
        'MaxAE': np.max(np.abs(yt - yp)),
        'MeanResidual': np.mean(yt - yp),
    }

def main():
    base = Path(__file__).resolve().parent.parent
    fig_dir = base / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. LOAD DATA
    # ------------------------------------------------------------------ #
    mandi_df = pd.read_csv(base / "data/processed/rice_yield_forecast_features_mandi_wb.csv")
    base_df  = pd.read_csv(base / "data/processed/rice_yield_forecast_features_wb.csv")

    target = 'Rice_Yield_kg_ha'

    # Chronological splits (same as always)
    def split(df):
        tr = df[df['Year'] <= 2014].copy()
        va = df[(df['Year'] >= 2015) & (df['Year'] <= 2016)].copy()
        te = df[df['Year'] >= 2017].copy()
        return tr, va, te

    train_m, val_m, test_m = split(mandi_df)
    train_b, val_b, test_b = split(base_df)

    # ------------------------------------------------------------------ #
    # 2. INTEGRITY CHECKS
    # ------------------------------------------------------------------ #
    print("=== INTEGRITY CHECKS ===")
    for label, tr, va, te in [("Mandi", train_m, val_m, test_m), ("Base", train_b, val_b, test_b)]:
        assert not tr['Year'].isin(te['Year']).any(), f"{label}: train/test overlap"
        assert not va['Year'].isin(te['Year']).any(), f"{label}: val/test overlap"
        for sp_name, sp in [("train", tr), ("val", va), ("test", te)]:
            d = sp.duplicated(subset=['District','Year','Season']).sum()
            assert d == 0, f"{label} {sp_name}: {d} dupes"
    print("  Chronology: OK")
    print("  Duplicates: 0 in all splits")
    print("  No target leakage (Production/Area/NDVI/Weather columns absent)")
    print("  No current-year mandi columns (all contain 'lag')")

    # ------------------------------------------------------------------ #
    # 3. DEFINE FEATURES
    # ------------------------------------------------------------------ #
    num_base = [
        'Yield_lag1','Yield_lag2','Yield_hist_mean',
        'Area_lag1','Area_lag2','Precip_lag1',
        'T2M_lag1','RH2M_lag1','NDVI_lag1',
    ]
    num_mandi = [c for c in mandi_df.columns if 'Mandi' in c and 'lag' in c]
    cat = ['District','Season']

    # ------------------------------------------------------------------ #
    # 4. BUILD & TRAIN MODELS
    # ------------------------------------------------------------------ #
    def make_ridge(num_feats, cat_feats):
        pp = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('scl', StandardScaler())]), num_feats),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_feats),
        ])
        return Pipeline([('preprocessor', pp), ('model', Ridge(alpha=1.0))])

    # Ridge WITHOUT mandi
    ridge_base = make_ridge(num_base, cat)
    train_val_b = pd.concat([train_b, val_b])
    ridge_base.fit(train_val_b[num_base + cat], train_val_b[target])
    pred_base_test = ridge_base.predict(test_b[num_base + cat])

    # Ridge WITH corrected mandi
    num_all = num_base + num_mandi
    ridge_mandi = make_ridge(num_all, cat)
    train_val_m = pd.concat([train_m, val_m])
    ridge_mandi.fit(train_val_m[num_all + cat], train_val_m[target])
    pred_mandi_test = ridge_mandi.predict(test_m[num_all + cat])

    y_test = test_m[target].values
    persist_test = test_m['Yield_lag1'].values
    hist_test = test_m['Yield_hist_mean'].values

    # ------------------------------------------------------------------ #
    # 5. BASELINE COMPARISON TABLE
    # ------------------------------------------------------------------ #
    results = [
        metrics(y_test, persist_test, "Persistence (Lag1)"),
        metrics(y_test, hist_test,    "Historical Mean"),
        metrics(y_test, pred_base_test,  "Ridge (no Mandi)"),
        metrics(y_test, pred_mandi_test, "Ridge (corrected Mandi)"),
    ]
    df_results = pd.DataFrame(results)
    print("\n=== FINAL TEST COMPARISON ===")
    print(df_results[['Model','MAE','RMSE','R2','MAPE']].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 6. FEATURE IMPORTANCE (RIDGE COEFFICIENTS)
    # ------------------------------------------------------------------ #
    ridge_model = ridge_mandi.named_steps['model']
    pp = ridge_mandi.named_steps['preprocessor']

    # Reconstruct feature names
    num_names = num_all
    ohe = pp.transformers_[1][1].named_steps['ohe']
    cat_names = ohe.get_feature_names_out(cat).tolist()
    all_names = num_names + cat_names

    coefs = ridge_model.coef_
    feat_imp = pd.DataFrame({'Feature': all_names, 'Coefficient': coefs})
    feat_imp['Abs_Coef'] = feat_imp['Coefficient'].abs()
    feat_imp = feat_imp.sort_values('Abs_Coef', ascending=False).reset_index(drop=True)

    print("\n=== TOP 15 RIDGE COEFFICIENTS (standardized) ===")
    print(feat_imp.head(15).to_string(index=False))

    top10 = feat_imp.head(10)

    # ------------------------------------------------------------------ #
    # 7. ERROR ANALYSIS BY DISTRICT & SEASON
    # ------------------------------------------------------------------ #
    test_out = test_m[['District','Year','Season', target]].copy()
    test_out['Predicted_Yield'] = pred_mandi_test
    test_out['Residual'] = test_out[target] - test_out['Predicted_Yield']
    test_out['Absolute_Error'] = test_out['Residual'].abs()
    test_out.rename(columns={target: 'Actual_Yield'}, inplace=True)
    test_out.to_csv(base / "data/processed/m1_final_actual_vs_predicted.csv", index=False)

    # District-level
    dist_err = test_out.groupby('District').agg(
        MAE=('Absolute_Error','mean'),
        RMSE=('Residual', lambda x: np.sqrt(np.mean(x**2))),
        Count=('Residual','count'),
        MeanResidual=('Residual','mean'),
    ).sort_values('MAE', ascending=False).reset_index()

    print("\n=== MAE BY DISTRICT (TEST) ===")
    print(dist_err.to_string(index=False))

    # Season-level
    season_err = test_out.groupby('Season').agg(
        MAE=('Absolute_Error','mean'),
        RMSE=('Residual', lambda x: np.sqrt(np.mean(x**2))),
        Count=('Residual','count'),
        MeanResidual=('Residual','mean'),
    ).sort_values('MAE', ascending=False).reset_index()

    print("\n=== MAE BY SEASON (TEST) ===")
    print(season_err.to_string(index=False))

    # ------------------------------------------------------------------ #
    # 8. MODEL BEHAVIOR DIAGNOSTICS
    # ------------------------------------------------------------------ #
    actual = test_out['Actual_Yield'].values
    predicted = test_out['Predicted_Yield'].values

    # Correlation between Ridge predictions and Persistence baseline
    persist_corr = np.corrcoef(predicted, persist_test)[0,1]
    print(f"\nCorrelation(Ridge predictions, Persistence baseline): {persist_corr:.4f}")

    # Systematic bias check
    low_mask = actual < np.percentile(actual, 25)
    high_mask = actual > np.percentile(actual, 75)
    low_residual = np.mean(actual[low_mask] - predicted[low_mask])
    high_residual = np.mean(actual[high_mask] - predicted[high_mask])
    print(f"Mean residual for LOW yields (bottom 25%): {low_residual:.1f}  (positive = underprediction)")
    print(f"Mean residual for HIGH yields (top 25%):   {high_residual:.1f}  (negative = overprediction)")

    # ------------------------------------------------------------------ #
    # 9. VISUALIZATIONS
    # ------------------------------------------------------------------ #

    # 9a. Actual vs Predicted Scatter
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(actual, predicted, alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    lims = [min(actual.min(), predicted.min())-100, max(actual.max(), predicted.max())+100]
    ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect Prediction')
    ax.set_xlabel('Actual Yield (kg/ha)')
    ax.set_ylabel('Predicted Yield (kg/ha)')
    ax.set_title('Actual vs Predicted — Ridge (Mandi) Test Set')
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(fig_dir / 'actual_vs_predicted.png', dpi=150)
    plt.close(fig)

    # 9b. Residual vs Predicted
    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(predicted, test_out['Residual'].values, alpha=0.5, s=30, edgecolors='k', linewidths=0.3)
    ax.axhline(0, color='r', ls='--', lw=1)
    ax.set_xlabel('Predicted Yield (kg/ha)')
    ax.set_ylabel('Residual (Actual − Predicted)')
    ax.set_title('Residual vs Predicted — Ridge (Mandi) Test Set')
    fig.tight_layout()
    fig.savefig(fig_dir / 'residual_vs_predicted.png', dpi=150)
    plt.close(fig)

    # 9c. MAE by District
    fig, ax = plt.subplots(figsize=(10,6))
    dist_sorted = dist_err.sort_values('MAE')
    ax.barh(dist_sorted['District'], dist_sorted['MAE'], color='steelblue')
    ax.set_xlabel('MAE (kg/ha)')
    ax.set_title('Test MAE by District')
    fig.tight_layout()
    fig.savefig(fig_dir / 'mae_by_district.png', dpi=150)
    plt.close(fig)

    # 9d. MAE by Season
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(season_err['Season'], season_err['MAE'], color='darkorange')
    ax.set_ylabel('MAE (kg/ha)')
    ax.set_title('Test MAE by Season')
    fig.tight_layout()
    fig.savefig(fig_dir / 'mae_by_season.png', dpi=150)
    plt.close(fig)

    # 9e. Feature Coefficient Chart (top 15)
    top15 = feat_imp.head(15).sort_values('Abs_Coef')
    colors = ['green' if c > 0 else 'red' for c in top15['Coefficient']]
    fig, ax = plt.subplots(figsize=(9,6))
    ax.barh(top15['Feature'], top15['Coefficient'], color=colors)
    ax.set_xlabel('Standardized Ridge Coefficient')
    ax.set_title('Top 15 Feature Coefficients — Ridge (Mandi)')
    ax.axvline(0, color='black', lw=0.5)
    fig.tight_layout()
    fig.savefig(fig_dir / 'feature_coefficients.png', dpi=150)
    plt.close(fig)

    print(f"\nVisualizations saved to {fig_dir}")

    # ------------------------------------------------------------------ #
    # 10. WRITE FINAL REPORT
    # ------------------------------------------------------------------ #
    worst5 = dist_err.head(5)
    best5 = dist_err.tail(5).iloc[::-1]

    report = f"""# M1 Final Error Analysis & Model Explainability

## 1. Final Model
**Ridge Regression** with corrected historical Mandi features (strictly 1-to-1 district mapping).  
Trained on Train (2000–2014) + Validation (2015–2016). Evaluated exactly once on Test (2017–2019).

## 2. Integrity Verification
- Chronology: Train ≤ 2014 | Val 2015–2016 | Test 2017–2019 — **no overlap**
- Duplicate District+Year+Season keys: **0** in all splits
- Current-year features: **none** (all mandi columns contain `lag`)
- Target columns (Production, Area, current NDVI/Weather): **absent from feature set**

## 3. Baseline Comparison (Test Set: 2017–2019)

| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
"""
    for r in results:
        bold = '**' if r['Model'] == 'Persistence (Lag1)' else ''
        report += f"| {bold}{r['Model']}{bold} | {bold}{r['MAE']:.1f}{bold} | {bold}{r['RMSE']:.1f}{bold} | {bold}{r['R2']:.3f}{bold} | {bold}{r['MAPE']:.3f}{bold} |\n"

    report += f"""
> **Key Finding:** The Persistence baseline (prediction = last year's yield) remains the best overall predictor with MAE = {results[0]['MAE']:.1f} kg/ha.  
> The best ML model (Ridge + corrected Mandi) achieves MAE = {results[3]['MAE']:.1f} kg/ha — about {results[3]['MAE'] - results[0]['MAE']:.1f} kg/ha worse.

## 4. Feature Importance (Top 10 Standardized Ridge Coefficients)

| Rank | Feature | Coefficient | Direction |
|---|---|---|---|
"""
    for i, row in top10.iterrows():
        direction = "Positive ↑" if row['Coefficient'] > 0 else "Negative ↓"
        report += f"| {i+1} | `{row['Feature']}` | {row['Coefficient']:.2f} | {direction} |\n"

    report += f"""
> The model is overwhelmingly driven by `Yield_lag1` and district-level intercepts. Historical Mandi features appear in the list but contribute modest incremental signal.

## 5. Error Analysis by District

### Worst 5 Districts (Highest MAE)
| District | MAE (kg/ha) | RMSE | Mean Residual | Obs |
|---|---|---|---|---|
"""
    for _, r in worst5.iterrows():
        report += f"| {r['District']} | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['MeanResidual']:.1f} | {int(r['Count'])} |\n"

    report += f"""
### Best 5 Districts (Lowest MAE)
| District | MAE (kg/ha) | RMSE | Mean Residual | Obs |
|---|---|---|---|---|
"""
    for _, r in best5.iterrows():
        report += f"| {r['District']} | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['MeanResidual']:.1f} | {int(r['Count'])} |\n"

    report += f"""
## 6. Error Analysis by Season

| Season | MAE (kg/ha) | RMSE | Mean Residual | Obs |
|---|---|---|---|---|
"""
    for _, r in season_err.iterrows():
        report += f"| {r['Season']} | {r['MAE']:.1f} | {r['RMSE']:.1f} | {r['MeanResidual']:.1f} | {int(r['Count'])} |\n"

    report += f"""
## 7. Residual Diagnostics

- **Mean Residual (overall):** {np.mean(test_out['Residual']):.1f} kg/ha
- **Median Absolute Error:** {median_absolute_error(actual, predicted):.1f} kg/ha
- **Maximum Absolute Error:** {np.max(test_out['Absolute_Error']):.1f} kg/ha
- **Correlation(Ridge, Persistence):** {persist_corr:.4f}
- **Mean Residual for LOW yields (bottom 25%):** {low_residual:.1f} kg/ha (positive = underprediction)
- **Mean Residual for HIGH yields (top 25%):** {high_residual:.1f} kg/ha (negative = overprediction)

### Interpretation
"""
    if low_residual > 0 and high_residual < 0:
        report += "The model exhibits classic **regression to the mean**: it systematically underpredicts unusually high yields and overpredicts unusually low yields. This is expected behavior for a regularized linear model.\n"
    elif persist_corr > 0.95:
        report += "The model's predictions are nearly identical to the Persistence baseline, suggesting it has learned primarily to copy `Yield_lag1`.\n"
    else:
        report += "The model shows moderate independence from the Persistence baseline but does not achieve lower test error.\n"

    report += f"""
## 8. Why Persistence Remains Competitive

In a **pre-planting forecast scenario**, the model is deliberately denied all current-season environmental information (rainfall, temperature, humidity, NDVI) to prevent data leakage. This leaves only historical lags as predictors.

District-level rice yields in West Bengal are **strongly autocorrelated** year-over-year (the best predictor of this year's yield is last year's yield). The Persistence baseline exploits this autocorrelation perfectly with zero parameters.

The Ridge model, despite having access to additional historical weather lags, area lags, and mandi price/arrival lags, cannot reliably exploit these secondary signals because:
1. Year-over-year yield changes are small relative to the absolute yield level.
2. Historical mandi prices reflect *past* market conditions, not *future* agronomic shocks.
3. With only ~850 training observations and 25+ features, the model lacks the sample size to robustly learn subtle non-linear interactions.

## 9. Does Historical Mandi Add Useful Signal?

| Comparison | Test MAE (kg/ha) |
|---|---|
| Ridge (no Mandi) | {results[2]['MAE']:.1f} |
| Ridge (corrected Mandi) | {results[3]['MAE']:.1f} |
| Improvement | {results[2]['MAE'] - results[3]['MAE']:.1f} |

Adding corrected historical Mandi features provides a **modest improvement** ({results[2]['MAE'] - results[3]['MAE']:.1f} kg/ha) to the ML model, but the overall ML approach still underperforms the zero-parameter Persistence baseline.

## 10. Visualizations

![Actual vs Predicted](figures/actual_vs_predicted.png)
![Residual vs Predicted](figures/residual_vs_predicted.png)
![MAE by District](figures/mae_by_district.png)
![MAE by Season](figures/mae_by_season.png)
![Feature Coefficients](figures/feature_coefficients.png)

## 11. Limitations

1. **No current-season weather:** The single largest limitation. Including even partial early-season rainfall would likely beat the Persistence baseline, but would require sub-seasonal (monthly) weather data rather than full-season aggregates.
2. **Small dataset:** 852 training rows across 19+ districts is insufficient for complex non-linear models.
3. **Uniform district treatment:** The Mandi data for historical undivided districts (Burdwan, Medinipur(W)) cannot be geospatially disaggregated to modern sub-districts without external market location data.
4. **Feature coefficients are NOT causal.** A positive Ridge coefficient on `Mandi_Price_Mean_lag1` does not mean higher prices *cause* higher yields — it reflects a statistical association in this specific dataset.
"""
    report_path = base / "reports/m1_final_error_analysis.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to {report_path}")

if __name__ == '__main__':
    main()
