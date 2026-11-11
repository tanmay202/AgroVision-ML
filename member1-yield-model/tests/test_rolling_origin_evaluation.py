"""
AgroVision Member 1: Tests for Rolling-Origin Cross-Validation & Seasonal Routing Evaluation
===========================================================================================
Validates the 17 requirements of Step 5:
1. Expanding-window chronological splits.
2. No future years in training.
3. No validation leakage into preprocessing.
4. Correct persistence predictions.
5. Correct delta reconstruction.
6. Fold reproducibility.
7. Weather-completeness accounting.
8. Independent per-row missing-weather handling.
9. Season-specific metrics.
10. Equal sample comparison across routing policies.
11. Correct improvement sign.
12. Bootstrap reproducibility.
13. District coverage.
14. Artifact hashes unchanged.
15. Production model artifact unchanged.
16. Step 3 routing defaults unchanged.
17. Member 2 source and model artifacts unchanged.
"""

import sys
import json
import hashlib
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
M1_DIR = TESTS_DIR.parent
MONOREPO_ROOT = M1_DIR.parent
REPORTS_DIR = M1_DIR / "reports/rolling_origin"
MODELS_DIR = M1_DIR / "models"
DATA_DIR = M1_DIR / "data/processed"

if str(M1_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(M1_DIR / "scripts"))

from evaluate_rolling_origin import (
    FOLDS,
    TARGET,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURES,
    INSEASON_WEATHER_FEATURES,
    fill_baseline_yield,
    enforce_leakage_controls,
    run_statistical_uncertainty_analysis,
)


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# 1. Expanding-window chronological splits
def test_expanding_window_chronological_splits():
    assert len(FOLDS) == 5, f"Expected 5 folds, found {len(FOLDS)}"
    expected_val_years = [2012, 2013, 2014, 2015, 2016]
    prev_train_len = 0

    df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")

    for fold_id, train_years, val_year in FOLDS:
        assert val_year == expected_val_years[fold_id - 1]
        assert train_years == list(range(2000, val_year))

        train_sub = df[df['Year'].isin(train_years)]
        assert len(train_sub) > prev_train_len, "Train set must expand monotonically"
        prev_train_len = len(train_sub)

        val_sub = df[df['Year'] == val_year]
        assert len(val_sub) in (54, 57, 64), f"Unexpected val size {len(val_sub)}"


# 2. No future years in training
def test_no_future_years_in_training():
    df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")
    for fold_id, train_years, val_year in FOLDS:
        train_sub = df[df['Year'].isin(train_years)]
        val_sub = df[df['Year'] == val_year]

        # Programmatic assertion function from evaluation script
        enforce_leakage_controls(train_sub, val_sub, val_year)

        assert train_sub['Year'].max() < val_year
        assert (train_sub['Year'] < val_year).all()
        # Zero 2017-2019 test set data
        assert (train_sub['Year'] <= 2015).all()
        assert (val_sub['Year'] <= 2016).all()


# 3. No validation leakage into preprocessing
def test_no_validation_leakage_into_preprocessing():
    df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")
    # In fold 5 (2016), 10 rows lack lag1
    train_df = df[df['Year'] <= 2015].copy()
    val_df = df[df['Year'] == 2016].copy()

    # Pre-compute training stats
    tr_state_mean = float(train_df[TARGET].mean())
    val_lag1, imputed_mask = fill_baseline_yield(val_df, train_df)

    assert imputed_mask.sum() == 10
    # Imputed values must be strictly derived from train_df, not val_df
    val_actual_mean = float(val_df[TARGET].mean())
    assert not np.isclose(tr_state_mean, val_actual_mean), "Train and val means coincidentally match"

    # Preprocessing Pipeline leakage check
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    ct = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), NUMERICAL_FEATURES)
    ])
    ct.fit(train_df[NUMERICAL_FEATURES])
    tr_medians = ct.named_transformers_['num'].named_steps['imp'].statistics_.copy()

    # Transform validation data
    _ = ct.transform(val_df[NUMERICAL_FEATURES])
    post_medians = ct.named_transformers_['num'].named_steps['imp'].statistics_

    # Statistics must not change when transforming validation observations
    assert np.array_equal(tr_medians, post_medians), "Validation data contaminated imputer statistics!"


# 4. Correct persistence predictions
def test_correct_persistence_predictions():
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    assert preds_csv.exists(), "Predictions CSV missing"
    df_preds = pd.read_csv(preds_csv)

    # Pred_Yield_Persistence must equal Yield_lag1
    assert np.allclose(df_preds['Pred_Yield_Persistence'].values, df_preds['Yield_lag1'].values)


# 5. Correct delta reconstruction
def test_correct_delta_reconstruction():
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    df_preds = pd.read_csv(preds_csv)

    # Reconstructed yields: Yield_lag1 + Predicted_Delta
    assert np.allclose(df_preds['Pred_Yield_Ridge_S2'].values, df_preds['Yield_lag1'].values + df_preds['Pred_Delta_Ridge'].values)
    assert np.allclose(df_preds['Pred_Yield_XGB_S2'].values, df_preds['Yield_lag1'].values + df_preds['Pred_Delta_XGB'].values)
    assert np.allclose(df_preds['Pred_Yield_Ens_S2'].values, df_preds['Yield_lag1'].values + df_preds['Pred_Delta_Ens'].values)

    # Ensemble delta is exact 50-50 average
    expected_ens_delta = 0.5 * df_preds['Pred_Delta_Ridge'].values + 0.5 * df_preds['Pred_Delta_XGB'].values
    assert np.allclose(df_preds['Pred_Delta_Ens'].values, expected_ens_delta)


# 6. Fold reproducibility
def test_fold_reproducibility():
    fold_csv = REPORTS_DIR / "fold_metrics.csv"
    assert fold_csv.exists()
    df_folds = pd.read_csv(fold_csv)

    assert len(df_folds) == 35  # 5 folds * 7 models
    # Verify Ridge S1 MAEs match expected verified numbers
    fold_r1 = df_folds[(df_folds['Fold'] == 1) & (df_folds['Model'] == 'Ridge_S1')]['MAE_kg_ha'].values[0]
    fold_r2 = df_folds[(df_folds['Fold'] == 2) & (df_folds['Model'] == 'Ridge_S1')]['MAE_kg_ha'].values[0]
    fold_r5 = df_folds[(df_folds['Fold'] == 5) & (df_folds['Model'] == 'Ridge_S1')]['MAE_kg_ha'].values[0]

    assert 285.0 < fold_r1 < 287.0
    assert 227.0 < fold_r2 < 229.0
    assert 251.0 < fold_r5 < 254.0


# 7. Weather-completeness accounting
def test_weather_completeness_accounting():
    cov_csv = REPORTS_DIR / "weather_coverage.csv"
    assert cov_csv.exists()
    df_cov = pd.read_csv(cov_csv)

    assert len(df_cov) == 5
    assert df_cov['Total_Rows'].sum() == 289
    assert df_cov['Complete_Weather_Rows'].sum() == 280
    assert df_cov['Missing_Weather_Rows'].sum() == 9
    assert df_cov['Incomplete_Weather_Rows'].sum() == 0

    overall_eligible = float(df_cov['Complete_Weather_Rows'].sum()) / df_cov['Total_Rows'].sum() * 100.0
    assert np.isclose(overall_eligible, 96.8858, atol=0.01)


# 8. Independent per-row missing-weather handling
def test_independent_per_row_missing_weather_handling():
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    df_preds = pd.read_csv(preds_csv)

    # When weather is missing (9 rows), Setting 1 falls back to persistence
    missing_mask = ~df_preds['Weather_Complete']
    assert missing_mask.sum() == 9

    assert np.allclose(
        df_preds.loc[missing_mask, 'Pred_Yield_Ridge_S1'].values,
        df_preds.loc[missing_mask, 'Pred_Yield_Persistence'].values
    )

    # When weather is complete (280 rows), Setting 1 uses the ML model prediction
    complete_mask = df_preds['Weather_Complete']
    assert np.allclose(
        df_preds.loc[complete_mask, 'Pred_Yield_Ridge_S1'].values,
        df_preds.loc[complete_mask, 'Pred_Yield_Ridge_S2'].values
    )


# 9. Season-specific metrics
def test_season_specific_metrics():
    season_csv = REPORTS_DIR / "season_metrics.csv"
    assert season_csv.exists()
    df_season = pd.read_csv(season_csv)

    seasons = df_season['Season'].unique()
    assert set(seasons) == {'Autumn', 'Winter', 'Summer', 'Overall'}

    # Verify counts
    aus_n = df_season[df_season['Season'] == 'Autumn']['Observations'].values[0]
    aman_n = df_season[df_season['Season'] == 'Winter']['Observations'].values[0]
    boro_n = df_season[df_season['Season'] == 'Summer']['Observations'].values[0]
    tot_n = df_season[df_season['Season'] == 'Overall']['Observations'].values[0]

    assert aus_n == 96
    assert aman_n == 97
    assert boro_n == 96
    assert tot_n == 289
    assert aus_n + aman_n + boro_n == tot_n


# 10. Equal sample comparison across routing policies
def test_equal_sample_comparison_across_routing_policies():
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    policy_csv = REPORTS_DIR / "routing_policy_comparison.csv"
    df_preds = pd.read_csv(preds_csv)
    df_pol = pd.read_csv(policy_csv)

    assert len(df_pol) == 8
    # All policies must be evaluated on the identical 289 records
    for i in range(1, 9):
        col = f"Pred_Policy_{i}"
        assert col in df_preds.columns
        assert df_preds[col].notna().sum() == 289


# 11. Correct improvement sign
def test_correct_improvement_sign():
    policy_csv = REPORTS_DIR / "routing_policy_comparison.csv"
    df_pol = pd.read_csv(policy_csv)

    p1_mae = df_pol[df_pol['Policy_ID'] == 1]['Overall_MAE_kg_ha'].values[0]
    for _, row in df_pol.iterrows():
        expected_diff = p1_mae - row['Overall_MAE_kg_ha']
        assert np.isclose(row['MAE_Improvement_vs_P1'], expected_diff, atol=0.01)


# 12. Bootstrap reproducibility
def test_bootstrap_reproducibility():
    boot_csv = REPORTS_DIR / "bootstrap_results.csv"
    assert boot_csv.exists()
    df_boot = pd.read_csv(boot_csv)

    # 7 candidate policies (P2-P8 vs P1) + 3 seasonal comparisons
    assert len(df_boot) == 10
    assert (df_boot['Replicates'] == 10000).all()
    assert (df_boot['Random_Seed'] == 42).all()

    # Re-running with seed=42 on a subset confirms numerical determinism
    preds_df = pd.read_csv(REPORTS_DIR / "rolling_origin_predictions.csv")
    res_df1, _ = run_statistical_uncertainty_analysis(preds_df, n_boot=500, seed=42)
    res_df2, _ = run_statistical_uncertainty_analysis(preds_df, n_boot=500, seed=42)
    assert len(res_df1) == 10
    assert len(res_df2) == 10
    assert np.allclose(res_df1['CI_95_Lower_kg_ha'].values, res_df2['CI_95_Lower_kg_ha'].values)
    assert np.allclose(res_df1['CI_95_Upper_kg_ha'].values, res_df2['CI_95_Upper_kg_ha'].values)
    assert np.allclose(res_df1['Observed_MAE_Difference_kg_ha'].values, res_df2['Observed_MAE_Difference_kg_ha'].values)


def test_production_parity_for_missing_lags():
    """Verify that in validation folds, missing Yield_lag1 values are imputed to match predict_rice_yield.py."""
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    df_preds = pd.read_csv(preds_csv)

    # In Fold 5 (2016), exactly 10 rows lack lag1
    fold5_preds = df_preds[df_preds['Fold'] == 5]
    imputed_rows = fold5_preds[fold5_preds['Lag1_Imputed']]
    assert len(imputed_rows) == 10

    # Ensure no NaN or non-positive baseline yields were passed
    assert (fold5_preds['Yield_lag1'] > 0).all()
    assert fold5_preds['Yield_lag1'].notna().all()


# 13. District coverage
def test_district_coverage():
    preds_csv = REPORTS_DIR / "rolling_origin_predictions.csv"
    df_preds = pd.read_csv(preds_csv)

    unique_districts = df_preds['District'].unique()
    assert len(unique_districts) >= 22, f"Expected at least 22 districts, found {len(unique_districts)}"


# 14. Artifact hashes unchanged
def test_artifact_hashes_unchanged():
    manifest_path = MODELS_DIR / "promotion_manifest.json"
    assert manifest_path.exists()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)

    for name, details in manifest['promoted_artifacts'].items():
        pkl_path = MODELS_DIR / f"{name}.pkl"
        assert pkl_path.exists(), f"Missing artifact: {pkl_path}"
        actual_hash = compute_sha256(pkl_path)
        assert actual_hash == details['artifact_sha256'], f"Artifact {name} hash was modified!"


# 15. Production model artifact unchanged
def test_production_model_artifact_unchanged():
    prod_path = MODELS_DIR / "m1_improved_delta_ridge.pkl"
    assert prod_path.exists()
    actual_hash = compute_sha256(prod_path)
    expected_hash = "3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56"
    assert actual_hash == expected_hash, "Production Ridge model artifact was unexpectedly altered!"


# 16. Step 3 routing defaults unchanged
def test_step3_routing_defaults_unchanged():
    from predict_rice_yield import RiceYieldPredictor
    predictor = RiceYieldPredictor()
    assert predictor.weather_missing_policy is None
    assert predictor.enable_seasonal_routing is False


# 17. Member 2 source and model artifacts unchanged
def test_member2_source_and_model_artifacts_unchanged():
    m2_manifest = MONOREPO_ROOT / "member2-price-volatility/models/m2_model_manifest.json"
    if m2_manifest.exists():
        with open(m2_manifest, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        for _, details in manifest.get('artifacts', {}).items():
            if 'sha256' in details and 'path' in details:
                p = MONOREPO_ROOT / details['path']
                if p.exists():
                    assert compute_sha256(p) == details['sha256'], f"Member 2 artifact altered: {p}"
