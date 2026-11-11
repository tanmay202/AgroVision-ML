"""
AgroVision Model 1 — Automated Reproducibility & Serialization Test Suite (Step 4)
==================================================================================
Tests covering all requirements from Part J:
1. Exact dependency version compatibility
2. New artifact loading
3. No InconsistentVersionWarning
4. Correct feature schema
5. Correct feature ordering
6. Correct target formulation
7. Prediction determinism
8. Artifact metadata
9. Artifact hash verification
10. Training reproducibility
11. Inference reproducibility
12. Existing production prediction compatibility
13. Weather-missing policy regression
14. Seasonal routing regression
15. Member 2 integration regression
"""

import sys
import json
import hashlib
import warnings
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import joblib
from sklearn.exceptions import InconsistentVersionWarning

TEST_DIR = Path(__file__).resolve().parent
BASE = TEST_DIR.parent
PROJECT_ROOT = BASE.parent
MODELS_DIR = BASE / "models"
DATA_DIR = BASE / "data/processed"

if str(BASE / "scripts") not in sys.path:
    sys.path.insert(0, str(BASE / "scripts"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict_rice_yield import (
    RiceYieldPredictor,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    AGRONOMIC_MIN_YIELD,
    AGRONOMIC_MAX_YIELD
)
from shared.predict_bridge import AgroVisionPredictor

SUPPORTED_MODELS = [
    "m1_improved_delta_ridge",
    "m1_improved_delta_xgboost_conservative",
    "m1_improved_delta_xgboost",
    "m1_best_ridge",
    "m1_inseason_xgboost"
]

IMPROVED_MODELS = [
    "m1_improved_delta_ridge",
    "m1_improved_delta_xgboost_conservative",
    "m1_improved_delta_xgboost"
]


@pytest.fixture(scope="module")
def sample_valid_input():
    return {
        'District': 'Hooghly',
        'Season': 'Winter',
        'Yield_lag1': 2850.0,
        'Yield_lag2': 2800.0,
        'Area_lag1': 320000.0,
        'Area_lag2': 318000.0,
        'Precip_lag1': 1400.0,
        'T2M_lag1': 25.5,
        'RH2M_lag1': 75.0,
        'NDVI_lag1': 0.62,
        'InSeason_Precip_Cum': 950.0,
        'InSeason_Precip_Mean': 7.8,
        'InSeason_Precip_Max': 85.0,
        'InSeason_T2M_Mean': 27.2,
        'InSeason_T2M_Min': 22.0,
        'InSeason_T2M_Max': 33.5,
        'InSeason_T2M_Range': 11.5,
        'InSeason_RH2M_Mean': 80.5,
        'InSeason_Solar_Mean': 18.2,
        'InSeason_Wind_Mean': 1.8
    }


# 1. Exact dependency version compatibility
def test_exact_dependency_version_compatibility():
    import sklearn
    import xgboost
    import pandas
    import scipy

    assert sys.version.startswith("3.13.3"), f"Unexpected Python version: {sys.version}"
    assert sklearn.__version__ == "1.7.2", f"Unexpected scikit-learn version: {sklearn.__version__}"
    assert xgboost.__version__ == "3.4.1", f"Unexpected xgboost version: {xgboost.__version__}"
    assert joblib.__version__ == "1.5.2", f"Unexpected joblib version: {joblib.__version__}"
    assert pandas.__version__ == "2.2.3", f"Unexpected pandas version: {pandas.__version__}"
    assert np.__version__ == "2.2.6", f"Unexpected numpy version: {np.__version__}"
    assert scipy.__version__ == "1.16.3", f"Unexpected scipy version: {scipy.__version__}"

    req_file = BASE / "requirements.txt"
    assert req_file.exists(), "member1-yield-model/requirements.txt is missing"
    req_text = req_file.read_text(encoding="utf-8")
    assert "numpy==2.2.6" in req_text
    assert "pandas==2.2.3" in req_text
    assert "scipy==1.16.3" in req_text
    assert "scikit-learn==1.7.2" in req_text
    assert "xgboost==3.4.1" in req_text
    assert "joblib==1.5.2" in req_text
    assert "pytest==9.1.1" in req_text
    assert "3.13.3" in req_text


# 2. New artifact loading and prediction execution
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_new_artifact_loading(model_name):
    art_path = MODELS_DIR / f"{model_name}.pkl"
    meta_path = MODELS_DIR / f"{model_name}.metadata.json"
    assert art_path.exists(), f"Model artifact missing: {art_path}"
    assert meta_path.exists(), f"Model metadata missing: {meta_path}"

    model = joblib.load(art_path)
    assert model is not None
    assert hasattr(model, "predict")

    # Load metadata and perform sample prediction to verify pipeline operational integrity
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["training_features"]
    inseason_df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")
    test_rows = inseason_df[inseason_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).head(5)

    if "Yield_hist_mean" in features and "InSeason_Precip_Cum" not in features:
        # Benchmark baseline Ridge
        test_base = pd.read_csv(DATA_DIR / "test.csv").sort_values(by=['District', 'Year', 'Season']).head(5)
        X = test_base[features]
    else:
        X = test_rows[features]

    preds = model.predict(X)
    assert isinstance(preds, np.ndarray), f"Predictions from {model_name} is not numpy array"
    assert len(preds) == len(X), f"Prediction count mismatch for {model_name}"
    assert np.isfinite(preds).all(), f"Predictions contain non-finite values for {model_name}"


# 3. No InconsistentVersionWarning
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_no_inconsistent_version_warning(model_name):
    art_path = MODELS_DIR / f"{model_name}.pkl"
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        _ = joblib.load(art_path)
        iv_warnings = [w for w in recorded if issubclass(w.category, InconsistentVersionWarning)]
        assert len(iv_warnings) == 0, f"InconsistentVersionWarning raised loading {model_name}: {iv_warnings}"


# 4. Correct feature schema
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_correct_feature_schema(model_name):
    meta_path = MODELS_DIR / f"{model_name}.metadata.json"
    assert meta_path.exists(), f"Metadata file missing: {meta_path}"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if model_name in IMPROVED_MODELS:
        assert len(meta["training_features"]) == 20
        assert len(meta["numerical_features"]) == 18
        assert len(meta["categorical_features"]) == 2
        assert meta["categorical_features"] == ["District", "Season"]
        assert meta["numerical_features"] == NUMERICAL_FEATURES
        assert meta["training_features"] == ALL_FEATURES
    elif model_name == "m1_best_ridge":
        assert len(meta["training_features"]) == 11
        assert len(meta["numerical_features"]) == 9
        assert len(meta["categorical_features"]) == 2
    elif model_name == "m1_inseason_xgboost":
        assert len(meta["training_features"]) == 38
        assert len(meta["numerical_features"]) == 36
        assert len(meta["categorical_features"]) == 2

    # Target leakage protection across all models
    assert "Rice_Yield_kg_ha" not in meta["training_features"]
    assert "Production" not in meta["training_features"]


# 5. Correct feature ordering
@pytest.mark.parametrize("model_name", IMPROVED_MODELS)
def test_correct_feature_ordering(model_name):
    meta_path = MODELS_DIR / f"{model_name}.metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_order = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    assert ALL_FEATURES == expected_order
    assert meta["training_features"] == ALL_FEATURES


# 6. Correct target formulation
@pytest.mark.parametrize("mtype", ['improved_ridge', 'improved_xgb_cons', 'improved_xgb'])
def test_correct_target_formulation(mtype, sample_valid_input):
    predictor = RiceYieldPredictor(model_type=mtype)
    res = predictor.predict(sample_valid_input)

    assert "predicted_delta_kg_ha" in res
    assert "baseline_yield_kg_ha" in res
    assert "predicted_yield_kg_ha" in res

    # Verification: predicted_yield == baseline + delta
    expected = res["baseline_yield_kg_ha"] + res["predicted_delta_kg_ha"]
    assert abs(res["predicted_yield_kg_ha"] - expected) < 1e-4


# 7. Prediction determinism
@pytest.mark.parametrize("mtype", ['improved_ridge', 'improved_xgb_cons'])
def test_prediction_determinism(mtype, sample_valid_input):
    predictor = RiceYieldPredictor(model_type=mtype)
    res1 = predictor.predict(sample_valid_input)
    res2 = predictor.predict(sample_valid_input)
    res3 = predictor.predict(sample_valid_input)

    assert res1["predicted_yield_kg_ha"] == res2["predicted_yield_kg_ha"] == res3["predicted_yield_kg_ha"]
    assert res1["predicted_delta_kg_ha"] == res2["predicted_delta_kg_ha"] == res3["predicted_delta_kg_ha"]


# 8. Artifact metadata
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_artifact_metadata(model_name):
    meta_path = MODELS_DIR / f"{model_name}.metadata.json"
    assert meta_path.exists(), f"Metadata file missing: {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    required_keys = [
        "model_name", "artifact_filename", "architecture", "target",
        "target_formula", "prediction_formula", "hyperparameters",
        "training_period", "training_rows", "test_period", "test_rows",
        "python_version", "sklearn_version", "dependencies", "training_features",
        "numerical_features", "categorical_features", "data_preprocessing",
        "training_script", "dataset_path", "dataset_sha256", "artifact_sha256",
        "evaluation_metrics", "created_at"
    ]
    for k in required_keys:
        assert k in meta, f"Metadata key '{k}' missing in {model_name}.metadata.json"

    assert meta["training_rows"] == 970
    assert meta["test_rows"] == 192


# 9. Artifact hash verification & tamper detection
@pytest.mark.parametrize("model_name", SUPPORTED_MODELS)
def test_artifact_hash_verification(model_name):
    art_path = MODELS_DIR / f"{model_name}.pkl"
    meta_path = MODELS_DIR / f"{model_name}.metadata.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    actual_hash = hashlib.sha256(art_path.read_bytes()).hexdigest()
    assert actual_hash == meta["artifact_sha256"], f"Hash mismatch for {model_name}"


def test_artifact_hash_tamper_detection():
    art_path = MODELS_DIR / "m1_improved_delta_ridge.pkl"
    meta_path = MODELS_DIR / "m1_improved_delta_ridge.metadata.json"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    actual_hash = hashlib.sha256(art_path.read_bytes()).hexdigest()
    corrupted_hash = actual_hash[:-1] + ('0' if actual_hash[-1] != '0' else '1')
    assert actual_hash != corrupted_hash
    with pytest.raises(AssertionError):
        assert corrupted_hash == meta["artifact_sha256"]


# 10. Training reproducibility
def test_training_reproducibility():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from xgboost import XGBRegressor

    inseason_df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")
    inseason_df = inseason_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)
    train_val = inseason_df[inseason_df['Year'] <= 2016].copy()
    test_df = inseason_df[inseason_df['Year'] >= 2017].copy()

    tv_lag1 = train_val['Yield_lag1'].fillna(train_val['Yield_hist_mean']).fillna(train_val['Rice_Yield_kg_ha'].mean()).values
    y_tv_delta = train_val['Rice_Yield_kg_ha'].values - tv_lag1

    # (A) Ridge Determinism
    ct_ridge = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), NUMERICAL_FEATURES),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), CATEGORICAL_FEATURES)
    ])
    pipe_r1 = Pipeline([('preprocessor', ct_ridge), ('model', Ridge(alpha=300.0, random_state=42))])
    pipe_r2 = Pipeline([('preprocessor', ct_ridge), ('model', Ridge(alpha=300.0, random_state=42))])

    pipe_r1.fit(train_val[ALL_FEATURES], y_tv_delta)
    pipe_r2.fit(train_val[ALL_FEATURES], y_tv_delta)

    coef1 = pipe_r1.named_steps['model'].coef_
    coef2 = pipe_r2.named_steps['model'].coef_
    assert np.array_equal(coef1, coef2), "Ridge training coefficients non-deterministic!"

    # (B) XGBoost Conservative Determinism
    ct_xgb = ColumnTransformer([
        ('num', 'passthrough', NUMERICAL_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
    ])
    pipe_x1 = Pipeline([('preprocessor', ct_xgb), ('model', XGBRegressor(
        max_depth=2, learning_rate=0.03, n_estimators=100, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, min_child_weight=5,
        random_state=42, n_jobs=4
    ))])
    pipe_x2 = Pipeline([('preprocessor', ct_xgb), ('model', XGBRegressor(
        max_depth=2, learning_rate=0.03, n_estimators=100, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, min_child_weight=5,
        random_state=42, n_jobs=4
    ))])

    pipe_x1.fit(train_val[ALL_FEATURES], y_tv_delta)
    pipe_x2.fit(train_val[ALL_FEATURES], y_tv_delta)

    pred_x1 = pipe_x1.predict(test_df[ALL_FEATURES])
    pred_x2 = pipe_x2.predict(test_df[ALL_FEATURES])
    assert np.max(np.abs(pred_x1 - pred_x2)) == 0.0, "XGBoost conservative retrain non-deterministic!"


# 11. Inference reproducibility
def test_inference_reproducibility(sample_valid_input):
    predictor = RiceYieldPredictor(model_type='improved_ridge')

    # Single dict
    single_res = predictor.predict(sample_valid_input)["predicted_yield_kg_ha"]

    # Batch DataFrame (10 rows)
    batch_df = pd.DataFrame([sample_valid_input] * 10)
    batch_res = predictor.predict(batch_df)["predicted_yield_kg_ha"].tolist()

    for val in batch_res:
        assert abs(val - single_res) < 1e-6, "Batch and single predictions diverge!"


# 12. Existing production prediction compatibility
def test_existing_production_prediction_compatibility():
    test_df = pd.read_csv(DATA_DIR / "rice_yield_inseason_features_wb.csv")
    test_df = test_df[test_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

    ridge = joblib.load(MODELS_DIR / "m1_improved_delta_ridge.pkl")
    preds_delta = ridge.predict(test_df[ALL_FEATURES])
    preds_total = test_df['Yield_lag1'].values + preds_delta

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
    y_true = test_df['Rice_Yield_kg_ha'].values
    mae = float(mean_absolute_error(y_true, preds_total))
    r2 = float(r2_score(y_true, preds_total))
    rmse = float(np.sqrt(mean_squared_error(y_true, preds_total)))
    mape = float(mean_absolute_percentage_error(y_true, preds_total) * 100.0)

    assert abs(mae - 246.364375) < 0.05, f"Production Ridge MAE changed: {mae}"
    assert abs(r2 - 0.660796) < 0.005, f"Production Ridge R2 changed: {r2}"
    assert abs(rmse - 334.1129) < 0.05, f"Production Ridge RMSE changed: {rmse}"
    assert abs(mape - 9.4847) < 0.05, f"Production Ridge MAPE changed: {mape}"


# 13. Weather-missing policy regression
def test_weather_missing_policy_regression(sample_valid_input):
    # Input missing all in-season weather
    missing_weather_input = {k: v for k, v in sample_valid_input.items() if not k.startswith("InSeason_")}

    # Default policy ('persistence') returns baseline
    p_default = RiceYieldPredictor(weather_missing_policy="persistence")
    res_persist = p_default.predict(missing_weather_input)
    assert res_persist["predicted_yield_kg_ha"] == missing_weather_input["Yield_lag1"]
    assert res_persist["prediction_method"] == "persistence"
    assert res_persist["weather_features_used"] is False

    # Explicit model imputation uses internal imputer
    p_impute = RiceYieldPredictor(weather_missing_policy="model_imputation")
    res_impute = p_impute.predict(missing_weather_input)
    assert res_impute["prediction_method"] == "model_imputation"
    assert res_impute["weather_features_used"] is False
    assert res_impute["predicted_delta_kg_ha"] != 0.0


# 14. Seasonal routing regression
def test_seasonal_routing_regression(sample_valid_input):
    p_routed = RiceYieldPredictor(enable_seasonal_routing=True)

    # Summer season routes to persistence fallback
    summer_input = dict(sample_valid_input)
    summer_input["Season"] = "Summer"
    res_summer = p_routed.predict(summer_input)
    assert res_summer["predicted_delta_kg_ha"] == 0.0
    assert res_summer["predicted_yield_kg_ha"] == summer_input["Yield_lag1"]
    assert any("Seasonal routing active" in w for w in res_summer["warnings"])

    # Autumn and Winter do NOT route to persistence fallback
    winter_input = dict(sample_valid_input)
    winter_input["Season"] = "Winter"
    res_winter = p_routed.predict(winter_input)
    assert not any("Seasonal routing active" in w for w in res_winter["warnings"])
    assert res_winter["predicted_delta_kg_ha"] != 0.0

    autumn_input = dict(sample_valid_input)
    autumn_input["Season"] = "Autumn"
    res_autumn = p_routed.predict(autumn_input)
    assert not any("Seasonal routing active" in w for w in res_autumn["warnings"])
    assert res_autumn["predicted_delta_kg_ha"] != 0.0


# 15. Member 2 integration regression
def test_member2_integration_regression():
    bridge = AgroVisionPredictor()

    y_res = bridge.predict_yield({
        "District": "Hooghly",
        "Season": "Winter",
        "Yield_lag1": 2850.0,
        "area_ha": 300000.0
    })
    assert "predicted_yield_tonnes_per_ha" in y_res
    assert "estimated_production_tonnes" in y_res
    assert 2.0 <= y_res["predicted_yield_tonnes_per_ha"] <= 4.0
    assert y_res["estimated_production_tonnes"] == round(y_res["predicted_yield_tonnes_per_ha"] * 300000.0, 2)
