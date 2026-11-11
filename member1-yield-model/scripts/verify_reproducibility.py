"""
AgroVision Member 1 Model Reproducibility & Serialization Verification
======================================================================
Executes Experiments 1 to 4 to verify reproducibility, numerical stability,
and absence of InconsistentVersionWarning.
"""

import sys
import json
import hashlib
import warnings
import subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
STAGING_DIR = BASE / "models/staging"
PRODUCTION_DIR = BASE / "models"


def run_experiment_1_fresh_loading(models_dir: Path) -> dict:
    """
    Experiment 1: Fresh artifact loading in fresh Python process.
    Verifies:
    - Loading succeeds
    - Zero InconsistentVersionWarning
    - Required features are recognized
    - Output shape is correct
    - Predictions contain only valid numerical values
    """
    print("\n--- Running Experiment 1: Fresh Artifact Loading ---")
    results = {}

    test_df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    test_df = test_df[test_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

    artifacts = [
        "m1_improved_delta_ridge.pkl",
        "m1_improved_delta_xgboost_conservative.pkl",
        "m1_improved_delta_xgboost.pkl",
        "m1_best_ridge.pkl",
        "m1_inseason_xgboost.pkl"
    ]

    for art in artifacts:
        art_path = models_dir / art
        meta_path = models_dir / f"{art_path.stem}.metadata.json"

        assert art_path.exists(), f"Artifact missing: {art_path}"
        assert meta_path.exists(), f"Metadata missing: {meta_path}"

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        features = meta["training_features"]

        # Run loading and prediction in isolated subprocess to verify zero warnings
        code = f"""
import sys, warnings, json, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.exceptions import InconsistentVersionWarning

art_path = Path(r'{str(art_path)}')
meta_path = Path(r'{str(meta_path)}')

with open(meta_path, 'r', encoding='utf-8') as f:
    meta = json.load(f)

features = meta['training_features']
test_df = pd.read_csv(Path(r'{str(BASE / "data/processed/rice_yield_inseason_features_wb.csv")}'))
test_df = test_df[test_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

# For baseline ridge test.csv is used if num_ridge_orig
if 'Yield_hist_mean' in features and 'InSeason_Precip_Cum' not in features:
    test_df = pd.read_csv(Path(r'{str(BASE / "data/processed/test.csv")}'))
    test_df = test_df.sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

X = test_df[features]

with warnings.catch_warnings(record=True) as recorded_warnings:
    warnings.simplefilter('always')
    model = joblib.load(art_path)
    inconsistent_warnings = [w for w in recorded_warnings if issubclass(w.category, InconsistentVersionWarning)]
    if len(inconsistent_warnings) > 0:
        print(f"FAIL_WARNING: {{len(inconsistent_warnings)}} InconsistentVersionWarning raised")
        sys.exit(2)

preds = model.predict(X)
if not isinstance(preds, np.ndarray):
    print("FAIL_TYPE: Output is not numpy array")
    sys.exit(3)
if len(preds) != len(X):
    print(f"FAIL_SHAPE: Expected {{len(X)}}, got {{len(preds)}}")
    sys.exit(4)
if np.isnan(preds).any() or np.isinf(preds).any():
    print("FAIL_NUMERICAL: Predictions contain NaN or Inf")
    sys.exit(5)

print(f"SUCCESS: {{len(preds)}} predictions valid")
"""
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert proc.returncode == 0, f"Loading failed for {art}: {proc.stderr}\n{proc.stdout}"
        print(f"  [PASS] {art}: Clean load, 0 InconsistentVersionWarning, {len(test_df)} valid predictions")
        results[art] = "PASS"

    return results


def run_experiment_2_training_reproducibility() -> dict:
    """
    Experiment 2: Repeated training reproducibility.
    Verifies that retraining under the identical certified environment produces
    identical model weights / estimators.
    """
    print("\n--- Running Experiment 2: Repeated Training Reproducibility ---")
    cmd = [sys.executable, str(BASE / "scripts/train_and_evaluate_improved.py"), "--output-dir", str(BASE / "models/staging_repeat")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Retraining failed: {proc.stderr}"

    repeat_dir = BASE / "models/staging_repeat"

    # Compare hashes of Ridge
    # Ridge with same seed and data is bitwise deterministic
    staging_ridge_hash = hashlib.sha256((STAGING_DIR / "m1_improved_delta_ridge.pkl").read_bytes()).hexdigest()
    repeat_ridge_hash = hashlib.sha256((repeat_dir / "m1_improved_delta_ridge.pkl").read_bytes()).hexdigest()

    # Compare predictions of both Ridge models
    r1 = joblib.load(STAGING_DIR / "m1_improved_delta_ridge.pkl")
    r2 = joblib.load(repeat_dir / "m1_improved_delta_ridge.pkl")

    test_df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    test_df = test_df[test_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

    with open(STAGING_DIR / "m1_improved_delta_ridge.metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    p1 = r1.predict(test_df[meta["training_features"]])
    p2 = r2.predict(test_df[meta["training_features"]])

    max_diff_ridge = np.max(np.abs(p1 - p2))
    assert max_diff_ridge == 0.0, f"Ridge prediction non-deterministic! Max diff: {max_diff_ridge}"
    print(f"  [PASS] Ridge training reproducibility: max prediction difference = {max_diff_ridge:.1e} (bitwise: {staging_ridge_hash == repeat_ridge_hash})")

    # Compare XGBoost conservative predictions
    x1 = joblib.load(STAGING_DIR / "m1_improved_delta_xgboost_conservative.pkl")
    x2 = joblib.load(repeat_dir / "m1_improved_delta_xgboost_conservative.pkl")
    px1 = x1.predict(test_df[meta["training_features"]])
    px2 = x2.predict(test_df[meta["training_features"]])
    max_diff_xgb = np.max(np.abs(px1 - px2))
    assert max_diff_xgb == 0.0, f"XGBoost conservative non-deterministic! Max diff: {max_diff_xgb}"
    print(f"  [PASS] XGBoost conservative reproducibility: max prediction difference = {max_diff_xgb:.1e}")

    # Clean up staging_repeat
    import shutil
    shutil.rmtree(repeat_dir, ignore_errors=True)

    return {"ridge_diff": max_diff_ridge, "xgb_diff": max_diff_xgb}


def run_experiment_3_prediction_reproducibility(models_dir: Path) -> dict:
    """
    Experiment 3: Prediction reproducibility.
    Verifies that identical input records evaluated repeatedly produce identical outputs.
    """
    print("\n--- Running Experiment 3: Prediction Reproducibility ---")
    sys.path.insert(0, str(BASE / "scripts"))
    from predict_rice_yield import RiceYieldPredictor, ALL_FEATURES

    sample_input = {
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

    # Load staging model
    ridge_path = models_dir / "m1_improved_delta_ridge.pkl"
    predictor = RiceYieldPredictor(model_path=ridge_path)

    run1 = predictor.predict(sample_input)
    run2 = predictor.predict(sample_input)
    run3 = predictor.predict(sample_input)

    assert run1["predicted_yield_kg_ha"] == run2["predicted_yield_kg_ha"] == run3["predicted_yield_kg_ha"]
    assert run1["predicted_delta_kg_ha"] == run2["predicted_delta_kg_ha"] == run3["predicted_delta_kg_ha"]
    print(f"  [PASS] Single record idempotency: predicted_yield={run1['predicted_yield_kg_ha']:.2f} kg/ha (repeated runs identical)")

    # Test batch DataFrame idempotency
    df_in = pd.DataFrame([sample_input] * 5)
    b1 = predictor.predict(df_in)["predicted_yield_kg_ha"].values
    b2 = predictor.predict(df_in)["predicted_yield_kg_ha"].values
    assert np.all(b1 == b2)
    print(f"  [PASS] Batch DataFrame idempotency: 5 repeated records produce identical predictions")

    return {"single_identical": True, "batch_identical": True}


def run_experiment_4_feature_schema_verification(models_dir: Path) -> dict:
    """
    Experiment 4: Feature schema verification.
    Compares inference feature names and order against newly generated training artifact metadata.
    """
    print("\n--- Running Experiment 4: Feature Schema Verification ---")
    sys.path.insert(0, str(BASE / "scripts"))
    from predict_rice_yield import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

    meta_path = models_dir / "m1_improved_delta_ridge.metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    meta_training_features = meta["training_features"]
    meta_num_features = meta["numerical_features"]
    meta_cat_features = meta["categorical_features"]

    assert ALL_FEATURES == meta_training_features, f"Feature mismatch! ALL_FEATURES != metadata features"
    assert NUMERICAL_FEATURES == meta_num_features, f"Numerical features mismatch!"
    assert CATEGORICAL_FEATURES == meta_cat_features, f"Categorical features mismatch!"
    print(f"  [PASS] Exact feature match: 20 features ({len(NUMERICAL_FEATURES)} numerical, {len(CATEGORICAL_FEATURES)} categorical) in identical order")

    # Verify SHA256 integrity
    art_path = models_dir / "m1_improved_delta_ridge.pkl"
    actual_hash = hashlib.sha256(art_path.read_bytes()).hexdigest()
    assert actual_hash == meta["artifact_sha256"], f"Artifact hash mismatch: {actual_hash} != {meta['artifact_sha256']}"
    print(f"  [PASS] Artifact SHA-256 matches metadata: {actual_hash}")

    return {"schema_matches": True, "hash_matches": True}


def main():
    print("=" * 80)
    print("MEMBER 1 MODEL REPRODUCIBILITY EXPERIMENTS (STAGING ARTIFACTS)")
    print("=" * 80)

    res1 = run_experiment_1_fresh_loading(STAGING_DIR)
    res2 = run_experiment_2_training_reproducibility()
    res3 = run_experiment_3_prediction_reproducibility(STAGING_DIR)
    res4 = run_experiment_4_feature_schema_verification(STAGING_DIR)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
