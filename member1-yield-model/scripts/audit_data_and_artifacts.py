"""
AgroVision Member 1 - Step 5 Data and Artifact Audit
====================================================
Comprehensive audit of datasets, features, coverage, scripts, and model artifacts:
1. data/processed/train.csv
2. data/processed/validation.csv
3. data/processed/test.csv
4. data/processed/rice_yield_inseason_features_wb.csv
5. scripts/train_and_evaluate_improved.py
6. scripts/verify_reproducibility.py
7. scripts/predict_rice_yield.py
8. src/weather_preprocessor.py
9. Current model metadata JSON files in models/
10. Current promotion manifest in models/promotion_manifest.json
"""

import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data/processed"
MODELS_DIR = BASE / "models"
SCRIPTS_DIR = BASE / "scripts"
SRC_DIR = BASE / "src"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

INSEASON_WEATHER_FEATURES = [
    'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
    'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
    'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
]


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(name: str, path: Path) -> dict:
    print(f"\n{'='*30} Auditing Dataset: {name} ({path.name}) {'='*30}")
    assert path.exists(), f"File not found: {path}"
    sha256 = compute_sha256(path)
    df = pd.read_csv(path)
    
    total_rows = len(df)
    columns = list(df.columns)
    
    # Check grain (District, Season, Year)
    has_grain = {'District', 'Season', 'Year'}.issubset(df.columns)
    assert has_grain, f"Missing grain columns in {name}"
    
    key_dupes = df.duplicated(subset=['District', 'Season', 'Year']).sum()
    unique_keys = total_rows - key_dupes
    
    years = sorted(df['Year'].unique().tolist())
    districts = sorted(df['District'].unique().tolist())
    seasons = sorted(df['Season'].unique().tolist())
    
    # Target and lag missingness
    target_col = 'Rice_Yield_kg_ha'
    target_missing = df[target_col].isna().sum() if target_col in df.columns else None
    
    lag1_missing = df['Yield_lag1'].isna().sum() if 'Yield_lag1' in df.columns else None
    lag2_missing = df['Yield_lag2'].isna().sum() if 'Yield_lag2' in df.columns else None
    
    # Weather completeness
    has_inseason_weather = all(c in df.columns for c in INSEASON_WEATHER_FEATURES)
    weather_summary = {}
    if has_inseason_weather:
        weather_df = df[INSEASON_WEATHER_FEATURES]
        null_counts = weather_df.isna().sum(axis=1)
        complete = (null_counts == 0).sum()
        missing = (null_counts == len(INSEASON_WEATHER_FEATURES)).sum()
        incomplete = total_rows - complete - missing
        weather_summary = {
            "all_10_weather_features_complete": int(complete),
            "incomplete_weather": int(incomplete),
            "completely_missing_weather": int(missing),
            "completeness_pct": round(float(complete) / total_rows * 100.0, 2)
        }
    
    info = {
        "file": str(path.relative_to(BASE)).replace("\\", "/"),
        "sha256": sha256,
        "total_rows": total_rows,
        "columns_count": len(columns),
        "columns": columns,
        "grain": "District x Season x Year",
        "duplicate_keys": int(key_dupes),
        "years_count": len(years),
        "years_range": [min(years), max(years)],
        "years": years,
        "districts_count": len(districts),
        "districts": districts,
        "seasons_count": len(seasons),
        "seasons": seasons,
        "target_missing": int(target_missing) if target_missing is not None else None,
        "lag1_missing": int(lag1_missing) if lag1_missing is not None else None,
        "lag2_missing": int(lag2_missing) if lag2_missing is not None else None,
        "weather_summary": weather_summary
    }
    
    print(f"Rows: {total_rows}")
    print(f"Years: {min(years)} - {max(years)} ({len(years)} distinct)")
    print(f"Districts: {len(districts)} distinct")
    print(f"Seasons: {seasons}")
    print(f"Duplicate keys (District, Season, Year): {key_dupes}")
    print(f"Target missing: {target_missing}")
    print(f"Yield_lag1 missing: {lag1_missing}")
    print(f"Yield_lag2 missing: {lag2_missing}")
    if has_inseason_weather:
        print(f"Weather completeness: Complete={weather_summary['all_10_weather_features_complete']}, Incomplete={weather_summary['incomplete_weather']}, Missing={weather_summary['completely_missing_weather']} ({weather_summary['completeness_pct']}%)")
    
    return info


def audit_scripts() -> dict:
    print(f"\n{'='*30} Auditing Core Pipeline Scripts {'='*30}")
    
    # 5. train_and_evaluate_improved.py
    train_script = SCRIPTS_DIR / "train_and_evaluate_improved.py"
    assert train_script.exists(), f"Missing: {train_script}"
    train_sha = compute_sha256(train_script)
    print(f"  [OK] train_and_evaluate_improved.py ({train_sha[:16]}...)")
    
    # 6. verify_reproducibility.py
    verify_script = SCRIPTS_DIR / "verify_reproducibility.py"
    assert verify_script.exists(), f"Missing: {verify_script}"
    verify_sha = compute_sha256(verify_script)
    print(f"  [OK] verify_reproducibility.py ({verify_sha[:16]}...)")
    
    # 7. predict_rice_yield.py
    predict_script = SCRIPTS_DIR / "predict_rice_yield.py"
    assert predict_script.exists(), f"Missing: {predict_script}"
    predict_sha = compute_sha256(predict_script)
    print(f"  [OK] predict_rice_yield.py ({predict_sha[:16]}...)")
    
    # 8. src/weather_preprocessor.py
    weather_script = SRC_DIR / "weather_preprocessor.py"
    assert weather_script.exists(), f"Missing: {weather_script}"
    weather_sha = compute_sha256(weather_script)
    print(f"  [OK] src/weather_preprocessor.py ({weather_sha[:16]}...)")
    
    # Introspect schemas from predict_rice_yield & weather_preprocessor
    from predict_rice_yield import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
    from weather_preprocessor import INSEASON_FEATURE_COLUMNS, SEASON_CONFIG
    
    assert len(NUMERICAL_FEATURES) == 18, f"Expected 18 numerical features, got {len(NUMERICAL_FEATURES)}"
    assert len(CATEGORICAL_FEATURES) == 2, f"Expected 2 categorical features, got {len(CATEGORICAL_FEATURES)}"
    assert len(ALL_FEATURES) == 20, f"Expected 20 total features, got {len(ALL_FEATURES)}"
    assert len(INSEASON_FEATURE_COLUMNS) == 10, f"Expected 10 in-season weather features, got {len(INSEASON_FEATURE_COLUMNS)}"
    assert set(INSEASON_FEATURE_COLUMNS) == set(INSEASON_WEATHER_FEATURES), "In-season weather features mismatch"
    
    expected_cutoffs = {
        'Autumn': (7, 31),
        'Winter': (9, 30),
        'Summer': (2, 28)
    }
    for s, (m, d) in expected_cutoffs.items():
        actual_m = SEASON_CONFIG[s]['cutoff_month']
        actual_d = SEASON_CONFIG[s]['cutoff_day']
        assert (actual_m, actual_d) == (m, d), f"Cutoff mismatch for {s}: {(actual_m, actual_d)} vs {(m, d)}"
    print(f"  [OK] Verified 20 production features (18 numeric, 2 categorical) and agronomic cutoffs {expected_cutoffs}")
    
    return {
        "train_and_evaluate_improved": {
            "path": str(train_script.relative_to(BASE)).replace("\\", "/"),
            "sha256": train_sha
        },
        "verify_reproducibility": {
            "path": str(verify_script.relative_to(BASE)).replace("\\", "/"),
            "sha256": verify_sha
        },
        "predict_rice_yield": {
            "path": str(predict_script.relative_to(BASE)).replace("\\", "/"),
            "sha256": predict_sha,
            "numerical_features_count": len(NUMERICAL_FEATURES),
            "categorical_features_count": len(CATEGORICAL_FEATURES),
            "total_features_count": len(ALL_FEATURES),
            "features": ALL_FEATURES
        },
        "weather_preprocessor": {
            "path": str(weather_script.relative_to(BASE)).replace("\\", "/"),
            "sha256": weather_sha,
            "inseason_features_count": len(INSEASON_FEATURE_COLUMNS),
            "features": INSEASON_FEATURE_COLUMNS,
            "cutoffs": {k: f"{v['cutoff_month']:02d}-{v['cutoff_day']:02d}" for k, v in SEASON_CONFIG.items()}
        }
    }


def audit_artifacts() -> dict:
    print(f"\n{'='*30} Auditing Model Artifacts and Manifest {'='*30}")
    manifest_path = MODELS_DIR / "promotion_manifest.json"
    assert manifest_path.exists(), f"Manifest missing: {manifest_path}"
    manifest_sha = compute_sha256(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    print(f"Promotion Manifest: {manifest_path.name} (SHA: {manifest_sha[:16]}...)")
    print(f"Promotion Timestamp: {manifest.get('promotion_timestamp')}")
    print(f"Approved Production Artifact: {manifest.get('approved_production_artifact')}")
    
    promoted = manifest.get("promoted_artifacts", {})
    results = {}
    for name, details in promoted.items():
        expected_sha = details["artifact_sha256"]
        pkl_path = MODELS_DIR / f"{name}.pkl"
        meta_path = MODELS_DIR / f"{name}.metadata.json"
        
        assert pkl_path.exists(), f"Model artifact missing: {pkl_path}"
        assert meta_path.exists(), f"Metadata missing: {meta_path}"
        
        actual_sha = compute_sha256(pkl_path)
        assert actual_sha == expected_sha, f"SHA-256 mismatch for {pkl_path.name}: {actual_sha} vs {expected_sha}"
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        assert meta.get("artifact_sha256") == actual_sha, f"Metadata SHA mismatch for {name}"
        
        print(f"  [OK] {pkl_path.name}: SHA matches ({actual_sha[:16]}...)")
        results[name] = {
            "artifact_file": pkl_path.name,
            "sha256": actual_sha,
            "metadata_file": meta_path.name,
            "architecture": meta.get("architecture"),
            "training_period": meta.get("training_period"),
            "training_rows": meta.get("training_rows"),
            "hyperparameters": meta.get("hyperparameters"),
            "hash_matches_manifest": True
        }
    
    return {
        "manifest_path": str(manifest_path.relative_to(BASE)).replace("\\", "/"),
        "manifest_sha256": manifest_sha,
        "approved_production_artifact": manifest.get("approved_production_artifact"),
        "promoted_artifacts": results
    }


def main():
    datasets = {
        "train": DATA_DIR / "train.csv",
        "validation": DATA_DIR / "validation.csv",
        "test": DATA_DIR / "test.csv",
        "inseason_features": DATA_DIR / "rice_yield_inseason_features_wb.csv"
    }
    
    dataset_results = {}
    for name, path in datasets.items():
        dataset_results[name] = audit_dataset(name, path)
        
    scripts_results = audit_scripts()
    artifact_results = audit_artifacts()
    
    summary = {
        "audit_version": "Step 5 Comprehensive Audit",
        "items_audited": [
            "data/processed/train.csv",
            "data/processed/validation.csv",
            "data/processed/test.csv",
            "data/processed/rice_yield_inseason_features_wb.csv",
            "scripts/train_and_evaluate_improved.py",
            "scripts/verify_reproducibility.py",
            "scripts/predict_rice_yield.py",
            "src/weather_preprocessor.py",
            "models/*.metadata.json",
            "models/promotion_manifest.json"
        ],
        "datasets": dataset_results,
        "scripts": scripts_results,
        "artifacts": artifact_results
    }
    
    out_path = BASE / "reports/rolling_origin/data_and_artifact_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAudit complete! Saved audit record across all 10 items to {out_path}")


if __name__ == "__main__":
    main()
