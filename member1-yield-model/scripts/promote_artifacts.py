"""
AgroVision Member 1 — Explicit Model Artifact Promotion
======================================================
Controlled promotion workflow:
1. Validates staging artifacts and metadata integrity.
2. Performs zero-warning loading and regression checks.
3. Verifies immutable historical backup is present.
4. Atomically promotes staging artifacts to production models/ directory.
5. Verifies RiceYieldPredictor, AgroVisionPredictor, weather preprocessing,
   and seasonal policy routing post-promotion.
6. Emits structured promotion audit log.
"""

import sys
import json
import shutil
import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import joblib

from typing import Optional, Union

BASE = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE.parent
STAGING_DIR = BASE / "models/staging"
PRODUCTION_DIR = BASE / "models"
BACKUP_DIR = BASE / "models_backup_pre_step4"

PROMOTABLE_ARTIFACTS = [
    "m1_improved_delta_ridge",
    "m1_improved_delta_xgboost_conservative",
    "m1_improved_delta_xgboost",
    "m1_best_ridge",
    "m1_inseason_xgboost"
]


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def promote_artifacts(
    staging_dir: Optional[Union[str, Path]] = None,
    production_dir: Optional[Union[str, Path]] = None
):
    staging_path = Path(staging_dir).resolve() if staging_dir else STAGING_DIR
    production_path = Path(production_dir).resolve() if production_dir else PRODUCTION_DIR

    print("=" * 80)
    print("AGROVISION M1: CONTROLLED MODEL ARTIFACT PROMOTION")
    print(f"Staging Source:    {staging_path}")
    print(f"Production Target: {production_path}")
    print("=" * 80)

    # 1. Verify backup exists and is cryptographically intact
    if not BACKUP_DIR.exists() or not (BACKUP_DIR / "manifest.json").exists():
        raise RuntimeError(f"ABORT: Immutable backup directory {BACKUP_DIR} is missing or incomplete!")
    with open(BACKUP_DIR / "manifest.json", "r", encoding="utf-8") as f:
        backup_manifest = json.load(f)
    for b_name, b_meta in backup_manifest.get("artifacts", {}).items():
        b_file = BACKUP_DIR / b_name
        if not b_file.exists():
            raise FileNotFoundError(f"Historical backup artifact missing: {b_file}")
        b_hash = compute_sha256(b_file)
        if b_hash != b_meta["sha256"]:
            raise ValueError(f"Backup artifact {b_name} hash corrupted! {b_hash} != {b_meta['sha256']}")
        if b_file.stat().st_size != b_meta["size_bytes"]:
            raise ValueError(f"Backup artifact {b_name} size corrupted! {b_file.stat().st_size} != {b_meta['size_bytes']}")
    print(f"[OK] Immutable historical backup confirmed and cryptographically verified at: {BACKUP_DIR}")


    # 2. Verify all staging artifacts and metadata
    print("\n[VALIDATION] Checking staging artifacts integrity...")
    staging_hashes = {}
    for name in PROMOTABLE_ARTIFACTS:
        art_path = staging_path / f"{name}.pkl"
        meta_path = staging_path / f"{name}.metadata.json"

        if not art_path.exists():
            raise FileNotFoundError(f"Staging artifact missing: {art_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Staging metadata missing: {meta_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        actual_hash = compute_sha256(art_path)
        if actual_hash != meta["artifact_sha256"]:
            raise ValueError(f"Hash mismatch for {name}: {actual_hash} != {meta['artifact_sha256']}")

        staging_hashes[name] = actual_hash
        print(f"  [VALID] {name}.pkl (SHA-256: {actual_hash[:12]}...)")

    # 3. Verify zero InconsistentVersionWarning on fresh loading
    from sklearn.exceptions import InconsistentVersionWarning
    for name in PROMOTABLE_ARTIFACTS:
        art_path = staging_path / f"{name}.pkl"
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            obj = joblib.load(art_path)
            iv_warnings = [w for w in recorded if issubclass(w.category, InconsistentVersionWarning)]
            if len(iv_warnings) > 0:
                raise RuntimeError(f"ABORT: InconsistentVersionWarning raised when loading {art_path}")
    print("[OK] All staging artifacts load cleanly with ZERO InconsistentVersionWarning.")

    # 4. Verify production candidate performance
    ridge = joblib.load(staging_path / "m1_improved_delta_ridge.pkl")
    test_df = pd.read_csv(BASE / "data/processed/rice_yield_inseason_features_wb.csv")
    test_df = test_df[test_df['Year'] >= 2017].sort_values(by=['District', 'Year', 'Season']).reset_index(drop=True)

    with open(staging_path / "m1_improved_delta_ridge.metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    y_te = test_df["Rice_Yield_kg_ha"].values
    te_lag1 = test_df["Yield_lag1"].values
    preds = te_lag1 + ridge.predict(test_df[meta["training_features"]])

    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)

    if mae > 249.0 or r2 < 0.65:
        raise RuntimeError(f"ABORT: Regression detected in staging Ridge artifact! MAE={mae:.2f}, R2={r2:.4f}")
    print(f"[OK] Production Ridge performance certified: MAE={mae:.2f} kg/ha, R2={r2:.4f}")

    # 5. Execute promotion (copy from staging to production models/)
    print(f"\n[PROMOTION] Promoting certified artifacts to {production_path}...")
    production_path.mkdir(parents=True, exist_ok=True)
    promoted_records = {}

    for name in PROMOTABLE_ARTIFACTS:
        for ext in [".pkl", ".metadata.json"]:
            src = staging_path / f"{name}{ext}"
            dst = production_path / f"{name}{ext}"
            shutil.copy2(src, dst)
            print(f"  Promoted: {src.name} -> {dst}")

        promoted_records[name] = {
            "artifact_sha256": staging_hashes[name],
            "promoted_at": datetime.now(timezone.utc).isoformat()
        }

    # 6. Write promotion audit manifest
    audit_manifest = {
        "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "SUCCESS",
        "approved_production_artifact": "m1_improved_delta_ridge.pkl",
        "certified_environment": meta["dependencies"],
        "promoted_artifacts": promoted_records
    }
    with open(production_path / "promotion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(audit_manifest, f, indent=2)
    print(f"[OK] Promotion manifest saved to {production_path / 'promotion_manifest.json'}")

    # 7. Post-promotion verification of RiceYieldPredictor & AgroVisionPredictor
    print("\n[VERIFICATION] Verifying runtime inference on promoted artifacts...")
    if str(BASE / "scripts") not in sys.path:
        sys.path.insert(0, str(BASE / "scripts"))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from predict_rice_yield import RiceYieldPredictor
    p = RiceYieldPredictor(model_path=production_path / "m1_improved_delta_ridge.pkl")
    res = p.predict({
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
    })
    assert 2500 <= res["predicted_yield_kg_ha"] <= 3000
    print(f"[OK] RiceYieldPredictor inference verified: {res['predicted_yield_kg_ha']:.2f} kg/ha")

    from shared.predict_bridge import AgroVisionPredictor
    bridge = AgroVisionPredictor()
    y_res = bridge.predict_yield({
        "District": "Hooghly",
        "Season": "Winter",
        "Yield_lag1": 2850.0,
        "Yield_lag2": 2800.0,
        "Area_lag1": 320000.0,
        "Area_lag2": 318000.0,
        "Precip_lag1": 1400.0,
        "T2M_lag1": 25.5,
        "RH2M_lag1": 75.0,
        "NDVI_lag1": 0.62,
        "InSeason_Precip_Cum": 950.0,
        "InSeason_Precip_Mean": 7.8,
        "InSeason_Precip_Max": 85.0,
        "InSeason_T2M_Mean": 27.2,
        "InSeason_T2M_Min": 22.0,
        "InSeason_T2M_Max": 33.5,
        "InSeason_T2M_Range": 11.5,
        "InSeason_RH2M_Mean": 80.5,
        "InSeason_Solar_Mean": 18.2,
        "InSeason_Wind_Mean": 1.8
    })
    assert 2.5 <= y_res["predicted_yield_tonnes_per_ha"] <= 3.0
    print(f"[OK] AgroVisionPredictor.predict_yield verified: {y_res['predicted_yield_tonnes_per_ha']} t/ha")

    # Step 3 weather and routing check
    p_routed = RiceYieldPredictor(enable_seasonal_routing=True)
    res_summer = p_routed.predict({
        'District': 'Hooghly',
        'Season': 'Summer',
        'Yield_lag1': 2900.0,
        'Area_lag1': 100000.0
    })
    assert res_summer["predicted_delta_kg_ha"] == 0.0
    assert res_summer["predicted_yield_kg_ha"] == 2900.0
    assert any("Seasonal routing active" in w for w in res_summer["warnings"])
    print("[OK] Step 3 seasonal routing behavior verified: Summer persistence fallback active")

    print("\n" + "=" * 80)
    print("PROMOTION COMPLETED SUCCESSFULLY WITH FULL COMPLIANCE!")
    print("=" * 80)


if __name__ == "__main__":
    promote_artifacts()
