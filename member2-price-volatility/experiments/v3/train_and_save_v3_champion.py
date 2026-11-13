"""
AgroVision Member 2 — Train and Persist V3 Champion Volatility Models

Trains and persists V3 experimental champion models:
1. rice_volatility_v3_7d_model.pkl & rice_volatility_v3_7d_config.json
2. rice_volatility_v3_14d_model.pkl & rice_volatility_v3_14d_config.json
3. Calibrated probability models (Isotonic & Sigmoid)
All artifacts are saved strictly into member2-price-volatility/artifacts/v3/.
Zero modifications to production artifacts.
"""

import sys
import os
import json
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight

V3_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = V3_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rice_features_final.csv"
V3_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "v3"

sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from volatility_v3_target_definitions import (
    compute_v3_calendar_targets,
    apply_v3_temporal_purge,
    VOLATILITY_CLASSES,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
    V3_7D_THRESHOLDS,
    V3_14D_THRESHOLDS,
)
from volatility_v3_features import extract_v3_volatility_features, V3_FEATURE_COLS
from v3_evaluation_utils import (
    compute_comprehensive_metrics,
    compute_multiclass_brier_and_ece,
)


def train_and_persist_v3_champions():
    V3_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAINING & PERSISTING V3 EXPERIMENTAL CHAMPION MODELS (7D & 14D)")
    print("=" * 80)

    # 1. Load data
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = compute_v3_calendar_targets(
        df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        horizons=[7, 14],
        min_future_obs=1,
    )
    df = extract_v3_volatility_features(
        df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
    )

    holdout_start = "2023-01-27"
    holdout_end = "2024-01-27"
    val_cal_start = "2021-01-27"
    val_cal_end = "2023-01-26"

    for H in [7, 14]:
        h_str = f"{H}d"
        target_col = f"target_{h_str}"
        target_date_col = f"future_target_date_{h_str}"
        persist_col = f"persistence_pred_{h_str}"
        thresh_dict = V3_7D_THRESHOLDS if H == 7 else V3_14D_THRESHOLDS

        print(f"\n--- Training Champion for Horizon {H}D ---")

        # 1. Full training set (< 2023-01-27 with boundary purging)
        clean_tr_mask, is_ho_mask, purged = apply_v3_temporal_purge(
            df, "Reported Date", target_date_col, holdout_start
        )
        train_df = df[clean_tr_mask].dropna(subset=V3_FEATURE_COLS + [target_col]).copy()
        holdout_df = df[is_ho_mask & (df["Reported Date"] <= pd.Timestamp(holdout_end))].dropna(
            subset=V3_FEATURE_COLS + [target_col, persist_col]
        ).copy()

        print(f"Train samples: {len(train_df):,} (Purged {purged:,} boundary-leak rows)")
        print(f"Holdout samples: {len(holdout_df):,}")

        X_tr = train_df[V3_FEATURE_COLS]
        y_tr_num = train_df[target_col].map(VOLATILITY_LABEL_MAP)
        X_ho = holdout_df[V3_FEATURE_COLS]
        y_ho_lbl = holdout_df[target_col]

        # 2. Fit base XGBoost Balanced
        weights = compute_sample_weight("balanced", y_tr_num)
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss"
        )
        model.fit(X_tr, y_tr_num, sample_weight=weights)

        # 3. Fit Calibrator using pre-2021 train and 2021-2023 validation
        clean_cal_tr, is_cal_val, purged_cal = apply_v3_temporal_purge(
            df, "Reported Date", target_date_col, val_cal_start
        )
        cal_train_df = df[clean_cal_tr].dropna(subset=V3_FEATURE_COLS + [target_col]).copy()
        cal_val_df = df[is_cal_val & (df["Reported Date"] <= pd.Timestamp(val_cal_end))].dropna(
            subset=V3_FEATURE_COLS + [target_col]
        ).copy()

        w_cal = compute_sample_weight("balanced", cal_train_df[target_col].map(VOLATILITY_LABEL_MAP))
        base_cal_model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss"
        )
        base_cal_model.fit(cal_train_df[V3_FEATURE_COLS], cal_train_df[target_col].map(VOLATILITY_LABEL_MAP), sample_weight=w_cal)

        calibrator_iso = CalibratedClassifierCV(base_cal_model, method="isotonic", cv="prefit")
        calibrator_iso.fit(cal_val_df[V3_FEATURE_COLS], cal_val_df[target_col].map(VOLATILITY_LABEL_MAP))

        calibrator_sig = CalibratedClassifierCV(base_cal_model, method="sigmoid", cv="prefit")
        calibrator_sig.fit(cal_val_df[V3_FEATURE_COLS], cal_val_df[target_col].map(VOLATILITY_LABEL_MAP))

        # 4. Evaluate on Holdout
        raw_proba = model.predict_proba(X_ho)
        iso_proba = calibrator_iso.predict_proba(X_ho)

        preds_raw = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in raw_proba.argmax(axis=1)]
        preds_iso = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in iso_proba.argmax(axis=1)]

        m_raw = compute_comprehensive_metrics(y_ho_lbl, preds_raw, f"V3 XGBoost ({H}D Raw)", "Holdout", f"{H}D")
        m_iso = compute_comprehensive_metrics(y_ho_lbl, preds_iso, f"V3 XGBoost ({H}D Calibrated)", "Holdout", f"{H}D")
        m_base = compute_comprehensive_metrics(y_ho_lbl, holdout_df[persist_col], f"Calendar Persistence ({H}D)", "Holdout", f"{H}D")

        cal_raw_eval = compute_multiclass_brier_and_ece(y_ho_lbl, raw_proba)
        cal_iso_eval = compute_multiclass_brier_and_ece(y_ho_lbl, iso_proba)

        print(f"Holdout Macro F1: ML Raw={m_raw['macro_f1']:.4f} | ML Iso={m_iso['macro_f1']:.4f} | Base={m_base['macro_f1']:.4f}")
        print(f"Holdout HIGH Rec: ML Raw={m_raw['high_recall']:.4f} | ML Iso={m_iso['high_recall']:.4f} | Base={m_base['high_recall']:.4f}")
        print(f"Holdout ECE:      ML Raw={cal_raw_eval['macro_ece']:.4f} | ML Iso={cal_iso_eval['macro_ece']:.4f}")

        # 5. Persist Model Artifacts (.pkl)
        model_artifact = {
            "model": model,
            "calibrator_isotonic": calibrator_iso,
            "calibrator_sigmoid": calibrator_sig,
            "feature_names": V3_FEATURE_COLS,
            "target": f"future_pct_range_{h_str}",
            "horizon_days": H,
            "volatility_version": f"v3_{h_str}",
            "commodity": "rice",
            "classes": VOLATILITY_CLASSES,
            "label_map": VOLATILITY_LABEL_MAP,
            "reverse_label_map": VOLATILITY_REVERSE_LABEL_MAP,
            "thresholds": thresh_dict,
            "training_metadata": {
                "train_observations": len(train_df),
                "purged_boundary_rows": purged,
                "train_period": [str(train_df["Reported Date"].min().date()), str(train_df["Reported Date"].max().date())],
                "weight_strategy": "balanced",
                "features_count": len(V3_FEATURE_COLS),
            },
            "holdout_benchmark": {
                "period": ["2023-01-27", "2024-01-27"],
                "observations": len(holdout_df),
                "ml_accuracy": m_raw["accuracy"],
                "ml_balanced_acc": m_raw["balanced_accuracy"],
                "ml_macro_f1": m_raw["macro_f1"],
                "ml_high_recall": m_raw["high_recall"],
                "ml_high_precision": m_raw["high_precision"],
                "ml_high_f1": m_raw["high_f1"],
                "ml_brier_score": cal_raw_eval["multiclass_brier_score"],
                "ml_macro_ece": cal_raw_eval["macro_ece"],
                "calibrated_macro_ece": cal_iso_eval["macro_ece"],
                "base_macro_f1": m_base["macro_f1"],
                "base_high_recall": m_base["high_recall"],
            },
        }

        artifact_pkl_path = V3_ARTIFACTS_DIR / f"rice_volatility_v3_{h_str}_model.pkl"
        joblib.dump(model_artifact, artifact_pkl_path, compress=3)
        print(f"Saved model artifact: {artifact_pkl_path}")

        # 6. Persist Config Artifact (.json)
        config_artifact = {
            "commodity": "rice",
            "volatility_version": f"v3_{h_str}",
            "horizon_days": H,
            "classes": VOLATILITY_CLASSES,
            "features": V3_FEATURE_COLS,
            "thresholds": thresh_dict,
            "training_metadata": model_artifact["training_metadata"],
            "holdout_benchmark": model_artifact["holdout_benchmark"],
        }
        config_json_path = V3_ARTIFACTS_DIR / f"rice_volatility_v3_{h_str}_config.json"
        with open(config_json_path, "w") as f:
            json.dump(config_artifact, f, indent=2)
        print(f"Saved config artifact: {config_json_path}")

    print("\nAll V3 artifacts successfully saved under member2-price-volatility/artifacts/v3/")


if __name__ == "__main__":
    train_and_persist_v3_champions()
