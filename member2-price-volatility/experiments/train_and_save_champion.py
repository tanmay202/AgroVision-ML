"""
AgroVision Member 2 — Train and Persist Champion Normalized Volatility Model

Trains the walk-forward selected champion model (XGBoost Balanced)
on the pre-holdout training data (< 2023-01-27) with boundary temporal purging.
Saves versioned artifacts to member2-price-volatility/artifacts/:
  - rice_volatility_normalized_model.pkl
  - rice_volatility_normalized_config.json

Includes full provenance metadata, feature order, training thresholds,
and holdout benchmark performance.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2

EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rice_features_final.csv"

sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import extract_volatility_features, EXPERIMENTAL_FEATURE_COLS
from target_definitions import (
    compute_forward_volatility_targets,
    compute_persistence_predictions,
    apply_temporal_purge,
    EXP_B_LOW_THRESH,
    EXP_B_HIGH_THRESH,
)
from experiment_models import train_xgboost
from metrics import evaluate_predictions
from config import (
    GROUP_COLUMNS,
    PRICE_COLUMN,
    DATE_COLUMN,
    VOLATILITY_CLASSES,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
)


def train_and_persist_champion():
    print("=" * 70)
    print("TRAINING CHAMPION PRICE-NORMALIZED VOLATILITY MODEL")
    print("=" * 70)

    # 1. Load and prepare dataset
    df = pd.read_csv(DATA_PATH)
    df = compute_forward_volatility_targets(df, GROUP_COLUMNS, DATE_COLUMN, PRICE_COLUMN, window=5)
    df = compute_persistence_predictions(df, GROUP_COLUMNS, PRICE_COLUMN)
    df = extract_volatility_features(df, GROUP_COLUMNS, DATE_COLUMN, PRICE_COLUMN)

    # 2. Strict temporal split and purging
    holdout_start = "2023-01-27"
    holdout_end = "2024-01-27"

    clean_train_mask, is_holdout, purged = apply_temporal_purge(df, DATE_COLUMN, holdout_start)
    train_df = df[clean_train_mask].dropna(subset=EXPERIMENTAL_FEATURE_COLS + ["target_B"]).copy()
    holdout_df = df[is_holdout & (df[DATE_COLUMN] <= pd.Timestamp(holdout_end))].dropna(
        subset=EXPERIMENTAL_FEATURE_COLS + ["target_B", "persistence_pred_B"]
    ).copy()

    print(f"Training observations: {len(train_df):,} (Purged {purged:,} horizon-leak rows)")
    print(f"Holdout observations:  {len(holdout_df):,}")

    # 3. Train Champion Model: XGBoost with Balanced Sample Weights
    y_train = train_df["target_B"].map(VOLATILITY_LABEL_MAP)
    y_holdout_true = holdout_df["target_B"]

    print("Fitting XGBoost (Balanced sample weights)...")
    model = train_xgboost(train_df[EXPERIMENTAL_FEATURE_COLS], y_train, weight_strategy="balanced")

    # 4. Evaluate on Holdout
    holdout_proba = model.predict_proba(holdout_df[EXPERIMENTAL_FEATURE_COLS])
    preds_num = holdout_proba.argmax(axis=1)
    holdout_preds = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in preds_num]

    ml_metrics = evaluate_predictions(y_holdout_true, holdout_preds, "Champion XGBoost Balanced", "Holdout (2023-2024)")
    base_metrics = evaluate_predictions(y_holdout_true, holdout_df["persistence_pred_B"], "Persistence Baseline", "Holdout (2023-2024)")

    # 5. Statistical Significance Test (McNemar)
    corr_ml = (pd.Series(holdout_preds).to_numpy() == y_holdout_true.to_numpy())
    corr_base = (holdout_df["persistence_pred_B"].to_numpy() == y_holdout_true.to_numpy())
    b = int((corr_ml & ~corr_base).sum())
    c = int((~corr_ml & corr_base).sum())
    mcnemar_chi2 = float(((abs(b - c) - 1)**2) / (b + c))
    mcnemar_p = float(1 - chi2.cdf(mcnemar_chi2, df=1))

    print(f"\nHoldout Accuracy:         ML {ml_metrics['accuracy']:.4f} vs Base {base_metrics['accuracy']:.4f}")
    print(f"Holdout Macro F1:         ML {ml_metrics['macro_f1']:.4f} vs Base {base_metrics['macro_f1']:.4f}")
    print(f"Holdout Balanced Acc:     ML {ml_metrics['balanced_accuracy']:.4f} vs Base {base_metrics['balanced_accuracy']:.4f}")
    print(f"Holdout HIGH Recall:      ML {ml_metrics['high_recall']:.4f} vs Base {base_metrics['high_recall']:.4f}")
    print(f"McNemar Chi2:             {mcnemar_chi2:.4f} (p-value: {mcnemar_p:.4e})")

    # 6. Save Model Artifact (.pkl)
    artifact_model_path = ARTIFACTS_DIR / "rice_volatility_normalized_model.pkl"
    model_artifact = {
        "model": model,
        "feature_names": EXPERIMENTAL_FEATURE_COLS,
        "target": "future_pct_range_class",
        "volatility_version": "v2_normalized",
        "commodity": "rice",
        "classes": VOLATILITY_CLASSES,
        "label_map": VOLATILITY_LABEL_MAP,
        "reverse_label_map": VOLATILITY_REVERSE_LABEL_MAP,
        "thresholds": {
            "low_medium_threshold": EXP_B_LOW_THRESH,
            "medium_high_threshold": EXP_B_HIGH_THRESH,
            "method": "zero-aware training split (56% zeros -> LOW, non-zeros split at median -> MEDIUM/HIGH)",
        },
        "target_definition": {
            "name": "future_pct_range",
            "formula": "(max(price[t+1..t+5]) - min(price[t+1..t+5])) / price[t] * 100",
            "window": 5,
        },
        "training_metadata": {
            "train_observations": len(train_df),
            "train_period": [str(train_df[DATE_COLUMN].min().date()), str(train_df[DATE_COLUMN].max().date())],
            "weight_strategy": "balanced",
            "n_features": len(EXPERIMENTAL_FEATURE_COLS),
        },
        "holdout_benchmark": {
            "period": ["2023-01-27", "2024-01-27"],
            "observations": len(holdout_df),
            "ml_accuracy": ml_metrics["accuracy"],
            "ml_macro_f1": ml_metrics["macro_f1"],
            "ml_balanced_acc": ml_metrics["balanced_accuracy"],
            "ml_high_recall": ml_metrics["high_recall"],
            "baseline_accuracy": base_metrics["accuracy"],
            "baseline_macro_f1": base_metrics["macro_f1"],
            "mcnemar_test": {"chi2": mcnemar_chi2, "p_value": mcnemar_p},
        },
    }
    joblib.dump(model_artifact, artifact_model_path)
    print(f"\nSaved model artifact to: {artifact_model_path}")

    # 7. Save Config Artifact (.json)
    artifact_config_path = ARTIFACTS_DIR / "rice_volatility_normalized_config.json"
    config_data = {
        "commodity": "rice",
        "volatility_version": "v2_normalized",
        "target_name": "future_pct_range_class",
        "volatility_definition": "percentage range of next 5 observations: (max - min) / price_t * 100",
        "volatility_window": 5,
        "classes": VOLATILITY_CLASSES,
        "thresholds": {
            "low_medium_threshold": EXP_B_LOW_THRESH,
            "medium_high_threshold": EXP_B_HIGH_THRESH,
            "derivation": "derived strictly from training set (<= 0.014286% is flat/LOW, non-zero median 1.960784% splits MEDIUM/HIGH)",
        },
        "features": EXPERIMENTAL_FEATURE_COLS,
        "feature_count": len(EXPERIMENTAL_FEATURE_COLS),
        "model_architecture": {
            "type": "XGBClassifier",
            "objective": "multi:softprob",
            "weight_strategy": "balanced",
            "n_estimators": 250,
            "max_depth": 4,
            "learning_rate": 0.05,
        },
        "walk_forward_validation_summary": {
            "mean_macro_f1": 0.6066,
            "mean_balanced_accuracy": 0.6147,
            "mean_accuracy": 0.6746,
            "mean_high_recall": 0.4680,
            "baseline_mean_macro_f1": 0.5832,
            "baseline_mean_balanced_acc": 0.5833,
        },
        "retrospective_holdout_summary": {
            "accuracy": ml_metrics["accuracy"],
            "macro_f1": ml_metrics["macro_f1"],
            "balanced_accuracy": ml_metrics["balanced_accuracy"],
            "high_recall": ml_metrics["high_recall"],
            "baseline_accuracy": base_metrics["accuracy"],
            "baseline_macro_f1": base_metrics["macro_f1"],
            "mcnemar_significance": {
                "chi2": round(mcnemar_chi2, 4),
                "p_value": mcnemar_p,
                "statistically_significant": mcnemar_p < 0.001,
            },
        },
    }
    with open(artifact_config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved configuration to:  {artifact_config_path}")

    return artifact_model_path, artifact_config_path


if __name__ == "__main__":
    train_and_persist_champion()
