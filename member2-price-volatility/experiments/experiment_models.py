"""
AgroVision Member 2 — Model Trainers and Threshold Tuning

Implements:
- XGBoost classifier with configurable weighting (none, balanced, sqrt)
- Random Forest classifier with configurable weighting
- Validation-based probability threshold tuning (macro F1 and constrained HIGH recall)
- Baseline predictors
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import VOLATILITY_CLASSES, VOLATILITY_LABEL_MAP, VOLATILITY_REVERSE_LABEL_MAP


def get_sample_weights(y, strategy="none"):
    """
    Computes sample weights:
    - 'none': uniform weights (all 1.0)
    - 'balanced': standard inverse frequency
    - 'sqrt': square root of inverse frequency (mild damping)
    """
    if strategy == "balanced":
        return compute_sample_weight("balanced", y)
    elif strategy == "sqrt":
        bw = compute_sample_weight("balanced", y)
        return np.sqrt(bw)
    else:
        return np.ones(len(y), dtype=float)


def train_xgboost(X_train, y_train, weight_strategy="none", n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42):
    """Fits an XGBoost classifier with multi:softprob."""
    weights = get_sample_weights(y_train, weight_strategy)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=len(VOLATILITY_CLASSES),
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train, sample_weight=weights)
    return model


def train_random_forest(X_train, y_train, class_weight=None, n_estimators=100, max_depth=12, random_state=42):
    """Fits a Random Forest classifier."""
    cw = "balanced" if class_weight == "balanced" else None
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=10,
        class_weight=cw,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def tune_decision_thresholds(y_val, proba_val, min_high_recall=0.0):
    """
    Grid searches probability multipliers for [LOW=1.0, MEDIUM=w_med, HIGH=w_high]
    on the VALIDATION set to maximize macro F1, optionally subject to HIGH recall floor.

    Returns:
    (best_weights, best_val_f1, best_val_high_rec)
    """
    y_val = np.asarray(y_val)
    grid = np.linspace(0.8, 3.5, 12)
    high_idx = VOLATILITY_LABEL_MAP["HIGH"]

    best_f1 = -1.0
    best_weights = np.array([1.0, 1.0, 1.0])
    best_high_rec = 0.0

    # Fallback tracker if min_high_recall is not strictly satisfied
    fallback_weights = np.array([1.0, 1.0, 1.0])
    fallback_high_rec = -1.0

    for w_med in grid:
        for w_high in grid:
            w = np.array([1.0, w_med, w_high])
            preds = (proba_val * w).argmax(axis=1)
            rec_high = recall_score(y_val, preds, labels=[high_idx], average="macro", zero_division=0)
            f1 = f1_score(y_val, preds, average="macro", zero_division=0)

            if rec_high > fallback_high_rec:
                fallback_high_rec = rec_high
                fallback_weights = w

            if rec_high >= min_high_recall:
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = w
                    best_high_rec = rec_high

    if best_f1 < 0.0:
        # None satisfied floor, return fallback
        best_weights = fallback_weights
        best_high_rec = fallback_high_rec
        preds = (proba_val * best_weights).argmax(axis=1)
        best_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

    return best_weights, best_f1, best_high_rec


def apply_decision_thresholds(proba, weights):
    """Applies class multipliers to predicted probabilities and returns class labels."""
    preds_num = (proba * weights).argmax(axis=1)
    return [VOLATILITY_REVERSE_LABEL_MAP[p] for p in preds_num]
