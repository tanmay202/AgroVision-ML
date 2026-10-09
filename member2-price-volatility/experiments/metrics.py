"""
AgroVision Member 2 — Volatility Comprehensive Evaluation Metrics

Computes:
- Overall accuracy
- Balanced accuracy
- Macro F1
- Matthews Correlation Coefficient (MCC)
- Per-class precision, recall, and F1
- Specific HIGH-class recall and precision
- Multi-class confusion matrix
- Class counts and distribution
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import VOLATILITY_CLASSES, VOLATILITY_LABEL_MAP


def evaluate_predictions(y_true, y_pred, model_name="Model", period_name="Evaluation"):
    """
    Computes all standard classification metrics on matching series.
    Returns a comprehensive metrics dictionary.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    n_obs = len(y_true)
    class_counts = y_true.value_counts().to_dict()
    pred_counts = y_pred.value_counts().to_dict()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=VOLATILITY_CLASSES, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=VOLATILITY_CLASSES, zero_division=0
    )

    per_class = {}
    for i, c in enumerate(VOLATILITY_CLASSES):
        per_class[c] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    high_recall = per_class.get("HIGH", {}).get("recall", 0.0)
    high_precision = per_class.get("HIGH", {}).get("precision", 0.0)
    high_f1 = per_class.get("HIGH", {}).get("f1", 0.0)

    med_recall = per_class.get("MEDIUM", {}).get("recall", 0.0)
    med_precision = per_class.get("MEDIUM", {}).get("precision", 0.0)
    med_f1 = per_class.get("MEDIUM", {}).get("f1", 0.0)

    low_recall = per_class.get("LOW", {}).get("recall", 0.0)
    low_precision = per_class.get("LOW", {}).get("precision", 0.0)
    low_f1 = per_class.get("LOW", {}).get("f1", 0.0)

    cm = confusion_matrix(y_true, y_pred, labels=VOLATILITY_CLASSES).tolist()

    return {
        "model": model_name,
        "period": period_name,
        "n_samples": n_obs,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mcc": round(mcc, 4),
        "high_recall": round(high_recall, 4),
        "high_precision": round(high_precision, 4),
        "high_f1": round(high_f1, 4),
        "medium_recall": round(med_recall, 4),
        "medium_precision": round(med_precision, 4),
        "medium_f1": round(med_f1, 4),
        "low_recall": round(low_recall, 4),
        "low_precision": round(low_precision, 4),
        "low_f1": round(low_f1, 4),
        "class_distribution": class_counts,
        "pred_distribution": pred_counts,
        "confusion_matrix": cm,
        "per_class_details": per_class,
    }


def format_confusion_matrix(cm, labels=VOLATILITY_CLASSES):
    """Returns a pandas DataFrame representation of confusion matrix."""
    return pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in labels],
        columns=[f"Pred {c}" for c in labels]
    )
