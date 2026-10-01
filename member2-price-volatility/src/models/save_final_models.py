"""Train and persist commodity-specific final models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (ARTIFACTS_DIR, PRICE_COLUMN, PRICE_FEATURES, PRICE_TARGET,
                    VOLATILITY_CLASSES, VOLATILITY_FEATURES,
                    VOLATILITY_LABEL_MAP, VOLATILITY_TARGET, feature_config_path,
                    get_commodity, price_model_path, train_path,
                    volatility_dataset_train_path, volatility_model_path)
from models.xgboost_price import make_price_model


def _volatility_model():
    return XGBClassifier(
        n_estimators=500, max_depth=3, learning_rate=0.03,
        min_child_weight=3, subsample=0.85, colsample_bytree=0.85,
        objective="multi:softprob", num_class=len(VOLATILITY_CLASSES),
        eval_metric="mlogloss", random_state=42, n_jobs=-1, tree_method="hist",
    )


def _classification_metrics(y_true, y_pred, proba=None):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=VOLATILITY_CLASSES, zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=VOLATILITY_CLASSES,
                                    average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f)}
            for cls, p, r, f in zip(VOLATILITY_CLASSES, precision, recall, f1)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=VOLATILITY_CLASSES
        ).tolist(),
    }
    if proba is not None:
        for cls in ("MEDIUM", "HIGH"):
            index = VOLATILITY_LABEL_MAP[cls]
            result[f"pr_auc_{cls.lower()}"] = float(average_precision_score(
                (y_true == cls).to_numpy(), proba[:, index]
            ))
    return result


def _print_volatility_metrics(name, metrics):
    print(f"\n{name}: accuracy={metrics['accuracy']:.4f}, "
          f"balanced_accuracy={metrics['balanced_accuracy']:.4f}, "
          f"macro_f1={metrics['macro_f1']:.4f}, MCC={metrics['mcc']:.4f}")
    if "pr_auc_medium" in metrics:
        print(f"   PR-AUC MEDIUM={metrics['pr_auc_medium']:.4f}; "
              f"HIGH={metrics['pr_auc_high']:.4f}")
    print(pd.DataFrame(metrics["confusion_matrix"],
          index=[f"Actual {c}" for c in VOLATILITY_CLASSES],
          columns=[f"Predicted {c}" for c in VOLATILITY_CLASSES]).to_string())


def save_models(train_df=None, vol_train_df=None, vol_test_df=None):
    """Fit final models and return test-set volatility metrics when supplied."""
    commodity = get_commodity()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if train_df is None:
        train_df = pd.read_csv(train_path())
    if vol_train_df is None:
        vol_train_df = pd.read_csv(volatility_dataset_train_path())

    required_price = PRICE_FEATURES + [PRICE_TARGET, PRICE_COLUMN]
    missing = [c for c in required_price if c not in train_df.columns]
    if missing:
        raise ValueError(f"Price training data missing columns: {missing}")
    price_df = train_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    y_price = price_df[PRICE_TARGET] - price_df[PRICE_COLUMN]
    price_model = make_price_model()
    price_model.fit(price_df[PRICE_FEATURES], y_price)
    joblib.dump(price_model, price_model_path())
    print(f"Saved Rice price-change model: {price_model_path()} ({len(price_df)} rows; no target clipping)")

    required_vol = VOLATILITY_FEATURES + [VOLATILITY_TARGET]
    missing = [c for c in required_vol if c not in vol_train_df.columns]
    if missing:
        raise ValueError(f"Volatility training data missing columns: {missing}")
    vol_train_df = vol_train_df[vol_train_df[VOLATILITY_TARGET].isin(VOLATILITY_CLASSES)].copy()
    y_vol = vol_train_df[VOLATILITY_TARGET].map(VOLATILITY_LABEL_MAP)
    vol_model = _volatility_model()
    vol_model.fit(vol_train_df[VOLATILITY_FEATURES], y_vol,
                  sample_weight=compute_sample_weight("balanced", y_vol))
    joblib.dump(vol_model, volatility_model_path())
    print(f"Saved Rice volatility model: {volatility_model_path()} ({len(vol_train_df)} rows; balanced classes)")

    volatility_metrics = None
    if vol_test_df is not None:
        vol_test_df = vol_test_df[vol_test_df[VOLATILITY_TARGET].isin(VOLATILITY_CLASSES)].copy()
        y_test = vol_test_df[VOLATILITY_TARGET]
        probability = vol_model.predict_proba(vol_test_df[VOLATILITY_FEATURES])
        predicted = pd.Series(probability.argmax(axis=1)).map(
            {value: key for key, value in VOLATILITY_LABEL_MAP.items()}
        )
        volatility_metrics = _classification_metrics(y_test.reset_index(drop=True), predicted, probability)
        majority = DummyClassifier(strategy="most_frequent", random_state=42)
        majority.fit(vol_train_df[VOLATILITY_FEATURES], vol_train_df[VOLATILITY_TARGET])
        volatility_metrics["majority_baseline"] = _classification_metrics(y_test, majority.predict(vol_test_df[VOLATILITY_FEATURES]))
        stratified = DummyClassifier(strategy="stratified", random_state=42)
        stratified.fit(vol_train_df[VOLATILITY_FEATURES], vol_train_df[VOLATILITY_TARGET])
        volatility_metrics["stratified_baseline"] = _classification_metrics(y_test, stratified.predict(vol_test_df[VOLATILITY_FEATURES]))
        _print_volatility_metrics("Volatility test model", volatility_metrics)
        _print_volatility_metrics("Volatility majority baseline", volatility_metrics["majority_baseline"])

    feature_config = {
        "commodity": commodity,
        "price_model": {
            "type": "XGBRegressor", "target": "future_price_change",
            "target_treatment": "unclipped", "features": PRICE_FEATURES,
            "reconstruction_column": PRICE_COLUMN,
            "note": "Add predicted future_price_change to current modal price.",
        },
        "volatility_model": {
            "type": "XGBClassifier", "target": VOLATILITY_TARGET,
            "classes": VOLATILITY_CLASSES, "features": VOLATILITY_FEATURES,
            "class_balanced": True,
        },
        "volatility_test_metrics": volatility_metrics,
    }
    with open(feature_config_path(), "w", encoding="utf-8") as handle:
        json.dump(feature_config, handle, indent=2)
    print(f"Saved feature configuration: {feature_config_path()}")
    return {"volatility_metrics": volatility_metrics}


if __name__ == "__main__":
    save_models()
