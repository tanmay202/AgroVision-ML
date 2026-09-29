"""
AgroVision — XGBoost Volatility Classifier (v2)

Fixes / improvements over v1:
1. Proper evaluation: macro-F1, balanced accuracy, MCC, per-class recall,
   PR-AUC for MEDIUM/HIGH, confusion matrix, and majority-class baselines.
2. Imbalance handled properly: balanced sample weights + probability
   re-weighting tuned for macro-F1 on a time-ordered validation set
   (never on the test set).
3. objective="multi:softprob" (v1 used softmax, which cannot give
   probabilities, so thresholds could not be tuned).
4. Early stopping on a time-ordered validation split instead of a fixed
   500 trees.
5. No rows lost to NaN features: inf -> NaN, and XGBoost handles NaN natively.
6. Gain-based feature importance (split-count importance is misleading).
7. The validation split is chronological, based on config.DATE_COLUMN
   ("Reported Date"), which is stored in the volatility CSVs as metadata
   only. It is never part of the feature matrix.

Usage:
    python xgboost_volatility.py            # evaluate only
    python xgboost_volatility.py --save     # also save tuned model
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    VOLATILITY_DATASET_TRAIN_PATH,
    VOLATILITY_DATASET_TEST_PATH,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
)

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

N_CLASSES = len(VOLATILITY_CLASSES)
MED_I = VOLATILITY_LABEL_MAP["MEDIUM"]
HIGH_I = VOLATILITY_LABEL_MAP["HIGH"]

GAP_DAYS = 7   # >= max forecast horizon; keeps val/train targets from overlapping
GAP_ROWS = 3   # fallback when no date column is available

# The date column is metadata for splitting only. Guard against it ever
# being added to the feature list by mistake.
assert DATE_COLUMN not in VOLATILITY_FEATURES, (
    f"'{DATE_COLUMN}' must not be a model feature"
)


def banner(text):
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def load(path):
    """Load a volatility dataset. Only rows without a target are dropped."""
    df = pd.read_csv(path)
    missing_cols = [c for c in VOLATILITY_FEATURES if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing feature columns in {path.name}: {missing_cols}")

    df[VOLATILITY_FEATURES] = df[VOLATILITY_FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df[df[VOLATILITY_TARGET].isin(VOLATILITY_CLASSES)].reset_index(drop=True)

    # Parse the date once (metadata only; never used as a feature).
    if DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        n_bad = int(df[DATE_COLUMN].isna().sum())
        if n_bad:
            print(f"   WARNING: {n_bad} rows in {path.name} have an "
                  f"unparseable '{DATE_COLUMN}'")
    return df


def time_ordered_val_split(df, frac):
    """
    Boolean masks (fit, val) with the validation set at the END of time.
    A small gap is purged from the fit set so targets do not leak.
    """
    if DATE_COLUMN in df.columns:
        dates = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
        if dates.notna().all():
            cutoff = dates.quantile(1 - frac)
            val = dates >= cutoff
            fit = dates < cutoff - pd.Timedelta(days=GAP_DAYS)
            if not fit.any() or not val.any():
                raise ValueError(
                    f"Date split produced an empty set (fit={int(fit.sum())}, "
                    f"val={int(val.sum())}); check '{DATE_COLUMN}' values and --val-frac."
                )
            return fit, val, f"date cutoff {cutoff.date()}"
        reason = (f"'{DATE_COLUMN}' has {int(dates.isna().sum())} unparseable "
                  "values")
    else:
        reason = f"'{DATE_COLUMN}' not in volatility dataset"

    print(
        f"   WARNING: {reason} - using the last rows of each group as "
        "validation (assumes chronological order).\n"
        "            Rebuild the volatility CSVs so the date column is kept "
        "for a cleaner split."
    )
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    if group_cols:
        pos = df.groupby(group_cols).cumcount()
        size = df.groupby(group_cols)[VOLATILITY_TARGET].transform("size")
    else:
        pos = pd.Series(np.arange(len(df)), index=df.index)
        size = pd.Series(len(df), index=df.index)
    start = np.floor(size * (1 - frac))
    return pos < start - GAP_ROWS, pos >= start, "tail of each group"


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
def make_model(n_estimators, early_stopping_rounds=None):
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=5.0,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        early_stopping_rounds=early_stopping_rounds,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )


def apply_weights(proba, w):
    return (proba * w).argmax(axis=1)


def make_w(w_med, w_high):
    w = np.ones(N_CLASSES)
    w[MED_I] = w_med
    w[HIGH_I] = w_high
    return w


def tune_prob_weights(y_val_idx, proba_val, min_high_recall=0.0):
    """
    Grid-search multipliers for MEDIUM/HIGH probabilities on the VALIDATION set.

    Objective: maximise macro-F1 SUBJECT TO HIGH recall >= min_high_recall.
    Ties prefer weights closest to 1.

    If no grid point meets the floor, fall back to the point with the highest
    HIGH recall (ties broken by macro-F1) and report feasible=False.
    min_high_recall=0.0 reproduces the old unconstrained macro-F1 tuning.

    Returns (w_med, w_high, macro_f1, high_recall, feasible).
    """
    grid = np.geomspace(0.25, 8, 11)
    labels = list(range(N_CLASSES))
    y_val_idx = np.asarray(y_val_idx)
    best_key, best = None, (1.0, 1.0, -1.0, 0.0, False)
    for w_med in grid:
        for w_high in grid:
            pred = apply_weights(proba_val, make_w(w_med, w_high))
            f1 = f1_score(y_val_idx, pred, labels=labels,
                          average="macro", zero_division=0)
            high_rec = recall_score(y_val_idx, pred, labels=[HIGH_I],
                                    average="macro", zero_division=0)
            feasible = high_rec >= min_high_recall
            closeness = -(abs(np.log(w_med)) + abs(np.log(w_high)))
            if feasible:
                key = (1, round(f1, 6), closeness)
            else:
                key = (0, round(high_rec, 6), round(f1, 6), closeness)
            if best_key is None or key > best_key:
                best_key, best = key, (w_med, w_high, f1, high_rec, feasible)
    return best


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def summarize(name, y_true, y_pred):
    rec = recall_score(y_true, y_pred, labels=VOLATILITY_CLASSES,
                       average=None, zero_division=0)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=VOLATILITY_CLASSES,
                             average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    for cls, r in zip(VOLATILITY_CLASSES, rec):
        row[f"recall_{cls}"] = r
    return row


def to_labels(idx_array):
    return pd.Series(idx_array).map(VOLATILITY_REVERSE_LABEL_MAP)


def date_range_str(df):
    if DATE_COLUMN not in df.columns or df[DATE_COLUMN].isna().all():
        return "n/a"
    return f"{df[DATE_COLUMN].min().date()} -> {df[DATE_COLUMN].max().date()}"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-frac", type=float, default=0.2,
                        help="fraction of the training period (latest) used for validation")
    parser.add_argument("--min-high-recall", type=float, default=0.50,
                        help="HIGH-recall floor (on validation) required when "
                             "tuning probability multipliers for macro-F1")
    parser.add_argument("--save", action="store_true",
                        help="refit on all training data and save model + weights")
    args = parser.parse_args()

    for p in (VOLATILITY_DATASET_TRAIN_PATH, VOLATILITY_DATASET_TEST_PATH):
        if not p.exists():
            print(f"ERROR: Missing {p}")
            return

    train_df = load(VOLATILITY_DATASET_TRAIN_PATH)
    test_df = load(VOLATILITY_DATASET_TEST_PATH)

    print("\nDEBUG TRAIN PATH:", VOLATILITY_DATASET_TRAIN_PATH)
    print("DEBUG TEST PATH :", VOLATILITY_DATASET_TEST_PATH)
    print("\nDEBUG TRAIN COLUMNS:")
    print(train_df.columns.tolist())
    print("\nDEBUG TEST COLUMNS:")
    print(test_df.columns.tolist())

    banner("XGBOOST VOLATILITY CLASSIFIER (v2)")
    print(f"Training rows: {len(train_df)}  ({date_range_str(train_df)})")
    print(f"Testing rows : {len(test_df)}  ({date_range_str(test_df)})")
    print(f"Rows with any NaN feature (kept, XGBoost handles NaN): "
          f"train={int(train_df[VOLATILITY_FEATURES].isna().any(axis=1).sum())}, "
          f"test={int(test_df[VOLATILITY_FEATURES].isna().any(axis=1).sum())}")
    print("\nClass distribution (train / test):")
    print(pd.DataFrame({
        "train": train_df[VOLATILITY_TARGET].value_counts(normalize=True),
        "test": test_df[VOLATILITY_TARGET].value_counts(normalize=True),
    }).reindex(VOLATILITY_CLASSES).round(3))

    # ---- time-ordered fit / validation split inside the training data ----
    fit_mask, val_mask, how = time_ordered_val_split(train_df, args.val_frac)
    fit_df, val_df = train_df[fit_mask], train_df[val_mask]
    purged = len(train_df) - len(fit_df) - len(val_df)
    print(f"\nInner split ({how}): fit={len(fit_df)}, val={len(val_df)}, "
          f"purged gap={purged}")

    # Features only: the date column is never passed to the model.
    X_fit, X_val, X_test = (d[VOLATILITY_FEATURES] for d in (fit_df, val_df, test_df))
    y_fit = fit_df[VOLATILITY_TARGET].map(VOLATILITY_LABEL_MAP)
    y_val = val_df[VOLATILITY_TARGET].map(VOLATILITY_LABEL_MAP)
    y_test_lbl = test_df[VOLATILITY_TARGET].reset_index(drop=True)

    w_fit = compute_sample_weight("balanced", y_fit)
    w_val = compute_sample_weight("balanced", y_val)

    # ---- train with early stopping ----
    banner("TRAINING (early stopping on time-ordered validation)")
    model = make_model(n_estimators=1000, early_stopping_rounds=50)
    model.fit(
        X_fit, y_fit,
        sample_weight=w_fit,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False,
    )
    best_iter = int(model.best_iteration) + 1
    print(f"Best number of trees: {best_iter}")

    # ---- tune MEDIUM/HIGH probability multipliers on validation ----
    proba_val = model.predict_proba(X_val)
    y_val_np = y_val.to_numpy()

    # Reference: old objective (macro-F1 only, no HIGH-recall floor).
    f_med, f_high, f_f1, f_rec, _ = tune_prob_weights(y_val_np, proba_val, 0.0)
    w_vec_f1 = make_w(f_med, f_high)
    print(f"F1-only multipliers      -> MEDIUM x{f_med:.2f}, HIGH x{f_high:.2f} "
          f"(val macro-F1 {f_f1:.3f}, val HIGH recall {f_rec:.3f})")

    # New objective: macro-F1 subject to HIGH recall >= floor (validation only).
    w_med, w_high, val_f1, val_high_rec, feasible = tune_prob_weights(
        y_val_np, proba_val, args.min_high_recall)
    w_vec = make_w(w_med, w_high)
    print(f"Constrained multipliers  -> MEDIUM x{w_med:.2f}, HIGH x{w_high:.2f} "
          f"(val macro-F1 {val_f1:.3f}, val HIGH recall {val_high_rec:.3f}, "
          f"floor {args.min_high_recall:.2f})")
    if not feasible:
        print(f"   WARNING: no multiplier pair reached HIGH recall >= "
              f"{args.min_high_recall:.2f} on validation; using the pair with "
              "the highest HIGH recall instead.")
    print(f"Validation has only {int((y_val == HIGH_I).sum())} HIGH rows, "
          "so treat the tuned multipliers as approximate.")

    # ---- predict on test ----
    proba_test = model.predict_proba(X_test)
    pred_raw = to_labels(proba_test.argmax(axis=1))
    pred_f1only = to_labels(apply_weights(proba_test, w_vec_f1))
    pred_tuned = to_labels(apply_weights(proba_test, w_vec))

    # ---- baselines ----
    majority = DummyClassifier(strategy="most_frequent").fit(X_fit, fit_df[VOLATILITY_TARGET])
    stratified = DummyClassifier(strategy="stratified", random_state=42).fit(X_fit, fit_df[VOLATILITY_TARGET])

    banner("TEST SUMMARY (compare against the baselines!)")
    summary = pd.DataFrame([
        summarize("Always-majority baseline", y_test_lbl, majority.predict(X_test)),
        summarize("Stratified-random baseline", y_test_lbl, stratified.predict(X_test)),
        summarize("XGBoost (argmax)", y_test_lbl, pred_raw),
        summarize("XGBoost (tuned, F1 only)", y_test_lbl, pred_f1only),
        summarize("XGBoost (tuned, F1 + HIGH floor)", y_test_lbl, pred_tuned),
    ]).set_index("model")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(3))

    banner("THRESHOLD-FREE RANKING QUALITY (PR-AUC vs prevalence)")
    for cls in ("MEDIUM", "HIGH"):
        idx = VOLATILITY_LABEL_MAP[cls]
        y_bin = (y_test_lbl == cls).to_numpy()
        ap = average_precision_score(y_bin, proba_test[:, idx])
        print(f"{cls:<7} PR-AUC = {ap:.3f}   (random = {y_bin.mean():.3f})")

    banner("TEST CLASSIFICATION REPORT (tuned, F1 + HIGH floor)")
    print(classification_report(
        y_test_lbl, pred_tuned, labels=VOLATILITY_CLASSES, digits=3, zero_division=0,
    ))

    banner("TEST CONFUSION MATRIX (tuned, F1 + HIGH floor)")
    cm = confusion_matrix(y_test_lbl, pred_tuned, labels=VOLATILITY_CLASSES)
    print(pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in VOLATILITY_CLASSES],
        columns=[f"Predicted {c}" for c in VOLATILITY_CLASSES],
    ))

    banner("TOP 10 FEATURES (gain)")
    gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame({
        "feature": VOLATILITY_FEATURES,
        "gain": [gain.get(f, 0.0) for f in VOLATILITY_FEATURES],
    })
    importance["share"] = importance["gain"] / max(importance["gain"].sum(), 1e-12)
    importance = importance.sort_values("gain", ascending=False)
    print(importance.head(10).to_string(index=False))
    unused = importance.loc[importance["gain"] == 0, "feature"].tolist()
    if unused:
        print(f"\nFeatures never used by the model (candidates to drop): {unused}")

    # ---- optional save ----
    if args.save:
        banner("SAVING")
        full = pd.concat([fit_df, val_df])
        X_full = full[VOLATILITY_FEATURES]
        y_full = full[VOLATILITY_TARGET].map(VOLATILITY_LABEL_MAP)
        final = make_model(n_estimators=best_iter)
        final.fit(X_full, y_full,
                  sample_weight=compute_sample_weight("balanced", y_full))

        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        model_path = ARTIFACT_DIR / "tea_volatility_model_v2.pkl"
        weights_path = ARTIFACT_DIR / "tea_volatility_prob_weights.json"
        joblib.dump(final, model_path)
        weights_path.write_text(json.dumps({
            "classes": VOLATILITY_CLASSES,
            "label_map": VOLATILITY_LABEL_MAP,
            "prob_weights": {"MEDIUM": float(w_med), "HIGH": float(w_high)},
            "objective": f"max macro-F1 s.t. validation HIGH recall >= {args.min_high_recall}",
            "note": "pred = argmax(predict_proba * weights); weights indexed by label_map",
        }, indent=2))
        print(f"Saved: {model_path}")
        print(f"Saved: {weights_path}")
        print("At inference: proba = model.predict_proba(X); "
              "pred = (proba * weights_vector).argmax(1)")


if __name__ == "__main__":
    main()