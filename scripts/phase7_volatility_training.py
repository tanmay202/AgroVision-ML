"""
Phase 7 & 8: Advanced Volatility Classification and Calibration
Compares:
1. Legacy V1 target (absolute rupee range, fixed thresholds [10, 30])
2. V2 target (percentage price range, training-derived quantile thresholds)
Evaluates:
- HistGradientBoostingClassifier, XGBoostClassifier, CatBoostClassifier
- Class weighting / sample weighting
- Platt scaling (Sigmoid) vs Isotonic calibration
- Metrics: Accuracy, Balanced Accuracy, Macro F1, Per-class F1, MCC, HIGH PR-AUC, Multi-class Brier Score, ECE.
"""

import sys
sys.path.insert(0, "member2-price-volatility/src")

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc, brier_score_loss, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from features_v2 import generate_candidate_price_features

def compute_ece(y_true, probs, n_bins=10):
    """Computes Expected Calibration Error on top-1 confidence."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(acc_in_bin - avg_conf_in_bin) * prop_in_bin
    return ece

def multiclass_brier_score(y_true, probs, num_classes=3):
    """Computes multiclass Brier score."""
    y_onehot = np.zeros((len(y_true), num_classes))
    for i in range(num_classes):
        y_onehot[y_true == i, i] = 1.0
    return float(np.mean(np.sum((y_onehot - probs) ** 2, axis=1)))

def train_and_eval_volatility(commodity="wheat"):
    print(f"\n=======================================================")
    print(f"=== PHASE 7 & 8 VOLATILITY CLASSIFICATION: {commodity.upper()} ===")
    print(f"=======================================================")
    
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    
    print(f"Generating features for {len(clean_df):,} rows...")
    feats_df = generate_candidate_price_features(clean_df)
    
    group_cols = ["Market Name", "Variety"]
    date_col = "Reported Date"
    price_col = "Modal Price (Rs./Quintal)"
    
    # Add extra volatility-specific ratio features
    feats_df["rolling_range_7"] = feats_df["rolling_max_7"] - feats_df["rolling_min_7"]
    feats_df["rolling_pct_range_7"] = (feats_df["rolling_range_7"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_7"] = (feats_df["rolling_std_7"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_14"] = (feats_df["rolling_std_14"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_30"] = (feats_df["rolling_std_30"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["price_to_rolling_mean_30"] = feats_df[price_col] / (feats_df["rolling_mean_30"] + 1e-4)
    
    feats_df[date_col] = pd.to_datetime(feats_df[date_col])
    feats_df = feats_df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    
    # 30-history requirement
    hist_mask = feats_df.groupby(group_cols).cumcount() >= 30
    df = feats_df[hist_mask].copy().reset_index(drop=True)
    
    # Compute 5-observation future max and min for volatility targets
    grp_p = df.groupby(group_cols)[price_col]
    future_p = [grp_p.shift(-i) for i in range(1, 6)]
    future_df = pd.concat(future_p, axis=1)
    
    # Require all 5 observations
    valid_mask = future_df.notna().all(axis=1)
    df = df[valid_mask].copy().reset_index(drop=True)
    future_df = future_df[valid_mask].reset_index(drop=True)
    
    f_max = future_df.max(axis=1)
    f_min = future_df.min(axis=1)
    df["future_abs_range"] = f_max - f_min
    df["future_pct_range"] = (df["future_abs_range"] / df[price_col]) * 100.0
    
    # Split dates
    train_orig = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    split_date = pd.to_datetime(train_orig[date_col]).max()
    print(f"Historical split date: {split_date.date()}")
    
    train_mask = df[date_col] <= split_date
    test_mask = df[date_col] > split_date
    
    train_data = df[train_mask].copy().reset_index(drop=True)
    test_data = df[test_mask].copy().reset_index(drop=True)
    
    print(f"Full Train rows: {len(train_data):,}, Holdout Test rows: {len(test_data):,}")
    
    # Derive V2 thresholds from training set non-zero medians / quantiles
    # Low: 0 to T_low, Medium: T_low to T_high, High: > T_high
    pct_train = train_data["future_pct_range"].values
    if commodity == "wheat":
        t_low, t_high = 0.05, 1.818
    elif commodity == "potato":
        t_low, t_high = 0.05, 5.556
    else:
        nz = pct_train[pct_train > 0.05]
        t_low = 0.05
        t_high = float(np.median(nz))
    
    print(f"V2 Percentage Thresholds: Low/Med = {t_low:.3f}%, Med/High = {t_high:.3f}%")
    
    # Create Targets
    # 0 = LOW, 1 = MEDIUM, 2 = HIGH
    def assign_classes(s, low_th, high_th):
        c = np.zeros(len(s), dtype=int)
        c[(s > low_th) & (s <= high_th)] = 1
        c[s > high_th] = 2
        return c
    
    # V2 Targets
    train_data["target_v2"] = assign_classes(train_data["future_pct_range"].values, t_low, t_high)
    test_data["target_v2"] = assign_classes(test_data["future_pct_range"].values, t_low, t_high)
    
    # Legacy V1 Targets (10, 30 Rs)
    train_data["target_v1"] = assign_classes(train_data["future_abs_range"].values, 10.0, 30.0)
    test_data["target_v1"] = assign_classes(test_data["future_abs_range"].values, 10.0, 30.0)
    
    print("\nV2 Class Distribution Train:")
    print(pd.Series(train_data["target_v2"]).value_counts(normalize=True).sort_index().to_dict())
    print("V2 Class Distribution Test:")
    print(pd.Series(test_data["target_v2"]).value_counts(normalize=True).sort_index().to_dict())
    
    # Candidate feature columns
    drop_cols = [
        "State Name", "District Name", "Market Name", "Variety", "Group",
        date_col, price_col, "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)",
        "future_abs_range", "future_pct_range", "target_v2", "target_v1"
    ]
    cand_feats = [c for c in train_data.columns if c not in drop_cols]
    print(f"Number of candidate features: {len(cand_feats)}")
    
    # Chronological validation split for tuning & calibration:
    # Use last 20% of training dates as validation fold
    val_split_date = train_data[date_col].quantile(0.80)
    tr_inner = train_data[train_data[date_col] <= val_split_date].copy().reset_index(drop=True)
    val_inner = train_data[train_data[date_col] > val_split_date].copy().reset_index(drop=True)
    print(f"Inner Train rows: {len(tr_inner):,}, Inner Val rows: {len(val_inner):,}")
    
    # Class weights for inner train
    counts = np.bincount(tr_inner["target_v2"], minlength=3)
    weights = len(tr_inner) / (3.0 * np.maximum(counts, 1))
    class_weights_dict = {i: float(weights[i]) for i in range(3)}
    print(f"Class weights: {class_weights_dict}")
    
    sample_weights_tr = np.array([class_weights_dict[c] for c in tr_inner["target_v2"]])
    
    # ------------------------------------------------------------
    # Models to evaluate
    # ------------------------------------------------------------
    models = {
        "HistGB_balanced": HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=250, max_depth=6, random_state=42
        ),
        "XGBoost_weighted": XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=3, random_state=42, n_jobs=-1
        ),
        "CatBoost_weighted": CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.06,
            loss_function="MultiClass", class_weights=list(weights),
            verbose=0, random_seed=42
        ),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} on Inner Train...")
        if "XGBoost" in name:
            model.fit(tr_inner[cand_feats], tr_inner["target_v2"], sample_weight=sample_weights_tr)
        else:
            model.fit(tr_inner[cand_feats], tr_inner["target_v2"])
            
        # Validation performance (uncalibrated)
        val_probs = model.predict_proba(val_inner[cand_feats])
        val_preds = np.argmax(val_probs, axis=1)
        val_f1 = f1_score(val_inner["target_v2"], val_preds, average="macro")
        val_ece = compute_ece(val_inner["target_v2"].values, val_probs)
        print(f"  Validation (Raw) - Macro F1: {val_f1:.4f}, ECE: {val_ece:.4f}")
        
        # Test calibration methods on Validation
        cal_sigmoid = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
        cal_sigmoid.fit(val_inner[cand_feats], val_inner["target_v2"])
        
        cal_isotonic = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
        cal_isotonic.fit(val_inner[cand_feats], val_inner["target_v2"])
        
        sig_probs = cal_sigmoid.predict_proba(val_inner[cand_feats])
        sig_ece = compute_ece(val_inner["target_v2"].values, sig_probs)
        
        iso_probs = cal_isotonic.predict_proba(val_inner[cand_feats])
        iso_ece = compute_ece(val_inner["target_v2"].values, iso_probs)
        
        print(f"  Calibration Val ECE: Raw={val_ece:.4f}, Sigmoid={sig_ece:.4f}, Isotonic={iso_ece:.4f}")
        
        # Select best calibrator for holdout
        best_cal = "sigmoid" if sig_ece < iso_ece else "isotonic"
        chosen_calibrator = cal_sigmoid if best_cal == "sigmoid" else cal_isotonic
        
        # Evaluate on Holdout Test
        test_y = test_data["target_v2"].values
        test_probs_raw = model.predict_proba(test_data[cand_feats])
        test_probs_cal = chosen_calibrator.predict_proba(test_data[cand_feats])
        
        test_preds = np.argmax(test_probs_cal, axis=1)
        
        acc = accuracy_score(test_y, test_preds)
        bal_acc = balanced_accuracy_score(test_y, test_preds)
        macro_f1 = f1_score(test_y, test_preds, average="macro")
        per_class_f1 = f1_score(test_y, test_preds, average=None)
        mcc = matthews_corrcoef(test_y, test_preds)
        
        # PR-AUC for HIGH volatility (class 2)
        y_high = (test_y == 2).astype(int)
        p_high = test_probs_cal[:, 2]
        precision, recall, _ = precision_recall_curve(y_high, p_high)
        high_pr_auc = auc(recall, precision)
        
        brier = multiclass_brier_score(test_y, test_probs_cal)
        ece = compute_ece(test_y, test_probs_cal)
        raw_ece = compute_ece(test_y, test_probs_raw)
        
        print(f"  --- Holdout Results for {name} ({best_cal} calibrated) ---")
        print(f"  Accuracy:          {acc*100:.2f}%")
        print(f"  Balanced Accuracy: {bal_acc*100:.2f}%")
        print(f"  Macro F1:          {macro_f1:.4f}")
        print(f"  Per-class F1:      LOW={per_class_f1[0]:.4f}, MED={per_class_f1[1]:.4f}, HIGH={per_class_f1[2]:.4f}")
        print(f"  MCC:               {mcc:.4f}")
        print(f"  HIGH PR-AUC:       {high_pr_auc:.4f}")
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  ECE:               {ece:.4f} (Raw: {raw_ece:.4f})")
        
        results[name] = {
            "model": model,
            "calibrator": chosen_calibrator,
            "calibrator_type": best_cal,
            "features": cand_feats,
            "thresholds": {"low_medium": t_low, "medium_high": t_high},
            "metrics": {
                "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
                "per_class_f1": per_class_f1.tolist(), "mcc": mcc, "high_pr_auc": high_pr_auc,
                "brier_score": brier, "ece": ece, "raw_ece": raw_ece
            }
        }
        
    return results

if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "wheat"
    train_and_eval_volatility(c)
