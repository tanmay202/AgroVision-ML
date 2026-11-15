"""
Master Production Script: Phase 9, 10, 11
Trains and saves final production-grade V2 models and metadata for Wheat and Potato.
Evaluates:
  - Overall Holdout
  - Known (Seen) Mandi-Variety groups
  - Previously Unseen Mandi-Variety groups
  - High-Volume Mandis
Generates:
  - Price Champion Model (Short-term next-valid observation)
  - Price 14-day Direct Horizon Model
  - Price 30-day Direct Horizon Model
  - Calibrated Volatility Champion Model
  - Empirical Residual Distributions (p05, p10, p25, p50, p75, p90, p95)
  - Feature Configs, Volatility Configs, Metrics JSON, and Manifest with SHA-256 checksums.
"""

import sys
sys.path.insert(0, "member2-price-volatility/src")

import os
import json
import hashlib
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc
)
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier

from features_v2 import generate_candidate_price_features
from horizon_targets import compute_single_horizon_targets, apply_horizon_temporal_purge


def compute_sha256(file_path):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_ece(y_true, probs, n_bins=10):
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
    return float(ece)


def multiclass_brier_score(y_true, probs, num_classes=3):
    y_onehot = np.zeros((len(y_true), num_classes))
    for i in range(num_classes):
        y_onehot[y_true == i, i] = 1.0
    return float(np.mean(np.sum((y_onehot - probs) ** 2, axis=1)))


def eval_price_subset(act_p, pred_p, cur_p):
    if len(act_p) == 0:
        return {}
    mae = float(mean_absolute_error(act_p, pred_p))
    rmse = float(root_mean_squared_error(act_p, pred_p))
    r2 = float(r2_score(act_p, pred_p)) if len(act_p) > 1 and np.var(act_p) > 0 else 0.0
    mape = float(np.mean(np.abs((act_p - pred_p) / np.maximum(act_p, 1e-4))) * 100.0)
    
    p_mae = float(mean_absolute_error(act_p, cur_p))
    p_rmse = float(root_mean_squared_error(act_p, cur_p))
    p_r2 = float(r2_score(act_p, cur_p)) if len(act_p) > 1 and np.var(act_p) > 0 else 0.0
    p_mape = float(np.mean(np.abs((act_p - cur_p) / np.maximum(act_p, 1e-4))) * 100.0)
    
    # Movement-only
    move_mask = np.abs(act_p - cur_p) > 1e-4
    if np.sum(move_mask) > 0:
        m_mae = float(mean_absolute_error(act_p[move_mask], pred_p[move_mask]))
        m_rmse = float(root_mean_squared_error(act_p[move_mask], pred_p[move_mask]))
        p_m_mae = float(mean_absolute_error(act_p[move_mask], cur_p[move_mask]))
        p_m_rmse = float(root_mean_squared_error(act_p[move_mask], cur_p[move_mask]))
    else:
        m_mae, m_rmse, p_m_mae, p_m_rmse = 0.0, 0.0, 0.0, 0.0
        
    pct_within_25 = float(np.mean(np.abs(act_p - pred_p) <= 25.0) * 100.0)
    p_pct_within_25 = float(np.mean(np.abs(act_p - cur_p) <= 25.0) * 100.0)
    
    return {
        "n_samples": int(len(act_p)),
        "mae": mae, "rmse": rmse, "r2": r2, "mape": mape,
        "persistence_mae": p_mae, "persistence_rmse": p_rmse, "persistence_r2": p_r2, "persistence_mape": p_mape,
        "movement_mae": m_mae, "movement_rmse": m_rmse,
        "persistence_movement_mae": p_m_mae, "persistence_movement_rmse": p_m_rmse,
        "pct_within_25": pct_within_25, "persistence_pct_within_25": p_pct_within_25,
    }


def eval_vol_subset(y_true, probs):
    if len(y_true) == 0:
        return {}
    preds = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    per_class_f1 = [float(x) for x in f1_score(y_true, preds, average=None, zero_division=0)]
    mcc = float(matthews_corrcoef(y_true, preds))
    
    # PR-AUC for HIGH (class 2)
    y_high = (y_true == 2).astype(int)
    if np.sum(y_high) > 0 and len(np.unique(y_high)) > 1:
        precision, recall, _ = precision_recall_curve(y_high, probs[:, 2])
        high_pr_auc = float(auc(recall, precision))
    else:
        high_pr_auc = 0.0
        
    brier = multiclass_brier_score(y_true, probs)
    ece = compute_ece(y_true, probs)
    
    return {
        "n_samples": int(len(y_true)),
        "accuracy": acc, "balanced_accuracy": bal_acc, "macro_f1": macro_f1,
        "per_class_f1": per_class_f1, "mcc": mcc, "high_pr_auc": high_pr_auc,
        "brier_score": brier, "ece": ece
    }


def train_and_export_v2(commodity="wheat"):
    print(f"\n=================================================================")
    print(f"=== FULL PRODUCTION PIPELINE: {commodity.upper()} V2 ARTIFACT GENERATION ===")
    print(f"=================================================================")
    
    out_dir = Path(f"member2-price-volatility/artifacts/{commodity}/v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    
    print(f"1. Generating candidate features for {len(clean_df):,} rows...")
    feats_df = generate_candidate_price_features(clean_df)
    
    group_cols = ["Market Name", "Variety"]
    date_col = "Reported Date"
    price_col = "Modal Price (Rs./Quintal)"
    
    # Add extra volatility features
    feats_df["rolling_range_7"] = feats_df["rolling_max_7"] - feats_df["rolling_min_7"]
    feats_df["rolling_pct_range_7"] = (feats_df["rolling_range_7"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_7"] = (feats_df["rolling_std_7"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_14"] = (feats_df["rolling_std_14"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["rolling_pct_std_30"] = (feats_df["rolling_std_30"] / (feats_df[price_col] + 1e-4)) * 100.0
    feats_df["price_to_rolling_mean_30"] = feats_df[price_col] / (feats_df["rolling_mean_30"] + 1e-4)
    
    feats_df[date_col] = pd.to_datetime(feats_df[date_col])
    feats_df = feats_df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    
    # Require 30 history
    hist_mask = feats_df.groupby(group_cols).cumcount() >= 30
    df = feats_df[hist_mask].copy().reset_index(drop=True)
    
    # Compute Short-term Target: next valid observation within 7 days
    grp_p = df.groupby(group_cols)[price_col]
    grp_d = df.groupby(group_cols)[date_col]
    
    next_p = grp_p.shift(-1)
    next_d = grp_d.shift(-1)
    delta_days = (next_d - df[date_col]).dt.days
    
    valid_st = next_p.notna() & (delta_days <= 7) & (delta_days >= 1)
    df_st = df[valid_st].copy().reset_index(drop=True)
    df_st["future_modal_price"] = next_p[valid_st].values
    df_st["future_price_change"] = df_st["future_modal_price"] - df_st[price_col]
    
    # Volatility Target: 5-observation percentage range
    future_p = [grp_p.shift(-i) for i in range(1, 6)]
    future_df = pd.concat(future_p, axis=1)
    valid_vol = future_df.notna().all(axis=1)
    
    df_vol = df[valid_vol].copy().reset_index(drop=True)
    f_max = future_df[valid_vol].max(axis=1).values
    f_min = future_df[valid_vol].min(axis=1).values
    df_vol["future_abs_range"] = f_max - f_min
    df_vol["future_pct_range"] = ((f_max - f_min) / df_vol[price_col].values) * 100.0
    
    # Historical Split Date
    train_orig = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    split_date = pd.to_datetime(train_orig[date_col]).max()
    print(f"Historical split date: {split_date.date()}")
    
    # Candidate features
    drop_cols = [
        "State Name", "District Name", "Market Name", "Variety", "Group",
        date_col, price_col, "Min Price (Rs./Quintal)", "Max Price (Rs./Quintal)",
        "future_modal_price", "future_price_change", "future_abs_range", "future_pct_range",
        "target_v2", "actual_horizon_14d", "target_date_14d", "future_modal_price_14d", "future_price_change_14d",
        "actual_horizon_30d", "target_date_30d", "future_modal_price_30d", "future_price_change_30d"
    ]
    cand_feats = [c for c in df.columns if c not in drop_cols]
    print(f"Candidate feature count: {len(cand_feats)}")
    
    # ------------------------------------------------------------
    # A. SHORT-TERM PRICE MODEL TRAINING & EVALUATION
    # ------------------------------------------------------------
    print("\n--- A. Short-Term Price Model ---")
    st_tr = df_st[df_st[date_col] <= split_date].copy().reset_index(drop=True)
    st_te = df_st[df_st[date_col] > split_date].copy().reset_index(drop=True)
    print(f"Price Train rows: {len(st_tr):,}, Holdout rows: {len(st_te):,}")
    
    if commodity == "wheat":
        # CatBoost MAE with deadband eps=3.0
        p_model = CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.04,
            loss_function="MAE", verbose=0, random_seed=42
        )
        p_model.fit(st_tr[cand_feats], st_tr["future_price_change"])
        deadband_eps = 3.0
    else:
        # Potato: XGBoost PseudoHuber with deadband eps=5.0
        p_model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            objective="reg:pseudohubererror", subsample=0.85, colsample_bytree=0.85,
            random_state=42, n_jobs=-1
        )
        p_model.fit(st_tr[cand_feats], st_tr["future_price_change"])
        deadband_eps = 5.0
        
    # Predict on test
    pred_delta = p_model.predict(st_te[cand_feats])
    # Apply deadband
    pred_delta[np.abs(pred_delta) < deadband_eps] = 0.0
    
    act_p = st_te["future_modal_price"].values
    cur_p = st_te[price_col].values
    pred_p = cur_p + pred_delta
    
    st_residuals = act_p - pred_p
    
    # Phase 10 Subsets for Price:
    train_groups = set(zip(st_tr["Market Name"], st_tr["Variety"]))
    test_groups = list(zip(st_te["Market Name"], st_te["Variety"]))
    seen_mask = np.array([g in train_groups for g in test_groups])
    unseen_mask = ~seen_mask
    
    # High-Volume Mandis (top 20% by observation count in test)
    mandi_counts = st_te["Market Name"].value_counts()
    top_mandis = set(mandi_counts.head(max(1, int(len(mandi_counts) * 0.2))).index)
    high_vol_mask = st_te["Market Name"].isin(top_mandis).values
    
    price_eval = {
        "overall": eval_price_subset(act_p, pred_p, cur_p),
        "seen_groups": eval_price_subset(act_p[seen_mask], pred_p[seen_mask], cur_p[seen_mask]),
        "unseen_groups": eval_price_subset(act_p[unseen_mask], pred_p[unseen_mask], cur_p[unseen_mask]),
        "high_volume_mandis": eval_price_subset(act_p[high_vol_mask], pred_p[high_vol_mask], cur_p[high_vol_mask]),
    }
    print(f"Price Overall: MAE=Rs.{price_eval['overall']['mae']:.2f} (P: Rs.{price_eval['overall']['persistence_mae']:.2f}), RMSE=Rs.{price_eval['overall']['rmse']:.2f} (P: Rs.{price_eval['overall']['persistence_rmse']:.2f})")
    print(f"Price Seen Groups ({price_eval['seen_groups']['n_samples']}): MAE=Rs.{price_eval['seen_groups']['mae']:.2f}")
    if price_eval['unseen_groups']['n_samples'] > 0:
        print(f"Price Unseen Groups ({price_eval['unseen_groups']['n_samples']}): MAE=Rs.{price_eval['unseen_groups']['mae']:.2f}")
        
    # ------------------------------------------------------------
    # B. DIRECT HORIZON MODELS (14-Day and 30-Day)
    # ------------------------------------------------------------
    print("\n--- B. Direct Horizon Models (14-Day and 30-Day) ---")
    horizon_models = {}
    horizon_residuals = {}
    horizon_eval = {}
    
    for h in [14, 30]:
        h_prefix = f"{h}d"
        h_df = compute_single_horizon_targets(
            df, horizon_days=h, max_tolerance_days=3,
            group_cols=group_cols, date_col=date_col, price_col=price_col
        )
        target_col = f"future_price_change_{h_prefix}"
        future_price_col = f"future_modal_price_{h_prefix}"
        
        valid_h = h_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
        tr_raw = valid_h[valid_h[date_col] <= split_date].copy().reset_index(drop=True)
        ho_data = valid_h[valid_h[date_col] > split_date].copy().reset_index(drop=True)
        
        tr_purged, p_count = apply_horizon_temporal_purge(
            tr_raw, split_date=str(split_date.date()), horizon_days=h, max_tolerance_days=3, date_col=date_col
        )
        
        h_cand = [c for c in cand_feats if c in tr_purged.columns]
        h_model = CatBoostRegressor(iterations=400, depth=6, learning_rate=0.04, loss_function="MAE", verbose=0, random_seed=42)
        h_model.fit(tr_purged[h_cand], tr_purged[target_col])
        
        h_pred_chg = h_model.predict(ho_data[h_cand])
        h_cur_p = ho_data[price_col].values
        h_act_p = ho_data[future_price_col].values
        h_pred_p = h_cur_p + h_pred_chg
        
        h_res = h_act_p - h_pred_p
        horizon_models[h_prefix] = h_model
        horizon_residuals[h_prefix] = h_res
        horizon_eval[h_prefix] = eval_price_subset(h_act_p, h_pred_p, h_cur_p)
        print(f"Horizon {h}d: MAE=Rs.{horizon_eval[h_prefix]['mae']:.2f} (P: Rs.{horizon_eval[h_prefix]['persistence_mae']:.2f}), RMSE=Rs.{horizon_eval[h_prefix]['rmse']:.2f} (P: Rs.{horizon_eval[h_prefix]['persistence_rmse']:.2f})")
        
    # ------------------------------------------------------------
    # C. VOLATILITY MODEL TRAINING & CALIBRATION
    # ------------------------------------------------------------
    print("\n--- C. Volatility Model Training & Calibration ---")
    vol_tr = df_vol[df_vol[date_col] <= split_date].copy().reset_index(drop=True)
    vol_te = df_vol[df_vol[date_col] > split_date].copy().reset_index(drop=True)
    
    t_low = 0.05
    t_high = 1.818 if commodity == "wheat" else 5.556
    
    def assign_classes(s, low_th, high_th):
        c = np.zeros(len(s), dtype=int)
        c[(s > low_th) & (s <= high_th)] = 1
        c[s > high_th] = 2
        return c
        
    vol_tr["target_v2"] = assign_classes(vol_tr["future_pct_range"].values, t_low, t_high)
    vol_te["target_v2"] = assign_classes(vol_te["future_pct_range"].values, t_low, t_high)
    
    # Chronological validation split for calibration
    val_split_date = vol_tr[date_col].quantile(0.80)
    v_tr_inner = vol_tr[vol_tr[date_col] <= val_split_date].copy().reset_index(drop=True)
    v_val_inner = vol_tr[vol_tr[date_col] > val_split_date].copy().reset_index(drop=True)
    
    counts = np.bincount(v_tr_inner["target_v2"], minlength=3)
    weights = len(v_tr_inner) / (3.0 * np.maximum(counts, 1))
    class_weights_dict = {i: float(weights[i]) for i in range(3)}
    sample_weights_tr = np.array([class_weights_dict[c] for c in v_tr_inner["target_v2"]])
    
    if commodity == "wheat":
        # CatBoost + Isotonic calibration
        base_v_model = CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.06,
            loss_function="MultiClass", class_weights=list(weights),
            verbose=0, random_seed=42
        )
        base_v_model.fit(v_tr_inner[cand_feats], v_tr_inner["target_v2"])
        cal_method = "isotonic"
    else:
        # Potato: XGBoost + Sigmoid calibration
        base_v_model = XGBClassifier(
            n_estimators=250, max_depth=5, learning_rate=0.05,
            objective="multi:softprob", num_class=3, random_state=42, n_jobs=-1
        )
        base_v_model.fit(v_tr_inner[cand_feats], v_tr_inner["target_v2"], sample_weight=sample_weights_tr)
        cal_method = "sigmoid"
        
    calibrator = CalibratedClassifierCV(estimator=base_v_model, method=cal_method, cv="prefit")
    calibrator.fit(v_val_inner[cand_feats], v_val_inner["target_v2"])
    
    # Test evaluation
    te_y = vol_te["target_v2"].values
    te_probs = calibrator.predict_proba(vol_te[cand_feats])
    
    vol_tr_groups = set(zip(vol_tr["Market Name"], vol_tr["Variety"]))
    vol_te_groups = list(zip(vol_te["Market Name"], vol_te["Variety"]))
    vol_seen_mask = np.array([g in vol_tr_groups for g in vol_te_groups])
    vol_unseen_mask = ~vol_seen_mask
    vol_high_vol_mask = vol_te["Market Name"].isin(top_mandis).values
    
    vol_eval = {
        "overall": eval_vol_subset(te_y, te_probs),
        "seen_groups": eval_vol_subset(te_y[vol_seen_mask], te_probs[vol_seen_mask]),
        "unseen_groups": eval_vol_subset(te_y[vol_unseen_mask], te_probs[vol_unseen_mask]),
        "high_volume_mandis": eval_vol_subset(te_y[vol_high_vol_mask], te_probs[vol_high_vol_mask]),
    }
    print(f"Volatility Overall: Acc={vol_eval['overall']['accuracy']*100:.2f}%, BalAcc={vol_eval['overall']['balanced_accuracy']*100:.2f}%, MacroF1={vol_eval['overall']['macro_f1']:.4f}, MCC={vol_eval['overall']['mcc']:.4f}, HIGH PR-AUC={vol_eval['overall']['high_pr_auc']:.4f}")
    
    # ------------------------------------------------------------
    # D. EXPORT PRODUCTION ARTIFACTS
    # ------------------------------------------------------------
    print("\n--- D. Exporting Production Artifacts ---")
    
    # 1. Price Model
    p_path = out_dir / f"{commodity}_price_model.pkl"
    p_wrapper = {
        "model": p_model,
        "feature_names": cand_feats,
        "target": "future_price_change",
        "deadband_eps": deadband_eps,
        "parameters": p_model.get_params() if hasattr(p_model, "get_params") else {},
        "metadata": {
            "commodity": commodity,
            "version": "2.0.0",
            "horizon": "next_valid_observation_7d"
        }
    }
    joblib.dump(p_wrapper, p_path)
    print(f"Saved: {p_path}")
    
    # 2. Horizon 14d and 30d Models
    p14_path = out_dir / f"{commodity}_price_14d_model.pkl"
    joblib.dump({"model": horizon_models["14d"], "feature_names": cand_feats, "horizon": 14}, p14_path)
    p30_path = out_dir / f"{commodity}_price_30d_model.pkl"
    joblib.dump({"model": horizon_models["30d"], "feature_names": cand_feats, "horizon": 30}, p30_path)
    print(f"Saved: {p14_path}")
    print(f"Saved: {p30_path}")
    
    # 3. Volatility Model
    v_path = out_dir / f"{commodity}_volatility_model.pkl"
    v_wrapper = {
        "model": calibrator,
        "base_model": base_v_model,
        "calibration_method": cal_method,
        "feature_names": cand_feats,
        "classes": ["LOW", "MEDIUM", "HIGH"],
        "thresholds": {"low_medium_pct": t_low, "medium_high_pct": t_high},
        "metadata": {
            "commodity": commodity,
            "version": "2.0.0",
            "target": "future_percentage_price_range_5obs"
        }
    }
    joblib.dump(v_wrapper, v_path)
    print(f"Saved: {v_path}")
    
    # 4. Feature Config
    feat_cfg = {
        "commodity": commodity,
        "version": "2.0.0",
        "price_model": {
            "type": type(p_model).__name__,
            "target": "future_price_change",
            "deadband_eps": deadband_eps,
            "features": cand_feats,
            "reconstruction_column": price_col,
            "n_features": len(cand_feats)
        },
        "volatility_model": {
            "type": f"{type(base_v_model).__name__}+{cal_method}_calibration",
            "target": "volatility_percentage_range",
            "classes": ["LOW", "MEDIUM", "HIGH"],
            "features": cand_feats,
            "n_features": len(cand_feats)
        },
        "horizons": {
            "short_term": "next valid observation in [T+1, T+7]",
            "14d": "first valid observation in [T+14, T+17]",
            "30d": "first valid observation in [T+30, T+33]"
        }
    }
    feat_cfg_path = out_dir / f"{commodity}_feature_config.json"
    with open(feat_cfg_path, "w", encoding="utf-8") as f:
        json.dump(feat_cfg, f, indent=2)
    print(f"Saved: {feat_cfg_path}")
    
    # 5. Volatility Config
    vol_cfg = {
        "commodity": commodity,
        "version": "2.0.0",
        "volatility_definition": "5-observation percentage price range ((max - min) / current_modal_price * 100)",
        "low_medium_threshold_pct": t_low,
        "medium_high_threshold_pct": t_high,
        "classes": ["LOW", "MEDIUM", "HIGH"],
        "calibrator": cal_method,
        "class_weights": class_weights_dict
    }
    vol_cfg_path = out_dir / f"{commodity}_volatility_config.json"
    with open(vol_cfg_path, "w", encoding="utf-8") as f:
        json.dump(vol_cfg, f, indent=2)
    print(f"Saved: {vol_cfg_path}")
    
    # 6. Empirical Holdout Residual Distributions
    def compute_residual_dict(residuals, horizon_name, n_samples):
        return {
            "crop": commodity,
            "horizon": horizon_name,
            "model_version": "2.0.0",
            "n_holdout_samples": int(n_samples),
            "quantiles": {
                "p05": float(np.percentile(residuals, 5)),
                "p10": float(np.percentile(residuals, 10)),
                "p25": float(np.percentile(residuals, 25)),
                "p50": float(np.percentile(residuals, 50)),
                "p75": float(np.percentile(residuals, 75)),
                "p90": float(np.percentile(residuals, 90)),
                "p95": float(np.percentile(residuals, 95)),
            },
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "iqr": float(np.percentile(residuals, 75) - np.percentile(residuals, 25))
        }
        
    residuals_summary = {
        "commodity": commodity,
        "short_term": compute_residual_dict(st_residuals, "short_term_7d", len(st_residuals)),
        "14d": compute_residual_dict(horizon_residuals["14d"], "14_calendar_days", len(horizon_residuals["14d"])),
        "30d": compute_residual_dict(horizon_residuals["30d"], "30_calendar_days", len(horizon_residuals["30d"])),
    }
    res_path = out_dir / f"{commodity}_residual_distributions.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(residuals_summary, f, indent=2)
    print(f"Saved: {res_path}")
    
    # 7. Metrics JSON (Full breakdown)
    full_metrics = {
        "commodity": commodity,
        "version": "2.0.0",
        "price_eval": price_eval,
        "horizon_eval": horizon_eval,
        "volatility_eval": vol_eval,
    }
    metrics_path = out_dir / f"{commodity}_v2_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"Saved: {metrics_path}")
    
    # 8. Manifest JSON with Checksums
    manifest = {
        "commodity": commodity,
        "version": "2.0.0",
        "created_at": "2026-11-15T20:30:00Z",
        "artifacts": {}
    }
    for item in sorted(out_dir.glob("*")):
        if item.name == "manifest.json":
            continue
        manifest["artifacts"][item.name] = {
            "size_bytes": item.stat().st_size,
            "sha256": compute_sha256(item)
        }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {manifest_path}")
    print(f"Successfully generated all V2 production artifacts for {commodity.upper()}!")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "wheat"
    train_and_export_v2(c)
