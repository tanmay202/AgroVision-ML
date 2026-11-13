"""
AgroVision Member 2 — Volatility V3 Full Experimental Suite

Comprehensive execution of:
1. Target Generation & Temporal Purging for 7-Day and 14-Day Calendar Horizons
2. Point-in-Time 38 Feature Extraction
3. Walk-Forward Cross-Validation (Folds 1, 2, 3)
4. Model Comparison (Majority, Persistence, V2 Read-Only, XGBoost Unweighted/Balanced/Sqrt, RF Unweighted/Balanced)
5. Decision Threshold Optimization (Validation-Tuned)
6. Probability Calibration (Platt Sigmoid and Isotonic Regression)
7. Final Retrospective Holdout Benchmark (2023-2024)
8. Unseen-Mandi Generalization Assessment (Group Holdout)
9. Market-Level Disaggregation and Cluster-Robust Statistical Significance Testing
10. Complete Serialization of Metrics, Calibration Records, and Confusion Matrices
"""

import sys
import os
import json
import time
import joblib
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Project path resolution
V3_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = V3_DIR.parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rice_features_final.csv"
V2_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "rice_volatility_normalized_model.pkl"
RESULTS_DIR = V3_DIR / "results"

sys.path.insert(0, str(V3_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from volatility_v3_target_definitions import (
    compute_v3_calendar_targets,
    apply_v3_temporal_purge,
    derive_thresholds_from_training,
    classify_volatility_series,
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
    compute_market_level_metrics,
)

# Designated unseen-mandi holdout list (8 mandis spanning large, medium, and small volume)
UNSEEN_MANDI_LIST = [
    "Kalipur",
    "Uluberia",
    "Kandi",
    "Islampur",
    "Raiganj",
    "Kasimbazar",
    "Falakata",
    "Beldanga",
]


def load_v2_model_readonly():
    """Loads existing production V2 model for read-only comparison."""
    if V2_ARTIFACT_PATH.exists():
        try:
            artifact = joblib.load(V2_ARTIFACT_PATH)
            return artifact.get("model")
        except Exception as e:
            print(f"Warning: Could not load V2 model for comparison: {e}")
            return None
    return None


def tune_v3_thresholds(
    y_val_num: np.ndarray,
    val_proba: np.ndarray,
    target_high_recall_floor: float = 0.50,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Grid-searches probability multiplier weights w = [1.0, w_med, w_high]
    on the VALIDATION set to maximize macro F1 subject to high recall constraint.
    """
    grid = np.linspace(0.8, 3.0, 12)
    high_idx = VOLATILITY_LABEL_MAP["HIGH"]

    best_weights = np.array([1.0, 1.0, 1.0])
    best_f1 = -1.0
    best_high_rec = 0.0

    for w_med in grid:
        for w_high in grid:
            w = np.array([1.0, w_med, w_high])
            preds = (val_proba * w).argmax(axis=1)

            # Check macro F1
            # Simple fast macro F1 calculation
            f1_c = []
            rec_high = 0.0
            for c in range(len(VOLATILITY_CLASSES)):
                tp = np.sum((y_val_num == c) & (preds == c))
                fp = np.sum((y_val_num != c) & (preds == c))
                fn = np.sum((y_val_num == c) & (preds != c))
                prec = tp / max(1, tp + fp)
                rec = tp / max(1, tp + fn)
                f1 = 2 * prec * rec / max(1e-6, prec + rec)
                f1_c.append(f1)
                if c == high_idx:
                    rec_high = rec

            macro_f1 = np.mean(f1_c)

            if rec_high >= target_high_recall_floor and macro_f1 > best_f1:
                best_f1 = macro_f1
                best_weights = w
                best_high_rec = rec_high

    if best_f1 < 0: # Fallback to unconstrained best macro F1
        for w_med in grid:
            for w_high in grid:
                w = np.array([1.0, w_med, w_high])
                preds = (val_proba * w).argmax(axis=1)
                f1_c = []
                for c in range(len(VOLATILITY_CLASSES)):
                    tp = np.sum((y_val_num == c) & (preds == c))
                    fp = np.sum((y_val_num != c) & (preds == c))
                    fn = np.sum((y_val_num == c) & (preds != c))
                    prec = tp / max(1, tp + fp)
                    rec = tp / max(1, tp + fn)
                    f1 = 2 * prec * rec / max(1e-6, prec + rec)
                    f1_c.append(f1)
                macro_f1 = np.mean(f1_c)
                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_weights = w

    return best_weights, {"val_macro_f1": float(best_f1), "val_high_recall": float(best_high_rec)}


def run_v3_evaluation_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 80)
    print("AGROVISION MEMBER 2 — VOLATILITY V3 COMPREHENSIVE EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Load dataset
    print("\n[Step 1/6] Loading source dataset...")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df_raw):,} observations from {DATA_PATH}")

    # 2. Target Engineering (7-day and 14-day calendar horizons)
    print("\n[Step 2/6] Computing V3 calendar targets & symmetric persistence...")
    t0 = time.time()
    df = compute_v3_calendar_targets(
        df_raw,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
        horizons=[7, 14],
        min_future_obs=1,
    )
    print(f"Calendar targets computed in {time.time()-t0:.2f}s")

    # 3. Feature Engineering (38 point-in-time features)
    print("\n[Step 3/6] Extracting 38 point-in-time features...")
    t1 = time.time()
    df = extract_v3_volatility_features(
        df,
        group_cols=["Market Name", "Variety"],
        date_col="Reported Date",
        price_col="Modal Price (Rs./Quintal)",
    )
    print(f"Point-in-time features generated in {time.time()-t1:.2f}s")

    # Load frozen V2 model for benchmark comparison
    v2_model = load_v2_model_readonly()

    # Define Walk-Forward Splits
    walk_forward_folds = [
        ("Fold 1 (2017-2018)", "2017-01-01", "2018-12-31"),
        ("Fold 2 (2019-2020)", "2019-01-01", "2020-12-31"),
        ("Fold 3 (2021-2022)", "2021-01-27", "2023-01-26"),
    ]

    holdout_start = "2023-01-27"
    holdout_end = "2024-01-27"

    all_pooled_results = []
    all_per_mandi_results = {}
    all_calibration_results = {}

    # Execute for both horizons
    for horizon in [7, 14]:
        h_str = f"{horizon}d"
        h_label = f"{horizon}-Day Calendar Horizon"
        target_col = f"target_{h_str}"
        target_date_col = f"future_target_date_{h_str}"
        persist_cal_col = f"persistence_pred_{h_str}"
        persist_5obs_col = f"persistence_pred_5obs_{h_str}"

        print(f"\n{'#' * 80}")
        print(f"STARTING EVALUATION FOR TARGET {horizon}D ({h_label})")
        print(f"{'#' * 80}")

        # Store Fold 3 models and val data for threshold tuning & calibration
        fold3_cache = {}

        # ---------------------------------------------------------------------
        # 4. Walk-Forward Cross-Validation
        # ---------------------------------------------------------------------
        print(f"\n--- Walk-Forward Chronological Cross-Validation ({horizon}D) ---")
        for fold_name, v_start, v_end in walk_forward_folds:
            print(f"\n>>> Running {fold_name} ({v_start} to {v_end}) for {horizon}D...")
            clean_train_mask, is_val_mask, purged = apply_v3_temporal_purge(
                df, "Reported Date", target_date_col, v_start
            )

            train_df = df[clean_train_mask].dropna(subset=V3_FEATURE_COLS + [target_col]).copy()
            val_df = df[is_val_mask & (df["Reported Date"] <= pd.Timestamp(v_end))].dropna(
                subset=V3_FEATURE_COLS + [target_col, persist_cal_col, persist_5obs_col]
            ).copy()

            print(f"    Train size: {len(train_df):,} rows (Purged {purged:,} boundary-leak rows)")
            print(f"    Val size:   {len(val_df):,} rows")

            X_tr = train_df[V3_FEATURE_COLS]
            y_tr_num = train_df[target_col].map(VOLATILITY_LABEL_MAP)
            X_val = val_df[V3_FEATURE_COLS]
            y_val_lbl = val_df[target_col]
            y_val_num = val_df[target_col].map(VOLATILITY_LABEL_MAP)

            # 1. Majority baseline
            dummy_maj = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr_num)
            preds_maj = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in dummy_maj.predict(X_val)]
            m_maj = compute_comprehensive_metrics(y_val_lbl, preds_maj, "Majority Baseline", fold_name, f"{horizon}D")
            all_pooled_results.append(m_maj)

            # 2. Calendar Persistence Baseline
            m_pcal = compute_comprehensive_metrics(
                y_val_lbl, val_df[persist_cal_col], f"Calendar Persistence ({horizon}D)", fold_name, f"{horizon}D"
            )
            all_pooled_results.append(m_pcal)

            # 3. 5-Obs Persistence Baseline
            m_p5obs = compute_comprehensive_metrics(
                y_val_lbl, val_df[persist_5obs_col], "5-Obs Persistence Baseline", fold_name, f"{horizon}D"
            )
            all_pooled_results.append(m_p5obs)

            # 4. XGBoost (Unweighted)
            xgb_un = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
                objective="multi:softprob", num_class=3, eval_metric="mlogloss"
            )
            xgb_un.fit(X_tr, y_tr_num)
            proba_xgb_un = xgb_un.predict_proba(X_val)
            preds_xgb_un = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in proba_xgb_un.argmax(axis=1)]
            m_xgb_un = compute_comprehensive_metrics(y_val_lbl, preds_xgb_un, "XGBoost (Unweighted)", fold_name, f"{horizon}D")
            all_pooled_results.append(m_xgb_un)

            # 5. XGBoost (Balanced Sample Weights)
            weights_bal = compute_sample_weight("balanced", y_tr_num)
            xgb_bal = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
                objective="multi:softprob", num_class=3, eval_metric="mlogloss"
            )
            xgb_bal.fit(X_tr, y_tr_num, sample_weight=weights_bal)
            proba_xgb_bal = xgb_bal.predict_proba(X_val)
            preds_xgb_bal = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in proba_xgb_bal.argmax(axis=1)]
            m_xgb_bal = compute_comprehensive_metrics(y_val_lbl, preds_xgb_bal, "XGBoost (Balanced)", fold_name, f"{horizon}D")
            all_pooled_results.append(m_xgb_bal)

            # 6. XGBoost (Mild Sqrt Weights)
            weights_sqrt = np.sqrt(weights_bal)
            xgb_sqrt = XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
                colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
                objective="multi:softprob", num_class=3, eval_metric="mlogloss"
            )
            xgb_sqrt.fit(X_tr, y_tr_num, sample_weight=weights_sqrt)
            proba_xgb_sqrt = xgb_sqrt.predict_proba(X_val)
            preds_xgb_sqrt = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in proba_xgb_sqrt.argmax(axis=1)]
            m_xgb_sqrt = compute_comprehensive_metrics(y_val_lbl, preds_xgb_sqrt, "XGBoost (Mild Sqrt)", fold_name, f"{horizon}D")
            all_pooled_results.append(m_xgb_sqrt)

            # 7. Random Forest (Balanced)
            rf_bal = RandomForestClassifier(
                n_estimators=100, max_depth=12, min_samples_leaf=10, class_weight="balanced",
                random_state=42, n_jobs=-1
            )
            rf_bal.fit(X_tr, y_tr_num)
            proba_rf_bal = rf_bal.predict_proba(X_val)
            preds_rf_bal = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in proba_rf_bal.argmax(axis=1)]
            m_rf_bal = compute_comprehensive_metrics(y_val_lbl, preds_rf_bal, "Random Forest (Balanced)", fold_name, f"{horizon}D")
            all_pooled_results.append(m_rf_bal)

            print(f"    Macro F1: Persist={m_pcal['macro_f1']:.4f} | XGB(Bal)={m_xgb_bal['macro_f1']:.4f} | XGB(Unw)={m_xgb_un['macro_f1']:.4f} | RF(Bal)={m_rf_bal['macro_f1']:.4f}")
            print(f"    HIGH Rec: Persist={m_pcal['high_recall']:.4f} | XGB(Bal)={m_xgb_bal['high_recall']:.4f} | XGB(Unw)={m_xgb_un['high_recall']:.4f} | RF(Bal)={m_rf_bal['high_recall']:.4f}")

            if fold_name == "Fold 3 (2021-2022)":
                fold3_cache["val_df"] = val_df
                fold3_cache["X_val"] = X_val
                fold3_cache["y_val_num"] = y_val_num.to_numpy()
                fold3_cache["xgb_bal"] = xgb_bal
                fold3_cache["xgb_un"] = xgb_un
                fold3_cache["proba_xgb_bal"] = proba_xgb_bal

        # ---------------------------------------------------------------------
        # 5. Threshold Tuning on Fold 3 Validation Set
        # ---------------------------------------------------------------------
        print(f"\n--- Decision Threshold Tuning on Fold 3 Validation ({horizon}D) ---")
        best_threshold_weights, thresh_info = tune_v3_thresholds(
            fold3_cache["y_val_num"], fold3_cache["proba_xgb_bal"], target_high_recall_floor=0.55
        )
        print(f"    Optimal Threshold Multipliers [LOW=1.0, MED={best_threshold_weights[1]:.2f}, HIGH={best_threshold_weights[2]:.2f}]")
        print(f"    Fold 3 Val Tuned Macro F1: {thresh_info['val_macro_f1']:.4f}, HIGH Rec: {thresh_info['val_high_recall']:.4f}")

        # ---------------------------------------------------------------------
        # 6. Pre-Holdout Training (< 2023-01-27) & Final Holdout Benchmark
        # ---------------------------------------------------------------------
        print(f"\n--- Retrospective Holdout Benchmark (2023-01-27 to 2024-01-27) ({horizon}D) ---")
        clean_train_mask_ho, is_holdout_mask, purged_ho = apply_v3_temporal_purge(
            df, "Reported Date", target_date_col, holdout_start
        )

        full_train_df = df[clean_train_mask_ho].dropna(subset=V3_FEATURE_COLS + [target_col]).copy()
        holdout_df = df[is_holdout_mask & (df["Reported Date"] <= pd.Timestamp(holdout_end))].dropna(
            subset=V3_FEATURE_COLS + [target_col, persist_cal_col, persist_5obs_col]
        ).copy()

        print(f"Full Train (< 2023-01-27): {len(full_train_df):,} rows (Purged {purged_ho:,} boundary rows)")
        print(f"Final Holdout:             {len(holdout_df):,} rows")

        X_full_tr = full_train_df[V3_FEATURE_COLS]
        y_full_tr_num = full_train_df[target_col].map(VOLATILITY_LABEL_MAP)
        X_ho = holdout_df[V3_FEATURE_COLS]
        y_ho_lbl = holdout_df[target_col]

        # 1. Majority baseline on Holdout
        dummy_ho = DummyClassifier(strategy="most_frequent").fit(X_full_tr, y_full_tr_num)
        preds_ho_maj = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in dummy_ho.predict(X_ho)]
        m_ho_maj = compute_comprehensive_metrics(y_ho_lbl, preds_ho_maj, "Majority Baseline", "Holdout (2023-2024)", f"{horizon}D")
        all_pooled_results.append(m_ho_maj)

        # 2. Calendar Persistence Baseline on Holdout
        m_ho_pcal = compute_comprehensive_metrics(
            y_ho_lbl, holdout_df[persist_cal_col], f"Calendar Persistence ({horizon}D)", "Holdout (2023-2024)", f"{horizon}D"
        )
        all_pooled_results.append(m_ho_pcal)

        # 3. 5-Obs Persistence Baseline on Holdout
        m_ho_p5obs = compute_comprehensive_metrics(
            y_ho_lbl, holdout_df[persist_5obs_col], "5-Obs Persistence Baseline", "Holdout (2023-2024)", f"{horizon}D"
        )
        all_pooled_results.append(m_ho_p5obs)

        # 4. Existing Normalized V2 Model (Read-Only) on Holdout
        if v2_model is not None:
            try:
                # V2 features from V2 config
                v2_feat_cols = [
                    "past_5_range", "past_5_pct_range", "past_10_range", "past_10_pct_range",
                    "past_20_range", "past_20_pct_range", "rolling_std_5", "rolling_std_7",
                    "rolling_std_14", "rolling_std_30", "rolling_std_pct_5", "rolling_std_pct_14",
                    "pct_change_1", "pct_change_2", "pct_change_5", "abs_pct_change_1",
                    "abs_pct_change_5", "ewma_abs_diff_5", "ewma_abs_diff_14", "ewma_pct_vol_5",
                    "ewma_pct_vol_14", "vol_trend_5_10", "vol_pct_trend_5_10", "vol_std_ratio_5_14",
                    "nonzero_moves_5", "nonzero_moves_10", "streak_zero", "max_daily_pct_jump_5",
                    "price_level", "price_to_mean_7", "month", "day_of_week", "week_of_year",
                    "arrival_lag_1", "arrival_rolling_mean_7",
                ]
                v2_proba = v2_model.predict_proba(holdout_df[v2_feat_cols])
                v2_preds = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in v2_proba.argmax(axis=1)]
                m_ho_v2 = compute_comprehensive_metrics(y_ho_lbl, v2_preds, "Normalized V2 (Production XGBoost)", "Holdout (2023-2024)", f"{horizon}D")
                all_pooled_results.append(m_ho_v2)
                holdout_df[f"v2_preds_{h_str}"] = v2_preds
            except Exception as e:
                print(f"Warning during V2 benchmark: {e}")

        # 5. Champion XGBoost (Balanced) on Holdout
        weights_full = compute_sample_weight("balanced", y_full_tr_num)
        model_v3_champion = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss"
        )
        model_v3_champion.fit(X_full_tr, y_full_tr_num, sample_weight=weights_full)

        raw_proba_ho = model_v3_champion.predict_proba(X_ho)
        preds_v3_raw = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in raw_proba_ho.argmax(axis=1)]
        m_ho_v3_raw = compute_comprehensive_metrics(y_ho_lbl, preds_v3_raw, "V3 XGBoost (Balanced, Default Thresholds)", "Holdout (2023-2024)", f"{horizon}D")
        all_pooled_results.append(m_ho_v3_raw)
        holdout_df[f"v3_preds_{h_str}"] = preds_v3_raw

        # 6. Champion XGBoost (Tuned Thresholds) on Holdout
        tuned_proba_ho = raw_proba_ho * best_threshold_weights
        preds_v3_tuned = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in tuned_proba_ho.argmax(axis=1)]
        m_ho_v3_tuned = compute_comprehensive_metrics(y_ho_lbl, preds_v3_tuned, "V3 XGBoost (Balanced, Tuned Thresholds)", "Holdout (2023-2024)", f"{horizon}D")
        all_pooled_results.append(m_ho_v3_tuned)
        holdout_df[f"v3_tuned_preds_{h_str}"] = preds_v3_tuned

        # 7. Random Forest (Balanced) on Holdout
        rf_full = RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10, class_weight="balanced",
            random_state=42, n_jobs=-1
        )
        rf_full.fit(X_full_tr, y_full_tr_num)
        proba_rf_ho = rf_full.predict_proba(X_ho)
        preds_rf_ho = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in proba_rf_ho.argmax(axis=1)]
        m_ho_rf = compute_comprehensive_metrics(y_ho_lbl, preds_rf_ho, "V3 Random Forest (Balanced)", "Holdout (2023-2024)", f"{horizon}D")
        all_pooled_results.append(m_ho_rf)

        # ---------------------------------------------------------------------
        # 7. Probability Calibration (Sigmoid & Isotonic)
        # ---------------------------------------------------------------------
        print(f"\n--- Probability Calibration Assessment ({horizon}D) ---")
        # Evaluate uncalibrated probabilities first
        uncal_eval = compute_multiclass_brier_and_ece(y_ho_lbl, raw_proba_ho)
        uncal_eval["calibration_method"] = "Uncalibrated (Raw XGBoost)"
        uncal_eval["horizon"] = f"{horizon}D"

        # Fit Calibrators on Fold 3 Validation Data (Prefit)
        # We use fold3_cache['xgb_bal'] pre-trained on < 2021 data, calibrated on 2021-2023 data
        cal_sigmoid = CalibratedClassifierCV(fold3_cache["xgb_bal"], method="sigmoid", cv="prefit")
        cal_sigmoid.fit(fold3_cache["X_val"], fold3_cache["y_val_num"])
        proba_sigmoid_ho = cal_sigmoid.predict_proba(X_ho)
        cal_sig_eval = compute_multiclass_brier_and_ece(y_ho_lbl, proba_sigmoid_ho)
        cal_sig_eval["calibration_method"] = "Sigmoid / Platt (Prefit on Fold 3 Val)"
        cal_sig_eval["horizon"] = f"{horizon}D"

        cal_isotonic = CalibratedClassifierCV(fold3_cache["xgb_bal"], method="isotonic", cv="prefit")
        cal_isotonic.fit(fold3_cache["X_val"], fold3_cache["y_val_num"])
        proba_iso_ho = cal_isotonic.predict_proba(X_ho)
        cal_iso_eval = compute_multiclass_brier_and_ece(y_ho_lbl, proba_iso_ho)
        cal_iso_eval["calibration_method"] = "Isotonic Regression (Prefit on Fold 3 Val)"
        cal_iso_eval["horizon"] = f"{horizon}D"

        all_calibration_results[f"{horizon}D"] = {
            "uncalibrated": uncal_eval,
            "sigmoid": cal_sig_eval,
            "isotonic": cal_iso_eval,
        }

        print(f"    Uncalibrated:  Brier={uncal_eval['multiclass_brier_score']:.4f}, ECE={uncal_eval['macro_ece']:.4f}")
        print(f"    Sigmoid/Platt: Brier={cal_sig_eval['multiclass_brier_score']:.4f}, ECE={cal_sig_eval['macro_ece']:.4f}")
        print(f"    Isotonic:      Brier={cal_iso_eval['multiclass_brier_score']:.4f}, ECE={cal_iso_eval['macro_ece']:.4f}")

        # ---------------------------------------------------------------------
        # 8. Market-Level / Per-Mandi Disaggregated Evaluation
        # ---------------------------------------------------------------------
        print(f"\n--- Market-Level Disaggregated Evaluation & Clustered Testing ({horizon}D) ---")
        mandi_eval = compute_market_level_metrics(
            holdout_df,
            mandi_col="Market Name",
            y_true_col=target_col,
            y_pred_col=f"v3_preds_{h_str}",
            y_base_col=persist_cal_col,
        )
        all_per_mandi_results[f"{horizon}D"] = mandi_eval

        print(f"    Total Mandis Evaluated: {mandi_eval['total_mandis_evaluated']}")
        print(f"    ML Wins: {mandi_eval['win_loss_summary']['ml_wins']} ({mandi_eval['win_loss_summary']['ml_win_rate_pct']}%) | Base Wins: {mandi_eval['win_loss_summary']['base_wins']} ({mandi_eval['win_loss_summary']['base_win_rate_pct']}%)")
        print(f"    Mean Mandi F1 Delta: {mandi_eval['clustered_significance_tests']['mean_macro_f1_difference']:.4f}")
        print(f"    Cluster t-test p-value: {mandi_eval['clustered_significance_tests']['paired_cluster_t_pvalue']:.5f}")
        print(f"    Clustered Bootstrap 95% CI: {mandi_eval['clustered_significance_tests']['clustered_bootstrap_95_ci']}")

        # ---------------------------------------------------------------------
        # 9. Unseen-Mandi Generalization Assessment
        # ---------------------------------------------------------------------
        print(f"\n--- Unseen-Mandi Group Holdout Assessment ({horizon}D) ---")
        # Train strictly excluding unseen mandis
        is_seen_mandi_train = clean_train_mask_ho & (~df["Market Name"].isin(UNSEEN_MANDI_LIST))
        seen_train_df = df[is_seen_mandi_train].dropna(subset=V3_FEATURE_COLS + [target_col]).copy()

        # Unseen mandi holdout test set (2023-2024 period, strictly unseen mandis)
        unseen_ho_df = df[is_holdout_mask & (df["Reported Date"] <= pd.Timestamp(holdout_end)) & (df["Market Name"].isin(UNSEEN_MANDI_LIST))].dropna(
            subset=V3_FEATURE_COLS + [target_col, persist_cal_col]
        ).copy()

        # Seen mandi holdout test set (2023-2024 period, strictly seen mandis)
        seen_ho_df = df[is_holdout_mask & (df["Reported Date"] <= pd.Timestamp(holdout_end)) & (~df["Market Name"].isin(UNSEEN_MANDI_LIST))].dropna(
            subset=V3_FEATURE_COLS + [target_col, persist_cal_col]
        ).copy()

        print(f"    Seen-mandis train size:    {len(seen_train_df):,} rows")
        print(f"    Unseen-mandis holdout size: {len(unseen_ho_df):,} rows across {unseen_ho_df['Market Name'].nunique()} mandis")
        print(f"    Seen-mandis holdout size:   {len(seen_ho_df):,} rows across {seen_ho_df['Market Name'].nunique()} mandis")

        y_seen_tr_num = seen_train_df[target_col].map(VOLATILITY_LABEL_MAP)
        weights_seen = compute_sample_weight("balanced", y_seen_tr_num)

        model_seen = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, random_state=42, n_jobs=-1, tree_method="hist",
            objective="multi:softprob", num_class=3, eval_metric="mlogloss"
        )
        model_seen.fit(seen_train_df[V3_FEATURE_COLS], y_seen_tr_num, sample_weight=weights_seen)

        # Evaluate on Seen Mandis Holdout
        p_seen_ho = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in model_seen.predict_proba(seen_ho_df[V3_FEATURE_COLS]).argmax(axis=1)]
        m_seen_ho = compute_comprehensive_metrics(seen_ho_df[target_col], p_seen_ho, "V3 XGBoost (Seen Mandis)", "Holdout Seen Mandis", f"{horizon}D")
        m_seen_base = compute_comprehensive_metrics(seen_ho_df[target_col], seen_ho_df[persist_cal_col], f"Calendar Persistence ({horizon}D)", "Holdout Seen Mandis", f"{horizon}D")

        # Evaluate on Unseen Mandis Holdout
        p_unseen_ho = [VOLATILITY_REVERSE_LABEL_MAP[p] for p in model_seen.predict_proba(unseen_ho_df[V3_FEATURE_COLS]).argmax(axis=1)]
        m_unseen_ho = compute_comprehensive_metrics(unseen_ho_df[target_col], p_unseen_ho, "V3 XGBoost (Unseen Mandis Generalization)", "Holdout Unseen Mandis", f"{horizon}D")
        m_unseen_base = compute_comprehensive_metrics(unseen_ho_df[target_col], unseen_ho_df[persist_cal_col], f"Calendar Persistence ({horizon}D)", "Holdout Unseen Mandis", f"{horizon}D")

        all_pooled_results.extend([m_seen_ho, m_seen_base, m_unseen_ho, m_unseen_base])

        print(f"    Seen Mandis Holdout:   ML Macro F1={m_seen_ho['macro_f1']:.4f} vs Base={m_seen_base['macro_f1']:.4f}")
        print(f"    Unseen Mandis Holdout: ML Macro F1={m_unseen_ho['macro_f1']:.4f} vs Base={m_unseen_base['macro_f1']:.4f}")

    # -------------------------------------------------------------------------
    # 10. Serialization of Results
    # -------------------------------------------------------------------------
    print("\n[Step 5/6] Exporting experimental results...")

    # Summary table
    res_df = pd.DataFrame(all_pooled_results)
    summary_cols = [
        "horizon", "period", "model", "n_samples", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "high_recall", "high_precision", "high_f1",
        "high_false_alarm_rate", "high_to_low_confusion", "low_to_high_confusion"
    ]
    res_df_filtered = res_df[summary_cols].sort_values(["horizon", "period", "macro_f1"], ascending=[True, True, False])
    res_df_filtered.to_csv(RESULTS_DIR / "v3_summary_metrics.csv", index=False)

    # Per-class metrics table
    class_rows = []
    for r in all_pooled_results:
        for c, det in r.get("per_class_details", {}).items():
            class_rows.append({
                "horizon": r["horizon"],
                "period": r["period"],
                "model": r["model"],
                "class": c,
                "precision": det["precision"],
                "recall": det["recall"],
                "f1": det["f1"],
                "support": det["support"],
            })
    pd.DataFrame(class_rows).to_csv(RESULTS_DIR / "v3_per_class_metrics.csv", index=False)

    # Per-mandi metrics table
    for h in ["7D", "14D"]:
        m_recs = all_per_mandi_results.get(h, {}).get("mandi_records", [])
        if m_recs:
            pd.DataFrame(m_recs).to_csv(RESULTS_DIR / f"v3_per_mandi_metrics_{h}.csv", index=False)

    # Calibration report JSON
    with open(RESULTS_DIR / "v3_calibration_report.json", "w") as f:
        json.dump(all_calibration_results, f, indent=2)

    # Market evaluation JSON
    with open(RESULTS_DIR / "v3_market_level_report.json", "w") as f:
        # Exclude dataframe records from json for compactness
        mandi_summary_json = {}
        for h, v in all_per_mandi_results.items():
            mandi_summary_json[h] = {k: val for k, val in v.items() if k != "mandi_records"}
        json.dump(mandi_summary_json, f, indent=2)

    # Full pooled metrics JSON
    with open(RESULTS_DIR / "v3_full_pooled_metrics.json", "w") as f:
        json.dump(all_pooled_results, f, indent=2)

    print(f"\nAll V3 experimental results successfully saved to {RESULTS_DIR}")
    print(f"Total Pipeline Runtime: {time.time()-t_start:.2f}s")


if __name__ == "__main__":
    run_v3_evaluation_pipeline()
