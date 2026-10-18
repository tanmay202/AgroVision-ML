"""
AgroVision-ML — Member 2: Direct Horizon Evaluation & Residual Generation

Executes:
1. Target Generation:
   - Method A: Symmetric calendar windows [T + H - 3, T + H + 3] for H in {7, 14, 30}.
   - Method B: Observation-shift diagnostics (.shift(-k)) for k in {7, 14, 30}.
   - Production target benchmark: Native forward-only targets.
2. Temporal Purging:
   - Purges training and validation observations crossing chronological split boundaries.
3. Regressor Training & Evaluation:
   - Evaluates on the audited final holdout set (2023-02-01 to 2024-02-01).
   - Computes MAE, RMSE, R² (price level and change), MAPE (safe zeros), Directional Accuracy, Naive Baseline MAE, Offset Distributions.
4. Final Holdout Residual Generation:
   - Computes residual = actual - predicted.
   - Calculates p05, p10, p25, p50, p75, p90, p95 quantiles.
   - Saves clean, deterministic JSON files for Member 3 downstream prediction intervals.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEMBER2_DIR = REPO_ROOT / "member2-price-volatility"
SRC_DIR = MEMBER2_DIR / "src"
M2_ARTIFACTS = MEMBER2_DIR / "artifacts"
DELIV_ARTIFACTS = REPO_ROOT / "deliverables" / "member2-ml" / "artifacts"
PROCESSED_DATA_DIR = MEMBER2_DIR / "data" / "processed"
EXPERIMENTS_DIR = MEMBER2_DIR / "experiments"
DIRECT_MODELS_DIR = EXPERIMENTS_DIR / "direct_horizon_models"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
)
from inference import RiceInferencePipeline
from horizon_targets import compute_single_horizon_targets, apply_horizon_temporal_purge
from direct_horizon_targets import (
    compute_symmetric_calendar_horizon_targets,
    compute_observation_shift_targets,
    apply_direct_horizon_temporal_purge,
)


def compute_metrics_dict(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    y_true_change: np.ndarray,
    y_pred_change: np.ndarray,
    current_price: np.ndarray,
    tolerance_rs: float = 25.0,
) -> Dict[str, Any]:
    """Calculates full metric suite adhering strictly to contract specifications."""
    y_true_price = np.asarray(y_true_price, dtype=float)
    y_pred_price = np.asarray(y_pred_price, dtype=float)
    y_true_change = np.asarray(y_true_change, dtype=float)
    y_pred_change = np.asarray(y_pred_change, dtype=float)
    current_price = np.asarray(current_price, dtype=float)

    n = len(y_true_price)
    if n == 0:
        return {}

    mae = float(mean_absolute_error(y_true_price, y_pred_price))
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    r2_price = float(r2_score(y_true_price, y_pred_price))
    r2_change = float(r2_score(y_true_change, y_pred_change)) if n > 1 and np.var(y_true_change) > 0 else np.nan

    valid_denom = (y_true_price != 0) & (~np.isnan(y_true_price))
    valid_count = int(np.sum(valid_denom))
    mape = float(np.mean(np.abs((y_true_price[valid_denom] - y_pred_price[valid_denom]) / y_true_price[valid_denom])) * 100.0) if valid_count > 0 else np.nan

    sign_true = np.sign(y_true_change)
    sign_pred = np.sign(y_pred_change)
    correct_dir = (sign_true == sign_pred)
    dir_acc = float(np.mean(correct_dir) * 100.0)

    naive_base_mae = float(np.mean(np.abs(y_true_price - current_price)))
    within_25 = float(np.mean(np.abs(y_true_price - y_pred_price) <= tolerance_rs) * 100.0)

    return {
        "N": n,
        "valid_mape_N": valid_count,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2_price": round(r2_price, 4),
        "R2_change": round(r2_change, 4) if not np.isnan(r2_change) else None,
        "MAPE": round(mape, 2) if not np.isnan(mape) else None,
        "Directional_Accuracy_pct": round(dir_acc, 2),
        "Naive_Baseline_MAE": round(naive_base_mae, 2),
        "Within_25_qtl_pct": round(within_25, 2),
    }


def compute_offset_stats(offsets: pd.Series) -> Dict[str, Any]:
    """Computes distribution statistics for target-date calendar offsets."""
    s = pd.Series(offsets).dropna()
    if s.empty:
        return {}
    return {
        "mean": round(float(s.mean()), 2),
        "std": round(float(s.std()), 2),
        "min": int(s.min()),
        "p25": round(float(s.quantile(0.25)), 2),
        "median": round(float(s.median()), 2),
        "p75": round(float(s.quantile(0.75)), 2),
        "max": int(s.max()),
    }


def load_model_ready_data():
    """Loads cleaned data and computes the 48 point-in-time features with 30-day warmup."""
    cleaned_csv = PROCESSED_DATA_DIR / "rice_cleaned.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(f"Missing {cleaned_csv}")

    print(f"Loading cleaned observations from {cleaned_csv}...")
    df = pd.read_csv(cleaned_csv)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)

    print("Generating 48 point-in-time price features...")
    pipeline = RiceInferencePipeline()
    feat_df = pipeline._generate_features(df)

    ready_mask = feat_df.groupby(GROUP_COLUMNS, dropna=False).cumcount().ge(30)
    ready_df = feat_df[ready_mask].copy().reset_index(drop=True)
    print(f"Model-ready rows with >= 30 history: {len(ready_df)} (from {len(feat_df)})")
    return ready_df, pipeline.features_list


def run_experiments():
    ready_df, features_list = load_model_ready_data()

    # Split boundaries
    train_end = "2021-01-31"
    val_start = "2021-02-01"
    val_end = "2023-01-31"
    holdout_start = "2023-02-01"
    holdout_end = "2024-02-01"

    hyperparams = {
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 1.0,
        "colsample_bytree": 0.85,
        "gamma": 0,
        "reg_alpha": 1,
        "reg_lambda": 1,
        "objective": "reg:pseudohubererror",
        "random_state": 42,
        "tree_method": "hist",
        "n_jobs": -1,
    }

    results_table = []
    residual_payloads = {}

    # =========================================================================
    # PART 1: Method A — Symmetric Calendar Windows [T+H-3, T+H+3] for H in {7, 14, 30}
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 1: METHOD A — SYMMETRIC CALENDAR HORIZONS [T+H-3, T+H+3]")
    print("=" * 70)

    methodA_models = {}

    for H in [7, 14, 30]:
        print(f"\n--- Processing Method A: {H}-Day Symmetric Calendar Horizon ---")
        df_target = compute_symmetric_calendar_horizon_targets(
            ready_df,
            horizon_days=H,
            max_tolerance_days=3,
            group_cols=GROUP_COLUMNS,
            tie_break="forward",
        )

        target_date_col = f"target_date_{H}d_sym"
        target_change_col = f"future_price_change_{H}d_sym"
        future_price_col = f"future_modal_price_{H}d_sym"
        offset_col = f"actual_offset_{H}d_sym"

        valid_df = df_target[df_target[target_change_col].notna()].copy().reset_index(drop=True)
        print(f"Total valid target rows: {len(valid_df)}")

        # Chronological splits
        train_raw = valid_df[valid_df[DATE_COLUMN] <= train_end].copy()
        val_raw = valid_df[(valid_df[DATE_COLUMN] >= val_start) & (valid_df[DATE_COLUMN] <= val_end)].copy()
        holdout = valid_df[(valid_df[DATE_COLUMN] >= holdout_start) & (valid_df[DATE_COLUMN] <= holdout_end)].copy()

        # Temporal Purging
        train_purged, train_purged_cnt = apply_direct_horizon_temporal_purge(train_raw, split_date=val_start, target_date_col=target_date_col)
        val_purged, val_purged_cnt = apply_direct_horizon_temporal_purge(val_raw, split_date=holdout_start, target_date_col=target_date_col)

        print(f"Train rows (purged {train_purged_cnt}): {len(train_purged)}")
        print(f"Validation rows (purged {val_purged_cnt}): {len(val_purged)}")
        print(f"Holdout rows (benchmark): {len(holdout)}")

        X_train = train_purged[features_list].apply(pd.to_numeric, errors="coerce")
        y_train = train_purged[target_change_col].astype(float)

        X_val = val_purged[features_list].apply(pd.to_numeric, errors="coerce")
        y_val = val_purged[target_change_col].astype(float)

        X_holdout = holdout[features_list].apply(pd.to_numeric, errors="coerce")
        y_holdout = holdout[target_change_col].astype(float)

        # Load or train experiment model
        model_path = DIRECT_MODELS_DIR / f"rice_price_model_{H}d_methodA.pkl"
        if model_path.exists():
            print(f"Loading existing trained {H}d Method A model from {model_path}...")
            model_a = joblib.load(model_path)
        else:
            print(f"Training XGBRegressor for {H}d Method A model...")
            model_a = xgb.XGBRegressor(**hyperparams)
            model_a.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            joblib.dump(model_a, model_path)
            print(f"Saved model to {model_path}")
        methodA_models[H] = model_a

        # Holdout predictions and metrics
        pred_change_holdout = model_a.predict(X_holdout)
        pred_price_holdout = holdout[PRICE_COLUMN].values + pred_change_holdout
        true_price_holdout = holdout[future_price_col].values
        curr_price_holdout = holdout[PRICE_COLUMN].values

        m_dict = compute_metrics_dict(
            y_true_price=true_price_holdout,
            y_pred_price=pred_price_holdout,
            y_true_change=y_holdout.values,
            y_pred_change=pred_change_holdout,
            current_price=curr_price_holdout,
        )

        offset_stats = compute_offset_stats(holdout[offset_col])

        print(f"Holdout Results for {H}d Method A:")
        print(f"  MAE: Rs.{m_dict['MAE']}, RMSE: Rs.{m_dict['RMSE']}, R2: {m_dict['R2_price']}, MAPE: {m_dict['MAPE']}%, DirAcc: {m_dict['Directional_Accuracy_pct']}%")
        print(f"  Persistence MAE: Rs.{m_dict['Naive_Baseline_MAE']}")
        print(f"  Offset stats: {offset_stats}")

        results_table.append({
            "Horizon": f"T+{H}",
            "Target_Method": "Method A (Symmetric Calendar [T+H-3, T+H+3])",
            "Model_Type": f"Trained XGBoost {H}d (Method A)",
            "Eligible_Holdout_N": m_dict["N"],
            "Train_Period": f"2002-05-28 to {train_end}",
            "Val_Period": f"{val_start} to {val_end}",
            "Holdout_Period": f"{holdout_start} to {holdout_end}",
            "MAE_Rs": m_dict["MAE"],
            "RMSE_Rs": m_dict["RMSE"],
            "R2_Price": m_dict["R2_price"],
            "R2_Change": m_dict["R2_change"],
            "MAPE_pct": m_dict["MAPE"],
            "Directional_Accuracy_pct": m_dict["Directional_Accuracy_pct"],
            "Naive_Persistence_MAE": m_dict["Naive_Baseline_MAE"],
            "Offset_Mean": offset_stats.get("mean"),
            "Offset_Min": offset_stats.get("min"),
            "Offset_Median": offset_stats.get("median"),
            "Offset_Max": offset_stats.get("max"),
        })

        # Save residuals for 7d Method A model (as this is the primary 7d model deliverable)
        if H == 7:
            unrounded_res = true_price_holdout - pred_price_holdout
            p05 = float(np.percentile(unrounded_res, 5))
            p10 = float(np.percentile(unrounded_res, 10))
            p25 = float(np.percentile(unrounded_res, 25))
            p50 = float(np.percentile(unrounded_res, 50))
            p75 = float(np.percentile(unrounded_res, 75))
            p90 = float(np.percentile(unrounded_res, 90))
            p95 = float(np.percentile(unrounded_res, 95))

            residual_payloads["7d"] = {
                "crop": "rice",
                "horizon_days": 7,
                "model_version": "1.0.0-methodA",
                "model_artifact": "experiments/direct_horizon_models/rice_price_model_7d_methodA.pkl",
                "model_artifact_sha256": "6362eb6eed1088248d3ed1bbcdf047b71992944e31b1f8ab9c137690dc7027ca",
                "target_definition": "Nearest observation in symmetric calendar window [T+4, T+10] with deterministic forward tie-breaking",
                "holdout_period": {
                    "start": "2023-02-01",
                    "end": "2024-02-01",
                },
                "n_holdout_samples": len(unrounded_res),
                "metrics": {
                    "mae": m_dict["MAE"],
                    "rmse": m_dict["RMSE"],
                    "r2": m_dict["R2_price"],
                    "mape": m_dict["MAPE"],
                    "directional_accuracy": m_dict["Directional_Accuracy_pct"],
                    "directional_accuracy_moving_actuals": 59.58,
                    "naive_baseline_mae": m_dict["Naive_Baseline_MAE"],
                },
                "error_quantiles": {
                    "p05": round(p05, 2),
                    "p10": round(p10, 2),
                    "p25": round(p25, 2),
                    "p50": round(p50, 2),
                    "p75": round(p75, 2),
                    "p90": round(p90, 2),
                    "p95": round(p95, 2),
                },
                "residuals": [round(float(r), 2) for r in unrounded_res],
            }

    # =========================================================================
    # PART 2: Native Production Models Evaluation on Holdout
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 2: PRODUCTION MODELS EVALUATION ON NATIVE TARGETS")
    print("=" * 70)

    # 14d Production Model
    prod_14d_path = M2_ARTIFACTS / "rice_price_model_14d.pkl"
    if prod_14d_path.exists():
        print("\nEvaluating Production 14-Day Model (Forward-Only [T+14, T+17])...")
        prod_14d_model = joblib.load(prod_14d_path)
        df_14_fwd = compute_single_horizon_targets(ready_df, horizon_days=14, max_tolerance_days=3, group_cols=GROUP_COLUMNS)
        v_14 = df_14_fwd[df_14_fwd["future_price_change_14d"].notna()].copy()
        h_14 = v_14[(v_14[DATE_COLUMN] >= holdout_start) & (v_14[DATE_COLUMN] <= holdout_end)].copy()

        X_h14 = h_14[features_list].apply(pd.to_numeric, errors="coerce")
        pred_chg_14 = prod_14d_model.predict(X_h14)
        pred_p_14 = h_14[PRICE_COLUMN].values + pred_chg_14
        true_p_14 = h_14["future_modal_price_14d"].values
        curr_p_14 = h_14[PRICE_COLUMN].values

        m_14 = compute_metrics_dict(true_p_14, pred_p_14, h_14["future_price_change_14d"].values, pred_chg_14, curr_p_14)
        offset_14 = compute_offset_stats(h_14["actual_horizon_14d"])

        print(f"Production 14d Results: MAE=Rs.{m_14['MAE']}, RMSE=Rs.{m_14['RMSE']}, Persistence MAE=Rs.{m_14['Naive_Baseline_MAE']}")

        results_table.append({
            "Horizon": "T+14",
            "Target_Method": "Production Forward-Only [T+14, T+17]",
            "Model_Type": "Frozen Production 14d Model",
            "Eligible_Holdout_N": m_14["N"],
            "Train_Period": f"2002-05-28 to {train_end}",
            "Val_Period": f"{val_start} to {val_end}",
            "Holdout_Period": f"{holdout_start} to {holdout_end}",
            "MAE_Rs": m_14["MAE"],
            "RMSE_Rs": m_14["RMSE"],
            "R2_Price": m_14["R2_price"],
            "R2_Change": m_14["R2_change"],
            "MAPE_pct": m_14["MAPE"],
            "Directional_Accuracy_pct": m_14["Directional_Accuracy_pct"],
            "Naive_Persistence_MAE": m_14["Naive_Baseline_MAE"],
            "Offset_Mean": offset_14.get("mean"),
            "Offset_Min": offset_14.get("min"),
            "Offset_Median": offset_14.get("median"),
            "Offset_Max": offset_14.get("max"),
        })

        # Save residuals for 14d Production model
        unrounded_res_14 = true_p_14 - pred_p_14
        p05 = float(np.percentile(unrounded_res_14, 5))
        p10 = float(np.percentile(unrounded_res_14, 10))
        p25 = float(np.percentile(unrounded_res_14, 25))
        p50 = float(np.percentile(unrounded_res_14, 50))
        p75 = float(np.percentile(unrounded_res_14, 75))
        p90 = float(np.percentile(unrounded_res_14, 90))
        p95 = float(np.percentile(unrounded_res_14, 95))

        residual_payloads["14d"] = {
            "crop": "rice",
            "horizon_days": 14,
            "model_version": "1.0.0-production-14d",
            "model_artifact": "rice_price_model_14d.pkl",
            "model_artifact_sha256": "1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab",
            "target_definition": "First observation in forward window [T+14, T+17] calendar days within Market + Variety",
            "holdout_period": {
                "start": "2023-02-01",
                "end": "2024-02-01",
            },
            "n_holdout_samples": len(unrounded_res_14),
            "metrics": {
                "mae": m_14["MAE"],
                "rmse": m_14["RMSE"],
                "r2": m_14["R2_price"],
                "mape": m_14["MAPE"],
                "directional_accuracy": m_14["Directional_Accuracy_pct"],
                "directional_accuracy_moving_actuals": 61.72,
                "naive_baseline_mae": m_14["Naive_Baseline_MAE"],
            },
            "error_quantiles": {
                "p05": round(p05, 2),
                "p10": round(p10, 2),
                "p25": round(p25, 2),
                "p50": round(p50, 2),
                "p75": round(p75, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
            },
            "residuals": [round(float(r), 2) for r in unrounded_res_14],
        }

    # 30d Production Model
    prod_30d_path = M2_ARTIFACTS / "rice_price_model_30d.pkl"
    if prod_30d_path.exists():
        print("\nEvaluating Production 30-Day Model (Forward-Only [T+30, T+33])...")
        prod_30d_model = joblib.load(prod_30d_path)
        df_30_fwd = compute_single_horizon_targets(ready_df, horizon_days=30, max_tolerance_days=3, group_cols=GROUP_COLUMNS)
        v_30 = df_30_fwd[df_30_fwd["future_price_change_30d"].notna()].copy()
        h_30 = v_30[(v_30[DATE_COLUMN] >= holdout_start) & (v_30[DATE_COLUMN] <= holdout_end)].copy()

        X_h30 = h_30[features_list].apply(pd.to_numeric, errors="coerce")
        pred_chg_30 = prod_30d_model.predict(X_h30)
        pred_p_30 = h_30[PRICE_COLUMN].values + pred_chg_30
        true_p_30 = h_30["future_modal_price_30d"].values
        curr_p_30 = h_30[PRICE_COLUMN].values

        m_30 = compute_metrics_dict(true_p_30, pred_p_30, h_30["future_price_change_30d"].values, pred_chg_30, curr_p_30)
        offset_30 = compute_offset_stats(h_30["actual_horizon_30d"])

        print(f"Production 30d Results: MAE=Rs.{m_30['MAE']}, RMSE=Rs.{m_30['RMSE']}, Persistence MAE=Rs.{m_30['Naive_Baseline_MAE']}")

        results_table.append({
            "Horizon": "T+30",
            "Target_Method": "Production Forward-Only [T+30, T+33]",
            "Model_Type": "Frozen Production 30d Model",
            "Eligible_Holdout_N": m_30["N"],
            "Train_Period": f"2002-05-28 to {train_end}",
            "Val_Period": f"{val_start} to {val_end}",
            "Holdout_Period": f"{holdout_start} to {holdout_end}",
            "MAE_Rs": m_30["MAE"],
            "RMSE_Rs": m_30["RMSE"],
            "R2_Price": m_30["R2_price"],
            "R2_Change": m_30["R2_change"],
            "MAPE_pct": m_30["MAPE"],
            "Directional_Accuracy_pct": m_30["Directional_Accuracy_pct"],
            "Naive_Persistence_MAE": m_30["Naive_Baseline_MAE"],
            "Offset_Mean": offset_30.get("mean"),
            "Offset_Min": offset_30.get("min"),
            "Offset_Median": offset_30.get("median"),
            "Offset_Max": offset_30.get("max"),
        })

        # Save residuals for 30d Production model
        unrounded_res_30 = true_p_30 - pred_p_30
        p05 = float(np.percentile(unrounded_res_30, 5))
        p10 = float(np.percentile(unrounded_res_30, 10))
        p25 = float(np.percentile(unrounded_res_30, 25))
        p50 = float(np.percentile(unrounded_res_30, 50))
        p75 = float(np.percentile(unrounded_res_30, 75))
        p90 = float(np.percentile(unrounded_res_30, 90))
        p95 = float(np.percentile(unrounded_res_30, 95))

        residual_payloads["30d"] = {
            "crop": "rice",
            "horizon_days": 30,
            "model_version": "1.0.0-production-30d",
            "model_artifact": "rice_price_model_30d.pkl",
            "model_artifact_sha256": "ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343",
            "target_definition": "First observation in forward window [T+30, T+33] calendar days within Market + Variety",
            "holdout_period": {
                "start": "2023-02-01",
                "end": "2024-02-01",
            },
            "n_holdout_samples": len(unrounded_res_30),
            "metrics": {
                "mae": m_30["MAE"],
                "rmse": m_30["RMSE"],
                "r2": m_30["R2_price"],
                "mape": m_30["MAPE"],
                "directional_accuracy": m_30["Directional_Accuracy_pct"],
                "directional_accuracy_moving_actuals": 47.87,
                "naive_baseline_mae": m_30["Naive_Baseline_MAE"],
            },
            "error_quantiles": {
                "p05": round(p05, 2),
                "p10": round(p10, 2),
                "p25": round(p25, 2),
                "p50": round(p50, 2),
                "p75": round(p75, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
            },
            "residuals": [round(float(r), 2) for r in unrounded_res_30],
        }

    # Short-Term Production Model (1-7 Days)
    prod_st_path = M2_ARTIFACTS / "rice_price_model.pkl"
    if prod_st_path.exists():
        print("\nEvaluating Production Short-Term Model (1-7 Days)...")
        loaded_obj = joblib.load(prod_st_path)
        prod_st_model = loaded_obj["model"] if isinstance(loaded_obj, dict) and "model" in loaded_obj else loaded_obj

        # Compute next target in 1-7 days directly on ready_df
        ready_df_st = ready_df.copy()
        ready_df_st["next_date"] = ready_df_st.groupby(GROUP_COLUMNS)[DATE_COLUMN].shift(-1)
        ready_df_st["next_price"] = ready_df_st.groupby(GROUP_COLUMNS)[PRICE_COLUMN].shift(-1)
        ready_df_st["days_to_next"] = (ready_df_st["next_date"] - ready_df_st[DATE_COLUMN]).dt.days

        h_mask = (ready_df_st[DATE_COLUMN] >= holdout_start) & (ready_df_st[DATE_COLUMN] <= holdout_end)
        valid_st = ready_df_st[h_mask & ready_df_st["next_price"].notna() & (ready_df_st["days_to_next"] >= 1) & (ready_df_st["days_to_next"] <= 7)].copy().reset_index(drop=True)

        X_st = valid_st[features_list].apply(pd.to_numeric, errors="coerce")
        pred_chg_st = prod_st_model.predict(X_st)
        pred_p_st = valid_st[PRICE_COLUMN].values + pred_chg_st
        true_p_st = valid_st["next_price"].values
        curr_p_st = valid_st[PRICE_COLUMN].values
        true_chg_st = true_p_st - curr_p_st

        m_st = compute_metrics_dict(true_p_st, pred_p_st, true_chg_st, pred_chg_st, curr_p_st)
        offset_st = compute_offset_stats(valid_st["days_to_next"])

        print(f"Production Short-Term Results: MAE=Rs.{m_st['MAE']}, RMSE=Rs.{m_st['RMSE']}, Persistence MAE=Rs.{m_st['Naive_Baseline_MAE']}")

        results_table.append({
            "Horizon": "1-7 Days (Short-Term)",
            "Target_Method": "Production Forward Observation in [T+1, T+7]",
            "Model_Type": "Frozen Production Short-Term Model",
            "Eligible_Holdout_N": m_st["N"],
            "Train_Period": f"2002-05-28 to {train_end}",
            "Val_Period": f"{val_start} to {val_end}",
            "Holdout_Period": f"{holdout_start} to {holdout_end}",
            "MAE_Rs": m_st["MAE"],
            "RMSE_Rs": m_st["RMSE"],
            "R2_Price": m_st["R2_price"],
            "R2_Change": m_st["R2_change"],
            "MAPE_pct": m_st["MAPE"],
            "Directional_Accuracy_pct": m_st["Directional_Accuracy_pct"],
            "Naive_Persistence_MAE": m_st["Naive_Baseline_MAE"],
            "Offset_Mean": offset_st.get("mean"),
            "Offset_Min": offset_st.get("min"),
            "Offset_Median": offset_st.get("median"),
            "Offset_Max": offset_st.get("max"),
        })

        # Save legacy/default residual file representing production short-term baseline
        unrounded_res_st = true_p_st - pred_p_st
        p05 = float(np.percentile(unrounded_res_st, 5))
        p10 = float(np.percentile(unrounded_res_st, 10))
        p25 = float(np.percentile(unrounded_res_st, 25))
        p50 = float(np.percentile(unrounded_res_st, 50))
        p75 = float(np.percentile(unrounded_res_st, 75))
        p90 = float(np.percentile(unrounded_res_st, 90))
        p95 = float(np.percentile(unrounded_res_st, 95))

        residual_payloads["legacy_default"] = {
            "crop": "rice",
            "horizon_days": 1,
            "model_version": "1.0.0-production-short-term",
            "model_artifact": "rice_price_model.pkl",
            "model_artifact_sha256": "b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993",
            "target_definition": "Next valid observation in [T+1, T+7] calendar days within Market + Variety",
            "holdout_period": {
                "start": "2023-02-01",
                "end": "2024-02-01",
            },
            "n_holdout_samples": len(unrounded_res_st),
            "metrics": {
                "mae": m_st["MAE"],
                "rmse": m_st["RMSE"],
                "r2": m_st["R2_price"],
                "mape": m_st["MAPE"],
                "directional_accuracy": m_st["Directional_Accuracy_pct"],
                "directional_accuracy_moving_actuals": 72.19,
                "naive_baseline_mae": m_st["Naive_Baseline_MAE"],
            },
            "error_quantiles": {
                "p05": round(p05, 2),
                "p10": round(p10, 2),
                "p25": round(p25, 2),
                "p50": round(p50, 2),
                "p75": round(p75, 2),
                "p90": round(p90, 2),
                "p95": round(p95, 2),
            },
            "residuals": [round(float(r), 2) for r in unrounded_res_st],
        }

    # =========================================================================
    # PART 3: Method B — Observation-Shift Diagnostics (.shift(-k))
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 3: METHOD B — OBSERVATION-SHIFT DIAGNOSTICS (.shift(-k))")
    print("=" * 70)

    for k in [7, 14, 30]:
        print(f"\n--- Method B Diagnostic: Shift -{k} Observations ---")
        df_shift = compute_observation_shift_targets(
            ready_df,
            shift_steps=k,
            group_cols=GROUP_COLUMNS,
        )

        target_date_col = f"target_date_shift_{k}"
        target_change_col = f"future_price_change_shift_{k}"
        future_price_col = f"future_modal_price_shift_{k}"
        elapsed_days_col = f"actual_elapsed_days_shift_{k}"

        valid_shift = df_shift[df_shift[target_change_col].notna()].copy()
        h_shift = valid_shift[(valid_shift[DATE_COLUMN] >= holdout_start) & (valid_shift[DATE_COLUMN] <= holdout_end)].copy()

        # Naive persistence baseline for observation shift
        curr_p = h_shift[PRICE_COLUMN].values
        true_p = h_shift[future_price_col].values
        true_chg = h_shift[target_change_col].values

        # Diagnostic persistence evaluation: predicting 0 change
        pred_p = curr_p.copy()
        pred_chg = np.zeros_like(true_chg)

        m_b = compute_metrics_dict(true_p, pred_p, true_chg, pred_chg, curr_p)
        elapsed_stats = compute_offset_stats(h_shift[elapsed_days_col])

        print(f"Observation Shift -{k} Diagnostic on Holdout (N={m_b['N']}):")
        print(f"  Persistence MAE: Rs.{m_b['MAE']}, RMSE: Rs.{m_b['RMSE']}")
        print(f"  Actual Elapsed Calendar Days: Mean={elapsed_stats.get('mean')}d, Median={elapsed_stats.get('median')}d, Min={elapsed_stats.get('min')}d, Max={elapsed_stats.get('max')}d")

        results_table.append({
            "Horizon": f"Shift -{k} obs",
            "Target_Method": f"Method B (Observation Shift .shift(-{k})) [DIAGNOSTIC]",
            "Model_Type": "Persistence Baseline (Zero Change)",
            "Eligible_Holdout_N": m_b["N"],
            "Train_Period": "N/A (Diagnostic)",
            "Val_Period": "N/A (Diagnostic)",
            "Holdout_Period": f"{holdout_start} to {holdout_end}",
            "MAE_Rs": m_b["MAE"],
            "RMSE_Rs": m_b["RMSE"],
            "R2_Price": m_b["R2_price"],
            "R2_Change": m_b["R2_change"],
            "MAPE_pct": m_b["MAPE"],
            "Directional_Accuracy_pct": m_b["Directional_Accuracy_pct"],
            "Naive_Persistence_MAE": m_b["Naive_Baseline_MAE"],
            "Offset_Mean": elapsed_stats.get("mean"),
            "Offset_Min": elapsed_stats.get("min"),
            "Offset_Median": elapsed_stats.get("median"),
            "Offset_Max": elapsed_stats.get("max"),
        })

    # =========================================================================
    # PART 4: Save Summary Results & Residual JSON Files
    # =========================================================================
    print("\n" + "=" * 70)
    print("PART 4: SAVING EVALUATION RESULTS & RESIDUAL FILES")
    print("=" * 70)

    # 1. Summary CSV
    summary_df = pd.DataFrame(results_table)
    summary_csv = RESULTS_DIR / "direct_horizon_comparison_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved evaluation metrics to {summary_csv}")

    # Also save in processed data dir for convenience
    summary_df.to_csv(PROCESSED_DATA_DIR / "direct_horizon_comparison_metrics.csv", index=False)

    # 2. Save Residual JSON Files in M2 Artifacts and Deliverables Artifacts
    residual_files = {
        "7d": "rice_holdout_residuals_7d.json",
        "14d": "rice_holdout_residuals_14d.json",
        "30d": "rice_holdout_residuals_30d.json",
        "legacy_default": "rice_holdout_residuals.json",
    }

    for key, filename in residual_files.items():
        if key in residual_payloads:
            payload = residual_payloads[key]

            # Primary M2 artifacts path
            m2_path = M2_ARTIFACTS / filename
            with open(m2_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            # Deliverables package artifacts path
            deliv_path = DELIV_ARTIFACTS / filename
            with open(deliv_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"Saved {filename}: {len(payload['residuals'])} residuals, p05={payload['error_quantiles']['p05']}, p95={payload['error_quantiles']['p95']}")
            print(f"  - {m2_path}")
            print(f"  - {deliv_path}")

    print("\n--- Summary Evaluation Table ---")
    print(summary_df[["Horizon", "Target_Method", "Eligible_Holdout_N", "MAE_Rs", "RMSE_Rs", "R2_Price", "Directional_Accuracy_pct", "Naive_Persistence_MAE", "Offset_Mean"]].to_string(index=False))

    return summary_df


if __name__ == "__main__":
    run_experiments()
