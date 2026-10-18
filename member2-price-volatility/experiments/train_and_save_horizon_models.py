"""
AgroVision — Member 2: Train and Evaluate 14-Day and 30-Day Rice Price Models

Trains two independent XGBoost regressors for direct 14-calendar-day and
30-calendar-day rice modal price forecasting.

Key Design Elements:
1. Target: Future price change relative to forecast origin:
   future_price_change_Hd = future_modal_price - origin_modal_price.
   Matches the first valid observation in [t + H, t + H + 3] calendar days.
2. Temporal Purging: Strictly drops rows whose future target dates reach across
   chronological partition boundaries (Train -> Validation -> Holdout).
3. Exact 48 Features: Reuses the frozen production 48-feature definitions and order.
4. Separate Artifacts:
   - artifacts/rice_price_model_14d.pkl
   - artifacts/rice_price_model_14d_config.json
   - artifacts/rice_price_model_30d.pkl
   - artifacts/rice_price_model_30d_config.json
5. Safety: NEVER touches or overwrites the frozen 1-7 day production model artifacts.
"""

import sys
import os
import json
import time
from pathlib import Path
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

EXPERIMENTS_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = EXPERIMENTS_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"
ARTIFACTS_DIR = MEMBER2_DIR / "artifacts"
PROCESSED_DATA_DIR = MEMBER2_DIR / "data" / "processed"

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


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Safe MAPE calculation excluding exact zero denominators."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    valid = y_true != 0
    if not np.any(valid):
        return np.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)


def compute_metrics(y_true_price: np.ndarray, y_pred_price: np.ndarray, tolerance_rs: float = 25.0) -> dict:
    """Computes comprehensive regression metrics for price levels."""
    y_true_price = np.asarray(y_true_price, dtype=float)
    y_pred_price = np.asarray(y_pred_price, dtype=float)

    mae = float(mean_absolute_error(y_true_price, y_pred_price))
    rmse = float(np.sqrt(mean_squared_error(y_true_price, y_pred_price)))
    r2 = float(r2_score(y_true_price, y_pred_price))
    mape = calculate_mape(y_true_price, y_pred_price)
    acc_100_mape = float(100.0 - mape) if not np.isnan(mape) else np.nan
    tolerance_acc = float(np.mean(np.abs(y_true_price - y_pred_price) <= tolerance_rs) * 100.0)

    return {
        "N": len(y_true_price),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "100_minus_MAPE": round(acc_100_mape, 2),
        "Within_25_qtl": round(tolerance_acc, 2),
    }


def prepare_horizon_data(cleaned_csv_path: Path):
    """Loads cleaned rice observations and generates 48 features and eligibility."""
    print(f"Loading cleaned dataset from {cleaned_csv_path}...")
    df = pd.read_csv(cleaned_csv_path)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)

    print("Generating 48 point-in-time features using RiceInferencePipeline...")
    pipe = RiceInferencePipeline()
    feat_df = pipe._generate_features(df)

    # 30-observation history requirement: drop early sparse rows per group
    print("Applying 30-observation history requirement...")
    ready_mask = feat_df.groupby(GROUP_COLUMNS, dropna=False).cumcount().ge(30)
    ready_df = feat_df[ready_mask].copy().reset_index(drop=True)
    print(f"Model-ready rows remaining: {len(ready_df)} (from {len(feat_df)})")

    return ready_df, pipe.features_list


def train_and_eval_single_horizon(
    ready_df: pd.DataFrame,
    features_list: list,
    horizon_days: int,
    max_tolerance_days: int = 3,
):
    """Trains and evaluates a single horizon model (14d or 30d)."""
    h_str = f"{horizon_days}d"
    target_col = f"future_price_change_{h_str}"
    target_date_col = f"target_date_{h_str}"
    future_price_col = f"future_modal_price_{h_str}"

    print(f"\n{'='*70}\nTRAINING AND EVALUATING {horizon_days}-DAY HORIZON MODEL\n{'='*70}")
    
    # 1. Compute targets
    df_h = compute_single_horizon_targets(
        ready_df,
        horizon_days=horizon_days,
        max_tolerance_days=max_tolerance_days,
        group_cols=GROUP_COLUMNS,
    )

    # Filter to rows with valid target
    valid_h = df_h[df_h[target_col].notna()].copy().reset_index(drop=True)
    print(f"Rows with valid {horizon_days}-day target in tolerance window: {len(valid_h)}")

    # 2. Chronological split dates
    train_end = "2021-01-31"
    val_start = "2021-02-01"
    val_end = "2023-01-31"
    holdout_start = "2023-02-01"
    holdout_end = "2024-02-01"

    # Raw partition assignments
    train_raw = valid_h[valid_h[DATE_COLUMN] <= train_end].copy()
    val_raw = valid_h[(valid_h[DATE_COLUMN] >= val_start) & (valid_h[DATE_COLUMN] <= val_end)].copy()
    holdout = valid_h[(valid_h[DATE_COLUMN] >= holdout_start) & (valid_h[DATE_COLUMN] <= holdout_end)].copy()

    # 3. Apply Temporal Purging: target_date must not cross into subsequent partition
    train_purged, train_purged_cnt = apply_horizon_temporal_purge(train_raw, split_date=val_start, horizon_days=horizon_days)
    val_purged, val_purged_cnt = apply_horizon_temporal_purge(val_raw, split_date=holdout_start, horizon_days=horizon_days)

    print(f"Train split: {len(train_purged)} rows (purged {train_purged_cnt} rows reaching >= {val_start})")
    print(f"Validation split: {len(val_purged)} rows (purged {val_purged_cnt} rows reaching >= {holdout_start})")
    print(f"Holdout split: {len(holdout)} rows (unpurged benchmark partition)")

    # 4. Feature matrices
    X_train = train_purged[features_list].apply(pd.to_numeric, errors="coerce")
    y_train = train_purged[target_col].astype(float)

    X_val = val_purged[features_list].apply(pd.to_numeric, errors="coerce")
    y_val = val_purged[target_col].astype(float)

    X_holdout = holdout[features_list].apply(pd.to_numeric, errors="coerce")
    y_holdout = holdout[target_col].astype(float)

    # 5. Fit XGBoost Regressor using controlled baseline parameters
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

    print(f"Training XGBRegressor for {horizon_days}-day horizon...")
    t0 = time.time()
    model = xgb.XGBRegressor(**hyperparams)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"Model trained in {round(time.time() - t0, 2)} s")

    # 6. Evaluate predictions (reconstruct price = current_price + predicted_change)
    pred_change_val = model.predict(X_val)
    pred_price_val = val_purged[PRICE_COLUMN].values + pred_change_val
    true_price_val = val_purged[future_price_col].values
    base_price_val = val_purged[PRICE_COLUMN].values  # Persistence baseline

    pred_change_holdout = model.predict(X_holdout)
    pred_price_holdout = holdout[PRICE_COLUMN].values + pred_change_holdout
    true_price_holdout = holdout[future_price_col].values
    base_price_holdout = holdout[PRICE_COLUMN].values  # Persistence baseline

    val_metrics = compute_metrics(true_price_val, pred_price_val)
    val_baseline = compute_metrics(true_price_val, base_price_val)

    holdout_metrics = compute_metrics(true_price_holdout, pred_price_holdout)
    holdout_baseline = compute_metrics(true_price_holdout, base_price_holdout)

    print(f"\n--- {horizon_days}-Day Holdout Results (N={holdout_metrics['N']}) ---")
    print(f"  Model       : MAE=Rs.{holdout_metrics['MAE']}, RMSE=Rs.{holdout_metrics['RMSE']}, R2={holdout_metrics['R2']}, MAPE={holdout_metrics['MAPE']}%, Within +/-Rs.25={holdout_metrics['Within_25_qtl']}%")
    print(f"  Persistence : MAE=Rs.{holdout_baseline['MAE']}, RMSE=Rs.{holdout_baseline['RMSE']}, R2={holdout_baseline['R2']}, MAPE={holdout_baseline['MAPE']}%, Within +/-Rs.25={holdout_baseline['Within_25_qtl']}%")

    # 7. Save separate artifacts
    model_artifact_path = ARTIFACTS_DIR / f"rice_price_model_{h_str}.pkl"
    config_artifact_path = ARTIFACTS_DIR / f"rice_price_model_{h_str}_config.json"

    joblib.dump(model, model_artifact_path)
    print(f"Saved model artifact: {model_artifact_path}")

    config_data = {
        "commodity": "rice",
        "horizon_days": horizon_days,
        "max_tolerance_days": max_tolerance_days,
        "target": f"future_price_change_{h_str} = future_modal_price_{h_str} - current_modal_price",
        "target_date_convention": f"First observation in [t + {horizon_days}, t + {horizon_days + max_tolerance_days}] calendar days within Market + Variety",
        "features": features_list,
        "hyperparameters": hyperparams,
        "train_period": ["2002-05-28", train_end],
        "validation_period": [val_start, val_end],
        "final_holdout_period": [holdout_start, holdout_end],
        "temporal_purge": {
            "train_purged_rows": train_purged_cnt,
            "val_purged_rows": val_purged_cnt,
            "train_cutoff": str(train_purged[DATE_COLUMN].max().date()),
            "val_cutoff": str(val_purged[DATE_COLUMN].max().date()),
        },
        "validation_metrics": {
            "model": val_metrics,
            "baseline": val_baseline,
        },
        "holdout_benchmark": {
            "model": holdout_metrics,
            "baseline": holdout_baseline,
        },
    }

    with open(config_artifact_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print(f"Saved config artifact: {config_artifact_path}")

    return {
        "horizon_days": horizon_days,
        "holdout_metrics": holdout_metrics,
        "holdout_baseline": holdout_baseline,
        "val_metrics": val_metrics,
        "val_baseline": val_baseline,
    }


def main():
    cleaned_csv = PROCESSED_DATA_DIR / "rice_cleaned.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(f"Cleaned rice dataset not found at {cleaned_csv}")

    ready_df, features_list = prepare_horizon_data(cleaned_csv)

    results_14d = train_and_eval_single_horizon(ready_df, features_list, horizon_days=14)
    results_30d = train_and_eval_single_horizon(ready_df, features_list, horizon_days=30)

    # Save summary CSV
    summary_rows = [
        {
            "Horizon": "14-Day",
            "Model": "Tuned XGBoost 14d",
            **results_14d["holdout_metrics"],
        },
        {
            "Horizon": "14-Day",
            "Model": "Persistence Baseline",
            **results_14d["holdout_baseline"],
        },
        {
            "Horizon": "30-Day",
            "Model": "Tuned XGBoost 30d",
            **results_30d["holdout_metrics"],
        },
        {
            "Horizon": "30-Day",
            "Model": "Persistence Baseline",
            **results_30d["holdout_baseline"],
        },
    ]

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PROCESSED_DATA_DIR / "rice_horizon_14d_30d_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary metrics to {summary_csv}")
    print("\n--- Final Summary Table ---")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
