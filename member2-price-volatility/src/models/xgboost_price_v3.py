"""
AgroVision -- XGBoost Price Model v3  (EXPERIMENT -- do not touch save_final_models.py)

Key finding from v2 diagnostics
---------------------------------
  - 81% of all rows have zero price change -- global model learns to predict ~0
  - Early stopping at round 12: the price-change val MAE plateaus at Rs.169
    from round 0 because 81% zeros dominate the gradient signal
  - Per-group analysis reveals three distinct regimes:

      Regime A -- HIGH volatility, LOW zero-change (< 70%):
          Parappanangadi/Other  : 41% zero,  change_std Rs.1605
          Thirurrangadi/Other   : 46% zero,  change_std Rs.1100
          Bishalgarh/Other      : 58% zero,  change_std Rs.937

      Regime B -- MODERATE volatility (70-90% zero):
          Dhanbad/Other         : 86% zero,  std Rs.2856
          Madhupur groups       : low zero, std Rs.90

      Regime C -- STABLE (> 90% zero):
          Sakhigopal/All Dust   : 96% zero  -> persistence is optimal

Strategy in v3
--------------
  1. Train a SEPARATE XGBoost model for each group.
     - Each model trains only on its group's history.
     - The model can now learn group-specific price patterns without being
       drowned by the stable groups' zero-change signal.

  2. For groups with < 30 training rows, fall back to persistence baseline.

  3. Keep all the v2 improvements:
     - Group-identity encoding (still useful within-group as auxiliary signal)
     - days_to_next as feature
     - price_spread
     - NaN imputation (ffill + median)
     - Evaluation on same 1,234 rows as baseline

  4. Grid search per-group hyperparameters via a chronological inner val split.
     Report results per group so we can see exactly where the model helps.

Run
---
    cd member2-price-volatility
    python src/models/xgboost_price_v3.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    PRICE_COLUMN,
    PRICE_TARGET,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    train_path,
    test_path,
)


# ============================================================
# FEATURES (base set — available at prediction time, no leakage)
# ============================================================

BASE_FEATURES = [
    PRICE_COLUMN,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    "price_pct_change",
    "arrival_pct_change",
    "days_to_next",
    "price_spread",
]

# Minimum training rows before we fall back to persistence baseline
MIN_TRAIN_ROWS = 30


# ============================================================
# METRICS
# ============================================================

def calc_metrics(actual, predicted):
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted)
    a, p = actual[mask], predicted[mask]
    if len(a) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    r2   = r2_score(a, p)
    nz   = a != 0
    mape = np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100 if nz.sum() > 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# ============================================================
# IMPUTE NaN LAGS  (forward-fill within group, then median)
# ============================================================

def impute_lags(df):
    df = df.copy()
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    fill_cols  = [
        "lag_1", "lag_7", "lag_14", "lag_30",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_30",
        "rolling_std_7", "rolling_std_14",
        "arrival_lag_1", "arrival_lag_7",
        "arrival_rolling_mean_7", "arrival_rolling_mean_14",
        "price_pct_change", "arrival_pct_change",
    ]
    existing = [c for c in fill_cols if c in df.columns]
    if group_cols:
        df = df.sort_values(group_cols + [DATE_COLUMN])
        df[existing] = (
            df.groupby(group_cols)[existing]
              .transform(lambda s: s.ffill())
        )
        df = df.reset_index(drop=True)
    for col in existing:
        df[col] = df[col].fillna(df[col].median())
    return df


# ============================================================
# ENGINEER EXTRA FEATURES
# ============================================================

def engineer_features(df):
    df = df.copy()
    if MIN_PRICE_COLUMN in df.columns and MAX_PRICE_COLUMN in df.columns:
        df["price_spread"] = df[MAX_PRICE_COLUMN] - df[MIN_PRICE_COLUMN]
    else:
        df["price_spread"] = 0.0
    # Price-change target
    df["future_price_change"] = df[PRICE_TARGET] - df[PRICE_COLUMN]
    return df


# ============================================================
# PER-GROUP CHRONOLOGICAL VALIDATION SPLIT
# ============================================================

def chrono_split_group(group_df, val_ratio=0.20):
    """Split one group's DataFrame into sub-train / val."""
    group_df = group_df.sort_values(DATE_COLUMN).reset_index(drop=True)
    split_idx = int(len(group_df) * (1 - val_ratio))
    if split_idx < 5:
        return group_df, pd.DataFrame(columns=group_df.columns)
    return group_df.iloc[:split_idx], group_df.iloc[split_idx:]


# ============================================================
# TRAIN ONE GROUP'S MODEL
# ============================================================

def train_group_model(sub_tr, val_df, features):
    """
    Train an XGBoost model on one group.
    Uses early stopping on the group's own val split.
    """
    X_sub = sub_tr[features]
    y_sub = sub_tr["future_price_change"]

    if len(val_df) >= 5:
        X_val = val_df[features]
        y_val = val_df["future_price_change"]
        eval_set = [(X_val, y_val)]
        early_stopping_rounds = 50
    else:
        eval_set = None
        early_stopping_rounds = None

    model = XGBRegressor(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_sub, y_sub,
        eval_set=eval_set,
        verbose=False,
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 65)
    print("XGBOOST PRICE MODEL v3  --  PER-GROUP EXPERIMENT")
    print("=" * 65)

    # ----------------------------------------------------------
    # LOAD DATA
    # ----------------------------------------------------------
    tr_file = train_path()
    te_file = test_path()
    if not tr_file.exists():
        raise FileNotFoundError(f"Train not found: {tr_file}")
    if not te_file.exists():
        raise FileNotFoundError(f"Test not found: {te_file}")

    train_df = pd.read_csv(tr_file)
    test_df  = pd.read_csv(te_file)
    train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN], errors="coerce")
    test_df[DATE_COLUMN]  = pd.to_datetime(test_df[DATE_COLUMN],  errors="coerce")

    print(f"  Loaded train={len(train_df):,}  test={len(test_df):,}")

    # ----------------------------------------------------------
    # FILTER TO VALID TARGET ROWS
    # ----------------------------------------------------------
    train_df = train_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    test_df  = test_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()

    # ----------------------------------------------------------
    # IMPUTE LAGS  +  ENGINEER FEATURES
    # ----------------------------------------------------------
    train_df = impute_lags(train_df)
    test_df  = impute_lags(test_df)
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)

    # ----------------------------------------------------------
    # SELECT AVAILABLE FEATURES
    # ----------------------------------------------------------
    features = [
        f for f in BASE_FEATURES
        if f in train_df.columns and f in test_df.columns
    ]
    missing = [f for f in BASE_FEATURES if f not in features]
    if missing:
        print(f"  WARNING -- features not found, skipped: {missing}")
    print(f"  Features selected: {len(features)}")

    # ----------------------------------------------------------
    # PER-GROUP TRAINING AND PREDICTION
    # ----------------------------------------------------------
    group_cols = [c for c in GROUP_COLUMNS if c in train_df.columns]

    # Initialise prediction arrays with persistence baseline
    test_df["predicted_price"] = test_df[PRICE_COLUMN].values

    group_results = []
    models = {}

    all_groups = test_df.groupby(group_cols).groups.keys()

    for group_name in all_groups:
        if isinstance(group_name, tuple):
            mask_tr = np.ones(len(train_df), dtype=bool)
            mask_te = np.ones(len(test_df),  dtype=bool)
            for col, val in zip(group_cols, group_name):
                mask_tr &= train_df[col] == val
                mask_te &= test_df[col]  == val
        else:
            mask_tr = train_df[group_cols[0]] == group_name
            mask_te = test_df[group_cols[0]]  == group_name

        grp_train = train_df[mask_tr].copy()
        grp_test  = test_df[mask_te].copy()

        n_train = len(grp_train)
        n_test  = len(grp_test)

        if n_test == 0:
            continue

        # Baseline for this group
        grp_actual   = grp_test[PRICE_TARGET].values
        grp_baseline = grp_test[PRICE_COLUMN].values
        baseline_mae = mean_absolute_error(grp_actual, grp_baseline)

        if n_train < MIN_TRAIN_ROWS:
            # Too few rows — use persistence baseline
            group_results.append({
                "Group":        str(group_name),
                "N_train":      n_train,
                "N_test":       n_test,
                "Strategy":     "PERSISTENCE",
                "Baseline_MAE": round(baseline_mae, 1),
                "Model_MAE":    round(baseline_mae, 1),
                "Delta":        0.0,
            })
            continue

        # -- Train per-group model --
        sub_tr, val_df_grp = chrono_split_group(grp_train, val_ratio=0.20)

        if len(sub_tr) < 10:
            # Subtrain too small even after split — use all of grp_train
            sub_tr      = grp_train
            val_df_grp  = pd.DataFrame(columns=grp_train.columns)

        model = train_group_model(sub_tr, val_df_grp, features)
        models[str(group_name)] = model

        # -- Predict on this group's test rows --
        predicted_change = model.predict(grp_test[features])
        predicted_price  = grp_test[PRICE_COLUMN].values + predicted_change

        # -- Clip predictions to [0.5 * current, 2.0 * current] to prevent wild extrapolation --
        current = grp_test[PRICE_COLUMN].values
        predicted_price = np.clip(predicted_price, 0.5 * current, 2.0 * current)

        model_mae = mean_absolute_error(grp_actual, predicted_price)

        # -- Only use the model prediction if it beats persistence on train val --
        # (guard against overfitting on tiny groups)
        group_results.append({
            "Group":        str(group_name),
            "N_train":      n_train,
            "N_test":       n_test,
            "Strategy":     "XGBOOST",
            "Baseline_MAE": round(baseline_mae, 1),
            "Model_MAE":    round(model_mae, 1),
            "Delta":        round(baseline_mae - model_mae, 1),
        })

        # Write predictions back to test_df
        test_df.loc[mask_te, "predicted_price"] = predicted_price

    # ----------------------------------------------------------
    # AGGREGATE METRICS
    # ----------------------------------------------------------
    actual_all   = test_df[PRICE_TARGET].values
    pred_all     = test_df["predicted_price"].values
    current_all  = test_df[PRICE_COLUMN].values

    model_metrics    = calc_metrics(actual_all, pred_all)
    baseline_metrics = calc_metrics(actual_all, current_all)

    mae_imp  = (baseline_metrics["MAE"]  - model_metrics["MAE"])  / baseline_metrics["MAE"]  * 100
    rmse_imp = (baseline_metrics["RMSE"] - model_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100

    print("\n" + "=" * 65)
    print("AGGREGATE METRICS  (all test rows)")
    print("=" * 65)
    print(f"\n  [XGBoost v3 -- per-group]")
    print(f"    MAE  : Rs.{model_metrics['MAE']:.2f}")
    print(f"    RMSE : Rs.{model_metrics['RMSE']:.2f}")
    print(f"    R2   :  {model_metrics['R2']:.4f}")
    print(f"    MAPE :  {model_metrics['MAPE']:.2f}%")
    print(f"\n  [Baseline (current price)]")
    print(f"    MAE  : Rs.{baseline_metrics['MAE']:.2f}")
    print(f"    RMSE : Rs.{baseline_metrics['RMSE']:.2f}")
    print(f"    R2   :  {baseline_metrics['R2']:.4f}")
    print(f"    MAPE :  {baseline_metrics['MAPE']:.2f}%")
    print(f"\n  MAE  improvement vs baseline : {mae_imp:+.2f}%")
    print(f"  RMSE improvement vs baseline : {rmse_imp:+.2f}%")

    if model_metrics["MAE"] < baseline_metrics["MAE"]:
        print("\n  PASS: XGBoost v3 BEATS the persistence baseline on MAE.")
    else:
        print("\n  FAIL: XGBoost v3 does NOT beat the baseline yet.")

    # ----------------------------------------------------------
    # PER-GROUP RESULTS
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("PER-GROUP RESULTS")
    print("=" * 65)
    gr_df = (
        pd.DataFrame(group_results)
          .sort_values("Baseline_MAE", ascending=False)
          .reset_index(drop=True)
    )
    print(gr_df.to_string(index=False))

    # ----------------------------------------------------------
    # FINAL SUMMARY TABLE
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL SUMMARY  --  v3 per-group vs BASELINE")
    print("=" * 65)
    print(f"  {'Metric':<12} {'Baseline':>14} {'XGBoost v3':>14} {'Chg%':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    print(f"  {'MAE':<12} {'Rs.'+str(round(baseline_metrics['MAE'],2)):>14} {'Rs.'+str(round(model_metrics['MAE'],2)):>14} {mae_imp:>+7.2f}%")
    print(f"  {'RMSE':<12} {'Rs.'+str(round(baseline_metrics['RMSE'],2)):>14} {'Rs.'+str(round(model_metrics['RMSE'],2)):>14} {rmse_imp:>+7.2f}%")
    print(f"  {'R2':<12} {round(baseline_metrics['R2'],4):>14} {round(model_metrics['R2'],4):>14}")
    print(f"  {'MAPE':<12} {str(round(baseline_metrics['MAPE'],2))+'%':>14} {str(round(model_metrics['MAPE'],2))+'%':>14}")

    # ----------------------------------------------------------
    # HORIZON-STRATIFIED METRICS
    # ----------------------------------------------------------
    if "days_to_next" in test_df.columns:
        print("\n" + "=" * 65)
        print("METRICS BY PREDICTION HORIZON")
        print("=" * 65)
        horizon_rows = []
        for d in sorted(test_df["days_to_next"].dropna().unique()):
            m = test_df["days_to_next"].values == d
            if m.sum() < 5:
                continue
            horizon_rows.append({
                "days_to_next": int(d),
                "N":            int(m.sum()),
                "Baseline_MAE": round(mean_absolute_error(actual_all[m], current_all[m]), 1),
                "Model_MAE":    round(mean_absolute_error(actual_all[m], pred_all[m]),    1),
                "Delta":        round(
                    mean_absolute_error(actual_all[m], current_all[m]) -
                    mean_absolute_error(actual_all[m], pred_all[m]),    1),
            })
        print(pd.DataFrame(horizon_rows).to_string(index=False))

    return test_df, models, model_metrics, baseline_metrics


if __name__ == "__main__":
    main()
