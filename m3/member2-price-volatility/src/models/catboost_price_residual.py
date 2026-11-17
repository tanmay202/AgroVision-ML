"""
AgroVision -- CatBoost Residual Price Model  (EXPERIMENT)

Motivation
----------
  XGBoost v2/v3 and blend experiments could not meaningfully beat the
  current-price persistence baseline (MAE Rs.279.04).  Two hypotheses:

  1. Tree-based gradient boosters with integer-coded categoricals cannot
     learn the per-market/per-variety intercept well -- they must spend
     splits rediscovering it.

  2. `days_to_next` leaks future information.

  CatBoost natively handles categorical features via ordered target
  statistics (TS), letting Market Name x Variety interact with the
  continuous features without explicit encoding.

Design
------
  Target:     residual = future_modal_price  -  current_price
  Reconstruct:  predicted_price = current_price + predicted_residual

  Features:
    - Current modal / min / max price, price_spread
    - Lag prices (1, 7, 14, 30)
    - Rolling mean & std (7, 14, 30)
    - Calendar (year, month, day, day_of_week, week_of_year)
    - Arrival lags (1, 7) and rolling means (7, 14)
    - price_pct_change, arrival_pct_change
    - Market Name, Variety  <- NATIVE categoricals (strings, not codes)

  Removed:
    - days_to_next   <- derived from the future, causes leakage

  Splitting:
    Chronological per-group 70 / 15 / 15  (train / val / test)
    The pre-split tea_train.csv and tea_test.csv already follow this;
    we carve a validation set chronologically from tea_train.csv.

  Tuning:
    Early stopping on validation MAE.  Test set is held out.

Run
---
    cd member2-price-volatility
    python src/models/catboost_price_residual.py
"""

import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -- Lazy CatBoost import with helpful error --
try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    print("ERROR: CatBoost is not installed.")
    print("       Install it with:  pip install catboost")
    sys.exit(1)

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


# =====================================================================
# FEATURE CONFIGURATION
# =====================================================================

# Numeric features  (all are known at prediction time, no future leakage)
NUMERIC_FEATURES = [
    PRICE_COLUMN,                   # current modal price
    MIN_PRICE_COLUMN,               # current min price  (same row)
    MAX_PRICE_COLUMN,               # current max price  (same row)
    "price_spread",                 # max - min  (engineered below)
    # Lag prices
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    # Rolling statistics (built from shift(1) -- no leakage)
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    # Date features
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    # Arrival lags
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    # Percentage changes (lagged -- no leakage)
    "price_pct_change",
    "arrival_pct_change",
]

# Categorical features  (native CatBoost support -- pass as strings)
CAT_FEATURES = [
    "Market Name",
    "Variety",
]

# Explicitly EXCLUDED (future leakage)
EXCLUDED = {"days_to_next"}


# =====================================================================
# METRICS
# =====================================================================

def calc_metrics(actual, predicted, label=""):
    """Return dict of MAE, RMSE, R2, MAPE."""
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


def print_metrics(metrics, label):
    print(f"\n  [{label}]")
    print(f"    MAE  : Rs.{metrics['MAE']:.2f}")
    print(f"    RMSE : Rs.{metrics['RMSE']:.2f}")
    print(f"    R2   :  {metrics['R2']:.4f}")
    print(f"    MAPE :  {metrics['MAPE']:.2f}%")


# =====================================================================
# FEATURE ENGINEERING
# =====================================================================

def engineer_features(df):
    """Add price_spread; ensure categoricals are strings."""
    df = df.copy()

    # Price spread (same-row, no leakage)
    if MIN_PRICE_COLUMN in df.columns and MAX_PRICE_COLUMN in df.columns:
        df["price_spread"] = df[MAX_PRICE_COLUMN] - df[MIN_PRICE_COLUMN]
    else:
        df["price_spread"] = 0.0

    # Residual target: how much does the price change?
    df["residual"] = df[PRICE_TARGET] - df[PRICE_COLUMN]

    # Ensure categoricals are proper strings (CatBoost requirement)
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str)

    return df


# =====================================================================
# NaN IMPUTATION  (forward-fill within group, then median)
# =====================================================================

def impute_lags(df):
    """
    Forward-fill lag/rolling columns within each group so that early
    rows (which have NaN lags) are kept instead of dropped.
    Then fill remaining NaNs with column median.
    """
    df = df.copy()
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]

    fill_cols = [
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
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        df[col] = df[col].fillna(median_val)

    return df


# =====================================================================
# CHRONOLOGICAL PER-GROUP VALIDATION SPLIT  (inside training set)
# =====================================================================

def chrono_val_split(df, val_ratio=0.15):
    """
    Per-group chronological split of the training set.
    Last val_ratio fraction of each group goes to validation.
    """
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    train_parts, val_parts = [], []

    if group_cols:
        for _, group in df.groupby(group_cols):
            group = group.sort_values(DATE_COLUMN).reset_index(drop=True)
            n = len(group)
            split_idx = int(n * (1 - val_ratio))
            if split_idx < 1:
                val_parts.append(group)
                continue
            if split_idx >= n:
                train_parts.append(group)
                continue
            train_parts.append(group.iloc[:split_idx])
            val_parts.append(group.iloc[split_idx:])
    else:
        df = df.sort_values(DATE_COLUMN).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_ratio))
        train_parts.append(df.iloc[:split_idx])
        val_parts.append(df.iloc[split_idx:])

    sub_tr = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    val_df = pd.concat(val_parts,   ignore_index=True) if val_parts   else pd.DataFrame(columns=df.columns)
    return sub_tr, val_df


# =====================================================================
# BUILD CATBOOST POOL  (handles categoricals + NaN properly)
# =====================================================================

def make_pool(df, features, cat_feature_names, y=None):
    """
    Create a CatBoost Pool with categorical feature indices.
    CatBoost requires cat features to have no NaN -- fill with __MISSING__.
    """
    X = df[features].copy()

    for col in cat_feature_names:
        if col in X.columns:
            X[col] = X[col].fillna("__MISSING__")

    cat_indices = [features.index(c) for c in cat_feature_names if c in features]

    pool = Pool(
        data=X,
        label=y.values if y is not None else None,
        cat_features=cat_indices,
    )
    return pool


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "=" * 70)
    print("  CATBOOST RESIDUAL PRICE MODEL  --  EXPERIMENT")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    tr_file = train_path()
    te_file = test_path()

    if not tr_file.exists():
        raise FileNotFoundError(f"Train file not found: {tr_file}")
    if not te_file.exists():
        raise FileNotFoundError(f"Test file not found: {te_file}")

    train_df = pd.read_csv(tr_file)
    test_df  = pd.read_csv(te_file)

    train_df[DATE_COLUMN] = pd.to_datetime(train_df[DATE_COLUMN], errors="coerce")
    test_df[DATE_COLUMN]  = pd.to_datetime(test_df[DATE_COLUMN],  errors="coerce")

    print(f"  Loaded  train={len(train_df):,} rows   test={len(test_df):,} rows")

    # ------------------------------------------------------------------
    # 2. FILTER TO ROWS WITH VALID TARGET
    # ------------------------------------------------------------------
    train_df = train_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    test_df  = test_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()

    print(f"  After target dropna: train={len(train_df):,}   test={len(test_df):,}")

    # ------------------------------------------------------------------
    # 3. IMPUTE NaN LAGS
    # ------------------------------------------------------------------
    train_df = impute_lags(train_df)
    test_df  = impute_lags(test_df)
    print("  Lag imputation complete.")

    # ------------------------------------------------------------------
    # 4. ENGINEER FEATURES
    # ------------------------------------------------------------------
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)
    print("  Feature engineering complete.")

    # ------------------------------------------------------------------
    # 5. SELECT FEATURES  (filter to those present in both splits)
    # ------------------------------------------------------------------
    all_features = NUMERIC_FEATURES + CAT_FEATURES
    features = [
        f for f in all_features
        if f in train_df.columns and f in test_df.columns
            and f not in EXCLUDED
    ]
    cat_in_features = [c for c in CAT_FEATURES if c in features]

    missing = [f for f in all_features if f not in features and f not in EXCLUDED]
    if missing:
        print(f"  WARNING -- features not found, skipped: {missing}")

    print(f"  Features selected: {len(features)}  ({len(cat_in_features)} categorical)")
    print(f"  Numeric : {[f for f in features if f not in cat_in_features]}")
    print(f"  Categor.: {cat_in_features}")
    print(f"  Excluded: {list(EXCLUDED)}")

    # ------------------------------------------------------------------
    # 6. CHRONOLOGICAL VALIDATION SPLIT  (inside training data)
    # ------------------------------------------------------------------
    sub_tr, val_df = chrono_val_split(train_df, val_ratio=0.15)

    print(f"\n  Sub-train rows : {len(sub_tr):,}")
    print(f"  Validation rows: {len(val_df):,}")
    print(f"  Test rows      : {len(test_df):,}")

    y_sub = sub_tr["residual"]
    y_val = val_df["residual"]

    print(f"\n  Residual target (sub-train) -- mean: {y_sub.mean():.1f}  std: {y_sub.std():.1f}")
    print(f"  Residual target (val)       -- mean: {y_val.mean():.1f}  std: {y_val.std():.1f}")
    print(f"  Pct zero-change (sub-train) : {(y_sub == 0).mean() * 100:.1f}%")

    # ------------------------------------------------------------------
    # 7. BUILD CATBOOST POOLS
    # ------------------------------------------------------------------
    pool_train = make_pool(sub_tr, features, cat_in_features, y=y_sub)
    pool_val   = make_pool(val_df, features, cat_in_features, y=y_val)

    # ------------------------------------------------------------------
    # 8. TRAIN CATBOOST WITH EARLY STOPPING ON VALIDATION MAE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING CatBoost  (early stopping on validation MAE)")
    print("=" * 70)

    model = CatBoostRegressor(
        iterations=3000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        border_count=128,
        loss_function="MAE",         # optimise for MAE directly
        eval_metric="MAE",
        random_seed=42,
        verbose=200,
        early_stopping_rounds=150,   # generous patience
        use_best_model=True,
    )

    model.fit(
        pool_train,
        eval_set=pool_val,
    )

    best_iter = model.get_best_iteration()
    print(f"\n  Best iteration (val MAE): {best_iter}")

    # ------------------------------------------------------------------
    # 9. RETRAIN FINAL MODEL ON FULL TRAINING DATA
    # ------------------------------------------------------------------
    print("\n  Retraining on full training set with best_iteration trees ...")

    final_n_iters = max(best_iter + 1, 300)   # at least 300 trees

    final_model = CatBoostRegressor(
        iterations=final_n_iters,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        border_count=128,
        loss_function="MAE",
        random_seed=42,
        verbose=200,
    )

    y_full = train_df["residual"]
    pool_full = make_pool(train_df, features, cat_in_features, y=y_full)
    final_model.fit(pool_full)

    print("  Final model trained.")

    # ------------------------------------------------------------------
    # 10. PREDICT ON TEST SET
    # ------------------------------------------------------------------
    pool_test = make_pool(test_df, features, cat_in_features)
    predicted_residual = final_model.predict(pool_test)

    predicted_price = test_df[PRICE_COLUMN].values + predicted_residual
    actual_price    = test_df[PRICE_TARGET].values
    current_price   = test_df[PRICE_COLUMN].values

    # ------------------------------------------------------------------
    # 11. AGGREGATE METRICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  AGGREGATE METRICS  (all test rows)")
    print("=" * 70)

    model_metrics    = calc_metrics(actual_price, predicted_price)
    baseline_metrics = calc_metrics(actual_price, current_price)

    print_metrics(model_metrics,    "CatBoost residual")
    print_metrics(baseline_metrics, "Baseline (current price)")

    mae_imp  = (baseline_metrics["MAE"]  - model_metrics["MAE"])  / baseline_metrics["MAE"]  * 100
    rmse_imp = (baseline_metrics["RMSE"] - model_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100

    print(f"\n  MAE  improvement vs baseline : {mae_imp:+.2f}%")
    print(f"  RMSE improvement vs baseline : {rmse_imp:+.2f}%")

    if model_metrics["MAE"] < baseline_metrics["MAE"]:
        print("\n  >>> PASS: CatBoost BEATS the persistence baseline on MAE.")
    else:
        print("\n  >>> FAIL: CatBoost does NOT beat the baseline yet.")

    # ------------------------------------------------------------------
    # 12. PER-GROUP METRICS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PER-GROUP METRICS")
    print("=" * 70)

    group_cols = [c for c in GROUP_COLUMNS if c in test_df.columns]
    results = []

    if group_cols:
        for name, grp_idx in test_df.groupby(group_cols).groups.items():
            grp_actual   = actual_price[grp_idx]
            grp_pred     = predicted_price[grp_idx]
            grp_baseline = current_price[grp_idx]

            if len(grp_actual) == 0:
                continue

            grp_mae_model    = mean_absolute_error(grp_actual, grp_pred)
            grp_mae_baseline = mean_absolute_error(grp_actual, grp_baseline)
            delta = grp_mae_baseline - grp_mae_model

            results.append({
                "Group":        str(name),
                "N":            len(grp_actual),
                "Baseline_MAE": round(grp_mae_baseline, 1),
                "CatBoost_MAE": round(grp_mae_model, 1),
                "Delta":        round(delta, 1),
                "Winner":       "CatBoost" if delta > 0 else "Baseline",
            })

    pg_df = (
        pd.DataFrame(results)
          .sort_values("Baseline_MAE", ascending=False)
          .reset_index(drop=True)
    )
    print(pg_df.to_string(index=False))

    wins   = (pg_df["Winner"] == "CatBoost").sum()
    losses = (pg_df["Winner"] == "Baseline").sum()
    print(f"\n  Group wins: CatBoost {wins} / Baseline {losses}")

    # ------------------------------------------------------------------
    # 13. FEATURE IMPORTANCE  (top 20)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TOP 20 FEATURE IMPORTANCES")
    print("=" * 70)

    imp_df = (
        pd.DataFrame({
            "feature":    features,
            "importance": final_model.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    print(imp_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 14. FINAL SUMMARY TABLE
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY  --  CatBoost Residual vs BASELINE")
    print("=" * 70)
    print(f"  {'Metric':<12} {'Baseline':>14} {'CatBoost':>14} {'Chg%':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    print(f"  {'MAE':<12} {'Rs.'+str(round(baseline_metrics['MAE'],2)):>14} "
          f"{'Rs.'+str(round(model_metrics['MAE'],2)):>14} {mae_imp:>+7.2f}%")
    print(f"  {'RMSE':<12} {'Rs.'+str(round(baseline_metrics['RMSE'],2)):>14} "
          f"{'Rs.'+str(round(model_metrics['RMSE'],2)):>14} {rmse_imp:>+7.2f}%")
    print(f"  {'R2':<12} {round(baseline_metrics['R2'],4):>14} "
          f"{round(model_metrics['R2'],4):>14}")
    print(f"  {'MAPE':<12} {str(round(baseline_metrics['MAPE'],2))+'%':>14} "
          f"{str(round(model_metrics['MAPE'],2))+'%':>14}")

    print(f"\n  Test rows evaluated: {len(test_df)}")
    print("  Baseline reference : MAE Rs.279.04, RMSE Rs.1011.10, R2 0.9589, MAPE 2.60%")
    print("=" * 70)

    return final_model, model_metrics, baseline_metrics


if __name__ == "__main__":
    main()
