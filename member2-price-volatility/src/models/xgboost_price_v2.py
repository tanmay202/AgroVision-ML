"""
AgroVision -- XGBoost Price Model v2  (EXPERIMENT -- do not touch save_final_models.py)

Improvements over previous experiments
---------------------------------------
1. Group identity encoded as ordinal features
   Market Name + Variety → integer codes the model can split on.
   Lets XGBoost learn per-group dynamics without one-hot explosion.

2. days_to_next encoded as a feature
   Horizon varies 1–7 days; baseline MAE swings ₹227→₹1186 with it.
   Giving the model the horizon lets it adjust confidence.

3. NaN imputation (ffill within group) instead of dropna
   Previous experiments dropped 14 test rows that the baseline counted.
   We now evaluate on the same 1,234 rows so the comparison is fair.

4. Price spread  (Max Price − Min Price, current row)
   Known at prediction time; captures real-time within-session volatility.
   No leakage — it is the same observation row.

5. XGBoost early stopping on a chronological val split of training data
   Avoids over-fitting the training residuals.

6. Group-stratified evaluation
   Shows per-group MAE so we can see if the model helps volatile groups
   even when aggregate numbers are close.

Target
------
    Predict  future_price_change = future_modal_price − current_price.
    Reconstruct: predicted_future_price = current_price + predicted_change.

Baseline
--------
    Persistence: predicted_future_price = current_price   (MAE ₹279.04)

Run
---
    cd member2-price-volatility
    python src/models/xgboost_price_v2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


# ============================================================
# CONFIG
# ============================================================

from config import (
    DATE_COLUMN,
    GROUP_COLUMNS,
    PRICE_COLUMN,
    PRICE_TARGET,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    ARRIVAL_COLUMN,
    train_path,
    test_path,
)

# ============================================================
# BASE FEATURES  (same leakage-safe set as before)
# ============================================================

BASE_FEATURES = [
    PRICE_COLUMN,           # current modal price (known at prediction time)
    MIN_PRICE_COLUMN,       # current min price  (same row — no leakage)
    MAX_PRICE_COLUMN,       # current max price  (same row — no leakage)
    # Lag prices
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    # Rolling statistics (built with shift(1) — no leakage)
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
    # Percentage changes (lagged — no leakage)
    "price_pct_change",
    "arrival_pct_change",
    # NEW: prediction horizon
    "days_to_next",
    # NEW: price spread (engineered below)
    "price_spread",
    # NEW: group identity codes (engineered below)
    "group_market_code",
    "group_variety_code",
    "group_pair_code",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(actual, predicted):
    """Return MAE, RMSE, R², MAPE as a dict."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2   = r2_score(actual, predicted)

    # MAPE — guard against zero actuals
    nonzero = actual != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
    else:
        mape = float("nan")

    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def print_metrics(metrics, label):
    print(f"\n  [{label}]")
    print(f"    MAE  : Rs.{metrics['MAE']:.2f}")
    print(f"    RMSE : Rs.{metrics['RMSE']:.2f}")
    print(f"    R2   :  {metrics['R2']:.4f}")
    print(f"    MAPE :  {metrics['MAPE']:.2f}%")


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df, encoder=None, fit_encoder=False):
    """
    Add engineered features to df.

    Parameters
    ----------
    df          : DataFrame (must contain GROUP_COLUMNS + price columns)
    encoder     : fitted OrdinalEncoder or None
    fit_encoder : if True, fit the encoder on df (use for training data only)

    Returns
    -------
    df, encoder
    """
    df = df.copy()

    # 1. Price spread (current-row Min/Max — safe, no leakage)
    if MIN_PRICE_COLUMN in df.columns and MAX_PRICE_COLUMN in df.columns:
        df["price_spread"] = df[MAX_PRICE_COLUMN] - df[MIN_PRICE_COLUMN]
    else:
        df["price_spread"] = 0.0

    # 2. Group identity encoding
    group_cols_present = [c for c in GROUP_COLUMNS if c in df.columns]
    market_col   = "Market Name"
    variety_col  = "Variety"

    if market_col in df.columns:
        if fit_encoder or encoder is None:
            enc_market = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df["group_market_code"] = enc_market.fit_transform(
                df[[market_col]]
            ).astype(float)
        else:
            # Use existing encoder — stored as tuple
            enc_market = encoder["market"]
            df["group_market_code"] = enc_market.transform(
                df[[market_col]]
            ).astype(float)
    else:
        enc_market = None
        df["group_market_code"] = -1.0

    if variety_col in df.columns:
        if fit_encoder or encoder is None:
            enc_variety = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df["group_variety_code"] = enc_variety.fit_transform(
                df[[variety_col]]
            ).astype(float)
        else:
            enc_variety = encoder["variety"]
            df["group_variety_code"] = enc_variety.transform(
                df[[variety_col]]
            ).astype(float)
    else:
        enc_variety = None
        df["group_variety_code"] = -1.0

    # Pair code: encode Market+Variety jointly
    if market_col in df.columns and variety_col in df.columns:
        pair_series = df[market_col] + " | " + df[variety_col]
        if fit_encoder or encoder is None:
            enc_pair = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )
            df["group_pair_code"] = enc_pair.fit_transform(
                pair_series.to_frame()
            ).astype(float)
        else:
            enc_pair = encoder["pair"]
            df["group_pair_code"] = enc_pair.transform(
                pair_series.to_frame()
            ).astype(float)
    else:
        enc_pair = None
        df["group_pair_code"] = -1.0

    new_encoder = {
        "market": enc_market,
        "variety": enc_variety,
        "pair": enc_pair,
    }

    return df, new_encoder


# ============================================================
# IMPUTE NaN LAGS  (forward-fill within each group)
# ============================================================

def impute_lags(df):
    """
    Forward-fill lag/rolling columns within each group so that early
    rows (which have NaN lags) are kept instead of dropped.
    This ensures we evaluate on the same rows as the persistence baseline.
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
        # Any remaining NaN (very first rows per group) — fill with column median
        for col in existing:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    else:
        for col in existing:
            df[col] = df[col].ffill().fillna(df[col].median())

    return df


# ============================================================
# CHRONOLOGICAL VALIDATION SPLIT (inside training set)
# ============================================================

def chrono_val_split(df, val_ratio=0.20):
    """Per-group chronological 80/20 split of the training set."""
    group_cols = [c for c in GROUP_COLUMNS if c in df.columns]
    train_parts, val_parts = [], []

    if group_cols:
        for _, group in df.groupby(group_cols):
            group = group.sort_values(DATE_COLUMN).reset_index(drop=True)
            split_idx = int(len(group) * (1 - val_ratio))
            if split_idx < 1:
                val_parts.append(group)
                continue
            if split_idx >= len(group):
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


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 65)
    print("XGBOOST PRICE MODEL v2  —  EXPERIMENT")
    print("=" * 65)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

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

    print(f"  Loaded  train={len(train_df):,} rows  test={len(test_df):,} rows")

    # --------------------------------------------------------
    # FILTER TO ROWS WITH VALID TARGET
    # --------------------------------------------------------

    train_df = train_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()
    test_df  = test_df.dropna(subset=[PRICE_TARGET, PRICE_COLUMN]).copy()

    print(f"  After target dropna: train={len(train_df):,}  test={len(test_df):,}")

    # --------------------------------------------------------
    # IMPUTE NaN LAGS (before engineering so spread uses clean data)
    # --------------------------------------------------------

    train_df = impute_lags(train_df)
    test_df  = impute_lags(test_df)

    print("  Lag imputation complete.")

    # --------------------------------------------------------
    # ENGINEER NEW FEATURES
    # --------------------------------------------------------

    train_df, encoder = engineer_features(train_df, fit_encoder=True)
    test_df,  _       = engineer_features(test_df,  encoder=encoder, fit_encoder=False)

    print("  Feature engineering complete.")

    # --------------------------------------------------------
    # SELECT FEATURE COLUMNS (those present in both splits)
    # --------------------------------------------------------

    features = [
        f for f in BASE_FEATURES
        if f in train_df.columns and f in test_df.columns
    ]
    missing = [f for f in BASE_FEATURES if f not in features]
    if missing:
        print(f"  WARNING — missing features skipped: {missing}")

    print(f"  Features selected: {len(features)}")
    print(f"  Feature list: {features}")

    # --------------------------------------------------------
    # CREATE PRICE-CHANGE TARGET
    # --------------------------------------------------------

    train_df["future_price_change"] = train_df[PRICE_TARGET] - train_df[PRICE_COLUMN]
    test_df["future_price_change"]  = test_df[PRICE_TARGET]  - test_df[PRICE_COLUMN]

    # --------------------------------------------------------
    # CHRONOLOGICAL VALIDATION SPLIT (inside training set)
    # --------------------------------------------------------

    sub_tr, val_df = chrono_val_split(train_df, val_ratio=0.20)

    print(f"\n  Subtrain rows : {len(sub_tr):,}")
    print(f"  Val rows      : {len(val_df):,}")

    X_sub   = sub_tr[features]
    y_sub   = sub_tr["future_price_change"]   # price-change target
    X_val   = val_df[features]
    y_val   = val_df["future_price_change"]   # price-change target

    print(f"  Target (train) -- mean: {y_sub.mean():.1f}  std: {y_sub.std():.1f}")
    print(f"  Target (val)   -- mean: {y_val.mean():.1f}  std: {y_val.std():.1f}")
    print(f"  Pct zero-change (train): {(y_sub==0).mean()*100:.1f}%")

    # --------------------------------------------------------
    # TRAIN WITH EARLY STOPPING ON CHANGE-RESIDUAL MAE
    # --------------------------------------------------------
    # We early-stop on the price-change MAE (not reconstructed-price MAE).
    # Reconstructed-price MAE moves from ~Rs.279 from round 1 and plateaus
    # immediately, giving best_round=12.  The change residual starts at the
    # naive-zero-prediction baseline and shows real learning signal.

    print("\n" + "=" * 65)
    print("TRAINING  (early stopping on val price-change MAE)")
    print("=" * 65)

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.02,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective="reg:pseudohubererror",
        eval_metric="mae",          # MAE on price-change residuals
        early_stopping_rounds=100,  # generous patience
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_sub, y_sub,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    best_round = model.best_iteration
    print(f"\n  Best round (on price-change val MAE): {best_round}")

    # --------------------------------------------------------
    # RETRAIN FINAL MODEL ON ALL TRAINING DATA
    # --------------------------------------------------------

    print("\n  Retraining on full train set with best_round+1 trees...")

    final_model = XGBRegressor(
        n_estimators=max(best_round + 1, 300),   # at least 300 trees

        max_depth=5,
        learning_rate=0.02,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:pseudohubererror",
        random_state=42,
        n_jobs=-1,
    )

    final_model.fit(train_df[features], train_df["future_price_change"])
    print("  Final model trained.")

    # --------------------------------------------------------
    # PREDICT ON TEST SET
    # --------------------------------------------------------

    predicted_change = final_model.predict(test_df[features])
    predicted_price  = test_df[PRICE_COLUMN].values + predicted_change
    actual_price     = test_df[PRICE_TARGET].values
    current_price    = test_df[PRICE_COLUMN].values

    # --------------------------------------------------------
    # AGGREGATE METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("AGGREGATE METRICS  (all test rows)")
    print("=" * 65)

    model_metrics    = calculate_metrics(actual_price, predicted_price)
    baseline_metrics = calculate_metrics(actual_price, current_price)

    print_metrics(model_metrics,    "XGBoost v2")
    print_metrics(baseline_metrics, "Baseline (current price)")

    # Improvement
    mae_imp  = (baseline_metrics["MAE"]  - model_metrics["MAE"])  / baseline_metrics["MAE"]  * 100
    rmse_imp = (baseline_metrics["RMSE"] - model_metrics["RMSE"]) / baseline_metrics["RMSE"] * 100

    print(f"\n  MAE  improvement vs baseline : {mae_imp:+.2f}%")
    print(f"  RMSE improvement vs baseline : {rmse_imp:+.2f}%")

    if model_metrics["MAE"] < baseline_metrics["MAE"]:
        print("\n  PASS: XGBoost v2 BEATS the persistence baseline on MAE.")
    else:
        print("\n  FAIL: XGBoost v2 does NOT beat the baseline yet.")

    # --------------------------------------------------------
    # PER-GROUP METRICS
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PER-GROUP METRICS")
    print("=" * 65)

    results = []
    group_cols = [c for c in GROUP_COLUMNS if c in test_df.columns]

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
                "Group": str(name),
                "N":     len(grp_actual),
                "Baseline_MAE": round(grp_mae_baseline, 1),
                "Model_MAE":    round(grp_mae_model, 1),
                "Delta_lower_is_better": round(delta, 1),
            })

    pg_df = (
        pd.DataFrame(results)
          .sort_values("Baseline_MAE", ascending=False)
          .reset_index(drop=True)
    )
    print(pg_df.to_string(index=False))

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("TOP 15 FEATURE IMPORTANCES")
    print("=" * 65)

    imp_df = (
        pd.DataFrame({
            "feature":    features,
            "importance": final_model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    print(imp_df.to_string(index=False))

    # --------------------------------------------------------
    # HORIZON-STRATIFIED METRICS
    # --------------------------------------------------------

    if "days_to_next" in test_df.columns:
        print("\n" + "=" * 65)
        print("METRICS BY PREDICTION HORIZON (days_to_next)")
        print("=" * 65)

        horizon_results = []
        for d in sorted(test_df["days_to_next"].dropna().unique()):
            mask = test_df["days_to_next"].values == d
            if mask.sum() < 5:
                continue
            h_mae_model    = mean_absolute_error(actual_price[mask], predicted_price[mask])
            h_mae_baseline = mean_absolute_error(actual_price[mask], current_price[mask])
            horizon_results.append({
                "days_to_next": int(d),
                "N":            int(mask.sum()),
                "Baseline_MAE": round(h_mae_baseline, 1),
                "Model_MAE":    round(h_mae_model, 1),
                "Delta":        round(h_mae_baseline - h_mae_model, 1),
            })

        print(pd.DataFrame(horizon_results).to_string(index=False))

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL SUMMARY  --  v2 vs BASELINE")
    print("=" * 65)
    print(f"  {'Metric':<12} {'Baseline':>14} {'XGBoost v2':>14} {'Chg%':>8}")
    print(f"  {'-'*12} {'-'*14} {'-'*14} {'-'*8}")
    print(f"  {'MAE':<12} {'Rs.'+str(round(baseline_metrics['MAE'],2)):>14} {'Rs.'+str(round(model_metrics['MAE'],2)):>14} {mae_imp:>+7.2f}%")
    print(f"  {'RMSE':<12} {'Rs.'+str(round(baseline_metrics['RMSE'],2)):>14} {'Rs.'+str(round(model_metrics['RMSE'],2)):>14} {rmse_imp:>+7.2f}%")
    print(f"  {'R2':<12} {round(baseline_metrics['R2'],4):>14} {round(model_metrics['R2'],4):>14}")
    print(f"  {'MAPE':<12} {str(round(baseline_metrics['MAPE'],2))+'%':>14} {str(round(model_metrics['MAPE'],2))+'%':>14}")

    return final_model, model_metrics, baseline_metrics


if __name__ == "__main__":
    main()
