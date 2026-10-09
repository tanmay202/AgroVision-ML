"""
AgroVision — Rice Price Forecast Horizon Evaluation

Evaluates Member 2's frozen production 48-feature Rice price model across
empirically observed forecast horizons on the audited chronological holdout set
(2023-02-01 to 2024-02-01).

Key evaluation characteristics:
1. Frozen Production Artifacts: Evaluates the frozen 48-feature XGBoost regressor
   ('artifacts/rice_price_model.pkl') without retraining or modifying features.
2. Horizon Stratification: Segregates predictions strictly by actual calendar
   horizon ('days_to_next'), calculating distinct metrics for 1-day, 7-day,
   intermediate horizons (2 to 6 days), and the overall holdout dataset.
3. Metric Suite:
   - MAE (₹/quintal)
   - RMSE (₹/quintal)
   - R² (coefficient of determination)
   - MAPE (%) with zero-actual price handling
   - MAPE-derived forecast accuracy convention: 100 - MAPE (%)
   - Tolerance-based accuracy: Percentage of predictions within ±₹25/quintal
4. Leakage Prevention: Features are reconstructed strictly from historical
   observations at each prediction timestamp. Ground truth target prices
   are never exposed to the model feature matrix.
5. 14-Day Forecast Transparency: Documents the limitation that the production model
   supports horizons up to 7 days, and details the prerequisites for 14-day forecasting.
"""

import sys
import os
import json
import argparse
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure utf-8 output in Windows environments
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

MEMBER2_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = MEMBER2_DIR / "src"
REPO_ROOT = MEMBER2_DIR.parent

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    GROUP_COLUMNS,
    FORECAST_HORIZON_MAX_DAYS,
)
from inference import RiceInferencePipeline


def discover_artifacts_dir():
    """Locate the artifacts directory containing production model and config."""
    candidates = [
        os.environ.get("AGROVISION_ARTIFACTS_DIR"),
        REPO_ROOT / "artifacts",
        MEMBER2_DIR / "artifacts",
        REPO_ROOT / "artifacts_backup" / "production",
    ]
    for c in candidates:
        if c:
            p = Path(c)
            if (p / "rice_price_final_config.json").exists() and (p / "rice_price_model.pkl").exists():
                return p
    raise FileNotFoundError("Could not locate rice_price_final_config.json and rice_price_model.pkl.")


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE), safely ignoring zeros in actual prices.
    Returns float (percentage, e.g. 0.46 for 0.46%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    nonzero = y_true != 0.0
    if nonzero.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)


def compute_regression_metrics(y_true, y_pred, tolerance_rs=25.0):
    """
    Compute full regression and forecast accuracy metric suite.

    Parameters
    ----------
    y_true : array-like
        Ground truth actual prices in ₹/quintal.
    y_pred : array-like
        Model predicted prices in ₹/quintal.
    tolerance_rs : float
        Absolute threshold in ₹/quintal for tolerance-based accuracy.

    Returns
    -------
    dict with N, MAE, RMSE, R2, MAPE, MAPE_accuracy, tolerance_accuracy
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {
            "N": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "MAPE": np.nan,
            "100 - MAPE (%)": np.nan,
            "Tolerance_Accuracy (%)": np.nan,
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = calculate_mape(y_true, y_pred)
    mape_acc = float(100.0 - mape) if not np.isnan(mape) else np.nan
    tol_acc = float(np.mean(np.abs(y_true - y_pred) <= tolerance_rs) * 100.0)

    return {
        "N": int(n),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "100 - MAPE (%)": mape_acc,
        "Tolerance_Accuracy (%)": tol_acc,
    }


def load_and_prepare_evaluation_dataset(
    holdout_start="2023-02-01",
    holdout_end="2024-02-01",
    cleaned_path=None,
    test_path=None,
):
    """
    Build the exact evaluation dataset for the holdout period with all 48 historical features.

    Parameters
    ----------
    holdout_start : str
        Start date (inclusive) of the audited holdout period.
    holdout_end : str
        End date (inclusive) of the audited holdout period.
    cleaned_path : Path or str, optional
        Path to rice_cleaned.csv.
    test_path : Path or str, optional
        Path to rice_test.csv.

    Returns
    -------
    eval_df : pd.DataFrame
        Holdout dataframe with historical features, target, horizon, and metadata.
    """
    if cleaned_path is None:
        cleaned_path = MEMBER2_DIR / "data" / "processed" / "rice_cleaned.csv"
    if test_path is None:
        test_path = MEMBER2_DIR / "data" / "processed" / "rice_test.csv"

    cleaned_path = Path(cleaned_path)
    test_path = Path(test_path)

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test split dataset not found at {test_path}")

    # 1. Load test split to identify audited holdout rows and target values
    test_df = pd.read_csv(test_path)
    test_df[DATE_COLUMN] = pd.to_datetime(test_df[DATE_COLUMN])

    holdout_mask = (test_df[DATE_COLUMN] >= holdout_start) & (test_df[DATE_COLUMN] <= holdout_end)
    holdout_test = test_df[holdout_mask].copy()

    if holdout_test.empty:
        raise ValueError(f"No holdout observations found between {holdout_start} and {holdout_end}")

    # 2. Load cleaned data to compute full 48 historical features up to each prediction date
    raw_df = pd.read_csv(cleaned_path)
    raw_df[DATE_COLUMN] = pd.to_datetime(raw_df[DATE_COLUMN])
    raw_df = raw_df.sort_values(by=GROUP_COLUMNS + [DATE_COLUMN]).reset_index(drop=True)

    # Use production inference feature generator
    pipeline = RiceInferencePipeline()
    feat_df = pipeline._generate_features(raw_df)

    # Record target date for verification
    feat_df["target_date"] = feat_df.groupby(GROUP_COLUMNS)[DATE_COLUMN].shift(-1)

    # Merge features with the audited holdout rows on Market, Variety, Date
    merge_cols = GROUP_COLUMNS + [DATE_COLUMN]
    target_cols = [
        "future_modal_price",
        "days_to_next",
    ]
    if "State Name" in holdout_test.columns:
        target_cols.append("State Name")
    if "District Name" in holdout_test.columns:
        target_cols.append("District Name")

    eval_df = pd.merge(
        holdout_test[merge_cols + [PRICE_COLUMN] + target_cols],
        feat_df[merge_cols + ["target_date"] + pipeline.features_list],
        on=merge_cols,
        how="inner",
    )

    eval_df["forecast_horizon_days"] = eval_df["days_to_next"].astype(int)
    return eval_df, pipeline


def run_evaluation(
    holdout_start="2023-02-01",
    holdout_end="2024-02-01",
    tolerance_rs=25.0,
    predictions_out_path=None,
    summary_out_path=None,
):
    """
    Executes the forecast evaluation across all horizons on the audited holdout dataset.

    Returns
    -------
    tuple of (summary_df, eval_df, model_metadata)
    """
    eval_df, pipeline = load_and_prepare_evaluation_dataset(
        holdout_start=holdout_start,
        holdout_end=holdout_end,
    )

    # Predict future price change using the frozen 48-feature XGBoost model
    X = eval_df[pipeline.features_list].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"NaN features detected in evaluation matrix: {nan_cols}")

    predicted_change = pipeline.price_model.predict(X)
    eval_df["predicted_price_change"] = predicted_change
    eval_df["predicted_modal_price"] = np.round(eval_df[PRICE_COLUMN] + predicted_change, 2)
    eval_df["actual_modal_price"] = eval_df["future_modal_price"]
    eval_df["actual_price_change"] = eval_df["actual_modal_price"] - eval_df[PRICE_COLUMN]
    eval_df["error"] = eval_df["predicted_modal_price"] - eval_df["actual_modal_price"]
    eval_df["absolute_error"] = np.abs(eval_df["error"])
    eval_df["percentage_error"] = np.where(
        eval_df["actual_modal_price"] != 0,
        (eval_df["absolute_error"] / eval_df["actual_modal_price"]) * 100.0,
        np.nan,
    )
    eval_df["within_25_tolerance"] = eval_df["absolute_error"] <= tolerance_rs

    # Compute metrics stratified by forecast horizon
    summary_rows = []
    supported_horizons = sorted(eval_df["forecast_horizon_days"].unique())

    for h in supported_horizons:
        sub = eval_df[eval_df["forecast_horizon_days"] == h]
        m = compute_regression_metrics(
            sub["actual_modal_price"].values,
            sub["predicted_modal_price"].values,
            tolerance_rs=tolerance_rs,
        )
        base_m = compute_regression_metrics(
            sub["actual_modal_price"].values,
            sub[PRICE_COLUMN].values,
            tolerance_rs=tolerance_rs,
        )
        summary_rows.append({
            "Horizon": f"{h}-Day",
            "N": m["N"],
            "MAE (₹/qtl)": round(m["MAE"], 2),
            "RMSE (₹/qtl)": round(m["RMSE"], 2),
            "R2": round(m["R2"], 4),
            "MAPE (%)": round(m["MAPE"], 2),
            "100 - MAPE (%)": round(m["100 - MAPE (%)"], 2),
            f"Within ₹{int(tolerance_rs)}/qtl (%)": round(m["Tolerance_Accuracy (%)"], 2),
            "Baseline_MAE (₹/qtl)": round(base_m["MAE"], 2),
            "Baseline_RMSE (₹/qtl)": round(base_m["RMSE"], 2),
            "Baseline_R2": round(base_m["R2"], 4),
        })

    # Overall holdout metrics
    all_m = compute_regression_metrics(
        eval_df["actual_modal_price"].values,
        eval_df["predicted_modal_price"].values,
        tolerance_rs=tolerance_rs,
    )
    all_base = compute_regression_metrics(
        eval_df["actual_modal_price"].values,
        eval_df[PRICE_COLUMN].values,
        tolerance_rs=tolerance_rs,
    )
    summary_rows.append({
        "Horizon": "Overall (Holdout)",
        "N": all_m["N"],
        "MAE (₹/qtl)": round(all_m["MAE"], 2),
        "RMSE (₹/qtl)": round(all_m["RMSE"], 2),
        "R2": round(all_m["R2"], 4),
        "MAPE (%)": round(all_m["MAPE"], 2),
        "100 - MAPE (%)": round(all_m["100 - MAPE (%)"], 2),
        f"Within ₹{int(tolerance_rs)}/qtl (%)": round(all_m["Tolerance_Accuracy (%)"], 2),
        "Baseline_MAE (₹/qtl)": round(all_base["MAE"], 2),
        "Baseline_RMSE (₹/qtl)": round(all_base["RMSE"], 2),
        "Baseline_R2": round(all_base["R2"], 4),
    })

    summary_df = pd.DataFrame(summary_rows)

    # Save output artifacts if paths provided
    if predictions_out_path:
        out_p = Path(predictions_out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Reorder columns for readability
        lead_cols = [
            "Market Name",
            "Variety",
            DATE_COLUMN,
            "target_date",
            "forecast_horizon_days",
            PRICE_COLUMN,
            "actual_modal_price",
            "predicted_modal_price",
            "predicted_price_change",
            "actual_price_change",
            "error",
            "absolute_error",
            "percentage_error",
            "within_25_tolerance",
        ]
        other_cols = [c for c in eval_df.columns if c not in lead_cols]
        eval_df[lead_cols + other_cols].to_csv(out_p, index=False)
        print(f"Saved row-level predictions to: {out_p}")

    if summary_out_path:
        out_s = Path(summary_out_path)
        out_s.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_s, index=False)
        print(f"Saved summary metrics to: {out_s}")

    return summary_df, eval_df, pipeline.config


def print_evaluation_report(summary_df, config):
    """Print terminal summary report with clear formatting."""
    print("\n" + "=" * 80)
    print("AGROVISION — RICE PRICE MODEL FORECAST HORIZON EVALUATION REPORT")
    print("=" * 80)
    print(f"Commodity: {config.get('commodity', 'rice').upper()}")
    print(f"Model: Frozen Production XGBoost Regressor (48 features)")
    print(f"Audited Final Holdout Period: 2023-02-01 through 2024-02-01")
    print("=" * 80)

    # Primary Horizon Metrics Table
    display_cols = [
        "Horizon",
        "N",
        "MAE (₹/qtl)",
        "RMSE (₹/qtl)",
        "R2",
        "MAPE (%)",
        "100 - MAPE (%)",
        "Within ₹25/qtl (%)",
    ]
    sub_summary = summary_df.rename(columns={
        summary_df.columns[7]: "Within ₹25/qtl (%)"
    })
    print("\n[PRODUCTION MODEL PERFORMANCE BY HORIZON]")
    print(sub_summary[display_cols].to_string(index=False))

    print("\n" + "-" * 80)
    print("[BASELINE COMPARISON (Current-Price Persistence)]")
    baseline_cols = [
        "Horizon",
        "N",
        "MAE (₹/qtl)",
        "Baseline_MAE (₹/qtl)",
        "RMSE (₹/qtl)",
        "Baseline_RMSE (₹/qtl)",
        "R2",
        "Baseline_R2",
    ]
    print(summary_df[baseline_cols].to_string(index=False))
    print("-" * 80)

    # 14-day analysis transparency note
    print("\n[STATUS OF 14-DAY FORECASTING]")
    print("  Status: UNSUPPORTED by the current frozen production model.")
    print("  Reason: The production pipeline sets FORECAST_HORIZON_MAX_DAYS = 7.")
    print("          Rows with gaps > 7 days are discarded during time-series preparation.")
    print("          The model was trained strictly on next-arrival targets within 1 to 7 days.")
    print("  Requirements to implement valid 14-day forecasts:")
    print("    1. Target Construction: Define explicit 14-day lookahead target (future_price_14d - current_price)")
    print("       or design an autoregressive multi-step recursive forecasting harness.")
    print("    2. Temporal Purge: Expand the train/test chronological purge gap from 7 days to >= 14 days.")
    print("    3. Retraining: Fit and tune a dedicated 14-day model on historical training folds.")
    print("       (Per task constraints, the frozen production model is NOT retrained here).")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Rice price forecast accuracy by horizon.")
    parser.add_argument("--holdout-start", default="2023-02-01", help="Holdout start date (YYYY-MM-DD)")
    parser.add_argument("--holdout-end", default="2024-02-01", help="Holdout end date (YYYY-MM-DD)")
    parser.add_argument("--tolerance", type=float, default=25.0, help="Absolute error tolerance in ₹/quintal")
    parser.add_argument(
        "--predictions-out",
        default=str(MEMBER2_DIR / "data" / "processed" / "rice_horizon_predictions.csv"),
        help="Output CSV for row-level predictions",
    )
    parser.add_argument(
        "--summary-out",
        default=str(MEMBER2_DIR / "data" / "processed" / "rice_horizon_metrics_summary.csv"),
        help="Output CSV for horizon summary metrics",
    )
    args = parser.parse_args()

    summary_df, eval_df, config = run_evaluation(
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        tolerance_rs=args.tolerance,
        predictions_out_path=args.predictions_out,
        summary_out_path=args.summary_out,
    )
    print_evaluation_report(summary_df, config)


if __name__ == "__main__":
    main()
