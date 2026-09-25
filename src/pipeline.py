"""
AgroVision — Unified ML Pipeline

Single entry point for the entire Member 2 pipeline.
Replaces the manual 13-script workflow.

Usage:
    from src.pipeline import run_pipeline
    run_pipeline("tea")           # Full pipeline for tea
    run_pipeline("onion")         # Full pipeline for onion

Or from CLI via run.py:
    python run.py --commodity tea
    python run.py --commodity all  # Run for all commodities in data/raw/
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    set_commodity,
    get_commodity,
    raw_data_path,
    RAW_DATA_DIR,
)


def run_pipeline(commodity=None, skip_volatility=False):
    """
    Run the complete price forecasting + volatility pipeline.

    Parameters
    ----------
    commodity : str, optional
        Commodity name (e.g., "tea", "onion").
        If None, uses current default.
    skip_volatility : bool
        If True, skip volatility model steps.

    Returns
    -------
    dict with pipeline results
    """
    if commodity:
        set_commodity(commodity)

    commodity = get_commodity()
    start_time = time.time()

    print("=" * 60)
    print(f"  AgroVision — ML Pipeline")
    print(f"  Commodity: {commodity}")
    print("=" * 60)

    # Verify raw data exists
    raw_path = raw_data_path()
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_path}\n"
            f"Place your CSV file at: data/raw/{commodity}.csv"
        )

    # Import step modules
    from data.clean_data import clean
    from data.prepare_timeseries import prepare
    from features.create_lag_features import create_lags
    from features.create_rolling_features import create_rolling
    from features.create_date_features import create_dates
    from features.create_arrival_features import create_arrivals
    from features.create_pct_change_features import create_pct_changes
    from evaluation.time_split import split
    from models.xgboost_price import train_xgboost_price
    from models.save_final_models import save_models

    results = {"commodity": commodity}

    # ============================================================
    # STEP 1: Clean raw data
    # ============================================================
    df = clean()

    # ============================================================
    # STEP 2: Create time-series target
    # ============================================================
    df = prepare(df)

    # ============================================================
    # STEP 3: Feature engineering (5 sub-steps)
    # ============================================================
    df = create_lags(df)
    df = create_rolling(df)
    df = create_dates(df)
    df = create_arrivals(df)
    df = create_pct_changes(df)

    # ============================================================
    # STEP 4: Train/test split
    # ============================================================
    train_df, test_df = split(df)

    # ============================================================
    # STEP 5: Train & evaluate price model
    # ============================================================
    price_model, price_metrics = train_xgboost_price(train_df, test_df)
    results["price_metrics"] = price_metrics

    # ============================================================
    # STEPS 6–8: Volatility pipeline
    # ============================================================
    if not skip_volatility:
        from features.create_volatility_target import create_volatility
        from features.create_volatility_dataset import build_volatility_train
        from features.create_volatility_test import build_volatility_test

        vol_df, low_thresh, high_thresh = create_volatility(train_df)
        vol_train = build_volatility_train(vol_df)
        vol_test = build_volatility_test(test_df)
        results["volatility_thresholds"] = {
            "low_medium": low_thresh,
            "medium_high": high_thresh,
        }
    else:
        vol_train = None

    # ============================================================
    # STEP 9: Save final models
    # ============================================================
    save_models(train_df, vol_train)

    # ============================================================
    # DONE
    # ============================================================
    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print(f"  PIPELINE COMPLETE — {commodity}")
    print(f"  Elapsed: {elapsed:.1f} seconds")
    print("=" * 60)

    print(f"\n  Price Model Metrics:")
    print(f"    MAE  : {price_metrics['mae']:.2f}")
    print(f"    RMSE : {price_metrics['rmse']:.2f}")
    print(f"    R²   : {price_metrics['r2']:.4f}")

    print(f"\n  Artifacts saved in: artifacts/")
    print(f"    {commodity}_price_model.pkl")
    print(f"    {commodity}_volatility_model.pkl")
    print(f"    {commodity}_feature_config.json")
    print(f"    {commodity}_volatility_config.json")

    return results


def discover_commodities():
    """Find all commodity CSV files in data/raw/."""
    raw_dir = RAW_DATA_DIR
    if not raw_dir.exists():
        return []
    return [
        f.stem
        for f in raw_dir.glob("*.csv")
        if not f.name.startswith(".")
    ]


def run_all():
    """Run the pipeline for all commodities found in data/raw/."""
    commodities = discover_commodities()
    if not commodities:
        print("No CSV files found in data/raw/")
        return {}

    print(f"Found {len(commodities)} commodities: {', '.join(commodities)}")
    all_results = {}

    for commodity in commodities:
        try:
            results = run_pipeline(commodity)
            all_results[commodity] = results
        except Exception as e:
            print(f"\nERROR processing {commodity}: {e}")
            all_results[commodity] = {"error": str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("  MULTI-COMMODITY SUMMARY")
    print("=" * 60)

    for commodity, results in all_results.items():
        if "error" in results:
            print(f"  {commodity:15s}  ERROR: {results['error']}")
        else:
            m = results.get("price_metrics", {})
            print(
                f"  {commodity:15s}  "
                f"R²={m.get('r2', 0):.4f}  "
                f"MAE={m.get('mae', 0):.2f}  "
                f"RMSE={m.get('rmse', 0):.2f}"
            )

    return all_results
