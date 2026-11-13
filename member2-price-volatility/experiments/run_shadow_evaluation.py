"""
AgroVision-ML — Member 2: Shadow Evaluation CLI Runner

Provides a runnable interface to:
1. Ingest newly reported mandi observations.
2. Generate point-in-time shadow forecasts (14d and 30d).
3. Evaluate pending forecasts against newly materialized outcome observations.
4. Output honest performance metrics comparing XGBoost vs Persistence baseline.
"""

import sys
import os
import time
import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

# Path resolution
EXP_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = EXP_DIR.parent
SRC_DIR = MEMBER2_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))

from config import (
    set_shadow_evaluation_enabled,
    set_horizon_forecasting_enabled,
    set_horizon_14d_enabled,
    set_horizon_30d_enabled,
    get_shadow_db_path,
    is_volatility_shadow_enabled,
)
from shadow_evaluation import (
    init_shadow_db,
    ingest_mandi_observations,
    generate_shadow_forecasts,
    evaluate_pending_shadow_forecasts,
    compute_shadow_evaluation_metrics,
    get_shadow_status,
    record_audit_log_entry,
    backup_shadow_database,
)
from data_connector import (
    ShadowExecutionLock,
    LockAcquisitionError,
    DataGovInConnector,
    normalize_and_map_records,
    audit_data_freshness_and_drift,
)


def print_metrics_summary(metrics: dict, title: str = "SHADOW EVALUATION LIVE PERFORMANCE SUMMARY"):
    """Pretty prints evaluation metrics comparing XGBoost vs Persistence baseline."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for h in ("14d", "30d"):
        if h not in metrics:
            continue
        h_data = metrics[h]
        counts = h_data.get("counts", {})
        print(f"\n[{h.upper()} HORIZON]")
        print(f"  Total Forecasts   : {counts.get('total_forecasts', 0)}")
        print(f"  Evaluated         : {counts.get('evaluated_forecasts', 0)}")
        print(f"  Pending           : {counts.get('pending_forecasts', 0)}")
        print(f"  Target Unavailable: {counts.get('unavailable_target_forecasts', 0)}")

        if h_data.get("status") == "pending_outcomes":
            print(f"  Notice: {h_data.get('message')}")
            continue

        if "model_performance" in h_data:
            m = h_data["model_performance"]
            b = h_data["persistence_baseline"]
            d = h_data.get("delta_model_minus_persistence", {})

            print(f"\n  {'Metric':<22} | {'XGBoost Model':<15} | {'Persistence':<15} | {'Delta':<15}")
            print("  " + "-" * 72)
            print(f"  {'MAE (Rs./qtl)':<22} | {m.get('mae_rs_qtl', 0.0):<15.2f} | {b.get('mae_rs_qtl', 0.0):<15.2f} | {d.get('mae', 0.0):<+15.2f}")
            if m.get("mae_95_ci") and b.get("mae_95_ci"):
                ci_m_str = f"[{m['mae_95_ci'][0]:.2f}, {m['mae_95_ci'][1]:.2f}]"
                ci_b_str = f"[{b['mae_95_ci'][0]:.2f}, {b['mae_95_ci'][1]:.2f}]"
                ci_d_str = f"[{d['mae_delta_95_ci'][0]:.2f}, {d['mae_delta_95_ci'][1]:.2f}]" if d.get("mae_delta_95_ci") else "N/A"
                print(f"  {'  MAE 95% CI':<22} | {ci_m_str:<15} | {ci_b_str:<15} | {ci_d_str:<15}")
            print(f"  {'RMSE (Rs./qtl)':<22} | {m.get('rmse_rs_qtl', 0.0):<15.2f} | {b.get('rmse_rs_qtl', 0.0):<15.2f} | {d.get('rmse', 0.0):<+15.2f}")
            r2_m_str = f"{m.get('r2_score'):.4f}" if m.get('r2_score') is not None else "N/A"
            r2_b_str = f"{b.get('r2_score'):.4f}" if b.get('r2_score') is not None else "N/A"
            r2_d_str = f"{d.get('r2'):+.4f}" if d.get('r2') is not None else "N/A"
            print(f"  {'R2 Score':<22} | {r2_m_str:<15} | {r2_b_str:<15} | {r2_d_str:<15}")
            print(f"  {'MAPE (%)':<22} | {m.get('mape_pct', 0.0):<15.2f} | {b.get('mape_pct', 0.0):<15.2f} | -")
            print(f"  {'Within +/- Rs.25 (%)':<22} | {m.get('within_25_rs_pct', 0.0):<15.2f} | {b.get('within_25_rs_pct', 0.0):<15.2f} | -")
            print(f"  {'Median Error (p50)':<22} | {m.get('median_error_p50', 0.0):<15.2f} | {b.get('median_error_p50', 0.0):<15.2f} | -")
            print(f"  {'90th Percentile Error':<22} | {m.get('error_p90', 0.0):<15.2f} | {b.get('error_p90', 0.0):<15.2f} | -")

            subsets = h_data.get("subsets", {})
            if "movement_market_days" in subsets:
                sub_mov = subsets["movement_market_days"]
                print(f"\n  [Subset: Movement Market Days (N={sub_mov['count']}, {sub_mov['pct_of_evaluated']}%) ]")
                print(f"    XGBoost MAE    : Rs.{sub_mov['model_mae']:.2f}")
                print(f"    Persistence MAE: Rs.{sub_mov['persistence_mae']:.2f} (Advantage: Rs.{sub_mov['persistence_mae'] - sub_mov['model_mae']:+.2f})")
                print(f"    XGBoost RMSE   : Rs.{sub_mov['model_rmse']:.2f}")
                print(f"    Persistence RMSE: Rs.{sub_mov['persistence_rmse']:.2f} (Advantage: Rs.{sub_mov['persistence_rmse'] - sub_mov['model_rmse']:+.2f})")

            if "flat_market_days" in subsets:
                sub_flat = subsets["flat_market_days"]
                print(f"\n  [Subset: Flat Market Days (N={sub_flat['count']}, {sub_flat['pct_of_evaluated']}%) ]")
                print(f"    XGBoost MAE    : Rs.{sub_flat['model_mae']:.2f}")
                print(f"    Persistence MAE: Rs.0.00 (Mathematical identity)")

            if "actual_horizon_offset_distribution" in h_data:
                print(f"\n  [Target-Date Offset Distribution]")
                for off, cnt in h_data["actual_horizon_offset_distribution"].items():
                    print(f"    Offset {off} days: {cnt} forecasts")

            if "variety_breakdown" in h_data and h_data["variety_breakdown"]:
                print(f"\n  [Variety Breakdown (N >= 5)]")
                for var, v_m in h_data["variety_breakdown"].items():
                    adv = v_m['persistence_mae'] - v_m['model_mae']
                    print(f"    {var:<20}: N={v_m['count']:<3} | XGBoost MAE=Rs.{v_m['model_mae']:<7.2f} | Pers MAE=Rs.{v_m['persistence_mae']:<7.2f} (Adv: Rs.{adv:+.2f})")

            if "market_breakdown" in h_data and h_data["market_breakdown"]:
                print(f"\n  [Market Breakdown (Top 5 by volume, N >= 5)]")
                sorted_mkts = sorted(h_data["market_breakdown"].items(), key=lambda x: x[1]["count"], reverse=True)[:5]
                for mkt, m_m in sorted_mkts:
                    adv = m_m['persistence_mae'] - m_m['model_mae']
                    print(f"    {mkt:<20}: N={m_m['count']:<3} | XGBoost MAE=Rs.{m_m['model_mae']:<7.2f} | Pers MAE=Rs.{m_m['persistence_mae']:<7.2f} (Adv: Rs.{adv:+.2f})")


def run_demo(db_path: Path):
    """Runs a self-contained demonstration using real historical observations."""
    print("=" * 80)
    print("RUNNING SHADOW EVALUATION DEMONSTRATION WORKFLOW")
    print("=" * 80)

    # 1. Load sample dataset from rice_cleaned.csv
    cleaned_csv = MEMBER2_DIR / "data" / "processed" / "rice_cleaned.csv"
    if not cleaned_csv.exists():
        print(f"Cleaned dataset not found at {cleaned_csv}. Skipping demo.")
        return

    print(f"\n1. Ingesting historical mandi observations from {cleaned_csv.name}...")
    df = pd.read_csv(cleaned_csv)
    # Filter to top market Burdwan / Common for rapid showcase
    sample_df = df[(df["Market Name"] == "Burdwan") & (df["Variety"] == "Common")].copy()
    sample_df["Reported Date"] = pd.to_datetime(sample_df["Reported Date"])
    sample_df = sample_df.sort_values("Reported Date").reset_index(drop=True)

    # Ingest historical observations
    n_ingested = ingest_mandi_observations(sample_df, db_path=db_path)
    print(f"   Successfully ingested {n_ingested} mandi records into SQLite database.")

    # 2. Generate point-in-time forecasts at an origin date with available past data
    origin_date = "2023-03-01"
    print(f"\n2. Generating shadow forecasts at origin date: {origin_date}...")
    gen_result = generate_shadow_forecasts(
        sample_df,
        horizons=(14, 30),
        origin_date=origin_date,
        db_path=db_path,
        force_enabled=True,
    )
    print(f"   Generated: {gen_result['generated_count']} forecasts, Skipped: {gen_result['skipped_existing_count']}")

    # 3. Delayed outcome evaluation
    print(f"\n3. Executing delayed outcome matching against subsequently reported observations...")
    eval_result = evaluate_pending_shadow_forecasts(db_path=db_path)
    print(f"   Evaluated: {eval_result['evaluated_count']}, Unavailable: {eval_result['unavailable_count']}, Still Pending: {eval_result['still_pending_count']}")

    # 4. Status and Metrics
    metrics = compute_shadow_evaluation_metrics(db_path=db_path)
    print_metrics_summary(metrics)


def run_pilot(db_path: Path, data_path: Optional[Path] = None):
    """
    Executes the comprehensive real-data shadow forecasting pilot workflow.
    Uses genuine West Bengal rice mandi observations.
    Explicitly labeled: Retrospective Shadow Simulation / Historical Replay Pilot.
    """
    print("=" * 80)
    print("AGROVISION-ML: REAL-DATA SHADOW FORECASTING PILOT")
    print("CLASSIFICATION: Retrospective Shadow Simulation / Historical Replay")
    print("=" * 80)

    csv_path = Path(data_path) if data_path else MEMBER2_DIR / "data" / "processed" / "rice_cleaned.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Mandi data file not found at {csv_path}")

    print(f"\n[PHASE 1: INGESTION OF GENUINE MANDI OBSERVATIONS]")
    print(f"  Source File: {csv_path.name}")
    ingest_stats = ingest_mandi_observations(csv_path, db_path=db_path, return_stats=True)
    print(f"  Total Rows Ingested     : {ingest_stats['accepted_count']}")
    print(f"  Rejected Rows (Invalid) : {ingest_stats['rejected_count']}")
    print(f"  Observation Date Range  : {ingest_stats['min_date']} to {ingest_stats['max_date']}")
    print(f"  Distinct Mandis         : {ingest_stats['markets_count']}")
    print(f"  Distinct Varieties      : {ingest_stats['varieties_count']}")

    print(f"\n[PHASE 2: POINT-IN-TIME FORECAST GENERATION]")
    pilot_origins = ["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01", "2024-01-20"]
    print(f"  Configured Origin Dates : {pilot_origins}")
    print(f"  Origin Rationale        : 4 historical origins with matured target windows, plus 1 late origin (2024-01-20) with pending target windows.")

    df_full = pd.read_csv(csv_path)
    total_gen = 0
    total_skip = 0

    for o_date in pilot_origins:
        res = generate_shadow_forecasts(
            df_full,
            horizons=(14, 30),
            origin_date=o_date,
            db_path=db_path,
            force_enabled=True,
        )
        total_gen += res["generated_count"]
        total_skip += res["skipped_existing_count"]
        print(f"  Origin {o_date:<12}: Generated {res['generated_count']:<3} forecasts, Skipped {res['skipped_existing_count']:<3} existing.")

    print(f"  Total Generated: {total_gen}, Total Skipped: {total_skip}")

    print(f"\n[PHASE 3: DELAYED OUTCOME EVALUATION]")
    eval_res = evaluate_pending_shadow_forecasts(db_path=db_path, recheck_unavailable=True)
    print(f"  Evaluated Outcomes      : {eval_res['evaluated_count']}")
    print(f"  Target Unavailable      : {eval_res['unavailable_count']}")
    print(f"  Awaiting Date (Pending) : {eval_res['still_pending_count']}")
    if eval_res.get("recovered_late_count", 0) > 0:
        print(f"  Recovered Late Targets  : {eval_res['recovered_late_count']}")

    print(f"\n[PHASE 4: IDEMPOTENCY RE-RUN VERIFICATION]")
    re_gen_count = 0
    re_skip_count = 0
    for o_date in pilot_origins:
        res_re = generate_shadow_forecasts(
            df_full,
            horizons=(14, 30),
            origin_date=o_date,
            db_path=db_path,
            force_enabled=True,
        )
        re_gen_count += res_re["generated_count"]
        re_skip_count += res_re["skipped_existing_count"]

    print(f"  Re-run Forecasts Generated: {re_gen_count} (Expected 0 - Zero duplication)")
    print(f"  Re-run Forecasts Skipped  : {re_skip_count} (Expected {total_gen} - Exactly all recognized)")

    re_eval_res = evaluate_pending_shadow_forecasts(db_path=db_path)
    print(f"  Re-run Evaluations Executed: {re_eval_res['evaluated_count']} (Expected 0 - Zero re-evaluation churn)")

    print(f"\n[PHASE 5: COMPREHENSIVE PERFORMANCE SUMMARY]")
    metrics = compute_shadow_evaluation_metrics(db_path=db_path)
    print_metrics_summary(metrics)

    status = get_shadow_status(db_path=db_path)
    print("\n" + "=" * 80)
    print(f"PILOT DATABASE FINAL AUDIT COUNTS: {status['database_path']}")
    print(f"  Total Mandi Observations : {status['counts']['total_observations']}")
    print(f"  14D Forecasts: Total={status['counts']['forecasts_14d']['total']}, Evaluated={status['counts']['forecasts_14d']['evaluated']}, Pending={status['counts']['forecasts_14d']['pending']}, Unavailable={status['counts']['forecasts_14d']['unavailable']}")
    print(f"  30D Forecasts: Total={status['counts']['forecasts_30d']['total']}, Evaluated={status['counts']['forecasts_30d']['evaluated']}, Pending={status['counts']['forecasts_30d']['pending']}, Unavailable={status['counts']['forecasts_30d']['unavailable']}")
    print("=" * 80)


def run_check_connector():
    """Checks and reports diagnostics for the official data.gov.in AGMARKNET connector."""
    print("=" * 80)
    print("CHECKING OFFICIAL DATA.GOV.IN / AGMARKNET DATA CONNECTOR")
    print("=" * 80)
    connector = DataGovInConnector()
    diag = connector.check_connection()
    print(f"Resource ID       : {diag.get('resource_id')}")
    print(f"Auth Configured   : {diag.get('auth_configured')}")
    print(f"Accessible        : {diag.get('accessible')}")
    print(f"Status Code       : {diag.get('status_code')}")
    print(f"Message           : {diag.get('message')}")
    if not diag.get("accessible"):
        print("\n[OPERATIONAL FALLBACK GUIDANCE]")
        print("  Official data.gov.in AGMARKNET queries require setting DATA_GOV_IN_API_KEY.")
        print("  For keyless / automated scheduled runs, use the documented fallback:")
        print("    python experiments/run_shadow_evaluation.py --daily-run --input-csv <path-to-batch.csv>")
    print("=" * 80)


def run_daily_prospective_workflow(
    db_path: Path,
    input_csv: Optional[Path] = None,
    max_lag_days: int = 14,
    origin_date: Optional[str] = None,
    allow_stale: bool = False,
    create_backup: bool = True,
) -> int:
    """
    Executes the automated daily prospective shadow evaluation workflow.
    - Protected by ShadowExecutionLock to ensure single-instance execution.
    - Automatic online database snapshot prior to ingestion.
    - Checks official data.gov.in AGMARKNET connector; falls back to input CSV if key absent.
    - Normalizes & strictly maps records; logs unmapped mandi/variety occurrences.
    - Audits reporting freshness and group-aware statistical distribution drift.
    - Data Quality Gate: Aborts on empty batches. Prevents prospective forecast emission
      on stale feeds unless --allow-stale is explicitly provided.
    - Evaluates pending outcomes against newly materialized observations.
    - Persists execution telemetry into shadow_audit_log for long-term observability.
    - Generates honest separated metrics report (Prospective vs Retrospective).
    
    Returns
    -------
    int
        Exit code: 0 for success, 2 for blocked gate (empty/stale feed or lock contention).
    """
    t_start = time.perf_counter()
    start_dt = datetime.now(timezone.utc)
    print("=" * 80)
    print("AGROVISION-ML: DAILY PROSPECTIVE SHADOW EVALUATION WORKFLOW")
    print(f"Start Timestamp: {start_dt.isoformat()}")
    print("=" * 80)

    lock = ShadowExecutionLock()
    try:
        lock.acquire()
    except LockAcquisitionError as e:
        print(f"\n[MUTEX LOCK BLOCKED] {e}")
        print("Another shadow execution process is currently running. Exiting safely.")
        return 2

    try:
        # Pre-ingestion Safety Backup
        if create_backup and db_path.exists():
            try:
                b_path = backup_shadow_database(db_path)
                print(f"[SAFETY BACKUP] Point-in-time database snapshot created: {b_path.name}")
            except Exception as e:
                print(f"[SAFETY BACKUP WARNING] Could not create snapshot: {e}")

        # Step 1: Connector Check & Data Source Acquisition
        print("\n[PHASE 1: DATA SOURCE ACQUISITION]")
        connector = DataGovInConnector()
        diag = connector.check_connection()
        print(f"  Official Data Source (data.gov.in) : {'CONNECTED' if diag['accessible'] else 'AUTH REQUIRED / KEYLESS'}")
        print(f"  Resource ID                        : {diag['resource_id']}")

        raw_df = None
        source_label = None

        if diag["accessible"]:
            print("  Fetching live daily observations from data.gov.in...")
            try:
                fetch_res = connector.fetch_records(limit=2000)
                rec_list = fetch_res.get("records", [])
                if rec_list:
                    raw_df = pd.DataFrame(rec_list)
                    source_label = "data_gov_in_api"
                    print(f"  Successfully fetched {len(raw_df)} live records from data.gov.in.")
                else:
                    print("  data.gov.in returned zero records for query. Falling back to CSV.")
            except Exception as e:
                print(f"  Error fetching from data.gov.in: {e}. Falling back to CSV.")

        if raw_df is None:
            if input_csv and Path(input_csv).exists():
                csv_file = Path(input_csv)
                print(f"  Using specified input CSV: {csv_file}")
                raw_df = pd.read_csv(csv_file)
                source_label = f"csv_file:{csv_file.name}"
            else:
                default_example_csv = MEMBER2_DIR / "examples" / "prospective_mandi_batch_example.csv"
                if default_example_csv.exists():
                    print(f"  Using documented prospective example feed: {default_example_csv}")
                    raw_df = pd.read_csv(default_example_csv)
                    source_label = f"example_csv:{default_example_csv.name}"
                else:
                    print("  No input CSV specified and no live API key. Using latest processed observations slice.")
                    default_proc = MEMBER2_DIR / "data" / "processed" / "rice_cleaned.csv"
                    df_all = pd.read_csv(default_proc)
                    raw_df = df_all.tail(1000).copy()
                    source_label = "rice_cleaned_tail_feed"

        # Step 2: Normalization and Mapping
        print("\n[PHASE 2: NORMALIZATION & CANONICAL MAPPING]")
        clean_df, norm_audit = normalize_and_map_records(raw_df, source_tag=source_label)
        print(f"  Raw Rows Received          : {norm_audit['raw_input_rows']}")
        print(f"  Accepted Valid Rows        : {norm_audit['accepted_valid_rows']}")
        print(f"  Rejected Corrupt Rows      : {norm_audit['rejected_corrupt_rows']} (Dates: {norm_audit['rejected_invalid_dates']}, Prices: {norm_audit['rejected_invalid_prices']})")
        print(f"  Deduplicated Within Batch  : {norm_audit['batch_duplicates_resolved']}")
        print(f"  Observation Date Range     : {norm_audit['min_date']} to {norm_audit['max_date']}")
        print(f"  Distinct Markets           : {norm_audit['distinct_mandis']}")
        print(f"  Distinct Varieties         : {norm_audit['distinct_varieties']}")
        if norm_audit['unmapped_mandi_labels']:
            print(f"  Unmapped Mandi Labels      : {norm_audit['unmapped_mandi_labels'][:5]}")
        if norm_audit['unmapped_variety_labels']:
            print(f"  Unmapped Variety Labels    : {norm_audit['unmapped_variety_labels'][:5]}")

        # Data Quality Gate: Abort if zero valid rows
        if clean_df.empty:
            print("\n[DATA QUALITY GATE BLOCKED]: Zero valid records extracted from incoming feed. Aborting run.")
            duration_s = time.perf_counter() - t_start
            record_audit_log_entry(
                db_path=db_path,
                execution_type="PROSPECTIVE_DAILY",
                status="ABORTED_EMPTY",
                source_label=source_label,
                duration_seconds=duration_s,
                details={"reason": "clean_df_empty", "raw_rows": norm_audit.get("raw_input_rows", 0)},
            )
            return 2

        # Ingest normalized records into SQLite database
        n_ingested = ingest_mandi_observations(clean_df, db_path=db_path)
        print(f"  Ingested into Shadow DB    : {n_ingested} records")

        # Step 3: Freshness & Drift Monitoring Audit
        print("\n[PHASE 3: FRESHNESS & STATISTICAL DRIFT AUDIT]")
        audit_res = audit_data_freshness_and_drift(clean_df, max_allowable_lag_days=max_lag_days)
        freshness = audit_res.get("freshness", {})
        drift = audit_res.get("drift", {})
        is_stale_feed = freshness.get("is_stale", False)
        print(f"  Freshness Status           : {'STALE' if is_stale_feed else 'FRESH'}")
        print(f"  Latest Reported Date       : {freshness.get('source_latest_date')}")
        print(f"  Reporting Lag              : {freshness.get('reporting_lag_days')} days (Threshold: {freshness.get('max_allowable_lag_days')} days)")
        print(f"  Freshness Message          : {freshness.get('message')}")
        print(f"  Price Distribution Shift   : {'DRIFT DETECTED' if drift.get('drift_detected') else 'NOMINAL'}")
        if "batch_stats" in drift:
            bs = drift["batch_stats"]
            base = drift.get("baseline_stats", {})
            print(f"  Batch Price Median         : Rs.{bs['median']:.2f} (Baseline: Rs.{base.get('price_median', 0.0):.2f}, Shift: {drift.get('median_pct_shift'):+.1f}%)")
            print(f"  Batch Price Mean (Std)     : Rs.{bs['mean']:.2f} (Rs.{bs['std']:.2f}) [Baseline Mean: Rs.{base.get('price_mean', 0.0):.2f}]")
            print(f"  Batch Price IQR            : Rs.{bs['iqr']:.2f} [p25: Rs.{bs['p25']:.2f}, p75: Rs.{bs['p75']:.2f}]")
        print(f"  Drift Evaluation Message   : {drift.get('message')}")
        if drift.get("group_drift_anomalies"):
            print(f"  Group Drift Anomalies      : {drift['group_drift_anomalies'][:3]}")

        # Data Quality Gate: Stale observation check for prospective generation
        gen_count = 0
        skip_count = 0
        if is_stale_feed and not allow_stale:
            print(f"\n[DATA QUALITY GATE WARNING]: Stale data detected ({freshness.get('reporting_lag_days')} days lag > {max_lag_days} days threshold).")
            print("  Prospective shadow forecasts are NOT emitted on stale feeds to protect live evaluation integrity.")
            print("  (Specify --allow-stale to override for historical simulations or backfill replays.)")
        else:
            # Step 4: Prospective Forecast Generation
            print("\n[PHASE 4: PROSPECTIVE SHADOW FORECAST GENERATION]")
            if origin_date is None:
                origin_date = norm_audit['max_date']
            print(f"  Forecast Origin Date       : {origin_date}")
            print(f"  Assigned Cohort Tag        : PROSPECTIVE")

            # Read history from database up to origin_date
            db_conn = sqlite3.connect(db_path)
            history_query = """
                SELECT market as "Market Name", variety as "Variety", 
                       reported_date as "Reported Date", modal_price as "Modal Price (Rs./Quintal)", 
                       arrivals as "Arrivals (Tonnes)"
                FROM mandi_observations
                WHERE reported_date <= ?
                ORDER BY market, variety, reported_date
            """
            history_df = pd.read_sql_query(history_query, db_conn, params=(origin_date,))
            db_conn.close()

            gen_res = generate_shadow_forecasts(
                history_df=history_df,
                horizons=(14, 30),
                origin_date=origin_date,
                db_path=db_path,
                force_enabled=True,
                cohort="PROSPECTIVE",
            )
            gen_count = gen_res['generated_count']
            skip_count = gen_res['skipped_existing_count']
            print(f"  Prospective Forecasts Gen  : {gen_count}")
            print(f"  Already Existing (Skipped) : {skip_count}")

            # Volatility V3 shadow prospective forecast generation
            if is_volatility_shadow_enabled():
                try:
                    from volatility_shadow import process_volatility_shadow_ingestion
                    vol_res = process_volatility_shadow_ingestion(
                        history_df=history_df,
                        origin_date=origin_date,
                        db_path=db_path,
                        cohort="PROSPECTIVE",
                        force_enabled=True,
                    )
                    print(f"  Volatility V3 Shadow Gen   : {vol_res.get('generated_count', 0)} (Skipped: {vol_res.get('skipped_duplicate_count', 0)})")
                except Exception as vol_err:
                    print(f"  [VOLATILITY SHADOW WARNING]: {vol_err}")

        # Step 5: Delayed Target Evaluation
        print("\n[PHASE 5: DELAYED OUTCOME EVALUATION]")
        eval_res = evaluate_pending_shadow_forecasts(db_path=db_path, recheck_unavailable=True)
        print(f"  Outcomes Evaluated         : {eval_res['evaluated_count']}")
        print(f"  Target Windows Unavailable : {eval_res['unavailable_count']}")
        print(f"  Awaiting Maturity (Pending): {eval_res['still_pending_count']}")
        if eval_res.get("recovered_late_count", 0) > 0:
            print(f"  Recovered Late Targets     : {eval_res['recovered_late_count']}")

        if is_volatility_shadow_enabled():
            try:
                from volatility_shadow import VolatilityShadowEvaluator
                vol_eval = VolatilityShadowEvaluator(db_path=db_path)
                vol_eval_res = vol_eval.evaluate_pending_forecasts(db_path=db_path)
                print(f"  Volatility Outcomes Eval   : {vol_eval_res.get('evaluated_count', 0)} (Empty: {vol_eval_res.get('window_empty_count', 0)}, Pending: {vol_eval_res.get('still_pending_count', 0)})")
            except Exception as vol_eval_err:
                print(f"  [VOLATILITY EVAL WARNING]  : {vol_eval_err}")

        # Step 6: Isolated Cohort Performance Reporting
        print("\n[PHASE 6: ISOLATED COHORT PERFORMANCE AUDIT]")

        # Prospective Cohort
        pro_metrics = compute_shadow_evaluation_metrics(db_path=db_path, cohort="PROSPECTIVE")
        print_metrics_summary(pro_metrics, title="PROSPECTIVE COHORT PERFORMANCE (Live Operational Feeds)")

        # Retrospective Cohort
        ret_metrics = compute_shadow_evaluation_metrics(db_path=db_path, cohort="RETROSPECTIVE")
        print_metrics_summary(ret_metrics, title="RETROSPECTIVE COHORT PERFORMANCE (Historical Simulation Benchmark)")

        if is_volatility_shadow_enabled():
            try:
                from volatility_shadow import generate_volatility_shadow_report
                vol_rep = generate_volatility_shadow_report(db_path=db_path, cohort="PROSPECTIVE")
                print("\n" + "=" * 80)
                print("VOLATILITY V3 SHADOW MONITORING SUMMARY (PROSPECTIVE)")
                print("=" * 80)
                print(vol_rep)
            except Exception as vol_rep_err:
                print(f"  [VOLATILITY REPORT WARNING]: {vol_rep_err}")

        # Record persistent audit entry
        duration_s = time.perf_counter() - t_start
        audit_status = "WARNING_STALE" if (is_stale_feed and not allow_stale) else "SUCCESS"
        record_audit_log_entry(
            db_path=db_path,
            execution_type="PROSPECTIVE_DAILY",
            status=audit_status,
            source_label=source_label,
            latest_observation_date=norm_audit["max_date"],
            reporting_lag_days=freshness.get("reporting_lag_days"),
            is_stale=is_stale_feed,
            drift_detected=drift.get("drift_detected", False),
            raw_rows=norm_audit["raw_input_rows"],
            accepted_rows=norm_audit["accepted_valid_rows"],
            rejected_rows=norm_audit["rejected_corrupt_rows"],
            forecasts_generated=gen_count,
            forecasts_skipped=skip_count,
            outcomes_evaluated=eval_res["evaluated_count"],
            outcomes_pending=eval_res["still_pending_count"],
            duration_seconds=duration_s,
            details={
                "source": source_label,
                "allow_stale": allow_stale,
                "freshness_message": freshness.get("message"),
                "drift_message": drift.get("message"),
                "group_drift_anomalies": drift.get("group_drift_anomalies", []),
            },
        )

        # Step 7: Final Status Summary
        status = get_shadow_status(db_path=db_path)
        print("\n" + "=" * 80)
        print(f"DAILY PROSPECTIVE WORKFLOW COMPLETE: {status['database_path']}")
        print(f"  Execution Duration         : {duration_s:.2f} seconds")
        print(f"  Total Observations in DB   : {status['counts']['total_observations']}")
        if "by_cohort" in status["counts"]:
            pro_cnt = status["counts"]["by_cohort"]["prospective"]
            ret_cnt = status["counts"]["by_cohort"]["retrospective"]
            print(f"  PROSPECTIVE Cohort Totals  : Total={pro_cnt['total']}, Evaluated={pro_cnt['evaluated']}, Pending={pro_cnt['pending']}, Unavailable={pro_cnt['unavailable']}")
            print(f"  RETROSPECTIVE Cohort Totals: Total={ret_cnt['total']}, Evaluated={ret_cnt['evaluated']}, Pending={ret_cnt['pending']}, Unavailable={ret_cnt['unavailable']}")
        print("=" * 80)
        return 0

    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description="AgroVision-ML Shadow Evaluation CLI")
    parser.add_argument("--ingest", type=str, help="Path to CSV containing newly arrived mandi observations")
    parser.add_argument("--generate", action="store_true", help="Generate shadow forecasts for available data")
    parser.add_argument("--origin-date", type=str, default=None, help="Forecast origin date (YYYY-MM-DD)")
    parser.add_argument("--data-path", type=str, default=None, help="Custom data source CSV for forecast generation")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate pending forecasts against stored outcomes")
    parser.add_argument("--recheck-unavailable", action="store_true", help="Recheck previously unavailable forecasts for late arrivals")
    parser.add_argument("--report", action="store_true", help="Display summary metrics report")
    parser.add_argument("--cohort", type=str, choices=["ALL", "RETROSPECTIVE", "PROSPECTIVE"], default="ALL", help="Cohort filter for report")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demonstration workflow")
    parser.add_argument("--pilot", action="store_true", help="Run comprehensive real-data shadow pilot workflow")
    parser.add_argument("--daily-run", action="store_true", help="Execute automated daily prospective monitoring workflow")
    parser.add_argument("--allow-stale", action="store_true", help="Allow prospective generation on older historical feeds")
    parser.add_argument("--backup", action="store_true", help="Create an atomic online backup snapshot of the shadow database")
    parser.add_argument("--backup-path", type=str, default=None, help="Custom target path for database backup")
    parser.add_argument("--check-connector", action="store_true", help="Check official data.gov.in AGMARKNET connector status")
    parser.add_argument("--input-csv", type=str, default=None, help="Path to input observations CSV for daily-run or ingest")
    parser.add_argument("--max-lag-days", type=int, default=14, help="Freshness threshold in calendar days (default 14)")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database path")

    args = parser.parse_args()

    # Enable switches for CLI run
    set_shadow_evaluation_enabled(True)
    set_horizon_forecasting_enabled(True)
    set_horizon_14d_enabled(True)
    set_horizon_30d_enabled(True)

    db_path = Path(args.db_path) if args.db_path else get_shadow_db_path()
    init_shadow_db(db_path)

    if args.backup:
        print("Creating online database backup snapshot...")
        b_file = backup_shadow_database(db_path=db_path, backup_path=args.backup_path)
        print(f"Database backup successfully created: {b_file}")
        sys.exit(0)

    if args.check_connector:
        run_check_connector()
        sys.exit(0)

    if args.daily_run:
        rc = run_daily_prospective_workflow(
            db_path=db_path,
            input_csv=Path(args.input_csv) if args.input_csv else None,
            max_lag_days=args.max_lag_days,
            origin_date=args.origin_date,
            allow_stale=args.allow_stale,
        )
        sys.exit(rc)

    if args.pilot:
        run_pilot(db_path, data_path=Path(args.data_path) if args.data_path else None)
        sys.exit(0)

    if args.demo:
        run_demo(db_path)
        sys.exit(0)

    if len(sys.argv) == 1:
        run_demo(db_path)
        sys.exit(0)

    if args.ingest:
        print(f"Ingesting observations from {args.ingest}...")
        stats = ingest_mandi_observations(args.ingest, db_path=db_path, return_stats=True)
        print(f"Ingestion complete: {stats}")

    if args.generate:
        print("Generating shadow forecasts...")
        src_csv = Path(args.data_path) if args.data_path else MEMBER2_DIR / "data" / "processed" / "rice_cleaned.csv"
        if src_csv.exists():
            df = pd.read_csv(src_csv)
            res = generate_shadow_forecasts(
                df,
                origin_date=args.origin_date,
                db_path=db_path,
                force_enabled=True,
            )
            print(f"Forecast generation complete: {res}")
        else:
            print(f"Data source {src_csv} not found.")

    if args.evaluate:
        print("Evaluating pending forecasts...")
        res = evaluate_pending_shadow_forecasts(db_path=db_path, recheck_unavailable=args.recheck_unavailable)
        print(f"Evaluation complete: {res}")

    if args.report:
        cohort_filter = None if args.cohort == "ALL" else args.cohort
        metrics = compute_shadow_evaluation_metrics(db_path=db_path, cohort=cohort_filter)
        cohort_title = f"SHADOW EVALUATION LIVE PERFORMANCE SUMMARY (COHORT: {args.cohort})"
        print_metrics_summary(metrics, title=cohort_title)


if __name__ == "__main__":
    main()
