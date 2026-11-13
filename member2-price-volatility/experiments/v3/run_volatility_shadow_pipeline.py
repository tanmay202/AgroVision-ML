"""
AgroVision-ML — Member 2: Prospective Shadow Monitoring Pipeline Runner for Volatility V3

Provides a runnable CLI interface to:
1. Ingest newly reported mandi observations into isolated shadow database.
2. Generate idempotent point-in-time V3 14-day shadow forecasts and persistence predictions.
3. Perform delayed outcome evaluation strictly after 14 complete calendar days have elapsed.
4. Compute and display comprehensive internal monitoring metrics and governance reports.

Usage Examples:
  # Ingest observations and generate shadow forecasts
  python run_volatility_shadow_pipeline.py --ingest data/processed/rice_features_final.csv --predict --cohort PROSPECTIVE

  # Perform delayed outcome evaluation for closed windows as of a given date
  python run_volatility_shadow_pipeline.py --evaluate --as-of-date 2024-02-01

  # Generate and view the internal monitoring report
  python run_volatility_shadow_pipeline.py --report --cohort PROSPECTIVE --output-report reports/volatility_shadow_monitoring_report.md
"""

import sys
import os
import argparse
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
MEMBER2_DIR = EXP_DIR.parent.parent
SRC_DIR = MEMBER2_DIR / "src"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(MEMBER2_DIR))
sys.path.insert(0, str(EXP_DIR))

from config import (
    get_shadow_db_path,
    is_volatility_shadow_enabled,
    set_volatility_shadow_enabled,
)
from volatility_shadow import (
    init_volatility_shadow_db,
    process_volatility_shadow_ingestion,
    VolatilityShadowEvaluator,
    compute_volatility_shadow_metrics,
    generate_volatility_shadow_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("volatility_shadow_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Member 2 Volatility V3 Prospective Shadow Monitoring Pipeline")
    parser.add_argument("--ingest", type=str, help="Path to CSV containing mandi observations to ingest")
    parser.add_argument("--predict", action="store_true", help="Generate shadow forecasts for eligible origin observations")
    parser.add_argument("--origin-date", type=str, default=None, help="Origin date for forecast generation (YYYY-MM-DD)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate closed 14-day calendar windows")
    parser.add_argument("--as-of-date", type=str, default=None, help="Evaluation as-of date (YYYY-MM-DD)")
    parser.add_argument("--recheck-empty", action="store_true", help="Recheck previously empty windows")
    parser.add_argument("--report", action="store_true", help="Print and export internal monitoring report")
    parser.add_argument("--cohort", type=str, default="PROSPECTIVE", help="Target cohort: PROSPECTIVE, RETROSPECTIVE, or ALL")
    parser.add_argument("--output-report", type=str, default=None, help="Output path for generated markdown report")
    parser.add_argument("--db-path", type=str, default=None, help="Custom SQLite database path")
    parser.add_argument("--force", action="store_true", help="Force execution even if operational switch is disabled")
    parser.add_argument("--margin", type=float, default=0.01, help="Predefined non-inferiority margin (default 0.01 = 1 percent)")

    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else get_shadow_db_path()
    init_volatility_shadow_db(db_path)
    logger.info("Operating on shadow database at: %s", db_path)

    if args.force:
        set_volatility_shadow_enabled(True)

    # 1. Ingestion & Prediction
    if args.ingest:
        csv_p = Path(args.ingest)
        if not csv_p.exists():
            logger.error("Ingestion file not found at %s", csv_p)
            sys.exit(1)

        logger.info("Loading observations from %s...", csv_p)
        df = pd.read_csv(csv_p)
        logger.info("Loaded %d rows. Ingesting and generating forecasts...", len(df))

        res = process_volatility_shadow_ingestion(
            history_df=df,
            origin_date=args.origin_date,
            db_path=db_path,
            cohort=args.cohort,
            force_enabled=args.force,
        )
        logger.info(
            "Ingestion Result: status=%s ingested=%d generated=%d skipped=%d errors=%d",
            res.get("status"),
            res.get("observations_ingested", 0),
            res.get("generated_count", 0),
            res.get("skipped_duplicate_count", 0),
            res.get("errors_count", 0),
        )

    elif args.predict:
        logger.warning("--predict specified without --ingest. Reading observations from database...")
        # If predicting from existing observations in db
        with sqlite3.connect(db_path) as conn:
            obs_df = pd.read_sql_query("""
                SELECT market as "Market Name", variety as "Variety",
                       reported_date as "Reported Date", modal_price as "Modal Price (Rs./Quintal)",
                       arrivals as "Arrivals (Tonnes)"
                FROM mandi_observations
                ORDER BY market, variety, reported_date ASC;
            """, conn)
        if obs_df.empty:
            logger.error("No mandi observations available in database.")
            sys.exit(1)
        res = process_volatility_shadow_ingestion(
            history_df=obs_df,
            origin_date=args.origin_date,
            db_path=db_path,
            cohort=args.cohort,
            force_enabled=args.force,
        )
        logger.info("Forecast generation result: %s", res)

    # 2. Delayed Evaluation
    if args.evaluate:
        logger.info("Running delayed outcome evaluation for complete 14-day calendar windows (as_of_date=%s)...", args.as_of_date)
        evaluator = VolatilityShadowEvaluator(db_path=db_path)
        eval_res = evaluator.evaluate_pending_forecasts(
            as_of_date=args.as_of_date,
            db_path=db_path,
            recheck_empty=args.recheck_empty,
        )
        logger.info(
            "Evaluation Complete: evaluated=%d empty_windows=%d still_pending=%d (as_of_date=%s)",
            eval_res["evaluated_count"],
            eval_res["window_empty_count"],
            eval_res["still_pending_count"],
            eval_res["as_of_date"],
        )

    # 3. Report Generation
    if args.report or args.output_report:
        logger.info("Computing metrics and generating monitoring report for cohort '%s' (margin=%.4f)...", args.cohort, args.margin)
        report_text = generate_volatility_shadow_report(
            db_path=db_path,
            cohort=args.cohort,
            output_path=args.output_report,
            non_inferiority_margin=args.margin,
        )
        print("\n" + "=" * 80)
        print(report_text)
        print("=" * 80 + "\n")
        if args.output_report:
            logger.info("Report saved to %s", args.output_report)


if __name__ == "__main__":
    main()
