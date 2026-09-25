"""
AgroVision-ML — Unified Root Orchestrator

Runs both ML pipelines from a single command at the project root.

Usage:
    python run_all.py                          # Run both pipelines (tea, full mode)
    python run_all.py --member 1               # Only Member 1 (yield model)
    python run_all.py --member 2               # Only Member 2 (price + volatility)
    python run_all.py --commodity onion        # Member 2 with different commodity
    python run_all.py --no-tune                # Skip XGBoost tuning (faster)
    python run_all.py --commodity all          # Member 2 batch for all commodities
"""

import sys
import time
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_member1(tune: bool = True) -> bool:
    """
    Run Member 1's yield/arrival forecasting pipeline.

    Parameters
    ----------
    tune : bool
        If True, runs XGBoost hyperparameter tuning (slower but better).

    Returns
    -------
    bool — True if pipeline succeeded, False on error.
    """
    print("\n" + "=" * 70)
    print("  MEMBER 1 — Yield/Arrival Forecasting Pipeline")
    print("=" * 70)

    # Add member1 to path so its internal imports work
    m1_root = PROJECT_ROOT / "member1-yield-model"
    if not m1_root.exists():
        print(f"[ERROR] Member 1 folder not found: {m1_root}")
        return False

    sys.path.insert(0, str(m1_root))

    try:
        from src.data_loader import load_all_data
        from src.data_cleaner import clean_and_save
        from src.feature_engineer import engineer_features
        from src.model_trainer import train_models
        from src.model_evaluator import evaluate_model

        start = time.time()
        data = load_all_data()
        cleaned_df = clean_and_save(data)
        feature_df = engineer_features(cleaned_df)
        results = train_models(feature_df, tune_xgboost=tune)
        evaluate_model(results)

        elapsed = time.time() - start
        print(f"\n[OK] Member 1 pipeline complete in {elapsed:.1f}s")
        print(f"     Model saved : member1-yield-model/models/yield_model.pkl")
        print(f"     Outputs     : member1-yield-model/outputs/")
        return True

    except FileNotFoundError as e:
        print(f"\n[ERROR] Member 1 — File not found: {e}")
        print("        → Make sure member1-yield-model/data/raw/tea_cleaned.csv exists")
        print("        → Run: python scripts/download_data.py --commodity tea")
        return False
    except Exception as e:
        print(f"\n[ERROR] Member 1 pipeline failed: {type(e).__name__}: {e}")
        return False
    finally:
        # Remove member1 path to avoid polluting Member 2 imports
        if str(m1_root) in sys.path:
            sys.path.remove(str(m1_root))


def run_member2(commodity: str = "tea", skip_volatility: bool = False) -> bool:
    """
    Run Member 2's price forecasting + volatility classification pipeline.

    Parameters
    ----------
    commodity : str
        Commodity name (e.g., 'tea', 'onion') or 'all' for batch mode.
    skip_volatility : bool
        If True, skips the volatility classification model.

    Returns
    -------
    bool — True if pipeline succeeded, False on error.
    """
    print("\n" + "=" * 70)
    print("  MEMBER 2 — Price Forecasting + Volatility Pipeline")
    print(f"  Commodity : {commodity}")
    print("=" * 70)

    m2_src = PROJECT_ROOT / "member2-price-volatility" / "src"
    if not m2_src.exists():
        print(f"[ERROR] Member 2 src folder not found: {m2_src}")
        return False

    sys.path.insert(0, str(m2_src))

    try:
        from pipeline import run_pipeline, run_all

        start = time.time()

        if commodity.lower() == "all":
            results = run_all()
        else:
            results = run_pipeline(
                commodity=commodity.lower(),
                skip_volatility=skip_volatility,
            )

        elapsed = time.time() - start
        print(f"\n[OK] Member 2 pipeline complete in {elapsed:.1f}s")
        print(f"     Models saved : member2-price-volatility/artifacts/")
        return True

    except FileNotFoundError as e:
        print(f"\n[ERROR] Member 2 — File not found: {e}")
        print(f"        → Make sure member2-price-volatility/data/raw/{commodity}.csv exists")
        print(f"        → Run: python scripts/download_data.py --commodity {commodity}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Member 2 pipeline failed: {type(e).__name__}: {e}")
        return False
    finally:
        if str(m2_src) in sys.path:
            sys.path.remove(str(m2_src))


def print_summary(m1_ok: bool, m2_ok: bool, elapsed_total: float) -> None:
    """Print final summary of both pipelines."""
    print("\n" + "=" * 70)
    print("  AGROVISION-ML — PIPELINE SUMMARY")
    print("=" * 70)
    print(f"  Member 1 (Yield)             : {'✓ SUCCESS' if m1_ok else '✗ FAILED / SKIPPED'}")
    print(f"  Member 2 (Price+Volatility)  : {'✓ SUCCESS' if m2_ok else '✗ FAILED / SKIPPED'}")
    print(f"  Total time                   : {elapsed_total:.1f}s")
    print("=" * 70)

    if m1_ok and m2_ok:
        print("\n  All models trained. Ready for Member 3 (FastAPI) integration.")
        print("  Use: from shared.predict_bridge import AgroVisionPredictor")


def main():
    parser = argparse.ArgumentParser(
        description="AgroVision-ML — Run both ML pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                        Run both pipelines (tea, full)
  python run_all.py --member 1             Only yield model
  python run_all.py --member 2             Only price+volatility model
  python run_all.py --no-tune              Skip XGBoost tuning (faster)
  python run_all.py --commodity onion      Member 2 with onion data
  python run_all.py --commodity all        Member 2 batch all commodities
        """,
    )
    parser.add_argument(
        "--member", "-m",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only Member 1 or Member 2. Default: both",
    )
    parser.add_argument(
        "--commodity", "-c",
        type=str,
        default="tea",
        help="Commodity for Member 2 (default: tea). Use 'all' for batch.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip XGBoost hyperparameter tuning (faster, for Member 1)",
    )
    parser.add_argument(
        "--skip-volatility",
        action="store_true",
        help="Skip volatility model (Member 2 price model only)",
    )
    args = parser.parse_args()

    total_start = time.time()
    m1_ok = False
    m2_ok = False

    if args.member is None or args.member == 1:
        m1_ok = run_member1(tune=not args.no_tune)

    if args.member is None or args.member == 2:
        m2_ok = run_member2(
            commodity=args.commodity,
            skip_volatility=args.skip_volatility,
        )

    elapsed_total = time.time() - total_start
    print_summary(m1_ok, m2_ok, elapsed_total)


if __name__ == "__main__":
    main()
