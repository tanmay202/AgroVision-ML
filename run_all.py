"""
AgroVision-ML -- Unified Root Orchestrator

Runs ML pipelines from a single command at the project root.

Usage:
    python run_all.py                          # Run Member 1 (inference) and Member 2 (pipeline)
    python run_all.py --member 1               # Only Member 1 (Rice yield model inference)
    python run_all.py --member 2               # Only Member 2 (price + volatility)
    python run_all.py --mode inference         # Member 1 inference mode (default)
    python run_all.py --mode train             # Member 1 candidate model training
    python run_all.py --mode validate          # Member 1 candidate validation
    python run_all.py --mode diagnose          # Member 1 diagnostics
    python run_all.py --commodity onion        # Member 2 with different commodity
    python run_all.py --no-tune                # Skip tuning (faster validation)
    python run_all.py --commodity all          # Member 2 batch for all commodities
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.paths import (
    get_project_root,
    get_member1_root,
    get_member2_root,
    get_m1_scripts_dir,
    get_m1_processed_dir,
)


def _clean_console_output(output_text: str) -> str:
    """Filter out third-party deprecation and unpickling version warnings for clean console output."""
    ignored_keywords = [
        "InconsistentVersionWarning",
        "warnings.warn",
        "scikit-learn.org",
        "model_persistence.html",
        "UserWarning",
    ]
    lines = [
        line for line in output_text.strip().splitlines()
        if not any(kw in line for kw in ignored_keywords)
    ]
    return "\n".join(lines)


def run_member1(
    mode: str = "inference",
    tune: bool = False,
    model_type: str = "improved_ridge",
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
) -> bool:
    """
    Run Member 1's Rice yield forecasting pipeline.

    Parameters
    ----------
    mode : str
        Execution mode: 'inference' (default), 'train', 'validate', or 'diagnose'.
        Ordinary inference never runs model training automatically.
    tune : bool
        If True, runs hyperparameter tuning during validation.
    model_type : str
        Candidate model to use ('improved_ridge', 'improved_xgb_cons', 'improved_xgb').
    input_file : str, optional
        Path to input CSV or JSON for batch inference.
    output_file : str, optional
        Path to write prediction results.

    Returns
    -------
    bool -- True if pipeline succeeded, False on error.
    """
    print("\n" + "=" * 70)
    print("  MEMBER 1 -- Rice Yield Forecasting Pipeline")
    print(f"  Mode : {mode}")
    print("=" * 70)

    m1_root = get_member1_root()
    if not m1_root.exists():
        print(f"[ERROR] Member 1 folder not found: {m1_root}")
        return False

    scripts_dir = get_m1_scripts_dir()
    data_processed = get_m1_processed_dir()

    try:
        start = time.time()

        if mode == "inference":
            predict_script = scripts_dir / "predict_rice_yield.py"
            if not predict_script.exists():
                print(f"[ERROR] Member 1 inference script not found: {predict_script}")
                return False

            cmd = [
                sys.executable,
                str(predict_script),
                "--model", model_type,
            ]

            if input_file:
                inp_path = Path(input_file).resolve()
                if not inp_path.exists():
                    print(f"[ERROR] Member 1 input file not found: {inp_path}")
                    return False
                cmd.extend(["--input", str(inp_path)])
                if output_file:
                    out_path = Path(output_file).resolve()
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cmd.extend(["--output", str(out_path)])
            elif output_file:
                test_csv = data_processed / "test.csv"
                if not test_csv.exists():
                    print(f"[ERROR] Default test dataset not found: {test_csv}")
                    return False
                out_path = Path(output_file).resolve()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cmd.extend(["--input", str(test_csv), "--output", str(out_path)])
            else:
                cmd.extend([
                    "--district", "Hooghly",
                    "--season", "Winter",
                    "--yield-lag1", "2800.0",
                ])

            print(f"[INFO] Running Member 1 inference with model '{model_type}'...")
            res = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if res.returncode != 0:
                print(f"[ERROR] Member 1 inference failed (exit code {res.returncode}):")
                if res.stderr:
                    print(res.stderr.strip())
                if res.stdout:
                    print(res.stdout.strip())
                return False

            if res.stdout:
                cleaned = _clean_console_output(res.stdout)
                if cleaned:
                    print(cleaned)

            elapsed = time.time() - start
            print(f"\n[OK] Member 1 inference complete in {elapsed:.1f}s")
            print(f"     Candidate model : member1-yield-model/models/ (type: {model_type})")
            if output_file:
                print(f"     Predictions     : {output_file}")
            elif input_file:
                print(f"     Batch inference : complete ({inp_path.name})")
            else:
                print(f"     Sample inference: complete (District: Hooghly, Season: Winter)")
            return True

        elif mode == "train":
            train_script = scripts_dir / "train_and_evaluate_improved.py"
            if not train_script.exists():
                print(f"[ERROR] Member 1 training script not found: {train_script}")
                return False

            print("[INFO] Member 1 training mode: Fitting candidate models on Train+Val (2000-2016).")
            print("[NOTE] Limitation: Member 1 does not provide an automated single-command raw ETL")
            print("       pipeline because raw weather acquisition requires external NASA POWER API calls.")
            print("       Training executes candidate model fitting on pre-built feature datasets.")

            cmd = [sys.executable, str(train_script)]
            res = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if res.returncode != 0:
                print(f"[ERROR] Member 1 training failed (exit code {res.returncode}):")
                if res.stderr:
                    print(res.stderr.strip())
                if res.stdout:
                    print(res.stdout.strip())
                return False

            if res.stdout:
                cleaned = _clean_console_output(res.stdout)
                if cleaned:
                    print(cleaned)

            elapsed = time.time() - start
            print(f"\n[OK] Member 1 training complete in {elapsed:.1f}s")
            return True

        elif mode == "validate":
            script_name = "tune_candidates.py" if tune else "validate_candidates.py"
            val_script = scripts_dir / script_name
            if not val_script.exists():
                print(f"[ERROR] Member 1 validation script not found: {val_script}")
                return False

            print(f"[INFO] Running Member 1 validation ({script_name})...")
            cmd = [sys.executable, str(val_script)]
            res = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if res.returncode != 0:
                print(f"[ERROR] Member 1 validation failed (exit code {res.returncode}):")
                if res.stderr:
                    print(res.stderr.strip())
                if res.stdout:
                    print(res.stdout.strip())
                return False

            if res.stdout:
                print(res.stdout.strip())

            elapsed = time.time() - start
            print(f"\n[OK] Member 1 validation complete in {elapsed:.1f}s")
            return True

        elif mode == "diagnose":
            diag_script = scripts_dir / "diagnose_models.py"
            if not diag_script.exists():
                print(f"[ERROR] Member 1 diagnostics script not found: {diag_script}")
                return False

            print("[INFO] Running Member 1 diagnostics...")
            cmd = [sys.executable, str(diag_script)]
            res = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
            )

            if res.returncode != 0:
                print(f"[ERROR] Member 1 diagnostics failed (exit code {res.returncode}):")
                if res.stderr:
                    print(res.stderr.strip())
                if res.stdout:
                    print(res.stdout.strip())
                return False

            if res.stdout:
                cleaned = _clean_console_output(res.stdout)
                if cleaned:
                    print(cleaned)

            elapsed = time.time() - start
            print(f"\n[OK] Member 1 diagnostics complete in {elapsed:.1f}s")
            return True

        else:
            print(f"[ERROR] Unknown mode '{mode}'. Choose from: inference, train, validate, diagnose.")
            return False

    except Exception as e:
        print(f"\n[ERROR] Member 1 pipeline failed: {type(e).__name__}: {e}")
        return False


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
    bool -- True if pipeline succeeded, False on error.
    """
    print("\n" + "=" * 70)
    print("  MEMBER 2 -- Price Forecasting + Volatility Pipeline")
    print(f"  Commodity : {commodity}")
    print("=" * 70)

    m2_src = get_member2_root() / "src"
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
        print(f"\n[ERROR] Member 2 -- File not found: {e}")
        print(f"        -> Make sure member2-price-volatility/data/raw/{commodity}.csv exists")
        print(f"        -> Run: python scripts/download_data.py --commodity {commodity}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Member 2 pipeline failed: {type(e).__name__}: {e}")
        return False
    finally:
        if str(m2_src) in sys.path:
            sys.path.remove(str(m2_src))


def print_summary(
    m1_ok: bool,
    m2_ok: bool,
    elapsed_total: float,
    m1_run: bool = True,
    m2_run: bool = True,
) -> None:
    """Print final summary of both pipelines using safe ASCII output."""
    print("\n" + "=" * 70)
    print("  AGROVISION-ML -- PIPELINE SUMMARY")
    print("=" * 70)
    status_m1 = "SUCCESS" if m1_ok else ("SKIPPED" if not m1_run else "FAILED")
    status_m2 = "SUCCESS" if m2_ok else ("SKIPPED" if not m2_run else "FAILED")
    print(f"  Member 1 (Yield)             : {status_m1}")
    print(f"  Member 2 (Price+Volatility)  : {status_m2}")
    print(f"  Total time                   : {elapsed_total:.1f}s")
    print("=" * 70)

    if (not m1_run or m1_ok) and (not m2_run or m2_ok) and (m1_ok or m2_ok):
        print("\n  Models ready for Member 3 (FastAPI) integration.")
        print("  Use: from shared.predict_bridge import AgroVisionPredictor")


def main():
    parser = argparse.ArgumentParser(
        description="AgroVision-ML -- Run ML pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py                        Run both pipelines (inference for Member 1, pipeline for Member 2)
  python run_all.py --member 1             Only Member 1 (Rice yield model inference)
  python run_all.py --member 2             Only Member 2 (Price + volatility model)
  python run_all.py --mode inference       Run Member 1 inference (default)
  python run_all.py --mode train           Run Member 1 candidate model training
  python run_all.py --mode validate        Run Member 1 candidate model validation
  python run_all.py --mode diagnose        Run Member 1 diagnostic analysis
  python run_all.py --commodity onion      Member 2 with onion data
  python run_all.py --commodity all        Member 2 batch all commodities
  python run_all.py --no-tune              Skip hyperparameter tuning during validation
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
        "--mode",
        type=str,
        choices=["inference", "train", "validate", "diagnose"],
        default="inference",
        help="Execution mode for Member 1 (default: inference). Training/validation are not run during inference.",
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
        help="Skip hyperparameter tuning (faster validation for Member 1)",
    )
    parser.add_argument(
        "--skip-volatility",
        action="store_true",
        help="Skip volatility model (Member 2 price model only)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="improved_ridge",
        choices=["improved_ridge", "improved_xgb_cons", "improved_xgb"],
        help="Candidate model type for Member 1 inference (default: improved_ridge)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input CSV or JSON path for Member 1 batch inference",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV or JSON path for Member 1 batch inference",
    )
    args = parser.parse_args()

    total_start = time.time()
    m1_ok = False
    m2_ok = False
    m1_run = args.member is None or args.member == 1
    m2_run = args.member is None or args.member == 2

    if m1_run:
        m1_ok = run_member1(
            mode=args.mode,
            tune=not args.no_tune,
            model_type=args.model_type,
            input_file=args.input,
            output_file=args.output,
        )

    if m2_run:
        m2_ok = run_member2(
            commodity=args.commodity,
            skip_volatility=args.skip_volatility,
        )

    elapsed_total = time.time() - total_start
    print_summary(m1_ok, m2_ok, elapsed_total, m1_run=m1_run, m2_run=m2_run)

    all_ok = (not m1_run or m1_ok) and (not m2_run or m2_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
