#!/usr/bin/env python
"""
AgroVision — CLI Entry Point

Usage:
    python run.py --commodity tea          # Single commodity
    python run.py --commodity onion        # Different commodity
    python run.py --commodity all          # All commodities in data/raw/
    python run.py                          # Default: tea
    python run.py --commodity tea --skip-volatility  # Price model only
"""

import sys
import argparse
from pathlib import Path

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pipeline import run_pipeline, run_all


def main():
    parser = argparse.ArgumentParser(
        description="AgroVision ML Pipeline — Multi-Commodity Price Forecasting"
    )
    parser.add_argument(
        "--commodity", "-c",
        type=str,
        default="tea",
        help="Commodity name (e.g., tea, onion, potato) or 'all' for batch processing. "
             "Must match filename in data/raw/{commodity}.csv"
    )
    parser.add_argument(
        "--skip-volatility",
        action="store_true",
        help="Skip volatility classification (price model only)"
    )

    args = parser.parse_args()

    if args.commodity.lower() == "all":
        run_all()
    else:
        run_pipeline(
            commodity=args.commodity.lower(),
            skip_volatility=args.skip_volatility,
        )


if __name__ == "__main__":
    main()
