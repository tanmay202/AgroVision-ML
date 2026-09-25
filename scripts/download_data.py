"""
AgroVision — Data Download Script
Downloads tea mandi price/arrival data from Data.gov.in API.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --commodity onion
    python scripts/download_data.py --limit 50000

Output:
    member2-price-volatility/data/raw/{commodity}.csv
    member1-yield-model/data/raw/tea_cleaned.csv   (if tea)

NOTE:
    API key is for the public Data.gov.in Agmarknet dataset.
    Resource ID: 9ef84268-d588-465a-a308-a864a43d0070
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import requests
    import pandas as pd
except ImportError:
    print("[ERROR] Missing dependencies. Run: pip install requests pandas")
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
BASE_URL = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def download_data(commodity: str = "tea", limit: int = 50000) -> pd.DataFrame:
    """
    Download mandi data from Data.gov.in API.

    Parameters
    ----------
    commodity : str
        Commodity name to filter (e.g., 'tea', 'onion').
    limit : int
        Max number of records to download.

    Returns
    -------
    pd.DataFrame with downloaded records.
    """
    params = {
        "api-key": API_KEY,
        "format": "json",
        "limit": limit,
    }

    if commodity.lower() != "all":
        params["filters[commodity]"] = commodity.capitalize()

    print(f"[DOWNLOAD] Fetching {commodity} data from Data.gov.in...")
    print(f"           URL: {BASE_URL}")
    print(f"           Limit: {limit:,} records")

    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=60)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print("[ERROR] Request timed out. Try reducing --limit or check your connection.")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Request failed: {e}")
        sys.exit(1)

    data = response.json()

    if "records" not in data:
        print(f"[ERROR] Unexpected API response. Keys: {list(data.keys())}")
        print(f"        Response: {str(data)[:500]}")
        sys.exit(1)

    records = data["records"]
    if not records:
        print(f"[WARN] No records returned for commodity='{commodity}'.")
        print("       Try --commodity all to see what is available.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"[OK]   Downloaded {len(df):,} rows × {len(df.columns)} columns")
    return df


def save_data(df: pd.DataFrame, commodity: str) -> None:
    """Save downloaded data to both member data directories."""
    commodity_lower = commodity.lower()

    # Member 2 path (primary)
    m2_dir = PROJECT_ROOT / "member2-price-volatility" / "data" / "raw"
    m2_dir.mkdir(parents=True, exist_ok=True)
    m2_path = m2_dir / f"{commodity_lower}.csv"
    df.to_csv(m2_path, index=False)
    print(f"[SAVE] Member 2 → {m2_path}")

    # Member 1 path (tea only, different expected filename)
    if commodity_lower == "tea":
        m1_dir = PROJECT_ROOT / "member1-yield-model" / "data" / "raw"
        m1_dir.mkdir(parents=True, exist_ok=True)
        m1_path = m1_dir / "tea_cleaned.csv"
        df.to_csv(m1_path, index=False)
        print(f"[SAVE] Member 1 → {m1_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AgroVision — Download mandi data from Data.gov.in"
    )
    parser.add_argument(
        "--commodity", "-c",
        default="tea",
        help="Commodity name (e.g., tea, onion, potato) or 'all'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50000,
        help="Maximum number of records to download (default: 50000)",
    )
    args = parser.parse_args()

    df = download_data(commodity=args.commodity, limit=args.limit)
    if not df.empty:
        save_data(df, commodity=args.commodity)
        print("\n[DONE] Data download complete.")
        print(f"       Columns: {list(df.columns)}")
        print(f"       Date range: {df.get('arrival_date', df.get('Reported Date', ['?'])).iloc[[0, -1]].tolist()}")


if __name__ == "__main__":
    main()
