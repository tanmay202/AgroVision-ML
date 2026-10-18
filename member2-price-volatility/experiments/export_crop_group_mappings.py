"""
AgroVision-ML — Member 2: Export Crop & Group Mappings
Author: Member 2 (Price Prediction & Volatility ML)
Target Consumer: Member 3 (Backend Scraper & Input Validation)

Exports the exact training-data string values for:
- State Name
- District Name
- Market Name
- Variety
- Group

Derived directly from authentic preprocessed training datasets for every supported crop.
Preserves original literal spelling, capitalization, and whitespace.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MEMBER2_DIR = REPO_ROOT / "member2-price-volatility"
DELIVERABLES_DIR = REPO_ROOT / "deliverables" / "member2-ml"

M2_ARTIFACTS = MEMBER2_DIR / "artifacts"
DELIV_ARTIFACTS = DELIVERABLES_DIR / "artifacts"
PROCESSED_DATA_DIR = MEMBER2_DIR / "data" / "processed"


def export_single_crop_mappings(
    crop_name: str,
    dataset_path: Path,
    train_end_date: Optional[str] = "2021-01-31",
    date_col: str = "Reported Date",
) -> Dict[str, Any]:
    """
    Exports deterministically sorted unique values for the 5 key categorical columns.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df[date_col] = pd.to_datetime(df[date_col])

    cols = ["State Name", "District Name", "Market Name", "Variety", "Group"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' missing from {dataset_path}")

    # Missing value counts across the dataset
    missing_counts = {c: int(df[c].isna().sum()) for c in cols}

    # Filter strictly to training observation partition
    if train_end_date:
        df_train = df[df[date_col] <= pd.to_datetime(train_end_date)].copy()
    else:
        df_train = df.copy()

    train_start = str(df_train[date_col].min().date()) if not df_train.empty else None
    train_end = str(df_train[date_col].max().date()) if not df_train.empty else None

    # Deterministic sorted unique values preserving exact literal string representation
    unique_columns = {}
    for c in cols:
        unique_vals = sorted([str(x) for x in df_train[c].dropna().unique()])
        unique_columns[c] = unique_vals

    # Observed valid combinations of the 5 fields
    combos = df_train[cols].drop_duplicates().sort_values(cols)
    combos_list = combos.to_dict(orient="records")

    rel_source_path = str(dataset_path.relative_to(REPO_ROOT)).replace("\\", "/")

    mapping_payload = {
        "crop": crop_name,
        "source_dataset": rel_source_path,
        "training_date_range": {
            "start": train_start,
            "end": train_end,
        },
        "total_training_records": len(df_train),
        "columns": unique_columns,
        "missing_value_counts": missing_counts,
        "observed_combinations_count": len(combos_list),
        "observed_combinations": combos_list,
    }
    return mapping_payload


def main():
    print("=" * 70)
    print("AGROVISION-ML: EXPORTING CROP GROUP MAPPINGS FOR MEMBER 3")
    print("=" * 70)

    # 1. Rice Mappings (Primary Crop)
    rice_csv = PROCESSED_DATA_DIR / "rice_cleaned.csv"
    print(f"\nProcessing Rice mappings from {rice_csv}...")
    rice_mappings = export_single_crop_mappings(
        crop_name="rice",
        dataset_path=rice_csv,
        train_end_date="2021-01-31",
        date_col="Reported Date",
    )

    rice_out_m2 = M2_ARTIFACTS / "rice_group_mappings.json"
    rice_out_deliv = DELIV_ARTIFACTS / "rice_group_mappings.json"

    with open(rice_out_m2, "w", encoding="utf-8") as f:
        json.dump(rice_mappings, f, indent=2)
    with open(rice_out_deliv, "w", encoding="utf-8") as f:
        json.dump(rice_mappings, f, indent=2)

    print(f"  Saved Rice mappings to:")
    print(f"    - {rice_out_m2}")
    print(f"    - {rice_out_deliv}")
    print(f"  Stats: {len(rice_mappings['columns']['Market Name'])} markets, "
          f"{len(rice_mappings['columns']['Variety'])} varieties, "
          f"{rice_mappings['observed_combinations_count']} observed combinations.")

    # 2. Tea Mappings (Secondary Supported Crop)
    tea_csv = PROCESSED_DATA_DIR / "tea_cleaned.csv"
    if tea_csv.exists():
        print(f"\nProcessing Tea mappings from {tea_csv}...")
        tea_mappings = export_single_crop_mappings(
            crop_name="tea",
            dataset_path=tea_csv,
            train_end_date="2013-08-07",
            date_col="Reported Date",
        )

        tea_out_m2 = M2_ARTIFACTS / "tea_group_mappings.json"
        tea_out_deliv = DELIV_ARTIFACTS / "tea_group_mappings.json"

        with open(tea_out_m2, "w", encoding="utf-8") as f:
            json.dump(tea_mappings, f, indent=2)
        with open(tea_out_deliv, "w", encoding="utf-8") as f:
            json.dump(tea_mappings, f, indent=2)

        print(f"  Saved Tea mappings to:")
        print(f"    - {tea_out_m2}")
        print(f"    - {tea_out_deliv}")
        print(f"  Stats: {len(tea_mappings['columns']['Market Name'])} markets, "
              f"{len(tea_mappings['columns']['Variety'])} varieties, "
              f"{tea_mappings['observed_combinations_count']} observed combinations.")

    print("\n" + "=" * 70)
    print("MAPPINGS EXPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
