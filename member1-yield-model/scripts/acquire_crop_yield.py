"""
AgroVision M1 — Crop Yield Acquisition
Pilot: Rice, West Bengal

Input:
    member1-yield-model/data/raw/rice.csv

Output:
    member1-yield-model/data/processed/rice_yield_wb.csv

No synthetic, estimated, interpolated, or imputed values are created.
"""

from pathlib import Path
import pandas as pd

from district_normalization import STANDARD_WB_DISTRICTS


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = BASE_DIR / "data" / "raw" / "rice.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "rice_yield_wb.csv"
REPORT_FILE = BASE_DIR / "reports" / "crop_yield_wb_audit.md"


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def normalize_text(value):
    if pd.isna(value):
        return None
    return " ".join(str(value).strip().split())


def normalize_district(value):
    """
    Normalize district names against the existing
    standard West Bengal district list.
    """
    value = normalize_text(value)

    if value is None:
        return None

    lookup = {
        d.lower(): d
        for d in STANDARD_WB_DISTRICTS
    }

    return lookup.get(value.lower(), value)


def main():
    print("=" * 70)
    print("AGROVISION M1 — RICE YIELD ACQUISITION")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load raw dataset
    # -----------------------------------------------------
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_FILE}"
        )

    print(f"\n[LOAD] {RAW_FILE}")

    df = pd.read_csv(RAW_FILE)

    print(f"[INFO] Raw rows: {len(df):,}")
    print(f"[INFO] Columns: {list(df.columns)}")

    required = [
        "State_Name",
        "District_Name",
        "Crop_Year",
        "Season",
        "Crop",
        "Area",
        "Production",
    ]

    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    # -----------------------------------------------------
    # 2. Normalize text fields
    # -----------------------------------------------------
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        df[col] = df[col].apply(normalize_text)

    # -----------------------------------------------------
    # 3. Filter West Bengal + Rice
    # -----------------------------------------------------
    rice_wb = df[
        (df["State_Name"].str.lower() == "west bengal")
        &
        (df["Crop"].str.lower() == "rice")
    ].copy()

    print(f"\n[FILTER] West Bengal + Rice: {len(rice_wb):,} rows")

    if rice_wb.empty:
        raise ValueError(
            "No West Bengal + Rice records found."
        )

    # -----------------------------------------------------
    # 4. Normalize district names
    # -----------------------------------------------------
    rice_wb["District"] = rice_wb["District_Name"].apply(
        normalize_district
    )

    # -----------------------------------------------------
    # 5. Numeric conversion
    # -----------------------------------------------------
    rice_wb["Area"] = pd.to_numeric(
        rice_wb["Area"], errors="coerce"
    )

    rice_wb["Production"] = pd.to_numeric(
        rice_wb["Production"], errors="coerce"
    )

    rice_wb["Crop_Year"] = pd.to_numeric(
        rice_wb["Crop_Year"], errors="coerce"
    )

    # -----------------------------------------------------
    # 6. Validate Area + Production
    # -----------------------------------------------------
    invalid_area = (
        rice_wb["Area"].isna()
        | (rice_wb["Area"] <= 0)
    )

    invalid_production = (
        rice_wb["Production"].isna()
        | (rice_wb["Production"] < 0)
    )

    print(
        f"[CHECK] Invalid/missing Area: "
        f"{invalid_area.sum():,}"
    )

    print(
        f"[CHECK] Missing/invalid Production: "
        f"{invalid_production.sum():,}"
    )

    # -----------------------------------------------------
    # 7. Calculate yield only for valid observations
    #
    # Production assumed tonnes
    # Area assumed hectares
    #
    # tonnes/hectare × 1000 = kg/hectare
    # -----------------------------------------------------
    valid = rice_wb[
        ~invalid_area & ~invalid_production
    ].copy()

    valid["Yield_Kg_Ha"] = (
        valid["Production"] / valid["Area"]
    ) * 1000.0

    # Invalid records remain excluded from final target.
    # No estimation or imputation is performed.

    # -----------------------------------------------------
    # 8. Detect duplicate modeling keys
    # -----------------------------------------------------
    duplicate_key = valid.duplicated(
        subset=["District", "Season", "Crop_Year"],
        keep=False
    )

    duplicate_count = duplicate_key.sum()

    print(
        f"[CHECK] Duplicate "
        f"(District, Season, Year): {duplicate_count:,}"
    )

    # -----------------------------------------------------
    # 9. Check existing yield column if present
    # -----------------------------------------------------
    yield_comparison = None

    if "yield" in rice_wb.columns:
        rice_wb["yield"] = pd.to_numeric(
            rice_wb["yield"], errors="coerce"
        )

        compare = rice_wb[
            ~invalid_area & ~invalid_production
        ].copy()

        compare["Calculated_Yield_Kg_Ha"] = (
            compare["Production"] / compare["Area"]
        ) * 1000.0

        compare["Yield_Difference"] = (
            compare["yield"]
            - compare["Calculated_Yield_Kg_Ha"]
        )

        yield_comparison = {
            "rows_compared": len(compare),
            "mean_absolute_difference": compare[
                "Yield_Difference"
            ].abs().mean(),
            "max_absolute_difference": compare[
                "Yield_Difference"
            ].abs().max(),
        }

        print("\n[CHECK] Existing yield column found")
        print(
            f"        Rows compared: "
            f"{yield_comparison['rows_compared']:,}"
        )
        print(
            f"        Mean absolute difference: "
            f"{yield_comparison['mean_absolute_difference']:.6f}"
        )
        print(
            f"        Max absolute difference: "
            f"{yield_comparison['max_absolute_difference']:.6f}"
        )

    # -----------------------------------------------------
    # 10. Build clean target dataset
    # -----------------------------------------------------
    output = valid[
        [
            "District",
            "Season",
            "Crop_Year",
            "Area",
            "Production",
            "Yield_Kg_Ha",
        ]
    ].copy()

    output = output.rename(
        columns={
            "Crop_Year": "Year",
            "Area": "Area_Hectares",
            "Production": "Production_Tonnes",
        }
    )

    output = output.sort_values(
        ["District", "Year", "Season"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 11. Save processed dataset
    # -----------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n[SAVE] Processed dataset:"
        f"\n       {OUTPUT_FILE}"
    )

    # -----------------------------------------------------
    # 12. Generate audit report
    # -----------------------------------------------------
    years = output["Year"].dropna()

    report = f"""# M1 Rice Yield — Data Audit

## Source

Raw file:
`{RAW_FILE}`

## Filter

- State: West Bengal
- Crop: Rice

## Counts

- Raw rows: {len(df):,}
- West Bengal Rice rows: {len(rice_wb):,}
- Final valid rows: {len(output):,}

## Coverage

- Years: {int(years.min()) if not years.empty else "N/A"} -
  {int(years.max()) if not years.empty else "N/A"}
- Districts: {output["District"].nunique():,}
- Seasons: {output["Season"].nunique():,}

## Missing / Invalid

- Invalid Area: {invalid_area.sum():,}
- Invalid Production: {invalid_production.sum():,}
- Duplicate modeling keys: {duplicate_count:,}

## Target

Yield is calculated only where Area > 0 and Production is valid:

`Yield_Kg_Ha = (Production_Tonnes / Area_Hectares) * 1000`

No missing values were estimated or imputed.

## Output

`{OUTPUT_FILE}`
"""

    if yield_comparison:
        report += f"""

## Existing `yield` Column Verification

The raw dataset contained an existing `yield` column.

- Rows compared: {yield_comparison["rows_compared"]:,}
- Mean absolute difference:
  {yield_comparison["mean_absolute_difference"]:.6f}
- Maximum absolute difference:
  {yield_comparison["max_absolute_difference"]:.6f}

The calculated value is used as the M1 target.
"""

    REPORT_FILE.write_text(
        report,
        encoding="utf-8"
    )

    print(
        f"[SAVE] Audit report:"
        f"\n       {REPORT_FILE}"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()