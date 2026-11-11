import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

YIELD_FILE = BASE_DIR / "data" / "processed" / "rice_yield_wb_clean.csv"
WEATHER_FILE = BASE_DIR / "data" / "processed" / "weather_rice_wb.csv"
NDVI_FILE = BASE_DIR / "data" / "raw" / "ndvi" / "AgroVision_WB_Rice_NDVI_2000_2019_LONG.csv"

OUTPUT_FILE = BASE_DIR / "data" / "processed" / "rice_yield_weather_ndvi_wb.csv"

KEYS = ["District", "Year", "Season"]


def merge_m1_datasets(
    yield_file: Path = YIELD_FILE,
    weather_file: Path = WEATHER_FILE,
    ndvi_file: Path = NDVI_FILE,
    output_file: Path = OUTPUT_FILE
) -> pd.DataFrame:
    yield_file = Path(yield_file)
    weather_file = Path(weather_file)
    ndvi_file = Path(ndvi_file)
    output_file = Path(output_file)

    # --------------------------------------------------
    # 1. Load
    # --------------------------------------------------

    yield_df = pd.read_csv(yield_file)
    weather_df = pd.read_csv(weather_file)
    ndvi_df = pd.read_csv(ndvi_file)

    # --------------------------------------------------
    # 2. Standardize Yield columns
    # --------------------------------------------------

    yield_df = yield_df.rename(columns={
        "District_Name": "District",
        "Crop_Year": "Year"
    })

    yield_df["District"] = (
        yield_df["District"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Known naming differences in source data
    district_map = {
        "24 PARAGANAS NORTH": "North 24 Parganas",
        "24 PARAGANAS SOUTH": "South 24 Parganas",
        "DARJEELING": "Darjeeling",
        "HAORA": "Howrah",
        "HUGLI": "Hooghly",
        "COOCH BEHAR": "Cooch Behar",
        "DAKSHIN DINAJPUR": "Dakshin Dinajpur",
        "JALPAIGURI": "Jalpaiguri",
        "JHARGRAM": "Jhargram",
        "KALIMPONG": "Kalimpong",
        "KOLKATA": "Kolkata",
        "MALDAH": "Maldah",
        "MURSHIDABAD": "Murshidabad",
        "NADIA": "Nadia",
        "PASCHIM BARDHAMAN": "Paschim Bardhaman",
        "PASCHIM MEDINIPUR": "Paschim Medinipur",
        "PURBA BARDHAMAN": "Purba Bardhaman",
        "PURBA MEDINIPUR": "Purba Medinipur",
        "PURULIA": "Purulia",
        "SOUTH 24 PARGANAS": "South 24 Parganas",
        "UTTAR DINAJPUR": "Uttar Dinajpur",
        "BANKURA": "Bankura",
        "BIRBHUM": "Birbhum",
    }

    yield_df["District"] = yield_df["District"].replace(district_map)

    # --------------------------------------------------
    # 3. Standardize Weather + NDVI
    # --------------------------------------------------

    weather_df["District"] = weather_df["District"].astype(str).str.strip()
    ndvi_df["District"] = ndvi_df["District"].astype(str).str.strip()

    yield_df["Year"] = pd.to_numeric(yield_df["Year"])
    weather_df["Year"] = pd.to_numeric(weather_df["Year"])
    ndvi_df["Year"] = pd.to_numeric(ndvi_df["Year"])

    # --------------------------------------------------
    # 4. Validate individual datasets
    # --------------------------------------------------

    print("========================================")
    print("INDIVIDUAL DATASET VALIDATION")
    print("========================================")

    for name, df in [
        ("Yield", yield_df),
        ("Weather", weather_df),
        ("NDVI", ndvi_df)
    ]:
        print(f"\n{name}")
        print("Rows:", len(df))
        print("Districts:", df["District"].nunique())
        print("Years:", df["Year"].min(), "-", df["Year"].max())
        print(
            "Duplicate keys:",
            df.duplicated(KEYS).sum()
        )

    # These must be unique
    assert yield_df.duplicated(KEYS).sum() == 0
    assert weather_df.duplicated(KEYS).sum() == 0
    assert ndvi_df.duplicated(KEYS).sum() == 0

    # --------------------------------------------------
    # 5. Merge Yield + Weather
    # --------------------------------------------------

    merged = yield_df.merge(
        weather_df,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # 6. Merge NDVI
    # --------------------------------------------------

    merged = merged.merge(
        ndvi_df,
        on=KEYS,
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------
    # 7. Final validation
    # --------------------------------------------------

    print("\n========================================")
    print("FINAL MERGED DATASET")
    print("========================================")

    print("Shape:", merged.shape)
    print("Districts:", merged["District"].nunique())
    print("Years:", merged["Year"].min(), "-", merged["Year"].max())
    print("Duplicate keys:", merged.duplicated(KEYS).sum())

    print("\nMissing values:")
    print(merged.isna().sum())

    print("\nNDVI missing by year:")
    print(
        merged.groupby("Year")["NDVI"]
        .apply(lambda x: x.isna().sum())
    )

    print("\nSample:")
    print(merged.head(10).to_string(index=False))

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:")
    print(output_file)

    return merged


if __name__ == "__main__":
    merge_m1_datasets()