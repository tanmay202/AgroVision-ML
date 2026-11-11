import json
import sys
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from district_normalization import DISTRICT_CENTROIDS

BASE_DIR = SCRIPT_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "weather"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "weather_rice_wb.csv"

SEASONS = {
    "Autumn": (5, 9),   # Aus
    "Winter": (6, 11),  # Aman
    "Summer": (3, 6)    # Boro
}


def aggregate_rice_weather(raw_dir: Path = RAW_DIR, output_file: Path = OUTPUT_FILE) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    output_file = Path(output_file)

    records = []

    for district in DISTRICT_CENTROIDS:

        raw_file = raw_dir / f"{district.lower().replace(' ', '_')}.json"

        if not raw_file.exists():
            raise FileNotFoundError(f"Missing file: {raw_file}")

        print(f"Processing: {district}")

        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        parameter_data = data["properties"]["parameter"]

        df = pd.DataFrame(parameter_data)

        df.index = pd.to_datetime(
            df.index,
            format="%Y%m%d"
        )

        df = df[
            (df.index.year >= 1997) &
            (df.index.year <= 2019)
        ].copy()

        df["District"] = district
        df["Year"] = df.index.year

        # Process each Rice season independently
        for season, (start_month, end_month) in SEASONS.items():

            seasonal = df[
                (df.index.month >= start_month) &
                (df.index.month <= end_month)
            ].copy()

            seasonal["Season"] = season

            agg = seasonal.groupby(
                ["District", "Year", "Season"]
            ).agg({
                "PRECTOTCORR": "sum",
                "T2M": "mean",
                "T2M_MIN": "mean",
                "T2M_MAX": "mean",
                "RH2M": "mean",
                "ALLSKY_SFC_SW_DWN": "mean",
                "WS2M": "mean"
            }).reset_index()

            day_count = seasonal.groupby(
                ["District", "Year", "Season"]
            ).size().reset_index(
                name="Weather_Days"
            )

            agg = agg.merge(
                day_count,
                on=["District", "Year", "Season"],
                how="left"
            )

            records.append(agg)

    final_df = pd.concat(
        records,
        ignore_index=True
    )

    final_df = final_df.sort_values(
        ["District", "Year", "Season"]
    ).reset_index(drop=True)

    print("\n========================================")
    print("RICE WEATHER AGGREGATION COMPLETE")
    print("========================================")

    print("Shape:", final_df.shape)
    print("Years:", final_df["Year"].min(), "-", final_df["Year"].max())
    print("Districts:", final_df["District"].nunique())
    print("Seasons:", sorted(final_df["Season"].unique()))

    print(
        "Duplicate keys:",
        final_df.duplicated(
            ["District", "Year", "Season"]
        ).sum()
    )

    print("\nMissing values:")
    print(final_df.isna().sum())

    print("\nWeather days by season:")
    print(
        final_df.groupby("Season")["Weather_Days"]
        .agg(["min", "max", "mean"])
    )

    print("\nSample:")
    print(
        final_df.head(10)
        .to_string(index=False)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        output_file,
        index=False
    )

    print("\nSaved:")
    print(output_file)

    return final_df


if __name__ == "__main__":
    aggregate_rice_weather()