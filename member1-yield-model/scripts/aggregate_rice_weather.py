import os
import json
import pandas as pd

from district_normalization import DISTRICT_CENTROIDS

RAW_DIR = "member1-yield-model/data/raw/weather/"
OUTPUT_FILE = "member1-yield-model/data/processed/weather_rice_wb.csv"

SEASONS = {
    "Autumn": (5, 9),   # Aus
    "Winter": (6, 11),  # Aman
    "Summer": (3, 6)    # Boro
}


records = []


for district in DISTRICT_CENTROIDS:

    raw_file = os.path.join(
        RAW_DIR,
        f"{district.lower().replace(' ', '_')}.json"
    )

    if not os.path.exists(raw_file):
        raise FileNotFoundError(f"Missing file: {raw_file}")

    print(f"Processing: {district}")

    with open(raw_file, "r") as f:
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


os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)