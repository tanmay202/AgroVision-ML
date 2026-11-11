import json
import time
import sys
from pathlib import Path
import requests
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from district_normalization import DISTRICT_CENTROIDS

BASE_DIR = SCRIPT_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "weather"
PROCESSED_FILE = BASE_DIR / "data" / "processed" / "weather_nasapower_wb.csv"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMS = [
    "PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX",
    "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"
]

def fetch_district_weather(district, lat, lon, start_year, end_year):
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = RAW_DIR / f"{district.lower().replace(' ', '_')}.json"

    params = {
        "parameters": ",".join(PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON"
    }

    print(f"Downloading {district}: {start_year}-{end_year}")

    response = requests.get(BASE_URL, params=params, timeout=120)
    response.raise_for_status()

    data = response.json()

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    time.sleep(1)
    return data


def get_season(month):
    if 6 <= month <= 11:
        return "Kharif"
    elif month in [12, 1, 2]:
        return "Rabi"
    else:
        return "Summer"


def process_weather_data(start_year, end_year):
    records = []

    for district, (lat, lon) in DISTRICT_CENTROIDS.items():

        data = fetch_district_weather(
            district, lat, lon, start_year, end_year
        )

        parameter_data = data["properties"]["parameter"]

        df_list = []

        for param in PARAMS:
            s = pd.Series(parameter_data[param], name=param)
            df_list.append(s)

        df = pd.concat(df_list, axis=1)

        df.index = pd.to_datetime(df.index, format="%Y%m%d")

        df["District"] = district
        df["Year"] = df.index.year
        df["Season"] = df.index.month.map(get_season)

        agg_df = df.groupby(
            ["District", "Year", "Season"]
        ).agg({
            "PRECTOTCORR": "sum",
            "T2M": "mean",
            "T2M_MIN": "min",
            "T2M_MAX": "max",
            "RH2M": "mean",
            "ALLSKY_SFC_SW_DWN": "mean",
            "WS2M": "mean"
        }).reset_index()

        records.append(agg_df)

    final_df = pd.concat(records, ignore_index=True)

    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("\nWeather processing complete.")
    print("Shape:", final_df.shape)
    print("Years:", final_df["Year"].min(), "-", final_df["Year"].max())

    return final_df


if __name__ == "__main__":

    if len(sys.argv) >= 3:
        start_year = int(sys.argv[1])
        end_year = int(sys.argv[2])
    else:
        start_year = 1997
        end_year = 2019

    process_weather_data(start_year, end_year)