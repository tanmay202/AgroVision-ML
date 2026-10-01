import os
import json
import time
import requests
import pandas as pd
from district_normalization import DISTRICT_CENTROIDS

RAW_DIR = "member1-yield-model/data/raw/weather/"
PROCESSED_FILE = "member1-yield-model/data/processed/weather_nasapower_wb.csv"
BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

PARAMS = [
    "PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX",
    "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"
]

def fetch_district_weather(district, lat, lon, start_year=2000, end_year=2023):
    os.makedirs(RAW_DIR, exist_ok=True)
    raw_path = os.path.join(RAW_DIR, f"{district.lower().replace(' ', '_')}.json")
    if os.path.exists(raw_path):
        with open(raw_path, "r") as f:
            return json.load(f)
    
    params = {
        "parameters": ",".join(PARAMS),
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON"
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()
    with open(raw_path, "w") as f:
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

def process_weather_data():
    records = []
    for district, (lat, lon) in DISTRICT_CENTROIDS.items():
        data = fetch_district_weather(district, lat, lon)
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
        
        agg_df = df.groupby(["District", "Year", "Season"]).agg({
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
    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    final_df.to_csv(PROCESSED_FILE, index=False)
    return final_df

if __name__ == "__main__":
    process_weather_data()
