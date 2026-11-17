import json
from pathlib import Path
import httpx
import pandas as pd

# Centroid for Hooghly district
LAT = 22.90
LON = 88.39
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
T_BASE_POTATO = 7.0  # T_base from features.crop_dim for Kufri Jyoti

DATA_DIR = Path.home() / "projects/agrovision/data/raw"
OUTPUT_DIR = Path.home() / "projects/agrovision/data/processed"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_weather_archive() -> dict:
  url = "https://archive-api.open-meteo.com/v1/archive"
  params = {
      "latitude": LAT,
      "longitude": LON,
      "start_date": START_DATE,
      "end_date": END_DATE,
      "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
      "timezone": "Asia/Kolkata",
  }
  print(f"[1/3] Fetching weather from Open-Meteo ({START_DATE} to {END_DATE})...")
  response = httpx.get(url, params=params, timeout=60.0)
  response.raise_for_status()
  return response.json()


def process_weather_data(payload: dict) -> pd.DataFrame:
  print("[2/3] Processing daily records and calculating GDD...")
  daily = payload["daily"]

  df = pd.DataFrame({
      "time": pd.to_datetime(daily["time"]),
      "t_max_c": daily["temperature_2m_max"],
      "t_min_c": daily["temperature_2m_min"],
      "precipitation_mm": daily["precipitation_sum"],
  })

  # GDD = max((Tmax + Tmin)/2 - T_base, 0)
  mean_temp = (df["t_max_c"] + df["t_min_c"]) / 2.0
  df["daily_gdd"] = (mean_temp - T_BASE_POTATO).clip(lower=0.0)

  # Handle missing values if any
  df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)
  df["t_max_c"] = df["t_max_c"].interpolate(method="linear")
  df["t_min_c"] = df["t_min_c"].interpolate(method="linear")

  # Add relational grain keys matching schema
  df["district_id"] = "WB_HG_01"
  df["crop_id"] = "POTATO_JYOTI"

  return df


def main():
  raw_payload = fetch_weather_archive()

  # Save raw JSON dump for provenance
  raw_path = DATA_DIR / "hooghly_weather_2023_2025_raw.json"
  with open(raw_path, "w") as f:
    json.dump(raw_payload, f, indent=2)
  print(f"      Saved raw payload to: {raw_path}")

  # Process and calculate features
  df = process_weather_data(raw_payload)

  # Save processed tabular CSV
  processed_path = OUTPUT_DIR / "hooghly_weather_daily_2023_2025.csv"
  df.to_csv(processed_path, index=False)
  print(f"[3/3] Successfully generated processed dataset: {processed_path}")
  print(f"      Total records: {len(df)} rows")
  print("\n--- First 5 rows ---")
  print(df.head())
  print("\n--- Summary Statistics ---")
  print(
      df[
          ["t_max_c", "t_min_c", "precipitation_mm", "daily_gdd"]
      ].describe().round(2)
  )


if __name__ == "__main__":
  main()
