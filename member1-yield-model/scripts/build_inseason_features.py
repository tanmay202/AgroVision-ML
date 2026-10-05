"""
M1 In-Season Forecasting: Build current-season weather features up to forecast cutoff.
Uses REAL daily NASA POWER JSON data.
Does NOT use any observation after the cutoff.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================
# CROP SEASON DEFINITIONS & CUTOFFS FOR WEST BENGAL RICE
# ============================================================
#
# Autumn (Aus):  Sown May–Jun,  Harvest Aug–Sep.  Season window: May 1 – Sep 30
#   Cutoff: July 31  (~2 months of growing-season weather)
#
# Winter (Aman): Sown Jun–Jul,  Harvest Nov–Dec.  Season window: Jun 1 – Dec 31
#   Cutoff: Sep 30   (~3–4 months of growing-season weather)
#
# Summer (Boro): Sown Nov–Dec,  Harvest Mar–May.  Season window: Nov 1 – May 31
#   Cutoff: Feb 28   (~3 months of growing-season weather)
#
# These cutoffs are early enough to be genuine forecasts but late
# enough to include meaningful weather observations.
# ============================================================

SEASON_CONFIG = {
    'Autumn': {
        'start_month': 5, 'start_day': 1,
        'cutoff_month': 7, 'cutoff_day': 31,
        'harvest_months': [8, 9],
        'year_offset': 0,  # season year = calendar year
    },
    'Winter': {
        'start_month': 6, 'start_day': 1,
        'cutoff_month': 9, 'cutoff_day': 30,
        'harvest_months': [11, 12],
        'year_offset': 0,
    },
    'Summer': {
        'start_month': 11, 'start_day': 1,
        'cutoff_month': 2, 'cutoff_day': 28,
        'harvest_months': [3, 4, 5],
        'year_offset': -1,  # season starts in Nov of previous calendar year
    },
}

# Map weather JSON filename stems to canonical district names
WEATHER_FILE_MAP = {
    'alipurduar': 'Alipurduar',
    'bankura': 'Bankura',
    'birbhum': 'Birbhum',
    'cooch_behar': 'Cooch Behar',
    'dakshin_dinajpur': 'Dakshin Dinajpur',
    'darjeeling': 'Darjeeling',
    'hooghly': 'Hooghly',
    'howrah': 'Howrah',
    'jalpaiguri': 'Jalpaiguri',
    'jhargram': 'Jhargram',
    'kalimpong': 'Kalimpong',
    'maldah': 'Maldah',
    'murshidabad': 'Murshidabad',
    'nadia': 'Nadia',
    'north_24_parganas': 'North 24 Parganas',
    'paschim_bardhaman': 'Paschim Bardhaman',
    'paschim_medinipur': 'Paschim Medinipur',
    'purba_bardhaman': 'Purba Bardhaman',
    'purba_medinipur': 'Purba Medinipur',
    'purulia': 'Purulia',
    'south_24_parganas': 'South 24 Parganas',
    'uttar_dinajpur': 'Uttar Dinajpur',
}


def load_daily_weather(weather_dir: Path) -> pd.DataFrame:
    """Parse all NASA POWER daily JSON files into a single long DataFrame."""
    all_rows = []
    params = ['PRECTOTCORR', 'T2M', 'T2M_MIN', 'T2M_MAX', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M']

    for json_file in sorted(weather_dir.glob('*.json')):
        stem = json_file.stem
        if stem not in WEATHER_FILE_MAP:
            continue
        district = WEATHER_FILE_MAP[stem]

        with open(json_file, 'r') as f:
            data = json.load(f)

        param_data = data['properties']['parameter']
        # Get all date keys from first parameter
        date_keys = list(param_data[params[0]].keys())

        for dk in date_keys:
            row = {'District': district, 'Date': dk}
            for p in params:
                val = param_data[p].get(dk, np.nan)
                row[p] = val if val != -999 else np.nan
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
    return df


def get_season_dates(year, season):
    """Return (start_date, cutoff_date) for a given crop year and season."""
    cfg = SEASON_CONFIG[season]
    
    if season == 'Summer':
        # Summer rice: sown Nov of (year-1), cutoff Feb of (year)
        start = datetime(year - 1, cfg['start_month'], cfg['start_day'])
        cutoff = datetime(year, cfg['cutoff_month'], cfg['cutoff_day'])
    else:
        start = datetime(year, cfg['start_month'], cfg['start_day'])
        cutoff = datetime(year, cfg['cutoff_month'], cfg['cutoff_day'])
    
    return pd.Timestamp(start), pd.Timestamp(cutoff)


def compute_weather_features(df_daily, district, year, season):
    """Compute weather features from daily data, strictly up to cutoff."""
    start, cutoff = get_season_dates(year, season)
    
    mask = (df_daily['District'] == district) & \
           (df_daily['Date'] >= start) & \
           (df_daily['Date'] <= cutoff)
    subset = df_daily.loc[mask]
    
    if len(subset) == 0:
        return None, 0, None
    
    max_date = subset['Date'].max()
    n_days = len(subset)
    
    features = {
        'InSeason_Precip_Cum': subset['PRECTOTCORR'].sum(),
        'InSeason_Precip_Mean': subset['PRECTOTCORR'].mean(),
        'InSeason_Precip_Max': subset['PRECTOTCORR'].max(),
        'InSeason_T2M_Mean': subset['T2M'].mean(),
        'InSeason_T2M_Min': subset['T2M_MIN'].min(),
        'InSeason_T2M_Max': subset['T2M_MAX'].max(),
        'InSeason_T2M_Range': subset['T2M_MAX'].max() - subset['T2M_MIN'].min(),
        'InSeason_RH2M_Mean': subset['RH2M'].mean(),
        'InSeason_Solar_Mean': subset['ALLSKY_SFC_SW_DWN'].mean(),
        'InSeason_Wind_Mean': subset['WS2M'].mean(),
        'InSeason_Weather_Days': n_days,
    }
    
    return features, n_days, max_date


def main():
    base_dir = Path(__file__).resolve().parent.parent
    weather_dir = base_dir / "data/raw/weather"
    existing_path = base_dir / "data/processed/rice_yield_forecast_features_mandi_wb.csv"
    output_path = base_dir / "data/processed/rice_yield_inseason_features_wb.csv"

    # 1. Load existing leakage-safe historical+mandi features
    print("Loading existing feature dataset...")
    df_existing = pd.read_csv(existing_path)
    print(f"  Existing shape: {df_existing.shape}")

    # 2. Load all daily weather
    print("Loading daily NASA POWER weather data...")
    df_daily = load_daily_weather(weather_dir)
    print(f"  Daily weather rows: {len(df_daily)}")
    print(f"  Date range: {df_daily['Date'].min()} to {df_daily['Date'].max()}")
    print(f"  Districts: {df_daily['District'].nunique()}")

    # 3. Compute in-season weather features for every District-Year-Season
    print("\nComputing in-season weather features (strictly up to cutoff)...")
    weather_rows = []
    audit_rows = []

    for _, row in df_existing[['District', 'Year', 'Season']].drop_duplicates().iterrows():
        district, year, season = row['District'], row['Year'], row['Season']
        features, n_days, max_date = compute_weather_features(df_daily, district, year, season)
        
        start, cutoff = get_season_dates(year, season)
        
        if features is not None:
            features['District'] = district
            features['Year'] = year
            features['Season'] = season
            weather_rows.append(features)
            
            audit_rows.append({
                'District': district, 'Year': year, 'Season': season,
                'Season_Start': start.date(), 'Cutoff': cutoff.date(),
                'Max_Obs_Date': max_date.date() if max_date is not None else None,
                'N_Days': n_days,
                'Cutoff_Respected': max_date <= cutoff if max_date is not None else True,
            })

    df_weather = pd.DataFrame(weather_rows)
    df_audit = pd.DataFrame(audit_rows)

    print(f"  Weather features computed for {len(df_weather)} District-Year-Season keys")

    # 4. LEAKAGE AUDIT: verify max observation date <= cutoff
    violations = df_audit[~df_audit['Cutoff_Respected']]
    if len(violations) > 0:
        print(f"  LEAKAGE VIOLATION: {len(violations)} rows have observations after cutoff!")
        print(violations)
        raise RuntimeError("Cutoff leakage detected!")
    else:
        print("  Leakage audit PASSED: all weather observations <= cutoff date")

    # 5. Handle NDVI
    # The MODIS NDVI dataset only has full-season aggregates (no daily/sub-seasonal dates).
    # We CANNOT construct a genuine "NDVI up to cutoff" feature from this data.
    # Using the full-season NDVI would constitute future leakage.
    # Therefore, we use ONLY historical NDVI (NDVI_lag1, already in the existing features).
    print("\n  NDVI: Full-season aggregates only available — cannot slice to cutoff.")
    print("  Using NDVI_lag1 (historical) only. No current-season NDVI included.")

    # 6. Merge weather features into existing dataset
    print("\nMerging in-season weather with existing features...")
    original_rows = len(df_existing)
    df_merged = pd.merge(df_existing, df_weather, on=['District', 'Year', 'Season'], how='left')
    
    assert len(df_merged) == original_rows, "Row multiplication during merge!"
    dupes = df_merged.duplicated(subset=['District', 'Year', 'Season']).sum()
    assert dupes == 0, f"Duplicate keys: {dupes}"
    
    print(f"  Rows before: {original_rows}")
    print(f"  Rows after: {len(df_merged)}")
    print(f"  Duplicate keys: {dupes}")

    # 7. Missingness report
    inseason_cols = [c for c in df_merged.columns if c.startswith('InSeason_')]
    print(f"\nIn-season weather feature missingness:")
    print(df_merged[inseason_cols].isna().sum())

    # 8. Save
    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Final shape: {df_merged.shape}")
    print(f"Final columns: {df_merged.columns.tolist()}")

    # 9. Save audit
    audit_path = base_dir / "data/processed/inseason_weather_audit.csv"
    df_audit.to_csv(audit_path, index=False)
    print(f"Audit saved to {audit_path}")

if __name__ == '__main__':
    main()
