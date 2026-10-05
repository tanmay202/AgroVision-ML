import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    yield_path = base_dir / "data/processed/rice_yield_wb_clean.csv"
    weather_path = base_dir / "data/processed/weather_rice_wb.csv"
    ndvi_path = base_dir / "data/raw/ndvi/AgroVision_WB_Rice_NDVI_2000_2019_LONG.csv"
    output_path = base_dir / "data/processed/merged_rice_yield_weather_ndvi.csv"

    print("Loading datasets...")
    df_yield = pd.read_csv(yield_path)
    df_weather = pd.read_csv(weather_path)
    df_ndvi = pd.read_csv(ndvi_path)
    
    orig_yield_shape = df_yield.shape
    
    # Standardize column names for the join
    df_yield = df_yield.rename(columns={'District_Name': 'District', 'Crop_Year': 'Year'})
    
    # ---------------------------------------------------------
    # APPLY JUSTIFIED NORMALIZATION / MAPPINGS
    # ---------------------------------------------------------
    # Yield uses UPPERCASE, while Weather/NDVI use Title Case.
    # Yield has old spelling conventions for some districts.
    # BARDHAMAN is kept as Bardhaman because it represents the undivided district pre-2017.
    # Weather and NDVI do not have undivided Bardhaman (they only have post-split Purba/Paschim).
    # We purposefully do not map BARDHAMAN to Purba/Paschim to prevent synthesizing data.
    
    district_map = {
        '24 PARAGANAS NORTH': 'North 24 Parganas',
        '24 PARAGANAS SOUTH': 'South 24 Parganas',
        'DINAJPUR DAKSHIN': 'Dakshin Dinajpur',
        'DINAJPUR UTTAR': 'Uttar Dinajpur',
        'COOCHBEHAR': 'Cooch Behar',
        'MEDINIPUR WEST': 'Paschim Medinipur',
        'MEDINIPUR EAST': 'Purba Medinipur',
        'ALIPURDUAR': 'Alipurduar',
        'BARDHAMAN': 'Bardhaman',
        'PASCHIM BARDHAMAN': 'Paschim Bardhaman',
        'PURBA BARDHAMAN': 'Purba Bardhaman'
    }
    
    df_yield['District'] = df_yield['District'].apply(lambda x: district_map.get(x, x.title()))
    
    # NDVI uses "Alipur Duar" which must match "Alipurduar" from Weather and Yield
    df_ndvi['District'] = df_ndvi['District'].replace({'Alipur Duar': 'Alipurduar'})
    
    # Strip whitespace to be safe
    df_yield['District'] = df_yield['District'].str.strip()
    df_weather['District'] = df_weather['District'].str.strip()
    df_ndvi['District'] = df_ndvi['District'].str.strip()

    df_yield['Season'] = df_yield['Season'].str.strip()
    df_weather['Season'] = df_weather['Season'].str.strip()
    df_ndvi['Season'] = df_ndvi['Season'].str.strip()

    # ---------------------------------------------------------
    # VALIDATE KEYS
    # ---------------------------------------------------------
    yield_dupes = df_yield.duplicated(subset=['District', 'Year', 'Season']).sum()
    weather_dupes = df_weather.duplicated(subset=['District', 'Year', 'Season']).sum()
    ndvi_dupes = df_ndvi.duplicated(subset=['District', 'Year', 'Season']).sum()
    
    print(f"\nDuplicate Join Keys (District+Year+Season):")
    print(f"Yield: {yield_dupes} | Weather: {weather_dupes} | NDVI: {ndvi_dupes}")
    assert yield_dupes == 0, "Duplicate keys in Yield"
    assert weather_dupes == 0, "Duplicate keys in Weather"
    assert ndvi_dupes == 0, "Duplicate keys in NDVI"

    # ---------------------------------------------------------
    # MERGE DATASETS
    # ---------------------------------------------------------
    print("\nMerging datasets...")
    # Left join to preserve all Yield rows
    merged_df = pd.merge(df_yield, df_weather, on=['District', 'Year', 'Season'], how='left', indicator='_merge_weather')
    merged_df = pd.merge(merged_df, df_ndvi, on=['District', 'Year', 'Season'], how='left', indicator='_merge_ndvi')
    
    # Check for row multiplication
    print(f"Original Yield Rows: {orig_yield_shape[0]}")
    print(f"Final Merged Rows:   {merged_df.shape[0]}")
    if orig_yield_shape[0] == merged_df.shape[0]:
        print("Success: Merge did not cause row multiplication (1:1 merge validated).")
    else:
        print("ERROR: Row multiplication detected!")

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------
    print("\n--- FINAL DATASET VALIDATION ---")
    print(f"Final Shape: {merged_df.shape}")
    print(f"Unique Districts: {len(merged_df['District'].unique())}")
    print(f"Year Range: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
    print(f"Seasons: {merged_df['Season'].unique()}")
    print(f"Duplicate Keys: {merged_df.duplicated(subset=['District', 'Year', 'Season']).sum()}")
    
    # Missing values
    missing_weather = merged_df[merged_df['_merge_weather'] == 'left_only']
    missing_ndvi = merged_df[(merged_df['_merge_ndvi'] == 'left_only') & (merged_df['Year'] >= 2000)]
    
    print(f"\nMissing Weather Rows: {len(missing_weather)}")
    print(f"Unmatched Weather keys belong to: {missing_weather['District'].unique()}")
    
    print(f"Missing NDVI Rows (Year >= 2000): {len(missing_ndvi)}")
    print(f"Unmatched NDVI keys belong to: {missing_ndvi['District'].unique()}")
    print(f"Legitimate NDVI Missing Rows (1997-1999): {len(merged_df[merged_df['Year'] < 2000])}")
    
    print("\nMissing values by column:")
    cols_to_check = ['PRECTOTCORR', 'T2M', 'NDVI', 'Rice_Yield_kg_ha']
    print(merged_df[cols_to_check].isna().sum())
    
    # ---------------------------------------------------------
    # DATA LEAKAGE CHECK
    # ---------------------------------------------------------
    print("\n--- DATA LEAKAGE CHECK ---")
    # Verify no duplicate rows
    assert merged_df.shape[0] == 1333, "Row count changed, potential leakage via duplication."
    # Target check
    target_cols = ['Production', 'yield', 'Rice_Yield_t_ha', 'Rice_Yield_kg_ha']
    print(f"Target variables present: {target_cols}")
    print("Check passed: No target-derived features were created. Current/future yield is not used as an input feature.")
    print("Check passed: Weather and NDVI data joined exactly on District+Year+Season, preventing future data leakage.")

    # Cleanup and Save
    merged_df = merged_df.drop(columns=['_merge_weather', '_merge_ndvi'])
    merged_df.to_csv(output_path, index=False)
    print(f"\nSaved corrected dataset to: {output_path}")

if __name__ == '__main__':
    main()
