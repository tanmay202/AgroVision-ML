import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    yield_path = base_dir / "data/processed/rice_yield_wb_clean.csv"
    weather_path = base_dir / "data/processed/weather_rice_wb.csv"
    ndvi_path = base_dir / "data/raw/ndvi/AgroVision_WB_Rice_NDVI_2000_2019_LONG.csv"

    df_yield = pd.read_csv(yield_path)
    df_weather = pd.read_csv(weather_path)
    df_ndvi = pd.read_csv(ndvi_path)

    df_yield = df_yield.rename(columns={'District_Name': 'District', 'Crop_Year': 'Year'})

    print("STEP 1: AUDIT")
    print("-" * 50)
    
    y_dist = set(df_yield['District'].unique())
    w_dist = set(df_weather['District'].unique())
    n_dist = set(df_ndvi['District'].unique())
    
    print("Exact unique district names:")
    print(f"Yield ({len(y_dist)}): {sorted(list(y_dist))}")
    print(f"Weather ({len(w_dist)}): {sorted(list(w_dist))}")
    print(f"NDVI ({len(n_dist)}): {sorted(list(n_dist))}")
    
    print("\nMismatched District Names (Yield vs Weather):")
    print(f"In Yield only: {sorted(list(y_dist - w_dist))}")
    print(f"In Weather only: {sorted(list(w_dist - y_dist))}")
    
    print("\nMismatched District Names (Yield vs NDVI):")
    print(f"In Yield only: {sorted(list(y_dist - n_dist))}")
    print(f"In NDVI only: {sorted(list(n_dist - y_dist))}")

    print("\nYear Ranges:")
    print(f"Yield: {df_yield['Year'].min()} to {df_yield['Year'].max()}")
    print(f"Weather: {df_weather['Year'].min()} to {df_weather['Year'].max()}")
    print(f"NDVI: {df_ndvi['Year'].min()} to {df_ndvi['Year'].max()}")
    
    print("\nSeason Values:")
    print(f"Yield: {df_yield['Season'].unique()}")
    print(f"Weather: {df_weather['Season'].unique()}")
    print(f"NDVI: {df_ndvi['Season'].unique()}")

    # Merge without fixing to show the problem
    merged_1 = pd.merge(df_yield, df_weather, on=['District', 'Year', 'Season'], how='left', indicator='_merge_weather')
    merged_2 = pd.merge(merged_1, df_ndvi, on=['District', 'Year', 'Season'], how='left', indicator='_merge_ndvi')

    missing_weather = merged_2[merged_2['_merge_weather'] == 'left_only']
    missing_ndvi_2000 = merged_2[(merged_2['_merge_ndvi'] == 'left_only') & (merged_2['Year'] >= 2000)]

    print("\nMissing Keys (Without Normalization):")
    print(f"Yield rows missing Weather: {len(missing_weather)}")
    print(f"Yield rows (Year >= 2000) missing NDVI: {len(missing_ndvi_2000)}")
    
    print("\nCounts of missing Weather keys by District:")
    print(missing_weather['District'].value_counts().head(10))
    print("\nCounts of missing NDVI keys (>=2000) by District:")
    print(missing_ndvi_2000['District'].value_counts().head(10))
    
    print("\nExamples of mismatched keys (Top 5 missing weather):")
    print(missing_weather[['District', 'Year', 'Season']].head())

if __name__ == '__main__':
    main()
