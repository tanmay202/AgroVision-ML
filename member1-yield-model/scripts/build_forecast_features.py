import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # 1. Load the merged dataset
    base_dir = Path(__file__).resolve().parent.parent
    input_path = base_dir / "data/processed/merged_rice_yield_weather_ndvi.csv"
    output_path = base_dir / "data/processed/rice_yield_forecast_features_wb.csv"
    
    print(f"Loading merged dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Initial shape: {df.shape}")
    
    # 2. Apply Selected Cutoff (Scenario: Pre-Planting / Start of Season)
    # Since the dataset only contains full-season aggregates, any current-season
    # weather or NDVI feature contains future information (data from during the season).
    # Therefore, we define the forecast cutoff as "Start of the Crop Season".
    # All current-season variables (except identifiers) are LEAKY and must be excluded.
    
    # Sort data chronologically to ensure safe historical shifts
    df = df.sort_values(by=['District', 'Season', 'Year']).reset_index(drop=True)
    
    # 3. Construct Leakage-Safe Features (Historical Lags)
    print("Constructing leakage-safe historical features...")
    
    # We group by District and Season, because rice varieties and conditions 
    # differ drastically between Autumn, Winter, and Summer seasons.
    group_cols = ['District', 'Season']
    
    # Historical Yield Lags
    df['Yield_lag1'] = df.groupby(group_cols)['Rice_Yield_kg_ha'].shift(1)
    df['Yield_lag2'] = df.groupby(group_cols)['Rice_Yield_kg_ha'].shift(2)
    
    # Historical Area Lags
    df['Area_lag1'] = df.groupby(group_cols)['Area'].shift(1)
    df['Area_lag2'] = df.groupby(group_cols)['Area'].shift(2)
    
    # Historical Weather Lags
    df['Precip_lag1'] = df.groupby(group_cols)['PRECTOTCORR'].shift(1)
    df['T2M_lag1'] = df.groupby(group_cols)['T2M'].shift(1)
    df['RH2M_lag1'] = df.groupby(group_cols)['RH2M'].shift(1)
    
    # Historical NDVI Lags
    df['NDVI_lag1'] = df.groupby(group_cols)['NDVI'].shift(1)
    
    # Expanding Historical Averages (safely shifted to exclude current year)
    # This computes the historical average yield up to, but not including, the current year.
    df['Yield_hist_mean'] = df.groupby(group_cols)['Rice_Yield_kg_ha'] \
                              .apply(lambda x: x.shift(1).expanding().mean()) \
                              .reset_index(level=[0, 1], drop=True)
    
    # 4. Exclude Forbidden / Leaky Columns
    leaky_cols = [
        'Production', 'yield', 'Rice_Yield_t_ha', 'Area',  # Current-season agricultural data
        'PRECTOTCORR', 'T2M', 'T2M_MIN', 'T2M_MAX', 'RH2M', 
        'ALLSKY_SFC_SW_DWN', 'WS2M', 'Weather_Days',       # Current-season weather
        'NDVI'                                             # Current-season NDVI
    ]
    df = df.drop(columns=leaky_cols)
    
    # 5. Preserve Identifiers & 6. Preserve Target
    # Remaining columns: State_Name, District, Year, Season, Crop, Rice_Yield_kg_ha, + Lags
    
    # Handle Missing Data Policy
    # We restrict the dataset to Year >= 2000 because MODIS NDVI data does not exist before 2000.
    # Lags for 2000 will naturally be NaN, which tree-based models can handle natively.
    # Undivided Bardhaman's missing environmental lags will remain NaN.
    print("Applying missing data policy (Filtering for Year >= 2000)...")
    df = df[df['Year'] >= 2000].copy()
    
    # 7. Validate Duplicates
    dupes = df.duplicated(subset=['District', 'Year', 'Season']).sum()
    print(f"Duplicate keys after feature engineering: {dupes}")
    assert dupes == 0, "Duplicate keys introduced!"
    
    # 8. Validate Missingness
    print("\nMissing values in the final forecast dataset:")
    print(df.isna().sum())
    
    # 9. Explicit Leakage Check
    assert 'Production' not in df.columns, "Leakage: Production found!"
    assert 'Area' not in df.columns, "Leakage: Current Area found!"
    assert 'PRECTOTCORR' not in df.columns, "Leakage: Current Weather found!"
    assert 'NDVI' not in df.columns, "Leakage: Current NDVI found!"
    
    # 10. Save
    print(f"\nSaving forecast-ready dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    
    print(f"Final shape: {df.shape}")
    print(f"Final features: {df.columns.tolist()}")

if __name__ == '__main__':
    main()
