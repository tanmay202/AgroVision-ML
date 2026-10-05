import pandas as pd
import numpy as np
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    mandi_path = base_dir.parent / "member2-price-volatility/data/processed/rice_cleaned.csv"
    yield_path = base_dir / "data/processed/rice_yield_forecast_features_wb.csv"
    out_mandi_path = base_dir / "data/processed/mandi_rice_historical_features_wb.csv"
    out_merged_path = base_dir / "data/processed/rice_yield_forecast_features_mandi_wb.csv"
    
    print("Loading Mandi data...")
    df_mandi = pd.read_csv(mandi_path)
    print(f"Original Mandi rows: {len(df_mandi)}")
    
    # --- 1. DISTRICT NORMALIZATION ---
    # Map Mandi district names to canonical Yield district names.
    # We expand districts that were split (e.g. Burdwan -> Bardhaman, Purba Bardhaman, Paschim Bardhaman)
    # so that the historical mandi data joins correctly to all fragments in the yield dataset.
    district_map = {
        'Coochbehar': ['Cooch Behar'],
        'Burdwan': ['Bardhaman'],
        'Puruliya': ['Purulia'],
        'Medinipur(E)': ['Purba Medinipur'],
        'Medinipur(W)': ['Paschim Medinipur'],
        'Sounth 24 Parganas': ['South 24 Parganas'],
        'Malda': ['Maldah'],
        'Jalpaiguri': ['Jalpaiguri'],
        'Darjeeling': ['Darjeeling'],
        'North 24 Parganas': ['North 24 Parganas'],
        'Bankura': ['Bankura'],
        'Birbhum': ['Birbhum'],
        'Dakshin Dinajpur': ['Dakshin Dinajpur'],
        'Hooghly': ['Hooghly'],
        'Howrah': ['Howrah'],
        'Murshidabad': ['Murshidabad'],
        'Nadia': ['Nadia'],
        'Uttar Dinajpur': ['Uttar Dinajpur']
    }
    
    # Explode the mapping so one mandi row becomes multiple rows for split districts
    df_mandi['Mapped_Districts'] = df_mandi['District Name'].map(district_map)
    df_mandi = df_mandi.explode('Mapped_Districts').dropna(subset=['Mapped_Districts'])
    df_mandi['District'] = df_mandi['Mapped_Districts']
    
    print(f"Rows after district expansion: {len(df_mandi)}")
    
    # --- 2. TEMPORAL AGGREGATION & SEASON MAPPING ---
    df_mandi['Reported Date'] = pd.to_datetime(df_mandi['Reported Date'])
    df_mandi['Cal_Year'] = df_mandi['Reported Date'].dt.year
    df_mandi['Month'] = df_mandi['Reported Date'].dt.month
    
    # Date Windows used for Crop Seasons:
    # Summer (Boro): Harvested Mar-May. Arrivals mapped to Mar, Apr, May, Jun, Jul (Same Cal Year)
    # Autumn (Aus): Harvested Aug-Sep. Arrivals mapped to Aug, Sep, Oct (Same Cal Year)
    # Winter (Aman): Harvested Nov-Dec. Arrivals mapped to Nov, Dec (Same Cal Year) and Jan, Feb (Cal Year - 1)
    
    def assign_season(row):
        m = row['Month']
        if m in [3, 4, 5, 6, 7]: return 'Summer', row['Cal_Year']
        elif m in [8, 9, 10]: return 'Autumn', row['Cal_Year']
        elif m in [11, 12]: return 'Winter', row['Cal_Year']
        elif m in [1, 2]: return 'Winter', row['Cal_Year'] - 1
        return None, None

    df_mandi[['Season', 'Year']] = df_mandi.apply(assign_season, axis=1, result_type='expand')
    
    # Aggregate features by District, Year, Season
    print("Aggregating Mandi features...")
    agg_funcs = {
        'Arrivals (Tonnes)': ['sum', 'mean'],
        'Modal Price (Rs./Quintal)': ['mean', 'min', 'max', 'std'],
        'Reported Date': ['count']
    }
    mandi_agg = df_mandi.groupby(['District', 'Year', 'Season']).agg(agg_funcs).reset_index()
    
    # Flatten multi-level columns
    mandi_agg.columns = ['District', 'Year', 'Season', 
                         'Mandi_Arrivals_Total', 'Mandi_Arrivals_Mean',
                         'Mandi_Price_Mean', 'Mandi_Price_Min', 'Mandi_Price_Max', 'Mandi_Price_Vol',
                         'Mandi_Obs_Count']
    
    # Add Price Range
    mandi_agg['Mandi_Price_Range'] = mandi_agg['Mandi_Price_Max'] - mandi_agg['Mandi_Price_Min']
    
    # Fill NaN volatility (std) with 0 where count is 1
    mandi_agg['Mandi_Price_Vol'] = mandi_agg['Mandi_Price_Vol'].fillna(0)
    
    mandi_agg.to_csv(out_mandi_path, index=False)
    
    # --- 3. STRICT PRE-PLANTING LEAKAGE RULE (LAGGING) ---
    print("Creating historical lags to strictly prevent leakage...")
    
    # We must construct lags so they can be joined to the main dataset
    # We sort by District, Season, Year and shift
    mandi_agg = mandi_agg.sort_values(['District', 'Season', 'Year']).reset_index(drop=True)
    
    lag_features = [
        'Mandi_Arrivals_Total', 'Mandi_Arrivals_Mean',
        'Mandi_Price_Mean', 'Mandi_Price_Min', 'Mandi_Price_Max', 'Mandi_Price_Vol',
        'Mandi_Obs_Count', 'Mandi_Price_Range'
    ]
    
    # We need to shift within the mandi dataset, but wait:
    # If the mandi dataset is missing some years, shifting blindly will map e.g. 2005 to 2010.
    # To shift safely in pandas when years might be missing, we create a Year + 1 target
    mandi_lag1 = mandi_agg[['District', 'Season', 'Year'] + lag_features].copy()
    mandi_lag1['Year'] = mandi_lag1['Year'] + 1
    mandi_lag1.columns = ['District', 'Season', 'Year'] + [f"{c}_lag1" for c in lag_features]
    
    mandi_lag2 = mandi_agg[['District', 'Season', 'Year'] + lag_features].copy()
    mandi_lag2['Year'] = mandi_lag2['Year'] + 2
    mandi_lag2.columns = ['District', 'Season', 'Year'] + [f"{c}_lag2" for c in lag_features]
    
    # --- 4. MERGE WITH EXISTING YIELD FEATURES ---
    print("Merging with existing leakage-safe feature dataset...")
    df_yield = pd.read_csv(yield_path)
    original_rows = len(df_yield)
    
    # Left join lag1
    df_merged = pd.merge(df_yield, mandi_lag1, on=['District', 'Season', 'Year'], how='left')
    # Left join lag2
    df_merged = pd.merge(df_merged, mandi_lag2, on=['District', 'Season', 'Year'], how='left')
    
    print(f"Original rows: {original_rows}")
    print(f"Merged rows: {len(df_merged)}")
    assert len(df_merged) == original_rows, "Row multiplication detected during merge!"
    
    # Check for duplicate keys
    dupes = df_merged.duplicated(subset=['District', 'Year', 'Season']).sum()
    assert dupes == 0, "Duplicate keys detected!"
    
    # Leakage check: ensure no current-year mandi features exist
    for col in df_merged.columns:
        if 'Mandi' in col:
            assert 'lag' in col, f"Leakage detected: Current-year mandi feature found: {col}"
    
    # Save
    df_merged.to_csv(out_merged_path, index=False)
    print(f"Saved merged features to {out_merged_path}")
    print("Done.")

if __name__ == '__main__':
    main()
