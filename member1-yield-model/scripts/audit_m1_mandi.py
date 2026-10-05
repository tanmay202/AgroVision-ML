import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    mandi_path = base_dir / "member2-price-volatility/data/processed/rice_cleaned.csv"
    yield_path = base_dir / "member1-yield-model/data/processed/rice_yield_forecast_features_wb.csv"
    
    print("Loading datasets...")
    df_mandi = pd.read_csv(mandi_path)
    df_yield = pd.read_csv(yield_path)
    
    print("\n--- MANDI AUDIT ---")
    print(f"Row count: {len(df_mandi)}")
    print(f"Date range: {df_mandi['Reported Date'].min()} to {df_mandi['Reported Date'].max()}")
    print(f"Districts: {sorted(df_mandi['District Name'].dropna().unique().tolist())}")
    print(f"Number of Markets: {df_mandi['Market Name'].nunique()}")
    print(f"Missing values:\n{df_mandi.isna().sum()}")
    print(f"Duplicate records: {df_mandi.duplicated().sum()}")
    print(f"Commodity consistency (Groups): {df_mandi['Group'].unique()}")
    print(f"Varieties: {df_mandi['Variety'].unique()}")
    
    print("\nPrice ranges:")
    print(df_mandi[['Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Modal Price (Rs./Quintal)']].describe())
    
    print("\nArrival ranges:")
    print(df_mandi['Arrivals (Tonnes)'].describe())
    
    # District matching
    mandi_districts = set(df_mandi['District Name'].unique())
    yield_districts = set(df_yield['District'].unique())
    
    print("\n--- DISTRICT INCONSISTENCIES ---")
    print(f"In Mandi, not in Yield: {mandi_districts - yield_districts}")
    print(f"In Yield, not in Mandi: {yield_districts - mandi_districts}")
    
    # Year/Season check
    df_mandi['Reported Date'] = pd.to_datetime(df_mandi['Reported Date'])
    df_mandi['Year'] = df_mandi['Reported Date'].dt.year
    df_mandi['Month'] = df_mandi['Reported Date'].dt.month
    
    print("\nMandi observations by month:")
    print(df_mandi['Month'].value_counts().sort_index())
    
    # Season mapping logic needed for Step 5
    # Standard West Bengal Rice Seasons:
    # Autumn (Aus): Sown May-June, Harvested Aug-Sep
    # Winter (Aman): Sown July-Aug, Harvested Nov-Dec
    # Summer (Boro): Sown Nov-Dec, Harvested March-May
    # For mandi data (market arrivals), arrivals usually happen during and after harvest.
    # We will map reported months to the corresponding crop season for aggregation.

if __name__ == '__main__':
    main()
