import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    mandi_path = base_dir.parent / "member2-price-volatility/data/processed/rice_cleaned.csv"
    
    df = pd.read_csv(mandi_path)
    print("--- MANDI DISTRICT MAPPING AUDIT ---")
    print(f"Original Mandi rows: {len(df)}")
    
    explosions = {
        'Burdwan': ['Bardhaman', 'Purba Bardhaman', 'Paschim Bardhaman'],
        'Medinipur(W)': ['Paschim Medinipur', 'Jhargram'],
        'Jalpaiguri': ['Jalpaiguri', 'Alipurduar'],
        'Darjeeling': ['Darjeeling', 'Kalimpong']
    }
    
    total_original = 0
    total_generated = 0
    
    for k, v in explosions.items():
        subset = df[df['District Name'] == k]
        count = len(subset)
        generated = count * len(v)
        total_original += count
        total_generated += generated
        years = sorted(pd.to_datetime(subset['Reported Date']).dt.year.unique().tolist())
        markets = subset['Market Name'].nunique()
        print(f"\nSource: {k}")
        print(f"Targets: {v}")
        print(f"Observation Count: {count}")
        print(f"Rows Generated: {generated}")
        print(f"Inflation/Duplicated: +{generated - count}")
        print(f"Years affected: {min(years)} - {max(years)}")
        print(f"Markets affected: {markets}")
        
    print(f"\nTotal rows before mapping: {len(df)}")
    print(f"Total rows inflated due to these 4 districts: +{total_generated - total_original}")
    print(f"Expected rows after old mapping: {len(df) + (total_generated - total_original)}")

if __name__ == '__main__':
    main()
