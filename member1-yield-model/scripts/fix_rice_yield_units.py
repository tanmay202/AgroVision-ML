import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    clean_path = base_dir / "data/processed/rice_yield_wb_clean.csv"
    
    print(f"Loading {clean_path}...")
    df = pd.read_csv(clean_path)
    
    # 1. Before Statistics
    print("\n--- BEFORE CORRECTION ---")
    print(df['Rice_Yield_kg_ha'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))
    
    # Identify anomalous rows
    # The anomaly is exactly a factor of 100 (e.g., 250,000 kg/ha instead of 2,500 kg/ha).
    # Anything above 10,000 kg/ha (10 tonnes/ha) is biologically impossible for standard 
    # district-level average rice yields in West Bengal, making it a safe threshold.
    anomaly_mask = df['Rice_Yield_kg_ha'] > 10000
    num_anomalies = anomaly_mask.sum()
    print(f"\nIdentified {num_anomalies} rows with impossible yields (> 10,000 kg/ha).")
    
    # Log which districts have these anomalies
    print("\nAnomalous rows by District:")
    print(df[anomaly_mask]['District_Name'].value_counts())
    
    # 2. Apply Deterministic Correction
    # The source of the error was a systematic factor-of-100 multiplication of the Production column
    # in the raw data for the vast majority of records.
    print("\nApplying deterministic correction: Production = Production / 100")
    df.loc[anomaly_mask, 'Production'] = df.loc[anomaly_mask, 'Production'] / 100
    
    # 3. Recalculate Yield Columns
    print("Recalculating yield derived columns...")
    df['Rice_Yield_t_ha'] = df['Production'] / df['Area']
    df['Rice_Yield_kg_ha'] = df['Rice_Yield_t_ha'] * 1000
    df['yield'] = df['Rice_Yield_t_ha']  # Some earlier steps might rely on 'yield' column
    
    # 4. After Statistics
    print("\n--- AFTER CORRECTION ---")
    print(df['Rice_Yield_kg_ha'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))
    
    assert df['Rice_Yield_kg_ha'].max() < 10000, "Correction failed, impossible yields remain."
    
    # Save the corrected dataframe in-place (as requested by 'rebuild downstream files')
    print(f"\nSaving corrected dataset to {clean_path}...")
    df.to_csv(clean_path, index=False)
    print("Done.")

if __name__ == '__main__':
    main()
