import sys
import pandas as pd
import numpy as np
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline
from config import raw_data_path, cleaned_data_path, features_final_path, set_commodity, PRICE_COLUMN, DATE_COLUMN

set_commodity("rice")

def debug():
    print("Loading datasets...")
    # Use cleaned data instead of raw data to skip complex outlier removal steps that inference shouldn't duplicate
    # Wait, the prompt says: "The input must represent historical observations available up to the prediction timestamp.
    # Current/future values must never be used as features for the frozen price model."
    # Let's use cleaned data to isolate feature-generation parity.
    raw_df = pd.read_csv(cleaned_data_path())
    feat_df = pd.read_csv(features_final_path())
    
    # Pick a random row from feat_df
    sample = feat_df.dropna().sample(1, random_state=42).iloc[0]
    market = sample["Market Name"]
    variety = sample["Variety"]
    pred_date = sample[DATE_COLUMN]
    
    print(f"Testing Market: {market}, Variety: {variety}, Date: {pred_date}")
    
    pipeline = RiceInferencePipeline()
    hist_df = raw_df[(raw_df["Market Name"] == market) & (raw_df["Variety"] == variety)]
    hist_df = hist_df[hist_df[DATE_COLUMN] <= pred_date].copy()
    
    # Run feature generation
    inf_clean = pipeline._validate_and_clean_input(raw_df, market, variety, pred_date)
    inf_feat = pipeline._generate_features(inf_clean).iloc[-1]
    
    print("\nComparing Features:")
    mismatches = 0
    for f in pipeline.features_list:
        train_val = sample[f]
        inf_val = inf_feat[f]
        diff = abs(train_val - inf_val) if pd.notna(train_val) and pd.notna(inf_val) else np.nan
        
        if pd.isna(train_val) and pd.isna(inf_val):
            status = "MATCH (NaN)"
        elif pd.isna(train_val) or pd.isna(inf_val):
            status = f"MISMATCH (NaN vs Val)"
            mismatches += 1
        elif diff < 1e-5:
            status = "MATCH"
        else:
            status = f"MISMATCH (diff: {diff:.6f})"
            mismatches += 1
            
        print(f"{f:<25} | Train: {train_val:<12.5f} | Inf: {inf_val:<12.5f} | {status}")
        
    print(f"\nTotal Mismatches: {mismatches}")

if __name__ == "__main__":
    debug()
