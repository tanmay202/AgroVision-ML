"""
Phase 1 Forensic Baseline Audit & Phase 2 Data Exploration
Investigates Wheat and Potato datasets and V1 models.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def audit_commodity(commodity):
    print(f"\n=======================================================")
    print(f"=== PHASE 1 FORENSIC AUDIT: {commodity.upper()} ===")
    print(f"=======================================================")
    
    raw_path = Path(f"member2-price-volatility/data/raw/{commodity}/{commodity}.csv")
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    art_dir = Path(f"member2-price-volatility/artifacts/{commodity}")
    
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    train_df = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    test_df = pd.read_csv(proc_dir / f"{commodity}_test.csv")
    vol_train_df = pd.read_csv(proc_dir / f"{commodity}_volatility_dataset_train.csv")
    vol_test_df = pd.read_csv(proc_dir / f"{commodity}_volatility_dataset_test.csv")
    
    # 1. Dates and Chronological Split
    clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"])
    train_df["Reported Date"] = pd.to_datetime(train_df["Reported Date"])
    test_df["Reported Date"] = pd.to_datetime(test_df["Reported Date"])
    
    print(f"\n1. CHRONOLOGICAL SPLIT:")
    print(f"   Cleaned date range: {clean_df['Reported Date'].min().date()} to {clean_df['Reported Date'].max().date()}")
    print(f"   Train date range:   {train_df['Reported Date'].min().date()} to {train_df['Reported Date'].max().date()} ({len(train_df):,} rows)")
    print(f"   Test date range:    {test_df['Reported Date'].min().date()} to {test_df['Reported Date'].max().date()} ({len(test_df):,} rows)")
    
    # Check split overlap
    split_overlap = set(train_df["Reported Date"]).intersection(set(test_df["Reported Date"]))
    print(f"   Date overlap between train and test: {len(split_overlap)} dates")
    
    # 2. Market-Variety Groups & Test-only Groups
    train_groups = set(zip(train_df["Market Name"], train_df["Variety"]))
    test_groups = set(zip(test_df["Market Name"], test_df["Variety"]))
    test_only_groups = test_groups - train_groups
    train_only_groups = train_groups - test_groups
    common_groups = train_groups.intersection(test_groups)
    
    print(f"\n2. MARKET-VARIETY GROUPS:")
    print(f"   Train groups: {len(train_groups)}")
    print(f"   Test groups: {len(test_groups)}")
    print(f"   Common groups: {len(common_groups)}")
    print(f"   Train-only groups: {len(train_only_groups)}")
    print(f"   Test-only groups ({len(test_only_groups)}): {test_only_groups}")
    
    # Check test-only group row counts in test
    test_only_mask = test_df.apply(lambda r: (r["Market Name"], r["Variety"]) in test_only_groups, axis=1)
    print(f"   Rows from test-only groups in test set: {test_only_mask.sum():,} ({test_only_mask.mean()*100:.2f}%)")
    
    # 3. Volatility Shift
    print(f"\n3. VOLATILITY TARGET & REGIME SHIFT:")
    train_vol_col = "volatility" if "volatility" in vol_train_df.columns else "future_volatility_class"
    test_vol_col = "volatility" if "volatility" in vol_test_df.columns else "future_volatility_class"
    
    train_dist = vol_train_df[train_vol_col].value_counts(normalize=True) * 100
    test_dist = vol_test_df[test_vol_col].value_counts(normalize=True) * 100
    print("   Train Volatility Distribution:")
    for k, v in train_dist.items():
        print(f"     {k}: {v:.2f}% ({vol_train_df[train_vol_col].value_counts()[k]:,} rows)")
    print("   Test Volatility Distribution:")
    for k, v in test_dist.items():
        print(f"     {k}: {v:.2f}% ({vol_test_df[test_vol_col].value_counts()[k]:,} rows)")
        
    # Volatility thresholds in config
    vol_cfg = json.loads((art_dir / f"{commodity}_volatility_config.json").read_text())
    print(f"   Configured thresholds: Low/Med = {vol_cfg.get('low_medium_threshold')}, Med/High = {vol_cfg.get('medium_high_threshold')}")
    print(f"   Threshold derivation note: {vol_cfg.get('threshold_method')}")
    
    # 4. Flat-price and price change frequency
    print(f"\n4. FLAT-PRICE & ZERO MOVEMENT FREQUENCY:")
    cur_price = clean_df["Modal Price (Rs./Quintal)"]
    # Grouped shift
    clean_sorted = clean_df.sort_values(["Market Name", "Variety", "Reported Date"]).copy()
    clean_sorted["prev_price"] = clean_sorted.groupby(["Market Name", "Variety"])["Modal Price (Rs./Quintal)"].shift(1)
    clean_sorted["price_diff"] = clean_sorted["Modal Price (Rs./Quintal)"] - clean_sorted["prev_price"]
    clean_valid = clean_sorted.dropna(subset=["price_diff"])
    zero_diff = (clean_valid["price_diff"] == 0).mean() * 100
    print(f"   Zero consecutive price movement across entire cleaned series: {zero_diff:.2f}%")
    
    train_zero = (train_df["future_modal_price"] == train_df["Modal Price (Rs./Quintal)"]).mean() * 100 if "future_modal_price" in train_df.columns else 0
    test_zero = (test_df["future_modal_price"] == test_df["Modal Price (Rs./Quintal)"]).mean() * 100 if "future_modal_price" in test_df.columns else 0
    print(f"   Zero future target change in Train: {train_zero:.2f}%")
    print(f"   Zero future target change in Test:  {test_zero:.2f}%")
    
    # 5. Feature Importances of V1 Models
    print(f"\n5. FEATURE IMPORTANCE AUDIT (V1 Models):")
    price_model = joblib.load(art_dir / f"{commodity}_price_model.pkl")
    vol_model = joblib.load(art_dir / f"{commodity}_volatility_model.pkl")
    feat_cfg = json.loads((art_dir / f"{commodity}_feature_config.json").read_text())
    
    p_feats = feat_cfg["price_model"]["features"] if "price_model" in feat_cfg else feat_cfg["price_features"]
    p_imp = pd.Series(price_model.feature_importances_, index=p_feats).sort_values(ascending=False)
    print("   Top 8 Price Model Features:")
    for f, imp in p_imp.head(8).items():
        print(f"     {f:25s}: {imp:.4f} ({imp*100:.1f}%)")
        
    v_feats = feat_cfg["volatility_model"]["features"] if "volatility_model" in feat_cfg else feat_cfg.get("volatility_features", p_feats)
    v_imp = pd.Series(vol_model.feature_importances_, index=v_feats).sort_values(ascending=False)
    print("   Top 8 Volatility Model Features:")
    for f, imp in v_imp.head(8).items():
        print(f"     {f:25s}: {imp:.4f} ({imp*100:.1f}%)")
        
    # Check date features dominance (Potato issue flagged in task)
    date_feats = ["year", "month", "day", "day_of_week", "week_of_year"]
    p_date_share = p_imp.loc[p_imp.index.intersection(date_feats)].sum() * 100
    v_date_share = v_imp.loc[v_imp.index.intersection(date_feats)].sum() * 100
    print(f"   Calendar features total importance: Price = {p_date_share:.1f}%, Volatility = {v_date_share:.1f}%")
    
    # 6. Cleaning Jumps / Outliers
    print(f"\n6. CLEANING STEP & JUMPS:")
    print(f"   Raw rows: {len(raw_df):,} -> Cleaned rows: {len(clean_df):,} (Dropped {len(raw_df)-len(clean_df):,} rows)")
    # Check raw vs clean price bounds
    print(f"   Raw Modal Price: min={raw_df['Modal Price (Rs./Quintal)'].min()}, max={raw_df['Modal Price (Rs./Quintal)'].max()}, median={raw_df['Modal Price (Rs./Quintal)'].median()}")
    print(f"   Clean Modal Price: min={clean_df['Modal Price (Rs./Quintal)'].min()}, max={clean_df['Modal Price (Rs./Quintal)'].max()}, median={clean_df['Modal Price (Rs./Quintal)'].median()}")
    
    # 7. Reporting Gaps (calendar days between observations)
    clean_sorted["date_diff"] = (clean_sorted["Reported Date"] - clean_sorted.groupby(["Market Name", "Variety"])["Reported Date"].shift(1)).dt.days
    print(f"\n7. REPORTING GAPS (Inter-observation calendar days):")
    gaps = clean_sorted["date_diff"].dropna()
    print(f"   Mean gap: {gaps.mean():.2f} days, Median: {gaps.median():.0f} days")
    print(f"   1 day (consecutive): {(gaps == 1).mean()*100:.1f}%")
    print(f"   2-3 days: {((gaps >= 2) & (gaps <= 3)).mean()*100:.1f}%")
    print(f"   4-7 days: {((gaps >= 4) & (gaps <= 7)).mean()*100:.1f}%")
    print(f"   >7 days: {(gaps > 7).mean()*100:.1f}% (max gap: {gaps.max():.0f} days)")

if __name__ == "__main__":
    audit_commodity("wheat")
    audit_commodity("potato")
