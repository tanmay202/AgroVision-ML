"""
Phase 6: Comprehensive Volatility Target Engineering & Horizon Analysis
Compares:
1. 5-observation absolute price range
2. 5-observation percentage price range ((max - min) / current_price * 100)
3. 7-calendar-day absolute price range
4. 7-calendar-day percentage price range
5. 14-calendar-day absolute price range
6. 14-calendar-day percentage price range
Computes calendar spans, zero-movement frequencies, quantile & zero-aware thresholds,
and regime shift stability from training to holdout test.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_volatility_targets(commodity):
    print(f"\n=======================================================")
    print(f"=== PHASE 6 VOLATILITY TARGET ANALYSIS: {commodity.upper()} ===")
    print(f"=======================================================")
    
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    train_df = pd.read_csv(proc_dir / f"{commodity}_train.csv")
    test_df = pd.read_csv(proc_dir / f"{commodity}_test.csv")
    
    clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"])
    train_df["Reported Date"] = pd.to_datetime(train_df["Reported Date"])
    test_df["Reported Date"] = pd.to_datetime(test_df["Reported Date"])
    
    group_cols = ["Market Name", "Variety"]
    price_col = "Modal Price (Rs./Quintal)"
    date_col = "Reported Date"
    
    # Analyze on clean_df sorted
    df = clean_df.sort_values(group_cols + [date_col]).reset_index(drop=True)
    grp_p = df.groupby(group_cols)[price_col]
    grp_d = df.groupby(group_cols)[date_col]
    
    # 1. 5-observation window
    p_next = [grp_p.shift(-i) for i in range(1, 6)]
    d_next = [grp_d.shift(-i) for i in range(1, 6)]
    
    p_next_df = pd.concat(p_next, axis=1)
    # Require all 5 observations
    valid_5obs = p_next_df.notna().all(axis=1)
    
    range_5obs = p_next_df.max(axis=1) - p_next_df.min(axis=1)
    pct_range_5obs = (range_5obs / df[price_col]) * 100.0
    cal_span_5obs = (d_next[-1] - df[date_col]).dt.days
    
    print(f"\n1. FIVE-OBSERVATION WINDOW (W=5):")
    valid_cal = cal_span_5obs[valid_5obs]
    print(f"   Calendar days represented: Mean={valid_cal.mean():.1f}d, Median={valid_cal.median():.0f}d, P90={valid_cal.quantile(0.90):.0f}d, Max={valid_cal.max():.0f}d")
    print(f"   Zero price range across 5 observations: {(range_5obs[valid_5obs] == 0).mean()*100:.2f}%")
    print(f"   Median non-zero absolute range: Rs.{range_5obs[valid_5obs & (range_5obs > 0)].median():.2f}")
    print(f"   Median non-zero percentage range: {pct_range_5obs[valid_5obs & (range_5obs > 0)].median():.2f}%")
    
    # 2. Direct calendar horizon windows: 7-day [T+1, T+7] and 14-day [T+1, T+14]
    # For each observation t, find all observations in (t, t + H days] within same group
    # Compute max price - min price in that calendar interval
    print(f"\n2. DIRECT CALENDAR HORIZON WINDOWS (7-day and 14-day):")
    
    def compute_calendar_window_volatility(df_sub, horizon_days):
        ranges = []
        pct_ranges = []
        counts = []
        for _, group in df_sub.groupby(group_cols, sort=False):
            dates = group[date_col].values
            prices = group[price_col].values
            n = len(dates)
            for i in range(n):
                t0 = dates[i]
                t_end = t0 + np.timedelta64(horizon_days, "D")
                # find observations strictly between t0 and t_end
                idx_start = i + 1
                if idx_start >= n:
                    ranges.append(np.nan)
                    pct_ranges.append(np.nan)
                    counts.append(0)
                    continue
                # searchsorted
                idx_end = np.searchsorted(dates, t_end, side="right")
                window_prices = prices[idx_start:idx_end]
                if len(window_prices) >= 1:
                    r = np.max(window_prices) - np.min(window_prices)
                    ranges.append(r)
                    pct_ranges.append((r / prices[i]) * 100.0)
                    counts.append(len(window_prices))
                else:
                    ranges.append(np.nan)
                    pct_ranges.append(np.nan)
                    counts.append(0)
        return np.array(ranges), np.array(pct_ranges), np.array(counts)

    # Sample if needed or run full
    sample_df = df.copy() if len(df) < 100000 else df.sample(80000, random_state=42).sort_values(group_cols + [date_col]).reset_index(drop=True)
    r7, pct7, c7 = compute_calendar_window_volatility(sample_df, 7)
    valid7 = ~np.isnan(r7)
    print(f"   7-Calendar-Day Window:")
    print(f"     Valid observations: {valid7.sum():,} / {len(sample_df):,} ({valid7.mean()*100:.1f}%)")
    print(f"     Observations in window: Mean={c7[valid7].mean():.1f}, Median={np.median(c7[valid7]):.0f}")
    print(f"     Zero price range: {(r7[valid7] == 0).mean()*100:.2f}%")
    print(f"     Median non-zero absolute range: Rs.{np.median(r7[valid7 & (r7 > 0)]):.2f}")
    print(f"     Median non-zero percentage range: {np.median(pct7[valid7 & (r7 > 0)]):.2f}%")
    
    r14, pct14, c14 = compute_calendar_window_volatility(sample_df, 14)
    valid14 = ~np.isnan(r14)
    print(f"   14-Calendar-Day Window:")
    print(f"     Valid observations: {valid14.sum():,} / {len(sample_df):,} ({valid14.mean()*100:.1f}%)")
    print(f"     Observations in window: Mean={c14[valid14].mean():.1f}, Median={np.median(c14[valid14]):.0f}")
    print(f"     Zero price range: {(r14[valid14] == 0).mean()*100:.2f}%")
    print(f"     Median non-zero absolute range: Rs.{np.median(r14[valid14 & (r14 > 0)]):.2f}")
    print(f"     Median non-zero percentage range: {np.median(pct14[valid14 & (r14 > 0)]):.2f}%")

    # 3. Training-Derived Thresholds Comparison (Absolute vs Normalized)
    # Split train and test
    split_date = train_df[date_col].max()
    train_mask = df[date_col] <= split_date
    test_mask = df[date_col] > split_date
    
    print(f"\n3. TRAINING-DERIVED THRESHOLDS & STABILITY AUDIT:")
    # A: 5-obs Absolute range (V1 method)
    train_r5 = range_5obs[valid_5obs & train_mask]
    test_r5 = range_5obs[valid_5obs & test_mask]
    
    # Zero-aware: <= 0 is LOW; non-zero median splits MED/HIGH
    low_thresh_abs = 0.5
    high_thresh_abs = train_r5[train_r5 > 0.5].median()
    print(f"   V1 Absolute Range Thresholds: Low/Med = Rs.{low_thresh_abs:.1f}, Med/High = Rs.{high_thresh_abs:.1f}")
    train_classes_abs = pd.cut(train_r5, bins=[-np.inf, low_thresh_abs, high_thresh_abs, np.inf], labels=["LOW", "MEDIUM", "HIGH"])
    test_classes_abs = pd.cut(test_r5, bins=[-np.inf, low_thresh_abs, high_thresh_abs, np.inf], labels=["LOW", "MEDIUM", "HIGH"])
    print(f"     Train distribution (Absolute): {train_classes_abs.value_counts(normalize=True).to_dict()}")
    print(f"     Test distribution (Absolute):  {test_classes_abs.value_counts(normalize=True).to_dict()}")
    
    # B: 5-obs Normalized Percentage range
    train_pct5 = pct_range_5obs[valid_5obs & train_mask]
    test_pct5 = pct_range_5obs[valid_5obs & test_mask]
    
    # Flat threshold: <= 0.05% is LOW (accommodating minor roundoff / flat)
    flat_thresh_pct = 0.05
    med_high_pct = train_pct5[train_pct5 > flat_thresh_pct].median()
    print(f"\n   Normalized Percentage Range Thresholds: Low/Med = {flat_thresh_pct:.3f}%, Med/High = {med_high_pct:.3f}%")
    train_classes_pct = pd.cut(train_pct5, bins=[-np.inf, flat_thresh_pct, med_high_pct, np.inf], labels=["LOW", "MEDIUM", "HIGH"])
    test_classes_pct = pd.cut(test_pct5, bins=[-np.inf, flat_thresh_pct, med_high_pct, np.inf], labels=["LOW", "MEDIUM", "HIGH"])
    print(f"     Train distribution (Normalized): {train_classes_pct.value_counts(normalize=True).to_dict()}")
    print(f"     Test distribution (Normalized):  {test_classes_pct.value_counts(normalize=True).to_dict()}")

if __name__ == "__main__":
    analyze_volatility_targets("wheat")
    analyze_volatility_targets("potato")
