"""
Phase 2: Advanced Data Quality and Exploration Script
Generates data quality statistics and plots for Wheat and Potato.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def explore_commodity(commodity):
    print(f"\n=======================================================")
    print(f"=== PHASE 2 DATA QUALITY & EXPLORATION: {commodity.upper()} ===")
    print(f"=======================================================")
    
    fig_dir = Path(f"reports/{commodity}/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    raw_path = Path(f"member2-price-volatility/data/raw/{commodity}/{commodity}.csv")
    proc_dir = Path(f"member2-price-volatility/data/processed/{commodity}")
    
    raw_df = pd.read_csv(raw_path)
    clean_df = pd.read_csv(proc_dir / f"{commodity}_cleaned.csv")
    
    raw_df["Reported Date"] = pd.to_datetime(raw_df["Reported Date"])
    clean_df["Reported Date"] = pd.to_datetime(clean_df["Reported Date"])
    
    # 1. Price Distribution Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(clean_df["Modal Price (Rs./Quintal)"], bins=60, kde=True, color="teal")
    plt.title(f"{commodity.capitalize()} Cleaned Modal Price Distribution (2001–2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Modal Price (Rs./Quintal)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "price_distribution.png", dpi=150)
    plt.close()
    
    # 2. Monthly and Yearly Trends
    clean_df["Year"] = clean_df["Reported Date"].dt.year
    clean_df["Month"] = clean_df["Reported Date"].dt.month
    
    yearly_avg = clean_df.groupby("Year")["Modal Price (Rs./Quintal)"].agg(["mean", "std", "count"]).reset_index()
    monthly_avg = clean_df.groupby("Month")["Modal Price (Rs./Quintal)"].agg(["mean", "std"]).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(yearly_avg["Year"], yearly_avg["mean"], marker="o", color="navy", linewidth=2)
    axes[0].fill_between(yearly_avg["Year"], yearly_avg["mean"] - yearly_avg["std"], yearly_avg["mean"] + yearly_avg["std"], alpha=0.2, color="navy")
    axes[0].set_title(f"Yearly Mean Modal Price (Rs./Quintal)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Price (Rs./Quintal)")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(monthly_avg["Month"], monthly_avg["mean"], marker="s", color="crimson", linewidth=2)
    axes[1].fill_between(monthly_avg["Month"], monthly_avg["mean"] - monthly_avg["std"], monthly_avg["mean"] + monthly_avg["std"], alpha=0.2, color="crimson")
    axes[1].set_title(f"Seasonal Monthly Modal Price (Rs./Quintal)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Month of Year")
    axes[1].set_xticks(range(1, 13))
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(fig_dir / "yearly_and_monthly_trends.png", dpi=150)
    plt.close()
    
    # 3. Market-wise Price Trends (Top 6 active markets)
    top_markets = clean_df["Market Name"].value_counts().head(6).index
    plt.figure(figsize=(12, 6))
    for mkt in top_markets:
        mkt_data = clean_df[clean_df["Market Name"] == mkt].sort_values("Reported Date")
        # Rolling 30d mean for smooth plot
        mkt_resampled = mkt_data.set_index("Reported Date")["Modal Price (Rs./Quintal)"].resample("ME").mean()
        plt.plot(mkt_resampled.index, mkt_resampled.values, label=mkt, linewidth=1.5)
    plt.title(f"Long-Term Modal Price by Top Markets ({commodity.capitalize()})", fontsize=14, fontweight="bold")
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Monthly Mean Modal Price (Rs./Quintal)", fontsize=12)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "market_wise_price_trends.png", dpi=150)
    plt.close()
    
    # 4. Arrival vs Modal Price
    plt.figure(figsize=(10, 5))
    sample_sub = clean_df.sample(min(15000, len(clean_df)), random_state=42)
    plt.scatter(np.log1p(sample_sub["Arrivals (Tonnes)"]), sample_sub["Modal Price (Rs./Quintal)"], alpha=0.15, s=10, color="purple")
    plt.title(f"Arrivals (log scale) vs Modal Price ({commodity.capitalize()})", fontsize=14, fontweight="bold")
    plt.xlabel("log(1 + Arrivals (Tonnes))", fontsize=12)
    plt.ylabel("Modal Price (Rs./Quintal)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "arrivals_vs_modal_price.png", dpi=150)
    plt.close()
    
    # 5. Price Movement Distribution
    clean_sorted = clean_df.sort_values(["Market Name", "Variety", "Reported Date"]).copy()
    clean_sorted["prev_price"] = clean_sorted.groupby(["Market Name", "Variety"])["Modal Price (Rs./Quintal)"].shift(1)
    clean_sorted["price_diff"] = clean_sorted["Modal Price (Rs./Quintal)"] - clean_sorted["prev_price"]
    clean_sorted["pct_diff"] = (clean_sorted["price_diff"] / clean_sorted["prev_price"]) * 100
    
    plt.figure(figsize=(10, 5))
    diffs = clean_sorted["price_diff"].dropna()
    sns.histplot(diffs[(diffs >= -100) & (diffs <= 100)], bins=80, color="darkgreen")
    plt.title(f"Inter-Day Price Movement Distribution [-100, +100 Rs.] ({commodity.capitalize()})", fontsize=14, fontweight="bold")
    plt.xlabel("Daily Price Change (Rs./Quintal)", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "price_movement_distribution.png", dpi=150)
    plt.close()
    
    # 6. Inter-observation Date Gaps
    clean_sorted["date_diff"] = (clean_sorted["Reported Date"] - clean_sorted.groupby(["Market Name", "Variety"])["Reported Date"].shift(1)).dt.days
    plt.figure(figsize=(10, 5))
    gaps = clean_sorted["date_diff"].dropna()
    sns.histplot(gaps[gaps <= 14], bins=14, discrete=True, color="coral")
    plt.title(f"Inter-Observation Calendar Gaps [<= 14 Days] ({commodity.capitalize()})", fontsize=14, fontweight="bold")
    plt.xlabel("Calendar Days to Next Observation", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "inter_observation_date_gaps.png", dpi=150)
    plt.close()
    
    print(f"Generated 6 diagnostic plots in {fig_dir}")

if __name__ == "__main__":
    explore_commodity("wheat")
    explore_commodity("potato")
