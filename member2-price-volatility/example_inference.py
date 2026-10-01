import sys
import pandas as pd
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline
from config import raw_data_path, set_commodity

set_commodity("rice")

def run():
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path())
    
    # Pick a known market and variety
    # Rice -> Burdwan / Rice
    market = "Burdwan"
    variety = "Common" # or check what exists
    
    # Filter just to see
    sub = df[df["Market Name"] == market]
    if not sub.empty:
        variety = sub["Variety"].iloc[0]
        
    print(f"Selected Market: {market}, Variety: {variety}")
    
    # Let's say prediction date is the last available date in that subset
    pred_date = sub["Reported Date"].max()
    print(f"Prediction Date: {pred_date}")
    
    pipeline = RiceInferencePipeline()
    result = pipeline.predict(df, market, variety, pred_date)
    
    print("\nInference Output:")
    for k, v in result.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    run()
