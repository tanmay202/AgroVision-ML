"""
AgroVision-ML Member 2 ML Quickstart Example for Member 3.

Demonstrates how to import and use RicePricePredictor in under 10 lines of code.
No FastAPI or uvicorn server needed.
"""

import sys
from pathlib import Path
import pandas as pd

# 1. Add deliverable root to Python path (if not installed as package)
DELIVERABLE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DELIVERABLE_ROOT))

from src.predictor import RicePricePredictor, InsufficientHistoryError

# 2. Instantiate the predictor (auto-discovers local artifacts directory)
predictor = RicePricePredictor()

# 3. Prepare historical mandi observations (pandas DataFrame or list of dicts with >= 30 rows)
# Sample data for Burdwan Common rice:
sample_history = []
base_price = 2500.0
for i in range(35):
    sample_history.append({
        "market": "Burdwan",
        "variety": "Common",
        "date": f"2024-01-{(i+1):02d}",
        "modal_price": base_price + (i % 5) * 15.0,
        "arrivals": 100.0 + (i % 3) * 10.0,
    })

print("=" * 60)
print("1. SHORT-TERM FORECAST (1–7 Days, Production)")
print("=" * 60)
short_term_result = predictor.predict(sample_history, horizon="short")
print(f"Market: {short_term_result['market']}")
print(f"Current Price: Rs. {short_term_result['current_modal_price']}")
print(f"Predicted Price: Rs. {short_term_result['predicted_modal_price']}")
print(f"Predicted Change: Rs. {short_term_result['predicted_price_change']}")
print(f"Volatility Class: {short_term_result.get('normalized_volatility_class')}")

print("\n" + "=" * 60)
print("2. 14-DAY CALENDAR HORIZON FORECAST (Opt-In)")
print("=" * 60)
res_14d = predictor.predict(sample_history, horizon=14)
print(f"Horizon Days: {res_14d['horizon_days']}")
print(f"Target Date: {res_14d['requested_target_date']}")
print(f"Predicted Price: Rs. {res_14d['predicted_modal_price']}")
print(f"Release Status: {res_14d['release_status']}")

print("\n" + "=" * 60)
print("3. 30-DAY CALENDAR HORIZON FORECAST (Experimental)")
print("=" * 60)
res_30d = predictor.predict(sample_history, horizon=30)
print(f"Horizon Days: {res_30d['horizon_days']}")
print(f"Target Date: {res_30d['requested_target_date']}")
print(f"Predicted Price: Rs. {res_30d['predicted_modal_price']}")
print(f"Release Status: {res_30d['release_status']}")
print("=" * 60)
