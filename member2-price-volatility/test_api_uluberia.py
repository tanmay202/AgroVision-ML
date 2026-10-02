import pandas as pd
import requests
import json
import time
import subprocess
import os

# Start the API server
print("Starting uvicorn server...")
proc = subprocess.Popen(["python", "-m", "uvicorn", "src.api:app", "--port", "8000"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(3)

try:
    print("\n--- Testing GET /health ---")
    resp = requests.get("http://127.0.0.1:8000/health")
    print(resp.status_code)
    print(resp.json())
    
    print("\n--- Testing POST /predict ---")
    # Fetch real Uluberia + Other data
    df = pd.read_csv("data/processed/rice_cleaned.csv")
    market = "Uluberia"
    variety = "Other"
    
    sub = df[(df["Market Name"] == market) & (df["Variety"] == variety)].sort_values("Reported Date")
    
    # Need at least 30 rows. Let's take 35 rows.
    if len(sub) < 35:
        print("Not enough history for Uluberia/Other.")
    else:
        history_rows = sub.iloc[:35]
        pred_date = history_rows.iloc[-1]["Reported Date"]
        
        history = []
        for _, row in history_rows.iterrows():
            history.append({
                "reported_date": str(row["Reported Date"]),
                "modal_price": float(row["Modal Price (Rs./Quintal)"]),
                "arrivals": float(row["Arrivals (Tonnes)"])
            })
            
        payload = {
            "market": market,
            "variety": variety,
            "prediction_date": str(pred_date),
            "history": history
        }
        
        resp = requests.post("http://127.0.0.1:8000/predict", json=payload)
        print(f"Status Code: {resp.status_code}")
        print(json.dumps(resp.json(), indent=2))
        
finally:
    proc.terminate()
