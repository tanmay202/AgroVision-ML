import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from inference import RiceInferencePipeline, InsufficientHistoryError
from config import set_commodity

set_commodity("rice")

@pytest.fixture
def dummy_historical_data():
    dates = pd.date_range("2024-01-01", periods=40)
    data = []
    
    # Market A, Variety X (40 rows)
    for i, d in enumerate(dates):
        data.append({
            "Market Name": "MarketA",
            "Variety": "VarietyX",
            "Reported Date": d,
            "Modal Price (Rs./Quintal)": 2000 + i * 10,
            "Arrivals (Tonnes)": 100 + i
        })
        
    # Market B, Variety Y (short history - 10 rows)
    for i, d in enumerate(dates[:10]):
        data.append({
            "Market Name": "MarketB",
            "Variety": "VarietyY",
            "Reported Date": d,
            "Modal Price (Rs./Quintal)": 3000,
            "Arrivals (Tonnes)": 50
        })
        
    return pd.DataFrame(data)

def test_insufficient_history(dummy_historical_data):
    pipeline = RiceInferencePipeline()
    with pytest.raises(InsufficientHistoryError):
        pipeline.predict(dummy_historical_data, "MarketB", "VarietyY", "2024-01-10")

def test_missing_values(dummy_historical_data):
    # Introduce NaNs
    df = dummy_historical_data.copy()
    df.loc[38, "Modal Price (Rs./Quintal)"] = np.nan
    df.loc[39, "Arrivals (Tonnes)"] = np.nan
    
    pipeline = RiceInferencePipeline()
    result = pipeline.predict(df, "MarketA", "VarietyX", "2024-02-09")
    
    assert "predicted_modal_price" in result
    assert not pd.isna(result["predicted_modal_price"])

def test_volatility_persistence(dummy_historical_data):
    # Setup specific last 5 prices to control volatility output
    df = dummy_historical_data.copy()
    mask = (df["Market Name"] == "MarketA")
    # Last 5 prices: indices 35 to 39
    # Let's set them to exactly 2000, 2000, 2000, 2000, 2000 (Range 0 -> LOW)
    df.loc[mask & (df.index >= 35), "Modal Price (Rs./Quintal)"] = 2000.0
    
    pipeline = RiceInferencePipeline()
    res1 = pipeline.predict(df, "MarketA", "VarietyX", "2024-02-09")
    assert res1["predicted_volatility_class"] == "LOW"
    assert res1["predicted_volatility_range"] == 0.0
    
    # Range 40 -> MEDIUM
    df.loc[mask & (df.index == 35), "Modal Price (Rs./Quintal)"] = 2000.0
    df.loc[mask & (df.index == 39), "Modal Price (Rs./Quintal)"] = 2040.0
    res2 = pipeline.predict(df, "MarketA", "VarietyX", "2024-02-09")
    assert res2["predicted_volatility_class"] == "MEDIUM"
    assert res2["predicted_volatility_range"] == 40.0
    
    # Range 100 -> HIGH
    df.loc[mask & (df.index == 35), "Modal Price (Rs./Quintal)"] = 2000.0
    df.loc[mask & (df.index == 39), "Modal Price (Rs./Quintal)"] = 2100.0
    res3 = pipeline.predict(df, "MarketA", "VarietyX", "2024-02-09")
    assert res3["predicted_volatility_class"] == "HIGH"
    assert res3["predicted_volatility_range"] == 100.0

def test_end_to_end_inference(dummy_historical_data):
    pipeline = RiceInferencePipeline()
    result = pipeline.predict(dummy_historical_data, "MarketA", "VarietyX", "2024-02-09")
    
    assert result["market"] == "MarketA"
    assert result["variety"] == "VarietyX"
    assert result["prediction_date"] == "2024-02-09"
    assert isinstance(result["predicted_modal_price"], float)
    assert isinstance(result["predicted_volatility_range"], float)
    assert result["predicted_volatility_class"] in ["LOW", "MEDIUM", "HIGH"]

def test_feature_ordering_and_leakage(dummy_historical_data):
    pipeline = RiceInferencePipeline()
    clean_df = pipeline._validate_and_clean_input(dummy_historical_data, "MarketA", "VarietyX", "2024-02-09")
    feat_df = pipeline._generate_features(clean_df)
    last_row = feat_df.iloc[-1]
    
    # Check that required features are present
    for f in pipeline.features_list:
        assert f in feat_df.columns
        
    # X construction respects the order in config
    X = last_row[pipeline.features_list].to_frame().T
    assert list(X.columns) == pipeline.features_list
    
    # Ensure no leakage (future targets or percentages are NOT in the feature list)
    assert "future_modal_price" not in pipeline.features_list
    assert "future_price_pct_change" not in pipeline.features_list
    assert "future_volatility" not in pipeline.features_list
