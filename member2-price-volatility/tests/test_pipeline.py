"""
AgroVision — Pipeline Integration Test

Tests the end-to-end pipeline with real tea data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def test_full_pipeline_tea():
    """Run the complete pipeline for tea and verify outputs."""
    from config import (
        set_commodity,
        raw_data_path,
        train_path,
        test_path,
        price_model_path,
        volatility_model_path,
        feature_config_path,
    )

    set_commodity("tea")

    # Verify raw data exists
    raw = raw_data_path()
    if not raw.exists():
        import pytest
        pytest.skip(f"Raw data not found: {raw}")

    # Run pipeline
    from pipeline import run_pipeline
    results = run_pipeline("tea")

    # Verify outputs
    assert results is not None
    assert "price_metrics" in results
    assert results["price_metrics"]["r2"] > 0, "R² should be positive"
    assert results["price_metrics"]["mae"] > 0, "MAE should be positive"

    # Verify files created
    assert train_path().exists(), "Train CSV not created"
    assert test_path().exists(), "Test CSV not created"
    assert price_model_path().exists(), "Price model not saved"
    assert volatility_model_path().exists(), "Volatility model not saved"
    assert feature_config_path().exists(), "Feature config not saved"

    print(f"\nPipeline test PASSED")
    print(f"  R²   = {results['price_metrics']['r2']:.4f}")
    print(f"  MAE  = {results['price_metrics']['mae']:.2f}")
    print(f"  RMSE = {results['price_metrics']['rmse']:.2f}")


def test_prediction_after_pipeline():
    """Test that predictions work after pipeline runs."""
    from config import set_commodity, test_path, PRICE_FEATURES
    import pandas as pd

    set_commodity("tea")

    t_path = test_path()
    if not t_path.exists():
        import pytest
        pytest.skip("Test data not found — run test_full_pipeline_tea first")

    test_df = pd.read_csv(t_path)

    # Find a row with all features present
    available = [f for f in PRICE_FEATURES if f in test_df.columns]
    clean_rows = test_df.dropna(subset=available)

    if len(clean_rows) == 0:
        import pytest
        pytest.skip("No complete rows in test data")

    sample = clean_rows.iloc[[0]]

    from models.predict import predict
    result = predict(sample, commodity="tea")

    assert "predicted_price" in result
    assert "volatility" in result
    assert isinstance(result["predicted_price"], float)
    assert result["volatility"] in {"LOW", "MEDIUM", "HIGH"}
    assert result["commodity"] == "tea"

    print(f"\nPrediction test PASSED")
    print(f"  Price: {result['predicted_price']}")
    print(f"  Volatility: {result['volatility']}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "-s"])
