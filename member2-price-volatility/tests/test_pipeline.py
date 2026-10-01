"""
AgroVision — Pipeline Integration Test

Tests the end-to-end Member 2 pipeline with Rice data.
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_full_pipeline_rice():
    """Run the complete pipeline for rice and verify outputs."""

    from config import (
        set_commodity,
        raw_data_path,
        train_path,
        test_path,
        price_model_path,
        volatility_model_path,
        feature_config_path,
        volatility_config_path,
    )

    # --------------------------------------------------
    # Use Rice
    # --------------------------------------------------
    set_commodity("rice")

    # --------------------------------------------------
    # Verify raw data exists
    # --------------------------------------------------
    raw = raw_data_path()

    if not raw.exists():
        import pytest
        pytest.skip(
            f"Rice raw data not found: {raw}"
        )

    # --------------------------------------------------
    # Run complete Rice pipeline
    # --------------------------------------------------
    from pipeline import run_pipeline

    results = run_pipeline("rice")

    # --------------------------------------------------
    # Verify returned results
    # --------------------------------------------------
    assert results is not None

    assert "price_metrics" in results

    assert (
        results["price_metrics"]["r2"] > 0
    ), "R² should be positive"

    assert (
        results["price_metrics"]["mae"] > 0
    ), "MAE should be positive"

    assert (
        results["price_metrics"]["rmse"] > 0
    ), "RMSE should be positive"

    # --------------------------------------------------
    # Verify processed files
    # --------------------------------------------------
    assert train_path().exists(), (
        "Rice train CSV not created"
    )

    assert test_path().exists(), (
        "Rice test CSV not created"
    )

    # --------------------------------------------------
    # Verify model artifacts
    # --------------------------------------------------
    assert price_model_path().exists(), (
        "Rice price model not saved"
    )

    assert volatility_model_path().exists(), (
        "Rice volatility model not saved"
    )

    assert feature_config_path().exists(), (
        "Rice feature config not saved"
    )

    assert volatility_config_path().exists(), (
        "Rice volatility config not saved"
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("RICE PIPELINE TEST PASSED")
    print("=" * 60)

    print(
        f"  R²   = "
        f"{results['price_metrics']['r2']:.4f}"
    )

    print(
        f"  MAE  = "
        f"{results['price_metrics']['mae']:.2f}"
    )

    print(
        f"  RMSE = "
        f"{results['price_metrics']['rmse']:.2f}"
    )


def test_prediction_after_pipeline():
    """
    Test that Rice predictions work after the pipeline runs.
    """

    import pandas as pd

    from config import (
        set_commodity,
        test_path,
        PRICE_FEATURES,
    )

    # --------------------------------------------------
    # Use Rice
    # --------------------------------------------------
    set_commodity("rice")

    # --------------------------------------------------
    # Verify Rice test data exists
    # --------------------------------------------------
    t_path = test_path()

    if not t_path.exists():
        import pytest
        pytest.skip(
            "Rice test data not found — "
            "run test_full_pipeline_rice first"
        )

    # --------------------------------------------------
    # Load Rice test data
    # --------------------------------------------------
    test_df = pd.read_csv(t_path)

    # --------------------------------------------------
    # Find configured price features that exist
    # --------------------------------------------------
    available = [
        feature
        for feature in PRICE_FEATURES
        if feature in test_df.columns
    ]

    if not available:
        import pytest
        pytest.skip(
            "No configured price features "
            "were found in Rice test data"
        )

    # --------------------------------------------------
    # Find row without missing required features
    # --------------------------------------------------
    clean_rows = test_df.dropna(
        subset=available
    )

    if len(clean_rows) == 0:
        import pytest
        pytest.skip(
            "No complete Rice rows available "
            "for prediction test"
        )

    sample = clean_rows.iloc[[0]].copy()

    # --------------------------------------------------
    # Run Rice prediction
    # --------------------------------------------------
    from models.predict import predict

    result = predict(
        sample,
        commodity="rice"
    )

    # --------------------------------------------------
    # Validate result
    # --------------------------------------------------
    assert "predicted_price" in result, (
        "Prediction result missing predicted_price"
    )

    assert "volatility" in result, (
        "Prediction result missing volatility"
    )

    assert "commodity" in result, (
        "Prediction result missing commodity"
    )

    assert isinstance(
        result["predicted_price"],
        (float, int)
    ), (
        "predicted_price should be numeric"
    )

    # Rice modal prices are in Rs./Quintal.  A small raw price-change value
    # such as -0.02 is not a valid public price prediction.
    assert result["predicted_price"] > 100, (
        "predicted_price must be reconstructed on the Rice price scale"
    )

    assert result["volatility"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }, (
        "Invalid volatility class"
    )

    assert result["commodity"] == "rice", (
        "Prediction returned wrong commodity"
    )

    # --------------------------------------------------
    # Print prediction result
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("RICE PREDICTION TEST PASSED")
    print("=" * 60)

    print(
        f"  Predicted Price: "
        f"{result['predicted_price']}"
    )

    print(
        f"  Volatility: "
        f"{result['volatility']}"
    )

    print(
        f"  Commodity: "
        f"{result['commodity']}"
    )


if __name__ == "__main__":
    import pytest

    pytest.main([
        __file__,
        "-v",
        "-s",
    ])
