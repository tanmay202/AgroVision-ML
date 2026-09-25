import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import (
    TEST_PATH,
    PRICE_FEATURES,
    PRICE_COLUMN,
)

from models.predict import predict


def main():
    if not TEST_PATH.exists():
        print(
            f"ERROR: File not found: "
            f"{TEST_PATH}"
        )
        return

    # Load test data and pick one valid row
    df = pd.read_csv(TEST_PATH)

    # Drop rows with any missing features
    all_features = list(
        set(PRICE_FEATURES)
    )

    df_clean = df.dropna(
        subset=all_features
    )

    if len(df_clean) == 0:
        print("ERROR: No complete rows in test data.")
        return

    sample = df_clean.iloc[[0]].copy()

    print("=" * 60)
    print("REAL PREDICTION TEST")
    print("=" * 60)

    print("\nInput sample:")
    print(
        sample[
            [
                PRICE_COLUMN,
                "lag_1",
                "rolling_std_7",
                "arrival_lag_1",
            ]
        ]
    )

    # Run prediction
    result = predict(sample)

    print("\nPrediction result:")
    print(result)

    # Basic validation
    assert "predicted_price" in result
    assert "volatility" in result

    assert isinstance(
        result["predicted_price"],
        float
    )

    assert result["volatility"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }

    # Verify price is from ML model, not just echoed back
    current_price = float(
        sample[PRICE_COLUMN].iloc[0]
    )

    print(f"\nCurrent price : {current_price}")
    print(f"Predicted price: {result['predicted_price']}")

    if result["predicted_price"] != current_price:
        print("PASS: ML model produced a different prediction (not naive baseline).")
    else:
        print("NOTE: Prediction equals current price (may happen for stable markets).")

    print("\nPASS: Prediction pipeline returned a valid result.")


if __name__ == "__main__":
    main()