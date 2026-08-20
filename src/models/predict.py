import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from config import (
    PRICE_MODEL_PATH,
    VOLATILITY_MODEL_PATH,
    PRICE_FEATURES,
    VOLATILITY_FEATURES,
    VOLATILITY_REVERSE_LABEL_MAP,
)


# ============================================================
# Model cache — loaded once, reused on every call.
# ============================================================

_price_model = None
_volatility_model = None


def _get_price_model():
    """Load the saved XGBoost price model (cached)."""
    global _price_model

    if _price_model is None:
        if not PRICE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Price model not found: {PRICE_MODEL_PATH}"
            )
        _price_model = joblib.load(PRICE_MODEL_PATH)

    return _price_model


def _get_volatility_model():
    """Load the saved XGBoost volatility model (cached)."""
    global _volatility_model

    if _volatility_model is None:
        if not VOLATILITY_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Volatility model not found: "
                f"{VOLATILITY_MODEL_PATH}"
            )
        _volatility_model = joblib.load(
            VOLATILITY_MODEL_PATH
        )

    return _volatility_model


def predict(data):
    """
    Generate price and volatility predictions.

    Parameters
    ----------
    data : pandas.DataFrame
        One or more rows containing all required
        features (both price and volatility features).

    Returns
    -------
    dict
        {
            "predicted_price": float,
            "volatility": str  ("LOW", "MEDIUM", or "HIGH")
        }
    """

    # --------------------------------------------------
    # 1. Validate price features
    # --------------------------------------------------
    missing_price_features = [
        feature
        for feature in PRICE_FEATURES
        if feature not in data.columns
    ]

    if missing_price_features:
        raise ValueError(
            "Missing price features: "
            + ", ".join(missing_price_features)
        )

    # --------------------------------------------------
    # 2. Validate volatility features
    # --------------------------------------------------
    missing_vol_features = [
        feature
        for feature in VOLATILITY_FEATURES
        if feature not in data.columns
    ]

    if missing_vol_features:
        raise ValueError(
            "Missing volatility features: "
            + ", ".join(missing_vol_features)
        )

    # --------------------------------------------------
    # 3. Check for missing values
    # --------------------------------------------------
    all_features = list(
        set(PRICE_FEATURES + VOLATILITY_FEATURES)
    )

    missing_values = (
        data[all_features]
        .isna()
        .any()
        .any()
    )

    if missing_values:
        raise ValueError(
            "Input contains missing feature values."
        )

    # --------------------------------------------------
    # 4. Price prediction (XGBoost model)
    # --------------------------------------------------
    price_model = _get_price_model()

    X_price = data[PRICE_FEATURES].copy()

    predicted_price = float(
        price_model.predict(X_price)[0]
    )

    # --------------------------------------------------
    # 5. Volatility prediction (XGBoost model)
    # --------------------------------------------------
    vol_model = _get_volatility_model()

    X_vol = data[VOLATILITY_FEATURES].copy()

    predicted_class = vol_model.predict(X_vol)[0]

    volatility = VOLATILITY_REVERSE_LABEL_MAP[
        int(predicted_class)
    ]

    # --------------------------------------------------
    # 6. Final output
    # --------------------------------------------------
    return {
        "predicted_price": round(predicted_price, 2),
        "volatility": volatility,
    }


def main():
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)

    print(
        "Prediction module loaded successfully."
    )

    print("\nPrice model features:")
    for feature in PRICE_FEATURES:
        print(f"  - {feature}")

    print("\nVolatility model features:")
    for feature in VOLATILITY_FEATURES:
        print(f"  - {feature}")


if __name__ == "__main__":
    main()