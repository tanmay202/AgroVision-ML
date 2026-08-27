"""
AgroVision - Predict
Load the saved model and make predictions on new data.
This module is what Member 3 (Engineering) will call from the FastAPI service.
"""

import os
import numpy as np
import pandas as pd
import joblib

from src.config import YIELD_MODEL_FILE


def load_model(model_path=None):
    """Load a trained model from disk."""
    model_path = model_path or YIELD_MODEL_FILE

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run 'python main.py' first to train and save a model."
        )

    model = joblib.load(model_path)
    print(f"[OK] Model loaded from: {model_path}")
    return model


def predict_yield(model, input_data, feature_columns=None):
    """
    Predict tea arrivals for a single input.

    Args:
        model: Trained model
        input_data: Dictionary of feature values
        feature_columns: List of feature column names the model expects

    Returns:
        Predicted arrival (float, tonnes)
    """
    if feature_columns is None:
        feature_columns = list(input_data.keys())

    input_df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    prediction = model.predict(input_df)[0]
    return float(prediction)


def predict_batch(model, input_df, feature_columns):
    """Predict for multiple inputs."""
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    X = input_df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return model.predict(X)


if __name__ == "__main__":
    model = load_model()
    print("[OK] Model loaded successfully. Ready for predictions.")
