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
    """
    Load a trained model from disk.

    Returns:
        dict with keys: 'model', 'feature_names', 'use_log_target', 'model_name'
    """
    model_path = model_path or YIELD_MODEL_FILE

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Run 'python main.py' first to train and save a model."
        )

    model_info = joblib.load(model_path)
    print(f"[OK] Model loaded from: {model_path}")

    # Backward compatibility: if saved as raw model (not dict), wrap it
    if not isinstance(model_info, dict):
        model_info = {
            "model": model_info,
            "feature_names": None,
            "use_log_target": False,
            "model_name": "unknown",
        }

    return model_info


def predict_yield(model_info, input_data, feature_columns=None):
    """
    Predict tea arrivals for a single input.

    Args:
        model_info: Dict from load_model() containing 'model', 'feature_names',
                    'use_log_target'. Also accepts a raw model for backward compat.
        input_data: Dictionary of feature values
        feature_columns: List of feature column names the model expects

    Returns:
        Predicted arrival (float, tonnes) — in original scale
    """
    # Extract the actual model and metadata from the dict
    if isinstance(model_info, dict):
        estimator = model_info["model"]
        use_log_target = model_info.get("use_log_target", False)
        if feature_columns is None:
            feature_columns = model_info.get("feature_names") or list(input_data.keys())
    else:
        # Backward compat: raw model object passed directly
        estimator = model_info
        use_log_target = False
        if feature_columns is None:
            feature_columns = list(input_data.keys())

    input_df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_columns]
    input_df = input_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    prediction = estimator.predict(input_df)[0]

    # Inverse the log1p transform if the model was trained on log-scaled target
    if use_log_target:
        prediction = np.expm1(prediction)

    return float(prediction)


def predict_batch(model_info, input_df, feature_columns=None):
    """
    Predict for multiple inputs.

    Args:
        model_info: Dict from load_model() or raw model object.
        input_df: DataFrame of inputs.
        feature_columns: List of feature column names.

    Returns:
        numpy array of predictions in original scale.
    """
    if isinstance(model_info, dict):
        estimator = model_info["model"]
        use_log_target = model_info.get("use_log_target", False)
        if feature_columns is None:
            feature_columns = model_info.get("feature_names")
    else:
        estimator = model_info
        use_log_target = False

    if feature_columns is None:
        raise ValueError("feature_columns must be provided for batch prediction.")

    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    X = input_df[feature_columns].copy()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    predictions = estimator.predict(X)

    if use_log_target:
        predictions = np.expm1(predictions)

    return predictions


if __name__ == "__main__":
    model_info = load_model()
    print(f"[OK] Model '{model_info.get('model_name', 'unknown')}' loaded successfully.")
    print(f"     Features: {len(model_info.get('feature_names', []))} columns")
    print(f"     Log target: {model_info.get('use_log_target', False)}")
    print("     Ready for predictions.")
