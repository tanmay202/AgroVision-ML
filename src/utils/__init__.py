"""
AgroVision — Shared Utilities

Common functions used across the pipeline:
- Metrics calculation (MAPE, RMSE, MAE, R2)
- Logging setup
- Path helpers
"""

import logging
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def setup_logger(name, level=logging.INFO):
    """Create a configured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(name)s — %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def calculate_mape(actual, predicted):
    """Calculate MAPE while ignoring zero actual values."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(
        np.abs((actual[mask] - predicted[mask]) / actual[mask])
    ) * 100


def calculate_metrics(actual, predicted):
    """
    Calculate regression metrics.

    Returns
    -------
    dict with keys: mae, rmse, r2, mape
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = calculate_mape(actual, predicted)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }


def print_metrics(metrics, model_name="Model"):
    """Print formatted metrics."""
    print(f"\n{'=' * 60}")
    print(f"{model_name} METRICS")
    print(f"{'=' * 60}")
    print(f"MAE  : {metrics['mae']:.2f}")
    print(f"RMSE : {metrics['rmse']:.2f}")
    print(f"R2   : {metrics['r2']:.4f}")
    if metrics['mape'] is not None and not np.isnan(metrics['mape']):
        print(f"MAPE : {metrics['mape']:.2f}%")
