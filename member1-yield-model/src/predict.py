"""
AgroVision Model 1: Production Predict Interface
===============================================
Exposes load_model() and predict_yield() for external bridge integration
(such as shared/predict_bridge.py) and imports core RiceYieldPredictor.
"""

from pathlib import Path
import sys

# Ensure scripts directory is importable
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from predict_rice_yield import (  # noqa: E402
    RiceYieldPredictor,
    predict_rice_yield,
    load_model,
    predict_yield,
    ALL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    VALID_DISTRICTS,
    VALID_SEASONS
)

__all__ = [
    "RiceYieldPredictor",
    "predict_rice_yield",
    "load_model",
    "predict_yield",
    "ALL_FEATURES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "VALID_DISTRICTS",
    "VALID_SEASONS"
]
