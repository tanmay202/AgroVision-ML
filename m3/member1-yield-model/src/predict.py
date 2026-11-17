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
    INSEASON_WEATHER_FEATURES,
    VALID_DISTRICTS,
    VALID_SEASONS,
    POLICY_VERSION,
    PROVISIONAL_SUMMER_ROUTING_POLICY,
    preprocess_inseason_weather,
    get_season_window,
    INSEASON_FEATURE_COLUMNS,
)

from weather_service import (  # noqa: E402
    NASAWeatherService,
    get_weather_service,
)

from wheat_predict import (  # noqa: E402
    WheatYieldPredictor,
    predict_wheat_yield,
)

from potato_predict import (  # noqa: E402
    PotatoYieldPredictor,
    predict_potato_yield,
)

__all__ = [
    "RiceYieldPredictor",
    "predict_rice_yield",
    "load_model",
    "predict_yield",
    "WheatYieldPredictor",
    "predict_wheat_yield",
    "PotatoYieldPredictor",
    "predict_potato_yield",
    "NASAWeatherService",
    "get_weather_service",
    "ALL_FEATURES",
    "NUMERICAL_FEATURES",
    "CATEGORICAL_FEATURES",
    "INSEASON_WEATHER_FEATURES",
    "VALID_DISTRICTS",
    "VALID_SEASONS",
    "POLICY_VERSION",
    "PROVISIONAL_SUMMER_ROUTING_POLICY",
    "preprocess_inseason_weather",
    "get_season_window",
    "INSEASON_FEATURE_COLUMNS",
]
