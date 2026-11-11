"""
AgroVision Model 1: Rice Yield Prediction Pipeline
==================================================
Production-grade inference pipeline for rice yield prediction (kg/ha and tonnes/ha)
in West Bengal using candidate residual-target models.

Target Formulation:
    Predicted Yield (kg/ha) = Yield_lag1 + Predicted Delta (kg/ha)
    Predicted Yield (t/ha)  = Predicted Yield (kg/ha) / 1000.0

Default Model:
    Improved Delta Ridge (alpha=300.0)
Alternative Models:
    Improved Delta XGBoost Conservative (max_depth=2, lr=0.03)
    Improved Delta XGBoost (max_depth=3, lr=0.05)

Features:
    18 numerical features (lags, in-season weather aggregates) + 2 categorical (District, Season)
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("predict_rice_yield")

# Base paths
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from weather_preprocessor import (
        preprocess_inseason_weather,
        get_season_window,
        INSEASON_FEATURE_COLUMNS,
    )
except ImportError:
    from src.weather_preprocessor import (
        preprocess_inseason_weather,
        get_season_window,
        INSEASON_FEATURE_COLUMNS,
    )

# Feature specifications
NUMERICAL_FEATURES = [
    'Yield_lag1', 'Yield_lag2',
    'Area_lag1', 'Area_lag2',
    'Precip_lag1', 'T2M_lag1', 'RH2M_lag1', 'NDVI_lag1',
    'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
    'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
    'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
]
INSEASON_WEATHER_FEATURES = [
    'InSeason_Precip_Cum', 'InSeason_Precip_Mean', 'InSeason_Precip_Max',
    'InSeason_T2M_Mean', 'InSeason_T2M_Min', 'InSeason_T2M_Max', 'InSeason_T2M_Range',
    'InSeason_RH2M_Mean', 'InSeason_Solar_Mean', 'InSeason_Wind_Mean'
]
CATEGORICAL_FEATURES = ['District', 'Season']
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

POLICY_VERSION = "2.1"
PROVISIONAL_SUMMER_ROUTING_POLICY = "provisional_summer_persistence"

VALID_SEASONS = ['Autumn', 'Summer', 'Winter']

VALID_DISTRICTS = [
    'Alipurduar', 'Bankura', 'Bardhaman', 'Birbhum', 'Cooch Behar',
    'Dakshin Dinajpur', 'Darjeeling', 'Hooghly', 'Howrah',
    'Jalpaiguri', 'Jhargram', 'Kalimpong', 'Maldah', 'Murshidabad',
    'Nadia', 'North 24 Parganas', 'Paschim Bardhaman',
    'Paschim Medinipur', 'Purba Bardhaman', 'Purba Medinipur',
    'Purulia', 'South 24 Parganas', 'Uttar Dinajpur'
]

# District aliases for robust normalization across government / mandi datasets
DISTRICT_ALIASES = {
    '24 paraganas north': 'North 24 Parganas',
    '24 parganas north': 'North 24 Parganas',
    'north 24 parganas': 'North 24 Parganas',
    'north 24-parganas': 'North 24 Parganas',
    'north 24 pargana': 'North 24 Parganas',
    'north 24 paraganas': 'North 24 Parganas',
    '24 paraganas south': 'South 24 Parganas',
    '24 parganas south': 'South 24 Parganas',
    'south 24 parganas': 'South 24 Parganas',
    'south 24-parganas': 'South 24 Parganas',
    'south 24 pargana': 'South 24 Parganas',
    'south 24 paraganas': 'South 24 Parganas',
    'dinajpur dakshin': 'Dakshin Dinajpur',
    'dakshin dinajpur': 'Dakshin Dinajpur',
    'south dinajpur': 'Dakshin Dinajpur',
    'dinajpur uttar': 'Uttar Dinajpur',
    'uttar dinajpur': 'Uttar Dinajpur',
    'north dinajpur': 'Uttar Dinajpur',
    'coochbehar': 'Cooch Behar',
    'cooch behar': 'Cooch Behar',
    'koch bihar': 'Cooch Behar',
    'medinipur west': 'Paschim Medinipur',
    'paschim medinipur': 'Paschim Medinipur',
    'west midnapore': 'Paschim Medinipur',
    'midnapore west': 'Paschim Medinipur',
    'paschim midnapore': 'Paschim Medinipur',
    'medinipur east': 'Purba Medinipur',
    'purba medinipur': 'Purba Medinipur',
    'east midnapore': 'Purba Medinipur',
    'midnapore east': 'Purba Medinipur',
    'purba midnapore': 'Purba Medinipur',
    'alipurduar': 'Alipurduar',
    'alipur duar': 'Alipurduar',
    'bardhaman': 'Bardhaman',
    'burdwan': 'Bardhaman',
    'paschim bardhaman': 'Paschim Bardhaman',
    'west bardhaman': 'Paschim Bardhaman',
    'bardhaman west': 'Paschim Bardhaman',
    'west burdwan': 'Paschim Bardhaman',
    'burdwan west': 'Paschim Bardhaman',
    'purba bardhaman': 'Purba Bardhaman',
    'east bardhaman': 'Purba Bardhaman',
    'bardhaman east': 'Purba Bardhaman',
    'east burdwan': 'Purba Bardhaman',
    'burdwan east': 'Purba Bardhaman',
    'bankura': 'Bankura',
    'birbhum': 'Birbhum',
    'darjeeling': 'Darjeeling',
    'hooghly': 'Hooghly',
    'hugli': 'Hooghly',
    'howrah': 'Howrah',
    'haora': 'Howrah',
    'jalpaiguri': 'Jalpaiguri',
    'jhargram': 'Jhargram',
    'kalimpong': 'Kalimpong',
    'maldah': 'Maldah',
    'malda': 'Maldah',
    'murshidabad': 'Murshidabad',
    'nadia': 'Nadia',
    'purulia': 'Purulia'
}

# Seasonal aliases for agronomic conventions
SEASON_ALIASES = {
    'autumn': 'Autumn',
    'aus': 'Autumn',
    'summer': 'Summer',
    'boro': 'Summer',
    'rabi': 'Summer',
    'winter': 'Winter',
    'aman': 'Winter',
    'kharif': 'Winter'
}

# Historical fallback yield values (from Train+Val 2000-2016, kg/ha)
DS_MEANS = {
    "Alipurduar_Autumn": 2195.98, "Alipurduar_Summer": 2832.7, "Alipurduar_Winter": 1985.09,
    "Bankura_Autumn": 2690.69, "Bankura_Summer": 2863.55, "Bankura_Winter": 2736.34,
    "Bardhaman_Autumn": 2884.08, "Bardhaman_Summer": 3235.11, "Bardhaman_Winter": 2913.38,
    "Birbhum_Autumn": 2595.05, "Birbhum_Summer": 3236.21, "Birbhum_Winter": 2962.99,
    "Cooch Behar_Autumn": 1553.73, "Cooch Behar_Summer": 2789.1, "Cooch Behar_Winter": 1964.72,
    "Dakshin Dinajpur_Autumn": 1884.58, "Dakshin Dinajpur_Summer": 3025.94, "Dakshin Dinajpur_Winter": 2494.03,
    "Darjeeling_Autumn": 1756.87, "Darjeeling_Summer": 2210.51, "Darjeeling_Winter": 1994.45,
    "Hooghly_Autumn": 2525.45, "Hooghly_Summer": 3005.59, "Hooghly_Winter": 2786.71,
    "Howrah_Autumn": 2077.79, "Howrah_Summer": 2827.53, "Howrah_Winter": 1892.19,
    "Jalpaiguri_Autumn": 1710.57, "Jalpaiguri_Summer": 2235.32, "Jalpaiguri_Winter": 1918.42,
    "Jhargram_Autumn": 2948.63, "Jhargram_Summer": 3541.11, "Jhargram_Winter": 2748.34,
    "Kalimpong_Winter": 1903.27, "Maldah_Autumn": 2006.02, "Maldah_Summer": 3705.11, "Maldah_Winter": 2717.12,
    "Murshidabad_Autumn": 2170.79, "Murshidabad_Summer": 3416.63, "Murshidabad_Winter": 2625.57,
    "Nadia_Autumn": 2272.66, "Nadia_Summer": 3519.21, "Nadia_Winter": 2490.08,
    "North 24 Parganas_Autumn": 2469.97, "North 24 Parganas_Summer": 3184.21, "North 24 Parganas_Winter": 2520.08,
    "Paschim Bardhaman_Autumn": 2925.08, "Paschim Bardhaman_Summer": 3296.7, "Paschim Bardhaman_Winter": 3406.98,
    "Paschim Medinipur_Autumn": 2197.7, "Paschim Medinipur_Summer": 3131.77, "Paschim Medinipur_Winter": 2491.7,
    "Purba Bardhaman_Autumn": 2916.76, "Purba Bardhaman_Summer": 3186.14, "Purba Bardhaman_Winter": 2966.3,
    "Purba Medinipur_Autumn": 1955.12, "Purba Medinipur_Summer": 3226.24, "Purba Medinipur_Winter": 1994.06,
    "Purulia_Autumn": 1715.32, "Purulia_Summer": 2242.38, "Purulia_Winter": 2262.22,
    "South 24 Parganas_Autumn": 2308.7, "South 24 Parganas_Summer": 3043.41, "South 24 Parganas_Winter": 2109.96,
    "Uttar Dinajpur_Autumn": 1659.53, "Uttar Dinajpur_Summer": 2887.89, "Uttar Dinajpur_Winter": 2238.14
}
D_MEANS = {
    "Alipurduar": 2337.92, "Bankura": 2763.53, "Bardhaman": 3010.86, "Birbhum": 2931.41,
    "Cooch Behar": 2102.52, "Dakshin Dinajpur": 2468.18, "Darjeeling": 1987.28,
    "Hooghly": 2772.58, "Howrah": 2265.84, "Jalpaiguri": 1954.77, "Jhargram": 3079.36,
    "Kalimpong": 1903.27, "Maldah": 2809.42, "Murshidabad": 2737.67, "Nadia": 2760.65,
    "North 24 Parganas": 2724.75, "Paschim Bardhaman": 3209.59, "Paschim Medinipur": 2607.06,
    "Purba Bardhaman": 3023.06, "Purba Medinipur": 2391.81, "Purulia": 2073.31,
    "South 24 Parganas": 2487.36, "Uttar Dinajpur": 2261.85
}
S_MEANS = {"Autumn": 2177.46, "Summer": 2998.44, "Winter": 2424.56}
STATE_MEAN = 2533.37

# Plausible agronomic bounds for rice in West Bengal (kg/ha)
AGRONOMIC_MIN_YIELD = 500.0
AGRONOMIC_MAX_YIELD = 6000.0

MODEL_ALIASES = {
    'improved_ridge': 'm1_improved_delta_ridge.pkl',
    'improved_xgb_cons': 'm1_improved_delta_xgboost_conservative.pkl',
    'improved_xgb': 'm1_improved_delta_xgboost.pkl',
    'ridge': 'm1_improved_delta_ridge.pkl',
    'xgboost_conservative': 'm1_improved_delta_xgboost_conservative.pkl',
    'xgboost': 'm1_improved_delta_xgboost.pkl',
}


def _parse_float(val: Any) -> Optional[float]:
    """Safely parse float from arbitrary input, returning None for invalid/nan/inf."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    # String handling
    val_str = str(val).strip().lower()
    if val_str in ("", "nan", "none", "null", "n/a", "na", "inf", "-inf", "+inf", "?", "-"):
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


class RiceYieldPredictor:
    """
    Rice yield prediction model wrapper.
    Handles data validation, missing feature imputation, delta inference,
    reconstruction of absolute yield, seasonal policy routing, and agronomic bounds checking.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_type: str = 'improved_ridge',
        clamp_yield: bool = True,
        min_yield: float = AGRONOMIC_MIN_YIELD,
        max_yield: float = AGRONOMIC_MAX_YIELD,
        enable_seasonal_routing: bool = False,
        weather_missing_policy: Optional[str] = None,
    ):
        self.model_type = model_type
        self.clamp_yield = clamp_yield
        self.min_yield = min_yield
        self.max_yield = max_yield
        self.enable_seasonal_routing = enable_seasonal_routing
        self.weather_missing_policy = (
            str(weather_missing_policy).lower() if weather_missing_policy else None
        )
        if self.weather_missing_policy and self.weather_missing_policy not in ("persistence", "model_imputation"):
            raise ValueError(
                f"Invalid weather_missing_policy '{weather_missing_policy}'. "
                f"Supported modes: 'persistence', 'model_imputation'."
            )

        self.model_path = self._resolve_model_path(model_path, model_type)
        self.model_name = self.model_path.name
        self.pipeline = self._load_pipeline(self.model_path)

    def _resolve_model_path(
        self,
        model_path: Optional[Union[str, Path]],
        model_type: str
    ) -> Path:
        if model_path is not None:
            p = Path(model_path)
            if p.exists():
                return p
            # Check within MODELS_DIR
            alt_p = MODELS_DIR / p.name
            if alt_p.exists():
                return alt_p
            # Bridge legacy alias: yield_model.pkl
            if p.name == "yield_model.pkl":
                default_model = MODELS_DIR / "m1_improved_delta_ridge.pkl"
                if default_model.exists():
                    logger.info(f"Mapping legacy '{p.name}' to {default_model.name}")
                    return default_model
            # Explicit path was specified but does not exist
            raise FileNotFoundError(f"Specified model path '{model_path}' does not exist.")

        # Check model_type alias
        filename = MODEL_ALIASES.get(model_type.lower(), f"{model_type}.pkl")
        resolved = MODELS_DIR / filename
        if resolved.exists():
            return resolved

        # Default fallback
        default_model = MODELS_DIR / "m1_improved_delta_ridge.pkl"
        if default_model.exists():
            logger.warning(f"Requested model '{model_type}' not found. Falling back to {default_model.name}")
            return default_model

        raise FileNotFoundError(f"No valid model artifact found at {model_path} or in {MODELS_DIR}")

    def _load_pipeline(self, path: Path):
        logger.info(f"Loading yield model artifact from: {path}")
        pipeline = joblib.load(path)
        return pipeline

    @staticmethod
    def _normalize_district(val: Any) -> str:
        """Standardizes district names across casing and regional spelling variations."""
        if val is None or pd.isna(val):
            return "Unknown"
        s = str(val).strip()
        if not s or s.lower() in ("none", "nan", "null", "unknown", "?"):
            return "Unknown"
        clean = s.lower().replace("-", " ")
        if clean in DISTRICT_ALIASES:
            return DISTRICT_ALIASES[clean]
        title = s.title()
        if title in VALID_DISTRICTS:
            return title
        return s

    @staticmethod
    def _normalize_season(val: Any) -> str:
        """Standardizes season names across casing, Aus/Aman/Boro, and Kharif/Rabi aliases."""
        if val is None or pd.isna(val):
            return "Unknown"
        s = str(val).strip()
        if not s or s.lower() in ("none", "nan", "null", "unknown", "?"):
            return "Unknown"
        clean = s.lower()
        if clean in SEASON_ALIASES:
            return SEASON_ALIASES[clean]
        title = s.title()
        if title in VALID_SEASONS:
            return title
        return s

    def _impute_yield_lag1(self, record: Dict[str, Any]) -> Tuple[float, bool, List[str]]:
        """
        Determines the baseline yield (Yield_lag1).
        If missing, non-numeric, or non-positive, uses a structured hierarchical fallback:
        1. Yield_lag1 (must be > 0 and finite)
        2. Yield_lag2 (must be > 0 and finite)
        3. Yield_hist_mean (if provided and > 0)
        4. Historical District x Season mean
        5. Historical District mean
        6. Historical Season mean
        7. State overall mean (2533.37 kg/ha)
        """
        warnings = []
        raw_lag1 = record.get('Yield_lag1')
        val1 = _parse_float(raw_lag1)

        if val1 is not None:
            if val1 > 0:
                return val1, False, warnings
            else:
                warnings.append(f"Non-positive Yield_lag1 ({val1}) rejected; falling back to secondary baseline.")

        # Fallback 1: Yield_lag2
        raw_lag2 = record.get('Yield_lag2')
        val2 = _parse_float(raw_lag2)
        if val2 is not None and val2 > 0:
            warnings.append(f"Yield_lag1 was missing or invalid; imputed using Yield_lag2 ({val2:.2f} kg/ha).")
            return val2, True, warnings

        # Fallback 2: Yield_hist_mean if passed in input
        raw_hist = record.get('Yield_hist_mean')
        val_hist = _parse_float(raw_hist)
        if val_hist is not None and val_hist > 0:
            warnings.append(f"Yield_lag1 missing or invalid; imputed using provided Yield_hist_mean ({val_hist:.2f} kg/ha).")
            return val_hist, True, warnings

        # Fallback 3: Lookup historical tables
        district = self._normalize_district(record.get('District'))
        season = self._normalize_season(record.get('Season'))

        key = f"{district}_{season}"
        if key in DS_MEANS:
            val = DS_MEANS[key]
            warnings.append(f"Yield_lag1 missing; imputed from historical {district} {season} mean ({val:.2f} kg/ha).")
            return val, True, warnings

        if district in D_MEANS:
            val = D_MEANS[district]
            warnings.append(f"Yield_lag1 missing; imputed from historical {district} district mean ({val:.2f} kg/ha).")
            return val, True, warnings

        if season in S_MEANS:
            val = S_MEANS[season]
            warnings.append(f"Yield_lag1 missing; imputed from historical {season} season mean ({val:.2f} kg/ha).")
            return val, True, warnings

        # Final state average
        val = STATE_MEAN
        warnings.append(f"Yield_lag1 missing & unknown district/season; imputed using state mean ({val:.2f} kg/ha).")
        return val, True, warnings

    def validate_and_prepare_features(
        self,
        df_input: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[List[str]]]:
        """
        Validates input columns, normalizes categories, sanitizes non-numeric and infinite values,
        and handles missing values.
        Returns:
            X_df: Prepared DataFrame with exact 20 training features in correct order
            baselines: 1D array of baseline yields (Yield_lag1)
            imputed_mask: Boolean array indicating where Yield_lag1 was imputed
            record_warnings: List of warning message lists per row
        """
        n_rows = len(df_input)
        baselines = np.zeros(n_rows, dtype=float)
        imputed_flags = np.zeros(n_rows, dtype=bool)
        all_warnings: List[List[str]] = [[] for _ in range(n_rows)]

        prepared_records = []

        for i in range(n_rows):
            row = df_input.iloc[i].to_dict()
            warnings = all_warnings[i]

            # Case-insensitive lookup map for input row
            row_lower = {str(k).lower().strip(): v for k, v in row.items()}

            def _get_field(name: str, aliases: Optional[List[str]] = None) -> Any:
                if name in row:
                    return row[name]
                if name.lower() in row_lower:
                    return row_lower[name.lower()]
                if aliases:
                    for a in aliases:
                        if a in row:
                            return row[a]
                        if a.lower() in row_lower:
                            return row_lower[a.lower()]
                return None

            # 1. Normalize categoricals
            district = self._normalize_district(_get_field('District', ['district', 'district_name', 'District Name']))
            season = self._normalize_season(_get_field('Season', ['season', 'season_name', 'Crop Season']))

            if district not in VALID_DISTRICTS:
                warnings.append(f"Unseen District '{district}'. Prediction will rely on state baseline / zero encoding.")
            if season not in VALID_SEASONS:
                warnings.append(f"Unseen Season '{season}'. Expected one of {VALID_SEASONS}.")

            # 2. Check and impute baseline yield
            row_norm = dict(row)
            row_norm['District'] = district
            row_norm['Season'] = season
            for k_canon, aliases in [
                ('Yield_lag1', ['yield_lag1', 'yield_lag_1', 'lag1', 'lag1_yield']),
                ('Yield_lag2', ['yield_lag2', 'yield_lag_2', 'lag2', 'lag2_yield']),
                ('Yield_hist_mean', ['yield_hist_mean', 'hist_mean', 'historical_mean']),
            ]:
                if k_canon not in row_norm:
                    v_alias = _get_field(k_canon, aliases)
                    if v_alias is not None:
                        row_norm[k_canon] = v_alias

            baseline, is_imputed, lag_warns = self._impute_yield_lag1(row_norm)
            baselines[i] = baseline
            imputed_flags[i] = is_imputed
            warnings.extend(lag_warns)

            # 3. Validate numerical bounds & sanitize infinite / NaN values
            rec = {}
            for num_col in NUMERICAL_FEATURES:
                val = _get_field(num_col, [num_col.lower()])
                val_float = _parse_float(val)

                # Check if raw val was infinite
                if val is not None and not pd.isna(val):
                    try:
                        raw_f = float(val)
                        if np.isinf(raw_f):
                            warnings.append(f"Infinite value in column '{num_col}' replaced with NaN.")
                    except (ValueError, TypeError):
                        pass

                if val_float is None and val is not None and not pd.isna(val):
                    val_str = str(val).strip().lower()
                    if val_str not in ("", "nan", "none", "null", "n/a", "na", "?", "-", "inf", "-inf", "+inf"):
                        warnings.append(f"Non-numeric value in column '{num_col}': {val}. Set to NaN.")

                if val_float is not None:
                    # Agronomic plausibility checks on input features
                    if 'Area' in num_col and val_float < 0:
                        warnings.append(f"Negative area in '{num_col}' ({val_float}).")
                    elif 'Precip' in num_col and val_float < 0:
                        warnings.append(f"Negative precipitation in '{num_col}' ({val_float}).")
                    elif 'NDVI' in num_col and (val_float < -1.0 or val_float > 1.0):
                        warnings.append(f"NDVI out of range [-1, 1] in '{num_col}' ({val_float}).")
                    elif 'T2M' in num_col and (val_float < -20.0 or val_float > 60.0):
                        warnings.append(f"Extreme temperature in '{num_col}' ({val_float}°C).")
                    elif 'RH2M' in num_col and (val_float < 0 or val_float > 100):
                        warnings.append(f"Relative humidity out of [0, 100]% in '{num_col}' ({val_float}%).")
                    rec[num_col] = val_float
                else:
                    rec[num_col] = np.nan

            rec['Yield_lag1'] = baseline
            rec['District'] = district
            rec['Season'] = season

            prepared_records.append(rec)

        X_df = pd.DataFrame(prepared_records)[ALL_FEATURES]
        return X_df, baselines, imputed_flags, all_warnings

    def predict(
        self,
        data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
        return_dict: bool = False,
        enable_seasonal_routing: Optional[bool] = None,
        weather_missing_policy: Optional[str] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
        """
        Execute prediction on single record, list of records, or DataFrame.

        Parameters:
            data: Single dict, list of dicts, or pandas DataFrame.
            return_dict: If True and input is DataFrame, returns list of dicts.
            enable_seasonal_routing: Optional override for provisional seasonal routing.
            weather_missing_policy: Optional override ('persistence' or 'model_imputation').

        Returns:
            Structured prediction results including canonical dual-unit yields and
            explicit weather/seasonal provenance metadata.
        """
        # Resolve active policies for this prediction invocation
        routing_enabled = (
            self.enable_seasonal_routing
            if enable_seasonal_routing is None
            else bool(enable_seasonal_routing)
        )
        if weather_missing_policy is not None:
            active_weather_policy = str(weather_missing_policy).lower()
        elif self.weather_missing_policy is not None:
            active_weather_policy = self.weather_missing_policy
        else:
            # Low-level fallback preserving backward compatibility for unconfigured instances
            active_weather_policy = "model_imputation"

        if active_weather_policy not in ("persistence", "model_imputation"):
            raise ValueError(
                f"Invalid weather_missing_policy '{active_weather_policy}'. "
                f"Supported modes: 'persistence', 'model_imputation'."
            )

        routing_policy_name = (
            PROVISIONAL_SUMMER_ROUTING_POLICY if routing_enabled else "disabled"
        )

        is_single_dict = isinstance(data, dict)

        if is_single_dict:
            df_input = pd.DataFrame([data])
        elif isinstance(data, list):
            df_input = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df_input = data.copy()
        else:
            raise TypeError(f"Expected dict, list of dicts, or pandas DataFrame, got {type(data)}")

        if len(df_input) == 0:
            if is_single_dict:
                return {}
            return pd.DataFrame()

        # Prepare and validate features
        X_df, baselines, imputed_flags, all_warnings = self.validate_and_prepare_features(df_input)

        # Predict delta deviation from baseline using trained pipeline
        predicted_deltas = np.array(self.pipeline.predict(X_df), dtype=float)

        # Audit weather availability and evaluate policies per record independently
        pred_methods = []
        weather_statuses = []
        weather_used_flags = []
        missing_weather_lists = []
        routing_applied_flags = []

        for i in range(len(df_input)):
            row_dict = df_input.iloc[i].to_dict()
            row_dict_lower = {str(k).lower().strip(): v for k, v in row_dict.items()}
            rec_season = X_df.iloc[i]['Season']

            # Check presence and validity of all 10 in-season weather features
            missing_weather = []
            for w_col in INSEASON_WEATHER_FEATURES:
                raw_w = row_dict.get(w_col)
                if raw_w is None:
                    raw_w = row_dict_lower.get(w_col.lower())
                parsed_w = _parse_float(raw_w)
                if parsed_w is None:
                    missing_weather.append(w_col)

            missing_weather_lists.append(missing_weather)

            if len(missing_weather) == 0:
                w_status = "complete"
            elif len(missing_weather) == len(INSEASON_WEATHER_FEATURES):
                w_status = "missing"
            else:
                w_status = "incomplete"
            weather_statuses.append(w_status)

            # 1. Provisional seasonal routing evaluation:
            # Summer yield underperforms persistence by 13.41 kg/ha in 2017-2019 test evaluations.
            # When seasonal routing is enabled, Summer requests bypass delta and use persistence directly.
            if routing_enabled and rec_season == 'Summer':
                predicted_deltas[i] = 0.0
                pred_method = PROVISIONAL_SUMMER_ROUTING_POLICY
                routing_applied = True
                weather_used = False
                all_warnings[i].append(
                    "Seasonal routing active: Provisional seasonal routing policy applied — Summer (Boro) yield "
                    "routed to persistence baseline. Scientific qualification: This policy is provisional and was "
                    "motivated by historical 2017-2019 test set evaluations; it is not independently validated and "
                    "is subject to revision after Step 5 rolling-origin evaluation."
                )
                if w_status == "complete":
                    all_warnings[i].append(
                        "In-season weather features were supplied but ML model bypassed by provisional seasonal routing."
                    )
                else:
                    all_warnings[i].append("In-season weather features unavailable or incomplete.")
            else:
                # Seasonal routing not applied (either disabled or season is Autumn/Winter)
                routing_applied = False
                if w_status == "complete":
                    # All 10 weather features available: use ML model prediction
                    pred_method = getattr(self, "model_type", "improved_delta_ridge")
                    if hasattr(self, "model_name") and self.model_name:
                        pred_method = self.model_name.replace(".pkl", "")
                    weather_used = True
                else:
                    # Weather incomplete or completely missing
                    if active_weather_policy == "persistence":
                        predicted_deltas[i] = 0.0
                        pred_method = "persistence"
                        weather_used = False
                        if w_status == "missing":
                            all_warnings[i].append(
                                "In-season weather features unavailable; persistence fallback applied."
                            )
                        else:
                            all_warnings[i].append(
                                f"Incomplete in-season weather features (missing: {', '.join(missing_weather)}); "
                                "persistence fallback applied."
                            )
                    else:  # model_imputation
                        pred_method = "model_imputation"
                        weather_used = False
                        if w_status == "missing":
                            all_warnings[i].append(
                                "In-season weather features unavailable; missing values filled using training-set "
                                "median imputation. Warning: This prediction does NOT represent observed weather."
                            )
                        else:
                            all_warnings[i].append(
                                f"Incomplete in-season weather features (missing: {', '.join(missing_weather)}); "
                                "values filled using training-set median imputation. Warning: This prediction does "
                                "NOT represent observed weather."
                            )

            pred_methods.append(pred_method)
            weather_used_flags.append(weather_used)
            routing_applied_flags.append(routing_applied)

        raw_predicted_yields = baselines + predicted_deltas

        # Agronomic bounds clamping
        final_yields_kg = raw_predicted_yields.copy()
        for i in range(len(final_yields_kg)):
            raw_val = final_yields_kg[i]
            if self.clamp_yield:
                if raw_val < self.min_yield:
                    all_warnings[i].append(
                        f"Predicted yield ({raw_val:.2f} kg/ha) below agronomic minimum ({self.min_yield}). Clamped."
                    )
                    final_yields_kg[i] = self.min_yield
                elif raw_val > self.max_yield:
                    all_warnings[i].append(
                        f"Predicted yield ({raw_val:.2f} kg/ha) above agronomic maximum ({self.max_yield}). Clamped."
                    )
                    final_yields_kg[i] = self.max_yield

        final_yields_tonnes = final_yields_kg / 1000.0

        # Construct result objects
        results = []
        for i in range(len(final_yields_kg)):
            kg_val = round(float(final_yields_kg[i]), 2)
            tonnes_val = round(float(final_yields_tonnes[i]), 4)
            res = {
                "predicted_yield_kg_per_ha": kg_val,
                "predicted_yield_tonnes_per_ha": tonnes_val,
                "predicted_yield_kg_ha": kg_val,
                "predicted_yield_tonnes_ha": tonnes_val,
                "baseline_yield_kg_ha": round(float(baselines[i]), 2),
                "predicted_delta_kg_ha": round(float(predicted_deltas[i]), 2),
                "is_yield_lag1_imputed": bool(imputed_flags[i]),
                "prediction_method": pred_methods[i],
                "weather_feature_status": weather_statuses[i],
                "weather_features_used": bool(weather_used_flags[i]),
                "missing_weather_features": missing_weather_lists[i],
                "prediction_policy_version": POLICY_VERSION,
                "seasonal_routing_applied": bool(routing_applied_flags[i]),
                "seasonal_routing_policy": routing_policy_name,
                "model_name": self.model_name,
                "warnings": all_warnings[i]
            }

            # Optional crop production calculation when valid area is explicitly supplied
            row_dict = df_input.iloc[i].to_dict()
            row_dict_lower = {str(k).lower().strip(): v for k, v in row_dict.items()}
            area_val = None
            for area_key in ("Area_ha", "area_ha", "Area", "area", "harvested_area_ha", "planted_area_ha"):
                raw_a = row_dict.get(area_key)
                if raw_a is None:
                    raw_a = row_dict_lower.get(area_key.lower())
                if raw_a is not None and not pd.isna(raw_a):
                    parsed_a = _parse_float(raw_a)
                    if parsed_a is not None and parsed_a > 0:
                        area_val = parsed_a
                        break
                    elif parsed_a is not None and parsed_a <= 0:
                        all_warnings[i].append(f"Non-positive area supplied in '{area_key}' ({parsed_a}). Production omitted.")

            if area_val is not None:
                # Total Production (tonnes) = Yield (tonnes/ha) * Area (ha)
                res["estimated_production_tonnes"] = round(tonnes_val * area_val, 4)
                res["area_ha"] = area_val

            results.append(res)

        if is_single_dict:
            return results[0]

        if return_dict:
            return results

        # Return enhanced DataFrame
        out_df = df_input.copy()
        out_df["predicted_yield_kg_per_ha"] = [r["predicted_yield_kg_per_ha"] for r in results]
        out_df["predicted_yield_tonnes_per_ha"] = [r["predicted_yield_tonnes_per_ha"] for r in results]
        out_df["predicted_yield_kg_ha"] = [r["predicted_yield_kg_ha"] for r in results]
        out_df["predicted_yield_tonnes_ha"] = [r["predicted_yield_tonnes_ha"] for r in results]
        out_df["predicted_delta_kg_ha"] = [r["predicted_delta_kg_ha"] for r in results]
        out_df["baseline_yield_kg_ha"] = [r["baseline_yield_kg_ha"] for r in results]
        out_df["is_yield_lag1_imputed"] = [r["is_yield_lag1_imputed"] for r in results]
        out_df["prediction_method"] = [r["prediction_method"] for r in results]
        out_df["weather_feature_status"] = [r["weather_feature_status"] for r in results]
        out_df["weather_features_used"] = [r["weather_features_used"] for r in results]
        out_df["missing_weather_features"] = [", ".join(r["missing_weather_features"]) if r["missing_weather_features"] else "none" for r in results]
        out_df["prediction_policy_version"] = [r["prediction_policy_version"] for r in results]
        out_df["seasonal_routing_applied"] = [r["seasonal_routing_applied"] for r in results]
        out_df["seasonal_routing_policy"] = [r["seasonal_routing_policy"] for r in results]
        out_df["prediction_warnings"] = ["; ".join(r["warnings"]) if r["warnings"] else "None" for r in results]
        if any("estimated_production_tonnes" in r for r in results):
            out_df["estimated_production_tonnes"] = [r.get("estimated_production_tonnes", np.nan) for r in results]
        return out_df


# -------------------------------------------------------------------------
# Integration / Bridge Convenience Functions
# -------------------------------------------------------------------------

_DEFAULT_PREDICTOR: Optional[RiceYieldPredictor] = None


def get_default_predictor(
    model_path: Optional[str] = None,
    enable_seasonal_routing: bool = True,
    weather_missing_policy: str = "persistence"
) -> RiceYieldPredictor:
    """Returns singleton instance of default predictor matching configuration."""
    global _DEFAULT_PREDICTOR
    if (
        _DEFAULT_PREDICTOR is None
        or (model_path and Path(model_path) != _DEFAULT_PREDICTOR.model_path)
        or (_DEFAULT_PREDICTOR.enable_seasonal_routing != enable_seasonal_routing)
        or (_DEFAULT_PREDICTOR.weather_missing_policy != weather_missing_policy)
    ):
        _DEFAULT_PREDICTOR = RiceYieldPredictor(
            model_path=model_path,
            enable_seasonal_routing=enable_seasonal_routing,
            weather_missing_policy=weather_missing_policy
        )
    return _DEFAULT_PREDICTOR


def predict_rice_yield(
    data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    model_path: Optional[str] = None,
    clamp_yield: bool = True,
    enable_seasonal_routing: bool = True,
    weather_missing_policy: str = "persistence"
) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """Convenience function to predict yield on data."""
    predictor = RiceYieldPredictor(
        model_path=model_path,
        clamp_yield=clamp_yield,
        enable_seasonal_routing=enable_seasonal_routing,
        weather_missing_policy=weather_missing_policy
    )
    return predictor.predict(data)


def load_model(
    model_path: Optional[str] = None,
    enable_seasonal_routing: bool = True,
    weather_missing_policy: str = "persistence"
) -> Dict[str, Any]:
    """
    Integration loader matching shared/predict_bridge.py contract.
    Returns model metadata dictionary with feature_names and pipeline.
    """
    predictor = get_default_predictor(
        model_path=model_path,
        enable_seasonal_routing=enable_seasonal_routing,
        weather_missing_policy=weather_missing_policy
    )
    return {
        "pipeline": predictor.pipeline,
        "feature_names": ALL_FEATURES,
        "predictor": predictor,
        "model_name": predictor.model_name
    }


def predict_yield(
    model_info: Dict[str, Any],
    input_dict: Dict[str, Any],
    as_dict: bool = False,
    enable_seasonal_routing: Optional[bool] = None,
    weather_missing_policy: Optional[str] = None,
) -> Union[float, Dict[str, Any]]:
    """
    Integration prediction function matching shared/predict_bridge.py contract.
    Returns predicted yield in tonnes/ha (or full prediction dict if as_dict=True).
    """
    predictor = model_info.get("predictor") or get_default_predictor(
        enable_seasonal_routing=enable_seasonal_routing if enable_seasonal_routing is not None else True,
        weather_missing_policy=weather_missing_policy if weather_missing_policy is not None else "persistence"
    )
    res = predictor.predict(
        input_dict,
        enable_seasonal_routing=enable_seasonal_routing,
        weather_missing_policy=weather_missing_policy
    )
    if isinstance(res, dict):
        if as_dict:
            return res
        return float(res["predicted_yield_tonnes_per_ha"])
    raise ValueError("Expected dict result from single input dict.")


# -------------------------------------------------------------------------
# CLI Application
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="AgroVision Model 1: Rice Yield Prediction CLI (kg/ha and tonnes/ha)"
    )
    parser.add_argument("--input", type=str, help="Path to input CSV or JSON file")
    parser.add_argument("--output", type=str, help="Path to write prediction results (CSV or JSON)")
    parser.add_argument("--data", type=str, help="Raw JSON string containing input record or list of records")
    parser.add_argument("--model", type=str, default="improved_ridge",
                        help="Candidate model name or path (improved_ridge, improved_xgb_cons, improved_xgb)")
    parser.add_argument("--no-clamp", action="store_true", help="Disable agronomic bounds clamping [500, 6000]")
    parser.add_argument("--weather-missing-policy", type=str, default="persistence",
                        choices=["persistence", "model_imputation"],
                        help="Weather missing policy: 'persistence' (default) or 'model_imputation'")
    parser.add_argument("--enable-seasonal-routing", action="store_true", default=True,
                        help="Enable provisional seasonal routing (Summer routes to persistence baseline; default: enabled)")
    parser.add_argument("--disable-seasonal-routing", dest="enable_seasonal_routing", action="store_false",
                        help="Disable provisional seasonal routing policy")

    # Single-instance CLI arguments
    parser.add_argument("--district", type=str, help="District name (e.g. Hooghly, Purba Bardhaman)")
    parser.add_argument("--season", type=str, help="Season name (Autumn, Summer, Winter)")
    parser.add_argument("--yield-lag1", type=float, help="Previous year yield (kg/ha)")
    parser.add_argument("--yield-lag2", type=float, help="Two years prior yield (kg/ha)")
    parser.add_argument("--area-lag1", type=float, help="Previous year area (ha)")
    parser.add_argument("--area-lag2", type=float, help="Two years prior area (ha)")
    parser.add_argument("--area-ha", "--area", dest="area_ha", type=float, default=None,
                        help="Planted or harvested area in hectares for total production estimation")
    parser.add_argument("--precip-lag1", type=float, help="Previous year precipitation (mm)")
    parser.add_argument("--t2m-lag1", type=float, help="Previous year mean temperature (°C)")
    parser.add_argument("--rh2m-lag1", type=float, help="Previous year relative humidity (pct)")
    parser.add_argument("--ndvi-lag1", type=float, help="Previous year NDVI")

    # In-season weather parameters
    parser.add_argument("--precip-cum", type=float, help="In-season cumulative precipitation (mm)")
    parser.add_argument("--precip-mean", type=float, help="In-season mean daily precipitation (mm)")
    parser.add_argument("--precip-max", type=float, help="In-season maximum daily precipitation (mm)")
    parser.add_argument("--t2m-mean", type=float, help="In-season mean temperature (°C)")
    parser.add_argument("--t2m-min", type=float, help="In-season minimum daily temperature (°C)")
    parser.add_argument("--t2m-max", type=float, help="In-season maximum daily temperature (°C)")
    parser.add_argument("--t2m-range", type=float, help="In-season temperature range (°C)")
    parser.add_argument("--rh2m-mean", type=float, help="In-season mean relative humidity (pct)")
    parser.add_argument("--solar-mean", type=float, help="In-season mean solar insolation (MJ/m²/day)")
    parser.add_argument("--wind-mean", type=float, help="In-season mean wind speed (m/s)")

    return parser.parse_args()


def main():
    args = parse_args()
    clamp = not args.no_clamp

    try:
        predictor = RiceYieldPredictor(
            model_type=args.model,
            clamp_yield=clamp,
            enable_seasonal_routing=args.enable_seasonal_routing,
            weather_missing_policy=args.weather_missing_policy,
        )
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        sys.exit(1)

    # Mode 1: File batch inference
    if args.input:
        in_path = Path(args.input)
        if not in_path.exists():
            logger.error(f"Input file not found: {in_path}")
            sys.exit(1)

        if in_path.suffix.lower() == ".csv":
            df = pd.read_csv(in_path)
            res_df = predictor.predict(df)
            if args.output:
                out_path = Path(args.output)
                res_df.to_csv(out_path, index=False)
                logger.info(f"Predictions written to {out_path}")
            else:
                print(res_df.to_string())
        else:
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            res = predictor.predict(data)
            if args.output:
                out_path = Path(args.output)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2)
                logger.info(f"Predictions written to {out_path}")
            else:
                print(json.dumps(res, indent=2))
        return

    # Mode 2: Direct raw JSON string input via --data
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as err:
            logger.error(f"Invalid JSON provided in --data: {err}")
            sys.exit(1)
        res = predictor.predict(data)
        if args.output:
            out_path = Path(args.output)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            logger.info(f"Predictions written to {out_path}")
        else:
            print(json.dumps(res, indent=2))
        return

    # Mode 3: Single prediction from CLI arguments
    record = {}
    if args.district:
        record["District"] = args.district
    if args.season:
        record["Season"] = args.season
    if args.yield_lag1 is not None:
        record["Yield_lag1"] = args.yield_lag1
    if args.yield_lag2 is not None:
        record["Yield_lag2"] = args.yield_lag2
    if args.area_lag1 is not None:
        record["Area_lag1"] = args.area_lag1
    if args.area_lag2 is not None:
        record["Area_lag2"] = args.area_lag2
    if args.area_ha is not None:
        record["Area_ha"] = args.area_ha
    if args.precip_lag1 is not None:
        record["Precip_lag1"] = args.precip_lag1
    if args.t2m_lag1 is not None:
        record["T2M_lag1"] = args.t2m_lag1
    if args.rh2m_lag1 is not None:
        record["RH2M_lag1"] = args.rh2m_lag1
    if args.ndvi_lag1 is not None:
        record["NDVI_lag1"] = args.ndvi_lag1

    if args.precip_cum is not None:
        record["InSeason_Precip_Cum"] = args.precip_cum
    if args.precip_mean is not None:
        record["InSeason_Precip_Mean"] = args.precip_mean
    if args.precip_max is not None:
        record["InSeason_Precip_Max"] = args.precip_max
    if args.t2m_mean is not None:
        record["InSeason_T2M_Mean"] = args.t2m_mean
    if args.t2m_min is not None:
        record["InSeason_T2M_Min"] = args.t2m_min
    if args.t2m_max is not None:
        record["InSeason_T2M_Max"] = args.t2m_max
    if args.t2m_range is not None:
        record["InSeason_T2M_Range"] = args.t2m_range
    if args.rh2m_mean is not None:
        record["InSeason_RH2M_Mean"] = args.rh2m_mean
    if args.solar_mean is not None:
        record["InSeason_Solar_Mean"] = args.solar_mean
    if args.wind_mean is not None:
        record["InSeason_Wind_Mean"] = args.wind_mean

    if not record:
        print("No input provided. Please pass --input <file>, --data '<json>', or CLI parameters like --district, --season, --yield-lag1.")
        print("Run with -h for help.")
        sys.exit(0)

    result = predictor.predict(record)
    print("\n" + "=" * 50)
    print("AGROVISION RICE YIELD PREDICTION RESULT")
    print("=" * 50)
    print(f"Model:                      {result['model_name']}")
    print(f"Prediction Method:          {result['prediction_method']}")
    print(f"Weather Feature Status:     {result['weather_feature_status']}")
    print(f"Weather Features Used:      {result['weather_features_used']}")
    print(f"Seasonal Routing Applied:   {result['seasonal_routing_applied']} ({result['seasonal_routing_policy']})")
    print(f"Prediction Policy Version:  {result['prediction_policy_version']}")
    kg_ha_display = result.get('predicted_yield_kg_per_ha', result.get('predicted_yield_kg_ha', 0.0))
    t_ha_display = result.get('predicted_yield_tonnes_per_ha', result.get('predicted_yield_tonnes_ha', 0.0))
    print(f"Predicted Yield (kg/ha):    {kg_ha_display:.2f} kg/ha")
    print(f"Predicted Yield (tonnes/ha):{t_ha_display:.4f} t/ha")
    if "estimated_production_tonnes" in result:
        print(f"Estimated Prod. (tonnes):  {result['estimated_production_tonnes']:.4f} tonnes (Area: {result.get('area_ha')} ha)")
    print(f"Baseline Yield (kg/ha):     {result['baseline_yield_kg_ha']:.2f} kg/ha")
    print(f"Predicted Delta (kg/ha):    {result['predicted_delta_kg_ha']:+.2f} kg/ha")
    print(f"Yield_lag1 Imputed?:        {result['is_yield_lag1_imputed']}")
    if result['warnings']:
        print("Warnings:")
        for w in result['warnings']:
            print(f"  - {w}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
