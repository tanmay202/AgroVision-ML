"""
Standalone Rice Price & Volatility Predictor for AgroVision-ML.
Author: Member 2 (Price Prediction & Volatility ML)
Target Consumer: Member 3 (Backend / Application Integration)

Pure Python inference class with zero FastAPI / web server dependencies.
Accepts historical mandi observations as pandas DataFrame or list of dicts.
Supports Short-Term (1-7d), 14-Day Calendar Horizon, and 30-Day Calendar Horizon forecasting.
"""

import os
import json
import warnings
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from .config import (
    DATE_COLUMN,
    PRICE_COLUMN,
    ARRIVAL_COLUMN,
    GROUP_COLUMNS,
    SUPPORTED_HORIZONS,
    TARGET_CONVENTIONS,
    RELEASE_STATUSES,
    MODEL_NAMES,
    MODEL_LIMITATIONS,
)
from .feature_engineering import (
    InsufficientHistoryError,
    normalize_input_data,
    generate_48_price_features,
    generate_normalized_volatility_features,
)

warnings.filterwarnings("ignore")


class RicePricePredictor:
    """
    Production-ready standalone rice price and volatility predictor.

    Parameters:
    -----------
    artifacts_dir : str or Path, optional
        Path to directory containing model `.pkl` and config `.json` files.
        If None, automatically searches:
        1. Environment variable `AGROVISION_ARTIFACTS_DIR`
        2. `deliverables/member2-ml/artifacts`
        3. `member2-price-volatility/artifacts`
        4. `artifacts` in repository root
    """

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
        self.artifacts_dir = self._resolve_artifacts_dir(artifacts_dir)

        # 1. Load frozen production Short-Term Price Model (48 features)
        self.short_term_model_path = self.artifacts_dir / "rice_price_model.pkl"
        self.short_term_config_path = self.artifacts_dir / "rice_price_final_config.json"

        if not self.short_term_model_path.exists():
            raise FileNotFoundError(f"Missing price model artifact: {self.short_term_model_path}")
        if not self.short_term_config_path.exists():
            raise FileNotFoundError(f"Missing price config artifact: {self.short_term_config_path}")

        with open(self.short_term_config_path, "r", encoding="utf-8") as f:
            self.short_term_config = json.load(f)

        if self.short_term_config.get("commodity") != "rice":
            raise ValueError(
                f"Invalid config at {self.short_term_config_path}: commodity is "
                f"'{self.short_term_config.get('commodity')}', expected 'rice'."
            )

        self.features_list = self.short_term_config.get("features", [])
        if len(self.features_list) != 48:
            raise ValueError(f"Invalid feature configuration: expected 48 features, found {len(self.features_list)}")

        loaded_obj = joblib.load(self.short_term_model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.short_term_model = loaded_obj["model"]
            self.short_term_metadata = loaded_obj.get("metadata", {})
        else:
            self.short_term_model = loaded_obj
            self.short_term_metadata = {}

        n_features = getattr(self.short_term_model, "n_features_in_", None)
        if n_features is not None and n_features != 48:
            raise ValueError(
                f"Incompatible model artifact at {self.short_term_model_path}: "
                f"model expects {n_features} features, but production requires 48 features."
            )

        # 2. Caches for lazy-loaded horizon and volatility models
        self._horizon_models: Dict[int, Any] = {}
        self._horizon_configs: Dict[int, Dict[str, Any]] = {}
        self._normalized_vol_model = None
        self._normalized_vol_config = None

    def _resolve_artifacts_dir(self, custom_path: Optional[Union[str, Path]]) -> Path:
        if custom_path is not None:
            p = Path(custom_path).resolve()
            if p.exists() and (p / "rice_price_model.pkl").exists():
                return p
            raise FileNotFoundError(f"Provided artifacts_dir does not contain rice_price_model.pkl: {p}")

        candidates = []
        env_val = os.environ.get("AGROVISION_ARTIFACTS_DIR")
        if env_val:
            candidates.append(Path(env_val).resolve())

        # Relative to current module
        module_dir = Path(__file__).resolve().parent
        candidates.append((module_dir.parent / "artifacts").resolve())
        candidates.append((module_dir.parent.parent.parent / "member2-price-volatility" / "artifacts").resolve())
        candidates.append((module_dir.parent.parent.parent / "artifacts").resolve())

        for cand in candidates:
            if cand.exists() and (cand / "rice_price_model.pkl").exists() and (cand / "rice_price_final_config.json").exists():
                return cand

        # Fallback to local package artifacts
        return (module_dir.parent / "artifacts").resolve()

    def _load_horizon_model(self, horizon_days: int) -> Tuple[Any, Dict[str, Any]]:
        """Lazy-loads and caches 14-day or 30-day XGBoost horizon models."""
        if horizon_days in self._horizon_models:
            return self._horizon_models[horizon_days], self._horizon_configs[horizon_days]

        model_path = self.artifacts_dir / f"rice_price_model_{horizon_days}d.pkl"
        cfg_path = self.artifacts_dir / f"rice_price_model_{horizon_days}d_config.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing {horizon_days}-day horizon model artifact at {model_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing {horizon_days}-day horizon config artifact at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            model = loaded_obj["model"]
        else:
            model = loaded_obj

        self._horizon_models[horizon_days] = model
        self._horizon_configs[horizon_days] = cfg
        return model, cfg

    def _load_normalized_volatility_model(self):
        """Lazy-loads and caches price-normalized volatility model and configuration."""
        if self._normalized_vol_model is not None:
            return

        model_path = self.artifacts_dir / "rice_volatility_normalized_model.pkl"
        cfg_path = self.artifacts_dir / "rice_volatility_normalized_config.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Missing normalized volatility model at {model_path}")
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing normalized volatility config at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            self._normalized_vol_config = json.load(f)

        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self._normalized_vol_model = loaded_obj["model"]
            self._normalized_vol_features = loaded_obj.get("feature_names", self._normalized_vol_config.get("features", []))
            self._normalized_vol_classes = loaded_obj.get("classes", ["Low", "Moderate", "High"])
            self._normalized_vol_thresholds = loaded_obj.get("thresholds", self._normalized_vol_config.get("thresholds", {}))
        else:
            self._normalized_vol_model = loaded_obj
            self._normalized_vol_features = self._normalized_vol_config.get("features", [])
            self._normalized_vol_classes = ["LOW", "MEDIUM", "HIGH"]
            self._normalized_vol_thresholds = self._normalized_vol_config.get("thresholds", {})

    def get_required_features(self) -> List[str]:
        """Returns the exact ordered list of 48 features expected by the price regressors."""
        return list(self.features_list)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns metadata about available models, versions, and statuses."""
        return {
            "commodity": "rice",
            "artifacts_dir": str(self.artifacts_dir),
            "horizons": {
                "short_term": {
                    "horizon_days": 1,
                    "target": "Next valid observation within 1-7 days",
                    "status": "production",
                    "features_count": 48,
                    "model_file": str(self.short_term_model_path.name),
                },
                "14d": {
                    "horizon_days": 14,
                    "target": TARGET_CONVENTIONS[14],
                    "status": "opt-in",
                    "features_count": 48,
                    "model_file": f"rice_price_model_14d.pkl",
                },
                "30d": {
                    "horizon_days": 30,
                    "target": TARGET_CONVENTIONS[30],
                    "status": "experimental",
                    "features_count": 48,
                    "model_file": f"rice_price_model_30d.pkl",
                },
            },
            "volatility": {
                "v1": "persistence baseline (5-day rolling range)",
                "v2_normalized": "price-normalized 3-class ML model (Low, Moderate, High)",
            }
        }

    def predict_short_term(
        self,
        history: Union[pd.DataFrame, List[Dict]],
        market: Optional[str] = None,
        variety: Optional[str] = None,
        prediction_date: Optional[Union[str, pd.Timestamp]] = None,
        include_volatility: bool = True,
        volatility_version: str = "v2_normalized",
    ) -> Dict[str, Any]:
        """
        Generates short-term price forecast (1-7 days) using frozen 48-feature XGBoost regressor.

        Parameters:
        -----------
        history : DataFrame or list of dicts
            Historical observations spanning >= 30 days.
        market : str, optional
            Mandi/market name (e.g. 'Burdwan'). If None, inferred from history.
        variety : str, optional
            Rice variety (e.g. 'Common'). If None, inferred from history.
        prediction_date : str, optional
            Origin date for prediction (YYYY-MM-DD). If None, inferred as max date.
        include_volatility : bool, default True
            Whether to compute and include volatility predictions.
        volatility_version : str, default 'v2_normalized'
            'v1' (persistence baseline) or 'v2_normalized' (normalized ML model).

        Returns:
        --------
        dict containing predicted modal price, change, metadata, and volatility.
        """
        clean_df, resolved_market, resolved_variety, resolved_date = normalize_input_data(
            history, market=market, variety=variety, prediction_date=prediction_date, min_observations=30
        )

        feat_df = generate_48_price_features(clean_df)
        X = feat_df.iloc[[-1]][self.features_list].copy()
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"NaN detected in generated features: {nan_cols}")

        predicted_change = float(self.short_term_model.predict(X)[0])
        current_price = float(feat_df.iloc[-1][PRICE_COLUMN])
        predicted_price = current_price + predicted_change

        response = {
            "market": resolved_market,
            "variety": resolved_variety,
            "prediction_date": resolved_date,
            "forecast_origin_date": resolved_date,
            "horizon_days": 1,
            "requested_target_date": (pd.to_datetime(resolved_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "current_modal_price": round(current_price, 2),
            "predicted_modal_price": round(predicted_price, 2),
            "predicted_price_change": round(predicted_change, 2),
            "target_date_convention": TARGET_CONVENTIONS[1],
            "model_name": MODEL_NAMES[1],
            "model_version": "1.0.0",
            "release_status": RELEASE_STATUSES[1],
            "limitations": MODEL_LIMITATIONS[1],
        }

        if include_volatility:
            # Baseline rolling range
            p_last5 = feat_df.iloc[-5:][PRICE_COLUMN]
            vol_range = float(p_last5.max() - p_last5.min()) if len(p_last5) > 0 else 0.0

            if volatility_version == "v2_normalized":
                self._load_normalized_volatility_model()
                norm_feat_df = generate_normalized_volatility_features(clean_df)
                X_norm = norm_feat_df.iloc[[-1]][self._normalized_vol_features].copy()
                X_norm = X_norm.apply(pd.to_numeric, errors="coerce")

                proba = self._normalized_vol_model.predict_proba(X_norm)[0]
                class_idx = int(np.argmax(proba))
                v2_class = self._normalized_vol_classes[class_idx]
                proba_dict = {
                    cls: round(float(proba[i]), 4)
                    for i, cls in enumerate(self._normalized_vol_classes)
                }
                hist_pct_range = float((vol_range / (current_price + 1e-4)) * 100.0)

                response.update({
                    "predicted_volatility_range": round(vol_range, 2),
                    "predicted_volatility_class": v2_class,
                    "volatility_version": "v2_normalized",
                    "normalized_volatility_class": v2_class,
                    "normalized_volatility_probabilities": proba_dict,
                    "historical_pct_range_5": round(hist_pct_range, 2),
                    "thresholds": self._normalized_vol_thresholds,
                })
            else:
                # v1 persistence
                v1_class = "High" if vol_range > 100 else ("Moderate" if vol_range > 30 else "Low")
                response.update({
                    "predicted_volatility_range": round(vol_range, 2),
                    "predicted_volatility_class": v1_class,
                    "volatility_version": "v1",
                })

        return response

    def predict_horizon(
        self,
        history: Union[pd.DataFrame, List[Dict]],
        horizon_days: int,
        market: Optional[str] = None,
        variety: Optional[str] = None,
        prediction_date: Optional[Union[str, pd.Timestamp]] = None,
    ) -> Dict[str, Any]:
        """
        Generates 14-day or 30-day calendar horizon modal price forecast.

        Parameters:
        -----------
        history : DataFrame or list of dicts
            Historical observations spanning >= 30 days.
        horizon_days : int
            14 or 30 calendar days.
        market : str, optional
            Mandi/market name. If None, inferred from history.
        variety : str, optional
            Rice variety. If None, inferred from history.
        prediction_date : str, optional
            Origin date for prediction (YYYY-MM-DD). If None, inferred as max date.

        Returns:
        --------
        dict containing predicted modal price, change, convention, and evaluation metadata.
        """
        try:
            h = int(horizon_days)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid horizon_days: {horizon_days}. Expected 14 or 30.")

        if h not in (14, 30):
            raise ValueError(f"Unsupported horizon_days: {horizon_days}. Supported horizons are 14 and 30.")

        model, cfg = self._load_horizon_model(h)

        clean_df, resolved_market, resolved_variety, resolved_date = normalize_input_data(
            history, market=market, variety=variety, prediction_date=prediction_date, min_observations=30
        )

        try:
            req_target_dt = (pd.to_datetime(resolved_date) + pd.Timedelta(days=h)).strftime("%Y-%m-%d")
        except Exception:
            req_target_dt = None

        feat_df = generate_48_price_features(clean_df)
        X = feat_df.iloc[[-1]][self.features_list].copy()
        X = X.apply(pd.to_numeric, errors="coerce")

        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"NaN detected in generated features: {nan_cols}")

        predicted_change = float(model.predict(X)[0])
        current_price = float(feat_df.iloc[-1][PRICE_COLUMN])
        predicted_price = current_price + predicted_change

        return {
            "market": resolved_market,
            "variety": resolved_variety,
            "prediction_date": resolved_date,
            "forecast_origin_date": resolved_date,
            "horizon_days": h,
            "requested_target_date": req_target_dt,
            "current_modal_price": round(current_price, 2),
            "predicted_modal_price": round(predicted_price, 2),
            "predicted_price_change": round(predicted_change, 2),
            "target_date_convention": TARGET_CONVENTIONS[h],
            "model_name": MODEL_NAMES[h],
            "model_version": "1.0.0",
            "release_status": RELEASE_STATUSES[h],
            "limitations": MODEL_LIMITATIONS[h],
            "horizon_metadata": {
                "max_tolerance_days": cfg.get("max_tolerance_days", 3),
                "train_period": cfg.get("train_period"),
                "holdout_benchmark": cfg.get("holdout_benchmark"),
            }
        }

    def predict(
        self,
        history: Union[pd.DataFrame, List[Dict]],
        horizon: Union[int, str] = "short",
        market: Optional[str] = None,
        variety: Optional[str] = None,
        prediction_date: Optional[Union[str, pd.Timestamp]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Unified prediction entry point across all three horizons.

        Parameters:
        -----------
        history : DataFrame or list of dicts
            Historical observations.
        horizon : int or str, default 'short'
            - 'short', 1, or '1-7': Short-term production forecast.
            - 14: 14-day calendar horizon forecast.
            - 30: 30-day calendar horizon forecast.
        """
        if horizon in ("short", "1-7", 1, "1"):
            return self.predict_short_term(
                history=history,
                market=market,
                variety=variety,
                prediction_date=prediction_date,
                include_volatility=kwargs.get("include_volatility", True),
                volatility_version=kwargs.get("volatility_version", "v2_normalized"),
            )
        elif horizon in (14, "14", "14d"):
            return self.predict_horizon(
                history=history,
                horizon_days=14,
                market=market,
                variety=variety,
                prediction_date=prediction_date,
            )
        elif horizon in (30, "30", "30d"):
            return self.predict_horizon(
                history=history,
                horizon_days=30,
                market=market,
                variety=variety,
                prediction_date=prediction_date,
            )
        else:
            raise ValueError(
                f"Unsupported horizon: '{horizon}'. Valid options are 'short' (1-7 days), 14, or 30."
            )
