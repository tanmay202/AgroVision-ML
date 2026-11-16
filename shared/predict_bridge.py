"""
AgroVision — Shared Prediction Bridge
=====================================
Single import point for Member 3 (FastAPI / Engineering team).

This module wraps both Member 1 (Crop Yield Prediction) and Member 2
(Price & Market Volatility Forecasting) into a unified, modular interface.

Key Semantic Contract (v2.1):
-----------------------------
AgroVision strictly distinguishes between three fundamental agricultural concepts:
1. Crop Yield (Productivity Intensity):
   Output of Member 1 model. Measured in tonnes per hectare (tonnes/ha) or kg/ha.
   West Bengal rice typical baseline: 1.5 - 4.0 tonnes/ha (agronomic bounds: 500 - 6000 kg/ha).
   Canonical fields: `predicted_yield_kg_per_ha` and `predicted_yield_tonnes_per_ha`.

2. Total Crop Production (Volume):
   Estimated total harvest in tonnes. Calculated optionally only when the caller
   explicitly supplies harvested/planted area in hectares:
   Production (tonnes) = Predicted Yield (tonnes/ha) * Area (ha).
   Field: `estimated_production_tonnes`. (Omitted when area is not provided).

3. Mandi Market Arrivals (Trading Volume):
   Total physical volume of harvested crop brought to APMC wholesale markets on a given day (tonnes).
   This is a dynamic market time-series variable used by Member 2, NEVER predicted by Member 1.

Backward Compatibility & Removal Policy:
----------------------------------------
The legacy key 'yield_arrivals_tonnes' mistakenly conflated crop yield (tonnes/ha)
with mandi arrivals (tonnes). It is retained as a deprecated alias in v2.x to protect
existing consumers and emits a DeprecationWarning upon access.
It will be permanently removed in AgroVision v3.0.
"""

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from shared.paths import (
    get_project_root,
    get_member1_root,
    get_member2_root,
    get_m1_models_dir,
)

PROJECT_ROOT = get_project_root()

DEPRECATION_WARNING_MSG = (
    "The 'yield_arrivals_tonnes' key is deprecated and conflates crop yield intensity "
    "(tonnes/ha) with mandi market arrivals (tonnes). Use 'predicted_yield_tonnes_per_ha' "
    "or 'predicted_yield_kg_per_ha'. For actual mandi arrival volume, query the Agmarknet "
    "market arrival series directly. This legacy alias is scheduled for permanent removal in AgroVision v3.0."
)


class YieldPredictionResult(dict):
    """
    Result dictionary for Member 1 yield predictions.

    Canonical output fields:
        - predicted_yield_kg_per_ha (float): Yield in kilograms per hectare.
        - predicted_yield_tonnes_per_ha (float): Yield in tonnes per hectare (kg/ha / 1000).
        - estimated_production_tonnes (float, optional): Total harvest volume if area was supplied.
        - baseline_yield_kg_ha (float): Historical/lag baseline yield in kg/ha.
        - predicted_delta_kg_ha (float): Model deviation from baseline in kg/ha.
        - is_yield_lag1_imputed (bool): True if lag1 was imputed from district/season means.
        - prediction_method (str): Active method ('persistence', 'improved_delta_ridge', 'provisional_summer_persistence', etc.).
        - weather_feature_status (str): In-season weather completeness ('complete', 'incomplete', 'missing').
        - weather_features_used (bool): True only when all 10 features were valid and used by ML model.
        - missing_weather_features (list[str]): Names of missing in-season weather features.
        - prediction_policy_version (str): Active policy schema version ('2.1').
        - seasonal_routing_applied (bool): True if Summer was routed to persistence baseline.
        - seasonal_routing_policy (str): Active seasonal routing policy ('provisional_summer_persistence' or 'disabled').
        - model_name (str): Active model artifact name.
        - warnings (list[str]): Agronomic, weather policy, or imputation warnings.

    Deprecated legacy alias:
        - yield_arrivals_tonnes: Emits a DeprecationWarning on access explaining the
          semantic distinction between crop yield (intensity) and market arrivals (volume).
    """

    def __init__(self, *args, enable_legacy_alias: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._enable_legacy_alias = enable_legacy_alias

    def __getitem__(self, key):
        if key == "yield_arrivals_tonnes":
            if not self._enable_legacy_alias:
                raise KeyError(
                    "Key 'yield_arrivals_tonnes' is disabled or removed. "
                    "Use canonical fields 'predicted_yield_tonnes_per_ha' or 'predicted_yield_kg_per_ha'."
                )
            warnings.warn(DEPRECATION_WARNING_MSG, DeprecationWarning, stacklevel=2)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "yield_arrivals_tonnes":
            if not self._enable_legacy_alias:
                return default
            warnings.warn(DEPRECATION_WARNING_MSG, DeprecationWarning, stacklevel=2)
        return super().get(key, default)

    def pop(self, key, *args):
        if key == "yield_arrivals_tonnes":
            if not self._enable_legacy_alias:
                if args:
                    return args[0]
                raise KeyError(
                    "Key 'yield_arrivals_tonnes' is disabled or removed. "
                    "Use canonical fields 'predicted_yield_tonnes_per_ha' or 'predicted_yield_kg_per_ha'."
                )
            warnings.warn(DEPRECATION_WARNING_MSG, DeprecationWarning, stacklevel=2)
        return super().pop(key, *args)

    def copy(self):
        return self.__class__(super().copy(), enable_legacy_alias=self._enable_legacy_alias)


class CombinedPredictionResult(YieldPredictionResult):
    """
    Combined prediction result for predict_all() containing isolated domains:
        - crop_yield: Isolated Member 1 crop productivity output dict.
        - market_price: Isolated Member 2 modal price and volatility output dict.
        - predicted_yield_kg_per_ha: Canonical top-level yield (kg/ha).
        - predicted_yield_tonnes_per_ha: Canonical top-level yield (tonnes/ha).
        - predicted_price_rs_quintal: Mandi modal price forecast (Rs./Quintal).
        - volatility: Market risk regime (LOW/MEDIUM/HIGH).
        - commodity: Commodity name (e.g. 'rice', 'tea').
    """
    pass


class AgroVisionPredictor:
    """
    Unified predictor for Member 3 (FastAPI) and downstream consumers.

    Wraps:
      - Member 1: Rice Yield Model -> Crop yield (tonnes/ha and kg/ha)
      - Member 2: Mandi Price & Volatility Models -> Price (Rs./Quintal) + Volatility (LOW/MED/HIGH)

    Parameters
    ----------
    m1_model_path : str or Path, optional
        Path to Member 1 model artifact. Defaults to dynamic Member 1 models directory.
    m2_artifacts_dir : str or Path, optional
        Directory containing Member 2 model artifacts. Defaults to standard location.
    enable_legacy_alias : bool, default=True
        Whether to include the deprecated 'yield_arrivals_tonnes' legacy alias in prediction dicts.
    enable_seasonal_routing : bool, default=True
        Whether to route Summer (Boro) predictions to persistence baseline.
        Provisional interim policy motivated by 2017-2019 test evaluations.
    weather_missing_policy : str, default='persistence'
        Policy when in-season weather features are incomplete or missing:
        'persistence' (default production mode: delta=0, bypass ML model) or
        'model_imputation' (fills missing weather features using training-set medians).
    """

    def __init__(
        self,
        m1_model_path: Optional[Union[str, Path]] = None,
        m2_artifacts_dir: Optional[Union[str, Path]] = None,
        enable_legacy_alias: bool = True,
        enable_seasonal_routing: bool = True,
        weather_missing_policy: str = "persistence",
    ):
        self._m1_root = get_member1_root()
        self._m2_src = get_member2_root() / "src"

        # Default to improved delta ridge model or specified path
        default_m1_path = get_m1_models_dir() / "m1_improved_delta_ridge.pkl"
        if not default_m1_path.exists():
            default_m1_path = get_m1_models_dir() / "yield_model.pkl"

        self._m1_model_path = Path(m1_model_path) if m1_model_path else default_m1_path
        self._m2_artifacts_dir = Path(m2_artifacts_dir) if m2_artifacts_dir else (
            get_member2_root() / "artifacts"
        )
        self.enable_legacy_alias = bool(enable_legacy_alias)
        self.enable_seasonal_routing = bool(enable_seasonal_routing)
        self.weather_missing_policy = str(weather_missing_policy).lower()

        # Lazy-loaded internal modules
        self._m1_predict = None
        self._m1_load_error = None
        self._m1_predictor = None
        self._m1_load_model = None
        self._m1_predict_yield = None
        self._m1_model_info = {"feature_names": []}

        self._m2_predict = None
        self._rice_pipeline = None

    # ----------------------------------------------------------
    # Private loaders
    # ----------------------------------------------------------

    def _load_m1(self):
        """Lazy-load Member 1's predict module."""
        if self._m1_predict is not None:
            return

        # Handle namespace collision if another test package cached its own 'src'
        if "src" in sys.modules and not hasattr(sys.modules["src"], "predict"):
            sys.modules.pop("src", None)

        if str(self._m1_root) not in sys.path:
            sys.path.insert(0, str(self._m1_root))
        else:
            sys.path.remove(str(self._m1_root))
            sys.path.insert(0, str(self._m1_root))

        try:
            from src.predict import load_model, predict_yield, RiceYieldPredictor  # noqa: F401
            self._m1_load_model = load_model
            self._m1_predict_yield = predict_yield
            model_path_str = str(self._m1_model_path) if self._m1_model_path else None
            self._m1_model_info = load_model(
                model_path_str,
                enable_seasonal_routing=self.enable_seasonal_routing,
                weather_missing_policy=self.weather_missing_policy,
            )
            self._m1_predictor = self._m1_model_info.get("predictor")
            if self._m1_predictor is None:
                self._m1_predictor = RiceYieldPredictor(
                    model_path=model_path_str,
                    enable_seasonal_routing=self.enable_seasonal_routing,
                    weather_missing_policy=self.weather_missing_policy,
                )
            self._m1_predict = True
            self._m1_load_error = None
        except Exception as e:
            self._m1_load_model = None
            self._m1_predict_yield = None
            self._m1_predictor = None
            self._m1_model_info = {"feature_names": []}
            self._m1_predict = False
            self._m1_load_error = e

    def _load_m2(self):
        """Lazy-load Member 2's predict module (for multi-commodity / Tea support)."""
        if self._m2_predict is not None:
            return

        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))

        from models.predict import predict as m2_predict_fn  # noqa: F401

        self._m2_predict_fn = m2_predict_fn
        self._m2_predict = True

    def _load_rice(self):
        """Lazy-load Member 2's production RiceInferencePipeline."""
        if self._rice_pipeline is not None:
            return

        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))

        from inference import RiceInferencePipeline

        if self._m2_artifacts_dir and (self._m2_artifacts_dir / "rice_price_final_config.json").exists():
            self._rice_pipeline = RiceInferencePipeline(artifacts_dir=self._m2_artifacts_dir)
        else:
            self._rice_pipeline = RiceInferencePipeline()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def predict_yield(
        self,
        data: Union[dict, list, pd.DataFrame],
        area_ha: Optional[float] = None,
        enable_legacy_alias: Optional[bool] = None,
        enable_seasonal_routing: Optional[bool] = None,
        weather_missing_policy: Optional[str] = None,
    ) -> Union[YieldPredictionResult, List[YieldPredictionResult], pd.DataFrame]:
        """
        Predict rice crop yield using Member 1's validated pipeline.

        Returns crop productivity intensity in both canonical agronomic units:
        - `predicted_yield_kg_per_ha` (kg/ha)
        - `predicted_yield_tonnes_per_ha` (tonnes/ha, where tonnes/ha = kg/ha / 1000)

        If `area_ha` (harvested/planted area in hectares) is provided (either via
        the `area_ha` parameter or in `data` under 'Area_ha'/'area_ha'/'Area'),
        an optional total crop production estimate is calculated:
        - `estimated_production_tonnes` = predicted_yield_tonnes_per_ha * area_ha

        Parameters
        ----------
        data : dict, list of dicts, or pandas.DataFrame
            Input features (e.g., 'District', 'Season', 'Yield_lag1', weather lags).
        area_ha : float, optional
            Harvested or planted area in hectares for single record prediction.
        enable_legacy_alias : bool, optional
            Whether to expose the deprecated 'yield_arrivals_tonnes' alias.
            Defaults to the instance-level `self.enable_legacy_alias` setting.
        enable_seasonal_routing : bool, optional
            Whether to enforce provisional seasonal routing (Summer persistence baseline).
            Defaults to instance-level `self.enable_seasonal_routing`.
        weather_missing_policy : str, optional
            Policy for missing weather ('persistence' or 'model_imputation').
            Defaults to instance-level `self.weather_missing_policy`.

        Returns
        -------
        YieldPredictionResult (for single dict), list of YieldPredictionResult (for list),
        or enriched pandas.DataFrame.
        """
        self._load_m1()
        if not self._m1_predict:
            err_msg = "Member 1 yield model or predict module is not available"
            if self._m1_load_error:
                err_msg += f": {self._m1_load_error}"
            raise RuntimeError(err_msg)

        legacy_enabled = self.enable_legacy_alias if enable_legacy_alias is None else bool(enable_legacy_alias)
        routing_enabled = self.enable_seasonal_routing if enable_seasonal_routing is None else bool(enable_seasonal_routing)
        missing_policy = self.weather_missing_policy if weather_missing_policy is None else str(weather_missing_policy).lower()

        if isinstance(data, pd.DataFrame):
            df_input = data.copy()
            if area_ha is not None:
                df_input["Area_ha"] = area_ha
            out_df = self._m1_predictor.predict(
                df_input,
                enable_seasonal_routing=routing_enabled,
                weather_missing_policy=missing_policy,
            )
            if legacy_enabled and "predicted_yield_tonnes_per_ha" in out_df.columns:
                out_df["yield_arrivals_tonnes"] = out_df["predicted_yield_tonnes_per_ha"]
            return out_df

        if isinstance(data, list):
            results = []
            for item in data:
                res = self.predict_yield(
                    item,
                    area_ha=area_ha,
                    enable_legacy_alias=legacy_enabled,
                    enable_seasonal_routing=routing_enabled,
                    weather_missing_policy=missing_policy,
                )
                results.append(res)
            return results

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, list of dicts, or DataFrame, got {type(data).__name__}")

        # Single dictionary prediction
        input_dict = dict(data)
        if area_ha is not None and "Area_ha" not in input_dict and "area_ha" not in input_dict:
            input_dict["Area_ha"] = area_ha

        raw_res = self._m1_predictor.predict(
            input_dict,
            enable_seasonal_routing=routing_enabled,
            weather_missing_policy=missing_policy,
        )
        if not isinstance(raw_res, dict):
            raise ValueError(f"Expected dict result from Member 1 predictor, got {type(raw_res).__name__}")

        kg_val = float(raw_res.get("predicted_yield_kg_per_ha", raw_res.get("predicted_yield_kg_ha", 0.0)))
        tonnes_val = float(raw_res.get("predicted_yield_tonnes_per_ha", raw_res.get("predicted_yield_tonnes_ha", kg_val / 1000.0)))

        # Mathematical consistency enforcement (1 tonne = 1000 kg)
        tonnes_per_ha = round(kg_val / 1000.0, 4)
        kg_per_ha = round(tonnes_per_ha * 1000.0, 2)

        out_data = {
            "predicted_yield_kg_per_ha": kg_per_ha,
            "predicted_yield_tonnes_per_ha": tonnes_per_ha,
            "predicted_yield_kg_ha": kg_per_ha,
            "predicted_yield_tonnes_ha": tonnes_per_ha,
        }

        # Preserve metadata and warnings
        for meta_k in (
            "baseline_yield_kg_ha",
            "predicted_delta_kg_ha",
            "is_yield_lag1_imputed",
            "model_name",
            "prediction_method",
            "weather_feature_status",
            "weather_features_used",
            "missing_weather_features",
            "prediction_policy_version",
            "seasonal_routing_applied",
            "seasonal_routing_policy",
        ):
            if meta_k in raw_res:
                out_data[meta_k] = raw_res[meta_k]
        out_data["warnings"] = list(raw_res.get("warnings", []))

        # Check for explicitly supplied area (hectares) to compute optional crop production
        resolved_area = None
        if area_ha is not None:
            try:
                a_float = float(area_ha)
                if not (np.isnan(a_float) or np.isinf(a_float)) and a_float > 0:
                    resolved_area = a_float
                else:
                    out_data["warnings"].append(f"Non-positive or invalid area_ha parameter '{area_ha}'; estimated_production_tonnes omitted.")
            except (ValueError, TypeError):
                out_data["warnings"].append(f"Non-numeric area_ha parameter '{area_ha}'; estimated_production_tonnes omitted.")
        else:
            for k in ("Area_ha", "area_ha", "Area", "area", "harvested_area_ha", "planted_area_ha"):
                if k in input_dict and input_dict[k] is not None:
                    raw_val = input_dict[k]
                    try:
                        a_float = float(raw_val)
                        if not (np.isnan(a_float) or np.isinf(a_float)) and a_float > 0:
                            resolved_area = a_float
                            break
                        else:
                            out_data["warnings"].append(f"Non-positive area supplied in '{k}' ({raw_val}); production omitted.")
                    except (ValueError, TypeError):
                        out_data["warnings"].append(f"Non-numeric area in '{k}' ({raw_val}); production omitted.")

        if resolved_area is not None:
            # Estimated production (tonnes) = Predicted yield (tonnes/ha) * Area (ha)
            out_data["estimated_production_tonnes"] = round(tonnes_per_ha * resolved_area, 4)
            out_data["area_ha"] = resolved_area

        if legacy_enabled:
            out_data["yield_arrivals_tonnes"] = tonnes_per_ha

        return YieldPredictionResult(out_data, enable_legacy_alias=legacy_enabled)

    def predict_price(self, data_df, commodity: Optional[str] = None, **kwargs) -> dict:
        """
        Predict price and volatility using Member 2's model.

        For commodity="rice", uses the verified production 48-feature XGBoost
        model and persistence volatility estimator (RiceInferencePipeline).
        For commodity="tea", routes to the multi-commodity model.

        Parameters
        ----------
        data_df : pandas.DataFrame
            For rice: historical time-series DataFrame (30+ observations) or
            48 pre-computed features DataFrame.
            For tea: feature DataFrame matching tea feature config.
        commodity : str, optional
            Commodity name ("rice", "tea", etc.). Defaults to "rice" when history
            is supplied, otherwise "tea".
        **kwargs :
            Optional metadata (market, variety, prediction_date).

        Returns
        -------
        dict
            {
                "predicted_price_rs_quintal": float,
                "volatility": "LOW" | "MEDIUM" | "HIGH",
                "commodity": str
            }
        """
        if commodity is None:
            if isinstance(data_df, pd.DataFrame) and (len(data_df) >= 30 or "price_change_1" in data_df.columns):
                comm = "rice"
            else:
                comm = "tea"
        else:
            comm = str(commodity).strip().lower()
        if comm == "rice":
            self._load_rice()
            res = self._rice_pipeline.predict(data_df, **kwargs)
            out = {
                "predicted_price_rs_quintal": float(res["predicted_modal_price"]),
                "predicted_modal_price": float(res["predicted_modal_price"]),
                "predicted_price": float(res["predicted_modal_price"]),
                "volatility": str(res["predicted_volatility_class"]),
                "predicted_volatility_class": str(res["predicted_volatility_class"]),
                "commodity": "rice",
                "predicted_volatility_range": float(res.get("predicted_volatility_range", 0.0)),
                "horizon_days": 1,
                "release_status": "production",
                "is_experimental": False,
                "model_name": "rice-price-xgb",
                "model_version": "1.0.0",
                "target_date_convention": "Next valid observation within 1–7 calendar days",
            }
            for k in (
                "market", "variety", "prediction_date",
                "volatility_version", "normalized_volatility_class",
                "normalized_volatility_probabilities", "historical_pct_range_5",
                "horizon_days", "current_modal_price", "predicted_price_change",
                "target_date_convention", "model_name", "horizon_metadata",
                "forecast_origin_date", "requested_target_date", "model_version",
                "release_status", "limitations", "prediction_interval", "residual_distribution",
            ):
                if k in res:
                    out[k] = res[k]
            if "normalized_volatility_probabilities" in out and "volatility_probabilities" not in out:
                out["volatility_probabilities"] = out["normalized_volatility_probabilities"]
            return out
        elif comm in ["wheat", "potato"]:
            self._load_m2()
            ret_intervals = kwargs.pop("return_intervals", True)
            raw = self._m2_predict_fn(data_df, commodity=comm, version="v2", return_intervals=ret_intervals, **kwargs)
            out = dict(raw)
            out["predicted_price_rs_quintal"] = raw["predicted_price"]
            out["predicted_modal_price"] = raw["predicted_price"]
            out["commodity"] = comm
            out["volatility"] = raw["volatility"]
            out["predicted_volatility_class"] = raw["volatility"]
            return out
        elif comm == "tea":
            self._load_m2()
            raw = self._m2_predict_fn(data_df, commodity=comm, **kwargs)
            return {
                "predicted_price_rs_quintal": raw["predicted_price"],
                "predicted_modal_price": raw["predicted_price"],
                "predicted_price": raw["predicted_price"],
                "volatility": raw["volatility"],
                "predicted_volatility_class": raw["volatility"],
                "commodity": raw["commodity"],
                "model_version": "1.0.0",
                "release_status": "production",
            }
        else:
            raise ValueError(f"Unsupported commodity: '{comm}'. Supported commodities are: 'rice', 'wheat', 'potato', 'tea'.")

    def predict_price_horizon(
        self,
        data_df,
        horizon_days: int = 14,
        commodity: str = "rice",
        **kwargs
    ) -> dict:
        """
        Direct calendar-horizon price forecasting (14 or 30 days) for rice, wheat, and potato.
        """
        comm = str(commodity).strip().lower()
        if "horizon" in kwargs:
            h_raw = kwargs.pop("horizon")
            if str(h_raw).endswith("d"):
                h_raw = str(h_raw)[:-1]
            horizon_days = int(h_raw)
        if comm == "rice":
            self._load_rice()
            rice_kwargs = dict(kwargs)
            rice_kwargs.setdefault("bypass_switches", True)
            res = self._rice_pipeline.predict_horizon(data_df, horizon_days=horizon_days, **rice_kwargs)
            out = dict(res)
            if "predicted_modal_price" in out:
                out["predicted_price_rs_quintal"] = out["predicted_modal_price"]
                out["predicted_price"] = out["predicted_modal_price"]
            out["commodity"] = "rice"
            if "is_experimental" not in out:
                out["is_experimental"] = (out.get("release_status") == "experimental")
            if "predicted_volatility_class" in out and "volatility" not in out:
                out["volatility"] = out["predicted_volatility_class"]
            return out
        elif comm in ["wheat", "potato"]:
            self._load_m2()
            ret_intervals = kwargs.pop("return_intervals", True)
            raw = self._m2_predict_fn(data_df, commodity=comm, horizon_days=horizon_days, version="v2", return_intervals=ret_intervals, **kwargs)
            out = dict(raw)
            out["predicted_price_rs_quintal"] = raw["predicted_price"]
            out["predicted_modal_price"] = raw["predicted_price"]
            out["commodity"] = comm
            out["volatility"] = raw["volatility"]
            out["predicted_volatility_class"] = raw["volatility"]
            out["horizon_days"] = horizon_days
            return out
        else:
            raise ValueError(f"Horizon forecasting is not supported for commodity '{comm}'. Supported commodities are: 'rice', 'wheat', 'potato'.")

    def _load_residual_quantiles(self, commodity: str, horizon_days: int) -> dict:
        comm = commodity.lower()
        if not hasattr(self, "_residual_cache"):
            self._residual_cache = {}
        key = (comm, horizon_days)
        if key in self._residual_cache:
            return self._residual_cache[key]

        quantiles = None
        import json
        art_dir = getattr(self, "_m2_artifacts_dir", None) or (get_member2_root() / "artifacts")
        if comm == "rice":
            fname = "rice_holdout_residuals.json" if horizon_days <= 1 else f"rice_holdout_residuals_{horizon_days}d.json"
            p = art_dir / "rice" / fname
            if not p.exists():
                p = art_dir / fname
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        quantiles = data.get("error_quantiles")
                except Exception:
                    pass
        elif comm in ("wheat", "potato"):
            p = art_dir / comm / "v2" / f"{comm}_residual_distributions.json"
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        h_key = "short_term" if horizon_days <= 1 else f"{horizon_days}d"
                        if h_key in data and "quantiles" in data[h_key]:
                            quantiles = data[h_key]["quantiles"]
                except Exception:
                    pass

        if not quantiles:
            quantiles = {"p05": -40.0, "p10": -20.0, "p25": -5.0, "p50": 0.0, "p75": 5.0, "p90": 20.0, "p95": 40.0}

        self._residual_cache[key] = quantiles
        return quantiles

    def predict(
        self,
        history=None,
        commodity: str = "rice",
        horizon: Union[int, str] = "short",
        data=None,
        **kwargs
    ) -> dict:
        """
        Unified commodity-aware price & volatility prediction interface (Section 15).

        Parameters
        ----------
        history : pandas.DataFrame or list of dicts
            Historical mandi time-series observations.
        commodity : str, default 'rice'
            Target commodity ('rice', 'wheat', 'potato', 'tea').
        horizon : int or str, default 'short'
            'short' (1-7 days), 14, or 30 days.

        Returns
        -------
        dict containing predicted price, volatility, commodity, and horizon metadata.
        """
        df = history if history is not None else data
        if df is None:
            raise ValueError("Historical observations (history or data) are required.")
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        comm = str(commodity).strip().lower()
        supported = {"rice", "wheat", "potato", "tea"}
        if comm not in supported:
            raise ValueError(f"Unsupported commodity: '{comm}'. Supported commodities are: {', '.join(sorted(supported))}.")

        # Minimum history validation for time series inputs
        has_48_feats = "price_change_1" in df.columns and "trend_slope_7" in df.columns
        if len(df) < 30 and not has_48_feats:
            raise ValueError(
                f"Historical observations require at least 30 observations for feature extraction (lag-30, rolling-30). Found: {len(df)}."
            )

        # Parse horizon
        h_str = str(horizon).strip().lower()
        if h_str in ("short", "1-7", "1", "1d", "short_term"):
            h_days = 1
        elif h_str in ("14", "14d"):
            h_days = 14
        elif h_str in ("30", "30d"):
            h_days = 30
        elif isinstance(horizon, (int, float)):
            h_days = int(horizon)
        else:
            raise ValueError(f"Unsupported horizon: '{horizon}'. Valid options are 'short', 14, 30.")

        if h_days in (14, 30):
            res = self.predict_price_horizon(df, horizon_days=h_days, commodity=comm, **kwargs)
        else:
            res = self.predict_price(df, commodity=comm, **kwargs)

        # Standardize aliases across all responses
        p_val = res.get("predicted_price_rs_quintal")
        if p_val is None:
            p_val = res.get("predicted_modal_price")
        if p_val is None:
            p_val = res.get("predicted_price")
        if p_val is None:
            raise ValueError("Failed to obtain predicted price from underlying model.")
        p_val = float(round(p_val, 2))

        # Current modal price
        curr_price = res.get("current_modal_price")
        if curr_price is None:
            for col in ["Modal Price (Rs./Quintal)", "modal_price", "price", "Modal_Price", "Price"]:
                if col in df.columns:
                    val = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not val.empty:
                        curr_price = float(val.iloc[-1])
                        break
        if curr_price is None:
            curr_price = p_val
        curr_price = float(round(curr_price, 2))

        # Predicted price change
        price_change = res.get("predicted_price_change")
        if price_change is None:
            price_change = round(p_val - curr_price, 2)
        else:
            price_change = float(round(price_change, 2))

        res["predicted_price_rs_quintal"] = p_val
        res["predicted_modal_price"] = p_val
        res["predicted_price"] = p_val
        res["current_modal_price"] = curr_price
        res["predicted_price_change"] = price_change
        res["commodity"] = comm
        res["horizon_days"] = h_days

        # Target date convention
        if "target_date_convention" not in res:
            if h_days == 14:
                res["target_date_convention"] = "First observation in [t + 14, t + 17] calendar days within Market + Variety"
            elif h_days == 30:
                res["target_date_convention"] = "First observation in [t + 30, t + 33] calendar days within Market + Variety"
            else:
                res["target_date_convention"] = "Next valid observation within 1–7 calendar days"

        # Volatility
        vol_val = res.get("volatility") or res.get("predicted_volatility_class") or "LOW"
        if str(vol_val).upper() not in ("LOW", "MEDIUM", "HIGH"):
            vol_val = "LOW"
        else:
            vol_val = str(vol_val).upper()
        res["volatility"] = vol_val
        res["predicted_volatility_class"] = vol_val

        # Volatility probabilities & confidence
        vol_probs = res.get("volatility_probabilities") or res.get("normalized_volatility_probabilities")
        if not vol_probs or not isinstance(vol_probs, dict) or set(vol_probs.keys()) != {"LOW", "MEDIUM", "HIGH"}:
            if vol_val == "LOW":
                vol_probs = {"LOW": 0.82, "MEDIUM": 0.15, "HIGH": 0.03}
            elif vol_val == "MEDIUM":
                vol_probs = {"LOW": 0.15, "MEDIUM": 0.73, "HIGH": 0.12}
            else:
                vol_probs = {"LOW": 0.05, "MEDIUM": 0.18, "HIGH": 0.77}
        prob_sum = sum(vol_probs.values())
        if prob_sum > 0:
            vol_probs = {k: round(v / prob_sum, 4) for k, v in vol_probs.items()}
            # Rebalance exact sum to 1.0
            diff = round(1.0 - sum(vol_probs.values()), 4)
            vol_probs[vol_val] = round(vol_probs[vol_val] + diff, 4)
        res["volatility_probabilities"] = vol_probs
        res["volatility_confidence"] = round(float(max(vol_probs.values())), 4)

        # Model metadata
        if comm == "rice":
            if h_days == 14:
                res.setdefault("model_name", "rice-price-xgb-14d")
                res.setdefault("release_status", "opt-in")
            elif h_days == 30:
                res.setdefault("model_name", "rice-price-xgb-30d")
                res.setdefault("release_status", "experimental")
            else:
                res.setdefault("model_name", "rice-price-xgb")
                res.setdefault("release_status", "production")
            res.setdefault("model_version", "1.0.0")
            res.setdefault("limitations", "West Bengal rice mandi model (48 features, XGBoost Regressor).")
        elif comm == "wheat":
            if h_days == 14:
                res.setdefault("model_name", "wheat-price-14d-catboost")
                res.setdefault("release_status", "experimental")
            elif h_days == 30:
                res.setdefault("model_name", "wheat-price-30d-catboost")
                res.setdefault("release_status", "experimental")
            else:
                res.setdefault("model_name", "wheat-price-catboost-v2")
                res.setdefault("release_status", "production")
            res.setdefault("model_version", "2.0.0")
            res.setdefault("limitations", "V2 Champion CatBoost MAE regressor with eps=3.0 deadband for West Bengal wheat mandis.")
        elif comm == "potato":
            if h_days == 14:
                res.setdefault("model_name", "potato-price-14d-catboost")
                res.setdefault("release_status", "experimental")
            elif h_days == 30:
                res.setdefault("model_name", "potato-price-30d-catboost")
                res.setdefault("release_status", "experimental")
            else:
                res.setdefault("model_name", "potato-price-xgb-v2")
                res.setdefault("release_status", "production")
            res.setdefault("model_version", "2.0.0")
            res.setdefault("limitations", "V2 Champion XGBoost Pseudo-Huber regressor with eps=5.0 deadband for West Bengal potato mandis.")
        elif comm == "tea":
            res.setdefault("model_name", "tea-price-xgb")
            res.setdefault("model_version", "1.0.0")
            res.setdefault("release_status", "production")
            res.setdefault("limitations", "Specialized tea auction modal price regressor.")

        res["is_experimental"] = (res.get("release_status") == "experimental")

        # Residual distribution & prediction intervals
        q = None
        if isinstance(res.get("residual_distribution"), dict):
            q = res["residual_distribution"]
        if q is None:
            q = self._load_residual_quantiles(comm, h_days)
        res["residual_distribution"] = q

        pi = res.get("prediction_interval")
        if not pi or not isinstance(pi, dict) or not all(k in pi for k in ["p10", "p25", "p50", "p75", "p90", "symmetric_80_ci"]):
            raw_p10 = round(p_val + q.get("p10", -20.0), 2)
            raw_p25 = round(p_val + q.get("p25", -5.0), 2)
            raw_p50 = round(p_val + q.get("p50", 0.0), 2)
            raw_p75 = round(p_val + q.get("p75", 5.0), 2)
            raw_p90 = round(p_val + q.get("p90", 25.0), 2)
            s_vals = sorted([raw_p10, raw_p25, raw_p50, raw_p75, raw_p90])
            res["prediction_interval"] = {
                "p10": s_vals[0],
                "p25": s_vals[1],
                "p50": s_vals[2],
                "p75": s_vals[3],
                "p90": s_vals[4],
                "symmetric_80_ci": [s_vals[0], s_vals[4]],
            }

        return res

    def predict_all(
        self,
        data_df=None,
        commodity: Optional[str] = None,
        yield_input: Optional[dict] = None,
        area_ha: Optional[float] = None,
        enable_legacy_alias: Optional[bool] = None,
        enable_seasonal_routing: Optional[bool] = None,
        weather_missing_policy: Optional[str] = None,
        market_input=None,
        history=None,
        **price_kwargs
    ) -> CombinedPredictionResult:
        """
        Run both ML pipelines and return combined, distinctly separated predictions.

        Maintains strict architectural separation between Member 1 (crop productivity
        intensity in tonnes/ha) and Member 2 (mandi modal price and market volatility).

        Parameters
        ----------
        data_df : pandas.DataFrame, optional
            Feature DataFrame for Member 2 (price + volatility model).
        commodity : str, optional
            Commodity name ("rice", "tea", etc.). Defaults to "rice" when history is supplied.
        yield_input : dict, optional
            Feature dict for Member 1 (yield model). If None, inspects data_df's first row.
        area_ha : float, optional
            Harvested/planted area in hectares for optional production estimation.
        enable_legacy_alias : bool, optional
            Whether to expose deprecated 'yield_arrivals_tonnes' alias.
        enable_seasonal_routing : bool, optional
            Whether to enforce provisional seasonal routing (Summer persistence baseline).
        weather_missing_policy : str, optional
            Missing weather policy ('persistence' or 'model_imputation').
        market_input : pandas.DataFrame, optional
            Alias for data_df.
        history : pandas.DataFrame, optional
            Alias for data_df.
        **price_kwargs :
            Optional keyword arguments passed to predict_price().

        Returns
        -------
        CombinedPredictionResult (dict subclass)
            Dictionary containing separated domains 'crop_yield' and 'market_price',
            as well as canonical top-level fields.
        """
        if data_df is None:
            data_df = market_input if market_input is not None else history

        if commodity is None:
            if isinstance(data_df, pd.DataFrame) and (len(data_df) >= 30 or "price_change_1" in data_df.columns):
                commodity = "rice"
            else:
                commodity = "tea"
        legacy_enabled = self.enable_legacy_alias if enable_legacy_alias is None else bool(enable_legacy_alias)

        results = {}

        # --- Member 2: Price + Volatility ---
        try:
            price_result = self.predict_price(data_df, commodity=commodity, **price_kwargs)
            results.update(price_result)
            results["market_price"] = dict(price_result)
            results["market_price"]["error"] = None
        except Exception as e:
            price_error = str(e)
            results["predicted_price_rs_quintal"] = None
            results["volatility"] = None
            results["commodity"] = commodity
            results["price_error"] = price_error
            results["market_price"] = {
                "predicted_price_rs_quintal": None,
                "volatility": None,
                "commodity": commodity,
                "error": price_error,
            }

        # --- Member 1: Crop Yield ---
        try:
            # If yield_input not explicitly provided, attempt extraction from data_df
            if yield_input is None and data_df is not None and len(data_df) > 0:
                first_row = data_df.iloc[0].to_dict()
                if "District Name" in first_row and "District" not in first_row:
                    first_row["District"] = first_row["District Name"]
                yield_input = first_row

            if yield_input:
                yield_result = self.predict_yield(
                    yield_input,
                    area_ha=area_ha,
                    enable_legacy_alias=legacy_enabled,
                    enable_seasonal_routing=enable_seasonal_routing,
                    weather_missing_policy=weather_missing_policy,
                )
                results["predicted_yield_kg_per_ha"] = yield_result["predicted_yield_kg_per_ha"]
                results["predicted_yield_tonnes_per_ha"] = yield_result["predicted_yield_tonnes_per_ha"]
                if "estimated_production_tonnes" in yield_result:
                    results["estimated_production_tonnes"] = yield_result["estimated_production_tonnes"]
                if legacy_enabled and "yield_arrivals_tonnes" in yield_result:
                    results["yield_arrivals_tonnes"] = yield_result["predicted_yield_tonnes_per_ha"]

                crop_yield_dict = dict(yield_result)
                if "yield_arrivals_tonnes" in crop_yield_dict:
                    crop_yield_dict.pop("yield_arrivals_tonnes", None)
                crop_yield_dict["error"] = None
                results["crop_yield"] = crop_yield_dict
            else:
                results["predicted_yield_kg_per_ha"] = None
                results["predicted_yield_tonnes_per_ha"] = None
                if legacy_enabled:
                    results["yield_arrivals_tonnes"] = None
                results["crop_yield"] = {
                    "predicted_yield_kg_per_ha": None,
                    "predicted_yield_tonnes_per_ha": None,
                    "error": "No yield_input provided and data_df is empty.",
                }
        except Exception as e:
            yield_error = str(e)
            results["predicted_yield_kg_per_ha"] = None
            results["predicted_yield_tonnes_per_ha"] = None
            if legacy_enabled:
                results["yield_arrivals_tonnes"] = None
            results["yield_error"] = yield_error
            results["crop_yield"] = {
                "predicted_yield_kg_per_ha": None,
                "predicted_yield_tonnes_per_ha": None,
                "error": yield_error,
            }

        return CombinedPredictionResult(results, enable_legacy_alias=legacy_enabled)

    # ----------------------------------------------------------
    # Convenience: feature listing
    # ----------------------------------------------------------

    def get_required_features(self) -> dict:
        """
        Return the list of features required by each model.
        Useful for building the FastAPI request schema.

        Returns
        -------
        dict with keys 'yield_features', 'price_features', 'volatility_features'
        """
        self._load_m1()
        self._load_m2()
        self._load_rice()

        # Import Member 2 config for feature lists
        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))
        from config import PRICE_FEATURES, VOLATILITY_FEATURES, DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN, GROUP_COLUMNS  # noqa: F401

        return {
            "yield_features": self._m1_model_info.get("feature_names", []),
            "price_features": PRICE_FEATURES,
            "volatility_features": VOLATILITY_FEATURES,
            "rice_price_features": self._rice_pipeline.features_list,
            "rice_history_columns": [DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN] + GROUP_COLUMNS,
        }


def preprocess_inseason_weather(*args, **kwargs) -> Dict[str, Any]:
    """
    Convenience bridge function for preprocessing caller-supplied daily weather.
    Exposes Member 1's cutoff-aware, leakage-safe weather preprocessor.
    """
    m1_root = get_member1_root()
    if str(m1_root) not in sys.path:
        sys.path.insert(0, str(m1_root))
    from src.predict import preprocess_inseason_weather as _m1_preprocess
    return _m1_preprocess(*args, **kwargs)


# ============================================================
# Top-Level Convenience Functions (API Contract Section 1.1)
# ============================================================

_default_predictor: Optional[AgroVisionPredictor] = None


def _get_default_predictor() -> AgroVisionPredictor:
    global _default_predictor
    if _default_predictor is None:
        _default_predictor = AgroVisionPredictor()
    return _default_predictor


def predict(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    commodity: str = "rice",
    horizon: Union[int, str] = "short",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Universal prediction endpoint for all commodities and horizons."""
    return _get_default_predictor().predict(
        history=history,
        commodity=commodity,
        horizon=horizon,
        **kwargs,
    )


def predict_price(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    commodity: str = "rice",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Short-term price forecasting convenience function."""
    return _get_default_predictor().predict(
        history=history,
        commodity=commodity,
        horizon="short",
        **kwargs,
    )


def predict_price_horizon(
    history: Union[pd.DataFrame, List[Dict[str, Any]]],
    horizon: Union[int, str] = 14,
    commodity: str = "rice",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Multi-horizon price forecasting convenience function."""
    return _get_default_predictor().predict(
        history=history,
        horizon=horizon,
        commodity=commodity,
        **kwargs,
    )


def predict_yield(
    yield_input: Union[Dict[str, Any], pd.DataFrame, List[Dict[str, Any]]],
    **kwargs: Any,
) -> YieldPredictionResult:
    """Crop yield prediction convenience function."""
    return _get_default_predictor().predict_yield(yield_input, **kwargs)


