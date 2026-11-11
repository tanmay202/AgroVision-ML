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
        - model_name (str): Active model artifact name.
        - warnings (list[str]): Agronomic or imputation warnings.

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
    """

    def __init__(
        self,
        m1_model_path: Optional[Union[str, Path]] = None,
        m2_artifacts_dir: Optional[Union[str, Path]] = None,
        enable_legacy_alias: bool = True,
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

        if str(self._m1_root) not in sys.path:
            sys.path.insert(0, str(self._m1_root))

        try:
            from src.predict import load_model, predict_yield, RiceYieldPredictor  # noqa: F401
            self._m1_load_model = load_model
            self._m1_predict_yield = predict_yield
            model_path_str = str(self._m1_model_path) if self._m1_model_path else None
            self._m1_model_info = load_model(model_path_str)
            self._m1_predictor = self._m1_model_info.get("predictor")
            if self._m1_predictor is None:
                self._m1_predictor = RiceYieldPredictor(model_path=model_path_str)
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

        if isinstance(data, pd.DataFrame):
            df_input = data.copy()
            if area_ha is not None:
                df_input["Area_ha"] = area_ha
            out_df = self._m1_predictor.predict(df_input)
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
                )
                results.append(res)
            return results

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, list of dicts, or DataFrame, got {type(data).__name__}")

        # Single dictionary prediction
        input_dict = dict(data)
        if area_ha is not None and "Area_ha" not in input_dict and "area_ha" not in input_dict:
            input_dict["Area_ha"] = area_ha

        raw_res = self._m1_predictor.predict(input_dict)
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
        for meta_k in ("baseline_yield_kg_ha", "predicted_delta_kg_ha", "is_yield_lag1_imputed", "model_name"):
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

    def predict_price(self, data_df, commodity: str = "tea", **kwargs) -> dict:
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
        commodity : str
            Commodity name ("rice", "tea", etc.). Defaults to "tea".
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
        comm = str(commodity).strip().lower()
        if comm == "rice":
            self._load_rice()
            res = self._rice_pipeline.predict(data_df, **kwargs)
            out = {
                "predicted_price_rs_quintal": float(res["predicted_modal_price"]),
                "volatility": str(res["predicted_volatility_class"]),
                "commodity": "rice",
                "predicted_volatility_range": float(res.get("predicted_volatility_range", 0.0)),
            }
            for k in (
                "market", "variety", "prediction_date",
                "volatility_version", "normalized_volatility_class",
                "normalized_volatility_probabilities", "historical_pct_range_5",
                "horizon_days", "current_modal_price", "predicted_price_change",
                "target_date_convention", "model_name", "horizon_metadata",
                "forecast_origin_date", "requested_target_date", "model_version",
                "release_status", "limitations",
            ):
                if k in res:
                    out[k] = res[k]
            return out
        else:
            self._load_m2()
            raw = self._m2_predict_fn(data_df, commodity=comm)
            return {
                "predicted_price_rs_quintal": raw["predicted_price"],
                "volatility": raw["volatility"],
                "commodity": raw["commodity"],
            }

    def predict_price_horizon(
        self,
        data_df,
        horizon_days: int = 14,
        commodity: str = "rice",
        **kwargs
    ) -> dict:
        """
        Direct calendar-horizon price forecasting (14 or 30 days) for rice.
        """
        comm = str(commodity).strip().lower()
        if comm != "rice":
            raise NotImplementedError(f"Horizon forecasting currently supported for 'rice', received '{comm}'")
        self._load_rice()
        return self._rice_pipeline.predict_horizon(data_df, horizon_days=horizon_days, **kwargs)

    def predict_all(
        self,
        data_df,
        commodity: str = "tea",
        yield_input: Optional[dict] = None,
        area_ha: Optional[float] = None,
        enable_legacy_alias: Optional[bool] = None,
        **price_kwargs
    ) -> CombinedPredictionResult:
        """
        Run both ML pipelines and return combined, distinctly separated predictions.

        Maintains strict architectural separation between Member 1 (crop productivity
        intensity in tonnes/ha) and Member 2 (mandi modal price and market volatility).

        Parameters
        ----------
        data_df : pandas.DataFrame
            Feature DataFrame for Member 2 (price + volatility model).
        commodity : str
            Commodity name ("rice", "tea", etc.).
        yield_input : dict, optional
            Feature dict for Member 1 (yield model). If None, inspects data_df's first row.
        area_ha : float, optional
            Harvested/planted area in hectares for optional production estimation.
        enable_legacy_alias : bool, optional
            Whether to expose deprecated 'yield_arrivals_tonnes' alias.
        **price_kwargs :
            Optional keyword arguments passed to predict_price().

        Returns
        -------
        CombinedPredictionResult (dict subclass)
            Dictionary containing separated domains 'crop_yield' and 'market_price',
            as well as canonical top-level fields.
        """
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
