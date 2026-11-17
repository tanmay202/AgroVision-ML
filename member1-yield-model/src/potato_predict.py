"""
AgroVision Model 1: Potato Yield Prediction Interface
======================================================
Production-grade inference interface for West Bengal Rabi Potato Yield Forecasting.
Uses certified Huber Regressor on Delta from 3-Year Rolling Mean,
with NASA POWER meteorological features up to the January 31 cutoff.

Operational Release Status: PRODUCTION READY
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("potato_predict")

SRC_DIR = Path(__file__).resolve().parent
M1_DIR = SRC_DIR.parent
MODELS_DIR = M1_DIR / "models"

try:
    from weather_preprocessor import normalize_district_name, POTATO_FEATURE_COLUMNS
    from weather_service import NASAWeatherService, get_weather_service
except ImportError:
    from src.weather_preprocessor import normalize_district_name, POTATO_FEATURE_COLUMNS
    from src.weather_service import NASAWeatherService, get_weather_service


class PotatoYieldPredictor:
    """
    Dedicated predictor for West Bengal Rabi Potato Yield.
    Loads certified production champion m1_potato_delta_huber.pkl (or legacy candidate)
    and evaluates predictions with empirical prediction intervals.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        prod_path = MODELS_DIR / "m1_potato_delta_huber.pkl"
        fallback_path = MODELS_DIR / "m1_potato_catboost.pkl"
        default_path = prod_path if prod_path.exists() else fallback_path

        self.model_path = Path(model_path) if model_path else default_path
        self.metadata_path = self.model_path.with_suffix(".metadata.json")
        self._load_artifacts()
        self._weather_service = None

    def _load_artifacts(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Potato model artifact not found: {self.model_path}")
        self.artifact = joblib.load(self.model_path)
        self.model = self.artifact['model']
        self.scaler = self.artifact.get('scaler', None)
        self.feature_cols = self.artifact['feature_cols']
        self.dist_means = self.artifact.get('dist_means', {})
        self.global_mean = self.artifact.get('global_mean', 23000.0)
        self.target_type = self.artifact.get('target_type', 'absolute')
        self.residuals_quantiles = self.artifact.get('residuals_quantiles', {})
        self.model_name = self.model_path.name

        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.release_status = self.metadata.get('release_status', 'PRODUCTION READY')
        else:
            self.release_status = "PRODUCTION READY" if "delta" in str(self.model_path) else "EXPERIMENTAL"
            self.metadata = {"release_status": self.release_status, "model_name": self.model_name}

    def _get_weather_service(self) -> NASAWeatherService:
        if self._weather_service is None:
            self._weather_service = get_weather_service()
        return self._weather_service

    def predict(
        self,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        area_ha: Optional[float] = None,
        auto_fetch_weather: bool = True,
        weather_missing_policy: str = "persistence",
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
        """
        Predict potato yield in kg/ha and tonnes/ha.
        """
        if isinstance(data, pd.DataFrame):
            results = []
            for _, row in data.iterrows():
                r = self.predict_single(
                    row.to_dict(),
                    area_ha=area_ha,
                    auto_fetch_weather=auto_fetch_weather,
                    weather_missing_policy=weather_missing_policy
                )
                results.append(r)
            return pd.DataFrame(results)

        if isinstance(data, list):
            return [
                self.predict_single(
                    item,
                    area_ha=area_ha,
                    auto_fetch_weather=auto_fetch_weather,
                    weather_missing_policy=weather_missing_policy
                )
                for item in data
            ]

        if isinstance(data, dict):
            return self.predict_single(
                data,
                area_ha=area_ha,
                auto_fetch_weather=auto_fetch_weather,
                weather_missing_policy=weather_missing_policy
            )

        raise TypeError(f"Expected dict, list of dicts, or DataFrame, got {type(data).__name__}")

    def predict_single(
        self,
        item: Dict[str, Any],
        area_ha: Optional[float] = None,
        auto_fetch_weather: bool = True,
        weather_missing_policy: str = "persistence",
    ) -> Dict[str, Any]:
        warnings = []
        d_raw = item.get("District", item.get("district", "Unknown"))
        district = normalize_district_name(d_raw)

        # Season validation
        season_raw = str(item.get("Season", item.get("season", "Rabi"))).strip().title()
        if season_raw.lower() not in ("rabi", "winter", "whole year"):
            warnings.append(f"Potato in plains of West Bengal is predominantly Rabi; season '{season_raw}' normalized to Rabi.")
        season = "Rabi"

        # Year validation
        year_raw = item.get("Crop_Year", item.get("crop_year", item.get("Year", item.get("year"))))
        try:
            year = int(year_raw) if year_raw is not None else None
        except (ValueError, TypeError):
            year = None

        # Lag yields
        lag1 = item.get("Yield_lag1", item.get("yield_lag1"))
        lag2 = item.get("Yield_lag2", item.get("yield_lag2"))
        roll3 = item.get("Yield_roll_mean3", item.get("yield_roll_mean3"))

        dist_mean = self.dist_means.get(district, self.global_mean)
        if lag1 is None or pd.isna(lag1):
            lag1 = dist_mean
            warnings.append(f"Yield_lag1 missing for {district}; imputed with historical district mean ({dist_mean:.1f} kg/ha).")
        else:
            lag1 = float(lag1)

        lag2 = float(lag2) if (lag2 is not None and not pd.isna(lag2)) else lag1
        roll3 = float(roll3) if (roll3 is not None and not pd.isna(roll3)) else lag1

        # Area handling
        target_area = area_ha if area_ha is not None else item.get("Area_ha", item.get("area_ha", item.get("Area", item.get("area"))))
        try:
            target_area = float(target_area) if target_area is not None and not pd.isna(target_area) else None
        except (ValueError, TypeError):
            target_area = None

        area_lag1 = item.get("Area_lag1", item.get("area_lag1", target_area if target_area else 5000.0))
        try:
            area_lag1 = float(area_lag1)
        except (ValueError, TypeError):
            area_lag1 = 5000.0
        log_area_lag1 = float(np.log1p(area_lag1))

        # Weather acquisition
        weather_feats = {}
        weather_source = "missing"
        weather_provenance = {}

        has_caller_weather = all(k in item for k in POTATO_FEATURE_COLUMNS)
        if has_caller_weather:
            weather_feats = {k: float(item[k]) for k in POTATO_FEATURE_COLUMNS}
            weather_source = "caller_supplied"
            weather_provenance = {"source": "caller_supplied"}
        elif auto_fetch_weather and year is not None:
            svc = self._get_weather_service()
            f_res = svc.get_inseason_features(district=district, year=year, season="Rabi", crop="potato")
            if f_res.get("is_complete"):
                weather_feats = f_res.get("features", {})
                weather_source = f_res.get("weather_source", "nasa_power")
                weather_provenance = f_res.get("metadata", {})
            else:
                warnings.append(f"Incomplete weather fetch for {district} Potato {year}; fallback policy '{weather_missing_policy}' active.")
                warnings.extend(f_res.get("warnings", []))

        # Prediction execution
        if weather_feats and all(k in weather_feats and not pd.isna(weather_feats[k]) for k in POTATO_FEATURE_COLUMNS):
            x_dict = {
                'yield_lag1': lag1,
                'yield_lag2': lag2,
                'yield_roll_mean3': roll3,
                'log_area_lag1': log_area_lag1,
            }
            if 'dist_mean' in self.feature_cols:
                x_dict['dist_mean'] = dist_mean
            x_dict.update(weather_feats)
            
            x_arr = np.array([[x_dict[col] for col in self.feature_cols]])
            x_scaled = self.scaler.transform(x_arr) if self.scaler else x_arr
            raw_pred = float(self.model.predict(x_scaled)[0])

            if self.target_type == 'delta_roll3':
                pred_kg = float(roll3 + raw_pred)
                pred_method = f"{self.model_name}_delta_weather"
            elif self.target_type == 'delta_lag1':
                pred_kg = float(lag1 + raw_pred)
                pred_method = f"{self.model_name}_delta_weather"
            else:
                pred_kg = raw_pred
                pred_method = f"{self.model_name}_weather"
        else:
            if weather_missing_policy == "district_mean":
                pred_kg = float(dist_mean)
                pred_method = "potato_district_mean_fallback"
            elif weather_missing_policy == "rolling_mean":
                pred_kg = float(roll3)
                pred_method = "potato_rolling_mean_fallback"
            else:
                pred_kg = float(lag1)
                pred_method = "potato_persistence_fallback"
            warnings.append(f"Weather-informed model inactive. Evaluated using {pred_method}.")

        # Agronomic clipping for West Bengal potato (2000 to 50000 kg/ha)
        pred_kg = max(2000.0, min(50000.0, pred_kg))
        pred_tonnes = round(pred_kg / 1000.0, 4)

        p10_offset = self.residuals_quantiles.get('p10', -3000.0)
        p90_offset = self.residuals_quantiles.get('p90', 3000.0)
        pred_interval = {
            "p10_kg_per_ha": round(max(1500.0, pred_kg + p10_offset), 2),
            "p90_kg_per_ha": round(min(55000.0, pred_kg + p90_offset), 2),
        }

        pred_prod = None
        if target_area is not None and target_area > 0:
            pred_prod = round(pred_tonnes * target_area, 2)

        return {
            "crop": "potato",
            "model_name": self.model_name,
            "district": district,
            "season": season,
            "predicted_yield_kg_per_ha": round(pred_kg, 2),
            "predicted_yield_tonnes_per_ha": pred_tonnes,
            "estimated_production_tonnes": pred_prod,
            "prediction_method": pred_method,
            "prediction_interval": pred_interval,
            "weather_provenance": {
                "source": weather_source,
                "metadata": weather_provenance
            },
            "release_status": self.release_status,
            "warnings": warnings,
        }


def predict_potato_yield(data: Any, **kwargs) -> Any:
    """Convenience function for potato yield prediction."""
    predictor = PotatoYieldPredictor()
    return predictor.predict(data, **kwargs)
