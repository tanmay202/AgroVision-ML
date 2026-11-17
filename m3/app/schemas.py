from datetime import date, datetime
from typing import Any, Optional
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    database: str
    timestamp: datetime


class MarketSeriesItem(BaseModel):
    mandi_id: str
    mandi_name: str
    crop_id: str
    crop_name: str
    district_id: str
    district_name: str
    total_observations: int
    latest_trade_date: Optional[date]


class TradeHistoryRecord(BaseModel):
    trade_date: date
    modal_price_inr: float
    arrivals_mt: float


class RiskAssessmentItem(BaseModel):
    v_norm: float
    h_anomaly: float
    storage_vuln: float
    composite_risk_score: float
    risk_regime: str  # 'LOW' | 'MEDIUM' | 'HIGH'
    forecast_rain_48h_mm: float
    trailing_rain_3d_mm: float
    rh_mean_pct: float
    storage_avail_index: float
    advisory_action: str
    advisory_headline: str
    advisory_reason: str
    weather_adjusted_lower: float
    is_simulated: bool = False


class PriceForecastItem(BaseModel):
    horizon_label: str  # 'short', '14d', '30d'
    forecast_origin_date: date
    target_date: date
    current_modal_price: float
    predicted_modal_price: float
    predicted_price_change: float
    predicted_price_lower: float
    predicted_price_upper: float
    naive_baseline_price: Optional[float] = None
    naive_baseline_mae: Optional[float] = None
    volatility_class: Optional[str] = None
    volatility_prob_low: Optional[float] = None
    volatility_prob_medium: Optional[float] = None
    volatility_prob_high: Optional[float] = None
    model_version: str
    is_experimental: bool
    risk: RiskAssessmentItem


class TelemetryProvenance(BaseModel):
    stale_data_warning: bool
    stale_reasons: list[str]
    weather_source: str
    weather_age_hours: float
    market_age_hours: float
    latest_ndvi: Optional[float] = None
    ndvi_anomaly: Optional[float] = None
    ndvi_date: Optional[date] = None
    ndvi_age_days: int


class ForecastResponse(BaseModel):
    mandi_id: str
    mandi_name: str
    district_id: str
    district_name: str
    storage_avail_index: float
    crop_id: str
    canonical_name: str
    predicted_yield_kg_ha: float
    predicted_yield_q_acre: float
    yield_method: str
    latest_run_id: str
    ran_at: datetime
    stale_data_warning: bool
    telemetry: TelemetryProvenance
    forecasts: list[PriceForecastItem]


class IngestionRunResponse(BaseModel):
    overall_status: str
    ran_at_ist: str
    district_id: str
    pipelines: dict[str, Any]


class DriftFeatureItem(BaseModel):
    feature: str
    label: str
    domain: str
    unit: str
    psi: float
    wasserstein_distance: float
    normalized_w1: float
    classification_metric: str
    status: str
    ref_mean: float
    live_mean: Optional[float] = None
    ref_p10: float
    ref_p90: float
    live_samples: int


class DriftReportResponse(BaseModel):
    overall_status: str
    max_psi: float
    drifted_feature_count: int
    moderate_shift_count: int
    window_days: int
    mandi_id: Optional[str] = None
    crop_id: Optional[str] = None
    reference_generated_at: str
    evaluated_at: str
    features: list[DriftFeatureItem]


class ExplainFeatureItem(BaseModel):
    feature: str
    label: str
    unit: str
    feature_value: float
    shap_value_inr: float
    direction: str


class ExplainResponse(BaseModel):
    mandi_id: str
    mandi_name: str
    crop_id: str
    variety: str
    horizon_label: str
    horizon_days: int
    forecast_origin_date: str
    model_version: str
    current_modal_price: float
    base_expected_change_inr: float
    predicted_change_inr: float
    predicted_modal_price: float
    other_features_shap_inr: float
    computed_at: str
    top_features: list[ExplainFeatureItem]
